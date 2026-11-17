"""Open-Meteo latest NWP run (GFS Global, ECMWF IFS 0.25 deg) for the 81 points -> R2.

  python -m src.realtime.openmeteo_latest [--store r2|local:DIR]

Run initialisation time comes from Open-Meteo's model metadata endpoint
(documented at https://open-meteo.com/en/docs/model-updates):
    https://api.open-meteo.com/data/{meta_model}/static/meta.json
    fields: last_run_initialisation_time, last_run_availability_time,
            last_run_modification_time (unix seconds), update_interval_seconds
gfs_global blends ncep_gfs013 and ncep_gfs025 -> issue time recorded as the newest
component initialisation (upper bound, same convention as P2A's ECMWF rule).
Metadata is read before and after the forecast request; if the run changed in
between, that model is skipped (the next hourly run stores the new run cleanly).

* Idempotency: one object per (model, run): id ``{model}_runYYYYMMDDTHHMM``.
  When every model's current run is already in R2 no forecast request is made.
* Kept rows: hourly intervals starting at or after the issue time (lead >= 0);
  ``precipitation`` is the sum over the preceding hour (label = interval end).
* ``available_time_utc`` = ``ingestion_time_utc`` (actual retrieval time on the runner).
* Politeness: one forecast request per run for all points and the models with
  a new run; HTTP 429 (any request) -> no retry, exit 4.
* P3B hotfix (2026-11-17): metadata timeout connect 10 s / read 30 s, one retry
  after 15 s (worst case 75 s per call); forecast timeout 10 s / 60 s, one retry
  after 60 s. Models are processed independently. When a model's metadata cannot
  be read, its run is determined by the conservative history rule
  (configs/openmeteo_availability.yaml: newest 6-hourly run with issue + delay <=
  now; issue = lower bound of the served run) and the manifest records
  ``availability_source = "rule_fallback"`` (normal path: ``"metadata"``).
  Exit 0 when at least one model was published or skipped; 3 when every model failed.

Objects: lake/v1/realtime/openmeteo/date=D/{id}.parquet + manifest JSON.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.realtime import common as C  # noqa: E402

SOURCE = "openmeteo"
RETRY_WAIT_S = 60                 # forecast request: one retry after 60 s
FORECAST_TIMEOUT = (10, 60)       # (connect, read) seconds; was 120 s (P3B hotfix: cap 60 s)
META_TIMEOUT = (10, 30)           # metadata request (P3B hotfix)
META_RETRY_WAIT_S = 15            # metadata: one retry after 15 s -> worst case 30 + 15 + 30 = 75 s
CYCLE_H = 6                       # GFS and ECMWF IFS runs at 00/06/12/18Z
RULE_FILE = ROOT / "configs" / "openmeteo_availability.yaml"
_sleep = time.sleep               # indirection so tests can simulate waits without sleeping

ISSUE_RULE = {
    "metadata": "metadata last_run_initialisation_time; max over components (upper bound)",
    "rule_fallback": "rule_fallback: newest 6-hourly run with issue + delay_min <= retrieval time "
                     "(configs/openmeteo_availability.yaml; lower bound of the served run)",
}

# errors that make a model's metadata unusable (-> rule fallback); anything else propagates
META_ERRORS = (requests.RequestException, ValueError, KeyError, TypeError)


class RateLimited(RuntimeError):
    pass


def ts(sec: int | None) -> datetime | None:
    return None if sec is None else datetime.fromtimestamp(int(sec), tz=timezone.utc)


def run_id(model: str, issue: datetime) -> str:
    return f"{model}_run{issue:%Y%m%dT%H%M}"


def _http_get(url: str, params: dict | None, ua: str, timeout=FORECAST_TIMEOUT) -> requests.Response:
    r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": ua})
    if r.status_code == 429 or (r.status_code >= 400 and "limit" in r.text[:500].lower()):
        raise RateLimited(f"HTTP {r.status_code}: {r.text[:200]}")
    r.raise_for_status()
    return r


def get_with_retry(url: str, params: dict | None, ua: str, getter=_http_get,
                   timeout=FORECAST_TIMEOUT, wait_s: float | None = None, retries: int = 1):
    """One attempt + ``retries`` retries after ``wait_s``; HTTP 429 is never retried."""
    wait_s = RETRY_WAIT_S if wait_s is None else wait_s
    for attempt in range(retries + 1):
        try:
            return getter(url, params, ua, timeout=timeout)
        except RateLimited:
            raise
        except Exception as e:
            if attempt == retries:
                raise
            C.log("retry", url=url.split("?")[0], error=repr(e)[:300], wait_s=wait_s)
            _sleep(wait_s)


def load_rule_delays(path: Path = RULE_FILE) -> dict[str, int]:
    """{model: delay_min} of the P2B.6 availability rule (same file the history Silver uses)."""
    if not Path(path).is_file():
        return {}
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return {m: int(v["delay_min"]) for m, v in (doc.get("models") or {}).items()}


def rule_run(now: datetime, delay_min: int) -> dict:
    """Conservative issue-time rule (P2A / P2B.6): the newest 6-hourly run whose rule availability
    (issue + delay) is not after ``now``. The run Open-Meteo actually serves is this one or newer,
    so the issue time is a lower bound; availability = issue + delay (never earlier than the rule)."""
    t = now - timedelta(minutes=delay_min)
    issue = t.replace(hour=t.hour - t.hour % CYCLE_H, minute=0, second=0, microsecond=0)
    return {"issue": issue, "availability": issue + timedelta(minutes=delay_min),
            "components": None, "availability_source": "rule_fallback", "rule_delay_min": delay_min}


def model_meta(oc: dict, model: str, getter=_http_get, retries: int = 1) -> dict:
    """Metadata of one model ({issue, availability, components}); raises on any component failure."""
    comps = {}
    for mm in oc["models"][model]["meta_models"]:
        j = get_with_retry(oc["metadata_endpoint"].format(meta_model=mm), None, oc["user_agent"], getter,
                           timeout=META_TIMEOUT, wait_s=META_RETRY_WAIT_S, retries=retries).json()
        comps[mm] = {k: j.get(k) for k in ("last_run_initialisation_time", "last_run_availability_time",
                                           "last_run_modification_time", "update_interval_seconds",
                                           "temporal_resolution_seconds")}
    issue = max(ts(c["last_run_initialisation_time"]) for c in comps.values())
    avail = max(ts(c["last_run_availability_time"]) for c in comps.values())
    return {"issue": issue, "availability": avail, "components": comps, "availability_source": "metadata"}


def model_runs(oc: dict, getter=_http_get, now: datetime | None = None, delays: dict | None = None) -> dict:
    """{model: run} — metadata per model, isolated; on failure the conservative rule (P3B hotfix).
    A model without metadata and without a rule gets {"error": ...}."""
    out = {}
    for model in oc["models"]:
        try:
            out[model] = model_meta(oc, model, getter)
        except RateLimited:
            raise
        except META_ERRORS as exc:
            C.log("metadata_failed", source=SOURCE, model=model, error=repr(exc)[:300])
            delays = load_rule_delays() if delays is None else delays
            if model not in delays:
                out[model] = {"error": f"metadata failed and no availability rule: {repr(exc)[:200]}"}
                continue
            r = rule_run(now or C.utcnow(), delays[model])
            r["metadata_error"] = repr(exc)[:300]
            out[model] = r
            C.log("rule_fallback", source=SOURCE, model=model, issue=C.iso(r["issue"]),
                  availability=C.iso(r["availability"]), delay_min=delays[model])
    return out


def request_params(oc: dict, models: list[str]) -> dict:
    pts = oc["points"]
    return {"latitude": ",".join(f"{p[1]:.2f}" for p in pts),
            "longitude": ",".join(f"{p[2]:.2f}" for p in pts),
            "hourly": oc["variable"], "models": ",".join(models), "forecast_days": oc["forecast_days"],
            "timezone": "GMT", "timeformat": "unixtime"}


def parse_response(blob: bytes, oc: dict, model: str, n_models: int, issue: datetime,
                   ingested: datetime, run_meta: dict) -> pd.DataFrame:
    data = json.loads(blob)
    if isinstance(data, dict):
        data = [data]
    pts = oc["points"]
    if len(data) != len(pts):
        raise ValueError(f"{len(data)} locations returned, {len(pts)} requested")
    var = oc["variable"]
    frames = []
    for p, loc in zip(pts, data):
        h = loc["hourly"]
        col = f"{var}_{model}" if f"{var}_{model}" in h else (var if n_models == 1 else None)
        if col is None:
            raise ValueError(f"no column for {model} in response")
        end = pd.to_datetime(np.asarray(h["time"], dtype="int64"), unit="s", utc=True)
        vals = np.asarray([np.nan if v is None else v for v in h[col]], dtype="float64")
        frames.append(pd.DataFrame({"pixel_id": int(p[0]), "role": p[3], "lat": p[1], "lon": p[2],
                                    "om_grid_lat": loc.get("latitude"), "om_grid_lon": loc.get("longitude"),
                                    "interval_end_utc": end, "precip_mm": vals}))
    df = pd.concat(frames, ignore_index=True)
    df["interval_start_utc"] = df["interval_end_utc"] - pd.Timedelta(hours=1)
    issue_ts = pd.Timestamp(issue)
    df = df[df["interval_start_utc"] >= issue_ts].copy()
    df["model"] = model
    df["forecast_issue_time_utc"] = issue_ts
    df["issue_time_rule"] = ISSUE_RULE[run_meta.get("availability_source", "metadata")]
    df["availability_source"] = run_meta.get("availability_source", "metadata")
    df["model_availability_time_utc"] = pd.Timestamp(run_meta["availability"])
    df["ingestion_time_utc"] = pd.Timestamp(ingested)
    df["available_time_utc"] = df["ingestion_time_utc"]
    df["lead_end_min"] = (df["interval_end_utc"] - df["forecast_issue_time_utc"]).dt.total_seconds() / 60
    return df.sort_values(["pixel_id", "interval_start_utc"]).reset_index(drop=True)


def to_parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


def _stored_issues(store: C.Store, model: str, dates: list[str]) -> list[datetime]:
    out = []
    for i in C.existing_ids(store, SOURCE, dates):
        if i.rsplit("_run", 1)[0] == model:
            out.append(datetime.strptime(i.rsplit("_run", 1)[1], "%Y%m%dT%H%M").replace(tzinfo=timezone.utc))
    return out


def run(store: C.Store, cfg: dict, getter=_http_get, now: datetime | None = None,
        delays: dict | None = None) -> dict:
    """Each model is processed independently; per-model status in ``res["models"]``:
    published | skipped_existing | skipped_changed | failed (P3B hotfix)."""
    oc = cfg["openmeteo"]
    t0 = time.time()
    res = {"source": SOURCE, "runs": {}, "models": {}, "published": [], "skipped_existing": [],
           "skipped_changed": [], "failed": [], "forecast_requests": 0}
    before = model_runs(oc, getter, now=now, delays=delays)

    def status(m, st, **kw):
        res["models"][m] = {"status": st, "availability_source": before[m].get("availability_source"), **kw}
        if st == "skipped_existing":
            res[st].append(res["runs"][m]["id"])
        elif st in ("skipped_changed", "failed"):
            res[st].append(m)

    todo = []
    for m, r in before.items():
        if "error" in r:
            status(m, "failed", error=r["error"])
            continue
        rid = run_id(m, r["issue"])
        res["runs"][m] = {"id": rid, "issue": C.iso(r["issue"]), "availability": C.iso(r["availability"]),
                          "availability_source": r["availability_source"]}
        d0 = r["issue"].strftime("%Y-%m-%d")
        if r["availability_source"] == "rule_fallback":
            # the rule issue is a lower bound: a stored run at least as new already holds this data
            days = sorted({d0, (r["issue"] + timedelta(days=1)).strftime("%Y-%m-%d"),
                           (now or C.utcnow()).strftime("%Y-%m-%d")})
            newer = [i for i in _stored_issues(store, m, days) if i >= r["issue"]]
            if newer:
                res["runs"][m]["stored_newer_or_equal"] = run_id(m, max(newer))
                status(m, "skipped_existing")
                continue
        elif rid in C.existing_ids(store, SOURCE, [d0]):
            status(m, "skipped_existing")
            continue
        todo.append(m)
    if todo:
        params = request_params(oc, todo)
        started = C.utcnow()
        try:
            r = get_with_retry(oc["api"], params, oc["user_agent"], getter, timeout=FORECAST_TIMEOUT)
        except RateLimited:
            raise
        except requests.RequestException as exc:
            C.log("forecast_failed", source=SOURCE, models=todo, error=repr(exc)[:300])
            for m in todo:
                status(m, "failed", error=f"forecast request: {repr(exc)[:200]}")
            todo = []
        else:
            ingested = C.utcnow()
            res["forecast_requests"] += 1
            blob = r.content
            url = requests.Request("GET", oc["api"], params=params).prepare().url
        for m in todo:
            b = before[m]
            after_check = "not_applicable"
            if b["availability_source"] == "metadata":
                # single attempt (no retry) keeps the worst-case job time bounded
                try:
                    after = model_meta(oc, m, getter, retries=0)
                    after_check = "same_run"
                    if after["issue"] != b["issue"]:
                        status(m, "skipped_changed")
                        continue
                except RateLimited:
                    raise
                except META_ERRORS as exc:
                    after_check = "failed"
                    C.log("metadata_after_failed", source=SOURCE, model=m, error=repr(exc)[:300])
            issue = b["issue"]
            rid = run_id(m, issue)
            try:
                df = parse_response(blob, oc, m, len(todo), issue, ingested, b)
            except (ValueError, KeyError, TypeError) as exc:
                C.log("parse_failed", source=SOURCE, model=m, error=repr(exc)[:300])
                status(m, "failed", error=f"parse: {repr(exc)[:200]}")
                continue
            data = to_parquet_bytes(df)
            man = {
                "source": SOURCE, "id": rid, "model": m, "variable": oc["variable"],
                "forecast_issue_time_utc": C.iso(issue),
                "issue_time_rule": ISSUE_RULE[b["availability_source"]],
                "availability_source": b["availability_source"],
                "model_availability_time_utc": C.iso(b["availability"]),
                "metadata": b["components"],
                "metadata_endpoint": oc["metadata_endpoint"],
                "metadata_after_check": after_check,
                "request_url": url, "request_started_utc": C.iso(started),
                "ingestion_time_utc": C.iso(ingested), "available_time_utc": C.iso(ingested),
                "raw_response_bytes": len(blob), "raw_response_sha256": C.sha256_bytes(blob),
                "rows": len(df), "points": int(df["pixel_id"].nunique()),
                "lead_end_min_range": [float(df["lead_end_min"].min()), float(df["lead_end_min"].max())]
                if len(df) else None,
                "nan_share": round(float(df["precip_mm"].isna().mean()), 6) if len(df) else None,
                "license": oc["license"],
                "data_time_utc": C.iso(issue), "run": C.run_context(),
            }
            if b["availability_source"] == "rule_fallback":
                man["rule_delay_min"] = b["rule_delay_min"]
                man["rule_file"] = "configs/openmeteo_availability.yaml"
                man["metadata_error"] = b.get("metadata_error")
            pub = C.publish(store, SOURCE, issue.strftime("%Y-%m-%d"), rid, "parquet", data, man)
            res["published"].append(pub)
            status(m, "published", id=rid)
            C.log("published", availability_source=b["availability_source"], **pub)
    res["seconds"] = round(time.time() - t0, 1)
    res["store"] = store.stats()
    return res


def exit_code(res: dict) -> int:
    """0 when at least one model was published or skipped (existing / changed); 3 when every model failed."""
    sts = [v["status"] for v in res["models"].values()]
    return 3 if sts and all(s == "failed" for s in sts) else 0


def main(argv=None, getter=_http_get) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="r2")
    a = ap.parse_args(argv)
    cfg = C.load_config()
    store = C.open_store(a.store, cfg)
    try:
        res = run(store, cfg, getter=getter)
    except RateLimited as exc:
        C.log("rate_limited", source=SOURCE, error=str(exc)[:300])
        return 4
    except requests.RequestException as exc:
        C.log("source_failed", source=SOURCE, error=repr(exc)[:300])
        return 3
    code = exit_code(res)
    C.log("summary", exit_code=code, **res)
    return code


if __name__ == "__main__":
    raise SystemExit(C.cli(main))
