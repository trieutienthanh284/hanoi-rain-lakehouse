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
  a new run; HTTP 429 -> no retry, exit 4; other errors: one retry after 60 s, exit 3.

Objects: lake/v1/realtime/openmeteo/date=D/{id}.parquet + manifest JSON.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.realtime import common as C  # noqa: E402

SOURCE = "openmeteo"
RETRY_WAIT_S = 60


class RateLimited(RuntimeError):
    pass


def ts(sec: int | None) -> datetime | None:
    return None if sec is None else datetime.fromtimestamp(int(sec), tz=timezone.utc)


def run_id(model: str, issue: datetime) -> str:
    return f"{model}_run{issue:%Y%m%dT%H%M}"


def _http_get(url: str, params: dict | None, ua: str, timeout: int = 120) -> requests.Response:
    r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": ua})
    if r.status_code == 429 or (r.status_code >= 400 and "limit" in r.text[:500].lower()):
        raise RateLimited(f"HTTP {r.status_code}: {r.text[:200]}")
    r.raise_for_status()
    return r


def get_with_retry(url: str, params: dict | None, ua: str, getter=_http_get):
    try:
        return getter(url, params, ua)
    except RateLimited:
        raise
    except Exception as e1:
        C.log("retry", url=url.split("?")[0], error=repr(e1)[:300], wait_s=RETRY_WAIT_S)
        time.sleep(RETRY_WAIT_S)
        return getter(url, params, ua)


def model_runs(oc: dict, getter=_http_get) -> dict:
    """{model: {issue, availability, components{meta_model: {...}}}} from metadata."""
    out = {}
    for model, spec in oc["models"].items():
        comps = {}
        for mm in spec["meta_models"]:
            j = get_with_retry(oc["metadata_endpoint"].format(meta_model=mm), None, oc["user_agent"], getter).json()
            comps[mm] = {k: j.get(k) for k in ("last_run_initialisation_time", "last_run_availability_time",
                                               "last_run_modification_time", "update_interval_seconds",
                                               "temporal_resolution_seconds")}
        issue = max(ts(c["last_run_initialisation_time"]) for c in comps.values())
        avail = max(ts(c["last_run_availability_time"]) for c in comps.values())
        out[model] = {"issue": issue, "availability": avail, "components": comps}
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
    df["issue_time_rule"] = "open-meteo metadata last_run_initialisation_time (max over components)"
    df["model_availability_time_utc"] = pd.Timestamp(run_meta["availability"])
    df["ingestion_time_utc"] = pd.Timestamp(ingested)
    df["available_time_utc"] = df["ingestion_time_utc"]
    df["lead_end_min"] = (df["interval_end_utc"] - df["forecast_issue_time_utc"]).dt.total_seconds() / 60
    return df.sort_values(["pixel_id", "interval_start_utc"]).reset_index(drop=True)


def to_parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


def run(store: C.Store, cfg: dict, getter=_http_get) -> dict:
    oc = cfg["openmeteo"]
    t0 = time.time()
    res = {"source": SOURCE, "runs": {}, "published": [], "skipped_existing": [], "skipped_changed": [],
           "forecast_requests": 0}
    before = model_runs(oc, getter)
    todo = []
    for m, r in before.items():
        rid = run_id(m, r["issue"])
        res["runs"][m] = {"id": rid, "issue": C.iso(r["issue"]), "availability": C.iso(r["availability"])}
        if rid in C.existing_ids(store, SOURCE, [r["issue"].strftime("%Y-%m-%d")]):
            res["skipped_existing"].append(rid)
        else:
            todo.append(m)
    if todo:
        params = request_params(oc, todo)
        started = C.utcnow()
        r = get_with_retry(oc["api"], params, oc["user_agent"], getter)
        ingested = C.utcnow()
        res["forecast_requests"] += 1
        blob = r.content
        after = model_runs(oc, getter)
        url = requests.Request("GET", oc["api"], params=params).prepare().url
        for m in todo:
            if after[m]["issue"] != before[m]["issue"]:
                res["skipped_changed"].append(m)
                continue
            issue = before[m]["issue"]
            df = parse_response(blob, oc, m, len(todo), issue, ingested, before[m])
            data = to_parquet_bytes(df)
            rid = run_id(m, issue)
            man = {
                "source": SOURCE, "id": rid, "model": m, "variable": oc["variable"],
                "forecast_issue_time_utc": C.iso(issue),
                "issue_time_rule": "metadata last_run_initialisation_time; max over components (upper bound)",
                "model_availability_time_utc": C.iso(before[m]["availability"]),
                "metadata": before[m]["components"],
                "metadata_endpoint": oc["metadata_endpoint"],
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
            pub = C.publish(store, SOURCE, issue.strftime("%Y-%m-%d"), rid, "parquet", data, man)
            res["published"].append(pub)
            C.log("published", **pub)
    res["seconds"] = round(time.time() - t0, 1)
    res["store"] = store.stats()
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="r2")
    a = ap.parse_args(argv)
    cfg = C.load_config()
    store = C.open_store(a.store, cfg)
    try:
        res = run(store, cfg)
    except RateLimited as exc:
        C.log("rate_limited", source=SOURCE, error=str(exc)[:300])
        return 4
    except requests.RequestException as exc:
        C.log("source_failed", source=SOURCE, error=repr(exc)[:300])
        return 3
    C.log("summary", **res)
    return 0


if __name__ == "__main__":
    raise SystemExit(C.cli(main))
