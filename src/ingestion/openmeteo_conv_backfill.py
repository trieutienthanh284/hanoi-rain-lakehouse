"""P4G.1 convective backfill (cloud job `nwpconv_backfill`): GFS CAPE and lifted index, *_previous_day1, from the
Open-Meteo Previous Runs API for the 81 set-B points, 2024-05-01..2025-10-31 (configs/model_p4fg.yaml
pre_registered.convective, DECISIONS D214). Never the Historical Forecast API (run-availability leakage).

Politeness (free non-commercial API, CC BY 4.0, "Weather data by Open-Meteo.com"; terms recorded in
docs/agent_runs/openmeteo_feasibility.md): strictly sequential, one request per block of <= 13 days (+1 day) for all 81
points, >= conv_store.min_interval_s between requests, one retry after 60 s on a non-429 error, HTTP 429 -> stop (no
retry, STOP_429.json under the run prefix, exit 4). Resumable: a block whose Bronze object exists is not requested again.

R2 (put-if-absent; identical bytes accepted, other bytes = error):
  Bronze  lake/v1/bronze/openmeteo_conv/block_start=YYYY-MM-DD/response.json.gz + request.json
  Silver  lake/v1/silver/nwp_conv/date=YYYY-MM-DD/part.parquet + dq.json
The 81 points are read from the Open-Meteo Silver on R2 (set B), not from local reference files.
Mapping (D214): time label T (instantaneous value); forecast_issue_time = floor_6h(T) - 24 h (as precipitation
previous_day1, P2A.2); available_time = issue + 420 min (configs/openmeteo_availability.yaml gfs_global).
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import time
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

API = "https://previous-runs-api.open-meteo.com/v1/forecast"
MODEL = "gfs_global"
VARS = ("cape", "lifted_index")
SUFFIX = "_previous_day1"
ISSUE_OFFSET_H = 24
AVAILABLE_DELAY_MIN = 420
USER_AGENT = "hanoi-flood-lakehouse-research/0.1 (non-commercial thesis prototype)"
POINTS_DAY = "2024-05-01"
LICENCE = "CC BY 4.0, Weather data by Open-Meteo.com (free non-commercial API)"


class RateLimited(RuntimeError):
    pass


def blocks(start: date, end: date, n: int = 13) -> list[tuple[str, str]]:
    out, d = [], start
    while d <= end:
        e = min(d + timedelta(days=n - 1), end)
        out.append((d.isoformat(), e.isoformat()))
        d = e + timedelta(days=1)
    return out


def points(store) -> pd.DataFrame:
    """The 81 set-B points (pixel_id, lat, lon) from the Open-Meteo Silver on R2."""
    k = f"lake/v1/data/silver/met/openmeteo/date={POINTS_DAY}/part.parquet"
    if k not in store.list(k):
        raise RuntimeError(f"{k} missing: cannot read the set-B points")
    df = pd.read_parquet(io.BytesIO(store.get(k)), columns=["pixel_id", "lat", "lon"])
    p = df.drop_duplicates("pixel_id").sort_values("pixel_id").reset_index(drop=True)
    if len(p) != 81:
        raise RuntimeError(f"expected 81 set-B points, found {len(p)}")
    return p


def request_params(a: str, b: str, pts: pd.DataFrame) -> dict:
    e = (date.fromisoformat(b) + timedelta(days=1)).isoformat()
    return {"latitude": ",".join(f"{x:.2f}" for x in pts["lat"]), "longitude": ",".join(f"{x:.2f}" for x in pts["lon"]),
            "hourly": ",".join(f"{v}{SUFFIX}" for v in VARS), "models": MODEL, "start_date": a, "end_date": e,
            "timezone": "GMT", "timeformat": "unixtime"}


def fetch(params: dict, http_get, sleep=time.sleep, retry_wait_s: float = 60.0) -> tuple[bytes, dict]:
    def once():
        r = http_get(API, params=params, timeout=180, headers={"User-Agent": USER_AGENT})
        if r.status_code == 429 or "limit exceeded" in (r.text or "")[:300].lower():
            raise RateLimited(f"HTTP {r.status_code}: {(r.text or '')[:200]}")
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {(r.text or '')[:200]}")
        return r
    note = None
    try:
        r = once()
    except RateLimited:
        raise
    except Exception as e1:                                       # noqa: BLE001 - one retry, then stop the source
        sleep(retry_wait_s)
        r = once()
        note = f"retried after: {e1!r}"[:300]
    return r.content, {"status": r.status_code, "retry_note": note, "url": getattr(r, "url", API),
                       "request_utc": datetime.now(timezone.utc).isoformat()}


def parse(blob: bytes, pts: pd.DataFrame, request_utc: str) -> pd.DataFrame:
    data = json.loads(blob)
    if isinstance(data, dict):
        data = [data]
    if len(data) != len(pts):
        raise ValueError(f"{len(data)} locations returned, {len(pts)} requested")
    frames = []
    for k, loc in enumerate(data):
        h = loc["hourly"]
        t = pd.to_datetime(np.asarray(h["time"], dtype="int64"), unit="s", utc=True)
        f = pd.DataFrame({"pixel_id": np.int64(pts["pixel_id"].iloc[k]), "lat": float(pts["lat"].iloc[k]),
                          "lon": float(pts["lon"].iloc[k]), "om_grid_lat": loc.get("latitude"),
                          "om_grid_lon": loc.get("longitude"), "model": MODEL, "time_utc": t})
        for v in VARS:
            col = next((c for c in (f"{v}{SUFFIX}", f"{v}{SUFFIX}_{MODEL}") if c in h), None)
            vals = h[col] if col else [None] * len(t)
            f[v] = np.asarray([np.nan if x is None else x for x in vals], dtype="float64")
        f["forecast_issue_time_utc"] = f["time_utc"].dt.floor("6h") - pd.Timedelta(hours=ISSUE_OFFSET_H)
        f["available_time_utc"] = f["forecast_issue_time_utc"] + pd.Timedelta(minutes=AVAILABLE_DELAY_MIN)
        f["issue_time_rule"] = "floor_6h(T) - 24 h (previous_day1, as precipitation P2A.2); available = issue + 420 min"
        f["source_offset"] = "previous_day1"
        f["ingestion_time_utc"] = pd.Timestamp(request_utc).tz_convert("UTC")
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def silver_day(df: pd.DataFrame, day: str) -> tuple[bytes, dict]:
    sel = df[df["time_utc"].dt.strftime("%Y-%m-%d") == day].sort_values(["pixel_id", "time_utc"]).reset_index(drop=True)
    for c in ("time_utc", "forecast_issue_time_utc", "available_time_utc", "ingestion_time_utc"):
        sel[c] = sel[c].astype("datetime64[us, UTC]")
    dups = int(sel.duplicated(["pixel_id", "model", "time_utc"]).sum())
    dq = {"source": "nwp_conv", "date": day, "rows": int(len(sel)), "points": int(sel["pixel_id"].nunique()),
          "hours": int(sel["time_utc"].nunique()), "duplicates": dups,
          "nan_share": {v: round(float(sel[v].isna().mean()), 6) for v in VARS},
          "available_before_time": bool((sel["available_time_utc"] <= sel["time_utc"]).all()),
          "licence": LICENCE}
    dq["dq_pass"] = bool(len(sel) and dups == 0 and dq["points"] == 81 and dq["hours"] == 24
                         and dq["available_before_time"])
    buf = io.BytesIO()
    sel.to_parquet(buf, index=False)
    return buf.getvalue(), dq


def put_ok(store, key: str, data: bytes) -> str:
    """put-if-absent; identical bytes already there = ok, other bytes raise (never overwritten)."""
    return store.put(key, data)


def run_block(store, cfg: dict, a: str, b: str, pts: pd.DataFrame, http_get, state: dict, sleep=time.sleep) -> dict:
    cs = cfg["conv_store"]
    bk = f"{cs['bronze_prefix']}block_start={a}/"
    have = store.list(bk)
    if f"{bk}response.json.gz" in have and f"{bk}request.json" in have:
        blob = gzip.decompress(store.get(f"{bk}response.json.gz"))
        req = json.loads(store.get(f"{bk}request.json"))
        fetched = False
    else:
        gap = time.time() - state.get("last", 0.0)
        if state.get("last") and gap < float(cs["min_interval_s"]):
            sleep(float(cs["min_interval_s"]) - gap)
        blob, meta = fetch(request_params(a, b, pts), http_get, sleep, float(cs["retry_wait_s"]))
        state["last"] = time.time()
        req = {"block": [a, b], "api": API, "model": MODEL, "variables": [f"{v}{SUFFIX}" for v in VARS],
               "points": pts["pixel_id"].astype(int).tolist(), "raw_sha256": hashlib.sha256(blob).hexdigest(),
               "raw_bytes": len(blob), "licence": LICENCE, **meta}
        put_ok(store, f"{bk}response.json.gz", gzip.compress(blob, mtime=0))
        put_ok(store, f"{bk}request.json", json.dumps(req, indent=1).encode())
        fetched = True
    df = parse(blob, pts, req["request_utc"])
    days = [d.date().isoformat() for d in pd.date_range(a, b, freq="D")]
    out = {"block": [a, b], "fetched": fetched, "days": {}}
    for d in days:
        data, dq = silver_day(df, d)
        sp = f"{cs['silver_prefix']}date={d}/"
        put_ok(store, sp + "part.parquet", data)
        put_ok(store, sp + "dq.json", json.dumps(dq, indent=1).encode())
        out["days"][d] = {k: dq[k] for k in ("rows", "points", "hours", "nan_share", "dq_pass")}
    return out
