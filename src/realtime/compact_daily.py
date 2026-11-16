"""Daily compaction of realtime GSMaP NOW windows into one backfill-format day file.

  python -m src.realtime.compact_daily [--store r2|local:DIR] [--date YYYY-MM-DD]

* Default: every day of the last ``compaction.lookback_days`` that is finished
  (now >= day end + ``min_hours_after_day_end``), has realtime windows and is
  not compacted yet. Put-if-absent: an existing compacted day is never replaced.
* Reads the one-window objects from lake/v1/realtime/gsmap_now/..., rebuilds the
  ``regional_crop.read_crop`` structure and writes the day file with
  ``regional_crop.write_day_file`` (identical layout to the backfill day files
  in data/bronze_regional/gsmap_now/, every array read back and compared).
* Output: lake/v1/realtime/compacted/gsmap_now/date=D/gsmap_now_YYYYMMDD.h5 and
  manifest JSON with one line per window (backfill manifest fields plus the
  realtime MDTM / ingestion times).
"""
from __future__ import annotations

import argparse
import io
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lake import regional_crop as rc  # noqa: E402
from src.realtime import common as C  # noqa: E402

SOURCE = "gsmap_now"
OUT_SOURCE = "compacted/gsmap_now"
VARS = ("hourlyPrecipRate", "hourlyPrecipRateGC")


def day_id(d: str) -> str:
    return f"gsmap_now_{d.replace('-', '')}"


def due_days(now: datetime, lookback: int, min_hours: float) -> list[str]:
    out = []
    for k in range(1, lookback + 1):
        d = (now - timedelta(days=k)).date()
        end = datetime(d.year, d.month, d.day, tzinfo=timezone.utc) + timedelta(days=1)
        if now >= end + timedelta(hours=min_hours):
            out.append(d.isoformat())
    return out


def window_from_manifest(m: dict) -> rc.Window:
    start, end = C.parse_iso(m["window_start_utc"]), C.parse_iso(m["window_end_utc"])
    return rc.Window(SOURCE, Path(m["original_name"]), start, end, m.get("product_version", "GSMaP_NOW_V8"))


def read_window_object(blob: bytes) -> dict:
    """One-window realtime object -> ``read_crop`` structure."""
    with h5py.File(io.BytesIO(blob), "r") as h:
        if int(h.attrs["n_windows"]) != 1:
            raise ValueError("expected a one-window object")
        vars_ = {n: (h[n][...], {k: h[n].attrs[k] for k in h[n].attrs}) for n in VARS}
        meta = {"lat": h["lat"][...], "lon": h["lon"][...]}
    return {"vars": vars_, "meta": meta}


def compact_day(store: C.Store, d: str, work: Path, dom: dict, dest: C.Store | None = None) -> dict | None:
    prefix = C.manifest_prefix(SOURCE, d)
    keys = sorted(k for k in store.list(prefix) if k.endswith(".json"))
    if not keys:
        return None
    mans = sorted((store.get_json(k) for k in keys), key=lambda m: m["window_start_utc"])
    mans = [m for m in mans if m["window_start_utc"].startswith(d)]
    ws, crops = [], []
    for m in mans:
        blob = store.get(m["object_key"])
        if C.sha256_bytes(blob) != m["object_sha256"]:
            raise RuntimeError(f"sha256 mismatch for {m['object_key']}")
        ws.append(window_from_manifest(m))
        crops.append(read_window_object(blob))
    with C.redirected(rc, REGIONAL_ROOT=work / "regional_tmp"):
        out, sha, var_names = rc.write_day_file(SOURCE, d, dom, ws, crops)
        data = out.read_bytes()
    now = C.iso(C.utcnow())
    lines = []
    for t, m in enumerate(mans):
        lines.append({
            "source": SOURCE, "date": d, "time_index": t,
            "original_name": m["original_name"], "original_path": f"jaxa-ftp:{m['remote_dir']}/{m['original_name']}",
            "original_size": m["original_size"], "original_sha256": m["original_sha256"],
            "window_start_utc": m["window_start_utc"], "window_end_utc": m["window_end_utc"],
            "product_version": m.get("product_version"),
            "remote_size": m["original_size"], "remote_mtime_utc": m["remote_mdtm_utc"],
            "ingestion_time_utc": m["ingestion_time_utc"], "available_time_utc": m["available_time_utc"],
            "realtime_object_key": m["object_key"], "realtime_object_sha256": m["object_sha256"],
            "crop_sha256": sha, "built_utc": now,
        })
    man = {"source": OUT_SOURCE, "id": day_id(d), "date": d, "n_windows": len(ws), "complete": len(ws) == 48,
           "variables": var_names, "format": "backfill day file (src/lake/regional_crop.write_day_file)",
           "crop_sha256": sha, "data_time_utc": mans[-1]["window_end_utc"],
           "ingestion_time_utc": max(m["ingestion_time_utc"] for m in mans),
           "lines": lines, "run": C.run_context()}
    pub = C.publish(dest or store, OUT_SOURCE, d, day_id(d), "h5", data, man)
    return pub | {"n_windows": len(ws), "complete": man["complete"]}


def run(store: C.Store, cfg: dict, days: list[str] | None = None, now: datetime | None = None,
        dest: C.Store | None = None) -> dict:
    cc = cfg["compaction"]
    now = now or C.utcnow()
    days = days or due_days(now, cc["lookback_days"], cc["min_hours_after_day_end"])
    done = C.existing_ids(dest or store, OUT_SOURCE, days)
    work = C.workdir(cfg, "compact")
    dom = rc.load_domain()
    res = {"source": OUT_SOURCE, "days": {}, "published": []}
    for d in days:
        if day_id(d) in done:
            res["days"][d] = "exists"
            continue
        pub = compact_day(store, d, work, dom, dest)
        if pub is None:
            res["days"][d] = "no realtime windows"
            continue
        res["days"][d] = f"compacted {pub['n_windows']} windows"
        res["published"].append(pub)
        C.log("published", **pub)
    res["store"] = store.stats()
    if dest is not None:
        res["dest_store"] = dest.stats()
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="r2")
    ap.add_argument("--date", action="append")
    ap.add_argument("--dest", default=None,
                    help="test aid: write the compacted day to another store (e.g. local:DIR); default = --store")
    a = ap.parse_args(argv)
    if a.date:
        for d in a.date:
            date.fromisoformat(d)
    cfg = C.load_config()
    store = C.open_store(a.store, cfg)
    dest = C.open_store(a.dest, cfg) if a.dest else None
    res = run(store, cfg, a.date, dest=dest)
    C.log("summary", **res)
    return 0


if __name__ == "__main__":
    raise SystemExit(C.cli(main))
