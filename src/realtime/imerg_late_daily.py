"""IMERG Late V07 daily ingestion (provisional labels) -> Hanoi day file -> R2.

  python -m src.realtime.imerg_late_daily [--store r2|local:DIR] [--date YYYY-MM-DD] [--max-days N]

* Candidate days: yesterday back to ``imerg_late.lookback_days`` (UTC), newest
  first. A day is processed when it is not yet in R2 and CMR lists all 48
  half-hourly granules (complete). At most ``max_days_per_run`` days per run,
  so a missed schedule is caught up on the following runs.
* Retrieval reuses the P2A OPeNDAP path unchanged
  (``imerg_subset_collector.crop_day_opendap``; ADD-004 provenance; the 1st of
  each month is also verified by full download with exact array match). Its
  writers are redirected into the runner working directory.
* IMERG Late is provisional: it is superseded later by IMERG Final, and labels
  derived from it are marked ``label_status = provisional``.

Objects: lake/v1/realtime/imerg_late/date=D/imerg_late_YYYYMMDD.h5 + manifest JSON
(same day-file format as the backfill; manifest carries the per-granule lines).
"""
from __future__ import annotations

import argparse
import contextlib
import sys
import time
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion import imerg_subset_collector as isc  # noqa: E402
from src.lake import regional_crop as rc  # noqa: E402
from src.realtime import common as C  # noqa: E402

SOURCE = "imerg_late"


def day_id(d: str) -> str:
    return f"imerg_late_{d.replace('-', '')}"


def candidate_days(today: date, lookback: int) -> list[str]:
    return [(today - timedelta(days=k)).isoformat() for k in range(1, lookback + 1)]


def select_days(cands: list[str], existing: set[str], n_granules, expected: int, max_days: int) -> tuple[list[str], dict]:
    """Newest complete days not yet in R2. ``n_granules(day)`` is only called for missing days."""
    chosen, notes = [], {}
    for d in cands:
        if len(chosen) >= max_days:
            break
        if day_id(d) in existing:
            notes[d] = "exists"
            continue
        n = n_granules(d)
        if n < expected:
            notes[d] = f"incomplete ({n}/{expected} granules in CMR)"
            continue
        chosen.append(d)
        notes[d] = "selected"
    return chosen, notes


def cmr_count(d: str) -> int:
    return len([g for g in isc.cmr_granules_umm("late", d) if g["opendap"]])


def regional_env(work: Path):
    """Redirect the P2A writers into the working directory. ``rc.ROOT`` is redirected
    too because ``rc.rel`` expresses paths relative to it, and on GitHub runners
    $RUNNER_TEMP lies outside the repository checkout."""
    stack = contextlib.ExitStack()
    stack.enter_context(C.redirected(rc, REGIONAL_ROOT=work / "regional", ROOT=work))
    stack.enter_context(C.redirected(isc, SCRATCH=work / "scratch"))
    return stack


def process_day(d: str, work: Path, dom: dict) -> tuple[bytes, list[dict], dict]:
    with regional_env(work):
        r = isc.crop_day_opendap("late", d, dom, None)
        lines = [x for x in rc.read_manifest(SOURCE) if x["date"] == d]
        data = rc.crop_path(SOURCE, d).read_bytes() if r.get("windows") else b""
    key = C.data_key(SOURCE, d, day_id(d), "h5")
    for x in lines:
        x["crop_file"] = key            # the runner-local path is meaningless after the run
    return data, lines, r


def run(store: C.Store, cfg: dict, days: list[str] | None = None, max_days: int | None = None,
        today: date | None = None, counter=cmr_count, processor=None) -> dict:
    ic = cfg["imerg_late"]
    expected = ic["expected_granules"]
    max_days = ic["max_days_per_run"] if max_days is None else max_days
    today = today or C.utcnow().date()
    cands = days or candidate_days(today, ic["lookback_days"])
    have = C.existing_ids(store, SOURCE, cands)
    chosen, notes = select_days(cands, have, counter, expected, max_days)
    work = C.workdir(cfg, SOURCE)
    dom = rc.load_domain()
    processor = processor or (lambda d: process_day(d, work, dom))
    res = {"source": SOURCE, "candidates": notes, "published": [], "failed": []}
    for d in chosen:
        t0 = time.time()
        started = C.utcnow()
        try:
            data, lines, r = processor(d)
        except Exception as exc:
            res["failed"].append({"date": d, "error": repr(exc)[:500]})
            C.log("day_failed", date=d, error=repr(exc)[:500])
            continue
        ingested = C.utcnow()
        if len(lines) < expected or not data:
            res["failed"].append({"date": d, "error": f"only {len(lines)} windows"})
            continue
        man = {
            "source": SOURCE, "id": day_id(d), "date": d, "product": "GPM_3IMERGHHL V07 (IMERG Late)",
            "label_status": cfg["imerg_late"]["label_status"],
            "retrieval_path": "opendap_dap4 (ADD-004)",
            "n_windows": len(lines),
            "window_start_utc": lines[0]["window_start_utc"], "window_end_utc": lines[-1]["window_end_utc"],
            "retrieval_started_utc": C.iso(started), "ingestion_time_utc": C.iso(ingested),
            "available_time_rule": "per window: CMR production time is the publication evidence; "
                                   "realtime use only after ingestion_time_utc",
            "verification": r.get("verification"), "download_bytes": r.get("download_bytes"),
            "seconds": round(time.time() - t0, 1),
            "crop_sha256": C.sha256_bytes(data),
            "data_time_utc": lines[-1]["window_end_utc"],
            "granules": lines, "run": C.run_context(),
        }
        pub = C.publish(store, SOURCE, d, day_id(d), "h5", data, man)
        res["published"].append(pub | {"seconds": man["seconds"]})
        C.log("published", **pub)
    res["store"] = store.stats()
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="r2")
    ap.add_argument("--date", action="append", help="process this UTC day (repeatable) instead of the lookback")
    ap.add_argument("--max-days", type=int, default=None)
    a = ap.parse_args(argv)
    cfg = C.load_config()
    store = C.open_store(a.store, cfg)
    res = run(store, cfg, a.date, a.max_days)
    C.log("summary", **res)
    return 1 if res["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(C.cli(main))
