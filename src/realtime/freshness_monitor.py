"""Freshness monitor: age of the newest realtime record per source -> lake/v1/status/freshness.json.

  python -m src.realtime.freshness_monitor [--store r2|local:DIR]

* Reads only manifest objects (newest date partitions first, within
  ``sources.<s>.lookback_days``), takes the newest manifest per stream
  (Open-Meteo: one stream per model) and its ``data_time_field``:
      gsmap_now  -> window_end_utc           (stale after 120 min by default)
      openmeteo  -> model_availability_time_utc (stale after 720 min; D27)
      imerg_late -> window_end_utc of the day (stale after 2880 min)
  Thresholds live in configs/realtime.yaml. Open-Meteo manifests written with
  ``availability_source = "rule_fallback"`` (P3B hotfix) are valid records: their
  ``model_availability_time_utc`` is the conservative rule time (issue + delay).
* Exit code 1 when any source/stream is stale or has no record: a failed
  scheduled run makes GitHub e-mail the repository owner (that is the alert).
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.realtime import common as C  # noqa: E402

STATUS_KEY = "lake/v1/status/freshness.json"


def stream_of(source: str, obj_id: str) -> str:
    return obj_id.rsplit("_run", 1)[0] if source == "openmeteo" else source


def newest_manifests(store: C.Store, source: str, now: datetime, lookback: int) -> dict[str, str]:
    """{stream: newest manifest key} scanning date partitions from today backwards.
    Ids embed a sortable timestamp, so the lexicographic max is the newest."""
    found: dict[str, str] = {}
    for k in range(lookback + 1):
        d = (now - timedelta(days=k)).strftime("%Y-%m-%d")
        for key in store.list(C.manifest_prefix(source, d)):
            if not key.endswith(".json"):
                continue
            s = stream_of(source, C.id_from_manifest_key(key))
            if s not in found or C.id_from_manifest_key(key) > C.id_from_manifest_key(found[s]):
                found[s] = key
        # partitions are scanned newest first; stop once every expected stream has a record
        if found and source != "openmeteo":
            break
    return found


def evaluate(now: datetime, records: dict[str, dict], cfg: dict) -> dict:
    """records: {source: {stream: {"data_time_utc": iso|None, "ingestion_time_utc": iso|None, "key": str}}}.
    Pure function (unit-tested)."""
    out, stale = {}, []
    for source, scfg in cfg["sources"].items():
        limit = scfg["stale_after_minutes"]
        streams = records.get(source) or {}
        if not streams:
            out[source] = {"stale": True, "reason": "no record found", "threshold_min": limit}
            stale.append(source)
            continue
        entry = {"threshold_min": limit, "streams": {}}
        for s, r in sorted(streams.items()):
            dt = r.get("data_time_utc")
            age = None if dt is None else round((now - C.parse_iso(dt)).total_seconds() / 60, 1)
            ing = r.get("ingestion_time_utc")
            entry["streams"][s] = {"key": r.get("key"), "data_time_utc": dt, "ingestion_time_utc": ing,
                                   "age_min": age,
                                   "ingestion_age_min": None if ing is None else
                                   round((now - C.parse_iso(ing)).total_seconds() / 60, 1),
                                   "stale": age is None or age > limit}
            if r.get("availability_source"):
                entry["streams"][s]["availability_source"] = r["availability_source"]
        entry["age_min"] = max((v["age_min"] for v in entry["streams"].values() if v["age_min"] is not None),
                               default=None)
        entry["stale"] = any(v["stale"] for v in entry["streams"].values())
        if entry["stale"]:
            stale.append(source)
        out[source] = entry
    return {"checked_utc": C.iso(now), "overall": "STALE" if stale else "OK", "stale_sources": stale,
            "sources": out}


def run(store: C.Store, cfg: dict, now: datetime | None = None) -> dict:
    now = now or C.utcnow()
    records = {}
    for source, scfg in cfg["sources"].items():
        keys = newest_manifests(store, source, now, scfg["lookback_days"])
        records[source] = {}
        for s, key in keys.items():
            m = store.get_json(key)
            records[source][s] = {"key": key, "data_time_utc": m.get(scfg["data_time_field"]),
                                  "ingestion_time_utc": m.get("ingestion_time_utc"),
                                  "availability_source": m.get("availability_source")}
    status = evaluate(now, records, cfg)
    store.put_json(STATUS_KEY, status, overwrite=True)
    status["store"] = store.stats()
    return status


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="r2")
    a = ap.parse_args(argv)
    cfg = C.load_config()
    store = C.open_store(a.store, cfg)
    status = run(store, cfg)
    C.log("freshness", **status)
    return 1 if status["stale_sources"] else 0


if __name__ == "__main__":
    raise SystemExit(C.cli(main))
