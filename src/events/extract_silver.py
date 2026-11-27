"""U1.2 — write Silver flood-evidence records from `reports/u1/silver_specs.yaml`.

Every record: extraction_method = "automated", review_status = "pending". The passage
(`extracted_text`) must be found verbatim (normalised substring) in the Bronze artifact text;
any miss aborts before writing. Records go through the existing writer
`flood_evidence_silver_builder.write_silver_record` (validator included, skip_existing).

Usage: .venv/Scripts/python.exe -m src.events.extract_silver [--check]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import yaml

from src.events.common import REPORTS, ROOT, rel, sha256_file
from src.events.text_extract import artifact_text, is_verbatim

SPECS = REPORTS / "silver_specs.yaml"
INGEST = REPORTS / "ingest_inbox_run1.csv"
OUT_CSV = REPORTS / "silver_records_u1.csv"


def file_index(ingest: Path = INGEST) -> dict[str, dict[str, str]]:
    with open(ingest, encoding="utf-8", newline="") as f:
        return {r["file_name"]: r for r in csv.DictReader(f)}


def build_record(evidence_id: str, s: dict) -> dict:
    return {
        "evidence_id": evidence_id,
        "event_type": s["type"],
        "observed_time_start": s.get("t0"),
        "observed_time_end": s.get("t1"),
        "temporal_text": s.get("tt"),
        "location_text": s.get("loc"),
        "latitude": None,
        "longitude": None,
        "flood_depth_m": s.get("depth"),
        "flood_depth_text": s.get("depth_text"),
        "inundation_duration_hours": s.get("inund"),
        "rainfall_mm": float(s["mm"]) if s.get("mm") is not None else None,
        "rainfall_duration_hours": float(s["dur"]) if s.get("dur") is not None else None,
        "extracted_text": s["text"],
        "evidence_locator": s.get("locator"),
        "extraction_method": "automated",
        "confidence": s.get("conf", "medium"),
        "review_status": "pending",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verbatim + validation only, no writes")
    ap.add_argument("--specs", default=str(SPECS), help="spec yaml (U1.1b: reports/u1/u1b/silver_specs_agent.yaml)")
    ap.add_argument("--ingest", default=str(INGEST), help="ingest result CSV mapping file_name -> Bronze artifact")
    ap.add_argument("--out", default=str(OUT_CSV), help="record table CSV")
    a = ap.parse_args()
    specs_path, ingest_path, out_csv = Path(a.specs), Path(a.ingest), Path(a.out)
    from flood_evidence_silver_builder import build_record_id, write_silver_record
    from flood_evidence_silver_validator import load_contract, validate_silver_record

    contract = load_contract()
    specs = yaml.safe_load(specs_path.read_text(encoding="utf-8"))
    idx = file_index(ingest_path)
    planned, misses = [], []
    for fname, recs in specs.items():
        row = idx.get(fname)
        if row is None:
            misses.append(f"{fname}: not in ingest table")
            continue
        text = artifact_text(ROOT / row["bronze_artifact"])
        for i, s in enumerate(recs):
            if not is_verbatim(s["text"], text):
                misses.append(f"{fname}#{i}: passage not verbatim: {s['text'][:70]}")
                continue
            rec = build_record(row["evidence_id"], s)
            rid, _ = build_record_id(record=rec, contract=contract)
            validate_silver_record(record={**rec, "record_id": rid}, contract=contract)
            planned.append((fname, rec, rid))
    ids = Counter(r for _, _, r in planned)
    dup = [r for r, n in ids.items() if n > 1]
    print(f"SPEC_FILES={len(specs)} RECORDS={len(planned)} VERBATIM_MISSES={len(misses)} DUP_IDS={len(dup)}")
    for m in misses[:20]:
        print("  MISS", m)
    if misses or dup:
        return 2
    if a.check:
        return 0
    rows, actions = [], Counter()
    for fname, rec, _ in planned:
        res = write_silver_record(record=rec, contract=contract)
        actions[res["action"]] += 1
        rows.append({"file_name": fname, "evidence_id": rec["evidence_id"], "record_id": res["record_id"],
                     "action": res["action"], "event_type": rec["event_type"],
                     "location_text": rec["location_text"], "temporal_text": rec["temporal_text"],
                     "rainfall_mm": rec["rainfall_mm"], "flood_depth_m": rec["flood_depth_m"],
                     "record_path": rel(res["record_path"])})
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    by_type = Counter(r["event_type"] for r in rows)
    print(f"ACTIONS={dict(actions)} BY_TYPE={dict(by_type)} TABLE={rel(out_csv)}")
    print(f"SPECS_SHA256={sha256_file(specs_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
