"""U1.1 — ingest inbox rows into Bronze flood evidence through the existing collector.

Thin wrapper: loops over `inbox.csv` (or `inbox_agent.csv`), filters valid rows and calls
`flood_evidence_collector.ingest_local_artifact` unchanged (manual_controlled, local file,
idempotent skip_existing). Never edits the inbox, never touches existing Bronze files.

Valid row: local file exists, `source_uri` non-empty, evidence_type supported, and the
notes do not end with the U0 pending marker.

Inbox dates are dates only; the Bronze contract needs tz-aware datetimes for
`published_time`, so those stay null and the dates go verbatim into `notes` (D102).

Usage: .venv/Scripts/python.exe -m src.events.ingest_inbox [--agent] [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from src.events.common import (INBOX_AGENT_CSV, INBOX_CSV, INBOX_DIR, PENDING_MARK, REPORTS, ROOT,
                               log_line, rel, sha256_file, write_marker)

COLUMNS = ["file_name", "evidence_type", "source_name", "source_uri", "published_date",
           "event_date", "location_hint", "notes"]


def read_inbox(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        return [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(f)]


def classify_row(row: dict[str, str], base_dir: Path) -> tuple[bool, str]:
    if not row.get("file_name"):
        return False, "no file_name"
    if not (base_dir / row["file_name"]).is_file():
        return False, "local file missing"
    if not row.get("source_uri"):
        return False, "no source_uri"
    if row.get("evidence_type") not in {"article", "pdf", "image"}:
        return False, f"unsupported evidence_type {row.get('evidence_type')!r}"
    if row.get("notes", "").endswith(PENDING_MARK):
        return False, "U0 pending marker not removed"
    return True, "valid"


def bronze_notes(row: dict[str, str]) -> str:
    parts = [f"inbox published_date={row.get('published_date') or 'n/a'}",
             f"event_date={row.get('event_date') or 'n/a'}",
             f"location_hint={row.get('location_hint') or 'n/a'}"]
    if row.get("notes"):
        parts.append(f"inbox notes: {row['notes']}")
    parts.append("Ingested by U1.1 (src/events/ingest_inbox.py); dates are dates only, "
                 "published_time/observed_time intentionally null.")
    return "; ".join(parts)


def verify_pair(source: Path, metrics: dict) -> dict:
    art = Path(metrics["bronze_artifact"])
    man = Path(metrics["bronze_manifest"])
    m = json.loads(man.read_text(encoding="utf-8"))
    src_sha = sha256_file(source)
    art_sha = sha256_file(art)
    return {"manifest_exists": man.is_file(), "artifact_exists": art.is_file(),
            "sha_source": src_sha, "sha_artifact": art_sha, "sha_manifest": m.get("sha256"),
            "sha_match": src_sha == art_sha == m.get("sha256"),
            "bytes_match": source.stat().st_size == art.stat().st_size == m.get("file_bytes")}


def run(agent: bool = False, dry_run: bool = False) -> dict:
    from flood_evidence_collector import ingest_local_artifact, load_config
    from flood_evidence_validator import load_contract

    inbox = INBOX_AGENT_CSV if agent else INBOX_CSV
    base_dir = INBOX_DIR / "agent" if agent else INBOX_DIR
    rows = read_inbox(inbox)
    config, contract = load_config(), load_contract()
    now = datetime.now(timezone.utc)
    out_rows, counts = [], {"INGESTED": 0, "SKIP_EXISTING": 0, "SKIPPED_ROW": 0, "ERROR": 0}
    for i, row in enumerate(rows, start=2):  # CSV line number (header = 1)
        ok, reason = classify_row(row, base_dir)
        rec = {"inbox": rel(inbox), "line": i, "file_name": row.get("file_name"),
               "evidence_type": row.get("evidence_type"), "status": "", "reason": reason,
               "evidence_id": "", "bronze_artifact": "", "bronze_manifest": "", "sha256": "",
               "sha_match": "", "bytes_match": ""}
        if not ok:
            rec["status"] = "SKIPPED_ROW"
            counts["SKIPPED_ROW"] += 1
            out_rows.append(rec)
            continue
        src = base_dir / row["file_name"]
        if dry_run:
            rec["status"] = "DRY_RUN"
            out_rows.append(rec)
            continue
        sha_before = sha256_file(src)
        try:
            metrics = ingest_local_artifact(
                source_path=src.resolve(), evidence_type=row["evidence_type"],
                source_name=row["source_name"], source_uri=row["source_uri"], media_type=None,
                ingestion_time=now, config=config, contract=contract, title=None, author=None,
                published_time=None, observed_time=None,
                location_text=row.get("location_hint") or None, notes=bronze_notes(row))
        except Exception as exc:  # report and continue with the next row
            rec["status"], rec["reason"] = "ERROR", f"{type(exc).__name__}: {exc}"[:300]
            counts["ERROR"] += 1
            out_rows.append(rec)
            log_line("u1_1_ingest", f"ERROR {row['file_name']}: {exc}")
            continue
        v = verify_pair(src, metrics)
        rec.update(status=metrics["action"], evidence_id=metrics["evidence_id"],
                   bronze_artifact=rel(Path(metrics["bronze_artifact"])),
                   bronze_manifest=rel(Path(metrics["bronze_manifest"])), sha256=v["sha_artifact"],
                   sha_match=v["sha_match"] and sha_before == v["sha_source"],
                   bytes_match=v["bytes_match"])
        counts[metrics["action"]] = counts.get(metrics["action"], 0) + 1
        out_rows.append(rec)
        log_line("u1_1_ingest", f"{metrics['action']} {row['file_name']} {metrics['evidence_id']}")
    return {"rows": out_rows, "counts": counts, "inbox": inbox}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agent", action="store_true", help="process inbox_agent.csv (U1.1b)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run-tag", default="run1", help="suffix of the result CSV")
    args = ap.parse_args()
    res = run(agent=args.agent, dry_run=args.dry_run)
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / f"ingest_{'agent' if args.agent else 'inbox'}_{args.run_tag}.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res["rows"][0].keys()) if res["rows"] else ["status"])
        w.writeheader()
        w.writerows(res["rows"])
    bad = [r for r in res["rows"] if r["status"] in ("INGESTED", "SKIP_EXISTING")
           and not (r["sha_match"] and r["bytes_match"])]
    print(f"INBOX={rel(res['inbox'])} ROWS={len(res['rows'])} COUNTS={json.dumps(res['counts'])}")
    print(f"VERIFY_FAIL={len(bad)} RESULT={rel(out)}")
    for r in res["rows"]:
        if r["status"] in ("SKIPPED_ROW", "ERROR"):
            print(f"  {r['status']} line {r['line']} {r['file_name']}: {r['reason'][:120]}")
    return 1 if (bad or res["counts"]["ERROR"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
