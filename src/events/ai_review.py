"""U1-AI — inputs and output validation for the independent AI review (user task 2026-11-26).

Scope: the needs_human records (review_effective pending) plus the U1-R2b audit sample. For each record the
input holds the record fields, the Bronze paragraph containing extracted_text, the sentence before and after
the passage, the article date and the failing checks / warnings of the automated pre-review. It deliberately
holds no previous verdict and no auto_review class. A separate fresh-context subagent reads only the input
file and writes reports/u1/review/ai_review.jsonl; `validate` checks that file. Never edits Silver.

Stages:
  prepare  → reports/u1/review/ai_review_input.jsonl
  validate → checks reports/u1/review/ai_review.jsonl (JSON per line, one line per input record, no duplicates)

Usage: .venv/Scripts/python.exe -m src.events.ai_review --stage prepare|validate
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from src.events.common import BRONZE_FE, CATALOG_DIR, REPORTS, ROOT

REVIEW_DIR = REPORTS / "review"
AUTO_CSV = REVIEW_DIR / "auto_review.csv"
AUDIT_CSV = REVIEW_DIR / "audit_sample.csv"
AI_IN = REVIEW_DIR / "ai_review_input.jsonl"
AI_OUT = REVIEW_DIR / "ai_review.jsonl"
VERDICTS = ("accept", "reject", "uncertain")
FIELDS = ("value", "unit", "qualifier", "place", "time", "event_type", "passage", "none")
MAX_PARAGRAPH = 2000
WINDOW = 700
LOCAL_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
_SENT = re.compile(r"(?<=[.!?…])\s+")


def _clean(x):
    if x is None:
        return None
    try:
        import pandas as pd
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(x, "isoformat"):
        return x.isoformat()
    if hasattr(x, "tolist"):
        return x.tolist()
    return x


def locate(blocks: list[str], passage: str) -> dict | None:
    """Paragraph (block(s) containing the passage, windowed when long), sentence before and after.
    `blocks` are normalised text blocks of the artifact in reading order."""
    from src.events.text_extract import normalise
    p = normalise(passage)
    if not p:
        return None
    joined, spans, pos0 = "", [], 0
    for b in blocks:
        if joined:
            joined += " "
            pos0 += 1
        spans.append((pos0, pos0 + len(b)))
        joined += b
        pos0 += len(b)
    i = joined.find(p)
    if i < 0:
        return None
    j = i + len(p)
    hit = [k for k, (a, b) in enumerate(spans) if a < j and b > i]
    a, b = spans[hit[0]][0], spans[hit[-1]][1]
    para, truncated = joined[a:b], False
    if len(para) > MAX_PARAGRAPH:
        a2, b2 = max(a, i - WINDOW), min(b, j + WINDOW)
        para = ("… " if a2 > a else "") + joined[a2:b2] + (" …" if b2 < b else "")
        truncated = True
    head = [s for s in _SENT.split(joined[max(0, i - 600):i].rstrip()) if s.strip()]
    tail = [s for s in _SENT.split(joined[j:j + 600].lstrip()) if s.strip()]
    # the passage may end mid-sentence: the "after" sentence is the first full sentence after its end
    rest = joined[j:j + 600]
    if rest and not re.match(r"^\s*$", rest) and not re.search(r"[.!?…]\s*$", p):
        tail = tail[1:] if len(tail) > 1 else tail
    return {"paragraph": para, "paragraph_truncated": truncated,
            "sentence_before": head[-1][-400:] if head else "",
            "sentence_after": tail[0][:400] if tail else ""}


def locate_any(blocks: list[str], passage: str) -> dict | None:
    """locate(); if the whole passage is not verbatim in the artifact, locate its longest sentence that is
    (≥ 20 characters). `passage_located` = exact | partial; None when nothing is found or there is no text."""
    if not blocks:
        return None
    loc = locate(blocks, passage)
    if loc is not None:
        return {**loc, "passage_located": "exact"}
    for s in sorted((x for x in _SENT.split(passage or "") if len(x.strip()) >= 20), key=len, reverse=True):
        loc = locate(blocks, s)
        if loc is not None:
            return {**loc, "passage_located": "partial"}
    return None


def _local(ts) -> str | None:
    if ts is None:
        return None
    t = ts if isinstance(ts, datetime) else datetime.fromisoformat(str(ts))
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    return t.astimezone(LOCAL_TZ).isoformat()


def build_item(rec: dict, cat: dict, checks: dict, ctx: dict, loc: dict | None, table: dict | None) -> dict:
    """One input line. No class, no previous verdict, no review status."""
    rain, depth = rec.get("rainfall_mm"), rec.get("flood_depth_m")
    value, unit = (rain, "mm") if rain is not None else ((depth, "m") if depth is not None else (None, None))
    fails = [c for c in str(checks.get("failing_checks") or "").split(",") if c]
    warns = [w for w in str(checks.get("warnings") or "").split(";") if w.strip()]
    return {
        "record_id": rec["record_id"],
        "source_name": cat.get("source_name"),
        "article_date": str(ctx.get("published_date")) if ctx.get("published_date") else None,
        "event_type": rec.get("event_type"),
        "value": value, "unit": unit,
        "qualifier": cat.get("rainfall_qualifier"),
        "location_text": rec.get("location_text"),
        "location_text_verbatim": cat.get("location_text_verbatim"),
        "temporal_text": rec.get("temporal_text"),
        "rain_period_text": cat.get("rain_period_text"),
        "rainfall_duration_hours": cat.get("rainfall_duration_hours"),
        "flood_depth_text": cat.get("flood_depth_text"),
        "observed": {"event_date_start": cat.get("event_date_start"), "event_date_end": cat.get("event_date_end"),
                     "time_start_utc": cat.get("time_start_utc"), "time_end_utc": cat.get("time_end_utc"),
                     "time_start_local": _local(cat.get("time_start_utc")),
                     "time_end_local": _local(cat.get("time_end_utc"))},
        "time_precision": cat.get("time_precision"),
        "extracted_text": rec.get("extracted_text"),
        "passage_located": loc["passage_located"] if loc and "passage_located" in loc else
                           ("exact" if loc else "not_found_in_artifact_text"),
        "paragraph": loc["paragraph"] if loc else None,
        "paragraph_truncated": loc["paragraph_truncated"] if loc else None,
        "sentence_before": loc["sentence_before"] if loc else None,
        "sentence_after": loc["sentence_after"] if loc else None,
        "table_context": ({"caption": table["caption"], "header": table["header"]} if table else None),
        "failing_checks": [{"check": c, "reason": checks.get(f"{c}_reason")} for c in fails],
        "warnings": [w.strip() for w in warns],
    }


def scope_ids(auto, audit_ids: set[str]) -> tuple[list[str], list[str]]:
    """(needs_human ids, audit-sample ids), sorted; the two sets are disjoint by construction."""
    nh = sorted(auto.loc[auto["class"] == "needs_human", "record_id"])
    au = sorted(i for i in audit_ids if i not in set(nh))
    return nh, au


def stage_prepare() -> dict:
    import pandas as pd
    from src.events.auto_review import table_row_context
    from src.events.build_event_catalog import manifest_context
    from src.events.text_extract import artifact_text, normalise
    from src.events.u1_followup import silver_records

    auto = pd.read_csv(AUTO_CSV, keep_default_na=False, na_values=[""])
    audit = set(pd.read_csv(AUDIT_CSV)["record_id"])
    nh, au = scope_ids(auto, audit)
    ids = sorted(nh + au)
    sil = silver_records()
    cat = pd.read_parquet(CATALOG_DIR / "events.parquet").set_index("silver_record_id")
    chk = auto.set_index("record_id")
    blocks, ctxs, items, missing = {}, {}, [], []
    for rid in ids:
        rec = sil[rid]
        eid = rec["evidence_id"]
        if eid not in blocks:
            man = json.loads(next((BRONZE_FE / "manifest").rglob(f"{eid}.json")).read_text(encoding="utf-8"))
            raw = artifact_text(ROOT / man["raw_artifact_path"])
            blocks[eid] = [b for b in (normalise(x) for x in raw.split("\n")) if b]
            ctxs[eid] = manifest_context(eid)
        ex = rec.get("extracted_text") or ""
        loc = locate_any(blocks[eid], ex)
        if loc is None:
            missing.append(rid)
        table = table_row_context(ex, normalise(" ".join(blocks[eid])))
        c = {k: _clean(v) for k, v in cat.loc[rid].to_dict().items()}
        ck = {k: _clean(v) for k, v in chk.loc[rid].to_dict().items()}
        items.append(build_item({k: _clean(v) for k, v in rec.items()}, c, ck, ctxs[eid], loc, table))
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    tmp = AI_IN.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    tmp.replace(AI_IN)
    from src.events.common import rss_gb
    return {"records": len(items), "needs_human": len(nh), "audit_sample": len(au), "rss_gb_end": round(rss_gb(), 3),
            "paragraph_not_found": missing, "truncated": sum(1 for i in items if i["paragraph_truncated"])}


def validate_lines(lines: list[str], expected: set[str]) -> dict:
    """Schema check of the subagent output. Raises ValueError on any problem."""
    seen, out = set(), {}
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"line {n}: not JSON ({e})") from e
        rid = o.get("record_id")
        if rid in seen:
            raise ValueError(f"line {n}: duplicate record_id {rid}")
        if rid not in expected:
            raise ValueError(f"line {n}: record_id {rid} not in the input")
        if o.get("verdict") not in VERDICTS:
            raise ValueError(f"line {n}: verdict {o.get('verdict')!r}")
        f = o.get("fields")
        if not isinstance(f, list) or not f or any(x not in FIELDS for x in f):
            raise ValueError(f"line {n}: fields {f!r}")
        if not isinstance(o.get("reason"), str) or not o["reason"].strip():
            raise ValueError(f"line {n}: empty reason")
        seen.add(rid)
        out[rid] = o
    if seen != expected:
        raise ValueError(f"missing record_ids: {sorted(expected - seen)}")
    return out


def load_verdicts(path) -> dict[str, dict]:
    """{record_id: {verdict, fields, reason}} from a validated ai_review.jsonl (no expected-set check)."""
    out = {}
    for n, line in enumerate(open(path, encoding="utf-8"), 1):
        if line.strip():
            o = json.loads(line)
            if o["record_id"] in out:
                raise ValueError(f"line {n}: duplicate record_id {o['record_id']}")
            if o.get("verdict") not in VERDICTS:
                raise ValueError(f"line {n}: verdict {o.get('verdict')!r}")
            out[o["record_id"]] = o
    return out


def stage_validate() -> dict:
    expected = {json.loads(l)["record_id"] for l in open(AI_IN, encoding="utf-8") if l.strip()}
    v = validate_lines(open(AI_OUT, encoding="utf-8").read().splitlines(), expected)
    return {"records": len(v), "verdicts": {k: sum(1 for o in v.values() if o["verdict"] == k) for k in VERDICTS}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["prepare", "validate"], required=True)
    a = ap.parse_args()
    res = stage_prepare() if a.stage == "prepare" else stage_validate()
    print(json.dumps(res, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
