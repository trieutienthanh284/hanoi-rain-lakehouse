"""U1-R — apply the user's review sheet to the review overlay (idempotent).

Parses reports/u1/U1_SILVER_REVIEW.md: a ticked box → human_reviewed; a note starting with "SAI" →
human_rejected (with the note), ticked or not; untouched records keep the automatic class
(auto_pass → auto_checked, needs_human → pending, duplicate → duplicate; U1-R2 D150). Writes data/gold/event_catalog/review_overlay.parquet
and the review_effective column of data/gold/event_catalog/events.parquet only when something changed.
Prints counts and the audit-sample error rate. Never edits Silver.

U1-AI (user 2026-11-26): `--ai-review <path>` applies the independent AI review (reports/u1/review/ai_review.jsonl):
accept → ai_reviewed, reject → ai_rejected (reason kept in note), uncertain → unchanged (pending / auto_checked);
reviewer "ai-subagent", reviewed_at = time of the first application. Human values (human_reviewed /
human_rejected) always win over AI values; duplicate stays duplicate. AI rows already in the overlay are carried
forward when the option is omitted, so a later run for new sheet ticks does not drop them. Idempotent.

Usage: .venv/Scripts/python.exe -m src.events.apply_review [--ai-review reports/u1/review/ai_review.jsonl]
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone

from src.events.common import CATALOG_DIR, REPORTS

OVERLAY = CATALOG_DIR / "review_overlay.parquet"
SHEET = REPORTS / "U1_SILVER_REVIEW.md"
AUTO_CSV = REPORTS / "review" / "auto_review.csv"
REVIEW_VALUES = ("auto_checked", "pending", "human_reviewed", "human_rejected", "duplicate", "ai_reviewed", "ai_rejected")
AI_REVIEWER = "ai-subagent"
AI_MAP = {"accept": "ai_reviewed", "reject": "ai_rejected"}
OVERLAY_COLS = ["record_id", "review_effective", "reasons", "reviewer", "reviewed_at", "in_audit_sample", "note"]

_HEAD = re.compile(r"^#{2,4} .*`(fer_[0-9a-f]+)`\s*$")
_BOX = re.compile(r"^\s*[-*] \[([ xX])\](.*)$")
_ID = re.compile(r"`(fer_[0-9a-f]+)`")


def parse_sheet(text: str) -> dict[str, tuple[bool, str]]:
    """{record_id: (ticked, note)} from full blocks (id in the heading) and compact lines (id on the line)."""
    out: dict[str, tuple[bool, str]] = {}
    cur = None
    for line in text.splitlines():
        m = _HEAD.match(line)
        if m:
            cur = m.group(1)
            continue
        m = _BOX.match(line)
        if not m:
            continue
        rest = m.group(2)
        mid = _ID.search(rest)
        rid = mid.group(1) if mid else cur
        if not rid:
            continue
        note = rest.split("Ghi chú:", 1)[1].strip() if "Ghi chú:" in rest else ""
        out[rid] = (m.group(1).lower() == "x", note)
        if not mid:
            cur = None
    return out


def is_rejection(note: str) -> bool:
    return note.strip().startswith("SAI")


def decide(base_class: str, ticked: bool, note: str, ai_verdict: str | None = None) -> str:
    """Precedence: human rejection > human tick > duplicate > AI accept/reject > automatic class.
    AI 'uncertain' (or no AI verdict) leaves the automatic class."""
    if is_rejection(note):
        return "human_rejected"
    if ticked:
        return "human_reviewed"
    if base_class == "duplicate":
        return "duplicate"
    if ai_verdict in AI_MAP:
        return AI_MAP[ai_verdict]
    return "auto_checked" if base_class == "auto_pass" else "pending"


def ai_from_overlay(previous) -> dict[str, dict]:
    """AI verdicts already applied in an overlay, for carry-forward when --ai-review is omitted."""
    if previous is None or previous.empty or "reviewer" not in previous.columns:
        return {}
    inv = {v: k for k, v in AI_MAP.items()}
    out = {}
    for r in previous[previous["review_effective"].isin(list(inv))].to_dict("records"):
        out[r["record_id"]] = {"verdict": inv[r["review_effective"]], "_reasons": r["reasons"],
                               "_note": r.get("note") or ""}
    return out


def ai_texts(v: dict) -> tuple[str, str]:
    """(reasons, note) of an AI-decided overlay row."""
    if "_reasons" in v:
        return v["_reasons"], v["_note"]
    fields = ",".join(x for x in (v.get("fields") or []) if x != "none")
    reason = str(v.get("reason") or "").strip()
    if v["verdict"] == "reject":
        return f"AI reject [{fields}]: {reason}", reason
    return f"AI accept: {reason}", ""


def build_overlay(auto, states: dict[str, tuple[bool, str]], previous=None, ai: dict[str, dict] | None = None):
    """Overlay rows from the auto-review table, the parsed sheet and (U1-AI) the AI verdicts. `previous` (overlay
    DataFrame) keeps reviewed_at of unchanged rows so a re-run without changes writes the same content; AI rows of
    `previous` are carried forward unless `ai` gives a new verdict for that record."""
    import pandas as pd
    prev = {} if previous is None or previous.empty else previous.set_index("record_id").to_dict("index")
    ai_all = {**ai_from_overlay(previous), **(ai or {})}
    rows = []
    for r in auto.to_dict("records"):
        rid = r["record_id"]
        ticked, note = states.get(rid, (False, ""))
        v = ai_all.get(rid)
        eff = decide(r["class"], ticked, note, v["verdict"] if v else None)
        if eff in ("ai_reviewed", "ai_rejected"):
            reasons, note = ai_texts(v)
            reviewer = AI_REVIEWER
        elif eff == "human_rejected":
            reasons, reviewer = note.strip(), "user"
        elif eff == "human_reviewed":
            reasons, reviewer = (note.strip() or "ticked in U1_SILVER_REVIEW.md"), "user"
        elif eff == "auto_checked":
            reasons, reviewer = "C1–C9 pass", str(r.get("reviewer") or "auto_review/v1")
        elif eff == "duplicate":
            reasons, reviewer = str(r.get("C8_reason") or "duplicate (overrides)"), str(r.get("reviewer") or "auto_review/v1")
        else:
            fails = str(r.get("failing_checks") or "")
            reasons = "; ".join(f"{k}: {r.get(k + '_reason')}" for k in fails.split(",") if k)
            reviewer = str(r.get("reviewer") or "auto_review/v1")
        p = prev.get(rid)
        if p and p.get("review_effective") == eff and p.get("reasons") == reasons and p.get("note", "") == note.strip():
            at = p["reviewed_at"]
        else:
            at = (datetime.now(timezone.utc).replace(microsecond=0).isoformat() if reviewer in ("user", AI_REVIEWER)
                  else str(r.get("run_utc")))
        rows.append({"record_id": rid, "review_effective": eff, "reasons": reasons, "reviewer": reviewer,
                     "reviewed_at": at, "in_audit_sample": bool(r.get("in_audit_sample")), "note": note.strip()})
    df = pd.DataFrame(rows, columns=OVERLAY_COLS)
    assert df["review_effective"].isin(REVIEW_VALUES).all()
    return df


def write_overlay(df) -> bool:
    """Write the overlay atomically; False (no write) when the content is unchanged."""
    import pandas as pd
    if OVERLAY.exists():
        old = pd.read_parquet(OVERLAY)
        if list(old.columns) == list(df.columns) and old.reset_index(drop=True).equals(df.reset_index(drop=True)):
            return False
    OVERLAY.parent.mkdir(parents=True, exist_ok=True)
    tmp = OVERLAY.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(OVERLAY)
    return True


def sync_catalog(df) -> bool:
    """Set events.parquet review_effective from the overlay (superseded rows: null). False if unchanged."""
    import pandas as pd
    path = CATALOG_DIR / "events.parquet"
    cat = pd.read_parquet(path)
    new = cat["silver_record_id"].map(df.set_index("record_id")["review_effective"])
    new = new.where(cat["superseded_by"].isna(), None)
    if "review_effective" in cat.columns and cat["review_effective"].fillna("∅").equals(new.fillna("∅")):
        return False
    cat["review_effective"] = new
    tmp = path.with_suffix(".parquet.tmp")
    cat.to_parquet(tmp, index=False)
    tmp.replace(path)
    return True


def audit_error_rate(df) -> dict:
    a = df[df["in_audit_sample"]]
    reviewed = a["review_effective"].isin(["human_reviewed", "human_rejected"])
    rej = int((a["review_effective"] == "human_rejected").sum())
    n = int(reviewed.sum())
    return {"audit_sample": int(len(a)), "audit_reviewed": n, "audit_rejected": rej,
            "audit_error_rate": None if n == 0 else round(rej / n, 4)}


def ai_audit_error_rate(df, ai: dict[str, dict]) -> dict:
    """AI audit error rate = AI rejects in the audit sample / audit records with a verdict (accept or reject;
    uncertain excluded and counted separately). Computed from the AI verdicts, whatever humans later decide."""
    ids = set(df.loc[df["in_audit_sample"], "record_id"])
    vs = [ai[i]["verdict"] for i in ids if i in ai]
    acc, rej, unc = vs.count("accept"), vs.count("reject"), vs.count("uncertain")
    return {"ai_audit_accept": acc, "ai_audit_reject": rej, "ai_audit_uncertain": unc,
            "ai_audit_error_rate": None if acc + rej == 0 else round(rej / (acc + rej), 4)}


def load_ai(path, known: set[str]) -> dict[str, dict]:
    from src.events.ai_review import load_verdicts
    ai = load_verdicts(path)
    unknown = sorted(set(ai) - known)
    if unknown:
        raise ValueError(f"AI review has record_ids not in the auto-review table: {unknown[:5]}")
    return ai


def main(argv: list[str] | None = None) -> int:
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai-review", metavar="PATH", help="apply the AI review JSONL (U1-AI)")
    a = ap.parse_args(argv)
    auto = pd.read_csv(AUTO_CSV, keep_default_na=False, na_values=[""])
    states = parse_sheet(SHEET.read_text(encoding="utf-8"))
    prev = pd.read_parquet(OVERLAY) if OVERLAY.exists() else None
    ai = load_ai(a.ai_review, set(auto["record_id"])) if a.ai_review else None
    df = build_overlay(auto, states, prev, ai)
    wrote = write_overlay(df)
    synced = sync_catalog(df)
    res = {"records": int(len(df)), "counts": {k: int((df["review_effective"] == k).sum()) for k in REVIEW_VALUES},
           "ticked": sum(1 for t, _ in states.values() if t),
           "notes": sum(1 for _, n in states.values() if n), "overlay_written": wrote, "catalog_updated": synced,
           **audit_error_rate(df)}
    if ai is not None:
        res.update(ai_audit_error_rate(df, ai))
        human = set(df.loc[df["review_effective"].isin(["human_reviewed", "human_rejected"]), "record_id"])
        res["ai_overridden_by_human"] = sorted(i for i, v in ai.items() if v["verdict"] in AI_MAP and i in human)
    print(json.dumps(res, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
