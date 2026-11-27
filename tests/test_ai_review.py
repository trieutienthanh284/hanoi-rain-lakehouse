"""U1-AI: AI verdict → review_effective mapping, human precedence, idempotency, carry-forward, audit error rate,
output validation and paragraph location (small synthetic fixtures, no real data)."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from src.events.ai_review import build_item, locate, locate_any, scope_ids, validate_lines
from src.events.apply_review import ai_audit_error_rate, build_overlay, decide


def _auto():
    return pd.DataFrame([
        {"record_id": f"fer_bbbb00000000000{i}", "class": c, "failing_checks": f, "C2_reason": "x",
         "in_audit_sample": a, "run_utc": "2026-11-20T00:00:00+00:00", "reviewer": "auto_review/v2"}
        for i, c, f, a in [(1, "needs_human", "C2", False), (2, "needs_human", "C2", False),
                           (3, "needs_human", "C2", False), (4, "auto_pass", "", True), (5, "auto_pass", "", True),
                           (6, "auto_pass", "", True), (7, "duplicate", "", False)]])


AI = {"fer_bbbb000000000001": {"record_id": "fer_bbbb000000000001", "verdict": "accept", "fields": ["none"],
                               "reason": "text says 'Hà Đông 124 mm'"},
      "fer_bbbb000000000002": {"record_id": "fer_bbbb000000000002", "verdict": "reject", "fields": ["value"],
                               "reason": "text says '123 mm' for Hoàng Mai"},
      "fer_bbbb000000000003": {"record_id": "fer_bbbb000000000003", "verdict": "uncertain", "fields": ["passage"],
                               "reason": "passage not in article"},
      "fer_bbbb000000000004": {"record_id": "fer_bbbb000000000004", "verdict": "reject", "fields": ["time"],
                               "reason": "text gives only a day"},
      "fer_bbbb000000000005": {"record_id": "fer_bbbb000000000005", "verdict": "accept", "fields": ["none"],
                               "reason": "ok"},
      "fer_bbbb000000000006": {"record_id": "fer_bbbb000000000006", "verdict": "uncertain", "fields": ["place"],
                               "reason": "ambiguous"},
      "fer_bbbb000000000007": {"record_id": "fer_bbbb000000000007", "verdict": "accept", "fields": ["none"],
                               "reason": "ok"}}


def _eff(df):
    return dict(zip(df["record_id"], df["review_effective"]))


def test_ai_mapping():
    assert decide("needs_human", False, "", "accept") == "ai_reviewed"
    assert decide("needs_human", False, "", "reject") == "ai_rejected"
    assert decide("needs_human", False, "", "uncertain") == "pending"
    assert decide("auto_pass", False, "", "uncertain") == "auto_checked"
    assert decide("auto_pass", False, "", "reject") == "ai_rejected"
    o = build_overlay(_auto(), {}, ai=AI)
    e = _eff(o)
    assert e["fer_bbbb000000000001"] == "ai_reviewed"
    assert e["fer_bbbb000000000002"] == "ai_rejected"
    assert e["fer_bbbb000000000003"] == "pending"
    assert e["fer_bbbb000000000006"] == "auto_checked"
    assert e["fer_bbbb000000000007"] == "duplicate"        # duplicate is not changed by the AI
    r = o.set_index("record_id")
    assert r.loc["fer_bbbb000000000002", "note"] == "text says '123 mm' for Hoàng Mai"   # reason kept in note
    assert r.loc["fer_bbbb000000000002", "reviewer"] == "ai-subagent"
    assert r.loc["fer_bbbb000000000002", "reasons"].startswith("AI reject [value]:")
    assert r.loc["fer_bbbb000000000001", "note"] == ""
    assert r.loc["fer_bbbb000000000003", "reviewer"] == "auto_review/v2"
    assert r.loc["fer_bbbb000000000002", "reviewed_at"] != "2026-11-20T00:00:00+00:00"   # = now


def test_human_values_win_over_ai():
    assert decide("needs_human", True, "", "reject") == "human_reviewed"
    assert decide("auto_pass", False, "SAI: sai giá trị", "accept") == "human_rejected"
    states = {"fer_bbbb000000000002": (True, ""), "fer_bbbb000000000001": (False, "SAI: địa danh sai")}
    o = build_overlay(_auto(), states, ai=AI)
    e = _eff(o)
    assert e["fer_bbbb000000000002"] == "human_reviewed"
    assert e["fer_bbbb000000000001"] == "human_rejected"
    assert o.set_index("record_id").loc["fer_bbbb000000000001", "reviewer"] == "user"
    # a human decision added after the AI run also wins over carried-forward AI rows
    o2 = build_overlay(_auto(), states, previous=build_overlay(_auto(), {}, ai=AI))
    assert _eff(o2)["fer_bbbb000000000002"] == "human_reviewed"


def test_ai_idempotent_and_carried_forward():
    o1 = build_overlay(_auto(), {}, ai=AI)
    o2 = build_overlay(_auto(), {}, previous=o1, ai=AI)
    assert o1.equals(o2)                                    # same reviewed_at, same content
    o3 = build_overlay(_auto(), {}, previous=o1)            # option omitted: AI rows kept unchanged
    assert o1.equals(o3)
    o4 = build_overlay(_auto(), {}, previous=o3, ai=AI)
    assert o1.equals(o4)


def test_ai_audit_error_rate():
    o = build_overlay(_auto(), {}, ai=AI)
    er = ai_audit_error_rate(o, AI)
    assert er == {"ai_audit_accept": 1, "ai_audit_reject": 1, "ai_audit_uncertain": 1, "ai_audit_error_rate": 0.5}
    assert ai_audit_error_rate(o, {})["ai_audit_error_rate"] is None


def test_validate_lines():
    exp = {"fer_1", "fer_2"}
    ok = [json.dumps({"record_id": "fer_1", "verdict": "accept", "fields": ["none"], "reason": "q"}),
          json.dumps({"record_id": "fer_2", "verdict": "reject", "fields": ["value", "time"], "reason": "q"})]
    assert set(validate_lines(ok, exp)) == exp
    with pytest.raises(ValueError, match="duplicate"):
        validate_lines(ok + [ok[0]], exp)
    with pytest.raises(ValueError, match="missing"):
        validate_lines(ok[:1], exp)
    with pytest.raises(ValueError, match="verdict"):
        validate_lines([ok[0], ok[1].replace("reject", "maybe")], exp)
    with pytest.raises(ValueError, match="not JSON"):
        validate_lines([ok[0], "{oops"], exp)
    with pytest.raises(ValueError, match="fields"):
        validate_lines([ok[0], ok[1].replace('"time"', '"colour"')], exp)


def test_locate_paragraph_and_neighbour_sentences():
    blocks = ["Tiêu đề bài.", "Câu một. Mưa lớn tại Hà Đông 124 mm; Hoàng Mai 123 mm. Câu ba.", "Đoạn sau. Câu nữa."]
    loc = locate(blocks, "Mưa lớn tại Hà Đông 124 mm; Hoàng Mai 123 mm.")
    assert loc["paragraph"] == blocks[1]
    assert loc["sentence_before"] == "Câu một."
    assert loc["sentence_after"] == "Câu ba."
    assert locate(blocks, "không có") is None
    part = locate_any(blocks, "Câu sai không có. Mưa lớn tại Hà Đông 124 mm; Hoàng Mai 123 mm.")
    assert part["passage_located"] == "partial" and part["paragraph"] == blocks[1]
    assert locate_any([], "x") is None


def test_input_item_has_no_class_or_verdict():
    rec = {"record_id": "fer_1", "event_type": "rainfall_observation", "rainfall_mm": 124.0, "flood_depth_m": None,
           "location_text": "Hà Đông", "temporal_text": "sáng nay", "extracted_text": "Hà Đông 124 mm",
           "review_status": "pending", "confidence": "high"}
    cat = {"source_name": "S", "rainfall_qualifier": "max", "time_start_utc": "2025-09-30T00:00:00+00:00"}
    checks = {"failing_checks": "C2,C9", "C2_reason": "r2", "C9_reason": "r9", "warnings": "a; b",
              "class": "needs_human", "prev_class": "needs_human", "in_audit_sample": True}
    it = build_item(rec, cat, checks, {"published_date": "2025-09-30"}, None, None)
    s = json.dumps(it, ensure_ascii=False)
    for banned in ("needs_human", "auto_pass", "review_status", "in_audit_sample", "class"):
        assert banned not in s
    assert it["value"] == 124.0 and it["unit"] == "mm"
    assert [c["check"] for c in it["failing_checks"]] == ["C2", "C9"]
    assert it["observed"]["time_start_local"] == "2025-09-30T07:00:00+07:00"


def test_scope_disjoint():
    auto = _auto()
    nh, au = scope_ids(auto, {"fer_bbbb000000000004", "fer_bbbb000000000005"})
    assert len(nh) == 3 and len(au) == 2 and not set(nh) & set(au)
