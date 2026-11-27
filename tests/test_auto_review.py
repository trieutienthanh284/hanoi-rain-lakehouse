"""U1-R / U1-R2 auto review: sheet parser, overlay decision, and checks C2–C6 / C9 on small fixtures."""
from __future__ import annotations

import pandas as pd

from src.events.apply_review import audit_error_rate, build_overlay, decide, parse_sheet
from src.events.auto_review import (Gazetteer, check_c2, check_c3, check_c4, check_c5, check_c6, check_c9,
                                    clause_bounds, draw_audit, expected_qualifier, moved_quota, number_forms,
                                    table_row_context, text_granularity, value_occurrences, warnings_for)

GAZ = Gazetteer(["Quận Hà Đông", "Quận Hoàng Mai", "Quận Thanh Xuân", "Quận Nam Từ Liêm", "Huyện Đông Anh",
                 "Xã Vĩnh Thanh", "Phường Yên Sở", "Quận Cầu Giấy", "Ha Dong", "Hoang Mai District"])


# ------------------------------------------------------------------------------------------ C2

def test_number_forms_comma_and_dot():
    assert {"104.2", "104,2"} <= set(number_forms(104.2))
    assert {"105", "105.0", "105,0"} <= set(number_forms(105.0))


def test_c2_value_with_unit_variants():
    assert value_occurrences("Hà Đông (68,5mm)", 68.5, "rain")
    assert value_occurrences("Hà Đông 68.5 mm", 68.5, "rain")
    assert value_occurrences("rain of 68.5 millimeters", 68.5, "rain")
    assert not value_occurrences("Hà Đông 168,5 mm", 68.5, "rain")       # no partial number
    assert not value_occurrences("Hà Đông 68,55 mm", 68.5, "rain")
    assert not value_occurrences("Thanh Trì 113,5 m", 113.5, "rain")     # truncated unit is not mm
    assert value_occurrences("ngập sâu 50 cm", 0.5, "depth")             # depth in cm
    assert value_occurrences("ngập 0,5m", 0.5, "depth")


def test_c2_nearest_place_before_value():
    t = "Lượng mưa cao nhất ghi nhận ở quận Hà Đông 124 mm; Hoàng Mai 123 mm."
    assert check_c2(t, 124.0, "rain", "quận Hà Đông", GAZ)[0] == "pass"
    assert check_c2(t, 123.0, "rain", "Hoàng Mai", GAZ)[0] == "pass"
    st, why, _ = check_c2(t, 123.0, "rain", "quận Hà Đông", GAZ)        # wrong place for the value
    assert st == "fail" and "hoàng mai" in why


def test_c2_longest_name_and_other_place_in_between():
    t = "Thanh Xuân 154,5 mm; Nam Từ Liêm 106,7 mm"
    assert check_c2(t, 106.7, "rain", "Nam Từ Liêm", GAZ)[0] == "pass"
    t2 = "Thanh Xuan, Cau Giay and Nam Tu Liem all received more than 60 mm"
    gaz = Gazetteer(["Thanh Xuân", "Cầu Giấy", "Nam Từ Liêm"])
    st, why, _ = check_c2(t2, 60.0, "rain", "Cau Giay", gaz)       # U1-R2: list sharing one value
    assert st == "pass" and "list" in why


def test_c2_parenthetical_qualifier_is_not_another_place():
    t = "Ô Chợ Dừa (176mm), xã Vĩnh Thanh (huyện Đông Anh cũ) 148mm, Yên Sở (Hoàng Mai cũ) 138mm."
    assert check_c2(t, 148.0, "rain", "xã Vĩnh Thanh", GAZ)[0] == "pass"
    assert check_c2(t, 138.0, "rain", "Yên Sở", GAZ)[0] == "pass"


def test_c2_place_after_value_same_clause_and_missing_cases():
    t = "rainfall of 105 mm from midnight to 6 a.m. in Hoang Mai District, the highest among all districts"
    st, why, _ = check_c2(t, 105.0, "rain", "Hoang Mai District", GAZ)   # U1-R2: place after, same clause
    assert st == "pass" and "after" in why
    assert check_c2("Hà Đông 124 mm", 124.0, "rain", None, GAZ)[0] == "fail"
    assert check_c2("Hà Đông 124 mm", 99.0, "rain", "Hà Đông", GAZ)[0] == "fail"
    assert check_c2("Hà Đông ngập", None, "rain", "Hà Đông", GAZ)[0] == "na"


# ------------------------------------------------------------------------------------------ C5

def test_c5_forecast_cue_requires_context():
    assert check_c5("Dự báo mưa 100 mm tại Hà Đông", "rainfall_observation", 100.0)[0] == "fail"
    assert check_c5("Dự báo mưa 100 mm tại Hà Đông", "context", None)[0] == "pass"
    assert check_c5("Heavy rain is expected", "flood_observation", None)[0] == "fail"


def test_c5_rainfall_needs_value_and_flood_needs_cue():
    assert check_c5("Hà Đông 124 mm", "rainfall_observation", None)[0] == "fail"
    assert check_c5("Hà Đông 124 mm", "rainfall_observation", 124.0)[0] == "pass"
    assert check_c5("Phố Huế ngập sâu", "flood_observation", None)[0] == "pass"
    assert check_c5("the street was flooded", "flood_observation", None)[0] == "pass"
    assert check_c5("houses were submerged", "flood_observation", None)[0] == "pass"
    assert check_c5("roads inundated", "flood_observation", None)[0] == "pass"
    assert check_c5("Đường úng nước", "flood_observation", None)[0] == "pass"
    assert check_c5("Đi đúng làn đường", "flood_observation", None)[0] == "fail"   # 'đúng' is not 'úng'
    assert check_c5("nước dâng lên quá nửa bánh xe", "flood_observation", None)[0] == "pass"   # U1-R2 cue


# ------------------------------------------------------------------------------------------ C6

def test_c6_qualifier_words():
    assert expected_qualifier("lượng mưa trên 100 mm", 15) == "gt"
    assert expected_qualifier("lượng mưa hơn 100 mm", 14) == "gt"
    assert expected_qualifier("received more than 60 mm", 19) == "gt"
    assert expected_qualifier("lượng mưa lên tới 300 mm", 18) == "max"
    assert expected_qualifier("rain of up to 300 mm", 14) == "max"
    assert expected_qualifier("lượng mưa khoảng 80 mm", 17) == "approx"
    assert expected_qualifier("gần 80 mm", 4) == "approx"
    assert expected_qualifier("about 80 mm", 6) == "approx"
    assert expected_qualifier("Hà Đông 124 mm", 8) == "eq"
    assert expected_qualifier("rainfall exceeded 150 mm", 18) == "eq"          # not in the rule list


def test_c6_compares_with_catalog_qualifier():
    assert check_c6("Hà Đông trên 100 mm", 100.0, None, "gt")[0] == "pass"
    st, why = check_c6("Hà Đông trên 100 mm", 100.0, None, "eq")
    assert st == "fail" and "gt" in why
    assert check_c6("Phố Huế ngập", None, None, None)[0] == "na"


# ------------------------------------------------------------------------------------------ C4 helper

def test_text_granularity_ignores_durations():
    assert text_granularity("Lượng mưa 6h qua")[0] == 0
    assert text_granularity("đến 6h30 sáng nay")[0] == 3
    assert text_granularity("trong 24 giờ từ trưa 25/8 đến trưa 26/8")[0] == 2
    assert text_granularity("ngày 30/9")[0] == 1


# ------------------------------------------------------------------------------------------ parser

SHEET = """# Phiếu

### 1.1. [TẬP CHÍNH] VnExpress — quận Hà Đông — `fer_aaaa000000000001`

- [x] Duyệt (đúng nguyên văn, đúng trường) — Ghi chú:

### 1.2. VnExpress — Hoàng Mai — `fer_aaaa000000000002`

- [ ] Duyệt (đúng nguyên văn, đúng trường) — Ghi chú: SAI: giá trị là 124 mm

### 2.1. Nhân Dân — Ba Đình — `fer_aaaa000000000003`

- [X] Duyệt (đúng nguyên văn, đúng trường) — Ghi chú: đã kiểm tra

## 3. Khác

- [x] `fer_aaaa000000000004` [TẬP CHÍNH] VnExpress — Láng — rainfall_observation — 144 mm — 6h-8h — Ghi chú:
- [ ] `fer_aaaa000000000005` VnExpress — Láng — rainfall_observation — 1 mm — — Ghi chú:
- [x] `fer_aaaa000000000006` VnExpress — Láng — rainfall_observation — 1 mm — — Ghi chú: SAI sai ngày
"""


def test_parse_sheet_blocks_and_compact_lines():
    st = parse_sheet(SHEET)
    assert st["fer_aaaa000000000001"] == (True, "")
    assert st["fer_aaaa000000000002"] == (False, "SAI: giá trị là 124 mm")
    assert st["fer_aaaa000000000003"] == (True, "đã kiểm tra")
    assert st["fer_aaaa000000000004"] == (True, "")
    assert st["fer_aaaa000000000005"] == (False, "")
    assert st["fer_aaaa000000000006"] == (True, "SAI sai ngày")
    assert len(st) == 6


def test_parse_sheet_old_format_heading():
    old = "### A0.1. [TẬP CHÍNH] VnExpress — x — `fer_bbbb000000000001`\n\n- [x] Duyệt (đúng nguyên văn, đúng trường) — Ghi chú: ok\n"
    assert parse_sheet(old) == {"fer_bbbb000000000001": (True, "ok")}


def test_decide_rules():
    assert decide("auto_pass", False, "") == "auto_checked"
    assert decide("needs_human", False, "") == "pending"
    assert decide("needs_human", True, "") == "human_reviewed"
    assert decide("auto_pass", True, "SAI: sai địa danh") == "human_rejected"
    assert decide("auto_pass", False, "SAI") == "human_rejected"
    assert decide("auto_pass", True, "sai chính tả nhỏ") == "human_reviewed"   # only a note starting with 'SAI'


def _auto():
    return pd.DataFrame([
        {"record_id": f"fer_aaaa00000000000{i}", "class": c, "failing_checks": f, "C2_reason": "x",
         "in_audit_sample": a, "run_utc": "2026-11-20T00:00:00+00:00"}
        for i, c, f, a in [(1, "needs_human", "C2", False), (2, "needs_human", "C2", False), (3, "auto_pass", "", True),
                           (4, "auto_pass", "", True), (5, "auto_pass", "", False), (6, "auto_pass", "", True)]])


def test_overlay_idempotent_and_audit_error_rate():
    st = parse_sheet(SHEET)
    o1 = build_overlay(_auto(), st)
    o2 = build_overlay(_auto(), st, previous=o1)
    assert o1.equals(o2)
    eff = dict(zip(o1["record_id"], o1["review_effective"]))
    assert eff["fer_aaaa000000000001"] == "human_reviewed"
    assert eff["fer_aaaa000000000002"] == "human_rejected"
    assert eff["fer_aaaa000000000005"] == "auto_checked"
    assert o1.set_index("record_id").loc["fer_aaaa000000000002", "reasons"] == "SAI: giá trị là 124 mm"
    er = audit_error_rate(o1)
    assert er == {"audit_sample": 3, "audit_reviewed": 3, "audit_rejected": 1, "audit_error_rate": 0.3333}


def test_audit_sample_reproducible_min_and_per_source():
    auto = [{"record_id": f"fer_{i:016x}", "source_name": "A" if i < 30 else "B", "warnings": "w" if i == 5 else ""}
            for i in range(40)]
    s1 = draw_audit(auto, primary={"fer_0000000000000005"})
    s2 = draw_audit(auto, primary={"fer_0000000000000005"})
    assert s1 == s2
    assert len(set(s1)) >= 10
    assert any(int(x[4:], 16) >= 30 for x in s1)            # at least one of source B
    assert "fer_0000000000000005" in s1                       # primary record with a warning


# ------------------------------------------------------------------------------------------ U1-R2

def test_c2_r2_after_value_needs_no_place_before_in_clause():
    gaz = Gazetteer(["Thanh Xuân", "Hoàng Mai", "Hoàng Cầu", "Nam Từ Liêm", "Hà Đông", "Hanoi"])
    t = ("According to the Hanoi Drainage Company, rainfall ranged from 80 mm to nearly 250 mm in several areas, "
         "including 245 mm in Hoang Cau, 207 mm in Nam Tu Liem, and 191 mm in Ha Dong.")
    assert check_c2(t, 245.0, "rain", "Hoang Cau", gaz)[0] == "pass"
    assert check_c2(t, 207.0, "rain", "Nam Tu Liem", gaz)[0] == "pass"
    assert check_c2(t, 207.0, "rain", "Ha Dong", gaz)[0] == "fail"            # Ha Dong belongs to 191 mm
    # a place before the value in the same clause blocks the 'after' reading
    t2 = "Thanh Xuân 154,5 mm, Hoàng Mai 126 mm"
    assert check_c2(t2, 154.5, "rain", "Hoàng Mai", gaz)[0] == "fail"
    assert check_c2(t2, 126.0, "rain", "Hoàng Mai", gaz)[0] == "pass"


def test_c2_r2_list_must_be_contiguous_and_in_the_clause():
    gaz = Gazetteer(["Thanh Xuân", "Cầu Giấy", "Ba Đình", "Nam Từ Liêm", "Hà Đông"])
    t = "Ha Dong saw 70 mm, while Thanh Xuan, Cau Giay, Ba Dinh and Nam Tu Liem all received more than 60 mm."
    for p in ("Thanh Xuan", "Cau Giay", "Ba Dinh", "Nam Tu Liem"):
        assert check_c2(t, 60.0, "rain", p, gaz)[0] == "pass", p
    assert check_c2(t, 60.0, "rain", "Ha Dong", gaz)[0] == "fail"              # not in the list
    t2 = "Thanh Xuan recorded 90 mm; Cau Giay 60 mm"                            # not a list
    assert check_c2(t2, 60.0, "rain", "Thanh Xuan", gaz)[0] == "fail"


def test_clause_bounds_ignore_am_and_decimals():
    t = "rainfall of 105 mm from midnight to 6 a.m. in Hoang Mai, the highest"
    a, b = clause_bounds(t, t.index("105"))
    assert t[a:b] == "rainfall of 105 mm from midnight to 6 a.m. in Hoang Mai"
    t2 = "Thanh Xuân 154,5 mm, Hoàng Mai 126 mm"
    a, b = clause_bounds(t2, t2.index("154"))
    assert t2[a:b] == "Thanh Xuân 154,5 mm"


ART = ("Table 6-12: Record of inundation time in Hanoi City (Source: JICA HP) Date Rain (mm) Inundation time "
       "Target rainfall 172 mm/2 days- - 11-12/06/1991 207 7 days 17/7/2009 119.2 2 hours 13/7/2010 179.8 4.3 hours")


def test_c2_r2_table_row_unit_from_header_city_from_caption():
    gaz = Gazetteer(["Hà Nội"])
    tab = table_row_context("17/7/2009 119.2 2 hours", ART)
    assert tab and tab["value"] == 119.2 and tab["unit"] == "mm" and "Hanoi City" in tab["caption"]
    st, why, off = check_c2("17/7/2009 119.2 2 hours", 119.2, "rain", None, gaz, tab, "Hà Nội")
    assert st == "pass" and off == 10 and "header" in why
    assert check_c2("17/7/2009 119.2 2 hours", 119.2, "rain", None, gaz, tab, "Hoàng Mai")[0] == "fail"
    assert check_c2("17/7/2009 119.2 2 hours", 2.0, "rain", None, gaz, tab, "Hà Nội")[0] == "fail"
    assert table_row_context("17/7/2009 119.2 2 hours", "no header here 17/7/2009 119.2 2 hours") is None
    assert table_row_context("Hà Đông 124 mm", ART) is None
    assert check_c6("17/7/2009 119.2 2 hours", 119.2, off, "eq")[0] == "pass"


def test_c3_r2_city_level_hanoi_spelling():
    art = "record of inundation time in hanoi city"
    assert check_c3(None, "Hà Nội", "city", "17/7/2009 119.2 2 hours", art)[0] == "pass"
    assert check_c3("Hà Nội", "Hà Nội", "city", "rain in ha noi", "rain in ha noi")[0] == "pass"
    assert check_c3(None, "Hà Nội", "district", "x", art)[0] == "fail"           # not city-level
    assert check_c3(None, "Hà Nội", "city", "x", "no city name")[0] == "fail"    # artifact never names it
    assert check_c3("Hoàng Mai", "Hoàng Mai", "district", "x", art)[0] == "fail"
    assert check_c3("Hoàng Mai", "Hoàng Mai", "district", "hoàng mai 126 mm", art)[0] == "pass"


def test_c4_r2_paraphrased_temporal_text_supported_by_extracted_text():
    ex = "Mưa chia làm hai đợt. Đợt đầu từ 0h đến 6h, đợt hai từ 8h đến 9h. Hà Đông 124 mm"
    rec = {"temporal_text": "hai đợt mưa: 0h-6h và 8h-9h ngày 28/09/2023", "extracted_text": ex}
    cat = {"time_precision": "timestamp", "time_start_utc": pd.Timestamp("2023-09-27T17:00Z"),
           "time_end_utc": pd.Timestamp("2023-09-28T02:00Z")}                    # 00:00 and 09:00 local
    st, why = check_c4(rec, cat, ex.lower())
    assert st == "warn" and "paraphrased" in why
    cat_bad = dict(cat, time_end_utc=pd.Timestamp("2023-09-28T05:00Z"))         # 12:00 local, not in the text
    assert check_c4(rec, cat_bad, ex.lower())[0] == "fail"
    rec2 = {"temporal_text": "sáng 28/09/2023", "extracted_text": "trường bị ngập ngang gối"}
    assert check_c4(rec2, {"time_precision": "part_of_day"}, "x")[0] == "fail"   # no time phrase in the passage
    rec3 = {"temporal_text": "sáng 28/09/2023", "extracted_text": "Sáng nay trường bị ngập ngang gối",
            "observed_time_start": "2023-09-28T07:00:00+07:00"}
    assert check_c4(rec3, {"time_precision": "timestamp"}, "x")[0] == "fail"     # observed time without a clock


def test_c5_r2_flood_cues():
    for t in ("Một số khu vực tồn tại điểm đọng nước", "nước dâng cao tới nửa người", "đường phố chìm trong biển nước",
              "điểm ngập úng", "inundation time", "the water level rose"):
        assert check_c5(t, "flood_observation", None)[0] == "pass", t
    assert check_c5("17/7/2009 119.2 2 hours", "flood_observation", 119.2)[0] == "fail"
    assert check_c5("17/7/2009 119.2 2 hours", "flood_observation", 119.2,
                    cue_context="Record of inundation time Rain (mm)")[0] == "pass"
    assert check_c5("nước sông trong xanh", "flood_observation", None)[0] == "fail"


def test_c5_r2_forecast_only_in_the_value_sentence():
    obs = ("Riêng khung giờ 6h-8h, Trung tâm Dự báo Khí tượng Thủy văn quốc gia ghi nhận mưa phổ biến 30-60 mm, "
           "Láng 144 mm.")
    assert check_c5(obs, "rainfall_observation", 144.0)[0] == "pass"            # agency name, not a forecast
    two = "Hà Đông đo được 124 mm. Dự báo đêm nay mưa tiếp 50 mm."
    assert check_c5(two, "rainfall_observation", 124.0)[0] == "pass"            # forecast in another sentence
    assert check_c5(two, "rainfall_observation", 50.0)[0] == "fail"             # the value sentence is a forecast
    assert check_c5("Dự báo mưa 100 mm tại Hà Đông", "rainfall_observation", 100.0)[0] == "fail"


def test_c9_r2_extra_information_is_a_warning():
    rec = {"event_type": "flood_observation", "rainfall_mm": None, "flood_depth_m": None, "temporal_text": "By 6:00 am",
           "location_text": "Xuan Dinh Street"}
    items = [{"place": "Xuan Dinh", "value": 0.5, "unit": "m", "period": "By 6:00 am", "event_type": "flood_observation"}]
    st, why = check_c9(rec, "Xuan Dinh Street", items)
    assert st == "warn" and "extra" in why
    # a real disagreement next to the extra value still fails
    items2 = [dict(items[0], period="đến 9h")]
    assert check_c9(dict(rec, temporal_text="lúc 7h"), "Xuan Dinh Street", items2)[0] == "fail"
    # the record stores a value and the blind pass found a different one → fail
    assert check_c9(dict(rec, flood_depth_m=0.3), "Xuan Dinh Street", items)[0] == "fail"
    # the blind pass found no value where the record has one → not 'extra' → fail
    items3 = [dict(items[0], value=None, unit=None)]
    assert check_c9(dict(rec, flood_depth_m=0.5), "Xuan Dinh Street", items3)[0] == "fail"
    # a period the record lacks → warning
    rec4 = dict(rec, temporal_text=None, flood_depth_m=0.5)
    items4 = [dict(items[0], period="Thursday evening")]
    assert check_c9(rec4, "Xuan Dinh Street", items4)[0] == "warn"


def test_c9_r2b_table_row_is_na_with_warning():
    rec = {"event_type": "rainfall_observation", "rainfall_mm": 119.2, "flood_depth_m": None, "temporal_text": None,
           "location_text": "Hà Nội"}
    items = [{"place": None, "value": 2.0, "unit": "hours", "period": "17/7/2009", "event_type": "other"}]
    assert table_row_context("17/7/2009 119.2 2 hours", ART) is not None
    st, why = check_c9(rec, None, items, table_row=True)
    assert st == "na" and why == "table row, no header visible to blind pass"
    assert check_c9(rec, None, None, table_row=True)[0] == "na"           # even without blind output
    assert check_c9(rec, None, items)[0] == "fail"                         # same items, not a table row → judged
    r = {"record_id": "fer_t", "C9": "na", "C9_reason": why}
    assert any("table row, no header visible to blind pass" in w for w in warnings_for(r, {}))


def test_u14_r2b_dedup_key_requires_same_basis_and_window():
    from src.events.u1_followup import rain_dup_key
    rv = pd.DataFrame({"gauge_key": ["o cho dua", "o cho dua", "tay mo", "tay mo", "hai ba trung", "hai ba trung"],
                       "period_basis": ["local_day", "start_only_unknown_end", "clock", "clock", "local_day", "local_day"],
                       "period_local": ["2025-10-07 00:00–2025-10-08 00:00"] * 2
                       + ["2025-09-29 07:00–2025-09-30 07:00"] * 2
                       + ["2025-09-30 00:00–2025-10-01 00:00", "2025-09-29 00:00–2025-09-30 00:00"]})
    k = rain_dup_key(rv)
    assert k[0] != k[1]          # 'ngày 7/10' vs 'từ 0h10 đêm 7/10': same window, different stated period
    assert k[2] == k[3]          # same gauge, same basis, same window → one record kept
    assert k[4] != k[5]          # same basis, different window


def test_c8_r2_duplicate_class_in_overlay():
    auto = pd.DataFrame([{"record_id": "fer_aaaa000000000009", "class": "duplicate", "failing_checks": "",
                          "C8_reason": "duplicate_of fer_x (user override)", "in_audit_sample": False,
                          "run_utc": "2026-11-20T00:00:00+00:00", "reviewer": "auto_review/v2"}])
    o = build_overlay(auto, {})
    assert o.loc[0, "review_effective"] == "duplicate" and "duplicate_of" in o.loc[0, "reasons"]
    assert decide("duplicate", False, "") == "duplicate"
    assert decide("duplicate", True, "") == "human_reviewed"
    assert decide("duplicate", False, "SAI: không trùng") == "human_rejected"


def test_audit_r2_includes_moved_records():
    assert moved_quota(3) == 3 and moved_quota(30) == 5 and moved_quota(80) == 8
    auto = [{"record_id": f"fer_{i:016x}", "source_name": "A", "warnings": ""} for i in range(120)]
    moved = [f"fer_{i:016x}" for i in range(60, 120)]
    s1 = draw_audit(auto, primary=set(), moved=moved)
    assert s1 == draw_audit(auto, primary=set(), moved=moved)
    assert len(set(s1) & set(moved)) >= 6
    assert len(set(s1)) == len(s1)
