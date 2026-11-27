"""U1.3 event catalog: temporal/spatial parsing rules and the built catalog."""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pytest

from src.events.build_event_catalog import (clock_windows, core_name, location_level, parse_dates,
                                            rainfall_qualifier, temporal)

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "data/gold/event_catalog/events.parquet"


def _t(tt, et="flood_observation", t0=None, t1=None, ctx=None):
    rec = {"temporal_text": tt, "event_type": et, "observed_time_start": t0, "observed_time_end": t1}
    return temporal(rec, ctx or {"event_date": None, "published_date": date(2023, 9, 28)})


def test_part_of_day_not_finer_than_text():
    r = _t("sáng 28/09/2023")
    assert r["time_precision"] == "part_of_day"
    assert r["event_date_start"] == date(2023, 9, 28) and r["time_start_utc"] is None


@pytest.mark.parametrize("tt,start,end", [("29-30/09/1993", date(1993, 9, 29), date(1993, 9, 30)),
                                          ("31/10-02/11/2008", date(2008, 10, 31), date(2008, 11, 2)),
                                          ("cuối tháng 10/2008", date(2008, 10, 1), date(2008, 10, 31))])
def test_date_ranges(tt, start, end):
    r = _t(tt, ctx={"event_date": None, "published_date": None})
    assert (r["time_precision"], r["event_date_start"], r["event_date_end"]) == ("date_range", start, end)


def test_single_date_is_date():
    assert _t("17/7/2009")["time_precision"] == "date"
    assert _t("ngày 23 tháng 07 năm 2024", et="context")["time_precision"] == "date"


def test_clock_windows_give_timestamp_in_utc():
    r = _t("hai đợt mưa: 0h-6h và 8h-9h ngày 28/09/2023", et="rainfall_observation")
    assert r["time_precision"] == "timestamp"
    assert r["time_start_utc"].isoformat() == "2023-09-27T17:00:00+00:00"
    assert r["time_end_utc"].isoformat() == "2023-09-28T02:00:00+00:00"
    assert len(clock_windows("0h-6h và 8h-9h", date(2023, 9, 28))) == 2


def test_observed_time_wins():
    r = _t("9h", t0="2023-09-28T09:00:00+07:00")
    assert r["time_precision"] == "timestamp" and r["time_start_utc"].hour == 2


def test_relative_text_uses_inbox_date_never_finer_than_part_of_day():
    ctx = {"event_date": date(2025, 8, 26), "published_date": date(2025, 8, 26)}
    assert _t("overnight", ctx=ctx)["time_precision"] == "date_range"
    assert _t("overnight", ctx=ctx)["event_date_start"] == date(2025, 8, 25)
    assert _t("lúc gần 14h", ctx=ctx)["time_precision"] == "part_of_day"
    assert _t("within six hours", ctx=ctx)["time_precision"] == "date"
    assert _t(None, et="context", ctx=ctx)["time_precision"] == "none"


def test_no_inbox_date_no_fallback():
    assert _t(None)["time_precision"] == "none"


def test_partial_date_takes_year_from_article():
    d0, d1, kind = parse_dates("sáng 24-7", 2024)
    assert (d0, kind) == (date(2024, 7, 24), "single")


@pytest.mark.parametrize("loc,level", [("Hà Nội", "city"), ("nội thành Hà Nội", "city"),
                                       ("quận Hoàng Mai, Hà Nội", "district"),
                                       ("phường Ô Chợ Dừa, Hà Nội", "ward"),
                                       ("đường Nguyễn Xiển, quận Thanh Xuân, Hà Nội", "street"),
                                       ("Ký túc xá Mễ Trì, quận Thanh Xuân, Hà Nội", "feature"),
                                       ("Luong The Vinh Street in Dai Mo Ward, Hanoi", "street"),
                                       ("My Duc District, Hanoi", "district"),
                                       ("Ô Chợ Dừa, Hà Nội", None)])
def test_location_level(loc, level):
    assert location_level(loc)[0] == level


def test_core_name_exact_normalisation():
    assert core_name("Quận Hoàng Mai") == core_name("Hoang Mai District, Hanoi") == "hoang mai"
    assert core_name("Hoàng Mai (tại Yên Sở), Hà Nội") == "hoang mai"
    assert core_name("phường Văn Miếu") != core_name("Phường Văn Miếu - Quốc Tử Giám")


def test_rainfall_qualifier():
    assert rainfall_qualifier("phường Từ Liêm trên 150 mm; phường Thanh Xuân 165 mm", 150) == "gt"
    assert rainfall_qualifier("phường Từ Liêm trên 150 mm; phường Thanh Xuân 165 mm", 165) == "eq"
    assert rainfall_qualifier("phường Ô Chợ Dừa gần 320 mm", 320) == "approx"
    assert rainfall_qualifier("had reached up to 270 mm", 270) == "max"


@pytest.mark.skipif(not CATALOG.exists(), reason="catalog not built")
def test_built_catalog():
    import pandas as pd
    d = pd.read_parquet(CATALOG)
    n_silver = len(list((ROOT / "data/silver/flood_evidence/records").rglob("fer_*.json")))
    assert len(d) == n_silver and d["silver_record_id"].is_unique
    sang = d[d["temporal_text"] == "sáng 28/09/2023"]
    assert len(sang) >= 2 and (sang["time_precision"] == "part_of_day").all()
    jica = d[d["evidence_id"] == "fe_pdf_2b85b288a97795b1"]
    assert len(jica) == 15 and (jica["location_level"] == "city").all()
    for _, r in jica.iterrows():  # range text → date_range; single date text → date (never finer)
        is_range = bool(re.search(r"\d\s*-\s*\d", r["temporal_text"]))
        assert r["time_precision"] == ("date_range" if is_range else "date")
    ts = d[d["time_precision"] == "timestamp"]
    assert ts["time_start_utc"].notna().sum() + ts["time_end_utc"].notna().sum() >= len(ts)
    assert set(d["time_precision"]) <= {"timestamp", "part_of_day", "date", "date_range", "none"}
    assert set(d["location_level"].dropna()) <= {"feature", "street", "ward", "district", "city"}
    hm = d[(d["temporal_text"] == "hai đợt mưa: 0h-6h và 8h-9h ngày 28/09/2023")]
    assert set(hm["feature_name"]) == {"Quận Hà Đông", "Quận Hoàng Mai"}


# ---- follow-up 2 (2026-11-20): exact-name alias, verbatim location span, superseded records

def test_tone_core_alias_is_exact_and_keeps_tone_marks():
    from src.events.build_event_catalog import tone_core
    assert tone_core("phường Văn Miếu, Hà Nội") == tone_core("Văn Miếu") == "văn miếu"
    assert tone_core("phường Văn Miêu, Hà Nội") != "văn miếu"          # different tone mark: no alias
    assert tone_core("Văn Miếu - Quốc Tử Giám") != "văn miếu"          # no partial matching


def test_location_verbatim_spans():
    from src.events.build_event_catalog import location_verbatim
    art = "Cụ thể lượng mưa đo được tại các phường: Ô Chợ Dừa 509 mm; nội thành Hà Nội và lưu vực sông Nhuệ"
    assert location_verbatim("nội thành Hà Nội", art) == ("nội thành Hà Nội", False)
    assert location_verbatim("phường Ô Chợ Dừa, Hà Nội", art) == ("Ô Chợ Dừa", True)
    assert location_verbatim("Nội thành Hà Nội, lưu vực sông Nhuệ", art) == ("nội thành Hà Nội | lưu vực sông Nhuệ", True)
    assert location_verbatim("quận Cầu Giấy, Hà Nội", art) == (None, True)
    assert location_verbatim("Hà Nội", "") == (None, None)               # no text layer (image)


@pytest.mark.skipif(not CATALOG.exists(), reason="catalog not built")
def test_superseded_and_alias_in_built_catalog():
    import pandas as pd
    import yaml
    d = pd.read_parquet(CATALOG).set_index("silver_record_id")
    ov = yaml.safe_load((ROOT / "configs/event_catalog_overrides.yaml").read_text(encoding="utf-8"))
    for old, v in ov["superseded"].items():
        assert d.loc[old, "superseded_by"] == v["superseded_by"] and len(d.loc[old, "usable_for"]) == 0
        new = d.loc[v["superseded_by"]]
        assert new["silver_valid"] and pd.isna(new["superseded_by"]) and "rain_validation" in list(new["usable_for"])
    vm = d[d["location_text"] == "phường Văn Miếu, Hà Nội"]
    assert len(vm) == 1 and vm["feature_name"].iloc[0] == "Phường Văn Miếu - Quốc Tử Giám"
    assert {"location_text_verbatim", "location_context_added"} <= set(d.columns)


def test_part_of_day_date_range_is_a_range():  # U1.1b: "trong 24 giờ từ trưa 25/8 đến trưa 26/8"
    from src.events.build_event_catalog import parse_dates
    assert parse_dates("trong 24 giờ từ trưa 25/8 đến trưa 26/8", 2025) == (date(2025, 8, 25), date(2025, 8, 26), "range")
    assert parse_dates("tính từ 7h ngày 6/10 đến 5h30 sáng 7/10", 2025)[2] == "range"
