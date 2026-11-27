"""U1.7 flood indicator: boundaries, long rule, accumulation and config invariants."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.events.flood_indicator import accumulate, indicator_level, indicator_levels, load_config

CFG = load_config()


def test_config_user_decision_unchanged():
    assert CFG["variable"] == "A6h" and CFG["a_mm"] == 50 and CFG["b_mm"] == 100
    assert CFG["long_rule"] == {"window_h": 24, "mm": 150, "level": "high"}
    assert CFG["scale"] == "gauge" and CFG["satellite_factor"] is None and CFG["status"] == "provisional"
    assert str(CFG["decided"]) == "2026-11-19"


@pytest.mark.parametrize("a6h,expected", [(49.9, "none"), (50.0, "possible"), (99.9, "possible"),
                                          (100.0, "high"), (0.0, "none")])
def test_a6h_boundaries(a6h, expected):
    assert indicator_level(a6h, 0.0, CFG) == expected


@pytest.mark.parametrize("a24h,expected", [(149.9, "none"), (150.0, "high")])
def test_long_rule_boundary(a24h, expected):
    assert indicator_level(10.0, a24h, CFG) == expected


def test_long_rule_lifts_possible_to_high():
    assert indicator_level(60.0, 150.0, CFG) == "high"


def test_missing_data_is_not_none_level():
    assert indicator_level(float("nan"), None, CFG) is None
    assert indicator_level(None, 160.0, CFG) == "high"


def test_vectorised_matches_scalar():
    a6 = np.array([49.9, 50, 99.9, 100, 10, np.nan, np.nan])
    a24 = np.array([0, 0, 0, 0, 150, 160, np.nan])
    codes = indicator_levels(a6, a24, 50, 100, 150)
    names = {0: "none", 1: "possible", 2: "high", -1: None}
    for x6, x24, c in zip(a6, a24, codes):
        assert names[int(c)] == indicator_level(x6, x24, CFG)


def test_accumulate_trailing_window_and_incomplete_nan():
    idx = pd.date_range("2025-09-30 00:30", periods=16, freq="30min", tz="UTC")  # interval ends
    s = pd.Series(1.0, index=idx)  # 1 mm per half hour
    a6 = accumulate(s, 6)
    assert math.isnan(a6.iloc[10]) and a6.iloc[11] == pytest.approx(12.0)  # 12 half hours
    s2 = s.drop(idx[13])  # a missing interval → incomplete windows are NaN
    assert math.isnan(accumulate(s2, 6).loc[idx[14]])


# ---- U1-D2 (user, 2026-11-20): per-product decided thresholds

from src.events.flood_indicator import ProductNotUsedForIcon, product_thresholds  # noqa: E402


@pytest.mark.parametrize("product", ["imerg_final", "imerg_late"])
def test_imerg_product_thresholds_are_gauge_scale(product):
    assert product_thresholds(CFG, product) == (50.0, 100.0, 150.0)


@pytest.mark.parametrize("product", ["imerg_final", "imerg_late"])
@pytest.mark.parametrize("a6h,a24h,expected", [(49.9, 0.0, 0), (50.0, 0.0, 1), (99.9, 0.0, 1), (100.0, 0.0, 2),
                                               (10.0, 149.9, 0), (10.0, 150.0, 2)])
def test_imerg_product_boundaries(product, a6h, a24h, expected):
    a, b, lm = product_thresholds(CFG, product)
    assert int(indicator_levels([a6h], [a24h], a, b, lm)[0]) == expected


def test_gsmap_now_gc_thresholds():
    assert product_thresholds(CFG, "gsmap_now_gc") == (110.0, 220.0, 270.0)


@pytest.mark.parametrize("a6h,a24h,expected", [(109.9, 0.0, 0), (110.0, 0.0, 1), (219.9, 0.0, 1), (220.0, 0.0, 2),
                                               (10.0, 269.9, 0), (10.0, 270.0, 2), (115.0, 269.9, 1)])
def test_gsmap_now_gc_boundaries(a6h, a24h, expected):
    a, b, lm = product_thresholds(CFG, "gsmap_now_gc")
    assert int(indicator_levels([a6h], [a24h], a, b, lm)[0]) == expected


def test_gsmap_now_raw_not_used_for_icon():
    with pytest.raises(ProductNotUsedForIcon, match="6/10"):
        product_thresholds(CFG, "gsmap_now")


def test_unknown_product_has_no_fallback():
    with pytest.raises(KeyError):
        product_thresholds(CFG, "satellite_proposal_imerg")  # proposals are never used as thresholds
