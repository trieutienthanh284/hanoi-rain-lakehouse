"""Leakage tests (temporal_contract.md v1.0 section 12). Synthetic data only."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.quality.temporal_checks import (  # noqa: E402
    FeatureGroup, TemporalContractError, as_of_latest, assert_feature_table, check_feature_table,
)

GSMAP = FeatureGroup("gsmap", "observation", ("gsmap_rate_mm_h", "gsmap_age_min"))
FC = FeatureGroup("fc", "forecast", ("fc_precip_mm_h",))
STATIC = FeatureGroup("static", "static", ("lat", "lon", "hour_utc"))
GROUPS = [GSMAP, FC, STATIC]


def utc(s):
    return pd.to_datetime(s, utc=True)


def valid_table(n: int = 8) -> pd.DataFrame:
    pt = pd.date_range("2025-09-30 00:00", periods=n, freq="30min", tz="UTC")
    h = np.where(np.arange(n) % 2 == 0, 30, 60)
    df = pd.DataFrame({
        "pixel_id": 1, "prediction_time": pt, "horizon_min": h,
        "target_time": pt + pd.to_timedelta(h, unit="min"),
        "gsmap_rate_mm_h": 1.0, "gsmap_age_min": 31.0,
        "gsmap__event_time": pt - pd.Timedelta("30min"),       # window end
        "gsmap__available_time": pt - pd.Timedelta("29min"),
        "fc_precip_mm_h": 0.5,
        "fc__issue_time": pt - pd.Timedelta("60min"),
        "fc__available_time": pt - pd.Timedelta("55min"),
        "fc__ingestion_time": pt - pd.Timedelta("55min"),
        "lat": 21.05, "lon": 105.85, "hour_utc": pt.hour,
        "target_max_rate_30m": 0.0,
    })
    return df


def rules(df, groups=GROUPS):
    return {v.rule for v in check_feature_table(df, groups)}


def test_valid_table_passes():
    assert check_feature_table(valid_table(), GROUPS) == []
    assert_feature_table(valid_table(), GROUPS)


def test_available_time_after_prediction_fails():
    df = valid_table()
    df.loc[3, "gsmap__available_time"] = df.loc[3, "prediction_time"] + pd.Timedelta("1min")
    assert "available_time" in rules(df)
    with pytest.raises(TemporalContractError):
        assert_feature_table(df, GROUPS)


def test_available_time_equal_prediction_is_allowed():
    df = valid_table()
    df["gsmap__available_time"] = df["prediction_time"]
    assert check_feature_table(df, GROUPS) == []


def test_missing_available_time_with_value_fails():
    df = valid_table()
    df.loc[2, "gsmap__available_time"] = pd.NaT
    assert "available_time" in rules(df)


def test_future_event_time_fails():
    df = valid_table()
    df.loc[1, "gsmap__event_time"] = df.loc[1, "prediction_time"] + pd.Timedelta("30min")
    df.loc[1, "gsmap__available_time"] = df.loc[1, "prediction_time"]
    assert "event_time" in rules(df)


def test_forecast_issued_after_prediction_fails():
    df = valid_table()
    df.loc[4, "fc__issue_time"] = df.loc[4, "prediction_time"] + pd.Timedelta("5min")
    assert "forecast_issue_time" in rules(df)


def test_ingestion_after_prediction_fails():
    df = valid_table()
    df.loc[5, "fc__ingestion_time"] = df.loc[5, "prediction_time"] + pd.Timedelta("4min")
    assert "ingestion_time" in rules(df)


def test_target_not_after_prediction_fails():
    df = valid_table()
    df.loc[0, "target_time"] = df.loc[0, "prediction_time"]
    assert "target_after_prediction" in rules(df)


def test_target_inconsistent_with_horizon_fails():
    df = valid_table()
    df.loc[0, "target_time"] = df.loc[0, "prediction_time"] + pd.Timedelta("45min")
    assert "target_equals_horizon" in rules(df)


@pytest.mark.parametrize("h", [0, 15, 45, 90, 120])
def test_invalid_horizon_fails(h):
    df = valid_table()
    df.loc[0, "horizon_min"] = h
    df.loc[0, "target_time"] = df.loc[0, "prediction_time"] + pd.Timedelta(minutes=max(h, 1))
    assert "horizon" in rules(df)


@pytest.mark.parametrize("col", ["imerg_final_rate_mm_h", "rain_final_mm_h", "IMERG_FINAL_max"])
def test_imerg_final_feature_fails(col):
    df = valid_table()
    df[col] = 1.0
    grp = GROUPS + [FeatureGroup("x", "static", (col,))]
    assert "forbidden_feature" in rules(df, grp)


def test_unlisted_feature_fails():
    df = valid_table()
    df["mystery_feature"] = 1.0
    grp = GROUPS + [FeatureGroup("x", "static", ("mystery_feature",))]
    assert "not_allowlisted" in rules(df, grp)


def test_target_columns_are_not_features():
    # target_* columns are labels, allowed in the table but never as features
    df = valid_table()
    assert "forbidden_feature" not in rules(df)


def test_naive_timestamp_fails():
    df = valid_table()
    df["prediction_time"] = df["prediction_time"].dt.tz_localize(None)
    assert "tz_utc" in rules(df)


def test_non_utc_timezone_fails():
    df = valid_table()
    df["gsmap__available_time"] = df["gsmap__available_time"].dt.tz_convert("Asia/Ho_Chi_Minh")
    assert "tz_utc" in rules(df)


def test_as_of_join_never_uses_future_records():
    rec = pd.DataFrame({
        "pixel_id": [1, 1, 1, 2],
        "window_end_utc": utc(["2025-09-30 00:30", "2025-09-30 01:00", "2025-09-30 01:30", "2025-09-30 00:30"]),
        "available_time_utc": utc(["2025-09-30 00:31", "2025-09-30 01:01", "2025-09-30 01:31", "2025-09-30 00:31"]),
        "rain_rate_mm_h": [1.0, 2.0, 3.0, 9.0],
    })
    times = pd.DataFrame({"pixel_id": [1, 1, 1, 2],
                          "prediction_time": utc(["2025-09-30 00:30", "2025-09-30 01:01",
                                                  "2025-09-30 01:30", "2025-09-30 01:00"])})
    out = as_of_latest(rec, times).sort_values(["pixel_id", "prediction_time"]).reset_index(drop=True)
    # 00:30 -> nothing available yet; 01:01 -> exact match allowed; 01:30 -> 01:31 record is future
    assert np.isnan(out.loc[0, "rain_rate_mm_h"])
    assert out.loc[1, "rain_rate_mm_h"] == 2.0
    assert out.loc[2, "rain_rate_mm_h"] == 2.0
    assert out.loc[3, "rain_rate_mm_h"] == 9.0
    assert (out["available_time_utc"].isna() | (out["available_time_utc"] <= out["prediction_time"])).all()


def test_as_of_join_rejects_naive_times():
    rec = pd.DataFrame({"pixel_id": [1], "available_time_utc": pd.to_datetime(["2025-09-30 00:31"]),
                        "rain_rate_mm_h": [1.0]})
    times = pd.DataFrame({"pixel_id": [1], "prediction_time": utc(["2025-09-30 01:00"])})
    with pytest.raises(TemporalContractError):
        as_of_latest(rec, times)
