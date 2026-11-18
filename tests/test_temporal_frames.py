"""Frame-mode (ADD-005, P4A) temporal leakage tests with synthetic leaked rows."""
import pandas as pd
import pytest

from src.quality.temporal_checks import (FeatureGroup, FrameConfig, TemporalContractError,
                                         assert_frame_table, check_frame_table)

CFG = FrameConfig.from_yaml("configs/model_pilot.yaml")
FEATS = ["gc_k0", "nb_r2_mean_k0", "cal_hour_sin", "lat", "nwp_gfs_precip_frame"]
GROUPS = [FeatureGroup("gsmap", "observation", ("gc_k0", "nb_r2_mean_k0")),
          FeatureGroup("calendar", "static", ("cal_hour_sin",)),
          FeatureGroup("static", "static", ("lat",)),
          FeatureGroup("nwp_gfs", "forecast", ("nwp_gfs_precip_frame",))]


def good() -> pd.DataFrame:
    t0 = pd.Timestamp("2024-09-06 12:00", tz="UTC")
    rows = []
    for lead in (0, 30, 60):
        fs = t0 + pd.Timedelta(minutes=lead)
        rows.append({"pixel_id": 1, "t0": t0, "prediction_time": t0 + pd.Timedelta(minutes=10), "lead_min": lead,
                     "frame_start": fs, "target_time": fs + pd.Timedelta(minutes=30),
                     "gc_k0": 1.0, "nb_r2_mean_k0": 0.5, "cal_hour_sin": 0.1, "lat": 21.0,
                     "nwp_gfs_precip_frame": 0.2, "y_2p5": 0,
                     "gsmap__event_time": t0, "gsmap__available_time": t0 + pd.Timedelta(minutes=5),
                     "nwp_gfs__issue_time": t0 - pd.Timedelta(hours=12),
                     "nwp_gfs__available_time": t0 - pd.Timedelta(hours=5)})
    return pd.DataFrame(rows)


def rules(df, feats=FEATS):
    return {x.rule for x in check_frame_table(df, GROUPS, feats, CFG)}


def test_clean_table_passes():
    assert_frame_table(good(), GROUPS, FEATS, CFG)


def test_lead_and_frame_geometry():
    d = good(); d.loc[0, "lead_min"] = 90
    assert "lead_min" in rules(d)
    d = good(); d.loc[1, "frame_start"] += pd.Timedelta(minutes=30)
    assert "frame_start" in rules(d)
    d = good(); d.loc[2, "target_time"] += pd.Timedelta(minutes=30)
    assert "target_time" in rules(d)
    d = good(); d["prediction_time"] = d["t0"]
    assert "issue_time" in rules(d)
    d = good(); d.loc[0, "t0"] += pd.Timedelta(minutes=5)
    assert "t0_grid" in rules(d)


def test_target_not_after_issue():
    d = good()
    d["prediction_time"] = d["t0"] + pd.Timedelta(minutes=40)   # also breaks issue_time
    assert "target_after_prediction" in rules(d)


def test_future_observation_rejected():
    d = good(); d.loc[1, "gsmap__available_time"] = d.loc[1, "prediction_time"] + pd.Timedelta(minutes=1)
    assert "available_time" in rules(d)
    d = good(); d.loc[0, "gsmap__event_time"] = d.loc[0, "t0"] + pd.Timedelta(minutes=30)
    d.loc[0, "gsmap__available_time"] = d.loc[0, "prediction_time"]
    assert "event_time" in rules(d)


def test_missing_availability_with_value_rejected():
    d = good(); d.loc[0, "gsmap__available_time"] = pd.NaT
    assert "available_time" in rules(d)


def test_missing_value_without_availability_ok():
    d = good(); d.loc[0, ["gc_k0", "nb_r2_mean_k0"]] = float("nan"); d.loc[0, "gsmap__available_time"] = pd.NaT
    assert rules(d) == set()


def test_forecast_issued_after_prediction_rejected():
    d = good(); d.loc[0, "nwp_gfs__issue_time"] = d.loc[0, "prediction_time"] + pd.Timedelta(hours=1)
    assert "forecast_issue_time" in rules(d)
    d = good(); d.loc[0, "nwp_gfs__available_time"] = d.loc[0, "prediction_time"] + pd.Timedelta(minutes=1)
    assert "available_time" in rules(d)


def test_label_columns_never_features():
    d = good()
    for bad in ("y_2p5", "imerg_final_rate", "label_rate_mm_h", "target_rate"):
        d[bad] = 0.0
        g = GROUPS + [FeatureGroup("x", "static", (bad,))]
        r = {x.rule for x in check_frame_table(d, g, FEATS + [bad], CFG)}
        assert "forbidden_feature" in r, bad


def test_not_allowlisted_and_undeclared():
    d = good(); d["mystery"] = 1.0
    r = rules(d, FEATS + ["mystery"])
    assert {"not_allowlisted", "undeclared_feature"} <= r


def test_naive_timestamps_rejected():
    d = good(); d["t0"] = d["t0"].dt.tz_localize(None)
    assert "tz_utc" in rules(d)
    with pytest.raises(TemporalContractError):
        assert_frame_table(d, GROUPS, FEATS, CFG)
