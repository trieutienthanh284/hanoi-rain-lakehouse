"""P4F/P4G pre-registered rules (D214) as pure functions, and the data helpers they rely on (synthetic, tiny)."""
import numpy as np
import pandas as pd
import pytest

from src.gold import features as F
from src.models import p4fg as FG
from src.models import p4fg_gates as G
from src.models import p4fg_replay as R

CFG = FG.load_cfg(smoke=False)
PR = CFG["pre_registered"]


def test_dt_verdict():
    r = PR["dt_check"]
    assert FG.dt_verdict(0.5, 1.305, r) == "ok"
    assert FG.dt_verdict(0.5, 2.0, r) == "dt_bug"
    assert FG.dt_verdict(1.0, 1.0, r) == "dt_bug"            # code Δt 1 h on 30-min files
    assert FG.dt_verdict(0.5, 1.6, r) == "needs_review" and FG.dt_verdict(0.5, 0.5, r) == "needs_review"
    assert FG.dt_verdict(0.5, float("nan"), r) == "needs_review"


def test_hp_select_rule_and_ties():
    assert G.hp_select_rule({(15, 500): 3.0, (31, 200): 2.0, (63, 2000): 2.5}) == (31, 200)
    assert G.hp_select_rule({(31, 500): 1.0, (15, 200): 1.0, (15, 2000): 1.0}) == (15, 2000)


def _acc(ll_hi=0.0002, csi_lo=-0.001, pooled_hi=-0.0001, pod=-0.01, csi=-0.005):
    p25 = [{"lead_min": L, "logloss_hi": ll_hi, "csi_lo": csi_lo} for L in (0, 30, 60)]
    p76 = [{"lead_min": L, "pod_diff": pod, "csi_diff": csi} for L in (0, 30, 60)]
    return G.accept_step(p25, {"diff": -0.001, "lo": -0.002, "hi": pooled_hi}, p76, PR["selection"])


def test_accept_step_boundaries():
    assert _acc()["accept"]
    assert not _acc(pooled_hi=0.0)["accept"]                  # strict < 0
    assert _acc(ll_hi=0.0005)["accept"] and not _acc(ll_hi=0.00051)["accept"]
    assert _acc(csi_lo=-0.005)["accept"] and not _acc(csi_lo=-0.0051)["accept"]
    assert _acc(pod=-0.02)["accept"] and not _acc(pod=-0.021)["accept"]
    assert _acc(csi=-0.01)["accept"] and not _acc(csi=-0.0101)["accept"]
    v = G.accept_step([], {"diff": 0, "lo": 0, "hi": -1}, [], PR["selection"])
    assert not v["accept"] and not v["complete"]


def test_union_adopt():
    rule = PR["union_rule"]
    ok = [{"lead_min": 30, "csi_diff": 0.01, "csi_lo": 0.0, "bias_union": 1.9},
          {"lead_min": 60, "csi_diff": 0.02, "csi_lo": 0.001, "bias_union": 2.0}]
    assert G.union_adopt(ok, rule)["adopt"]
    assert not G.union_adopt([ok[0] | {"csi_lo": -1e-6}, ok[1]], rule)["adopt"]
    assert not G.union_adopt([ok[0] | {"csi_diff": 0.0}, ok[1]], rule)["adopt"]
    assert not G.union_adopt([ok[0], ok[1] | {"bias_union": 2.01}], rule)["adopt"]
    assert not G.union_adopt([ok[0]], rule)["adopt"]          # F60 missing -> not adopted


def test_regional_skill():
    rule = PR["regional"]
    r = [{"lead_min": L, "bss": 0.1, "bss_lo": 0.01, "bss_hi": 0.2, "dbss": 0.05, "dbss_lo": 0.001, "dbss_hi": 0.1}
         for L in (30, 60)]
    assert G.regional_skill(r, rule)["skill"]
    assert not G.regional_skill([r[0], r[1] | {"dbss_lo": -0.001}], rule)["skill"]
    assert not G.regional_skill([r[0] | {"bss_lo": 0.0}, r[1]], rule)["skill"]


def test_choose_cut_csi_finds_the_separating_rate():
    x = np.r_[np.full(50, 1.0), np.full(50, 9.0)]
    y = np.r_[np.zeros(50, bool), np.ones(50, bool)]
    c = G.choose_cut_csi(x, y)
    assert 1.0 < c <= 9.0


def test_rain_class_and_wettest_days():
    out = R.rain_class(np.array([1, 1, 0, 0], bool), np.array([1, 0, 0, 0], bool), np.array([1, 0, 0, 0], bool))
    assert list(out) == ["heavy", "light", "none", "none"]
    b = pd.DataFrame({"day": ["2025-09-30"] * 2 + ["2025-08-01"] * 2 + ["2025-07-01"] * 2 + ["2025-06-01"] * 2 +
                      ["2025-05-01"] * 2, "role": "eval",
                      "label_rate_mm_h": [99, 99, 5, 5, 3, 3, 5, 5, 1, 1]})
    assert R.wettest_days(b) == ["2025-06-01", "2025-08-01", "2025-07-01"]     # tie -> earlier day first


def test_final_rounds_and_setting():
    assert R.final_rounds([{"median_rounds": 100}, {"median_rounds": 151}]) == 126   # 125.5 -> half up
    sel = {"model2_specs": {"2024": "ta", "2025": "ta"},
           "hp_select": {"2024": {"setting": [31, 200], "totals": {"15x500": 5.0, "31x200": 4.0, "63x200": 4.5}},
                         "2025": {"setting": [63, 200], "totals": {"15x500": 5.0, "31x200": 4.4, "63x200": 4.0}}}}
    assert R.resolved_specs(sel)["model2"] == {2024: "b31x200a", 2025: "b63x200a"}
    # folds differ -> lowest summed inner-OOF log-loss: 31x200 = 8.4 < 63x200 = 8.5 < 15x500 = 10.0
    assert R.final_setting(sel, "model2") == "b31x200a"
    assert R.final_setting(sel, "model1b") == "b15x500"


def test_slot_rates_and_dt_rule():
    """slot [s, s+30) = 0.5 x (r(s-30) + r(s)); its mm with Δt 0.5 h per file = 0.25 x (r(s-30) + r(s))."""
    w = pd.date_range("2024-07-01", periods=4, freq="30min", tz="UTC")
    gs = pd.DataFrame({"pixel_id": 1, "window_start_utc": w, "v": [2.0, 4.0, 6.0, np.nan]})
    r = FG.slot_rates(gs, "v")
    assert len(r) == 3 and np.isnan(r.iloc[0, 0]) and r.iloc[1, 0] == 3.0 and r.iloc[2, 0] == 5.0
    assert np.isnan(r.reindex(w).iloc[3, 0])                 # a missing window leaves its slots NaN (strict)
    assert FG.GSMAP_SLOT_DT_H == 0.5


def test_grid_max_regional_label_and_nb3():
    G_ = F.Grid()
    base = (G_.lat_idx_min + 20) * 3600 + G_.lon_idx_min + 20
    pix = np.array([base + di * 3600 + dj for di in range(3) for dj in range(3)], "int64")
    t = pd.Timestamp("2024-07-01", tz="UTC")
    meta = pd.DataFrame({"pixel_id": pix, "t0": t, "label_rate_mm_h": np.r_[np.zeros(8), 8.0]})
    lab = FG.regional_label(meta, 7.6)
    centre = pix[4]
    assert lab[list(pix).index(centre)] == 1 and lab[0] == 0       # corner (0,0) is 2 cells from (2,2)
    rate = pd.DataFrame([np.arange(9, dtype=float)], columns=pix)
    m = FG.nb3_mean(rate, np.array([centre]))
    assert m.iloc[0, 0] == pytest.approx(4.0)


def _conv_rows(pt):
    return pd.DataFrame({"pixel_id": np.array([10, 10], "int64"), "prediction_time": pd.to_datetime(pt, utc=True),
                         "frame_start": pd.to_datetime(pt, utc=True) + pd.Timedelta(minutes=50)})


def test_conv_features_admissibility():
    """value of the hour containing the frame start; issue <= prediction_time - 22 h and available <= prediction_time."""
    T = pd.Timestamp("2024-07-02 12:00", tz="UTC")
    cv = pd.DataFrame({"pixel_id": np.int64(10), "time_utc": [T], "cape": [1500.0], "lifted_index": [-3.0],
                       "forecast_issue_time_utc": [T - pd.Timedelta(hours=24)],
                       "available_time_utc": [T - pd.Timedelta(hours=24) + pd.Timedelta(minutes=420)]})
    pmap = {10: {"near": 10, "within": [10]}}
    rows = _conv_rows(["2024-07-02 11:20", "2024-07-01 12:30"])     # frame start 12:10 -> hour 12; 13:20 -> no data
    f = FG.compute_conv_features(rows, cv, pmap, np.array([1.0, 0.0]))
    assert f["nwp_conv_gfs_cape_near"].iloc[0] == 1500.0 and np.isnan(f["nwp_conv_gfs_cape_near"].iloc[1])
    assert f["nwp_conv_gfs_cape_x_rain"].iloc[0] == 1500.0
    late = cv.assign(forecast_issue_time_utc=[T - pd.Timedelta(hours=21)])     # issue only 21 h before -> refused
    f2 = FG.compute_conv_features(rows, late, pmap, np.array([1.0, 0.0]))
    assert np.isnan(f2["nwp_conv_gfs_cape_near"].iloc[0])
    f3 = FG.compute_conv_features(rows, cv, pmap, np.array([0.2, np.nan]))
    assert f3["nwp_conv_gfs_cape_x_rain"].iloc[0] == 0.0 and np.isnan(f3["nwp_conv_gfs_cape_x_rain"].iloc[1])


def test_conv_g2_flags_late_issue():
    t0 = pd.Timestamp("2024-07-02 11:00", tz="UTC")
    df = pd.DataFrame({"pixel_id": [10], "t0": [t0], "lead_min": [30], "prediction_time": [t0 + pd.Timedelta(minutes=10)],
                       "frame_start": [t0 + pd.Timedelta(minutes=30)], "target_time": [t0 + pd.Timedelta(minutes=60)],
                       "nwp_conv_gfs_cape_near": [100.0],
                       "nwp_conv_gfs__issue_time": [t0 - pd.Timedelta(hours=20)],
                       "nwp_conv_gfs__available_time": [t0 - pd.Timedelta(hours=13)]})
    v = FG.conv_g2(df, ["nwp_conv_gfs_cape_near"])
    assert any("22 h" in str(x) for x in v)
    ok = df.assign(nwp_conv_gfs__issue_time=[t0 - pd.Timedelta(hours=24)])
    assert FG.conv_g2(ok, ["nwp_conv_gfs_cape_near"]) == []


def test_icon_columns_operational_and_hindsight():
    """Operational GC: 10 slots ending at t0 - 30 min + 1 h persistence; hindsight IMERG: 12 slots ending at t0."""
    import yaml
    fi = yaml.safe_load((FG.CODE_ROOT / R.FI_CFG).read_text(encoding="utf-8"))
    ends = pd.date_range("2024-07-01 00:30", periods=60, freq="30min", tz="UTC")
    pix = np.array([1], "int64")
    t0 = pd.Series([pd.Timestamp("2024-07-02 06:00", tz="UTC")])
    gc_last = (ends > t0[0] - pd.Timedelta(hours=5, minutes=30)) & (ends <= t0[0] - pd.Timedelta(minutes=30))
    gc = pd.DataFrame({1: np.where(gc_last, 12.0, 0.0)}, index=ends)    # 12 mm in each of the 10 operational slots
    im_last = (ends > t0[0] - pd.Timedelta(hours=6)) & (ends <= t0[0])
    im = pd.DataFrame({1: np.where(im_last, 5.0, 0.0)}, index=ends)     # 5 mm in each of the 12 hindsight slots
    ic = R.icon_columns(t0, pix, gc, im, np.array([10.0]), "imerg_final", fi)
    assert ic["a6h_mm"].iloc[0] == pytest.approx(130.0) and ic["icon"].iloc[0] == "possible"     # 110 <= 130 < 220
    assert ic["a24h_mm"].iloc[0] == pytest.approx(130.0)                                      # < 270
    assert ic["a6h_mm_hindsight"].iloc[0] == pytest.approx(60.0) and ic["icon_hindsight"].iloc[0] == "possible"
    hi = R.icon_columns(t0, pix, gc * 2, im * 2, np.array([10.0]), "imerg_final", fi)
    assert hi["icon"].iloc[0] == "high" and hi["icon_hindsight"].iloc[0] == "high"         # 250 >= 220, 120 >= 100
    assert ic["icon_product"].iloc[0] == "gsmap_now_gc" and ic["icon_product_hindsight"].iloc[0] == "imerg_final"
