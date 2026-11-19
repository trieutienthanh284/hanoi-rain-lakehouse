"""P4B.2 — v1 features of src/gold/features.py on synthetic fields: moving blobs with known speed and direction,
zero-rain and noise fields (motion_valid = false), no future window, serving (T = 1) == batch, v0 unchanged."""
import numpy as np
import pandas as pd
import pytest
import yaml

from src.gold import features as F
from src.gold.p4a_common import ROOT

CFG1 = yaml.safe_load((ROOT / "configs" / "model_v1.yaml").read_text(encoding="utf-8"))
CFG0 = yaml.safe_load((ROOT / "configs" / "model_pilot.yaml").read_text(encoding="utf-8"))
P1 = F.feature_params(CFG1)
P0 = F.feature_params(CFG0)
G = F.Grid()
T0 = pd.Timestamp("2024-07-10 06:00", tz="UTC")


def pid(i, j):
    return (G.lat_idx_min + i) * 3600 + G.lon_idx_min + j


def blob(ci, cj, peak=10.0, sigma=2.0):
    gi, gj = np.meshgrid(np.arange(G.n_lat), np.arange(G.n_lon), indexing="ij")
    return (peak * np.exp(-((gi - ci) ** 2 + (gj - cj) ** 2) / (2 * sigma ** 2))).astype("float32")


def moving(u, v, ci=15.0, cj=14.0, t0=T0, n=7):
    """Windows k = 0..n-1 of a blob at (ci, cj) at t0 moving u (east) / v (north) pixels per hour."""
    out = {}
    for k in range(n):
        a = blob(ci - v * 0.5 * k, cj - u * 0.5 * k)
        out[t0 - pd.Timedelta(minutes=30 * k)] = (a, a.copy())
    return out


def feats(w, pix, t0s=None, p=P1):
    t0s = t0s if t0s is not None else pd.DatetimeIndex([T0])
    gc, raw = F.stack_windows(w, t0s, p, G.n_lat, G.n_lon)
    return F.compute_features(gc, raw, t0s, np.asarray(pix), p, G)


@pytest.mark.parametrize("u,v", [(3.0, 2.0), (-2.0, -3.0), (1.0, -4.0), (0.0, 0.0)])
def test_motion_vector_of_moving_blob(u, v):
    r = feats(moving(u, v), [pid(15, 14)]).iloc[0]
    assert r["motion_valid"] == 1.0 and r["motion_corr"] > 0.9
    assert abs(r["motion_u"] - u) < 0.35 and abs(r["motion_v"] - v) < 0.35


def test_advected_persistence_moves_rain_downstream():
    u, v = 3.0, 2.0                                    # blob centre (12, 11) at t0
    w = moving(u, v, ci=12.0, cj=11.0)
    tgt = (15, 16)                                     # centre at F60 mid-frame (1.75 h): (15.5, 16.25)
    r = feats(w, [pid(*tgt)]).iloc[0]
    f0 = w[T0][0]
    assert r["adv_rate_F60"] > 0.5 * f0.max() and r["adv_rate_F60"] > 5 * r["gc_k0"]
    assert r["adv_rate_F0"] < r["adv_rate_F30"] < r["adv_rate_F60"]     # blob approaches the target
    assert r["adv_max_r1_F60"] >= r["adv_rate_F60"]
    # exact bilinear value: shift by the estimated vector over the frame centre time
    h = F.advect_hours(60, P1)
    exp = F.bilinear(f0[None], np.array([[tgt[0] - r["motion_v"] * h]]), np.array([[tgt[1] - r["motion_u"] * h]]))[0, 0]
    assert abs(r["adv_rate_F60"] - exp) < 1e-4


def test_upwind_sector_is_against_the_motion():
    w = moving(3.0, 0.0, ci=15.0, cj=10.0)              # moving east
    df = feats(w, [pid(15, 15), pid(15, 5)])            # downstream (east) and upstream (west) targets
    down, up = df.iloc[0], df.iloc[1]
    assert down["upwind_mean"] > 1.0 and down["upwind_wet"] > 0.2
    assert up["upwind_mean"] < 0.05 and up["upwind_wet"] == 0.0


def test_zero_rain_is_invalid_and_advection_falls_back_to_persistence():
    z = np.zeros((G.n_lat, G.n_lon), "float32")
    w = {T0 - pd.Timedelta(minutes=30 * k): (z, z.copy()) for k in range(7)}
    r = feats(w, [pid(15, 14)]).iloc[0]
    assert r["motion_valid"] == 0.0 and np.isnan(r["motion_u"]) and np.isnan(r["upwind_mean"])
    assert r["adv_rate_F30"] == 0.0 and r["extent_wet0p5_k0"] == 0.0


def test_noise_fields_are_invalid():
    rng = np.random.default_rng(7)
    t0s = pd.DatetimeIndex(pd.date_range("2024-07-10 06:00", periods=4, freq="30min", tz="UTC"))
    w = {}
    for k in range(-1, 10):                            # independent noise in every window
        a = rng.gamma(0.5, 2.0, size=(G.n_lat, G.n_lon)).astype("float32")
        w[t0s[-1] - pd.Timedelta(minutes=30 * k)] = (a, a.copy())
    df = feats(w, [pid(15, 14)], t0s)
    assert (df["motion_valid"] == 0.0).all() and df["motion_u"].isna().all()


def test_no_future_window_is_used():
    w = moving(2.0, 1.0)
    base = feats(w, [pid(15, 14), pid(5, 20)])
    fut = dict(w)
    for m in (30, 60, 90):                             # windows ending after t0 (and an inadmissible one)
        a = blob(3.0, 3.0, peak=80.0)
        fut[T0 + pd.Timedelta(minutes=m)] = (a, a.copy())
    pd.testing.assert_frame_equal(base, feats(fut, [pid(15, 14), pid(5, 20)]))
    # k = 0 published too late for prediction_time -> not used by the motion or advection either
    late = dict(P1, avail_after=15)
    r = feats(w, [pid(15, 14)], p=late).iloc[0]
    assert np.isnan(r["gc_k0"]) and np.isnan(r["adv_rate_F0"]) and np.isnan(r["extent_wet0p5_k0"])


def test_extent_features():
    a = np.zeros((G.n_lat, G.n_lon), "float32"); a[:15] = 3.0
    z = np.zeros_like(a)
    w = {T0: (a, a.copy())} | {T0 - pd.Timedelta(minutes=30 * k): (z, z.copy()) for k in range(1, 7)}
    r = feats(w, [pid(20, 14)]).iloc[0]
    assert r["extent_wet0p5_k0"] == 0.5 and r["extent_wet2p5_k0"] == 0.5 and r["extent_wet2p5_d20"] == 0.5


def test_no_window_means_all_v1_columns_nan():
    r = feats({}, [pid(15, 14)]).iloc[0]
    v1 = [c for c in r.index if c.startswith(("motion_", "adv_", "upwind_", "extent_"))]
    assert len(v1) == 19 and r[v1].isna().all()


def test_v1_columns_and_groups_and_v0_unchanged():
    d1 = feats(moving(1.0, 1.0), [pid(15, 14)])
    d0 = feats(moving(1.0, 1.0), [pid(15, 14)], p=P0)
    c1, c0 = F.feature_columns(d1), F.feature_columns(d0)
    assert "cal_doy_sin" in c0 and "cal_doy_cos" in c0 and not any(c.startswith("cal_doy") for c in c1)
    assert "cal_month" in c1 and d1["cal_month"].iloc[0] == 7.0
    common = [c for c in c0 if c in c1]
    assert len(common) == len(c0) - 2                  # every v0 column except doy is identical in v1
    pd.testing.assert_frame_equal(d0[common], d1[common])
    for c in c1:
        assert F.group_of(c) in ("gsmap", "gsmap_meta", "calendar", "static")
        assert c.startswith(tuple(CFG1["features"]["allowlist_prefixes"]))


def test_batch_equals_single_v1():
    w = moving(2.0, -1.0, n=12)
    t0s = pd.DatetimeIndex(pd.date_range(T0 - pd.Timedelta(minutes=90), periods=4, freq="30min"))
    pix = [pid(15, 14), pid(3, 25)]
    batch = feats(w, pix, t0s)
    for n, t in enumerate(t0s):
        one = feats(w, pix, pd.DatetimeIndex([t]))
        pd.testing.assert_frame_equal(batch.iloc[2 * n:2 * n + 2].reset_index(drop=True), one)
