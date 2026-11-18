"""Feature set A (src/gold/features.py) — synthetic checks: admissibility, indexing, neighbourhood, cap."""
import numpy as np
import pandas as pd

from src.gold import features as F
from src.gold.p4a_common import load_cfg

CFG = load_cfg()
P = F.feature_params(CFG)
G = F.Grid()
PIX = np.array([(1095 + 15) * 3600 + 2843 + 14])          # centre of the domain
T0 = pd.DatetimeIndex([pd.Timestamp("2024-09-06 12:00", tz="UTC")])


def wins(values: dict) -> dict:
    out = {}
    for we, v in values.items():
        a = np.full((G.n_lat, G.n_lon), v, "float32")
        out[pd.Timestamp(we, tz="UTC")] = (a, a.copy())
    return out


def test_future_window_never_used():
    w = wins({"2024-09-06 12:00": 1.0, "2024-09-06 12:30": 99.0, "2024-09-06 11:30": 2.0})
    gc, raw = F.stack_windows(w, T0, P, G.n_lat, G.n_lon)
    df = F.compute_features(gc, raw, T0, PIX, P)
    assert df["gc_k0"].iloc[0] == 1.0 and df["gc_k1"].iloc[0] == 2.0
    assert (df.filter(like="gc_").max(axis=1) < 99).all()
    assert df["gsmap__event_time"].iloc[0] == T0[0]
    assert df["gsmap__available_time"].iloc[0] == T0[0] + pd.Timedelta(minutes=5)


def test_admissibility_rule():
    assert F.admissible(T0[0], T0[0], P)
    assert not F.admissible(T0[0] + pd.Timedelta(minutes=30), T0[0], P)
    p = dict(P, avail_after=15)                 # later publication than the issue offset → k0 inadmissible
    assert not F.admissible(T0[0], T0[0], p)


def test_missing_and_age():
    w = wins({"2024-09-06 11:00": 3.0})        # only k = 2 present
    gc, raw = F.stack_windows(w, T0, P, G.n_lat, G.n_lon)
    df = F.compute_features(gc, raw, T0, PIX, P)
    r = df.iloc[0]
    assert np.isnan(r["gc_k0"]) and r["gc_k2"] == 3.0
    assert r["gsmap_n_missing"] == 6 and r["gsmap_age_min"] == 10 + 60
    assert r["gc_min_since_wet"] == 60


def test_neighbourhood_and_ring():
    w = wins({"2024-09-06 12:00": 0.0})
    a = w[pd.Timestamp("2024-09-06 12:00", tz="UTC")][0]
    i, j = 15, 14
    a[i + 1, j] = 4.0                            # inside radius 1
    a[i + 5, j] = 10.0                           # 0.5 deg north: ring N, radius 5
    gc, raw = F.stack_windows(w, T0, P, G.n_lat, G.n_lon)
    r = F.compute_features(gc, raw, T0, PIX, P).iloc[0]
    assert r["nb_r1_max_k0"] == 4.0 and abs(r["nb_r1_wet_k0"] - 1 / 9) < 1e-6
    assert r["nb_r5_max_k0"] == 10.0
    assert r["ring_N_wet_k0"] > 0 and r["ring_S_wet_k0"] == 0 and r["ring_E_wet_k0"] == 0


def test_cap_flag_and_clipping():
    w = wins({"2024-09-06 12:00": 200.0})
    gc, raw = F.stack_windows(w, T0, P, G.n_lat, G.n_lon)
    r = F.compute_features(gc, raw, T0, PIX, P).iloc[0]
    assert r["raw_capped"] == 1.0 and r["raw_k0"] == 200.0 and r["nb_r2_max_k0"] <= 200.0


def test_batch_equals_single():
    w = wins({f"2024-09-06 {h:02d}:{mm:02d}": float(h + mm / 60) for h in range(8, 13) for mm in (0, 30)})
    t0s = pd.DatetimeIndex(pd.date_range("2024-09-06 11:00", periods=3, freq="30min", tz="UTC"))
    gc, raw = F.stack_windows(w, t0s, P, G.n_lat, G.n_lon)
    batch = F.compute_features(gc, raw, t0s, PIX, P)
    for n, t in enumerate(t0s):
        g1, r1 = F.stack_windows(w, pd.DatetimeIndex([t]), P, G.n_lat, G.n_lon)
        one = F.compute_features(g1, r1, pd.DatetimeIndex([t]), PIX, P)
        pd.testing.assert_frame_equal(batch.iloc[[n]].reset_index(drop=True), one)
