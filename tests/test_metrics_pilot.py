import numpy as np
import pandas as pd

from src.models import metrics as M


def test_contingency_and_scores():
    y = np.array([1, 1, 0, 0, 1, 0]); f = np.array([1, 0, 1, 0, 1, 0])
    c = M.contingency(y, f)
    assert c == {"tp": 2, "fp": 1, "fn": 1, "tn": 2}
    s = M.scores(**c)
    assert abs(s["pod"] - 2 / 3) < 1e-12 and abs(s["far"] - 1 / 3) < 1e-12
    assert abs(s["csi"] - 0.5) < 1e-12 and abs(s["bias"] - 1.0) < 1e-12


def test_brier_bss_accuracy():
    y = np.array([1, 0, 1, 0]); p = np.array([1.0, 0.0, 0.5, 0.5])
    assert M.brier(y, p) == 0.125
    assert abs(M.bss(y, p, np.full(4, 0.5)) - 0.5) < 1e-12
    assert M.accuracy(y, p >= 0.5) == 0.75


def test_reliability_bins():
    p = np.array([0.05, 0.05, 0.95, 0.95]); y = np.array([0, 1, 1, 1])
    r = M.reliability(y, p, 10)
    assert r.loc[0, "n"] == 2 and r.loc[0, "obs_freq"] == 0.5 and r.loc[9, "obs_freq"] == 1.0


def test_bootstrap_identical_models_zero_difference():
    rng = np.random.default_rng(0)
    n = 5000; day = np.repeat(np.arange(50), 100)
    y = rng.random(n) < 0.1; p = np.clip(y * 0.6 + rng.random(n) * 0.3, 0, 1)
    ds = M.day_stats(day, y, p >= 0.5, p, np.full(n, 0.1))
    d = M.paired_bootstrap(ds, ds, n=200)
    assert d["csi"]["diff"] == 0 and d["csi"]["lo"] == 0 and d["csi"]["hi"] == 0
    ci = M.bootstrap_ci(ds, n=200)
    full = M.summary(y, p, p >= 0.5, np.full(n, 0.1))
    assert ci["csi"][0] <= full["csi"] <= ci["csi"][1]


def test_bootstrap_detects_better_model():
    rng = np.random.default_rng(1)
    n = 20000; day = np.repeat(np.arange(200), 100)
    y = rng.random(n) < 0.1
    good = np.clip(y * 0.7 + rng.random(n) * 0.3, 0, 1); bad = rng.random(n)
    a = M.day_stats(day, y, good >= 0.5, good, np.full(n, 0.1))
    b = M.day_stats(day, y, bad >= 0.9, bad, np.full(n, 0.1))
    d = M.paired_bootstrap(a, b, n=300)
    assert d["csi"]["lo"] > 0 and d["brier"]["hi"] < 0


def test_choose_cut():
    y = np.array([1, 1, 0, 0, 0, 0, 1, 0]); p = np.array([.9, .8, .7, .1, .2, .3, .6, .05])
    c = M.choose_cut(y, p, "max_csi")
    assert c["cut"] == 0.6 and abs(c["csi"] - 0.75) < 1e-12
    b = M.choose_cut(y, p, "freq_bias_1.5")
    assert abs(b["bias"] - 1.33333) < 1e-3 or abs(b["bias"] - 1.66667) < 1e-3
