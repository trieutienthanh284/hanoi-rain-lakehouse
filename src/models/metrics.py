"""Verification metrics for the pilot nowcast (P4A): categorical, probabilistic, reliability,
and day-block bootstrap (confidence intervals and paired differences between two forecasts).

Bootstrap works on per-day sufficient statistics, so 1,000 resamples are a matrix product.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SUFF = ["n", "tp", "fp", "fn", "tn", "bs", "bs_clim", "obs"]


def contingency(y: np.ndarray, yhat: np.ndarray) -> dict:
    y = np.asarray(y).astype(bool); f = np.asarray(yhat).astype(bool)
    return {"tp": int((f & y).sum()), "fp": int((f & ~y).sum()), "fn": int((~f & y).sum()), "tn": int((~f & ~y).sum())}


def _div(a, b):
    a = np.asarray(a, dtype="float64"); b = np.asarray(b, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)


def scores(tp, fp, fn, tn=None) -> dict:
    """POD, FAR, CSI, frequency bias (arrays allowed)."""
    return {"pod": _div(tp, tp + fn), "far": _div(fp, tp + fp), "csi": _div(tp, tp + fp + fn),
            "bias": _div(tp + fp, tp + fn)}


def brier(y, p) -> float:
    y = np.asarray(y, dtype="float64"); p = np.asarray(p, dtype="float64")
    return float(np.mean((p - y) ** 2))


def bss(y, p, p_ref) -> float:
    b, r = brier(y, p), brier(y, p_ref)
    return float(1 - b / r) if r > 0 else float("nan")


def auc_pr(y, p) -> float:
    from sklearn.metrics import average_precision_score
    y = np.asarray(y).astype(int)
    return float(average_precision_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


def accuracy(y, yhat) -> float:
    return float(np.mean(np.asarray(y).astype(bool) == np.asarray(yhat).astype(bool)))


def reliability(y, p, bins: int = 10) -> pd.DataFrame:
    y = np.asarray(y, dtype="float64"); p = np.asarray(p, dtype="float64")
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        rows.append({"bin": b, "p_lo": edges[b], "p_hi": edges[b + 1], "n": int(m.sum()),
                     "mean_p": float(p[m].mean()) if m.any() else np.nan,
                     "obs_freq": float(y[m].mean()) if m.any() else np.nan})
    return pd.DataFrame(rows)


def summary(y, p, yhat, p_clim) -> dict:
    c = contingency(y, yhat)
    s = scores(c["tp"], c["fp"], c["fn"], c["tn"])
    return {"n": len(y), "base_rate": float(np.mean(y)), **c, **{k: float(v) for k, v in s.items()},
            "brier": brier(y, p), "bss_clim": bss(y, p, p_clim), "auc_pr": auc_pr(y, p),
            "accuracy": accuracy(y, yhat), "accuracy_always_no": accuracy(y, np.zeros(len(y), bool))}


# ---------------------------------------------------------------- day-block bootstrap
def day_stats(day, y, yhat, p, p_clim) -> pd.DataFrame:
    """Per-day sufficient statistics (index = day)."""
    y = np.asarray(y).astype(bool); f = np.asarray(yhat).astype(bool)
    p = np.asarray(p, dtype="float64"); pc = np.asarray(p_clim, dtype="float64")
    d = pd.DataFrame({"day": np.asarray(day), "n": 1, "tp": f & y, "fp": f & ~y, "fn": ~f & y, "tn": ~f & ~y,
                      "bs": (p - y) ** 2, "bs_clim": (pc - y) ** 2, "obs": y})
    return d.groupby("day")[SUFF].sum().astype("float64")


def _resample_weights(n_days: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_days, np.full(n_days, 1 / n_days), size=n).astype("float64")


def _metrics_from(tot: np.ndarray) -> dict:
    """tot[..., SUFF] -> metric arrays."""
    n, tp, fp, fn, tn, bs, bsc, _ = (tot[..., i] for i in range(len(SUFF)))
    s = scores(tp, fp, fn, tn)
    return {"csi": s["csi"], "pod": s["pod"], "far": s["far"], "bias": s["bias"], "brier": _div(bs, n),
            "bss_clim": 1 - _div(bs, bsc)}


def bootstrap_ci(ds: pd.DataFrame, n: int = 1000, seed: int = 42, level: float = 0.90) -> dict:
    w = _resample_weights(len(ds), n, seed)
    tot = w @ ds[SUFF].to_numpy()
    m = _metrics_from(tot)
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    return {k: (float(np.nanquantile(v, lo)), float(np.nanquantile(v, hi))) for k, v in m.items()}


def paired_bootstrap(ds_a: pd.DataFrame, ds_b: pd.DataFrame, n: int = 1000, seed: int = 42,
                     level: float = 0.90) -> dict:
    """Differences A − B (CSI, Brier, BSS) on the same days; returns point estimate and interval."""
    days = ds_a.index.intersection(ds_b.index)
    a = ds_a.loc[days, SUFF].to_numpy(); b = ds_b.loc[days, SUFF].to_numpy()
    w = _resample_weights(len(days), n, seed)
    ma, mb = _metrics_from(w @ a), _metrics_from(w @ b)
    pa, pb = _metrics_from(a.sum(0)), _metrics_from(b.sum(0))
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    out = {}
    for k in ("csi", "brier", "bss_clim", "pod", "far"):
        d = ma[k] - mb[k]
        out[k] = {"diff": float(pa[k] - pb[k]), "lo": float(np.nanquantile(d, lo)), "hi": float(np.nanquantile(d, hi))}
    return out


# ---------------------------------------------------------------- probability cuts
def cut_curve(y, p) -> pd.DataFrame:
    """CSI and frequency bias for every distinct cut c (forecast yes when p >= c)."""
    y = np.asarray(y).astype(bool); p = np.asarray(p, dtype="float64")
    o = np.argsort(-p, kind="mergesort")
    ps, ys = p[o], y[o]
    tp = np.cumsum(ys); fp = np.cumsum(~ys)
    last = np.r_[ps[1:] != ps[:-1], True]           # keep the last index of each distinct value
    tp, fp, c = tp[last], fp[last], ps[last]
    P = y.sum()
    fn = P - tp
    return pd.DataFrame({"cut": c, "tp": tp, "fp": fp, "fn": fn,
                         "csi": _div(tp, tp + fp + fn), "bias": _div(tp + fp, P)})


def choose_cut(y, p, rule: str) -> dict:
    cc = cut_curve(y, p)
    if rule == "max_csi":
        r = cc.iloc[int(np.nanargmax(cc["csi"].to_numpy()))]
    elif rule.startswith("freq_bias_"):
        target = float(rule.split("_")[-1])
        r = cc.iloc[int(np.nanargmin(np.abs(cc["bias"].to_numpy() - target)))]
    else:
        raise ValueError(rule)
    return {"rule": rule, "cut": float(r["cut"]), "csi": float(r["csi"]), "bias": float(r["bias"])}
