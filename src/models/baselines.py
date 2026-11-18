"""P4A.4 — baselines on validation and test (pilot v0), and reproduction of the P2B persistence numbers.

Baselines per frame (lead 0/30/60) and threshold (0.5/2.5/7.6 mm/h):
  clim     climatology = base rate by month x local hour of frame start, from the tune period
  pers     persistence: newest admissible GC window >= thr (asof_latest for F0/F30, max_avail_0_60 for F60)
  calpers  calibrated persistence: gc_k0 >= c, c maximises CSI vs IMERG >= thr on tune
  logit    logistic regression on 6 features (fit on tune, NaN -> 0, standardised)
Probabilistic baselines (clim, logit) get a cut on validation by the cut rule (as the models do).

Outputs: reports/p4a/baselines/{baseline_metrics.csv, p2b_reproduction.json, calpers_cuts.csv},
data/gold/met_training/pilot_v0/preds/baselines_<split>.parquet (row-level predictions for P4A.5).
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.gold import p4a_common as P  # noqa: E402
from src.models import data as D  # noqa: E402
from src.models import metrics as M  # noqa: E402

OUT = ROOT / "reports" / "p4a" / "baselines"
PREDS = P.GOLD / "preds"
LOCAL = "Asia/Ho_Chi_Minh"


# ---------------------------------------------------------------- P2B reproduction (matching definition)
def p2b_reproduction(cfg: dict, season: int = 2025) -> dict:
    """P2B §6 definition: prediction time t on the IMERG half-hour grid (no issue offset), GSMaP GC windows
    ending t-L (L = 0..180) admissible when archive available_time_utc (Silver, ADD-003) <= t;
    h = 30: asof_latest vs Final[t]; h = 60: max_avail_0_60 vs max(Final[t], Final[t+30]). City pixels."""
    P.guard_source("gsmap_now")
    s, e = date(season, 5, 1), date(season, 9, 30) if season == 2025 else date(season, 10, 31)
    parts, d = [], s - timedelta(days=1)
    while d <= e:
        f = P.SILVER / "gsmap_now" / f"date={d.isoformat()}" / "part.parquet"
        if f.is_file():
            g = pd.read_parquet(f, columns=["pixel_id", "window_end_utc", "available_time_utc", "rain_rate_gc_mm_h",
                                            "is_city_pixel"])
            parts.append(g[g["is_city_pixel"]])
        d += timedelta(days=1)
    gs = pd.concat(parts, ignore_index=True)
    gs["window_end_utc"] = gs["window_end_utc"].astype("datetime64[ns, UTC]")
    gs["available_time_utc"] = gs["available_time_utc"].astype("datetime64[ns, UTC]")
    gs = gs.drop_duplicates(["pixel_id", "window_end_utc"]).set_index(["pixel_id", "window_end_utc"])
    lab = []
    for m in range(5, 10 if season == 2025 else 11):
        lab.append(pd.read_parquet(P.GOLD / "labels" / f"season={season}" / f"month={m:02d}" / "part.parquet",
                                   columns=["pixel_id", "t0", "lead_min", "label_rate_mm_h"]))
    lab = pd.concat(lab, ignore_index=True)
    lab["t0"] = lab["t0"].astype("datetime64[ns, UTC]")
    wide = lab.pivot_table(index=["pixel_id", "t0"], columns="lead_min", values="label_rate_mm_h").reset_index()
    vals = []
    for L in range(0, 181, 30):
        idx = pd.MultiIndex.from_arrays([wide["pixel_id"], wide["t0"] - pd.Timedelta(minutes=L)])
        r = gs.reindex(idx)
        ok = (r["available_time_utc"] <= wide["t0"].to_numpy()).to_numpy()
        vals.append(np.where(ok, r["rain_rate_gc_mm_h"].to_numpy("float64"), np.nan))
    arr = np.vstack(vals)
    first = np.argmax(~np.isnan(arr), axis=0)
    asof = np.where((~np.isnan(arr)).any(0), arr[first, np.arange(arr.shape[1])], np.nan)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mx60 = np.nanmax(arr[:3], axis=0)
    y30 = wide[0].to_numpy("float64")
    y60 = np.where(np.isnan(wide[0]) | np.isnan(wide[30]), np.nan, np.fmax(wide[0], wide[30]))
    out = {"season": season, "definition": "P2B §6 (prediction_time = t, archive MDTM availability, cumulative h)"}
    for h, (x, y) in {30: (asof, y30), 60: (mx60, y60)}.items():
        m = np.isfinite(x) & np.isfinite(y)
        out[f"h{h}"] = {"n": int(m.sum()),
                        **{f"csi_{t:g}": float(M.scores(**M.contingency(y[m] >= t, x[m] >= t))["csi"])
                           for t in cfg["targets"]["thresholds_mm_h"]}}
    ref = {30: [0.32, 0.24, 0.08], 60: [0.32, 0.23, 0.09]}
    out["reference_p2b"] = ref
    out["max_abs_diff_h30"] = float(max(abs(out["h30"][f"csi_{t:g}"] - r)
                                        for t, r in zip(cfg["targets"]["thresholds_mm_h"], ref[30])))
    out["max_abs_diff_h60"] = float(max(abs(out["h60"][f"csi_{t:g}"] - r)
                                        for t, r in zip(cfg["targets"]["thresholds_mm_h"], ref[60])))
    out["within_0p02"] = out["max_abs_diff_h30"] <= 0.02 + 1e-9
    return out


# ---------------------------------------------------------------- baselines
def clim_table(tune: pd.DataFrame, ycol: str) -> pd.Series:
    loc = tune["frame_start"].dt.tz_convert(LOCAL)
    return tune.groupby([loc.dt.month.rename("m"), loc.dt.hour.rename("h")])[ycol].mean()


def clim_lookup(tab: pd.Series, df: pd.DataFrame) -> np.ndarray:
    loc = df["frame_start"].dt.tz_convert(LOCAL)
    idx = pd.MultiIndex.from_arrays([loc.dt.month, loc.dt.hour])
    return tab.reindex(idx).to_numpy("float64")


def calpers_cut(x: np.ndarray, y: np.ndarray) -> dict:
    x = np.nan_to_num(x, nan=-1.0)
    cands = np.unique(np.r_[np.round(np.arange(0.05, 2.0, 0.05), 2), np.round(np.arange(2.0, 40.01, 0.2), 2)])
    best = None
    for c in cands:
        s = M.scores(**M.contingency(y, x >= c))
        if best is None or s["csi"] > best["csi_tune"]:
            best = {"cut_mm_h": float(c), "csi_tune": float(s["csi"]), "bias_tune": float(s["bias"])}
    return best


def logit_fit(X: np.ndarray, y: np.ndarray, seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=seed))
    return m.fit(np.nan_to_num(X, nan=0.0), y)


def evaluate(split: str, lead: int, thr: float, model: str, df: pd.DataFrame, y, p, yhat, pclim, cfg) -> tuple[dict, pd.DataFrame]:
    s = M.summary(y, p, yhat, pclim)
    ds = M.day_stats(df["day"].to_numpy(), y, yhat, p, pclim)
    ci = M.bootstrap_ci(ds, n=cfg["evaluation"]["bootstrap_resamples"], seed=cfg["random_seed"],
                        level=cfg["evaluation"]["ci_level"])
    row = {"split": split, "lead_min": lead, "threshold": thr, "model": model, **s,
           "csi_lo": ci["csi"][0], "csi_hi": ci["csi"][1], "bss_lo": ci["bss_clim"][0], "bss_hi": ci["bss_clim"][1]}
    return row, ds


def main(argv=None) -> int:
    P.require_guard()
    t_start = time.time()
    cfg = P.load_cfg()
    lg = P.get_logger("p4a4_baselines")
    OUT.mkdir(parents=True, exist_ok=True)
    rep = p2b_reproduction(cfg)
    P.write_json(OUT / "p2b_reproduction.json", rep)
    print(f"P2B repro 2025 h30 CSI: {[round(rep['h30'][f'csi_{t:g}'], 3) for t in cfg['targets']['thresholds_mm_h']]} "
          f"h60: {[round(rep['h60'][f'csi_{t:g}'], 3) for t in cfg['targets']['thresholds_mm_h']]} "
          f"within ±0.02: {rep['within_0p02']}")
    lf = cfg["baselines"]["logistic_features"]
    cols = sorted(set([f"gc_k{k}" for k in range(7)] + lf))
    thrs = cfg["targets"]["thresholds_mm_h"]
    rules = cfg["evaluation"]["cut_rule"]
    rows, cuts, preds = [], [], {"validation": [], "test": []}
    for lead in cfg["frames"]["lead_min"]:
        tune = D.load_split(cfg, "tune", cols, lead=lead)
        ev = {s: D.load_split(cfg, s, cols, lead=lead) for s in ("validation", "test")}
        out = {s: ev[s][["pixel_id", "t0", "lead_min", "day", "season"] + D.LABELS].copy() for s in ev}
        pers_fn = D.max_avail_0_60 if lead == 60 else D.asof_latest
        for thr in thrs:
            yc = D.thr_col(thr)
            tab = clim_table(tune, yc)
            cp = calpers_cut(tune["gc_k0"].to_numpy("float64"), tune[yc].to_numpy().astype(bool))
            lm = logit_fit(tune[lf].to_numpy("float64"), tune[yc].to_numpy(), cfg["random_seed"])
            pc_val = clim_lookup(tab, ev["validation"])
            pl_val = lm.predict_proba(np.nan_to_num(ev["validation"][lf].to_numpy("float64"), nan=0.0))[:, 1]
            yv = ev["validation"][yc].to_numpy().astype(bool)
            cut_clim = M.choose_cut(yv, pc_val, rules[f"{thr:g}"])
            cut_logit = M.choose_cut(yv, pl_val, rules[f"{thr:g}"])
            cuts.append({"lead_min": lead, "threshold": thr, "calpers_cut_mm_h": cp["cut_mm_h"],
                         "calpers_csi_tune": cp["csi_tune"], "calpers_bias_tune": cp["bias_tune"],
                         "clim_cut": cut_clim["cut"], "logit_cut": cut_logit["cut"], "cut_rule": rules[f"{thr:g}"]})
            for s, df in ev.items():
                y = df[yc].to_numpy().astype(bool)
                pc = clim_lookup(tab, df)
                pl = lm.predict_proba(np.nan_to_num(df[lf].to_numpy("float64"), nan=0.0))[:, 1]
                fc = {"clim": (pc, pc >= cut_clim["cut"]),
                      "pers": None, "calpers": None,
                      "logit": (pl, pl >= cut_logit["cut"])}
                xp = np.nan_to_num(pers_fn(df), nan=-1.0)
                fc["pers"] = ((xp >= thr).astype(float), xp >= thr)
                x0 = np.nan_to_num(df["gc_k0"].to_numpy("float64"), nan=-1.0)
                fc["calpers"] = ((x0 >= cp["cut_mm_h"]).astype(float), x0 >= cp["cut_mm_h"])
                k = f"{thr:g}".replace(".", "p")
                out[s][f"p_clim_{k}"] = pc
                for name, (p, yh) in fc.items():
                    r, _ = evaluate(s, lead, thr, name, df, y, p, yh, pc, cfg)
                    rows.append(r)
                    out[s][f"yhat_{name}_{k}"] = yh.astype("int8")
                    if name == "logit":
                        out[s][f"p_logit_{k}"] = p.astype("float32")
            lg.info("lead %s thr %s done rss=%.2f", lead, thr, P.rss_gb())
        for s in out:
            preds[s].append(out[s])
        P.check_rss(cfg["resources"]["max_rss_gb"])
    for s, parts in preds.items():
        P.write_parquet(pd.concat(parts, ignore_index=True), PREDS / f"baselines_{s}.parquet")
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "baseline_metrics.csv", index=False)
    pd.DataFrame(cuts).to_csv(OUT / "calpers_cuts.csv", index=False)
    piv = res[(res["split"] == "test")].pivot_table(index=["threshold", "lead_min"], columns="model", values="csi")
    print("TEST CSI (rows: threshold, lead):")
    print(piv.round(3).to_string())
    print(f"done in {time.time() - t_start:.0f}s rss={P.rss_gb():.2f}GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
