"""P4A.6 (preliminary) and P4A.7 (evaluation only) — no model is trained here (user, 2026-11-18: ~1 GB free RAM;
heavy training moves to TASK_P4C_CLOUD_TRAINING.md). Every probability metric is reported for the raw LightGBM
output (`p_raw`) next to the isotonic-calibrated one (`p`).

  python src/models/evaluate_extra.py prelim-B              # the set-B models that exist (F0 0.5 / 2.5)
  python src/models/evaluate_extra.py strata                # by month, local-hour block, 10 wettest test days
  python src/models/evaluate_extra.py stale --month 2025-07 # one test month per invocation (resumable)
  python src/models/evaluate_extra.py stale-summary

Outputs in reports/p4a/: lgbm_B_preliminary/, robustness/. Reads one prediction file / one month of table_A
at a time; RSS is checked against the user limit (0.7 GB) after every partition.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.gold import p4a_common as P  # noqa: E402
from src.models import data as D  # noqa: E402
from src.models import metrics as M  # noqa: E402
from src.models.evaluate_lgbm import BASELINES, baseline_p, load_pair  # noqa: E402
from src.models.train_lgbm import MODELS, kstr  # noqa: E402

RSS_LIMIT_GB = 0.7                       # user instruction 2026-11-18
REP = ROOT / "reports" / "p4a"
ROB = REP / "robustness"
TZ = "Asia/Ho_Chi_Minh"
# local hour of the frame start -> block (D72)
BLOCKS = [("night", 0, 6), ("morning", 6, 12), ("afternoon", 12, 18), ("evening", 18, 24)]
# models evaluated in P4A.7: chosen variant A_no2023 (2.5, 7.6) and A (all three thresholds; the only 0.5 models)
P47_MODELS = [("A_no2023", 2.5), ("A_no2023", 7.6), ("A", 0.5), ("A", 2.5), ("A", 7.6)]


def rss_check() -> None:
    P.check_rss(RSS_LIMIT_GB)


def ece(y, p, bins: int = 10) -> float:
    r = M.reliability(np.asarray(y), np.asarray(p, "float64"), bins).dropna()
    return float((r["n"] / r["n"].sum() * (r["mean_p"] - r["obs_freq"]).abs()).sum())


def prob_metrics(y, p_raw, p_cal, p_clim) -> dict:
    y = np.asarray(y).astype(bool)
    return {"brier_raw": M.brier(y, p_raw), "brier_cal": M.brier(y, p_cal),
            "bss_raw": M.bss(y, p_raw, p_clim), "bss_cal": M.bss(y, p_cal, p_clim),
            "auc_pr_raw": M.auc_pr(y, p_raw), "auc_pr_cal": M.auc_pr(y, p_cal),
            "ece_raw": ece(y, p_raw), "ece_cal": ece(y, p_cal),
            "mean_p_raw": float(np.mean(p_raw)), "mean_p_cal": float(np.mean(p_cal)), "base_rate": float(np.mean(y))}


def cat_metrics(y, yhat) -> dict:
    c = M.contingency(np.asarray(y).astype(bool), np.asarray(yhat).astype(bool))
    s = M.scores(c["tp"], c["fp"], c["fn"], c["tn"])
    return {"csi": float(s["csi"]), "pod": float(s["pod"]), "far": float(s["far"]), "bias": float(s["bias"])}


def raw_cut(variant: str, lead: int, thr: float, rule: str) -> float:
    """Cut on the RAW probability chosen on validation with the same rule (choice on validation only)."""
    k = kstr(thr)
    v = pd.read_parquet(P.GOLD / "preds" / f"lgbm_{variant}_lead{lead}_thr{k}_validation.parquet",
                        columns=[D.thr_col(thr), "p_raw"])
    return M.choose_cut(v[D.thr_col(thr)].to_numpy(), v["p_raw"].to_numpy("float64"), rule)["cut"]


# ------------------------------------------------------------------------------------------- P4A.6 preliminary
def prelim_b(cfg: dict) -> int:
    ev = cfg["evaluation"]
    out = REP / "lgbm_B_preliminary"
    out.mkdir(parents=True, exist_ok=True)
    rows, diffs = [], []
    for d in sorted((MODELS / "B").glob("lead*_thr*")):
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        lead, thr = int(meta["lead_min"]), float(meta["threshold_mm_h"])
        k = kstr(thr); yc = D.thr_col(thr)
        df = load_pair("B", lead, thr)
        y = df[yc].to_numpy().astype(bool); pc = df[f"p_clim_{k}"].to_numpy("float64")
        p = df["p"].to_numpy("float64"); pr = df["p_raw"].to_numpy("float64"); yh = df["yhat"].to_numpy().astype(bool)
        ds_m = M.day_stats(df["day"].to_numpy(), y, yh, p, pc)
        ci = M.bootstrap_ci(ds_m, ev["bootstrap_resamples"], cfg["random_seed"], ev["ci_level"])
        rc = raw_cut("B", lead, thr, meta["cut"]["rule"])
        r = {"variant": "B", "lead_min": lead, "threshold": thr, "status": "PRELIMINARY (2/9 models)",
             "val_csi_at_cut": meta["cut"]["csi"], "best_iteration": meta["best_iteration"],
             "cut_cal": meta["cut"]["cut"], **cat_metrics(y, yh), "csi_lo": ci["csi"][0], "csi_hi": ci["csi"][1],
             "cut_raw_val": rc, **{f"{kk}_rawcut": vv for kk, vv in cat_metrics(y, pr >= rc).items()},
             **prob_metrics(y, pr, p, pc)}
        a_path = P.GOLD / "preds" / f"lgbm_A_no2023_lead{lead}_thr{k}_test.parquet"
        if a_path.is_file():                       # descriptive only: no paired test (compare_A_B not run)
            a = pd.read_parquet(a_path, columns=["pixel_id", "t0", "p", "p_raw", "yhat"]).merge(
                df[["pixel_id", "t0", yc, f"p_clim_{k}"]], on=["pixel_id", "t0"], how="inner", validate="one_to_one")
            ya = a[yc].to_numpy().astype(bool); pca = a[f"p_clim_{k}"].to_numpy("float64")
            r |= {"A_no2023_n_common": len(a), "A_no2023_csi": cat_metrics(ya, a["yhat"])["csi"],
                  "A_no2023_bss_raw": M.bss(ya, a["p_raw"].to_numpy("float64"), pca),
                  "A_no2023_bss_cal": M.bss(ya, a["p"].to_numpy("float64"), pca),
                  "A_no2023_val_csi_at_cut": json.loads((MODELS / "A_no2023" / f"lead{lead}_thr{k}" / "meta.json")
                                                        .read_text(encoding="utf-8"))["cut"]["csi"]}
            del a
        rows.append(r)
        for b in BASELINES:
            yb = df[f"yhat_{b}_{k}"].to_numpy().astype(bool)
            ds_b = M.day_stats(df["day"].to_numpy(), y, yb, baseline_p(df, b, k), pc)
            dd = M.paired_bootstrap(ds_m, ds_b, ev["bootstrap_resamples"], cfg["random_seed"], ev["ci_level"])
            diffs.append({"variant": "B", "lead_min": lead, "threshold": thr, "baseline": b,
                          **{f"{m}_{q}": dd[m][q] for m in ("csi", "brier") for q in ("diff", "lo", "hi")}})
        del df
        rss_check()
    pd.DataFrame(rows).to_csv(out / "test_metrics.csv", index=False)
    pd.DataFrame(diffs).to_csv(out / "diffs_vs_baselines.csv", index=False)
    t = pd.DataFrame(rows)
    print(t[["threshold", "lead_min", "csi", "csi_lo", "csi_hi", "bss_raw", "bss_cal", "auc_pr_raw",
             "A_no2023_csi" if "A_no2023_csi" in t else "csi"]].round(3).to_string(index=False))
    print(f"rss={P.rss_gb():.2f}GB")
    return 0


# ------------------------------------------------------------------------------------------- P4A.7 strata
def block_of(local_hour: np.ndarray) -> np.ndarray:
    out = np.empty(len(local_hour), dtype=object)
    for name, a, b in BLOCKS:
        out[(local_hour >= a) & (local_hour < b)] = name
    return out


def wettest_days(n: int = 10) -> pd.DataFrame:
    """Test days (local date of t0) ranked by the mean IMERG Final rate over the city pixels (frame F0)."""
    p = P.GOLD / "preds" / "baselines_test.parquet"
    b = pd.read_parquet(p, columns=["day", "lead_min", "label_rate_mm_h"])
    b = b[b["lead_min"] == 0]
    g = b.groupby("day")["label_rate_mm_h"].agg(["mean", "max", "size"]).sort_values("mean", ascending=False)
    g["city_mean_mm_day"] = g["mean"] * 24.0
    return g.head(n).reset_index()


def strata(cfg: dict) -> int:
    ROB.mkdir(parents=True, exist_ok=True)
    wd = wettest_days(10)
    wd.to_csv(ROB / "wettest_test_days.csv", index=False)
    wset = set(wd["day"])
    rows = []
    for variant, thr in P47_MODELS:
        k = kstr(thr); yc = D.thr_col(thr)
        for lead in cfg["frames"]["lead_min"]:
            df = load_pair(variant, lead, thr)
            meta = json.loads((MODELS / variant / f"lead{lead}_thr{k}" / "meta.json").read_text(encoding="utf-8"))
            rc = raw_cut(variant, lead, thr, meta["cut"]["rule"])
            fs = (df["t0"] + pd.to_timedelta(df["lead_min"].astype(int), unit="min")).dt.tz_convert(TZ)
            strat = {"all": np.full(len(df), "all", dtype=object),
                     "month": df["t0"].dt.tz_convert(TZ).dt.month.astype(str).to_numpy(object),
                     "hour_block": block_of(fs.dt.hour.to_numpy()),
                     "wettest10": np.where(df["day"].isin(wset), "wettest10", "other")}
            y = df[yc].to_numpy().astype(bool)
            for kind, lab in strat.items():
                for val in pd.unique(lab):
                    s = lab == val
                    if s.sum() == 0:
                        continue
                    d = df.loc[s]
                    r = {"variant": variant, "threshold": thr, "lead_min": lead, "stratum": kind, "value": val,
                         "n": int(s.sum()), **cat_metrics(y[s], d["yhat"]),
                         "csi_rawcut": cat_metrics(y[s], d["p_raw"].to_numpy() >= rc)["csi"],
                         "csi_pers": cat_metrics(y[s], d[f"yhat_pers_{k}"])["csi"],
                         "csi_calpers": cat_metrics(y[s], d[f"yhat_calpers_{k}"])["csi"],
                         "csi_logit": cat_metrics(y[s], d[f"yhat_logit_{k}"])["csi"],
                         **prob_metrics(y[s], d["p_raw"].to_numpy("float64"), d["p"].to_numpy("float64"),
                                        d[f"p_clim_{k}"].to_numpy("float64"))}
                    rows.append(r)
            # per wettest day (CSI only)
            for day in wd["day"]:
                s = (df["day"] == day).to_numpy()
                rows.append({"variant": variant, "threshold": thr, "lead_min": lead, "stratum": "wettest_day",
                             "value": day, "n": int(s.sum()), **cat_metrics(y[s], df.loc[s, "yhat"]),
                             "csi_pers": cat_metrics(y[s], df.loc[s, f"yhat_pers_{k}"])["csi"],
                             "base_rate": float(y[s].mean())})
            del df
            rss_check()
    t = pd.DataFrame(rows)
    t.to_csv(ROB / "strata_metrics.csv", index=False)
    sel = t[(t["variant"] == "A_no2023") & (t["threshold"] == 2.5) & (t["stratum"].isin(["all", "hour_block", "wettest10"]))]
    print(sel[["lead_min", "stratum", "value", "csi", "csi_pers", "bss_raw", "bss_cal"]].round(3).head(20).to_string(index=False))
    print(f"rows={len(t)} rss={P.rss_gb():.2f}GB")
    return 0


# ------------------------------------------------------------------------------------------- P4A.7 stale mode
def stale_month(cfg: dict, month: str) -> int:
    """Stale mode: the newest GSMaP window is removed. GSMaP features of (pixel, t0) are replaced by those computed
    at t0 - 30 min (same stack minus the window ending at t0; every older window was already admissible then),
    gsmap_age_min + 30; calendar and static features unchanged. Fresh predictions are recomputed too and must
    equal the stored test predictions (check of the model files)."""
    import lightgbm as lgb
    y_, m_ = (int(x) for x in month.split("-"))
    out = ROB / "stale" / f"month={month}.parquet"
    if out.is_file():
        print(f"skip {month} (exists)"); return 0
    feats = D.feature_list("table_A")
    gs = [f for f in feats if not (f.startswith("cal_") or f in ("lat", "lon"))]
    part = P.GOLD / "table_A" / f"season={y_}" / f"month={m_:02d}" / "part.parquet"
    cols = ["pixel_id", "t0", "lead_min", "target_time"] + D.LABELS + feats
    tb = pd.read_parquet(part, columns=cols)
    lo, hi = D.split_range(cfg, "test")
    tb = tb[(tb["t0"] >= lo) & (tb["t0"] < hi) & (tb["target_time"] <= hi)]
    # GSMaP features do not depend on the lead: take them once per (pixel, t0) from F0 rows
    g = tb.loc[tb["lead_min"] == 0, ["pixel_id", "t0"] + gs].copy()
    pm = pd.Timestamp(year=y_, month=m_, day=1, tz="UTC") - pd.Timedelta("30min")
    ppart = P.GOLD / "table_A" / f"season={pm.year}" / f"month={pm.month:02d}" / "part.parquet"
    if ppart.is_file():                            # last t0 of the previous month (needed for the first t0)
        pp = pd.read_parquet(ppart, columns=["pixel_id", "t0", "lead_min"] + gs,
                             filters=[("t0", "==", pm), ("lead_min", "==", 0)])
        g = pd.concat([pp.drop(columns="lead_min"), g], ignore_index=True)
    g["t0"] = g["t0"] + pd.Timedelta("30min")
    g["gsmap_age_min"] = g["gsmap_age_min"] + 30.0
    g = g.rename(columns={c: c + "__stale" for c in gs})
    tb = tb.merge(g, on=["pixel_id", "t0"], how="left")
    del g
    n_nostale = int(tb["gsmap_age_min__stale"].isna().sum())
    res = []
    for variant, thr in P47_MODELS:
        k = kstr(thr); yc = D.thr_col(thr)
        for lead in cfg["frames"]["lead_min"]:
            md = MODELS / variant / f"lead{lead}_thr{k}"
            bst = lgb.Booster(model_file=str(md / "model.txt"))
            iso = pickle.loads((md / "calibrator.pkl").read_bytes())
            cut = json.loads((md / "meta.json").read_text(encoding="utf-8"))["cut"]["cut"]
            s = tb[tb["lead_min"] == lead]
            Xf = s[feats].to_numpy("float32")
            Xs = s[[c + "__stale" if c in gs else c for c in feats]].to_numpy("float32")
            pf = bst.predict(Xf); ps = bst.predict(Xs)
            del Xf, Xs
            res.append(pd.DataFrame({"variant": variant, "threshold": thr, "lead_min": np.int16(lead),
                                     "pixel_id": s["pixel_id"].to_numpy(), "t0": s["t0"].to_numpy(),
                                     "y": s[yc].to_numpy().astype("int8"),
                                     "p_raw_fresh": pf.astype("float32"), "p_raw_stale": ps.astype("float32"),
                                     "p_fresh": iso.predict(pf).astype("float32"),
                                     "p_stale": iso.predict(ps).astype("float32"), "cut": np.float32(cut),
                                     "stale_missing": s["gsmap_age_min__stale"].isna().to_numpy()}))
            rss_check()
    r = pd.concat(res, ignore_index=True)
    del tb, res
    out.parent.mkdir(parents=True, exist_ok=True)
    P.write_parquet(r, out)
    print(f"{month}: rows={len(r)} no_previous_t0={n_nostale} rss={P.rss_gb():.2f}GB")
    return 0


def stale_summary(cfg: dict) -> int:
    ev = cfg["evaluation"]
    parts = sorted((ROB / "stale").glob("month=*.parquet"))
    rows, maxdiff = [], 0.0
    for variant, thr in P47_MODELS:
        k = kstr(thr)
        for lead in cfg["frames"]["lead_min"]:
            r = pd.concat([pd.read_parquet(p, filters=[("variant", "==", variant), ("threshold", "==", thr),
                                                       ("lead_min", "==", lead)]) for p in parts], ignore_index=True)
            st = pd.read_parquet(P.GOLD / "preds" / f"lgbm_{variant}_lead{lead}_thr{k}_test.parquet",
                                 columns=["pixel_id", "t0", "p_raw", "day"])
            r = r.merge(st, on=["pixel_id", "t0"], how="inner", validate="one_to_one")
            maxdiff = max(maxdiff, float(np.abs(r["p_raw"] - r["p_raw_fresh"]).max()))
            r = r[~r["stale_missing"]]
            b = pd.read_parquet(P.GOLD / "preds" / "baselines_test.parquet", columns=["pixel_id", "t0", "lead_min", f"p_clim_{k}"],
                                filters=[("lead_min", "==", lead)])
            r = r.merge(b.drop(columns="lead_min"), on=["pixel_id", "t0"], how="left")
            y = r["y"].to_numpy().astype(bool); pc = r[f"p_clim_{k}"].to_numpy("float64")
            yf = r["p_fresh"].to_numpy() >= r["cut"].to_numpy(); ys = r["p_stale"].to_numpy() >= r["cut"].to_numpy()
            dsf = M.day_stats(r["day"].to_numpy(), y, yf, r["p_fresh"].to_numpy("float64"), pc)
            dss = M.day_stats(r["day"].to_numpy(), y, ys, r["p_stale"].to_numpy("float64"), pc)
            d = M.paired_bootstrap(dss, dsf, ev["bootstrap_resamples"], cfg["random_seed"], ev["ci_level"])
            rows.append({"variant": variant, "threshold": thr, "lead_min": lead, "n": len(r),
                         "csi_fresh": cat_metrics(y, yf)["csi"], "csi_stale": cat_metrics(y, ys)["csi"],
                         "csi_diff_stale_minus_fresh": d["csi"]["diff"], "csi_lo": d["csi"]["lo"], "csi_hi": d["csi"]["hi"],
                         "brier_cal_fresh": M.brier(y, r["p_fresh"]), "brier_cal_stale": M.brier(y, r["p_stale"]),
                         "brier_diff": d["brier"]["diff"], "brier_lo": d["brier"]["lo"], "brier_hi": d["brier"]["hi"],
                         "brier_raw_fresh": M.brier(y, r["p_raw_fresh"]), "brier_raw_stale": M.brier(y, r["p_raw_stale"]),
                         "bss_raw_fresh": M.bss(y, r["p_raw_fresh"], pc), "bss_raw_stale": M.bss(y, r["p_raw_stale"], pc),
                         "bss_cal_fresh": M.bss(y, r["p_fresh"], pc), "bss_cal_stale": M.bss(y, r["p_stale"], pc)})
            del r, st, b
            rss_check()
    t = pd.DataFrame(rows)
    t.to_csv(ROB / "stale_mode.csv", index=False)
    print(t[["variant", "threshold", "lead_min", "csi_fresh", "csi_stale", "csi_lo", "csi_hi",
             "bss_raw_fresh", "bss_raw_stale"]].round(3).to_string(index=False))
    print(f"months={len(parts)} max|p_raw stored - recomputed|={maxdiff:.2e} rss={P.rss_gb():.2f}GB")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prelim-B", "strata", "stale", "stale-summary"])
    ap.add_argument("--month")
    a = ap.parse_args(argv)
    P.require_guard()
    cfg = P.load_cfg()
    if a.cmd == "prelim-B":
        return prelim_b(cfg)
    if a.cmd == "strata":
        return strata(cfg)
    if a.cmd == "stale":
        return stale_month(cfg, a.month)
    return stale_summary(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
