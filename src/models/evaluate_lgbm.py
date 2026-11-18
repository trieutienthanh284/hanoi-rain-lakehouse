"""P4A.5 — evaluate LightGBM pilot models on test 2025 against every baseline (no choice is made here).

  python src/models/evaluate_lgbm.py --variant A
Outputs in reports/p4a/lgbm_<variant>/: test_metrics.csv, test_by_month.csv, diffs.csv, verdict.csv,
reliability.csv; reports/p4a/importance_<variant>_thr<k>.png. Verdict per (frame, threshold):
BEATS_ALL = the 90 % interval of the CSI difference (> 0) and of the Brier difference (< 0) exclude zero
against every baseline; LOSES = significantly worse (CSI interval < 0 or Brier interval > 0) than at least
one baseline; TIES otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.gold import p4a_common as P  # noqa: E402
from src.models import data as D  # noqa: E402
from src.models import metrics as M  # noqa: E402
from src.models.train_lgbm import MODELS, kstr  # noqa: E402

BASELINES = ("clim", "pers", "calpers", "logit")


def baseline_p(b: pd.DataFrame, name: str, k: str) -> np.ndarray:
    if name == "clim":
        return b[f"p_clim_{k}"].to_numpy("float64")
    if name == "logit":
        return b[f"p_logit_{k}"].to_numpy("float64")
    return b[f"yhat_{name}_{k}"].to_numpy("float64")


def load_pair(variant: str, lead: int, thr: float, split: str = "test") -> pd.DataFrame:
    k = kstr(thr)
    m = pd.read_parquet(P.GOLD / "preds" / f"lgbm_{variant}_lead{lead}_thr{k}_{split}.parquet")
    b = pd.read_parquet(P.GOLD / "preds" / f"baselines_{split}.parquet")
    b = b[b["lead_min"] == lead].drop(columns=["day", "season", *D.LABELS], errors="ignore")
    out = m.merge(b, on=["pixel_id", "t0", "lead_min"], how="inner", validate="one_to_one")
    assert len(out) == len(m), "baseline rows missing"
    return out


def importance_figure(variant: str, thr: float, leads) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k = kstr(thr)
    fig, axes = plt.subplots(1, len(leads), figsize=(5 * len(leads), 6), constrained_layout=True)
    for ax, lead in zip(np.atleast_1d(axes), leads):
        imp = pd.read_csv(MODELS / variant / f"lead{lead}_thr{k}" / "importance.csv").head(20)[::-1]
        ax.barh(imp["feature"], imp["gain"] / imp["gain"].sum(), color="#4C72B0")
        ax.set_title(f"F{lead} — {thr:g} mm/h"); ax.set_xlabel("share of gain (top 20)")
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle(f"PILOT v0 — LightGBM set {variant}: gain importance (top 20)")
    p = ROOT / "reports" / "p4a" / f"importance_{variant}_thr{k}.png"
    fig.savefig(p, dpi=110); plt.close(fig)
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="A")
    ap.add_argument("--thresholds", nargs="*", type=float)
    a = ap.parse_args(argv)
    P.require_guard()
    cfg = P.load_cfg()
    ev = cfg["evaluation"]
    out = ROOT / "reports" / "p4a" / f"lgbm_{a.variant}"
    out.mkdir(parents=True, exist_ok=True)
    thrs = a.thresholds or cfg["targets"]["thresholds_mm_h"]
    leads = cfg["frames"]["lead_min"]
    rows, months, diffs, verdicts, rel = [], [], [], [], []
    for thr in thrs:
        k = kstr(thr); yc = D.thr_col(thr)
        for lead in leads:
            meta = json.loads((MODELS / a.variant / f"lead{lead}_thr{k}" / "meta.json").read_text(encoding="utf-8"))
            df = load_pair(a.variant, lead, thr)
            y = df[yc].to_numpy().astype(bool)
            pc = df[f"p_clim_{k}"].to_numpy("float64")
            p = df["p"].to_numpy("float64"); yh = df["yhat"].to_numpy().astype(bool)
            s = M.summary(y, p, yh, pc)
            ds_m = M.day_stats(df["day"].to_numpy(), y, yh, p, pc)
            ci = M.bootstrap_ci(ds_m, ev["bootstrap_resamples"], cfg["random_seed"], ev["ci_level"])
            r = {"variant": a.variant, "lead_min": lead, "threshold": thr, "model": "lgbm", "cut": meta["cut"]["cut"],
                 "cut_rule": meta["cut"]["rule"], **s, "csi_lo": ci["csi"][0], "csi_hi": ci["csi"][1],
                 "bss_lo": ci["bss_clim"][0], "bss_hi": ci["bss_clim"][1],
                 "brier_raw_uncalibrated": M.brier(y, df["p_raw"].to_numpy("float64"))}
            if "yhat_maxcsi" in df:
                c2 = M.scores(**M.contingency(y, df["yhat_maxcsi"].to_numpy()))
                r |= {"cut_max_csi": meta["cut_max_csi"]["cut"], "csi_at_max_csi_cut": float(c2["csi"]),
                      "bias_at_max_csi_cut": float(c2["bias"])}
            rows.append(r)
            mon = df["t0"].dt.month.to_numpy()
            for mm in np.unique(mon):
                sel = mon == mm
                months.append({"variant": a.variant, "lead_min": lead, "threshold": thr, "month": int(mm),
                               **M.summary(y[sel], p[sel], yh[sel], pc[sel])})
            wins, worse = [], []
            for b in BASELINES:
                pb = baseline_p(df, b, k)
                yb = df[f"yhat_{b}_{k}"].to_numpy().astype(bool)
                ds_b = M.day_stats(df["day"].to_numpy(), y, yb, pb, pc)
                d = M.paired_bootstrap(ds_m, ds_b, ev["bootstrap_resamples"], cfg["random_seed"], ev["ci_level"])
                win = d["csi"]["lo"] > 0 and d["brier"]["hi"] < 0
                bad = d["csi"]["hi"] < 0 or d["brier"]["lo"] > 0
                wins.append(win); worse.append(bad)
                diffs.append({"variant": a.variant, "lead_min": lead, "threshold": thr, "baseline": b,
                              "csi_model": s["csi"], "csi_diff": d["csi"]["diff"], "csi_lo": d["csi"]["lo"],
                              "csi_hi": d["csi"]["hi"], "brier_diff": d["brier"]["diff"], "brier_lo": d["brier"]["lo"],
                              "brier_hi": d["brier"]["hi"], "win": win, "significantly_worse": bad})
            verdict = "BEATS_ALL" if all(wins) else ("LOSES" if any(worse) else "TIES")
            verdicts.append({"variant": a.variant, "lead_min": lead, "threshold": thr, "verdict": verdict,
                             "wins_vs": ",".join(b for b, w in zip(BASELINES, wins) if w),
                             "worse_than": ",".join(b for b, w in zip(BASELINES, worse) if w)})
            for nm, pp in (("test_raw", df["p_raw"].to_numpy("float64")), ("test_calibrated", p)):
                rel.append(M.reliability(y, pp, ev["reliability_bins"]).assign(
                    variant=a.variant, lead_min=lead, threshold=thr, which=nm))
            v = pd.read_parquet(P.GOLD / "preds" / f"lgbm_{a.variant}_lead{lead}_thr{k}_validation.parquet")
            for nm, col in (("val_raw", "p_raw"), ("val_calibrated_insample", "p")):
                rel.append(M.reliability(v[yc].to_numpy(), v[col].to_numpy("float64"), ev["reliability_bins"]).assign(
                    variant=a.variant, lead_min=lead, threshold=thr, which=nm))
        importance_figure(a.variant, thr, leads)
    pd.DataFrame(rows).to_csv(out / "test_metrics.csv", index=False)
    pd.DataFrame(months).to_csv(out / "test_by_month.csv", index=False)
    pd.DataFrame(diffs).to_csv(out / "diffs.csv", index=False)
    pd.DataFrame(verdicts).to_csv(out / "verdict.csv", index=False)
    pd.concat(rel, ignore_index=True).to_csv(out / "reliability.csv", index=False)
    v = pd.DataFrame(verdicts)
    t = pd.DataFrame(rows)
    print(t[["threshold", "lead_min", "csi", "csi_lo", "csi_hi", "bss_clim", "auc_pr"]].round(3).to_string(index=False))
    print(v[["threshold", "lead_min", "verdict", "wins_vs", "worse_than"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
