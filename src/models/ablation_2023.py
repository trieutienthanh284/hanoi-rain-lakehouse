"""P4A.5b — ablation with / without season 2023 in the tune period (thresholds 2.5, 7.6; all frames).
Variant choice by VALIDATION only (rule D67); test differences are reported as an outcome."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.gold import p4a_common as P  # noqa: E402
from src.models import data as D  # noqa: E402
from src.models import metrics as M  # noqa: E402
from src.models.train_lgbm import MODELS, kstr  # noqa: E402

VARS = ("A", "A_no2023")


def ece(y, p, bins=10):
    r = M.reliability(y, p, bins); r = r[r["n"] > 0]
    return float((r["n"] * (r["mean_p"] - r["obs_freq"]).abs()).sum() / r["n"].sum())


def main() -> int:
    P.require_guard()
    cfg = P.load_cfg(); ev = cfg["evaluation"]
    out = ROOT / "reports" / "p4a" / "ablation_2023"; out.mkdir(parents=True, exist_ok=True)
    val_rows, test_rows, diff_rows = [], [], []
    for thr in (2.5, 7.6):
        k = kstr(thr); yc = D.thr_col(thr)
        for lead in cfg["frames"]["lead_min"]:
            preds = {}
            for v in VARS:
                meta = json.loads((MODELS / v / f"lead{lead}_thr{k}" / "meta.json").read_text(encoding="utf-8"))
                pv = pd.read_parquet(P.GOLD / "preds" / f"lgbm_{v}_lead{lead}_thr{k}_validation.parquet")
                y = pv[yc].to_numpy(); pr = pv["p_raw"].to_numpy("float64")
                val_rows.append({"variant": v, "lead_min": lead, "threshold": thr, "val_logloss": meta["val_logloss_best"],
                                 "val_auc_pr_raw": M.auc_pr(y, pr), "val_brier_raw": M.brier(y, pr),
                                 "val_csi_maxcut_insample": M.choose_cut(y, pr, "max_csi")["csi"],
                                 "best_iteration": meta["best_iteration"], "tune_rows": meta["rows"]["tune"],
                                 "season_2023_share_of_tune": meta["season_2023_share_of_training"]})
                pt = pd.read_parquet(P.GOLD / "preds" / f"lgbm_{v}_lead{lead}_thr{k}_test.parquet")
                b = pd.read_parquet(P.GOLD / "preds" / "baselines_test.parquet")
                b = b[b["lead_min"] == lead][["pixel_id", "t0", f"p_clim_{k}"]]
                pt = pt.merge(b, on=["pixel_id", "t0"], validate="one_to_one")
                preds[v] = pt
                y = pt[yc].to_numpy(); p = pt["p"].to_numpy("float64"); yh = pt["yhat"].to_numpy()
                s = M.summary(y, p, yh, pt[f"p_clim_{k}"].to_numpy("float64"))
                test_rows.append({"variant": v, "lead_min": lead, "threshold": thr, "csi": s["csi"], "bias": s["bias"],
                                  "brier": s["brier"], "bss_clim": s["bss_clim"], "auc_pr": s["auc_pr"],
                                  "ece_calibrated": ece(y, p), "ece_raw": ece(y, pt["p_raw"].to_numpy("float64")),
                                  "bss_clim_raw": M.bss(y, pt["p_raw"].to_numpy("float64"), pt[f"p_clim_{k}"].to_numpy("float64"))})
            a, b2 = preds["A_no2023"], preds["A"]
            assert a[["pixel_id", "t0"]].equals(b2[["pixel_id", "t0"]])
            y = a[yc].to_numpy(); pc = a[f"p_clim_{k}"].to_numpy("float64")
            dsa = M.day_stats(a["day"].to_numpy(), y, a["yhat"].to_numpy(), a["p"].to_numpy("float64"), pc)
            dsb = M.day_stats(b2["day"].to_numpy(), y, b2["yhat"].to_numpy(), b2["p"].to_numpy("float64"), pc)
            d = M.paired_bootstrap(dsa, dsb, ev["bootstrap_resamples"], cfg["random_seed"], ev["ci_level"])
            diff_rows.append({"lead_min": lead, "threshold": thr, "comparison": "A_no2023 - A",
                              **{f"{m}_{q}": d[m][q] for m in ("csi", "brier", "bss_clim") for q in ("diff", "lo", "hi")}})
    vr, tr, dr = pd.DataFrame(val_rows), pd.DataFrame(test_rows), pd.DataFrame(diff_rows)
    vr.to_csv(out / "validation.csv", index=False); tr.to_csv(out / "test.csv", index=False); dr.to_csv(out / "test_diffs.csv", index=False)
    agg = vr.groupby("variant")[["val_logloss", "val_auc_pr_raw"]].mean()
    ll = agg["val_logloss"]
    rel_gap = abs(ll["A"] - ll["A_no2023"]) / ll.min()
    chosen = (agg["val_auc_pr_raw"].idxmax() if rel_gap < 0.005 else ll.idxmin())
    dec = {"rule": "D67: lower mean validation log-loss (pre-refit, raw); tie (<0.5 %) -> higher mean AUC-PR",
           "mean_val_logloss": ll.to_dict(), "mean_val_auc_pr": agg["val_auc_pr_raw"].to_dict(),
           "relative_gap": rel_gap, "chosen_variant": chosen}
    P.write_json(out / "decision.json", dec)
    print(vr.pivot_table(index=["threshold", "lead_min"], columns="variant", values=["val_logloss", "val_auc_pr_raw"]).round(4).to_string())
    print("CHOSEN:", chosen, {k: round(v, 5) for k, v in ll.items()})
    print(dr[["threshold", "lead_min", "csi_diff", "csi_lo", "csi_hi", "brier_diff", "brier_lo", "brier_hi"]].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
