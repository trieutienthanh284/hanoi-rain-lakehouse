"""P4A.6 — equal-period comparison A-2024 (= A_no2023, same tune 2024-05..08) vs B-2024 (with NWP) on test 2025,
plus the baseline 'raw NWP >= threshold' (nearest point, hour containing the frame) per model."""
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


def main() -> int:
    P.require_guard()
    cfg = P.load_cfg(); ev = cfg["evaluation"]
    out = ROOT / "reports" / "p4a" / "compare_A_B"; out.mkdir(parents=True, exist_ok=True)
    rows, diffs, val = [], [], []
    for thr in cfg["targets"]["thresholds_mm_h"]:
        k = kstr(thr); yc = D.thr_col(thr)
        for lead in cfg["frames"]["lead_min"]:
            pa = pd.read_parquet(P.GOLD / "preds" / f"lgbm_A_no2023_lead{lead}_thr{k}_test.parquet")
            pb = pd.read_parquet(P.GOLD / "preds" / f"lgbm_B_lead{lead}_thr{k}_test.parquet")
            bl = pd.read_parquet(P.GOLD / "preds" / "baselines_test.parquet")
            bl = bl[bl["lead_min"] == lead][["pixel_id", "t0", f"p_clim_{k}", f"yhat_pers_{k}"]]
            nwp = D.load_split(cfg, "test", ["nwp_gfs_near_h0", "nwp_ecmwf_near_h0"], table="table_B", lead=lead)
            m = pa.merge(pb[["pixel_id", "t0", "p", "yhat", "p_raw"]], on=["pixel_id", "t0"], suffixes=("_A", "_B"),
                         validate="one_to_one").merge(bl, on=["pixel_id", "t0"], validate="one_to_one") \
                  .merge(nwp[["pixel_id", "t0", "nwp_gfs_near_h0", "nwp_ecmwf_near_h0"]], on=["pixel_id", "t0"],
                         validate="one_to_one")
            assert len(m) == len(pa)
            y = m[yc].to_numpy().astype(bool); pc = m[f"p_clim_{k}"].to_numpy("float64"); day = m["day"].to_numpy()
            fc = {"A_2024": (m["p_A"].to_numpy("float64"), m["yhat_A"].to_numpy()),
                  "B_2024": (m["p_B"].to_numpy("float64"), m["yhat_B"].to_numpy()),
                  "pers": (m[f"yhat_pers_{k}"].to_numpy("float64"), m[f"yhat_pers_{k}"].to_numpy())}
            for mod in ("gfs", "ecmwf"):
                x = np.nan_to_num(m[f"nwp_{mod}_near_h0"].to_numpy("float64"), nan=-1) >= thr
                fc[f"nwp_raw_{mod}"] = (x.astype(float), x)
            ds = {}
            for name, (p, yh) in fc.items():
                s = M.summary(y, p, yh, pc)
                ds[name] = M.day_stats(day, y, yh, p, pc)
                ci = M.bootstrap_ci(ds[name], ev["bootstrap_resamples"], cfg["random_seed"], ev["ci_level"])
                rows.append({"lead_min": lead, "threshold": thr, "model": name, "csi": s["csi"], "csi_lo": ci["csi"][0],
                             "csi_hi": ci["csi"][1], "pod": s["pod"], "far": s["far"], "bias": s["bias"],
                             "brier": s["brier"], "bss_clim": s["bss_clim"], "auc_pr": s["auc_pr"]})
            for a_, b_ in (("B_2024", "A_2024"), ("B_2024", "pers"), ("B_2024", "nwp_raw_ecmwf"), ("B_2024", "nwp_raw_gfs")):
                d = M.paired_bootstrap(ds[a_], ds[b_], ev["bootstrap_resamples"], cfg["random_seed"], ev["ci_level"])
                diffs.append({"lead_min": lead, "threshold": thr, "comparison": f"{a_} - {b_}",
                              **{f"{mm}_{q}": d[mm][q] for mm in ("csi", "brier") for q in ("diff", "lo", "hi")}})
            for v in ("A_no2023", "B"):
                meta = json.loads((MODELS / v / f"lead{lead}_thr{k}" / "meta.json").read_text(encoding="utf-8"))
                pv = pd.read_parquet(P.GOLD / "preds" / f"lgbm_{v}_lead{lead}_thr{k}_validation.parquet")
                val.append({"variant": v, "lead_min": lead, "threshold": thr, "val_logloss": meta["val_logloss_best"],
                            "val_auc_pr_raw": M.auc_pr(pv[yc].to_numpy(), pv["p_raw"].to_numpy("float64"))})
    r, d, v = pd.DataFrame(rows), pd.DataFrame(diffs), pd.DataFrame(val)
    r.to_csv(out / "test_metrics.csv", index=False); d.to_csv(out / "test_diffs.csv", index=False); v.to_csv(out / "validation.csv", index=False)
    print(r.pivot_table(index=["threshold", "lead_min"], columns="model", values="csi").round(3).to_string())
    x = d[d["comparison"] == "B_2024 - A_2024"]
    print(x[["threshold", "lead_min", "csi_diff", "csi_lo", "csi_hi", "brier_diff", "brier_lo", "brier_hi"]].round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
