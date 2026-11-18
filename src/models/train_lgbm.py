"""P4A.5 / P4A.5b — LightGBM per (frame, threshold), pilot v0.

  python src/models/train_lgbm.py --variant A --leads 0            # one invocation per lead (< 20 min)
  python src/models/train_lgbm.py --variant A_no2023 --thresholds 2.5 7.6

Per model: train on tune with early stopping on validation log-loss -> isotonic calibrator fitted on the
validation predictions of this pre-refit model -> probability cut chosen on validation (cut rule) ->
refit on tune + validation with the selected rounds. Test is only predicted, never used for a choice.
Outputs: models/pilot_v0/<variant>/lead<L>_thr<k>/{model.txt, calibrator.pkl, meta.json, importance.csv},
data/gold/met_training/pilot_v0/preds/lgbm_<variant>_lead<L>_thr<k>_{validation,test}.parquet,
MLflow run in mlruns/ (JSON record in reports/p4a/runs/ as well). Resumable: a model with meta.json
of the same config hash is skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.gold import p4a_common as P  # noqa: E402
from src.models import data as D  # noqa: E402
from src.models import metrics as M  # noqa: E402

MODELS = ROOT / "models" / "pilot_v0"
RUNS = ROOT / "reports" / "p4a" / "runs"
CODE = [ROOT / "src/models/train_lgbm.py", ROOT / "src/models/data.py"]
VARIANTS = {"A": "tune", "A_no2023": "ablation_tune_2024_only", "B": "ablation_tune_2024_only"}
TABLES = {"A": "table_A", "A_no2023": "table_A", "B": "table_B"}


def kstr(t: float) -> str:
    return f"{t:g}".replace(".", "p")


def lgb_params(cfg: dict) -> dict:
    g = dict(cfg["lightgbm"])
    for k in ("num_boost_round", "early_stopping_rounds"):
        g.pop(k)
    g["seed"] = cfg["random_seed"]; g["deterministic"] = True; g["force_row_wise"] = True
    return g


def mlflow_log(name: str, params: dict, metrics: dict, artifacts: list[Path]) -> str:
    try:
        import mlflow
        (ROOT / "mlruns").mkdir(exist_ok=True)
        mlflow.set_tracking_uri("sqlite:///" + (ROOT / "mlruns" / "mlflow.db").as_posix())
        if mlflow.get_experiment_by_name("pilot_v0") is None:
            mlflow.create_experiment("pilot_v0", artifact_location=(ROOT / "mlruns" / "artifacts").as_uri())
        mlflow.set_experiment("pilot_v0")
        with mlflow.start_run(run_name=name):
            mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v == v})
            for a in artifacts:
                mlflow.log_artifact(str(a))
        return "mlflow"
    except Exception as exc:                                    # fallback: JSON record only
        return f"json_only ({type(exc).__name__}: {str(exc)[:120]})"


def train_one(cfg, variant, lead, thr, tune, val, test, feats, ch, lg) -> dict:
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression
    k = kstr(thr); yc = D.thr_col(thr)
    out = MODELS / variant / f"lead{lead}_thr{k}"
    t1 = time.time()
    params = lgb_params(cfg)
    dtr = lgb.Dataset(tune[feats].to_numpy("float32"), tune[yc].to_numpy(), feature_name=feats, free_raw_data=True)
    dva = lgb.Dataset(val[feats].to_numpy("float32"), val[yc].to_numpy(), reference=dtr)
    ev = {}
    m0 = lgb.train(params, dtr, num_boost_round=cfg["lightgbm"]["num_boost_round"], valid_sets=[dva], valid_names=["val"],
                   callbacks=[lgb.early_stopping(cfg["lightgbm"]["early_stopping_rounds"], verbose=False),
                              lgb.record_evaluation(ev)])
    best = int(m0.best_iteration or m0.current_iteration())
    p_val_raw = m0.predict(val[feats].to_numpy("float32"), num_iteration=best)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p_val_raw, val[yc].to_numpy())
    p_val = iso.predict(p_val_raw)
    rule = cfg["evaluation"]["cut_rule"][f"{thr:g}"]
    cut = M.choose_cut(val[yc].to_numpy(), p_val, rule)
    cut_maxcsi = M.choose_cut(val[yc].to_numpy(), p_val, "max_csi")
    # refit on tune + validation with the selected number of rounds
    both = pd.concat([tune[feats + [yc]], val[feats + [yc]]], ignore_index=True)
    m1 = lgb.train(params, lgb.Dataset(both[feats].to_numpy("float32"), both[yc].to_numpy(), feature_name=feats),
                   num_boost_round=best)
    del both
    p_test_raw = m1.predict(test[feats].to_numpy("float32"))
    p_test = iso.predict(p_test_raw)
    out.mkdir(parents=True, exist_ok=True)
    m1.save_model(str(out / "model.txt"))
    m0.save_model(str(out / "model_prerefit.txt"), num_iteration=best)
    P.write_atomic_bytes(out / "calibrator.pkl", pickle.dumps(iso))
    imp = pd.DataFrame({"feature": feats, "gain": m1.feature_importance("gain"),
                        "split": m1.feature_importance("split")}).sort_values("gain", ascending=False)
    imp.to_csv(out / "importance.csv", index=False)
    key = ["pixel_id", "t0", "lead_min", "day", "season"]
    P.write_parquet(val[key + [yc]].assign(p_raw=p_val_raw.astype("float32"), p=p_val.astype("float32"),
                                           yhat=(p_val >= cut["cut"]).astype("int8")),
                    P.GOLD / "preds" / f"lgbm_{variant}_lead{lead}_thr{k}_validation.parquet")
    P.write_parquet(test[key + [yc]].assign(p_raw=p_test_raw.astype("float32"), p=p_test.astype("float32"),
                                            yhat=(p_test >= cut["cut"]).astype("int8"),
                                            yhat_maxcsi=(p_test >= cut_maxcsi["cut"]).astype("int8")),
                    P.GOLD / "preds" / f"lgbm_{variant}_lead{lead}_thr{k}_test.parquet")
    tr = cfg["splits"][VARIANTS[variant]]
    meta = {"model": f"pilot_v0/{variant}/lead{lead}_thr{k}", "status": "PILOT — not a product model",
            "variant": variant, "table": TABLES[variant], "lead_min": lead, "threshold_mm_h": thr, "label": "IMERG Final V07 (satellite estimate)",
            "config_hash": ch, "code_hash": P.code_hash(*CODE), "features": feats, "n_features": len(feats),
            "training_period": {"tune": tr, "validation": cfg["splits"]["validation"],
                                "refit": [tr[0], cfg["splits"]["validation"][1]]},
            "rows": {"tune": len(tune), "validation": len(val), "test": len(test)},
            "season_2023_share_of_training": float((tune["season"] == 2023).mean()),
            "best_iteration": best, "val_logloss_best": float(min(ev["val"]["binary_logloss"])),
            "cut": cut, "cut_max_csi": cut_maxcsi, "calibration": "isotonic on validation (pre-refit model)",
            "lgb_params": params, "seconds": round(time.time() - t1, 1)}
    P.write_json(out / "meta.json", meta)
    run_metrics = {"best_iteration": best, "val_logloss": meta["val_logloss_best"], "val_cut": cut["cut"],
                   "val_csi_at_cut": cut["csi"]}
    # MLflow is logged afterwards in a separate process (src/models/log_mlflow.py): importing MLflow in the
    # training process crashed LightGBM once with a native access violation (D65).
    meta["tracking"] = "json; mlflow via src/models/log_mlflow.py"
    P.write_json(out / "meta.json", meta)
    P.write_json(RUNS / f"{variant}_lead{lead}_thr{k}.json", {**meta, "metrics": run_metrics})
    lg.info("%s lead%s thr%s best=%d cut=%.4f valCSI=%.3f %.0fs tracking=%s", variant, lead, k, best, cut["cut"],
            cut["csi"], meta["seconds"], meta["tracking"])
    return meta


def wait_for_memory(min_gb: float, max_wait_s: int) -> None:
    import psutil
    t = time.time()
    while psutil.virtual_memory().available / 2**30 < min_gb:
        if time.time() - t > max_wait_s:            # D69: never compete with the user's backfill for RAM
            print(f"STOP: system free RAM < {min_gb} GB for {max_wait_s} s; rerun later (resumable)")
            raise SystemExit(3)
        time.sleep(10)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="A", choices=list(VARIANTS))
    ap.add_argument("--leads", nargs="*", type=int)
    ap.add_argument("--thresholds", nargs="*", type=float)
    ap.add_argument("--max-minutes", type=float, default=18.0)
    a = ap.parse_args(argv)
    P.require_guard()
    cfg = P.load_cfg()
    lg = P.get_logger("p4a5_train")
    feats = D.feature_list(TABLES[a.variant])
    ch = P.cfg_hash({"lightgbm": cfg["lightgbm"], "splits": cfg["splits"], "evaluation": cfg["evaluation"],
                     "seed": cfg["random_seed"], "variant": a.variant, "features": feats, "code": P.code_hash(*CODE)})
    leads = a.leads if a.leads else cfg["frames"]["lead_min"]
    thrs = a.thresholds if a.thresholds else cfg["targets"]["thresholds_mm_h"]
    t0 = time.time()
    done = skipped = 0
    for lead in leads:
        todo = [t for t in thrs if not ((MODELS / a.variant / f"lead{lead}_thr{kstr(t)}" / "meta.json").is_file() and
                json.loads((MODELS / a.variant / f"lead{lead}_thr{kstr(t)}" / "meta.json").read_text())["config_hash"] == ch)]
        skipped += len(thrs) - len(todo)
        if not todo:
            continue
        tb = TABLES[a.variant]
        tune = D.load_split(cfg, VARIANTS[a.variant], feats, table=tb, lead=lead)
        val = D.load_split(cfg, "validation", feats, table=tb, lead=lead)
        test = D.load_split(cfg, "test", feats, table=tb, lead=lead)
        for thr in todo:
            if (time.time() - t0) / 60 > a.max_minutes:
                print(f"time budget reached; rerun to continue (done={done})"); return 0
            m = None
            for attempt in (1, 2):                 # D69: system-wide memory pressure -> one retry after 60 s
                wait_for_memory(1.0, 300)
                try:
                    m = train_one(cfg, a.variant, lead, thr, tune, val, test, feats, ch, lg)
                    break
                except Exception as exc:
                    if "bad allocation" not in str(exc) or attempt == 2:
                        raise
                    lg.warning("bad allocation (%s lead%s thr%s); retry in 60 s", a.variant, lead, thr)
                    import gc; gc.collect(); time.sleep(60)
            done += 1
            print(f"{a.variant} lead{lead} thr{thr:g}: best_iter={m['best_iteration']} cut={m['cut']['cut']:.3f} "
                  f"valCSI={m['cut']['csi']:.3f} {m['seconds']:.0f}s rss={P.rss_gb():.2f}GB [{m['tracking'][:12]}]")
            P.check_rss(cfg["resources"]["max_rss_gb"])
        del tune, val, test
    print(f"done={done} skipped={skipped} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
