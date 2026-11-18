"""Log pilot run records (reports/p4a/runs/*.json) to MLflow (local SQLite in mlruns/), in its own process.
Idempotent: a run whose name + config hash already exists is skipped."""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "reports" / "p4a" / "runs"


def main() -> int:
    try:
        import mlflow
    except ImportError:
        print("mlflow not installed: JSON records only (NEEDS_REVIEW)"); return 0
    (ROOT / "mlruns").mkdir(exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///" + (ROOT / "mlruns" / "mlflow.db").as_posix())
    if mlflow.get_experiment_by_name("pilot_v0") is None:
        mlflow.create_experiment("pilot_v0", artifact_location=(ROOT / "mlruns" / "artifacts").as_uri())
    exp = mlflow.set_experiment("pilot_v0")
    have = {(r["tags.mlflow.runName"], r.get("params.config_hash")) for _, r in
            mlflow.search_runs([exp.experiment_id]).iterrows()} if True else set()
    n = 0
    for f in sorted(RUNS.glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        name = f.stem
        if (name, r["config_hash"]) in have:
            continue
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"variant": r["variant"], "lead_min": r["lead_min"], "threshold_mm_h": r["threshold_mm_h"],
                               "config_hash": r["config_hash"], "code_hash": r["code_hash"],
                               "training_period": json.dumps(r["training_period"]), "n_features": r["n_features"],
                               **{k: v for k, v in r["lgb_params"].items()}})
            mlflow.log_metrics({k: float(v) for k, v in r["metrics"].items()})
            mlflow.set_tag("status", "PILOT")
            d = ROOT / "models" / "pilot_v0" / r["variant"] / f"lead{r['lead_min']}_thr{name.split('_thr')[-1]}"
            for a in ("meta.json", "importance.csv"):
                if (d / a).is_file():
                    mlflow.log_artifact(str(d / a))
        n += 1
    print(f"mlflow: logged {n} new runs; total records {len(list(RUNS.glob('*.json')))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
