"""P4C.2 — cloud training entry point (GitHub Actions runner or local), PILOT v0.

  python -m src.models.cloud_train --job p4a6 [--part 1/2] [--store r2|local:<dir>] [--plan]
  python -m src.models.cloud_train --job p4a7 --part 2/3
  python -m src.models.cloud_train --job p4b [--final]          # reserved, see D77

Jobs (definitions exactly as in TASK_P4A_PILOT_MODEL.md, code imported, never copied):
  p4a6  remaining set-B models (all 9 are (re)trained: the 2 local ones serve as a determinism check),
        the A_no2023 0.5 mm/h models (3), then the final unit: compare_A_B (A-2024 vs B-2024, raw NWP
        baseline) + evaluate_lgbm for B and for A_no2023 0.5.
  p4a7  leave-one-season-out for set A: 3 held-out seasons x 3 frames x 3 thresholds = 27 fits, then the final
        unit: per-season out-of-sample metrics and spread (fold definitions: D76).

Remote layout (R2, everything under lake/v1/runs/<job>/; nothing is ever deleted):
  lake/v1/runs/<job>/<run_id>/files/<repo-relative path>     outputs (models, preds, reports, run records)
  lake/v1/runs/<job>/<run_id>/markers/<unit>.done.json       one per finished unit (commit marker)
  lake/v1/runs/<job>/<run_id>/summary.json, done.json        written by the final unit
  lake/v1/runs/<job>/latest.json                             pointer to the newest run id (only overwritable key)
Inputs are read from lake/v1/<repo-relative path> (synced by src/lake/r2_sync.py, P4C.1).
run_id = hash of the job, configs/model_pilot.yaml and the training code: re-running the same code resumes the
same run (finished units are skipped); changed code starts a new run. A local mirror of every uploaded file is
kept in cloud_out/<job>/<run_id>/ (GitHub Actions artifact, 7 days).
Secrets: only from the environment; all output goes through src.realtime.common's redacting streams.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from src.realtime import common as C  # noqa: E402

LAKE = "lake/v1/"
RUNS = LAKE + "runs/"
JOBS = ("p4a6", "p4a7", "p4b")
GOLD_REL = "data/gold/met_training/pilot_v0/"
CODE_FILES = ["src/models/cloud_train.py", "src/models/train_lgbm.py", "src/models/data.py", "src/models/metrics.py",
              "src/models/compare_A_B.py", "src/models/evaluate_lgbm.py", "src/gold/p4a_common.py",
              "configs/model_pilot.yaml"]
# leave-one-season-out folds (D76): inner validation = last two months of the most recent training season
LOSO = {
    2023: {"tune": [("2024-05-01", "2024-10-31"), ("2025-05-01", "2025-07-31")], "validation": ("2025-08-01", "2025-09-30"),
           "heldout": ("2023-05-01", "2023-10-31")},
    2024: {"tune": [("2023-05-01", "2023-10-31"), ("2025-05-01", "2025-07-31")], "validation": ("2025-08-01", "2025-09-30"),
           "heldout": ("2024-05-01", "2024-10-31")},
    2025: {"tune": [("2023-05-01", "2024-08-31")], "validation": ("2024-09-01", "2024-10-31"),
           "heldout": ("2025-05-01", "2025-09-30")},
}
# inputs per job: repo-relative prefixes downloaded from lake/v1/
INPUTS = {
    "p4a6": [GOLD_REL + "table_B/", GOLD_REL + "table_A/schema.json", GOLD_REL + "table_A/season=2024/",
             GOLD_REL + "table_A/season=2025/", GOLD_REL + "preds/baselines_", GOLD_REL + "preds/lgbm_A_no2023_",
             "models/pilot_v0/A_no2023/"],
    "p4a7": [GOLD_REL + "table_A/", GOLD_REL + "preds/lgbm_A_lead"],
    "p4b": [],
}


# ------------------------------------------------------------------------------------------- units
@dataclass(frozen=True)
class Unit:
    id: str
    kind: str                     # train | loso | final
    params: dict = field(default_factory=dict, hash=False, compare=False)


def kstr(t: float) -> str:
    return f"{t:g}".replace(".", "p")


def job_units(job: str, cfg: dict) -> list[Unit]:
    leads = list(cfg["frames"]["lead_min"])
    thrs = [float(t) for t in cfg["targets"]["thresholds_mm_h"]]
    u: list[Unit] = []
    if job == "p4a6":
        for lead in leads:                                   # ordered by lead: one data load per (variant, lead)
            for t in thrs:
                u.append(Unit(f"train-B-lead{lead}-thr{kstr(t)}", "train", {"variant": "B", "lead": lead, "thr": t}))
            u.append(Unit(f"train-A_no2023-lead{lead}-thr0p5", "train", {"variant": "A_no2023", "lead": lead, "thr": 0.5}))
    elif job == "p4a7":
        for s in sorted(LOSO):
            for lead in leads:
                for t in thrs:
                    u.append(Unit(f"loso-{s}-lead{lead}-thr{kstr(t)}", "loso", {"season": s, "lead": lead, "thr": t}))
    elif job == "p4b":
        return []
    u.append(Unit("final", "final", {}))
    return u


def parse_part(part: str | None) -> tuple[int, int]:
    if not part:
        return 1, 1
    i, n = (int(x) for x in part.split("/"))
    if not (n >= 1 and 1 <= i <= n):
        raise ValueError(f"bad --part {part!r} (expected i/n with 1 <= i <= n)")
    return i, n


def part_units(units: list[Unit], i: int, n: int) -> list[Unit]:
    """Trainable units of part i/n: contiguous blocks (keeps units of one (variant, lead) together, so a part loads
    each data slice once). The final unit is not assigned to a part: any part runs it once every trainable unit
    has its marker."""
    tr = [x for x in units if x.kind != "final"]
    lo, hi = (len(tr) * (i - 1)) // n, (len(tr) * i) // n
    return tr[lo:hi]


# ------------------------------------------------------------------------------------------- keys / store
def run_id_for(job: str) -> str:
    import hashlib
    h = hashlib.sha256(job.encode())
    for f in CODE_FILES:
        h.update((ROOT / f).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:12]


def run_prefix(job: str, run_id: str) -> str:
    p = f"{RUNS}{job}/{run_id}/"
    assert p.startswith(RUNS) and ".." not in p
    return p


def file_key(job: str, run_id: str, rel: str) -> str:
    rel = Path(rel).as_posix()
    if rel.startswith("/") or ".." in rel.split("/"):
        raise ValueError(f"bad relative path {rel}")
    return run_prefix(job, run_id) + "files/" + rel


def marker_key(job: str, run_id: str, unit_id: str) -> str:
    return run_prefix(job, run_id) + f"markers/{unit_id}.done.json"


def latest_key(job: str) -> str:
    return f"{RUNS}{job}/latest.json"


def open_run_store(spec: str, job: str):
    """Store restricted to lake/v1/runs/<job>/ (puts only; reads anywhere). Only latest.json is overwritable."""
    cfg = {"remote": {"realtime_prefix": f"{RUNS}{job}/", "status_prefix": f"{RUNS}{job}/",
                      "overwritable_keys": [latest_key(job)], "bucket_env": "R2_BUCKET"}}
    return C.open_store(spec, cfg)


def done_units(store, job: str, run_id: str) -> set[str]:
    pre = run_prefix(job, run_id) + "markers/"
    return {k[len(pre):-len(".done.json")] for k in store.list(pre) if k.endswith(".done.json")}


# ------------------------------------------------------------------------------------------- transfer
def download_inputs(store, job: str) -> dict:
    n = b = 0
    for rel in INPUTS[job]:
        for key, size in sorted(store.list(LAKE + rel).items()):
            dst = ROOT / key[len(LAKE):]
            if dst.is_file() and dst.stat().st_size == size:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".part")
            tmp.write_bytes(store.get(key))
            os.replace(tmp, dst)
            n, b = n + 1, b + size
    return {"downloaded_objects": n, "downloaded_bytes": b}


def overlay_run_files(store, job: str, run_id: str) -> int:
    """Bring every output of earlier units of this run (any part) into the workspace."""
    pre = run_prefix(job, run_id) + "files/"
    n = 0
    for key, size in sorted(store.list(pre).items()):
        dst = ROOT / key[len(pre):]
        if dst.is_file() and dst.stat().st_size == size:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(store.get(key))
        n += 1
    return n


def upload_outputs(store, job: str, run_id: str, rels: list[str], mirror: Path) -> dict:
    """Upload unit outputs. A unit without marker may be re-run after an interruption: its own output keys are
    then overwritable (never anything else, never markers)."""
    out = {}
    for rel in rels:
        p = ROOT / rel
        if not p.is_file():
            continue
        key = file_key(job, run_id, rel)
        data = p.read_bytes()
        store.overwritable.add(key)
        store.put(key, data, overwrite=True)
        store.overwritable.discard(key)
        out[key] = C.sha256_bytes(data)
        m = mirror / "files" / rel
        m.parent.mkdir(parents=True, exist_ok=True)
        m.write_bytes(data)
    return out


def write_marker(store, job: str, run_id: str, unit: Unit, outputs: dict, metrics: dict, mirror: Path) -> None:
    obj = {"job": job, "run_id": run_id, "unit": unit.id, "finished_utc": C.iso(C.utcnow()), "outputs": outputs,
           "metrics": metrics, "context": C.run_context()}
    store.put_json(marker_key(job, run_id, unit.id), obj)
    m = mirror / "markers" / f"{unit.id}.done.json"
    m.parent.mkdir(parents=True, exist_ok=True)
    m.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")


# ------------------------------------------------------------------------------------------- training units
class DataCache:
    def __init__(self):
        self.key, self.val = None, None

    def get(self, key, loader):
        if key != self.key:
            self.key, self.val = None, None
            import gc; gc.collect()
            self.val, self.key = loader(), key
        return self.val


def _rels_for(variant: str, lead: int, thr: float) -> list[str]:
    k = kstr(thr)
    d = f"models/pilot_v0/{variant}/lead{lead}_thr{k}/"
    return [d + f for f in ("model.txt", "model_prerefit.txt", "calibrator.pkl", "meta.json", "importance.csv")] + \
        [GOLD_REL + f"preds/lgbm_{variant}_lead{lead}_thr{k}_{s}.parquet" for s in ("validation", "test")] + \
        [f"reports/p4a/runs/{variant}_lead{lead}_thr{k}.json"]


def run_train(unit: Unit, cfg: dict, cache: DataCache, lg) -> tuple[list[str], dict]:
    from src.gold import p4a_common as P
    from src.models import data as D
    from src.models import train_lgbm as TL
    v, lead, thr = unit.params["variant"], unit.params["lead"], unit.params["thr"]
    feats = D.feature_list(TL.TABLES[v])
    # same config hash as src/models/train_lgbm.py main()
    ch = P.cfg_hash({"lightgbm": cfg["lightgbm"], "splits": cfg["splits"], "evaluation": cfg["evaluation"],
                     "seed": cfg["random_seed"], "variant": v, "features": feats, "code": P.code_hash(*TL.CODE)})
    tb = TL.TABLES[v]
    tune, val, test = cache.get((v, lead), lambda: (D.load_split(cfg, TL.VARIANTS[v], feats, table=tb, lead=lead),
                                                     D.load_split(cfg, "validation", feats, table=tb, lead=lead),
                                                     D.load_split(cfg, "test", feats, table=tb, lead=lead)))
    m = TL.train_one(cfg, v, lead, thr, tune, val, test, feats, ch, lg)
    return _rels_for(v, lead, thr), {"best_iteration": m["best_iteration"], "val_logloss": m["val_logloss_best"],
                                     "cut": m["cut"]["cut"], "val_csi_at_cut": m["cut"]["csi"], "seconds": m["seconds"]}


def _load_ranges(cfg, ranges, feats, lead):
    import pandas as pd
    from src.models import data as D
    return pd.concat([D.load_split(cfg, r, feats, table="table_A", lead=lead) for r in ranges], ignore_index=True)


def run_loso(unit: Unit, cfg: dict, cache: DataCache, lg) -> tuple[list[str], dict]:
    from src.gold import p4a_common as P
    from src.models import data as D
    from src.models import train_lgbm as TL
    s, lead, thr = unit.params["season"], unit.params["lead"], unit.params["thr"]
    f = LOSO[s]
    v = f"LOSO_{s}"
    feats = D.feature_list("table_A")
    # register the fold as a variant of the unchanged training code (in memory only)
    TL.VARIANTS[v], TL.TABLES[v] = f"loso_{s}_tune", "table_A"
    cfg = json.loads(json.dumps(cfg))
    cfg["splits"][f"loso_{s}_tune"] = [f["tune"][0][0], f["tune"][-1][1]]
    ch = P.cfg_hash({"lightgbm": cfg["lightgbm"], "evaluation": cfg["evaluation"], "seed": cfg["random_seed"],
                     "fold": {k: f[k] for k in ("tune", "validation", "heldout")}, "features": feats,
                     "code": P.code_hash(*TL.CODE)})
    tune, val, test = cache.get((v, lead), lambda: (_load_ranges(cfg, f["tune"], feats, lead),
                                                     D.load_split(cfg, f["validation"], feats, table="table_A", lead=lead),
                                                     D.load_split(cfg, f["heldout"], feats, table="table_A", lead=lead)))
    m = TL.train_one(cfg, v, lead, thr, tune, val, test, feats, ch, lg)
    # correct the training-period block of meta.json (train_one writes the P4A split names)
    mp = TL.MODELS / v / f"lead{lead}_thr{kstr(thr)}" / "meta.json"
    meta = json.loads(mp.read_text(encoding="utf-8"))
    meta["training_period"] = {"tune": f["tune"], "validation": list(f["validation"]),
                               "refit": f["tune"] + [list(f["validation"])], "heldout_season": list(f["heldout"])}
    meta["loso"] = {"heldout_season": s, "note": "leave-one-season-out fold (D76); not a forward-in-time test"}
    P.write_json(mp, meta)
    rels = _rels_for(v, lead, thr)
    return rels, {"best_iteration": m["best_iteration"], "val_logloss": m["val_logloss_best"], "cut": m["cut"]["cut"],
                  "seconds": m["seconds"]}


# ------------------------------------------------------------------------------------------- final units
def final_p4a6(cfg: dict) -> tuple[list[str], dict]:
    import pandas as pd
    from src.models import compare_A_B, evaluate_lgbm
    compare_A_B.main()
    evaluate_lgbm.main(["--variant", "B"])
    evaluate_lgbm.main(["--variant", "A_no2023", "--thresholds", "0.5"])
    rels = []
    for d in ("reports/p4a/compare_A_B", "reports/p4a/lgbm_B", "reports/p4a/lgbm_A_no2023"):
        rels += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / d).glob("*")) if p.is_file()]
    rels += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "reports/p4a").glob("importance_*.png"))]
    t = pd.read_csv(ROOT / "reports/p4a/compare_A_B/test_diffs.csv")
    t = t[t["comparison"] == "B_2024 - A_2024"]
    table = [{"threshold": r.threshold, "lead_min": int(r.lead_min), "csi_diff_B_minus_A": round(r.csi_diff, 4),
              "csi_ci90": [round(r.csi_lo, 4), round(r.csi_hi, 4)], "brier_diff": round(r.brier_diff, 5),
              "brier_ci90": [round(r.brier_lo, 5), round(r.brier_hi, 5)]} for r in t.itertuples()]
    v = pd.read_csv(ROOT / "reports/p4a/lgbm_B/verdict.csv")
    return rels, {"B_minus_A_2024_test": table,
                  "verdict_B": [{"threshold": r.threshold, "lead_min": int(r.lead_min), "verdict": r.verdict}
                                for r in v.itertuples()]}


def loso_metrics(cfg: dict) -> "pd.DataFrame":
    import numpy as np
    import pandas as pd
    from src.models import baselines as BL
    from src.models import data as D
    from src.models import metrics as M
    from src.gold import p4a_common as P
    rows = []
    gk = [f"gc_k{k}" for k in range(7)]
    for s, f in sorted(LOSO.items()):
        train_lab = pd.concat([D.load_split(cfg, r, [], table="table_A") for r in f["tune"] + [f["validation"]]],
                              ignore_index=True)
        held = D.load_split(cfg, f["heldout"], gk, table="table_A")
        for lead in cfg["frames"]["lead_min"]:
            h = held[held["lead_min"] == lead].reset_index(drop=True)
            tr = train_lab[train_lab["lead_min"] == lead]
            pers_x = D.asof_latest(h) if lead in (0, 30) else D.max_avail_0_60(h)
            for thr in cfg["targets"]["thresholds_mm_h"]:
                k = kstr(thr); yc = D.thr_col(thr)
                pr = pd.read_parquet(P.GOLD / "preds" / f"lgbm_LOSO_{s}_lead{lead}_thr{k}_test.parquet")
                j = pr.merge(h[["pixel_id", "t0", "frame_start"]].assign(pers=pers_x), on=["pixel_id", "t0"],
                             how="inner", validate="one_to_one")
                y = j[yc].to_numpy().astype(bool)
                pc = BL.clim_lookup(BL.clim_table(tr, yc), j)
                pc = np.where(np.isnan(pc), tr[yc].mean(), pc)
                yp = np.nan_to_num(j["pers"].to_numpy("float64"), nan=-1.0) >= thr
                sm = M.summary(y, j["p"].to_numpy("float64"), j["yhat"].to_numpy().astype(bool), pc)
                rows.append({"heldout_season": s, "lead_min": lead, "threshold": float(thr), "n": len(j),
                             "base_rate": sm["base_rate"], "csi": sm["csi"], "pod": sm["pod"], "far": sm["far"],
                             "bias": sm["bias"], "csi_pers": float(M.scores(**M.contingency(y, yp))["csi"]),
                             "brier_cal": sm["brier"], "bss_cal": sm["bss_clim"], "auc_pr_cal": sm["auc_pr"],
                             "brier_raw": M.brier(y, j["p_raw"].to_numpy("float64")),
                             "bss_raw": M.bss(y, j["p_raw"].to_numpy("float64"), pc),
                             "auc_pr_raw": M.auc_pr(y, j["p_raw"].to_numpy("float64"))})
    return pd.DataFrame(rows)


def final_p4a7(cfg: dict) -> tuple[list[str], dict]:
    import numpy as np
    import pandas as pd
    from src.gold import p4a_common as P
    out = ROOT / "reports" / "p4a" / "loso"
    out.mkdir(parents=True, exist_ok=True)
    t = loso_metrics(cfg)
    t.to_csv(out / "loso_metrics.csv", index=False)
    spread = t.groupby(["threshold", "lead_min"]).agg(
        csi_min=("csi", "min"), csi_max=("csi", "max"), csi_mean=("csi", "mean"),
        bss_raw_min=("bss_raw", "min"), bss_raw_max=("bss_raw", "max"),
        csi_minus_pers_min=("csi", lambda s: float((s - t.loc[s.index, "csi_pers"]).min())),
        csi_minus_pers_max=("csi", lambda s: float((s - t.loc[s.index, "csi_pers"]).max()))).reset_index()
    spread.to_csv(out / "loso_spread.csv", index=False)
    # determinism / code-path check: fold 2025 has exactly the P4A set-A data, splits and parameters
    rep = {}
    for lead in cfg["frames"]["lead_min"]:
        for thr in cfg["targets"]["thresholds_mm_h"]:
            k = kstr(thr)
            a = P.GOLD / "preds" / f"lgbm_A_lead{lead}_thr{k}_test.parquet"
            b = P.GOLD / "preds" / f"lgbm_LOSO_2025_lead{lead}_thr{k}_test.parquet"
            if a.is_file() and b.is_file():
                x = pd.read_parquet(a, columns=["pixel_id", "t0", "p_raw"]).merge(
                    pd.read_parquet(b, columns=["pixel_id", "t0", "p_raw"]), on=["pixel_id", "t0"], suffixes=("_A", "_L"))
                rep[f"lead{lead}_thr{k}"] = float(np.abs(x["p_raw_A"] - x["p_raw_L"]).max())
    P.write_json(out / "repro_fold2025_vs_A.json", {"max_abs_diff_p_raw": rep,
                                                   "note": "fold 2025 = P4A set A (same data, splits, parameters)"})
    rels = [p.relative_to(ROOT).as_posix() for p in sorted(out.glob("*")) if p.is_file()]
    x = t[t["threshold"] == 2.5][["heldout_season", "lead_min", "csi", "csi_pers", "bss_raw", "bss_cal"]].round(3)
    return rels, {"csi_2p5": x.to_dict("records"), "repro_fold2025_max_abs_diff": max(rep.values()) if rep else None}


FINALS = {"p4a6": final_p4a6, "p4a7": final_p4a7}


# ------------------------------------------------------------------------------------------- driver
def step_summary(job: str, run_id: str, done_now: list[str], todo_left: int, final_metrics: dict | None) -> str:
    lines = [f"### cloud_train `{job}` run `{run_id}` (PILOT v0)", "",
             f"- units finished in this invocation: {len(done_now)}; trainable units still missing: {todo_left}"]
    if done_now:
        lines.append(f"- {', '.join(done_now[:30])}{' …' if len(done_now) > 30 else ''}")
    if final_metrics:
        lines += ["", "| key | value |", "|---|---|"]
        for k, v in final_metrics.items():
            if isinstance(v, list):
                for r in v[:30]:
                    lines.append(f"| {k} | {json.dumps(r, default=str)} |")
            else:
                lines.append(f"| {k} | {v} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, choices=JOBS)
    ap.add_argument("--part", default="1/1")
    ap.add_argument("--final", action="store_true", help="p4b only: run the P4B.6 test evaluation")
    ap.add_argument("--store", default="r2", help="r2 (default) or local:<dir>")
    ap.add_argument("--run-id", default=None, help="override the code-derived run id")
    ap.add_argument("--max-minutes", type=float, default=320.0, help="stop starting new units after this time")
    ap.add_argument("--plan", action="store_true", help="print the units of this part and their state; train nothing")
    a = ap.parse_args(argv)
    t_start = time.time()
    from src.gold import p4a_common as P
    P.require_guard()
    cfg = P.load_cfg()
    if a.job == "p4b":
        C.log("job_reserved", job="p4b", final=a.final,
              reason="P4B code (CV folds, motion features, grid, calibration, ablations) is developed under "
                     "TASK_P4B_MODEL_V1.md after the p4a6 results are read back (P4B.4 depends on P4A.6); D77")
        return 2
    i, n = parse_part(a.part)
    units = job_units(a.job, cfg)
    mine = part_units(units, i, n)
    run_id = a.run_id or run_id_for(a.job)
    store = open_run_store(a.store, a.job)
    done = done_units(store, a.job, run_id)
    todo = [u for u in mine if u.id not in done]
    C.log("plan", job=a.job, run_id=run_id, part=f"{i}/{n}", units_in_part=len(mine), done=len(mine) - len(todo),
          todo=[u.id for u in todo][:40], final_done="final" in done)
    if a.plan:
        return 0
    mirror = ROOT / "cloud_out" / a.job / run_id
    store.put_json(latest_key(a.job), {"job": a.job, "run_id": run_id, "updated_utc": C.iso(C.utcnow()),
                                       "context": C.run_context()}, overwrite=True)
    if todo or "final" not in done:
        C.log("inputs", **download_inputs(store, a.job))
    lg = P.get_logger(f"cloud_{a.job}")
    cache = DataCache()
    done_now = []
    for u in todo:
        if (time.time() - t_start) / 60 > a.max_minutes:
            C.log("time_budget_reached", remaining=len(todo) - len(done_now), hint="run the job again: it resumes")
            break
        t1 = time.time()
        rels, metrics = (run_train if u.kind == "train" else run_loso)(u, cfg, cache, lg)
        outs = upload_outputs(store, a.job, run_id, rels, mirror)
        write_marker(store, a.job, run_id, u, outs, metrics, mirror)
        done_now.append(u.id)
        C.log("unit_done", unit=u.id, seconds=round(time.time() - t1, 1), rss_gb=round(P.rss_gb(), 2), **{
            k: v for k, v in metrics.items() if k in ("best_iteration", "val_logloss", "cut")})
    done = done_units(store, a.job, run_id)
    missing = [u.id for u in units if u.kind != "final" and u.id not in done]
    final_metrics = None
    if not missing and "final" not in done and (time.time() - t_start) / 60 <= a.max_minutes:
        C.log("final_start", job=a.job, overlay_files=overlay_run_files(store, a.job, run_id))
        rels, final_metrics = FINALS[a.job](cfg)
        outs = upload_outputs(store, a.job, run_id, rels, mirror)
        summary = {"job": a.job, "run_id": run_id, "status": "PILOT v0 — labels IMERG Final (satellite estimate)",
                   "units": len(units), "metrics": final_metrics, "finished_utc": C.iso(C.utcnow()),
                   "context": C.run_context()}
        store.put_json(run_prefix(a.job, run_id) + "summary.json", summary)
        write_marker(store, a.job, run_id, Unit("final", "final"), outs, final_metrics, mirror)
        store.put_json(run_prefix(a.job, run_id) + "done.json", {"job": a.job, "run_id": run_id,
                                                                   "finished_utc": summary["finished_utc"]})
        (mirror / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    elif "final" in done:
        C.log("final_already_done", job=a.job, run_id=run_id)
    md = step_summary(a.job, run_id, done_now, len(missing), final_metrics)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write(C.redact(md))
    C.log("end", job=a.job, run_id=run_id, finished_now=len(done_now), missing_trainable=len(missing),
          final=("done" if final_metrics or "final" in done else "waiting"), store=store.stats(),
          minutes=round((time.time() - t_start) / 60, 1))
    return 0


if __name__ == "__main__":
    raise SystemExit(C.cli(main))
