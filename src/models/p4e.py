"""P4E — v2 challenger (Model 1 = M0 + buffer ring), units of the cloud job `p4e_smoke` (driver:
src/models/cloud_train.py, main_p4e). TASK_P4E_MODEL_V2_CHALLENGER.md; configs/model_p4e.yaml merged on top of
configs/model_v2.yaml. Not a flood model; labels are IMERG (satellite estimate), not rain gauges. v1 stays official.

P4E.0 content (this module, smoke job):
  * every shared output has exactly one writer (cloud_train.commit_unit_single, D188 fix);
  * one learning rate / round cap for all P4E models (lightgbm.learning_rate, inner_cv.max_rounds, D190) and the
    pre-registered rounds check (D189): M0 / Model 1 median rounds <= 0.6 x max_rounds, else the comparison stops;
  * cost model from the P4D logs (planning.cost_model, D191) used for the per-part runtime estimate;
  * downsampling of dry scenes (inputs only; P4E.1) and the two-stage 7.6 model (P4E.3), one unit each in the smoke.

Smoke units (fold 2024, F0, 2.5 and 7.6 mm/h, P4D smoke days):
  0 preflight            gate (P4D preflight: R2 budget, ring geometry)
  1 feat-YYYY-MM         Gold table_v2 month; REUSED when part.parquet + part.meta.json exist for the gold id
  2 g2                   gate (P4D G2: partitions, NaN share, A1 equality, inner-CV label check, "plain" variant only)
  3 base-s2024-lead0     evaluation rows (P4D)
    fit-m0 / fit-m1      M0 (47 city pixels) and Model 1 (+ ring) at 2.5 and 7.6
    fit-m1ds10           Model 1 with dry scenes downsampled (p = 0.1, weight 1/p), 2.5
    fit-m1ts             Model 1 two-stage 7.6: P(R) x P(>=7.6 | R), R = label >= 0.5 at the pixel or a neighbour
  4 cmp                  gate: metrics, bootstrap differences, rounds check, equivalence (informative in the smoke),
                         two-stage violation rate, dry-scene share, cost actual vs predicted
  5 compare-done         gate: compare_summary.json; its winner writes summary.json and COMPARE_DONE.json
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.models import p4b as PB
from src.models import p4d as PD

CODE_ROOT = Path(__file__).resolve().parents[2]
CFG_PATH = CODE_ROOT / "configs" / "model_p4e.yaml"
LAKE = "lake/v1/"
GOLD_REMOTE = PD.GOLD_REMOTE                                     # lake/v1/gold/met_training/table_v2/
GOLD_REMOTE_PREFIX = LAKE + "gold/met_training/table_v2"         # store restriction: table_v2*/ (CLAUDE.md §8)
OUT = "reports/p4e"
NOT_FLOOD = PD.NOT_FLOOD
CODE_FILES = PD.CODE_FILES + ["src/models/p4e.py", "configs/model_p4e.yaml"]
# D189: sha256 of the pre-registered sections (JSON, sort_keys) written to DECISIONS before any P4E training code
PREREG_SHA = "58eb5a7ee66974f209a9ad0e361fbd9543bce46da79db48a7646fef6e47022b2"
PREREG_SECTIONS = {"rounds_check": ("rounds_check",), "equivalence": ("downsampling", "equivalence"),
                   "propose_rule": ("diagnostics", "propose_rule"), "use_rule": ("two_stage", "use_rule"),
                   "shadow_gate": ("shadow_gate",), "planning": ("planning",)}
DRILL_GATE = "cmp"
LAST_OPENING_STAGE = 2           # preflight, feat, g2: every part waits for them; later gates close the run
# rounds priors from the 216 P4D fits (lr 0.03, early stopping 150; weight settings with alpha/beta > 0 excluded):
# (mean optimal inner-fold rounds, median chosen rounds) per pixel family and threshold (D191)
ROUNDS_PRIOR_LR003 = {("city", 0.5): (227, 177), ("city", 2.5): (216, 161), ("city", 7.6): (202, 110),
                      ("ring", 0.5): (376, 283), ("ring", 2.5): (312, 195), ("ring", 7.6): (261, 149)}
INNER_ROWS_SHARE = 0.907         # mean inner-fold training rows / fit training rows (P4D)
DRY_SHARE_PRIOR = 0.30           # planning only; measured by the smoke and P4E.1 (dry rows / training rows)
RAIN_SCENE_SHARE_PRIOR = 0.35    # planning only; rows in R (label >= 0.5 at the pixel or a neighbour)


def root() -> Path:
    return PB.ROOT


# ============================================================================ configuration
def _prereg(raw: dict) -> dict:
    out = {}
    for name, path in PREREG_SECTIONS.items():
        v = raw
        for k in path:
            v = v[k]
        out[name] = v
    return out


def prereg_sha(raw: dict) -> str:
    return hashlib.sha256(json.dumps(_prereg(raw), sort_keys=True).encode()).hexdigest()


def load_cfg(smoke: bool = False, path: Path | None = None, check_prereg: bool = True) -> dict:
    raw = yaml.safe_load(Path(path or CFG_PATH).read_text(encoding="utf-8"))
    if check_prereg and prereg_sha(raw) != PREREG_SHA:
        raise RuntimeError("pre-registered sections of configs/model_p4e.yaml changed after DECISIONS D189 "
                           f"({prereg_sha(raw)} != {PREREG_SHA}): stop and ask (TASK P4E)")
    base = yaml.safe_load((CODE_ROOT / raw["base_config"]).read_text(encoding="utf-8"))
    for k in list(raw.get("not_used_from_base", [])) + ["smoke"]:
        base.pop(k, None)
    top = {k: v for k, v in raw.items() if k not in ("smoke", "base_config", "not_used_from_base")}
    cfg = PB.deep_merge(base, top)
    if smoke:
        cfg = PB.deep_merge(cfg, raw.get("smoke", {}))
        cfg["smoke"] = True
    return cfg


kstr, thr_col = PB.kstr, PB.thr_col
thresholds, primary, heavy, leads, seasons = PD.thresholds, PD.primary, PD.heavy, PD.leads, PD.seasons
lgb_params = PD.lgb_params
gold_id = PD.gold_id                                 # same gold keys as P4D: same table_v2 gid for the full job


@contextlib.contextmanager
def p4d_env():
    """Run shared P4D helpers with the P4E output folder and without the P4D cut variant (P4E has no cut)."""
    old = PD.OUT, PD.VARIANTS
    PD.OUT, PD.VARIANTS = OUT, {"plain": False}
    try:
        yield
    finally:
        PD.OUT, PD.VARIANTS = old


def inner_cv_plan_check(cfg: dict) -> list[dict]:
    with p4d_env():
        return PD.inner_cv_plan_check(cfg)


# ============================================================================ units
def fit_id(spec: str, s: int, lead: int, thr: float) -> str:
    return PD.fit_id(spec, s, lead, thr)


def dataset_def(spec: str) -> dict:
    if spec == "m0":
        return {"ring": False}
    if spec == "m1":
        return {"ring": True}
    m = re.fullmatch(r"m1ds(\d+)", spec)
    if m:
        return {"ring": True, "ds_p": int(m.group(1)) / 100}
    if spec == "m1ts":
        return {"ring": True, "two_stage": True}
    raise ValueError(spec)


def smoke_fit_specs(cfg: dict) -> list[tuple[str, float]]:
    pt, hv = primary(cfg), heavy(cfg)
    p = int(round(100 * min(cfg["downsampling"]["p_grid"])))
    return [("m0", pt), ("m0", hv), ("m1", pt), ("m1", hv), (f"m1ds{p:02d}", pt), ("m1ts", hv)]


def units(cfg: dict) -> list:
    from src.gold import build_table_v2 as BT2
    if not cfg.get("smoke"):
        raise NotImplementedError("the full p4e job is defined after the smoke (TASK P4E order)")
    U = PB.Unit
    u = [U("preflight", "gate", 0)]
    feat = [U(f"feat-{y}-{m:02d}", "feat", 1, ("preflight",), {"y": y, "m": m}) for y, m in BT2.build_months(cfg)]
    u += feat
    u.append(U("g2", "gate", 2, tuple(x.id for x in feat)))
    work = []
    for L in leads(cfg):
        for s in seasons(cfg):
            work.append(U(PD.base_id(s, L), "base", 3, ("g2",), {"season": s, "lead": L}))
            for spec, t in smoke_fit_specs(cfg):
                work.append(U(fit_id(spec, s, L, t), "fit", 3, ("g2",), {"spec": spec, "season": s, "lead": L,
                                                                           "thr": t}))
    u += work
    u.append(U("cmp", "gate", 4, tuple(x.id for x in work)))
    u.append(U("compare-done", "gate", 5, ("cmp",)))
    return u


part_units = PB.part_units


# ============================================================================ run context
class Ctx(PD.Ctx):
    """p4d.Ctx (table_v2 gold id and layout) for P4E."""


def write_json(p: Path, obj) -> None:
    PB.write_json(p, obj)


def rel(p: Path) -> str:
    return Path(p).resolve().relative_to(root().resolve()).as_posix()


def out_dir(*parts: str) -> Path:
    p = root().joinpath(OUT, *parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ============================================================================ Gold reuse
def run_feat(unit, ctx: Ctx) -> tuple[list[str], dict]:
    """Reuse a table_v2 partition that already exists on R2 for this gold id (gold id = same code and config as when
    it was built); build it with the P4D code path only when it is missing."""
    y, m = unit.params["y"], unit.params["m"]
    keys = [ctx.gold_key(y, m, n) for n in ("part.parquet", "part.meta.json")]
    have = ctx.store.list(f"{GOLD_REMOTE}gid={ctx.gid}/season={y}/month={m:02d}/")
    if all(k in have for k in keys):
        return [], {"reused": True, "gold_keys": keys, "bytes": int(sum(have[k] for k in keys))}
    with p4d_env():
        rels, met = PD.run_feat(unit, ctx)
    return rels, {**met, "reused": False}


# ============================================================================ downsampling and rain scenes
def rain_columns(feats: list[str], cfg: dict) -> list[str]:
    pats = [re.compile(p) for p in cfg["downsampling"]["dry_columns_regex"]]
    return [c for c in feats if any(p.search(c) for p in pats)]


def dry_scene_rows(meta: pd.DataFrame, X: np.ndarray, feats: list[str], cfg: dict) -> np.ndarray:
    """True for every row whose time t0 is a dry scene: all rows of t0 in the loaded frame (city + ring pixels) have
    exactly 0 in every rain-valued input column (NaN = not dry). Inputs only; labels are never read."""
    cols = rain_columns(feats, cfg)
    if not cols:
        raise RuntimeError("no rain-valued input column matched downsampling.dry_columns_regex")
    idx = [feats.index(c) for c in cols]
    row_dry = np.ones(len(meta), bool)
    for i in idx:                                      # column by column: no copy of the feature matrix (RAM)
        row_dry &= X[:, i] == 0                        # NaN == 0 is False
    codes, uniq = pd.factorize(meta["t0"])
    scene = np.ones(len(uniq), bool)
    np.logical_and.at(scene, codes, row_dry)
    return scene[codes]


def keep_draw(t0: pd.Series, seed: str) -> np.ndarray:
    """Deterministic uniform number in [0, 1) per row from sha256(seed | t0 UTC ISO): same for every pixel of t0,
    every fold, every run."""
    codes, uniq = pd.factorize(t0)
    u = np.array([int.from_bytes(hashlib.sha256(f"{seed}|{pd.Timestamp(t).tz_convert('UTC').isoformat()}".encode())
                                 .digest()[:8], "big") / 2.0 ** 64 for t in uniq])
    return u[codes]


def downsample(meta: pd.DataFrame, X: np.ndarray, feats: list[str], cfg: dict, p: float) -> tuple:
    """(drop mask, weights, dry-scene mask, stats): dry scenes are kept with probability p (whole scene), kept dry rows weigh 1/p."""
    dry = dry_scene_rows(meta, X, feats, cfg)
    u = keep_draw(meta["t0"], str(cfg["downsampling"]["seed"]))
    drop = dry & (u >= p)
    w = np.where(dry & ~drop, 1.0 / p, 1.0).astype("float32")
    return drop, w, dry, {"p": p, "rain_columns": len(rain_columns(feats, cfg)), "dry_rows": int(dry.sum()),
                     "rows": int(len(dry))}


def rain_scene(meta: pd.DataFrame, cfg: dict) -> np.ndarray:
    """R: label >= scene_mm_h at the pixel or at any of its 8 neighbours, same t0 and frame (IMERG labels).
    Neighbours outside the loaded pixel set are ignored (outer ring pixels only; city pixels have all 8)."""
    from src.gold import features as F
    thr = float(cfg["two_stage"]["scene_mm_h"])
    i, j = F.Grid().ij(meta["pixel_id"].to_numpy("int64"))
    i, j = i - i.min() + 1, j - j.min() + 1
    codes, uniq = pd.factorize(meta["t0"])
    wet = np.nan_to_num(meta["label_rate_mm_h"].to_numpy("float32"), nan=0.0) >= thr
    g = np.zeros((len(uniq), i.max() + 2, j.max() + 2), bool)
    g[codes, i, j] = wet
    nb = np.zeros_like(g)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            nb[:, 1:-1, 1:-1] |= g[:, 1 + di:g.shape[1] - 1 + di, 1 + dj:g.shape[2] - 1 + dj]
    return nb[codes, i, j]


# ============================================================================ fits
def _month_bounds(ym: int):
    return PD._month_bounds(ym)


def cv_fit(X, y, w, keep, inner_eval, meta, params, ic, feats) -> dict:
    """Inner blocked CV by calendar month (P4D design): early stopping on the held-out month (inner_eval rows of that
    month: city, all days, never downsampled or weighted), median of the fold optima; returns the boosters for the
    inner OOF, the per-fold record and the boosting work (rows x rounds run)."""
    import lightgbm as lgb
    buf = pd.Timedelta(days=int(ic["buffer_days"]))
    ym = meta["ym"].to_numpy()
    t0 = meta["t0"]
    boosters, best, folds, work = [], [], [], 0
    cap, es = int(ic["max_rounds"]), int(ic["early_stopping_rounds"])
    for mo in sorted(set(ym[inner_eval].tolist())):
        a, b = _month_bounds(mo)
        ho = inner_eval & (ym == mo)
        tr = keep & (ym != mo) & ((t0 < a - buf) | (t0 >= b + buf)).to_numpy()
        if ho.sum() == 0 or tr.sum() == 0 or y[tr].sum() == 0 or y[ho].sum() == 0:
            folds.append({"month": int(mo), "train_rows": int(tr.sum()), "held_rows": int(ho.sum()), "used": False})
            continue
        dtr = lgb.Dataset(X[tr], y[tr], weight=None if w is None else w[tr], feature_name=feats, free_raw_data=True)
        dva = lgb.Dataset(X[ho], y[ho], reference=dtr)
        bst = lgb.train(params, dtr, num_boost_round=cap, valid_sets=[dva], valid_names=["held"],
                        callbacks=[lgb.early_stopping(es, verbose=False)])
        bi = int(bst.best_iteration or bst.current_iteration())
        work += int(tr.sum()) * int(bst.current_iteration())
        boosters.append((mo, ho, bst))
        best.append(bi)
        folds.append({"month": int(mo), "train_rows": int(tr.sum()), "held_rows": int(ho.sum()),
                      "train_pos": int(y[tr].sum()), "best_iteration": bi, "rounds_run": int(bst.current_iteration()),
                      "at_cap": bi >= cap - es, "used": True})
        del dtr, dva
    if not boosters:
        raise RuntimeError("no usable inner fold")
    med = max(int(np.median(best)), 1)
    return {"boosters": boosters, "best": best, "folds": folds, "median": med, "work": work}


def inner_oof(cv: dict, X) -> dict:
    """{month: (held mask, prediction at the median rounds)}."""
    out = {}
    for mo, ho, bst in cv["boosters"]:
        out[mo] = (ho, bst.predict(X[ho], num_iteration=max(min(cv["median"], bst.current_iteration()), 1)))
    return out


def refit(X, y, w, keep, rounds: int, params, feats):
    import lightgbm as lgb
    d = lgb.Dataset(X[keep], y[keep], weight=None if w is None else w[keep], feature_name=feats)
    return lgb.train(params, d, num_boost_round=rounds)


def _family(dd: dict) -> str:
    return "ring" if dd.get("ring") else "city"


def run_fit(unit, ctx: Ctx) -> tuple[list[str], dict]:
    import time
    from src.gold import p4a_common as P
    from src.models import metrics as M
    cfg = ctx.cfg
    spec, s, L, thr = unit.params["spec"], unit.params["season"], unit.params["lead"], unit.params["thr"]
    dd = dataset_def(spec)
    t1 = time.time()
    with p4d_env():
        ctx.fetch_gold()
        feats = PD.feature_list(ctx, dd["ring"])
        meta, X = ctx.cache.get((L, tuple(feats)), lambda: PD.load_frame(ctx, L, feats))
    y = meta[thr_col(thr)].to_numpy("int8")
    city = meta["is_city"].to_numpy("float32") == 1
    trainable = (meta["season"].to_numpy() != s) & meta["train_month"].to_numpy(bool)
    keep = trainable & (city | dd["ring"])
    held = (meta["season"].to_numpy() == s) & city
    inner_eval = trainable & city
    params, ic = lgb_params(cfg), cfg["inner_cv"]
    w, ds = None, None
    if dd.get("ds_p"):
        drop, w, dry, ds = downsample(meta, X, feats, cfg, dd["ds_p"])
        ds |= {"training_rows_full": int(keep.sum()), "dry_training_rows": int((keep & dry).sum())}
        keep = keep & ~drop
        ds |= {"training_rows_kept": int(keep.sum()), "rows_saved": ds["training_rows_full"] - int(keep.sum()),
               "dry_share_training": ds["dry_training_rows"] / max(ds["training_rows_full"], 1)}
    info = {"unit": unit.id, "spec": spec, "dataset": dd, "heldout_season": s, "lead_min": L, "threshold": thr,
            "labels": "training: IMERG Final V07 rows only; held-out: IMERG Final (Late for 2025-10, secondary)",
            "features": feats, "n_features": len(feats), "lgb_params": params,
            "inner_cv": {k: ic[k] for k in ("max_rounds", "early_stopping_rounds", "buffer_days")},
            "downsampling": ds, "held_rows": int(held.sum()), "note": NOT_FLOOD}
    if dd.get("two_stage"):
        rs = rain_scene(meta, cfg)
        y1 = rs.astype("int8")
        cv1 = cv_fit(X, y1, w, keep, inner_eval, meta, params, ic, feats)
        cv2 = cv_fit(X, y, w, keep & rs, inner_eval & rs, meta, params, ic, feats)
        o1 = inner_oof(cv1, X)
        iy, ip = [], []
        for mo, (ho, p1) in o1.items():                 # product on every held-out city row of the month
            if mo not in {m for m, _, _ in cv2["boosters"]}:
                continue
            b2 = [b for m, _, b in cv2["boosters"] if m == mo][0]
            p2 = b2.predict(X[ho], num_iteration=max(min(cv2["median"], b2.current_iteration()), 1))
            iy.append(y[ho])
            ip.append(p1 * p2)
        iy, ip = np.concatenate(iy).astype(bool), np.concatenate(ip)
        b1 = refit(X, y1, w, keep, cv1["median"], params, feats)
        b2 = refit(X, y, w, keep & rs, cv2["median"], params, feats)
        p1h, p2h = b1.predict(X[held]), b2.predict(X[held])
        p_out = p1h * p2h
        work = cv1["work"] + cv2["work"] + int(keep.sum()) * cv1["median"] + int((keep & rs).sum()) * cv2["median"]
        info |= {"two_stage": {"scene_mm_h": cfg["two_stage"]["scene_mm_h"], "rain_scene_share_training":
                               float(rs[keep].mean()), "stage1": {k: cv1[k] for k in ("folds", "best", "median")},
                               "stage2": {k: cv2[k] for k in ("folds", "best", "median")},
                               "stage2_training_rows": int((keep & rs).sum())},
                 "median_rounds": int(max(cv1["median"], cv2["median"]))}
        extra = {"p_stage1": p1h.astype("float32"), "p_stage2": p2h.astype("float32")}
        folds_all = cv1["folds"] + cv2["folds"]
    else:
        cv = cv_fit(X, y, w, keep, inner_eval, meta, params, ic, feats)
        o = inner_oof(cv, X)
        iy = np.concatenate([y[ho] for ho, _ in o.values()]).astype(bool)
        ip = np.concatenate([p for _, p in o.values()])
        bst = refit(X, y, w, keep, cv["median"], params, feats)
        p_out = bst.predict(X[held])
        work = cv["work"] + int(keep.sum()) * cv["median"]
        info |= {"inner_folds": cv["folds"], "inner_fold_best_iterations": cv["best"], "median_rounds": cv["median"]}
        extra = {}
        folds_all = cv["folds"]
    platt = PB.fit_calibrator("platt", ip, iy)
    ipc = PB.apply_calibrator(platt, ip)
    rule = cfg["evaluation"]["cut_rules"][f"{thr:g}"]
    cut = M.choose_cut(iy, ipc, rule)
    oof = pd.DataFrame({"pixel_id": meta["pixel_id"].to_numpy("int64")[held],
                        "t0": meta.loc[held, "t0"].reset_index(drop=True).astype("datetime64[us, UTC]"),
                        "p_raw": p_out.astype("float32"), **extra})
    po = out_dir("oof") / f"{unit.id}.parquet"
    PB.write_parquet(oof, po)
    used = [f for f in folds_all if f.get("used")]
    fam = _family(dd)
    cm = cfg["planning"]["cost_model"][fam]
    info |= {"platt": platt, "cut": cut, "train_rows": int(keep.sum()), "train_rows_ring": int((keep & ~city).sum()),
             "train_pos": int(y[keep].sum()), "train_weight_sum": float(w[keep].sum()) if w is not None else None,
             "inner_oof": {"n": int(len(iy)), "base_rate": float(iy.mean()), "logloss_raw": PB.logloss(iy, ip),
                           "logloss_cal": PB.logloss(iy, ipc), "brier_raw": M.brier(iy, ip),
                           "brier_cal": M.brier(iy, ipc)},
             "rounds": {"max_rounds": int(ic["max_rounds"]), "inner_at_cap": int(sum(f["at_cap"] for f in used)),
                        "inner_used": len(used), "median_share_of_cap": info["median_rounds"] / int(ic["max_rounds"])},
             "work_grows": work / 1e9, "cost_family": fam,
             "predicted_seconds_from_work": float(cm["a_s"]) + float(cm["c_s_per_grow"]) * work / 1e9}
    ip_ = out_dir("fit") / f"{unit.id}.json"
    write_json(ip_, info)
    rss = P.rss_gb()
    if np.isfinite(rss) and rss > float(cfg["resources"]["max_rss_gb"]):
        raise RuntimeError(f"{unit.id}: RSS {rss:.2f} GB above resources.max_rss_gb")
    return [rel(po), rel(ip_)], {"median_rounds": info["median_rounds"], "train_rows": int(keep.sum()),
                                 "rss_gb": round(rss, 2), "work_grows": round(work / 1e9, 4),
                                 "seconds": round(time.time() - t1, 1)}


# ============================================================================ comparison gate
def smoke_models(cfg: dict) -> dict:
    """{threshold: [(name, spec, nested_spec or None), ...]}; the first entry of each list is the reference."""
    pt, hv = primary(cfg), heavy(cfg)
    p = int(round(100 * min(cfg["downsampling"]["p_grid"])))
    return {pt: [("M0", "m0", None), ("Model1", "m1", None), (f"Model1-ds{p:02d}", f"m1ds{p:02d}", None)],
            hv: [("M0", "m0", "m0"), ("Model1", "m1", "m1"), ("Model1-twostage", "m1ts", None)]}


def _frame(ctx, spec, L, thr, nested, roles=("eval",)):
    with p4d_env():
        return PD.pooled(ctx, spec, L, thr, roles=roles, nested_spec=nested)


def paired(ctx, ja: pd.DataFrame, jb: pd.DataFrame, thr: float) -> dict:
    from src.models import metrics as M
    ev, seed = ctx.cfg["evaluation"], ctx.cfg["random_seed"]
    da = M.day_stats(ja["day"].to_numpy(), ja["y"], ja["yhat"], ja["p_cal"], ja["p_clim"])
    db = M.day_stats(jb["day"].to_numpy(), jb["y"], jb["yhat"], jb["p_cal"], jb["p_clim"])
    d = M.paired_bootstrap(da, db, ev["bootstrap_resamples"], seed, ev["ci_level"])
    ll = PD.ll_paired(PD.ll_day(ja), PD.ll_day(jb), ev["bootstrap_resamples"], seed, ev["ci_level"])
    bins = ev["reliability_bins"]
    return {"logloss_cal_diff": ll["diff"], "logloss_lo": ll["lo"], "logloss_hi": ll["hi"],
            **{f"{k}_{x}": d[k][x] for k in ("csi", "brier", "bss_clim", "pod") for x in ("diff", "lo", "hi")},
            "ece_cal_diff": PD.ece(ja["y"], ja["p_cal"], bins) - PD.ece(jb["y"], jb["p_cal"], bins)}


def equivalence_verdict(rows: list[dict], eq: dict) -> dict:
    """Pre-registered P4E.1 rule (D189) on (downsampled - full) differences, one row per frame with keys lead_min,
    logloss_lo, logloss_hi, bss_clim_diff, csi_diff, ece_cal_diff (2.5) and csi76_diff (7.6, may be None)."""
    lo, hi = eq["logloss_2p5_ci90_within"]
    checks = []
    for r in rows:
        c = {"lead_min": r["lead_min"],
             "logloss_ci_within": lo <= r["logloss_lo"] and r["logloss_hi"] <= hi,
             "abs_bss_ok": abs(r["bss_clim_diff"]) <= eq["abs_bss_2p5_max"],
             "abs_csi_ok": abs(r["csi_diff"]) <= eq["abs_csi_2p5_max"],
             "ece_ok": r["ece_cal_diff"] <= eq["ece_2p5_increase_max"],
             "csi76_ok": None if r.get("csi76_diff") is None else r["csi76_diff"] >= eq["csi_7p6_min_diff"]}
        checks.append(c)
    complete = bool(checks) and all(c["csi76_ok"] is not None for c in checks)
    passed = complete and all(all(v for k, v in c.items() if k != "lead_min") for c in checks)
    return {"complete": complete, "pass": passed, "checks": checks}


def rounds_check(ctx) -> dict:
    rc, cap = ctx.cfg["rounds_check"], int(ctx.cfg["inner_cv"]["max_rounds"])
    lim = float(rc["max_share_of_cap"]) * cap
    rows = []
    for u in units(ctx.cfg):
        if u.kind == "fit" and u.params["spec"] in rc["specs"]:
            f = ctx.read_json(f"{OUT}/fit/{u.id}.json")
            rows.append({"unit": u.id, "median_rounds": f["median_rounds"], "limit": lim,
                         "inner_at_cap": f["rounds"]["inner_at_cap"], "ok": f["median_rounds"] <= lim})
    return {"ok": all(r["ok"] for r in rows), "limit": lim, "max_rounds": cap, "units": rows}


def cost_table(ctx) -> pd.DataFrame:
    from src.models import cloud_train as CT
    est = unit_minutes(ctx.cfg, units(ctx.cfg))
    rows = []
    for u in units(ctx.cfg):
        if u.kind != "fit":
            continue
        f = ctx.read_json(f"{OUT}/fit/{u.id}.json")
        mk = json.loads(ctx.store.get(CT.marker_key(ctx.job, ctx.run_id, u.id)))
        rows.append({"unit": u.id, "family": f["cost_family"], "work_grows": f["work_grows"],
                     "median_rounds": f["median_rounds"], "seconds_actual": mk["metrics"].get("seconds"),
                     "seconds_from_work_model": f["predicted_seconds_from_work"],
                     "seconds_planned": round(60 * est[u.id], 1)})
    return pd.DataFrame(rows)


def gate_cmp(ctx: Ctx) -> tuple[list[str], dict]:
    cfg = ctx.cfg
    rc = rounds_check(ctx)
    out = out_dir("compare")
    write_json(out / "rounds_check.json", rc)
    if not rc["ok"]:
        raise RuntimeError(f"rounds check FAIL (pre-registered D189): {[r for r in rc['units'] if not r['ok']]}; "
                           "stop and ask before any comparison")
    bins = cfg["evaluation"]["reliability_bins"]
    met, diffs, eqrows = [], [], {}
    frames = {}
    for thr, models in smoke_models(cfg).items():
        for L in leads(cfg):
            for period, roles in (("eval", ("eval",)), ("secondary", ("secondary",))):
                fr = {n: _frame(ctx, sp, L, thr, nest, roles) for n, sp, nest in models}
                if not len(fr[models[0][0]]):
                    continue
                frames[(thr, L, period)] = fr
                for n, sp, _ in models:
                    with p4d_env():
                        met.append({"period": period, "lead_min": L, "threshold": thr, "model": n, "spec": sp,
                                    **PD.metric_row(fr[n], thr, bins)})
                ref = "M0"
                pairs = [(models[1][0], ref), (models[2][0], models[1][0])]
                for a, b in pairs:
                    diffs.append({"period": period, "lead_min": L, "threshold": thr, "comparison": f"{a} - {b}",
                                  **paired(ctx, fr[a], fr[b], thr)})
    pt, hv = primary(cfg), heavy(cfg)
    ds_name = smoke_models(cfg)[pt][2][0]
    for L in leads(cfg):
        d = [r for r in diffs if r["period"] == "eval" and r["lead_min"] == L and r["threshold"] == pt and
             r["comparison"] == f"{ds_name} - Model1"]
        if d:
            eqrows[L] = {"lead_min": L, **{k: d[0][k] for k in ("logloss_lo", "logloss_hi", "bss_clim_diff",
                                                                    "csi_diff", "ece_cal_diff")}, "csi76_diff": None}
    eqv = equivalence_verdict(list(eqrows.values()), cfg["downsampling"]["equivalence"])
    viol = []                                          # two-stage P(>=7.6) <= Model 1 P(>=2.5) after combination
    for L in leads(cfg):
        if (hv, L, "eval") in frames and (pt, L, "eval") in frames:
            a = frames[(hv, L, "eval")]["Model1-twostage"][["pixel_id", "t0", "p_cal"]]
            b = frames[(pt, L, "eval")]["Model1"][["pixel_id", "t0", "p_cal"]]
            j = a.merge(b, on=["pixel_id", "t0"], suffixes=("_76", "_25"))
            viol.append({"lead_min": L, "rows": len(j), "violation_rate": float((j["p_cal_76"] > j["p_cal_25"]).mean())})
    ds_info = [ctx.read_json(f"{OUT}/fit/{u.id}.json").get("downsampling") for u in units(cfg)
               if u.kind == "fit" and u.params["spec"].startswith("m1ds")]
    pd.DataFrame(met).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(diffs).to_csv(out / "diffs.csv", index=False)
    ct = cost_table(ctx)
    ct.to_csv(out / "cost.csv", index=False)
    checks = {"rounds_check_ok": rc["ok"], "equivalence_smoke_informative": eqv, "two_stage_violation": viol,
              "downsampling": ds_info, "note": "smoke: one fold, one frame; verdicts here are informative only, "
                                               "never a P4E.1 / P4E.3 decision. " + NOT_FLOOD}
    write_json(out / "checks.json", checks)
    rels = [rel(out / f) for f in ("rounds_check.json", "metrics.csv", "diffs.csv", "cost.csv", "checks.json")]
    return rels, {"rounds_check_ok": rc["ok"], "equivalence_complete": eqv["complete"],
                  "two_stage_violation": [round(v["violation_rate"], 4) for v in viol],
                  "dry_share_training": [round(x["dry_share_training"], 4) for x in ds_info if x],
                  "fits": len(ct)}


def gate_compare_done(ctx: Ctx) -> tuple[list[str], dict]:
    ch = ctx.read_json(f"{OUT}/compare/checks.json")
    rc = ctx.read_json(f"{OUT}/compare/rounds_check.json")
    summ = {"job": ctx.job, "run_id": ctx.run_id, "gold_id": ctx.gid, "smoke": bool(ctx.cfg.get("smoke")),
            "prereg_sha": PREREG_SHA, "rounds_check": rc, "checks": ch, "note": NOT_FLOOD}
    p = out_dir() / "compare_summary.json"
    write_json(p, summ)
    return [rel(p)], {"compare_complete": True, "rounds_check_ok": rc["ok"]}


# ============================================================================ dispatch
def gate_preflight(ctx: Ctx):
    with p4d_env():
        return PD.gate_preflight(ctx)


def gate_g2(ctx: Ctx):
    with p4d_env():
        return PD.gate_g2(ctx)


GATES = {"preflight": gate_preflight, "g2": gate_g2, "cmp": gate_cmp, "compare-done": gate_compare_done}


def run_unit(unit, ctx: Ctx) -> tuple[list[str], dict]:
    if unit.kind == "feat":
        return run_feat(unit, ctx)
    if unit.kind == "base":
        with p4d_env():
            return PD.run_base(unit, ctx)
    if unit.kind == "fit":
        return run_fit(unit, ctx)
    if unit.kind == "gate":
        return GATES[unit.id](ctx)
    raise ValueError(unit.kind)


# ============================================================================ planning (cost model, D191)
def training_rows(cfg: dict, s: int, ring: bool, n_ring: int | None = None) -> int:
    from src.gold import build_table_v2 as BT2
    days = [d for d in BT2.build_days(cfg) if d.year != s and BT2.is_training_month(cfg, d.year, d.month)]
    nr = cfg["pixels"]["ring"].get("max_pixels") or cfg["pixels"]["ring"]["expected_pixels"]
    npx = 47 + ((nr if n_ring is None else n_ring) if ring else 0)
    return len(days) * 48 * npx


def inner_folds(cfg: dict, s: int) -> int:
    from src.gold import build_table_v2 as BT2
    return len({(d.year, d.month) for d in BT2.build_days(cfg) if d.year != s and BT2.is_training_month(cfg, d.year,
                                                                                                         d.month)})


def fit_work(cfg: dict, family: str, thr: float, rows: int, nfold: int) -> float:
    """Expected boosting work (rows x rounds, 1e9) of one fit: inner folds run (optimum + early stopping) rounds on
    ~INNER_ROWS_SHARE of the rows, the refit runs the median; rounds scale with 0.03 / learning rate (D190)."""
    mean_best, med = ROUNDS_PRIOR_LR003[(family, float(thr))]
    sc = 0.03 / float(cfg["lightgbm"]["learning_rate"])
    es, cap = int(cfg["inner_cv"]["early_stopping_rounds"]), int(cfg["inner_cv"]["max_rounds"])
    per_inner = min(mean_best * sc + es, cap)
    return rows * (nfold * INNER_ROWS_SHARE * per_inner + min(med * sc, cap)) / 1e9


def fit_minutes(cfg: dict, family: str, thr: float, rows: int, nfold: int) -> float:
    cm = cfg["planning"]["cost_model"][family]
    return (float(cm["a_s"]) + float(cm["c_s_per_grow"]) * fit_work(cfg, family, thr, rows, nfold)) / 60


def spec_minutes(cfg: dict, spec: str, s: int, thr: float, n_ring: int | None = None) -> float:
    dd = {"m0": {"ring": False}, "m1": {"ring": True}, "m1ts": {"ring": True, "two_stage": True},
          "v1b": {"ring": False}, "m1b": {"ring": True}, "wout": {"ring": True}, "rnear": {"ring": True}}.get(spec) \
        or dataset_def(spec)
    fam = _family(dd)
    rows = training_rows(cfg, s, dd["ring"], n_ring)
    nf = inner_folds(cfg, s)
    if dd.get("ds_p"):
        rows = int(rows * (1 - DRY_SHARE_PRIOR * (1 - dd["ds_p"])))
    if dd.get("two_stage"):
        return fit_minutes(cfg, fam, 0.5, rows, nf) + fit_minutes(cfg, fam, float(thr), int(rows *
                                                                  RAIN_SCENE_SHARE_PRIOR), nf)
    return fit_minutes(cfg, fam, float(thr), rows, nf)


def unit_minutes(cfg: dict, us: list) -> dict:
    gm = cfg["planning"]["gate_minutes"]
    out = {}
    for u in us:
        if u.kind == "fit":
            out[u.id] = spec_minutes(cfg, u.params["spec"], u.params["season"], u.params["thr"])
        elif u.kind == "feat":
            out[u.id] = float(gm["feat"])
        elif u.kind == "base":
            out[u.id] = float(gm["base"])
        else:
            out[u.id] = float(gm["done"] if u.id == "compare-done" else gm[u.id])
    return out


def lpt(minutes: list[tuple[str, float]], n: int) -> list[float]:
    """Longest-processing-time-first assignment of work units to n parts; returns the load of each part."""
    load = [0.0] * n
    for _, m in sorted(minutes, key=lambda x: -x[1]):
        load[int(np.argmin(load))] += m
    return load


def plan_estimate(cfg: dict, us: list, n: int, um: dict | None = None) -> dict:
    """Per-part minutes when n parts run together: every part waits for the opening gates (preflight, g2), runs its
    share of the work units (feat, base, fit), and the part that finishes last also runs the closing gates; a part
    with nothing left exits (no idle waiting, unlike P4D). Runner minutes = sum of the part walls."""
    um = um or unit_minutes(cfg, us)
    open_g = sum(um[u.id] for u in us if u.kind == "gate" and u.stage <= 2)
    close_g = sum(um[u.id] for u in us if u.kind == "gate" and u.stage > 2)
    feat = [(u.id, um[u.id]) for u in us if u.kind == "feat"]
    work = [(u.id, um[u.id]) for u in us if u.kind in ("fit", "base")]
    fl, wl = lpt(feat, n), lpt(work, n)
    walls = [open_g + f + w for f, w in zip(sorted(fl), sorted(wl, reverse=True))]
    k = int(np.argmax(walls))
    walls[k] += close_g
    walls = [round(x, 1) for x in walls]
    pl = cfg["planning"]
    return {"parts": n, "part_minutes": walls, "max_part_minutes": max(walls), "runner_minutes": round(sum(walls), 0),
            "within_target": max(walls) <= float(pl["part_target_minutes"]),
            "unit_minutes_total": round(sum(um.values()), 1)}


def choose_parts(cfg: dict, us: list, um: dict | None = None) -> dict:
    tgt = float(cfg["planning"]["part_target_minutes"])
    for n in range(1, 61):
        e = plan_estimate(cfg, us, n, um)
        if e["max_part_minutes"] <= tgt:
            return e
    raise ValueError("no part count meets planning.part_target_minutes")


def full_run_plan(cfg: dict, run: str, ds_p: float | None = None, n_ring_near: int = 73) -> tuple[list, dict]:
    """Planning-only unit lists of the later P4E runs (not executable yet): 'A' = P4E.1 (full Model 1 and the
    downsampled Model 1 at p in p_grid, 2.5 / 7.6, LOSO 3 folds); 'B' = P4E.2-P4E.4 (+ M0 / Model 1 references at
    every threshold, R-near, W-out, two-stage, Model 1-B and v1 (B) on folds 2024 / 2025). ds_p: downsampling on.
    Set-B NWP for the ring is NOT included (unknown, P4E.4 stop condition); R-near ring size is a placeholder."""
    U = PB.Unit
    from src.gold import build_table_v2 as BT2
    us = [U("preflight", "gate", 0)] + [U(f"feat-{y}-{m:02d}", "feat", 1, ("preflight",), {})
                                        for y, m in BT2.build_months(cfg)] + [U("g2", "gate", 2)]
    um = {"preflight": 1.0, "g2": float(cfg["planning"]["gate_minutes"]["g2"])}
    um |= {u.id: float(cfg["planning"]["gate_minutes"]["feat"]) for u in us if u.kind == "feat"}
    S, Ls = seasons(cfg), leads(cfg)

    def add(spec, ss, thrs, m_spec=None, n_ring=None):
        for s in ss:
            for L in Ls:
                for t in thrs:
                    uid = fit_id(spec, s, L, t)
                    us.append(U(uid, "fit", 3, ("g2",), {"spec": spec, "season": s, "lead": L, "thr": t}))
                    um[uid] = spec_minutes(cfg, m_spec or spec, s, t, n_ring)
    for s in S:
        for L in Ls:
            us.append(U(PD.base_id(s, L), "base", 3, ("g2",)))
            um[PD.base_id(s, L)] = float(cfg["planning"]["gate_minutes"]["base"])
    pt, hv = primary(cfg), heavy(cfg)
    if run == "A":
        add("m1", S, [pt, hv])
        for p in cfg["downsampling"]["p_grid"]:
            add(f"m1ds{int(round(100 * p)):02d}", S, [pt, hv])
    elif run == "B":
        suf = f"ds{int(round(100 * ds_p)):02d}" if ds_p else ""
        sp = (lambda b: f"{b}{suf}") if ds_p else (lambda b: b)
        for spec, ring, thrs, nr in (("m0", False, thresholds(cfg), None), ("m1", True, thresholds(cfg), None),
                                     ("wout", True, [pt, hv], None), ("rnear", True, [pt, hv], n_ring_near)):
            for s in S:
                for L in Ls:
                    for t in thrs:
                        uid = fit_id(sp(spec), s, L, t)
                        us.append(U(uid, "fit", 3, ("g2",), {}))
                        m = spec_minutes(cfg, spec, s, t, nr)
                        um[uid] = m * ((1 - DRY_SHARE_PRIOR * (1 - ds_p)) if ds_p and ring else 1.0)
        for s in S:
            for L in Ls:
                uid = fit_id(sp("m1ts"), s, L, hv)
                us.append(U(uid, "fit", 3, ("g2",), {}))
                um[uid] = spec_minutes(cfg, "m1ts", s, hv) * ((1 - DRY_SHARE_PRIOR * (1 - ds_p)) if ds_p else 1.0)
        for spec, ring in (("m1B", True), ("v1B", False)):
            for s in (2024, 2025):
                for L in Ls:
                    for t in thresholds(cfg):
                        uid = fit_id(sp(spec), s, L, t)
                        us.append(U(uid, "fit", 3, ("g2",), {}))
                        m = spec_minutes(cfg, "m1" if ring else "m0", s, t) * 1.15     # ~15 % more columns (set B)
                        um[uid] = m * ((1 - DRY_SHARE_PRIOR * (1 - ds_p)) if ds_p and ring else 1.0)
    else:
        raise ValueError(run)
    for gid_, mins in (("cmp", float(cfg["planning"]["gate_minutes"]["cmp"]) * (2 if run == "B" else 1)),
                       ("compare-done", float(cfg["planning"]["gate_minutes"]["done"]))):
        us.append(U(gid_, "gate", 4))
        um[gid_] = mins
    return us, um
