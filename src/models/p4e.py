"""P4E — v2 challenger (Model 1 = M0 + buffer ring), units of the cloud jobs `p4e_smoke` and `p4e` (driver:
src/models/cloud_train.py, main_p4e). TASK_P4E_MODEL_V2_CHALLENGER.md; configs/model_p4e.yaml merged on top of
configs/model_v2.yaml. Not a flood model; labels are IMERG (satellite estimate), not rain gauges. v1 stays official.

P4E.0 content:
  * every shared output has exactly one writer (cloud_train.commit_unit_single, D188 fix);
  * one learning rate / round cap for all P4E models (lightgbm.learning_rate, inner_cv.max_rounds, D190) and the
    pre-registered rounds check (D189): M0 / Model 1 median rounds <= 0.6 x max_rounds, else the comparison stops;
  * cost model from the P4D logs (planning.cost_model, D191) used for the per-part runtime estimate.
D196 (dry-scene definition, downsampling) is kept but UNUSED: D198 turns downsampling off for every P4E run.

D198 (user, 2026-11-23) — job p4e (full data) = P4E.2 + P4E.3 + P4E.4 + P4E.6; p4e_smoke runs every code path of it:
  0 preflight            gate (P4D preflight: R2 budget, ring geometry)
  1 feat-YYYY-MM         Gold table_v2 month; REUSED when part.parquet + part.meta.json exist for the gold id
    nwp-YYYY-MM          set B (P4E.4): NWP columns for every table_v2 row (city + ring) from the existing 81
                         Open-Meteo points (option A, v1 code path features.nwp_point_map / compute_nwp_features),
                         written to lake/v1/gold/met_training/table_v2_nwp/gid=<gid>/nid=<id>/ (reused if present)
  2 g2                   gate (P4D G2 + set-B inner-CV label check and NWP NaN share)
  3 base-s<fold>-lead<L> evaluation rows (P4D)
    fit-<spec>-...       configs: fits. m0, m1 (Model 1 (A)), rnear (R-near: 90 ring pixels), wout (W-out: ring
                         weight 0.5), m1ts (two-stage 7.6), v1b (v1 (B): 47 px + NWP), m1b (Model 1-B), m1bts
    dry-full-YYYY-MM     D198: dry-scene share (scenes, rows) per month and frame of the full Gold (read only)
  4 cmp-diag             gate P4E.2: M0 / Model 1 / R-near / W-out, pre-registered propose rule, event diagnosis
    cmp-2stage           gate P4E.3: two-stage vs nested 7.6, pre-registered use rule, violation rate
  5 cmp                  gate P4E.4 + P4E.6: v1 (B) / Model 1-B / Model 1 (A) / persistence, comparison table,
                         pre-registered shadow gate, rounds check, cost actual vs planned, dry-share finding
  6 figs                 gate: P4E.6 figures (src/models/p4e_figs.py), PNG + CSV under reports/p4e/figs/
  7 compare-done         gate: compare_summary.json; its winner writes summary.json and COMPARE_DONE.json
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import re
import time
from datetime import date, timedelta
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
NWP_REMOTE = LAKE + "gold/met_training/table_v2_nwp/"            # set-B columns (D198), inside table_v2*/
NWP_CODE = ["src/gold/features.py"]
OUT = "reports/p4e"
NOT_FLOOD = PD.NOT_FLOOD
CODE_FILES = PD.CODE_FILES + ["src/models/p4e.py", "src/models/p4e_figs.py", "configs/model_p4e.yaml"]
# D189: sha256 of the pre-registered sections (JSON, sort_keys) written to DECISIONS before any P4E training code
PREREG_SHA = "58eb5a7ee66974f209a9ad0e361fbd9543bce46da79db48a7646fef6e47022b2"
PREREG_SECTIONS = {"rounds_check": ("rounds_check",), "equivalence": ("downsampling", "equivalence"),
                   "propose_rule": ("diagnostics", "propose_rule"), "use_rule": ("two_stage", "use_rule"),
                   "shadow_gate": ("shadow_gate",), "planning": ("planning",)}
DRILL_GATE = "cmp"
LAST_OPENING_STAGE = 2           # preflight, feat, nwp, g2: every part waits for them; later gates close the run
# rounds priors from the 216 P4D fits (lr 0.03, early stopping 150; weight settings with alpha/beta > 0 excluded):
# (mean optimal inner-fold rounds, median chosen rounds) per pixel family and threshold (D191)
ROUNDS_PRIOR_LR003 = {("city", 0.5): (227, 177), ("city", 2.5): (216, 161), ("city", 7.6): (202, 110),
                      ("ring", 0.5): (376, 283), ("ring", 2.5): (312, 195), ("ring", 7.6): (261, 149)}
INNER_ROWS_SHARE = 0.907         # mean inner-fold training rows / fit training rows (P4D)
DRY_SHARE_PRIOR = 0.30           # planning only (D196 code path, unused since D198)
RAIN_SCENE_SHARE_PRIOR = 0.35    # planning only; rows in R (label >= 0.5 at the pixel or a neighbour)
SETB_COST_FACTOR = 1.18          # planning only: 16 NWP columns on top of ~89 set-A columns (per-round cost)
# display-only minutes of units added after the pre-registration (planning.gate_minutes is pre-registered and is
# used as a floor: max(pre-registered, value below); D199)
EXTRA_UNIT_MINUTES = {"dry": 3.0, "nwp": 4.0, "figs": 10.0, "cmp-diag": 15.0, "cmp-2stage": 8.0, "cmp": 35.0}
COST_RATIO_STOP = 1.5            # D196: actual / planned minutes of any unit outside [1/1.5, 1.5] -> flagged
PART_MARGIN = 0.8                # D198: proposed part count keeps the slowest part <= 0.8 x part_target_minutes
# model names of the comparison gates -> spec; nested 2.5 spec for the 7.6 alarm (two-stage: own cut, not nested)
MODEL_SPECS = {"M0": "m0", "Model1": "m1", "R-near": "rnear", "W-out": "wout", "Model1-twostage": "m1ts",
               "v1B": "v1b", "Model1-B": "m1b", "Model1-B-twostage": "m1bts"}
NESTED = {"m0": "m0", "m1": "m1", "rnear": "rnear", "wout": "wout", "v1b": "v1b", "m1b": "m1b"}


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


# ============================================================================ fit plan (configs: fits, D198)
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
    if spec == "rnear":
        return {"ring": True, "ring_subset": "r_near"}
    if spec == "wout":
        return {"ring": True, "w_out": True}
    if spec == "v1b":
        return {"ring": False, "set_b": True}
    if spec == "m1b":
        return {"ring": True, "set_b": True}
    if spec == "m1bts":
        return {"ring": True, "set_b": True, "two_stage": True}
    raise ValueError(spec)


def fit_plan(cfg: dict) -> list[tuple[str, int, float]]:
    """(spec, held-out fold, threshold) of every fit (each frame of frames.lead_min), in configuration order."""
    out, S = [], set(seasons(cfg))
    for e in cfg["fits"]:
        dataset_def(e["spec"])
        for s in e["folds"]:
            if int(s) not in S:
                raise ValueError(f"fits: fold {s} of {e['spec']} is not in evaluation.loso_seasons {sorted(S)}")
            for t in e["thr"]:
                if float(t) not in thresholds(cfg):
                    raise ValueError(f"fits: threshold {t} of {e['spec']} not in targets")
                k = (e["spec"], int(s), float(t))
                if k not in out:
                    out.append(k)
    return out


def fit_folds(cfg: dict, spec: str, thr: float) -> list[int]:
    return sorted({s for sp, s, t in fit_plan(cfg) if sp == spec and t == float(thr)})


def has_fit(cfg: dict, spec: str, thr: float, folds) -> bool:
    return bool(folds) and set(folds) <= set(fit_folds(cfg, spec, thr))


def uses_set_b(cfg: dict) -> bool:
    return any(dataset_def(sp).get("set_b") for sp, _, _ in fit_plan(cfg))


def nwp_from(cfg: dict) -> date:
    return pd.Timestamp(cfg["set_b"]["nwp_from"]).date()


def nwp_months(cfg: dict) -> list[tuple[int, int]]:
    from src.gold import build_table_v2 as BT2
    a = nwp_from(cfg)
    return [(y, m) for y, m in BT2.build_months(cfg) if date(y, m, 1) >= date(a.year, a.month, 1)]


def nwp_id(cfg: dict) -> str:
    """Identity of the set-B columns: Gold id, set_b settings (without the inner-CV block) and the feature code."""
    sb = {k: v for k, v in cfg["set_b"].items() if k != "inner_cv_block"}
    h = hashlib.sha256(json.dumps({"gid": gold_id(cfg), "set_b": sb}, sort_keys=True).encode())
    for f in NWP_CODE:
        h.update((CODE_ROOT / f).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:12]


def share_months(cfg: dict) -> list[tuple[int, int]]:
    from src.gold import build_table_v2 as BT2
    mo = cfg["downsampling"]["share_units"]["months"]
    if mo == "all":
        return BT2.build_months(cfg)
    return [(int(y), int(m)) for y, m in mo]


def _blocks(days, block: str):
    if block == "calendar_month":
        return sorted({(d.year, d.month) for d in days})
    if block == "utc_day":
        return sorted(set(days))
    raise ValueError(block)


def _block_bounds(b, block: str):
    if block == "calendar_month":
        a = date(b[0], b[1], 1)
        return a, date(b[0] + (b[1] == 12), b[1] % 12 + 1, 1)
    return b, b + timedelta(days=1)


def inner_cv_plan_check(cfg: dict) -> list[dict]:
    """P4D structural check (set A, variant plain) + the same check for every set-B fold: the training days of a
    set-B fit are the other seasons' training days on or after set_b.nwp_from; >= 2 usable blocks (training days
    outside the block and its 1-day buffer) and, for calendar-month blocks, >= 2 calendar months (D185)."""
    from src.gold import build_table_v2 as BT2
    with p4d_env():
        rows = PD.inner_cv_plan_check(cfg)
    if not uses_set_b(cfg):
        return rows
    days = BT2.build_days(cfg)
    buf = int(cfg["inner_cv"]["buffer_days"])
    blk = cfg["set_b"]["inner_cv_block"]
    folds = sorted({s for sp, s, _ in fit_plan(cfg) if dataset_def(sp).get("set_b")})
    for s in folds:
        tr = [d for d in days if d.year != s and BT2.is_training_month(cfg, d.year, d.month) and d >= nwp_from(cfg)]
        ub = []
        for b in _blocks(tr, blk):
            a, e = _block_bounds(b, blk)
            if [d for d in tr if d <= a - timedelta(days=buf + 1) or d >= e + timedelta(days=buf)]:
                ub.append(b)
        cal = sorted({(b[1] if blk == "calendar_month" else b.month) for b in ub})
        ok = len(ub) >= 2 and (len(cal) >= 2 if blk == "calendar_month" else True)
        rows.append({"heldout_season": s, "variant": "set_b", "block": blk, "inner_blocks": [str(b) for b in
                     _blocks(tr, blk)], "usable_blocks": [str(b) for b in ub], "calendar_months": cal, "ok": ok})
    return rows


def setb_label_check(cfg: dict, lab: pd.DataFrame) -> list[dict]:
    """Gate g2: for every set-B fit (fold, frame, threshold) the inner CV needs >= 2 usable blocks, i.e. held-out
    block with a positive city row AND training rows (other blocks, 1-day buffer) with a positive city row.
    lab: city rows with season, t0 (UTC), lead_min, train_month and y_* columns."""
    buf = pd.Timedelta(days=int(cfg["inner_cv"]["buffer_days"]))
    blk = cfg["set_b"]["inner_cv_block"]
    t = lab["t0"].dt.tz_convert("UTC")
    key = ((t.dt.year * 100 + t.dt.month) if blk == "calendar_month" else
           (t.dt.year * 10000 + t.dt.month * 100 + t.dt.day)).to_numpy()
    nb = (lab["t0"] >= pd.Timestamp(nwp_from(cfg), tz="UTC")).to_numpy()
    rows = []
    for sp, s, thr in fit_plan(cfg):
        if not dataset_def(sp).get("set_b"):
            continue
        trainable = (lab["season"].to_numpy() != s) & lab["train_month"].to_numpy(bool) & nb
        for L in leads(cfg):
            lm = lab["lead_min"].to_numpy() == L
            y = lab[thr_col(thr)].to_numpy() > 0
            usable = []
            for k in sorted(set(key[trainable & lm].tolist())):
                a, b = _key_bounds(k, blk)
                ho = trainable & lm & (key == k)
                tr = trainable & lm & (key != k) & ((t < a - buf) | (t >= b + buf)).to_numpy()
                if y[ho].any() and y[tr].any():
                    usable.append(int(k))
            rows.append({"spec": sp, "heldout_season": s, "lead_min": L, "threshold": thr, "block": blk,
                         "usable_blocks": usable, "ok": len(usable) >= 2})
    return rows


def _key_bounds(k: int, block: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    if block == "calendar_month":
        return PD._month_bounds(int(k))
    a = pd.Timestamp(year=k // 10000, month=(k // 100) % 100, day=k % 100, tz="UTC")
    return a, a + pd.Timedelta(days=1)


# ============================================================================ units
def fit_id(spec: str, s: int, lead: int, thr: float) -> str:
    return PD.fit_id(spec, s, lead, thr)


def dry_unit_id(y: int, m: int) -> str:
    return f"dry-full-{int(y)}-{int(m):02d}"


def dry_full_id(cfg: dict) -> str:
    fm = cfg["downsampling"]["full_month_check"]
    return dry_unit_id(fm["season"], fm["month"])


def units(cfg: dict) -> list:
    from src.gold import build_table_v2 as BT2
    U = PB.Unit
    u = [U("preflight", "gate", 0)]
    feat = [U(f"feat-{y}-{m:02d}", "feat", 1, ("preflight",), {"y": y, "m": m}) for y, m in BT2.build_months(cfg)]
    nwp = [U(f"nwp-{y}-{m:02d}", "nwp", 1, (f"feat-{y}-{m:02d}",), {"y": y, "m": m})
           for y, m in nwp_months(cfg)] if uses_set_b(cfg) else []
    u += feat + nwp
    u.append(U("g2", "gate", 2, tuple(x.id for x in feat + nwp)))
    base, fits = [], []
    for L in leads(cfg):
        for s in seasons(cfg):
            base.append(U(PD.base_id(s, L), "base", 3, ("g2",), {"season": s, "lead": L}))
        for sp, s, t in fit_plan(cfg):
            fits.append(U(fit_id(sp, s, L, t), "fit", 3, ("g2",), {"spec": sp, "season": s, "lead": L, "thr": t}))
    gid = str(cfg["downsampling"]["share_units"]["gold_id"])
    dry = [U(dry_unit_id(y, m), "dry", 3, ("preflight",), {"y": y, "m": m, "gid": gid}) for y, m in share_months(cfg)]
    u += base + fits + dry
    rounds = [f.id for f in fits if f.params["spec"] in cfg["rounds_check"]["specs"]]
    bid = tuple(b.id for b in base)

    def deps(specs):
        return tuple(dict.fromkeys(rounds + [f.id for f in fits if f.params["spec"] in specs])) + bid
    u.append(U("cmp-diag", "gate", 4, deps(("m0", "m1", "rnear", "wout"))))
    u.append(U("cmp-2stage", "gate", 4, deps(("m1", "m1ts", "m1b", "m1bts"))))
    u.append(U("cmp", "gate", 5, tuple(f.id for f in fits) + bid + tuple(d.id for d in dry) +
               ("cmp-diag", "cmp-2stage")))
    u.append(U("figs", "gate", 6, ("cmp",)))
    u.append(U("compare-done", "gate", 7, ("figs",)))
    return u


def _work_sort_key(u) -> tuple:
    """Order of a part's work units: base rows, dry share, then fits grouped by frame and feature set so that
    consecutive fits reuse the loaded frame (DataCache)."""
    if u.kind == "base":
        return (0, u.params["lead"], 0, 0, "", 0, 0.0)
    if u.kind == "dry":
        return (1, 0, 0, 0, u.id, 0, 0.0)
    dd = dataset_def(u.params["spec"])
    return (2, u.params["lead"], int(bool(dd.get("set_b"))), int(dd["ring"]), u.params["spec"], u.params["season"],
            u.params["thr"])


def lpt_assign(minutes: list[tuple[str, float]], n: int) -> tuple[dict, list[float]]:
    """Deterministic longest-processing-time-first assignment: (unit id -> part index 0..n-1, load per part)."""
    load = [0.0] * n
    out = {}
    for uid, m in sorted(minutes, key=lambda x: (-x[1], x[0])):
        k = int(np.argmin(load))
        out[uid] = k
        load[k] += m
    return out, load


def part_units(us: list, i: int, n: int, cfg: dict | None = None) -> list:
    """Work units of part i/n. With cfg (P4E driver): LPT on the planned minutes, separately for the opening work
    (feat, nwp) and the stage-3 work (base, fit, dry), the same assignment as plan_estimate; without cfg: the P4B
    round-robin."""
    if cfg is None:
        return PB.part_units(us, i, n)
    um = unit_minutes(cfg, us)
    out = []
    for kinds in (("feat", "nwp"), ("base", "fit", "dry")):
        work = [u for u in us if u.kind in kinds]
        a, _ = lpt_assign([(u.id, um[u.id]) for u in work], n)
        mine = [u for u in work if a[u.id] == i - 1]
        if kinds[0] == "feat":
            mine.sort(key=lambda u: (u.kind != "feat", u.id))
        else:
            mine.sort(key=_work_sort_key)
        out += mine
    return out


# ============================================================================ run context
class Ctx(PD.Ctx):
    """p4d.Ctx (table_v2 gold id and layout) for P4E, + the set-B NWP columns (table_v2_nwp)."""

    def nwp_prefix(self) -> str:
        return f"{NWP_REMOTE}gid={self.gid}/nid={nwp_id(self.cfg)}/"

    def nwp_key(self, y: int, m: int, name: str = "part.parquet") -> str:
        return f"{self.nwp_prefix()}season={y}/month={m:02d}/{name}"

    def nwp_local(self) -> Path:
        return (root() / "data" / "gold" / "met_training" / "v2" / "table_v2_nwp" / f"gid={self.gid}" /
                f"nid={nwp_id(self.cfg)}")

    def fetch_nwp(self) -> int:
        pre = self.nwp_prefix()
        n = 0
        for k, size in sorted(self.store.list(pre).items()):
            n += self._download(k, size, self.nwp_local() / k[len(pre):])
        return n


def write_json(p: Path, obj) -> None:
    PB.write_json(p, obj)


def rel(p: Path) -> str:
    return Path(p).resolve().relative_to(root().resolve()).as_posix()


def out_dir(*parts: str) -> Path:
    p = root().joinpath(OUT, *parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _local_csv(ctx, r: str) -> pd.DataFrame:
    if not (root() / r).is_file():
        ctx.fetch_run_files(r)
    return pd.read_csv(root() / r)


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


# ============================================================================ set B: NWP columns (P4E.4, D198)
def run_nwp(unit, ctx: Ctx) -> tuple[list[str], dict]:
    """NWP features of every table_v2 row of one month (city + ring), option A (D198): the existing 81 Open-Meteo
    points mapped to every pixel by features.nwp_point_map (nearest point + mean within 0.3 deg) and
    features.compute_nwp_features, i.e. the v1 set-B code path; frame-mode G2 check before writing. Stops if an
    Open-Meteo Silver day of the month is missing (the source would not cover the rows)."""
    from src.gold import build_table_B as TB
    from src.gold import build_table_v2 as BT2
    from src.gold import features as F
    from src.quality.temporal_checks import FrameConfig, check_frame_table
    cfg, y, m = ctx.cfg, unit.params["y"], unit.params["m"]
    keys = [ctx.nwp_key(y, m, n) for n in ("part.parquet", "part.meta.json")]
    have = ctx.store.list(ctx.nwp_key(y, m, ""))
    if all(k in have for k in keys):
        return [], {"reused": True, "nwp_keys": keys, "bytes": int(sum(have[k] for k in keys))}
    gk = ctx.gold_key(y, m)
    gl = ctx.store.list(gk)
    if gk not in gl:
        raise RuntimeError(f"{unit.id}: {gk} missing")
    gp = ctx.gold_local() / f"season={y}" / f"month={m:02d}" / "part.parquet"
    ctx._download(gk, gl[gk], gp)
    A = pd.read_parquet(gp, columns=["pixel_id", "t0", "lead_min", "prediction_time", "frame_start", "target_time"])
    days = [d for d in BT2.build_days(cfg) if (d.year, d.month) == (y, m)]
    dd = sorted({d + timedelta(days=k) for d in days for k in (-1, 0, 1)})
    rels = [PD._silver_rel("openmeteo", d) for d in dd]
    ctx.fetch_lake(rels, "data/silver/met/openmeteo/")
    missing = [d.isoformat() for d in days if not (root() / PD._silver_rel("openmeteo", d)).is_file()]
    if missing:
        raise RuntimeError(f"{unit.id}: Open-Meteo Silver missing for {missing[:5]} ({len(missing)} days): NWP "
                           "cannot cover these rows (TASK P4E.4 stop condition)")
    om = pd.concat([pd.read_parquet(root() / r, columns=TB.OM_COLS) for r in rels if (root() / r).is_file()],
                   ignore_index=True)
    for c in ("interval_start_utc", "forecast_issue_time_utc", "available_time_utc"):
        om[c] = om[c].astype("datetime64[ns, UTC]")
    pix = np.sort(A["pixel_id"].unique().astype("int64"))
    pmap = F.nwp_point_map(om, pix)
    parts = []
    for L in sorted(A["lead_min"].unique()):
        a = A[A["lead_min"] == L].reset_index(drop=True)
        rows = a[["pixel_id", "prediction_time", "frame_start"]].copy()
        for c in ("prediction_time", "frame_start"):
            rows[c] = rows[c].astype("datetime64[ns, UTC]")
        nw = F.compute_nwp_features(rows, om, pmap)
        for c in [c for c in nw.columns if c.endswith("_time")]:
            nw[c] = nw[c].astype("datetime64[us, UTC]")
        parts.append(pd.concat([a, nw], axis=1))
        del rows, nw
    B = pd.concat(parts, ignore_index=True)
    del parts
    ncols = [c for c in B.columns if c.startswith(cfg["set_b"]["feature_prefix"]) and "__" not in c]
    for c in ("t0", "prediction_time", "frame_start", "target_time"):
        B[c] = B[c].astype("datetime64[us, UTC]")
    # frame rules of v1 set B (configs/model_v1.yaml: same ADD-005 frames as table_v2, allowlist includes nwp_)
    v = check_frame_table(B, PB._groups(ncols), ncols, FrameConfig.from_yaml(PB.CFG_PATH))
    if v:
        raise RuntimeError(f"G2 violations in table_v2_nwp {y}-{m:02d}: {[str(x) for x in v[:3]]}")
    pts = om[["pixel_id", "lat", "lon"]].drop_duplicates("pixel_id")
    lat, lon = F.Grid().latlon(pix)
    dist = np.hypot(pts["lat"].to_numpy()[None, :] - lat[:, None], pts["lon"].to_numpy()[None, :] - lon[:, None]).min(1)
    nwin = np.array([len(pmap[int(p)]["within"]) for p in pix])
    lp = ctx.nwp_local() / f"season={y}" / f"month={m:02d}" / "part.parquet"
    PB.write_parquet(B.drop(columns=["prediction_time", "frame_start", "target_time"]), lp)
    nan = {c: round(float(B[c].isna().mean()), 5) for c in ncols}
    meta = {"table": cfg["set_b"]["table"], "gold_id": ctx.gid, "nwp_id": nwp_id(cfg), "season": y, "month": m,
            "rows": int(len(B)), "pixels": int(len(pix)), "features": ncols, "option": cfg["set_b"]["option"],
            "points": int(len(pts)), "nearest_point_deg": {"mean": float(dist.mean()), "max": float(dist.max())},
            "points_within_0p3": {"mean": float(nwin.mean()), "min": int(nwin.min())}, "nan_share": nan,
            "g2_violations": 0, "rule": "Open-Meteo previous_day1; forecast_issue_time <= prediction_time and "
            "available_time <= prediction_time (features.compute_nwp_features)", "note": NOT_FLOOD}
    write_json(lp.with_name("part.meta.json"), meta)
    for name, k in zip(("part.parquet", "part.meta.json"), keys):
        ctx.put_own(k, (lp.parent / name).read_bytes())
    return [], {"rows": int(len(B)), "pixels": int(len(pix)), "points": int(len(pts)),
                "nan_share_max": max(nan.values()) if nan else None, "nwp_keys": keys}


def nwp_feature_list(ctx: Ctx) -> list[str]:
    parts = sorted(ctx.nwp_local().glob("season=*/month=*/part.parquet"))
    if not parts:
        raise RuntimeError(f"no table_v2_nwp partition for gid {ctx.gid} nid {nwp_id(ctx.cfg)}")
    pre = ctx.cfg["set_b"]["feature_prefix"]
    return [c for c in PB._schema_names(parts[0]) if c.startswith(pre) and "__" not in c]


def load_frame(ctx: Ctx, lead: int, feats: list[str], months: list[tuple[int, int]] | None = None):
    """All table_v2 rows of one frame (only `months` if given) + the set-B NWP columns of `feats` joined on
    (pixel_id, t0); metadata frame + float32 feature matrix in the same row order (filled in place: RAM)."""
    from src.gold import build_table_v2 as BT2
    pre = ctx.cfg["set_b"]["feature_prefix"]
    fn = [c for c in feats if c.startswith(pre)]
    fa = [c for c in feats if not c.startswith(pre)]
    if not fn and months is None:
        with p4d_env():
            return PD.load_frame(ctx, lead, feats)
    cfg = ctx.cfg
    ycols = [thr_col(t) for t in thresholds(cfg)]
    with p4d_env():
        parts = [(y, m, p) for y, m, p in PD.gold_parts(ctx) if months is None or (y, m) in set(months)]
    if not parts:
        raise RuntimeError("load_frame: no table_v2 partition in the requested months")
    counts = [int((pd.read_parquet(p, columns=["lead_min"])["lead_min"] == lead).sum()) for _, _, p in parts]
    X = np.empty((sum(counts), len(feats)), "float32")
    metas, at = [], 0
    for (y, m, part), n in zip(parts, counts):
        df = pd.read_parquet(part, columns=list(dict.fromkeys(PD.META + ycols + fa)), filters=[("lead_min", "==", lead)])
        if fn:
            npth = ctx.nwp_local() / f"season={y}" / f"month={m:02d}" / "part.parquet"
            nw = pd.read_parquet(npth, columns=["pixel_id", "t0"] + fn, filters=[("lead_min", "==", lead)])
            df = df.merge(nw, on=["pixel_id", "t0"], how="left", validate="one_to_one", indicator=True)
            if (df["_merge"] != "both").any():
                raise RuntimeError(f"set B {y}-{m:02d} lead {lead}: {(df['_merge'] != 'both').sum()} rows without NWP")
            df = df.drop(columns="_merge")
        assert len(df) == n, (part, len(df), n)
        meta = df[PD.META + ycols].copy()
        meta["ym"] = np.int32(y * 100 + m)
        meta["train_month"] = BT2.is_training_month(cfg, y, m)
        X[at:at + n] = df[feats].to_numpy("float32")
        at += n
        metas.append(meta)
        del df
    meta = pd.concat(metas, ignore_index=True)
    t = meta["t0"].dt.tz_convert("UTC")
    meta["md"] = (t.dt.month * 100 + t.dt.day).astype("int16").to_numpy()
    meta["t0"] = t
    return meta, X


# ============================================================================ downsampling (D196; unused, D198)
def rain_columns(feats: list[str], cfg: dict) -> list[str]:
    pats = [re.compile(p) for p in cfg["downsampling"]["dry_columns_regex"]]
    return [c for c in feats if any(p.search(c) for p in pats)]


def motion_columns(cols: list[str], cfg: dict) -> list[str]:
    pats = [re.compile(p) for p in cfg["downsampling"].get("motion_columns_regex", [])]
    return [c for c in cols if any(p.search(c) for p in pats)]


def dry_rows(X: np.ndarray, feats: list[str], cfg: dict) -> tuple[np.ndarray, dict]:
    """Row-level dry-compatibility (D196): every RAW rain input window (rain-valued column that is not
    motion-derived) is non-NaN and exactly 0, and every motion-derived column (adv_*, upwind_*) is 0 or NaN. NaN in a
    motion column is accepted only through the first condition (all raw windows of the row are non-NaN zeros); NaN in
    any raw rain column makes the row not dry. Inputs only; labels are never read. Also returns row counts that
    explain the result (raw all zero, motion NaN among those)."""
    cols = rain_columns(feats, cfg)
    mot = motion_columns(cols, cfg)
    raw = [c for c in cols if c not in mot]
    if not raw:
        raise RuntimeError("no raw rain input column matched downsampling.dry_columns_regex")
    raw_zero = np.ones(len(X), bool)
    for c in raw:                                      # column by column: no copy of the feature matrix (RAM)
        raw_zero &= X[:, feats.index(c)] == 0          # NaN == 0 is False: missing raw input is never dry
    mot_ok = np.ones(len(X), bool)
    mot_nan = np.zeros(len(X), bool)
    for c in mot:
        v = X[:, feats.index(c)]
        nan = np.isnan(v)
        mot_ok &= (v == 0) | nan
        mot_nan |= nan
    row = raw_zero & mot_ok
    return row, {"raw_columns": len(raw), "motion_columns": len(mot), "rows": int(len(X)),
                 "rows_raw_all_zero": int(raw_zero.sum()), "rows_dry": int(row.sum()),
                 "rows_dry_with_motion_nan": int((row & mot_nan).sum())}


def dry_scene_rows(meta: pd.DataFrame, X: np.ndarray, feats: list[str], cfg: dict) -> np.ndarray:
    """True for every row whose time t0 is a dry scene: all rows of t0 in the loaded frame (city + ring pixels) are
    dry-compatible (dry_rows, D196). Inputs only; labels are never read."""
    row_dry, _ = dry_rows(X, feats, cfg)
    codes, uniq = pd.factorize(meta["t0"])
    scene = np.ones(len(uniq), bool)
    np.logical_and.at(scene, codes, row_dry)
    return scene[codes]


def dry_share_stats(meta: pd.DataFrame, X: np.ndarray, feats: list[str], cfg: dict) -> dict:
    """Dry-scene share of one loaded frame (D196 report): scenes (t0) and rows, with the row-level breakdown."""
    row_dry, st = dry_rows(X, feats, cfg)
    codes, uniq = pd.factorize(meta["t0"])
    scene = np.ones(len(uniq), bool)
    np.logical_and.at(scene, codes, row_dry)
    return {**st, "scenes": int(len(uniq)), "dry_scenes": int(scene.sum()),
            "dry_scene_share": float(scene.mean()) if len(uniq) else float("nan"),
            "dry_scene_row_share": float(scene[codes].mean()) if len(codes) else float("nan")}


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
    return drop, w, dry, {"p": p, "rain_columns": len(rain_columns(feats, cfg)),
                          "motion_columns": len(motion_columns(rain_columns(feats, cfg), cfg)),
                          "dry_rows": int(dry.sum()), "rows": int(len(dry))}


def run_dry_full(unit, ctx: Ctx) -> tuple[list[str], dict]:
    """D196 / D198 finding: dry-scene share (scenes and rows) of ONE full month of the full Gold table_v2 (read only),
    per frame, streamed one frame at a time. The D196 definition is unchanged; downsampling itself is off (D198)."""
    import pyarrow.parquet as pq
    from src.gold import build_table_v2 as BT2
    fm = ctx.cfg["downsampling"]["full_month_check"]
    y = int(unit.params.get("y", fm["season"]))
    m = int(unit.params.get("m", fm["month"]))
    gid = str(unit.params.get("gid", fm["gold_id"]))
    key = f"{GOLD_REMOTE}gid={gid}/season={y}/month={m:02d}/part.parquet"
    have = ctx.store.list(key)
    if key not in have:
        raise RuntimeError(f"{unit.id}: {key} not found (full Gold month for the dry-share check)")
    dst = root() / "data" / "_scratch" / "p4e" / "dry_full" / f"gid={gid}" / f"season={y}_month={m:02d}.parquet"
    ctx._download(key, have[key], dst)
    names = pq.read_schema(dst).names
    feats = BT2.feature_columns(names)
    cols = rain_columns(feats, ctx.cfg)
    lead_vals = sorted(int(v) for v in pd.read_parquet(dst, columns=["lead_min"])["lead_min"].unique())
    per = []
    for L in lead_vals:
        df = pd.read_parquet(dst, columns=["t0", "pixel_id", "is_city"] + cols, filters=[("lead_min", "==", L)])
        X = df[cols].to_numpy("float32")
        st = dry_share_stats(df[["t0"]], X, cols, ctx.cfg)
        city = df["is_city"].to_numpy("float32") == 1
        per.append({"lead_min": L, "pixels": int(df["pixel_id"].nunique()), "city_pixels":
                    int(df.loc[city, "pixel_id"].nunique()), **st})
        del df, X
    out = {"unit": unit.id, "gold_key": key, "gold_id": gid, "season": y, "month": m,
           "definition": "D196: raw rain windows non-NaN and exactly 0, motion-derived (adv_*, upwind_*) 0 or NaN, "
                         "every pixel of the scene (city + ring); inputs only. Finding only: downsampling is off (D198)",
           "rain_columns": cols, "motion_columns": motion_columns(cols, ctx.cfg), "per_frame": per, "note": NOT_FLOOD}
    p = out_dir("compare") / f"{unit.id}.json"
    write_json(p, out)
    return [rel(p)], {"dry_scene_share": [round(r["dry_scene_share"], 4) for r in per],
                      "scenes": [r["scenes"] for r in per]}


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


def r_near_pixels(meta: pd.DataFrame, cfg: dict) -> np.ndarray:
    """R-near (D194 / D198): ring pixels of the loaded frame with Chebyshev distance <= r_near_chebyshev_max to the
    nearest city pixel. Must equal the recorded diagnostics.r_near_pixels (count r_near_count), else the fit stops."""
    from src.gold import features as F
    dg = cfg["diagnostics"]
    pix = meta[["pixel_id", "is_city"]].drop_duplicates("pixel_id")
    city = pix.loc[pix["is_city"].to_numpy("float32") == 1, "pixel_id"].to_numpy("int64")
    ring = np.sort(pix.loc[pix["is_city"].to_numpy("float32") != 1, "pixel_id"].to_numpy("int64"))
    G = F.Grid()
    ci, cj = G.ij(city)
    ri, rj = G.ij(ring)
    d = np.maximum(np.abs(ri[:, None] - ci[None, :]), np.abs(rj[:, None] - cj[None, :])).min(1)
    got = ring[d <= int(dg["r_near_chebyshev_max"])]
    want = np.sort(np.asarray(dg["r_near_pixels"], "int64"))
    if len(got) != int(dg["r_near_count"]) or not np.array_equal(np.sort(got), want):
        raise RuntimeError(f"R-near: {len(got)} ring pixels within Chebyshev {dg['r_near_chebyshev_max']} of the city "
                           f"in the data != recorded list ({len(want)}, count {dg['r_near_count']}); stop (D198)")
    return want


# ============================================================================ fits
def _month_bounds(ym: int):
    return PD._month_bounds(ym)


def cv_fit(X, y, w, keep, inner_eval, meta, params, ic, feats, block: str = "calendar_month") -> dict:
    """Inner blocked CV (P4D design: calendar-month blocks; the smoke set-B fits use UTC-day blocks, D199): early
    stopping on the held-out block (inner_eval rows of that block: city, all days, never downsampled or weighted),
    median of the fold optima; returns the boosters for the inner OOF, the per-fold record and the boosting work
    (rows x rounds run)."""
    import lightgbm as lgb
    buf = pd.Timedelta(days=int(ic["buffer_days"]))
    t0 = meta["t0"]
    if block == "calendar_month":
        key = meta["ym"].to_numpy()
    else:
        t = t0.dt.tz_convert("UTC")
        key = (t.dt.year * 10000 + t.dt.month * 100 + t.dt.day).to_numpy()
    boosters, best, folds, work = [], [], [], 0
    cap, es = int(ic["max_rounds"]), int(ic["early_stopping_rounds"])
    for mo in sorted(set(key[inner_eval].tolist())):
        a, b = _key_bounds(int(mo), block)
        ho = inner_eval & (key == mo)
        tr = keep & (key != mo) & ((t0 < a - buf) | (t0 >= b + buf)).to_numpy()
        if ho.sum() == 0 or tr.sum() == 0 or y[tr].sum() == 0 or y[ho].sum() == 0:
            folds.append({"month": int(mo), "block": block, "train_rows": int(tr.sum()), "held_rows": int(ho.sum()),
                          "used": False})
            continue
        dtr = lgb.Dataset(X[tr], y[tr], weight=None if w is None else w[tr], feature_name=feats, free_raw_data=True)
        dva = lgb.Dataset(X[ho], y[ho], reference=dtr)
        bst = lgb.train(params, dtr, num_boost_round=cap, valid_sets=[dva], valid_names=["held"],
                        callbacks=[lgb.early_stopping(es, verbose=False)])
        bi = int(bst.best_iteration or bst.current_iteration())
        work += int(tr.sum()) * int(bst.current_iteration())
        boosters.append((mo, ho, bst))
        best.append(bi)
        folds.append({"month": int(mo), "block": block, "train_rows": int(tr.sum()), "held_rows": int(ho.sum()),
                      "train_pos": int(y[tr].sum()), "best_iteration": bi, "rounds_run": int(bst.current_iteration()),
                      "at_cap": bi >= cap - es, "used": True})
        del dtr, dva
    if not boosters:
        raise RuntimeError("no usable inner fold")
    med = max(int(np.median(best)), 1)
    return {"boosters": boosters, "best": best, "folds": folds, "median": med, "work": work}


def inner_oof(cv: dict, X) -> dict:
    """{block: (held mask, prediction at the median rounds)}."""
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
    from src.gold import p4a_common as P
    from src.models import metrics as M
    cfg = ctx.cfg
    spec, s, L, thr = unit.params["spec"], unit.params["season"], unit.params["lead"], unit.params["thr"]
    dd = dataset_def(spec)
    setb = bool(dd.get("set_b"))
    t1 = time.time()
    with p4d_env():
        ctx.fetch_gold()
        feats = PD.feature_list(ctx, dd["ring"])
    months = None
    if setb:
        ctx.fetch_nwp()
        feats = feats + nwp_feature_list(ctx)
        months = nwp_months(cfg)
    meta, X = ctx.cache.get((L, tuple(feats), bool(months)), lambda: load_frame(ctx, L, feats, months))
    y = meta[thr_col(thr)].to_numpy("int8")
    city = meta["is_city"].to_numpy("float32") == 1
    trainable = (meta["season"].to_numpy() != s) & meta["train_month"].to_numpy(bool)
    if setb:
        trainable &= (meta["t0"] >= pd.Timestamp(nwp_from(cfg), tz="UTC")).to_numpy()
    ring_rows = ~city if dd["ring"] else np.zeros(len(city), bool)
    rn = None
    if dd.get("ring_subset") == "r_near":
        rn = r_near_pixels(meta, cfg)
        ring_rows &= meta["pixel_id"].isin(rn).to_numpy()
    keep = trainable & (city | ring_rows)
    held = (meta["season"].to_numpy() == s) & city
    inner_eval = trainable & city
    params, ic = lgb_params(cfg), cfg["inner_cv"]
    block = cfg["set_b"]["inner_cv_block"] if setb else "calendar_month"
    w, ds = None, None
    if dd.get("w_out"):
        w = np.where(city, 1.0, float(cfg["diagnostics"]["w_out_ring_weight"])).astype("float32")
    if dd.get("ds_p"):
        drop, w, dry, ds = downsample(meta, X, feats, cfg, dd["ds_p"])
        ds |= {"training_rows_full": int(keep.sum()), "dry_training_rows": int((keep & dry).sum())}
        keep = keep & ~drop
        ds |= {"training_rows_kept": int(keep.sum()), "rows_saved": ds["training_rows_full"] - int(keep.sum()),
               "dry_share_training": ds["dry_training_rows"] / max(ds["training_rows_full"], 1)}
    info = {"unit": unit.id, "spec": spec, "dataset": dd, "heldout_season": s, "lead_min": L, "threshold": thr,
            "labels": "training: IMERG Final V07 rows only; held-out: IMERG Final (Late for 2025-10, secondary)",
            "features": feats, "n_features": len(feats), "lgb_params": params,
            "inner_cv": {k: ic[k] for k in ("max_rounds", "early_stopping_rounds", "buffer_days")} | {"block": block},
            "downsampling": ds, "held_rows": int(held.sum()), "note": NOT_FLOOD}
    if setb:
        info["set_b"] = {"nwp_id": nwp_id(cfg), "nwp_from": str(nwp_from(cfg)), "option": cfg["set_b"]["option"],
                         "nwp_features": [c for c in feats if c.startswith(cfg["set_b"]["feature_prefix"])]}
    if rn is not None:
        info["r_near"] = {"pixels": int(len(rn)), "chebyshev_max": int(cfg["diagnostics"]["r_near_chebyshev_max"])}
    if w is not None and dd.get("w_out"):
        info["ring_weight"] = float(cfg["diagnostics"]["w_out_ring_weight"])
    if dd.get("two_stage"):
        rs = rain_scene(meta, cfg)
        y1 = rs.astype("int8")
        cv1 = cv_fit(X, y1, w, keep, inner_eval, meta, params, ic, feats, block)
        cv2 = cv_fit(X, y, w, keep & rs, inner_eval & rs, meta, params, ic, feats, block)
        o1 = inner_oof(cv1, X)
        iy, ip = [], []
        for mo, (ho, p1) in o1.items():                 # product on every held-out city row of the block
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
        cv = cv_fit(X, y, w, keep, inner_eval, meta, params, ic, feats, block)
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


# ============================================================================ pooled OOF access (gates)
def pooled(ctx, spec: str, L: int, thr: float, roles=("eval",), folds=None) -> pd.DataFrame:
    """p4d.pooled restricted to `folds` (default: every fold this spec was fitted on at thr); the 7.6 alarm of a
    non-two-stage spec is nested with its own 2.5 alarm (heavy_rule nested, as v1)."""
    cfg = ctx.cfg
    folds = list(folds) if folds is not None else fit_folds(cfg, spec, thr)
    nested = NESTED.get(spec) if (thr == heavy(cfg) and cfg["evaluation"]["heavy_rule"] == "nested") else None
    if nested and not has_fit(cfg, nested, primary(cfg), folds):
        raise RuntimeError(f"{spec} 7.6 on folds {folds}: nested rule needs {nested} at 2.5 on the same folds")
    out = []
    with p4d_env():
        for s in folds:
            b = PD.base_rows(ctx, s, L)
            b = b[b["role"].isin(roles)]
            o, info = PD.fold_pred(ctx, spec, s, L, thr)
            j = b.merge(o, on=["pixel_id", "t0"], how="inner", validate="one_to_one")
            if len(j) != len(b):
                raise RuntimeError(f"{spec} s{s} lead{L} thr{thr:g}: {len(b) - len(j)} base rows without prediction")
            j["y"] = j[thr_col(thr)].astype(bool)
            j["p_clim"] = j[f"pclim_{kstr(thr)}"].astype("float64")
            j["p_raw"] = j["p_raw"].astype("float64")
            j["p_cal"] = PB.apply_calibrator(info["platt"], j["p_raw"].to_numpy())
            j["yhat"] = j["p_cal"].to_numpy() >= info["cut"]["cut"]
            j["season_fold"] = s
            if nested:
                o2, i2 = PD.fold_pred(ctx, nested, s, L, primary(cfg))
                q = j[["pixel_id", "t0"]].merge(o2, on=["pixel_id", "t0"], how="left")
                pq = PB.apply_calibrator(i2["platt"], q["p_raw"].to_numpy("float64"))
                j["yhat"] = j["yhat"].to_numpy() & (np.nan_to_num(pq, nan=-1.0) >= i2["cut"]["cut"])
            out.append(j)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _frames(ctx, names: dict, L: int, thr: float, roles, folds) -> dict:
    """{model name: pooled frame} for the names whose spec (names[name], or MODEL_SPECS) has fits on `folds`."""
    out = {}
    for n, sp in names.items():
        if has_fit(ctx.cfg, sp, thr, folds):
            j = pooled(ctx, sp, L, thr, roles, folds)
            if len(j):
                out[n] = j
    return out


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


def metric_ci(ctx, j: pd.DataFrame, thr: float) -> dict:
    from src.models import metrics as M
    ev, cfg = ctx.cfg["evaluation"], ctx.cfg
    ds = M.day_stats(j["day"].to_numpy(), j["y"], j["yhat"], j["p_cal"], j["p_clim"])
    ci = M.bootstrap_ci(ds, ev["bootstrap_resamples"], cfg["random_seed"], ev["ci_level"])
    return {**PD.metric_row(j, thr, ev["reliability_bins"]), "csi_lo": ci["csi"][0], "csi_hi": ci["csi"][1],
            "bss_lo": ci["bss_clim"][0], "bss_hi": ci["bss_clim"][1]}


def _cat(j: pd.DataFrame, yhat: np.ndarray, p: np.ndarray) -> dict:
    from src.models import metrics as M
    y = j["y"].to_numpy(bool)
    sc = M.scores(**M.contingency(y, yhat))
    return {"n": int(len(j)), "positives": int(y.sum()), "base_rate": float(y.mean()) if len(j) else np.nan,
            "csi": float(sc["csi"]), "pod": float(sc["pod"]), "far": float(sc["far"]), "bias": float(sc["bias"]),
            "brier": M.brier(y, p) if len(j) else np.nan}


def event_day_sets(ctx, folds) -> dict:
    """Named events of evaluation.events (local days) and the wettest days (evaluation.wettest_days, city-mean label at
    the first frame, eval role) of every fold in `folds` (P4D definition). Descriptive only (hub 6)."""
    cfg = ctx.cfg
    out = {n: {d.strftime("%Y-%m-%d") for d in pd.date_range(a, b, freq="D")}
           for n, (a, b) in cfg["evaluation"]["events"].items()}
    with p4d_env():
        for s in folds:
            lab = PD.base_rows(ctx, s, leads(cfg)[0])
            lab = lab[lab["role"] == "eval"].groupby("day")["label_rate_mm_h"].mean()
            out[f"wettest{s}"] = set(lab.sort_values(ascending=False).index[:int(cfg["evaluation"]["wettest_days"])])
    return out


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


def require_rounds(ctx) -> dict:
    rc = rounds_check(ctx)
    write_json(out_dir("compare") / "rounds_check.json", rc)
    if not rc["ok"]:
        raise RuntimeError(f"rounds check FAIL (pre-registered D189): {[r for r in rc['units'] if not r['ok']]}; "
                           "stop and ask before any comparison")
    return rc


def equivalence_verdict(rows: list[dict], eq: dict) -> dict:
    """Pre-registered P4E.1 rule (D189) on (downsampled - full) differences, one row per frame with keys lead_min,
    logloss_lo, logloss_hi, bss_clim_diff, csi_diff, ece_cal_diff (2.5) and csi76_diff (7.6, may be None).
    Not applied since D198 (P4E.1 not run); kept with its tests."""
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


def ci_rule(diff: dict, upper_below: float) -> dict:
    """Pre-registered propose / use rules (D189): the upper 90 % CI bound of (candidate - reference) < upper_below."""
    return {"diff": diff["diff"], "lo": diff["lo"], "hi": diff["hi"], "days": diff.get("days"),
            "upper_below": upper_below, "pass": bool(diff["hi"] < upper_below)}


def _ll_pooled_frames(ctx, fa: list[pd.DataFrame], fb: list[pd.DataFrame]) -> dict:
    """Day-block bootstrap of the log-loss difference A - B, day sums pooled over frames (as sel-m1a/m1b of P4D)."""
    ev = ctx.cfg["evaluation"]
    return PD.ll_paired(PD._sum_days([PD.ll_day(j) for j in fa]), PD._sum_days([PD.ll_day(j) for j in fb]),
                        ev["bootstrap_resamples"], ctx.cfg["random_seed"], ev["ci_level"])


def _diag_folds(cfg: dict) -> list[int]:
    pt = primary(cfg)
    return sorted(set(fit_folds(cfg, "m0", pt)) & set(fit_folds(cfg, "m1", pt)))


def _b_folds(cfg: dict) -> list[int]:
    pt = primary(cfg)
    return sorted(set(fit_folds(cfg, "v1b", pt)) & set(fit_folds(cfg, "m1b", pt)))


# ============================================================================ P4E.2 gate (diagnosis)
def gate_cmp_diag(ctx: Ctx) -> tuple[list[str], dict]:
    from src.models import metrics as M
    cfg, ev = ctx.cfg, ctx.cfg["evaluation"]
    require_rounds(ctx)
    pt, hv = primary(cfg), heavy(cfg)
    folds = _diag_folds(cfg)
    names = {n: MODEL_SPECS[n] for n in ("M0", "Model1", "R-near", "W-out")}
    met, diffs, dist, hist, relr = [], [], [], [], []
    ll = {n: [] for n in names}
    days = event_day_sets(ctx, folds)
    groups = {"final_events": ("eval", set().union(*[v for k, v in days.items() if k != "matmo"])),
              "matmo_late": ("secondary", days.get("matmo", set()))}
    edges = np.linspace(0, 1, 21)
    classes = {">=7.6": lambda r: r >= 7.6, ">=2.5": lambda r: r >= 2.5, "<0.5": lambda r: r < 0.5}

    def describe(fr, t, L, folds_used, grp):
        for n, j in fr.items():
            for g, (role, ds) in grp.items():
                e = j[(j["role"] == role).to_numpy() & j["day"].isin(ds).to_numpy()]
                if not len(e):
                    continue
                lab = e["label_rate_mm_h"].to_numpy("float64")
                for cn, f in classes.items():
                    sel = f(np.nan_to_num(lab, nan=-1.0)) if cn != "<0.5" else (lab < 0.5)
                    for kind in ("cal", "raw"):
                        v = e.loc[sel, f"p_{kind}"].to_numpy("float64")
                        dist.append({"group": g, "label_class": cn, "threshold": t, "lead_min": L, "model": n,
                                     "prob": kind, "folds": str(folds_used), "n": int(len(v)),
                                     "mean": float(v.mean()) if len(v) else np.nan,
                                     **{f"q{int(q * 100):02d}": float(np.quantile(v, q)) if len(v) else np.nan
                                        for q in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)}})
                    v = e.loc[sel, "p_cal"].to_numpy("float64")
                    h, _ = np.histogram(v, bins=edges)
                    hist.extend([{"group": g, "label_class": cn, "threshold": t, "lead_min": L, "model": n,
                              "bin_lo": float(edges[k]), "bin_hi": float(edges[k + 1]), "count": int(h[k])}
                             for k in range(len(h))])
                r = M.reliability(e["y"].to_numpy(bool), e["p_cal"].to_numpy("float64"), ev["reliability_bins"])
                relr.append(r.assign(group=g, threshold=t, lead_min=L, model=n))
    for t in (pt, hv):
        for L in leads(cfg):
            for period, roles in (("eval", ("eval",)), ("secondary", ("secondary",))):
                fr = _frames(ctx, names, L, t, roles, folds)
                if "Model1" not in fr:
                    continue
                for n, j in fr.items():
                    met.append({"period": period, "lead_min": L, "threshold": t, "model": n, "spec": names[n],
                                "folds": str(folds), **metric_ci(ctx, j, t)})
                    if period == "eval" and t == pt:
                        ll[n].append(j)
                for a, b in (("Model1", "M0"), ("R-near", "Model1"), ("W-out", "Model1")):
                    if a in fr and b in fr:
                        diffs.append({"period": period, "lead_min": L, "threshold": t, "comparison": f"{a} - {b}",
                                      **paired(ctx, fr[a], fr[b], t)})
            fr = _frames(ctx, names, L, t, ("eval", "secondary"), folds)
            describe(fr, t, L, folds, groups)
            # Matmo (Late labels, secondary) on the folds of Model 1 that contain it, if not in `folds`
            mf = [s for s in fit_folds(cfg, "m1", t) if s not in folds and any(d.startswith(str(s)) for d in
                                                                              days.get("matmo", set()))]
            if mf:
                frm = _frames(ctx, names, L, t, ("secondary",), mf)
                describe(frm, t, L, mf, {"matmo_late": groups["matmo_late"]})
    propose = {}
    pr = cfg["diagnostics"]["propose_rule"]
    for v in ("R-near", "W-out"):
        if ll[v] and len(ll[v]) == len(ll["Model1"]):
            propose[v] = ci_rule(_ll_pooled_frames(ctx, ll[v], ll["Model1"]), float(pr["ci90_upper_below"])) | {
                "metric": pr["metric"], "folds": folds, "frames": leads(cfg)}
    missed = matmo_missed(ctx)
    f1 = [ctx.read_json(f"{OUT}/fit/{fit_id('m1', s, L, pt)}.json") for s in fit_folds(cfg, "m1", pt)
          for L in leads(cfg)]
    ring_share = float(np.mean([f["train_rows_ring"] / max(f["train_rows"], 1) for f in f1])) if f1 else None
    d = pd.DataFrame(dist)
    under = []
    if len(d):
        q = d[(d["label_class"] == ">=7.6") & (d["prob"] == "cal") & d["model"].isin(["M0", "Model1"])]
        for (g, t, L), x in q.groupby(["group", "threshold", "lead_min"]):
            m = x.set_index("model")["mean"]
            under.append({"group": g, "threshold": t, "lead_min": L, "mean_p_M0": m.get("M0"),
                          "mean_p_Model1": m.get("Model1")})
    summ = {"folds": folds, "propose": propose, "proposed": [k for k, v in propose.items() if v["pass"]],
            "rule": "pre-registered D189: proposed only if the upper 90 % CI of (variant - Model 1) pooled OOF "
                    "log-loss(2.5), day-summed over the frames, is < 0; events are descriptive only (hub 6)",
            "ring_share_of_training_rows_model1": ring_share, "mean_p_on_label_ge_7p6": under,
            "matmo_missed_rows": int(len(missed)), "note": NOT_FLOOD}
    out = out_dir("compare")
    files = {"diag_metrics.csv": pd.DataFrame(met), "diag_diffs.csv": pd.DataFrame(diffs),
             "diag_event_dist.csv": d, "diag_event_hist.csv": pd.DataFrame(hist),
             "diag_event_reliability.csv": pd.concat(relr, ignore_index=True) if relr else pd.DataFrame(),
             "diag_matmo_missed.csv": missed}
    for k, v in files.items():
        v.to_csv(out / k, index=False)
    write_json(out / "diag_summary.json", summ)
    return [rel(out / k) for k in list(files) + ["diag_summary.json", "rounds_check.json"]], {
        "proposed": summ["proposed"], "ring_share": ring_share, "matmo_missed_rows": int(len(missed))}


def matmo_missed(ctx) -> pd.DataFrame:
    """Input values on the Matmo city pixel-times with label >= 7.6 that Model 1 (nested 7.6 alarm) missed, every
    frame, with the M0 / Model 1 probabilities; Gold features read from table_v2 (Late labels, descriptive)."""
    cfg = ctx.cfg
    pt, hv = primary(cfg), heavy(cfg)
    mdays = {d.strftime("%Y-%m-%d") for d in pd.date_range(*cfg["evaluation"]["events"]["matmo"], freq="D")}
    folds = [s for s in fit_folds(cfg, "m1", hv) if any(d.startswith(str(s)) for d in mdays)]
    if not folds:
        return pd.DataFrame()
    rows = []
    for L in leads(cfg):
        j = pooled(ctx, "m1", L, hv, ("secondary", "eval"), folds)
        j = j[j["day"].isin(mdays).to_numpy() & j["y"].to_numpy() & ~j["yhat"].to_numpy(bool)]
        if not len(j):
            continue
        k = j[["pixel_id", "t0", "day", "label_rate_mm_h", "label_source", "p_cal", "p_raw"]].rename(
            columns={"p_cal": "p76_cal_Model1", "p_raw": "p76_raw_Model1"})
        k["lead_min"] = L
        for n, sp in (("Model1", "m1"), ("M0", "m0")):
            for t in (pt, hv):
                if has_fit(cfg, sp, t, folds) and not (n == "Model1" and t == hv):
                    q = pooled(ctx, sp, L, t, ("secondary", "eval"), folds)[["pixel_id", "t0", "p_cal"]]
                    k = k.merge(q.rename(columns={"p_cal": f"p{kstr(t)}_cal_{n}"}), on=["pixel_id", "t0"], how="left")
        rows.append(k)
    if not rows:
        return pd.DataFrame(columns=["pixel_id", "t0", "lead_min"])
    from src.gold import build_table_v2 as BT2
    k = pd.concat(rows, ignore_index=True)
    t = k["t0"].dt.tz_convert("UTC")
    gold = []
    for y, m in sorted(set(zip(t.dt.year, t.dt.month))):
        gk = ctx.gold_key(int(y), int(m))
        ls = ctx.store.list(gk)
        if gk not in ls:
            continue
        p = ctx.gold_local() / f"season={int(y)}" / f"month={int(m):02d}" / "part.parquet"
        ctx._download(gk, ls[gk], p)
        allc = BT2.feature_columns(PB._schema_names(p))           # Model 1 inputs (set d + ring extras)
        feats = [c for c in allc if c != "cal_month" and c not in BT2.STATIC] + \
            [c for c in cfg["features"]["ring_model_extra"] if c in allc]
        g = pd.read_parquet(p, columns=list(dict.fromkeys(["pixel_id", "t0", "lead_min"] + feats)))
        g["t0"] = g["t0"].astype("datetime64[ns, UTC]")
        gold.append(g.merge(k[["pixel_id", "t0", "lead_min"]], on=["pixel_id", "t0", "lead_min"]))
    if gold:
        k = k.merge(pd.concat(gold, ignore_index=True), on=["pixel_id", "t0", "lead_min"], how="left")
    k["note"] = "Matmo, IMERG Late labels (provisional), descriptive only; " + NOT_FLOOD
    return k.sort_values(["lead_min", "t0", "pixel_id"]).reset_index(drop=True)


# ============================================================================ P4E.3 gate (two-stage)
def gate_cmp_2stage(ctx: Ctx) -> tuple[list[str], dict]:
    from src.models import metrics as M
    cfg, ev = ctx.cfg, ctx.cfg["evaluation"]
    require_rounds(ctx)
    pt, hv = primary(cfg), heavy(cfg)
    ur = cfg["two_stage"]["use_rule"]
    met, diffs, relr, viol, evs, use = [], [], [], [], [], {}
    for a, b, setname in (("Model1-twostage", "Model1", "A"), ("Model1-B-twostage", "Model1-B", "B")):
        sa, sb = MODEL_SPECS[a], MODEL_SPECS[b]
        folds = sorted(set(fit_folds(cfg, sa, hv)) & set(fit_folds(cfg, sb, hv)))
        if not folds:
            continue
        days = event_day_sets(ctx, folds)
        la, lb = [], []
        for L in leads(cfg):
            for period, roles in (("eval", ("eval",)), ("secondary", ("secondary",))):
                fr = _frames(ctx, {a: sa, b: sb}, L, hv, roles, folds)
                if len(fr) < 2:
                    continue
                for n, j in fr.items():
                    met.append({"set": setname, "period": period, "lead_min": L, "threshold": hv, "model": n,
                                "folds": str(folds), **metric_ci(ctx, j, hv)})
                    if period == "eval":
                        relr.append(M.reliability(j["y"].to_numpy(bool), j["p_cal"].to_numpy("float64"),
                                                  ev["reliability_bins"]).assign(set=setname, lead_min=L, model=n))
                diffs.append({"set": setname, "period": period, "lead_min": L, "threshold": hv,
                              "comparison": f"{a} - {b} (nested)", **paired(ctx, fr[a], fr[b], hv)})
                if period == "eval":
                    la.append(fr[a]); lb.append(fr[b])
                if has_fit(cfg, sb, pt, folds):            # P(>=7.6) of the two-stage vs P(>=2.5) of the base model
                    p25 = pooled(ctx, sb, L, pt, roles, folds)[["pixel_id", "t0", "p_cal", "p_raw"]]
                    jj = fr[a][["pixel_id", "t0", "p_cal", "p_raw"]].merge(p25, on=["pixel_id", "t0"],
                                                                           suffixes=("_76", "_25"))
                    viol.append({"set": setname, "period": period, "lead_min": L, "rows": int(len(jj)),
                                 "violation_rate_cal": float((jj["p_cal_76"] > jj["p_cal_25"]).mean()),
                                 "violation_rate_raw": float((jj["p_raw_76"] > jj["p_raw_25"]).mean())})
            fr = _frames(ctx, {a: sa, b: sb}, L, hv, ("eval", "secondary"), folds)
            for n, j in fr.items():
                for ename, ds in days.items():
                    sel = j["day"].isin(ds).to_numpy()
                    if sel.any():
                        evs.append({"set": setname, "event": ename, "lead_min": L, "model": n,
                                    "role": ",".join(sorted(set(j.loc[sel, "role"]))),
                                    **_cat(j[sel], j["yhat"].to_numpy(bool)[sel], j["p_cal"].to_numpy()[sel])})
        if la and len(la) == len(leads(cfg)):
            r = ci_rule(_ll_pooled_frames(ctx, la, lb), float(ur["ci90_upper_below"]))
            use[setname] = r | {"metric": ur["metric"], "folds": folds,
                                "pre_registered": setname == "A",
                                "role": "decides the 7.6 model of Model 1-B in the shadow gate" if setname == "A"
                                else "informative only (set B two-stage); the use rule is the set-A comparison"}
    a_ok = bool(use.get("A", {}).get("pass", False))
    summ = {"use_two_stage": a_ok, "complete": "A" in use, "rules": use, "violation": viol,
            "rule": "pre-registered D189: the two-stage 7.6 is used in the shadow gate only if (two-stage - nested "
                    "Model 1) pooled OOF log-loss(7.6), day-summed over the frames, has an upper 90 % CI < 0",
            "note": NOT_FLOOD}
    out = out_dir("compare")
    files = {"twostage_metrics.csv": pd.DataFrame(met), "twostage_diffs.csv": pd.DataFrame(diffs),
             "twostage_reliability.csv": pd.concat(relr, ignore_index=True) if relr else pd.DataFrame(),
             "twostage_events.csv": pd.DataFrame(evs)}
    for k, v in files.items():
        v.to_csv(out / k, index=False)
    write_json(out / "twostage_use.json", summ)
    return [rel(out / k) for k in list(files) + ["twostage_use.json", "rounds_check.json"]], {
        "use_two_stage": a_ok, "violation_rate": [round(v["violation_rate_cal"], 4) for v in viol
                                                  if v["period"] == "eval"]}


# ============================================================================ P4E.4 + P4E.6 gate (comparison)
def b_models(cfg: dict, use_two_stage: bool) -> dict:
    """{name: {threshold: spec}} of the P4E.4 / P4E.6 table; Model 1-B at 7.6 = set-B two-stage only if the
    pre-registered P4E.3 rule passed (and the fit exists)."""
    hv = heavy(cfg)
    out = {}
    for n in ("v1B", "Model1-B", "Model1-A", "M0", "R-near", "W-out", "Model1-twostage", "Model1-B-twostage"):
        sp = MODEL_SPECS.get(n, "m1")
        out[n] = {t: sp for t in thresholds(cfg)}
    if use_two_stage and fit_folds(cfg, "m1bts", hv):
        out["Model1-B"][hv] = "m1bts"
    return out


def shadow_gate(cfg: dict, diffs: pd.DataFrame, met: pd.DataFrame, ev7: pd.DataFrame, matmo: pd.DataFrame) -> dict:
    """Pre-registered shadow gate (D189) for Model 1-B vs v1 (B); all must hold. ev7: pooled event-set POD / CSI at 7.6
    per frame and model (Final labels); matmo: CSI 7.6 at F0 on the Matmo days (Late), flag only."""
    sg = cfg["shadow_gate"]
    a, e = sg["at_2p5_every_frame"], sg["events_7p6_every_frame"]
    pt = primary(cfg)
    rows, complete = [], True
    for L in leads(cfg):
        d = diffs[(diffs["period"] == "eval") & (diffs["lead_min"] == L) & (diffs["threshold"] == pt) &
                  (diffs["comparison"] == "Model1-B - v1B")]
        m = met[(met["period"] == "eval") & (met["lead_min"] == L) & (met["threshold"] == pt) &
                (met["model"] == "Model1-B")]
        x = ev7[ev7["lead_min"] == L].set_index("model") if len(ev7) else pd.DataFrame()
        r = {"lead_min": L}
        if len(d) and len(m):
            r |= {"logloss_hi": float(d["logloss_hi"].iloc[0]), "csi_lo": float(d["csi_lo"].iloc[0]),
                  "bss_cal": float(m["bss_cal"].iloc[0])}
            r |= {"ok_logloss": r["logloss_hi"] <= a["logloss_diff_ci90_upper_max"],
                  "ok_csi": r["csi_lo"] >= a["csi_diff_ci90_lower_min"], "ok_bss": r["bss_cal"] > a["bss_min_exclusive"]}
        else:
            complete = False
        if len(x) and {"Model1-B", "v1B"} <= set(x.index):
            r |= {"pod76_diff": float(x.loc["Model1-B", "pod"] - x.loc["v1B", "pod"]),
                  "csi76_diff": float(x.loc["Model1-B", "csi"] - x.loc["v1B", "csi"])}
            r |= {"ok_pod76": r["pod76_diff"] >= e["pod_diff_min"], "ok_csi76": r["csi76_diff"] >= e["csi_diff_min"]}
        else:
            complete = False
        rows.append(r)
    oks = [v for r in rows for k, v in r.items() if k.startswith("ok_")]
    flag = None
    if len(matmo) and {"Model1-B", "v1B"} <= set(matmo["model"]):
        mm = matmo.set_index("model")
        drop = float(mm.loc["v1B", "csi"] - mm.loc["Model1-B", "csi"])
        flag = {"csi76_F0_drop": drop, "flag": drop > float(sg["matmo_flag_csi_7p6_F0_drop"]),
                "note": "Late labels: flagged for the hub, never an automatic fail"}
    return {"complete": complete, "pass": bool(complete and oks and all(oks)), "frames": rows, "matmo": flag}


def gate_cmp(ctx: Ctx) -> tuple[list[str], dict]:
    from src.models import metrics as M
    cfg, ev = ctx.cfg, ctx.cfg["evaluation"]
    rc = require_rounds(ctx)
    pt, hv = primary(cfg), heavy(cfg)
    ts = ctx.read_json(f"{OUT}/compare/twostage_use.json")
    dg = ctx.read_json(f"{OUT}/compare/diag_summary.json")
    use = bool(ts["use_two_stage"])
    bm = b_models(cfg, use)
    folds = _b_folds(cfg)
    days = event_day_sets(ctx, folds)
    pool_days = set(days.get("yagi", set())).union(*[v for k, v in days.items() if k.startswith("wettest")])
    met, diffs, byfold, relr, half, evs, ev7, mat = [], [], [], [], [], [], [], []
    for t in thresholds(cfg):
        for L in leads(cfg):
            names = {n: s[t] for n, s in bm.items()}
            for period, roles in (("eval", ("eval",)), ("secondary", ("secondary",))):
                fr = _frames(ctx, names, L, t, roles, folds)
                if not fr:
                    continue
                for n, j in fr.items():
                    met.append({"period": period, "lead_min": L, "threshold": t, "model": n, "spec": names[n],
                                "folds": str(folds), **metric_ci(ctx, j, t)})
                    if period == "eval":
                        relr.append(M.reliability(j["y"].to_numpy(bool), j["p_cal"].to_numpy("float64"),
                                                  ev["reliability_bins"]).assign(threshold=t, lead_min=L, model=n))
                        for s, g in j.groupby("season_fold"):
                            byfold.append({"heldout_season": s, "lead_min": L, "threshold": t, "model": n,
                                           **PD.metric_row(g, t, ev["reliability_bins"])})
                j0 = next(iter(fr.values()))
                yp = np.nan_to_num(j0["pers"].to_numpy("float64"), nan=-1.0) >= t
                sp_ = M.scores(**M.contingency(j0["y"], yp))
                met.append({"period": period, "lead_min": L, "threshold": t, "model": "persistence", "n": len(j0),
                            "csi": float(sp_["csi"]), "pod": float(sp_["pod"]), "far": float(sp_["far"]),
                            "bias": float(sp_["bias"]), "brier_cal": M.brier(j0["y"], yp.astype(float))})
                if "v1B" in fr:
                    for n in fr:
                        if n != "v1B":
                            diffs.append({"period": period, "lead_min": L, "threshold": t,
                                          "comparison": f"{n} - v1B", **paired(ctx, fr[n], fr["v1B"], t)})
            fr = _frames(ctx, names, L, t, ("eval", "secondary"), folds)
            if not fr:
                continue
            j0 = next(iter(fr.values()))
            pers = np.nan_to_num(j0["pers"].to_numpy("float64"), nan=-1.0) >= t
            for n, j in list(fr.items()) + [("persistence", j0)]:
                yh = pers if n == "persistence" else j["yhat"].to_numpy(bool)
                p = pers.astype(float) if n == "persistence" else j["p_cal"].to_numpy()
                for (role, h), idx in j.groupby(["role", "half"]).indices.items():
                    half.append({"role": role, "half_month": h, "lead_min": L, "threshold": t, "model": n,
                                 **_cat(j.iloc[idx], yh[idx], p[idx])})
                for ename, ds in days.items():
                    sel = j["day"].isin(ds).to_numpy()
                    if sel.any():
                        evs.append({"event": ename, "lead_min": L, "threshold": t, "model": n,
                                    "role": ",".join(sorted(set(j.loc[sel, "role"]))),
                                    "label_source": ",".join(sorted(set(j.loc[sel, "label_source"].astype(str)))),
                                    **_cat(j[sel], yh[sel], p[sel])})
                if t == hv:
                    sel = j["day"].isin(pool_days).to_numpy() & (j["role"] == "eval").to_numpy()
                    ev7.append({"lead_min": L, "model": n, **_cat(j[sel], yh[sel], p[sel])})
                    if L == leads(cfg)[0]:
                        sel = j["day"].isin(days.get("matmo", set())).to_numpy() & (j["role"] == "secondary").to_numpy()
                        if sel.any():
                            mat.append({"lead_min": L, "model": n, **_cat(j[sel], yh[sel], p[sel])})
    met, diffs, ev7, mat = pd.DataFrame(met), pd.DataFrame(diffs), pd.DataFrame(ev7), pd.DataFrame(mat)
    gate = shadow_gate(cfg, diffs, met, ev7, mat) if len(met) and len(diffs) else {"complete": False, "pass": False}
    table = comparison_table(cfg, met, diffs, dg, ts)
    ct = cost_table(ctx)
    dry = dry_share_table(ctx)
    out = out_dir("compare")
    files = {"metrics.csv": met, "diffs.csv": diffs, "by_fold.csv": pd.DataFrame(byfold),
             "reliability.csv": pd.concat(relr, ignore_index=True) if relr else pd.DataFrame(),
             "halfmonth.csv": pd.DataFrame(half), "events.csv": pd.DataFrame(evs), "events_pooled_7p6.csv": ev7,
             "matmo_7p6_F0.csv": mat, "comparison_table.csv": table, "cost.csv": ct, "dry_share_all.csv": dry}
    for k, v in files.items():
        v.to_csv(out / k, index=False)
    checks = {"rounds_check_ok": rc["ok"], "p4e1_downsampling": {"status": cfg["downsampling"]["status"],
              "equivalence_rule": cfg["downsampling"]["equivalence_status"], "decision": "D198"},
              "p4e2_propose": dg.get("propose"), "p4e2_proposed": dg.get("proposed"),
              "p4e3_use_two_stage": use, "p4e3_rules": ts.get("rules"), "model1b_7p6_spec": bm["Model1-B"][hv],
              "shadow_gate": gate, "folds_set_b": folds,
              "dry_share_finding": {"months": int(dry["month"].nunique()) if len(dry) else 0,
                                    "max_dry_scene_share": float(dry["dry_scene_share"].max()) if len(dry) else None,
                                    "mean_dry_scene_share": float(dry["dry_scene_share"].mean()) if len(dry) else None},
              "cost_units_off_1p5x": int(ct["off_1p5x"].sum()) if len(ct) else 0,
              "smoke": bool(cfg.get("smoke")), "note": ("smoke: verdicts are informative only. " if cfg.get("smoke")
                                                         else "") + NOT_FLOOD}
    write_json(out / "checks.json", checks)
    return [rel(out / k) for k in list(files) + ["checks.json", "rounds_check.json"]], {
        "shadow_gate_pass": gate["pass"], "shadow_gate_complete": gate["complete"], "use_two_stage": use,
        "proposed": dg.get("proposed"), "fits": int(len(ct[ct["kind"] == "fit"])) if len(ct) else 0}


def comparison_table(cfg: dict, met: pd.DataFrame, diffs: pd.DataFrame, dg: dict, ts: dict) -> pd.DataFrame:
    """One comparison table (P4E.6): every model at 2.5 and 7.6 per frame (eval, set-B folds) with the difference vs
    v1 (B) and its CI, + the P4E.1 / P4E.2 / P4E.3 verdict rows."""
    rows = []
    src = {"v1B": "P4E.4 reference", "Model1-B": "P4E.4", "Model1-A": "P4E.4 (set A)", "M0": "P4E.2",
           "R-near": "P4E.2 variant", "W-out": "P4E.2 variant", "Model1-twostage": "P4E.3",
           "Model1-B-twostage": "P4E.3 (set B)", "persistence": "reference"}
    if len(met):
        e = met[met["period"] == "eval"]
        for _, r in e[e["threshold"].isin([primary(cfg), heavy(cfg)])].iterrows():
            d = diffs[(diffs["period"] == "eval") & (diffs["lead_min"] == r["lead_min"]) &
                      (diffs["threshold"] == r["threshold"]) & (diffs["comparison"] == f"{r['model']} - v1B")] \
                if len(diffs) else pd.DataFrame()
            row = {"part": src.get(r["model"], ""), "model": r["model"], "threshold": r["threshold"],
                   "lead_min": r["lead_min"], **{k: r.get(k) for k in ("logloss_cal", "bss_cal", "csi", "pod", "far",
                                                                      "bias", "ece_cal", "auc_pr")}}
            if len(d):
                row |= {k: d[k].iloc[0] for k in ("logloss_cal_diff", "logloss_lo", "logloss_hi", "csi_diff",
                                                   "csi_lo", "csi_hi", "pod_diff", "bss_clim_diff")}
            rows.append(row)
    rows.append({"part": "P4E.1", "model": "downsampling", "verdict": "not run, not applied (D198): full data"})
    for v, r in (dg.get("propose") or {}).items():
        rows.append({"part": "P4E.2 propose rule", "model": v, "verdict": "proposed" if r["pass"] else "not proposed",
                     "logloss_cal_diff": r["diff"], "logloss_lo": r["lo"], "logloss_hi": r["hi"]})
    for k, r in (ts.get("rules") or {}).items():
        rows.append({"part": f"P4E.3 use rule (set {k})", "model": "two-stage vs nested",
                     "verdict": "beats nested" if r["pass"] else "does not beat nested",
                     "logloss_cal_diff": r["diff"], "logloss_lo": r["lo"], "logloss_hi": r["hi"]})
    rows.append({"part": "P4E.5", "model": "Tweedie", "verdict": "skipped (D198)"})
    return pd.DataFrame(rows)


def dry_share_table(ctx) -> pd.DataFrame:
    rows = []
    for u in units(ctx.cfg):
        if u.kind == "dry":
            d = ctx.read_json(f"{OUT}/compare/{u.id}.json")
            for r in d["per_frame"]:
                rows.append({"gold_id": d["gold_id"], "season": d["season"], "month": d["month"],
                             **{k: r[k] for k in ("lead_min", "pixels", "scenes", "dry_scenes", "dry_scene_share",
                                                  "dry_scene_row_share", "rows", "rows_raw_all_zero", "rows_dry")}})
    return pd.DataFrame(rows)


def cost_table(ctx) -> pd.DataFrame:
    """Planned (D191 cost model, unit_minutes) vs actual seconds of every timed unit (fits, NWP, dry share, Gold)."""
    from src.models import cloud_train as CT
    us = units(ctx.cfg)
    est = unit_minutes(ctx.cfg, us)
    rows = []
    for u in us:
        if u.kind not in ("fit", "nwp", "dry", "feat", "base"):
            continue
        mk = json.loads(ctx.store.get(CT.marker_key(ctx.job, ctx.run_id, u.id)))
        r = {"unit": u.id, "kind": u.kind, "seconds_actual": mk["metrics"].get("seconds"),
             "seconds_planned": round(60 * est[u.id], 1)}
        if u.kind == "fit":
            f = ctx.read_json(f"{OUT}/fit/{u.id}.json")
            r |= {"family": f["cost_family"], "work_grows": f["work_grows"], "median_rounds": f["median_rounds"],
                  "seconds_from_work_model": f["predicted_seconds_from_work"]}
        else:
            r["reused"] = mk["metrics"].get("reused")
        rows.append(r)
    t = pd.DataFrame(rows)
    if len(t):                                         # D196 / D198: per-unit predicted vs actual, > 1.5x flagged
        t["ratio_actual_planned"] = t["seconds_actual"] / t["seconds_planned"]
        t["off_1p5x"] = (t["ratio_actual_planned"] > COST_RATIO_STOP) | (t["ratio_actual_planned"] < 1 / COST_RATIO_STOP)
    return t


FIG_COLS = ["pixel_id", "t0", "day", "half", "role", "label_source", "label_rate_mm_h", "pers", "y", "p_raw", "p_cal",
            "yhat", "season_fold"]


def fig_spec(cfg: dict) -> dict:
    """Models of the P4E.6 figures (D196 / D198): v1 (B), Model 1-B, Model 1 (A) (+ persistence) on the set-B folds;
    replay pair v1 (B) / Model 1-B; diagnosis pair M0 / Model 1 on the P4E.2 folds. Same names in the smoke."""
    return {"models": [("v1B", "v1b"), ("Model1-B", "m1b"), ("Model1-A", "m1")], "folds": _b_folds(cfg),
            "replay": ["v1B", "Model1-B"], "diag": [("M0", "m0"), ("Model1", "m1")], "diag_folds": _diag_folds(cfg),
            "title": ("P4E SMOKE (F0; set-B models at 2.5 only)" if cfg.get("smoke") else "P4E")}


def gate_figs(ctx: Ctx) -> tuple[list[str], dict]:
    """P4E.6 figures on the cloud (src/models/p4e_figs.py), PNG + CSV under reports/p4e/figs/."""
    from src.models import p4e_figs as FG
    cfg, ev = ctx.cfg, ctx.cfg["evaluation"]
    fs = fig_spec(cfg)
    pt, hv = primary(cfg), heavy(cfg)
    thrs, Ls = [pt, hv], leads(cfg)
    out = out_dir("figs")
    n, seed, lvl = int(ev["bootstrap_resamples"]), int(cfg["random_seed"]), float(ev["ci_level"])

    def frames(spec_list, roles, folds):
        F = {}
        for name, spec in spec_list:
            for t in thrs:
                if not has_fit(cfg, spec, t, folds):
                    continue
                for L in Ls:
                    j = pooled(ctx, spec, L, t, roles, folds)
                    F[(name, float(t), int(L))] = j[[c for c in FIG_COLS if c in j.columns]].copy()
        return F
    models = [m for m, _ in fs["models"]]
    Fe = frames(fs["models"], ("eval",), fs["folds"])
    paths, tables = [], {}
    p, tables["reliability"] = FG.reliability(Fe, models, thrs, Ls, out, fs["title"], ev["reliability_bins"])
    paths += p
    p, tables["scores_ci"] = FG.scores_ci(Fe, models, thrs, Ls, out, fs["title"], n, seed, lvl)
    paths += p
    p, tables["fss"] = FG.fss(Fe, models, thrs, Ls, out, fs["title"], [1, 3, 5], n, seed, lvl)
    paths += p
    p, tables["halfmonth_csi"] = FG.halfmonth_csi(Fe, models, thrs, Ls, out, fs["title"])
    paths += p
    del Fe
    Fa = frames([x for x in fs["models"] if x[0] in fs["replay"]], ("eval", "secondary"), fs["folds"])
    days = FG.event_days(cfg, Fa, fs["replay"][0], pt, Ls[0])
    p, tables["replay"] = FG.replay(Fa, fs["replay"], pt, Ls, days, out, fs["title"])
    paths += p
    del Fa
    Fd = frames(fs["diag"], ("eval",), fs["diag_folds"])
    p, tables["diagnosis"] = FG.diagnosis(Fd, [m for m, _ in fs["diag"]], thrs, Ls, out, fs["title"])
    paths += p
    for k, t in tables.items():
        t = t.copy()
        t["note"] = NOT_FLOOD
        t.to_csv(out / f"{k}.csv", index=False)
    rels = [rel(x) for x in paths] + [rel(out / f"{k}.csv") for k in tables]
    rp = tables["replay"]
    return rels, {"png": len(paths), "replay_ok": int((rp["status"] == "ok").sum()) if len(rp) else 0,
                  "replay_skipped": int((rp["status"] != "ok").sum()) if len(rp) else 0}


def gate_compare_done(ctx: Ctx) -> tuple[list[str], dict]:
    ch = ctx.read_json(f"{OUT}/compare/checks.json")
    rc = ctx.read_json(f"{OUT}/compare/rounds_check.json")
    summ = {"job": ctx.job, "run_id": ctx.run_id, "gold_id": ctx.gid, "smoke": bool(ctx.cfg.get("smoke")),
            "prereg_sha": PREREG_SHA, "nwp_id": nwp_id(ctx.cfg) if uses_set_b(ctx.cfg) else None,
            "stage": "P4E.6 comparison complete (stop point: the hub decides on shadow mode)",
            "rounds_check": rc, "checks": ch, "note": NOT_FLOOD}
    p = out_dir() / "compare_summary.json"
    write_json(p, summ)
    return [rel(p)], {"compare_complete": True, "rounds_check_ok": rc["ok"],
                      "shadow_gate_pass": ch["shadow_gate"]["pass"], "shadow_gate_complete": ch["shadow_gate"]["complete"]}


# ============================================================================ dispatch
def gate_preflight(ctx: Ctx):
    with p4d_env():
        return PD.gate_preflight(ctx)


def gate_g2(ctx: Ctx):
    """P4D G2 (partitions, NaN share, A1 equality, inner-CV label check) + for set B: NWP NaN share per month and the
    set-B inner-CV label check (setb_label_check), before any fit."""
    from src.gold import build_table_v2 as BT2
    with p4d_env():
        rels, m = PD.gate_g2(ctx)
    if not uses_set_b(ctx.cfg):
        return rels, m
    cfg = ctx.cfg
    ctx.fetch_nwp()
    out = out_dir("g2")
    nan = []
    for y, mo in nwp_months(cfg):
        mp = ctx.nwp_local() / f"season={y}" / f"month={mo:02d}" / "part.meta.json"
        if not mp.is_file():
            raise RuntimeError(f"G2 gate FAIL: set-B NWP partition {y}-{mo:02d} missing")
        d = json.loads(mp.read_text(encoding="utf-8"))
        nan.append({"partition": f"{y}-{mo:02d}", "rows": d["rows"], "pixels": d["pixels"], "points": d["points"],
                    "nearest_point_deg_max": d["nearest_point_deg"]["max"], **d["nan_share"]})
    labs = []
    ycols = [thr_col(t) for t in thresholds(cfg)]
    ctx.fetch_gold()                                   # already local after PD.gate_g2 (no download then)
    with p4d_env():
        parts = PD.gold_parts(ctx)
    for y, mo, part in parts:
        df = pd.read_parquet(part, columns=["season", "t0", "lead_min", "is_city"] + ycols)
        df = df[df["is_city"] == 1].drop(columns="is_city")
        df["train_month"] = BT2.is_training_month(cfg, y, mo)
        labs.append(df)
    lab = pd.concat(labs, ignore_index=True)
    lab["t0"] = lab["t0"].astype("datetime64[ns, UTC]")
    chk = setb_label_check(cfg, lab)
    pd.DataFrame(nan).to_csv(out / "nwp_nan_share.csv", index=False)
    pd.DataFrame(chk).to_csv(out / "set_b_inner_cv_check.csv", index=False)
    bad = [r for r in chk if not r["ok"]]
    m = m | {"nwp_partitions": len(nan), "nwp_nan_share_max": float(pd.DataFrame(nan).drop(
        columns=["partition", "rows", "pixels", "points", "nearest_point_deg_max"]).max().max()) if nan else None,
             "set_b_inner_cv_ok": not bad}
    if bad:
        raise RuntimeError(f"G2 gate FAIL: set-B inner CV has < 2 usable blocks for {bad[:3]}; no fit was started")
    return rels + [rel(out / "nwp_nan_share.csv"), rel(out / "set_b_inner_cv_check.csv")], m


GATES = {"preflight": gate_preflight, "g2": gate_g2, "cmp-diag": gate_cmp_diag, "cmp-2stage": gate_cmp_2stage,
         "cmp": gate_cmp, "figs": gate_figs, "compare-done": gate_compare_done}


def run_unit(unit, ctx: Ctx) -> tuple[list[str], dict]:
    t1 = time.time()
    if unit.kind == "gate":
        return GATES[unit.id](ctx)
    if unit.kind == "fit":
        return run_fit(unit, ctx)
    if unit.kind == "feat":
        rels, met = run_feat(unit, ctx)
    elif unit.kind == "nwp":
        rels, met = run_nwp(unit, ctx)
    elif unit.kind == "base":
        with p4d_env():
            rels, met = PD.run_base(unit, ctx)
    elif unit.kind == "dry":
        rels, met = run_dry_full(unit, ctx)
    else:
        raise ValueError(unit.kind)
    return rels, {**met, "seconds": round(time.time() - t1, 1)}


# ============================================================================ planning (cost model, D191)
def training_days(cfg: dict, s: int, set_b: bool = False) -> list:
    from src.gold import build_table_v2 as BT2
    return [d for d in BT2.build_days(cfg) if d.year != s and BT2.is_training_month(cfg, d.year, d.month)
            and (not set_b or d >= nwp_from(cfg))]


def training_rows(cfg: dict, s: int, ring: bool, n_ring: int | None = None, set_b: bool = False) -> int:
    nr = cfg["pixels"]["ring"].get("max_pixels") or cfg["pixels"]["ring"]["expected_pixels"]
    npx = 47 + ((nr if n_ring is None else n_ring) if ring else 0)
    return len(training_days(cfg, s, set_b)) * 48 * npx


def inner_folds(cfg: dict, s: int, set_b: bool = False) -> int:
    days = training_days(cfg, s, set_b)
    if set_b and cfg["set_b"]["inner_cv_block"] == "utc_day":
        return len(set(days))
    return len({(d.year, d.month) for d in days})


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
    dd = dataset_def(spec)
    fam = _family(dd)
    sb = bool(dd.get("set_b"))
    if dd.get("ring_subset") == "r_near" and n_ring is None:
        n_ring = int(cfg["diagnostics"]["r_near_count"])
    rows = training_rows(cfg, s, dd["ring"], n_ring, sb)
    nf = inner_folds(cfg, s, sb)
    if dd.get("ds_p"):
        rows = int(rows * (1 - DRY_SHARE_PRIOR * (1 - dd["ds_p"])))
    f = SETB_COST_FACTOR if sb else 1.0
    if dd.get("two_stage"):
        return f * (fit_minutes(cfg, fam, 0.5, rows, nf) + fit_minutes(cfg, fam, float(thr), int(rows *
                                                                       RAIN_SCENE_SHARE_PRIOR), nf))
    return f * fit_minutes(cfg, fam, float(thr), rows, nf)


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
        elif u.kind in ("dry", "nwp"):
            out[u.id] = EXTRA_UNIT_MINUTES[u.kind]
        else:
            pre = float(gm["done"] if u.id == "compare-done" else gm.get(u.id, 0.0))
            out[u.id] = max(pre, EXTRA_UNIT_MINUTES.get(u.id, 0.0))
    return out


def lpt(minutes: list[tuple[str, float]], n: int) -> list[float]:
    """Longest-processing-time-first assignment of work units to n parts; returns the load of each part."""
    return lpt_assign(minutes, n)[1]


def plan_estimate(cfg: dict, us: list, n: int, um: dict | None = None) -> dict:
    """Per-part minutes when n parts run together: every part waits for the opening gates (preflight, g2), runs its
    share of the opening work (feat, nwp) and of the stage-3 work (base, fit, dry) — the LPT assignment of
    part_units — and the part that finishes last also runs the closing gates; a part with nothing left exits (no idle
    waiting, unlike P4D). Runner minutes = sum of the part walls."""
    um = um or unit_minutes(cfg, us)
    open_g = sum(um[u.id] for u in us if u.kind == "gate" and u.stage <= LAST_OPENING_STAGE)
    close_g = sum(um[u.id] for u in us if u.kind == "gate" and u.stage > LAST_OPENING_STAGE)
    _, fl = lpt_assign([(u.id, um[u.id]) for u in us if u.kind in ("feat", "nwp")], n)
    _, wl = lpt_assign([(u.id, um[u.id]) for u in us if u.kind in ("fit", "base", "dry")], n)
    walls = [open_g + f + w for f, w in zip(fl, wl)]
    k = int(np.argmax(walls))
    walls[k] += close_g
    walls = [round(x, 1) for x in walls]
    pl = cfg["planning"]
    return {"parts": n, "part_minutes": walls, "max_part_minutes": max(walls), "runner_minutes": round(sum(walls), 0),
            "within_target": max(walls) <= float(pl["part_target_minutes"]),
            "unit_minutes_total": round(sum(um.values()), 1)}


def choose_parts(cfg: dict, us: list, um: dict | None = None, margin: float = 1.0) -> dict:
    """Smallest part count whose slowest part is <= margin x part_target_minutes (margin 0.8 = proposed count with
    20 % reserve, D198)."""
    tgt = float(cfg["planning"]["part_target_minutes"]) * margin
    for n in range(1, 61):
        e = plan_estimate(cfg, us, n, um)
        if e["max_part_minutes"] <= tgt:
            return e
    raise ValueError("no part count meets planning.part_target_minutes")


def full_run_plan(cfg: dict, run: str, ds_p: float | None = None, n_ring_near: int = 90) -> tuple[list, dict]:
    """Planning-only unit lists written for P4E.0 (kept for the record; the executable plan is units()): 'A' = P4E.1
    (full Model 1 and the downsampled Model 1 at p in p_grid, 2.5 / 7.6, LOSO 3 folds); 'B' = P4E.2-P4E.4."""
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
