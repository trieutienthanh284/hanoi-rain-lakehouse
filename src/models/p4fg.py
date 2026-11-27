"""P4F + P4G — final model round (TASK_P4F_P4G_FINAL_MODEL.md). Units of the cloud jobs `p4fg_smoke` and `p4fg`
(driver: src/models/cloud_train.py, main_p4fg); gates in src/models/p4fg_gates.py, final fit + REPLAY in
src/models/p4fg_replay.py (job p4fg_final). Not a flood model; labels are IMERG (satellite estimate), not rain gauges.
v1 stays the official model; nothing here goes to live serving.

Configuration: configs/model_p4e.yaml through p4e.load_cfg (D189 hash checked) + configs/model_p4fg.yaml on top; the
`pre_registered` block (D214) is hash-checked by load_cfg. Gold table_v2 (gid a4a6329bcf13) and the P4E set-B NWP
columns are REUSED, never built: R2 writes of P4F/P4G are approved only under runs/p4fg*/ and table_v3*/ (CLAUDE.md
§8); a missing partition makes the reuse unit fail (the store refuses table_v2 keys). The smoke filters the full
Gold to `smoke_days` at load time and chooses its frame / thresholds with run_leads / run_thresholds (the gold id
stays the full one).

Units (stage):
  0 preflight            gate (P4D preflight: R2 budget, ring geometry)
  1 feat-YYYY-MM         Gold table_v2 month, reused (never built)
    nwp-YYYY-MM          set-B NWP columns (P4E table_v2_nwp), reused (never built)
    conv-YYYY-MM         P4G.1 convective columns (GFS CAPE / LI previous_day1) for every table_v2 row (city + ring)
                         from Silver nwp_conv -> lake/v1/gold/met_training/table_v3_conv/gid=<gid>/cid=<cid>/
    orc-YYYY-MM          P4G.0 oracle series (target-frame GSMaP GC / raw and 3x3 means, deliberately leaky) -> run
                         files reports/p4fg/oracle/series/ (oracle = true); never a Gold table
  2 g2                   gate: P4D G2 + set-B / conv NaN shares, conv G2 = 0, set-B inner-CV label check
  3 base-s<fold>-lead<L> evaluation rows (city, never cut / sampled / weighted)
    fit-<spec>-...       B0 b15x500, hp grid b<nl>x<md>, conv on B0 b15x500{a,c,ac}, oracle orc, regional reg, v1b
    dtcheck              P4F.0 cloud check (season totals GC / raw / IMERG Final, Δt 0.5 h and 1 h)
    fssv1                P4F.2 FSS of v1 on the P4B row-level predictions (OOF 2024 set B, test 2025) + persistence
  4 hp-select            gate P4G.3: one setting per fold by the fold's own inner-OOF log-loss
  5 fit-t{a,c,ac}-...    conv steps on the tuned model (alias of the B0 version when the fold chose g1)
  6 cmp                  gate: oracle A/B, regional, union rule, selection chain P4G.5, P4E shadow gate, table
    fss                  gate: FSS of every candidate (3x3, 5x5) per frame, day-block CI
  7 figs                 gate: P4FG figures + P4E figures re-rendered with the display names (P4F.3)
    replay-smoke         p4fg_smoke only: REPLAY export of one event day under the run prefix (never serving)
    finalfit-smoke       p4fg_smoke only: one final-fit unit (code path of p4fg_final)
  8 compare-done         gate: summary; the winner writes SELECTION.json + COMPARE_DONE.json at the run prefix
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import re
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.models import p4b as PB
from src.models import p4d as PD
from src.models import p4e as PE

CODE_ROOT = Path(__file__).resolve().parents[2]
CFG_PATH = CODE_ROOT / "configs" / "model_p4fg.yaml"
LAKE = "lake/v1/"
OUT = "reports/p4fg"
NOT_FLOOD = PD.NOT_FLOOD
# D214: sha256 of json.dumps(pre_registered, sort_keys=True), written to DECISIONS before any P4FG training code
PREREG_SHA = "fbc7ec898884519b4970fa786ef099c70e7be636df38c533d5173caf7f33335a"
CODE_FILES = PE.CODE_FILES + ["src/models/p4fg.py", "src/models/p4fg_gates.py", "src/models/p4fg_replay.py",
                              "src/models/rss_gate.py", "configs/model_p4fg.yaml"]
CONV_CODE = ["src/models/p4fg.py"]
DEFAULT_HP = (15, 500)
CONV_GROUPS = {"a": "g_a", "c": "g_c"}
ORACLE_PREFIX = "oracle_"
CONV_PREFIX = "nwp_conv_"
GSMAP_SLOT_DT_H = 0.5           # P4F.0: code Δt per 30-min GSMaP file (src/events/indicator_analysis.py:88)
LAST_OPENING_STAGE = 2
MID_GATES = ("hp-select",)      # stage-4 gate every part waits for (stage-5 fits depend on it)
DRILL_GATE = "cmp"
STAGE5 = 5


def root() -> Path:
    return PB.ROOT


@contextlib.contextmanager
def env():
    """Shared P4D / P4E helpers write and read under reports/p4fg (module globals OUT), P4D cut variant off."""
    old = PE.OUT, PD.OUT, PD.VARIANTS
    PE.OUT, PD.OUT, PD.VARIANTS = OUT, OUT, {"plain": False}
    try:
        yield
    finally:
        PE.OUT, PD.OUT, PD.VARIANTS = old


# ============================================================================ configuration
def prereg_sha(raw: dict) -> str:
    return hashlib.sha256(json.dumps(raw["pre_registered"], sort_keys=True).encode()).hexdigest()


def load_cfg(smoke: bool = False, path: Path | None = None, check_prereg: bool = True) -> dict:
    raw = yaml.safe_load(Path(path or CFG_PATH).read_text(encoding="utf-8"))
    if check_prereg and prereg_sha(raw) != PREREG_SHA:
        raise RuntimeError("pre-registered block of configs/model_p4fg.yaml changed after DECISIONS D214 "
                           f"({prereg_sha(raw)} != {PREREG_SHA}): stop and ask (TASK P4F/P4G)")
    base = PE.load_cfg(smoke=False)                    # D189 check of the P4E pre-registration
    for k in ("fits", "two_stage", "diagnostics"):    # P4E-only sections, not used by P4FG
        base.pop(k, None)
    top = {k: v for k, v in raw.items() if k not in ("smoke", "base_config")}
    cfg = PB.deep_merge(base, top)
    pr = cfg["pre_registered"]
    cfg["folds"] = [int(s) for s in pr["folds"]]
    cfg["evaluation"]["loso_seasons"] = list(cfg["folds"])
    cfg["evaluation"]["bootstrap_resamples"] = int(pr["bootstrap"]["resamples"])
    cfg["evaluation"]["ci_level"] = float(pr["bootstrap"]["ci_level"])
    bm = pr["base_model"]
    cfg["lightgbm"] |= dict(bm["lightgbm"])
    cfg["inner_cv"] |= {"max_rounds": int(bm["max_rounds"]), "early_stopping_rounds": int(bm["early_stopping_rounds"])}
    cfg["hp_grid_points"] = [[int(a), int(b)] for a in pr["hp_grid"]["num_leaves"]
                             for b in pr["hp_grid"]["min_data_in_leaf"]]
    if smoke:
        sm = raw.get("smoke", {})
        cfg = PB.deep_merge(cfg, {k: v for k, v in sm.items() if k not in ("folds", "hp_grid_points")})
        cfg["folds"] = [int(s) for s in sm["folds"]]
        cfg["evaluation"]["loso_seasons"] = list(cfg["folds"])
        cfg["hp_grid_points"] = [[int(a), int(b)] for a, b in sm["hp_grid_points"]]
        cfg["smoke"] = True
    if list(DEFAULT_HP) not in cfg["hp_grid_points"]:
        raise ValueError("hp grid must contain the B0 setting (15, 500)")
    return cfg


def leads(cfg: dict) -> list[int]:
    return [int(x) for x in (cfg.get("run_leads") or PD.leads(cfg))]


def thresholds(cfg: dict) -> list[float]:
    return [float(x) for x in (cfg.get("run_thresholds") or PD.thresholds(cfg))]


primary, heavy, kstr, thr_col = PD.primary, PD.heavy, PB.kstr, PB.thr_col
gold_id = PD.gold_id


def folds(cfg: dict) -> list[int]:
    return list(cfg["folds"])


def smoke_days(cfg: dict) -> set | None:
    rr = cfg.get("smoke_days")
    if not rr:
        return None
    return {d.date() for a, b in rr for d in pd.date_range(a, b, freq="D")}


def run_months(cfg: dict) -> list[tuple[int, int]]:
    """Set-B months (>= set_b.nwp_from) used by this run (smoke: months holding a smoke day)."""
    ms = PE.nwp_months(cfg)
    sd = smoke_days(cfg)
    return [x for x in ms if sd is None or any((d.year, d.month) == x for d in sd)]


def conv_id(cfg: dict) -> str:
    """Identity of the convective columns: gold id, the pre-registered convective block, build version, code."""
    h = hashlib.sha256(json.dumps({"gid": gold_id(cfg), "conv": cfg["pre_registered"]["convective"],
                                   "v": cfg["conv_store"]["build_version"]}, sort_keys=True).encode())
    for f in CONV_CODE:
        h.update(_conv_code_bytes(CODE_ROOT / f))
    return h.hexdigest()[:12]


def _conv_code_bytes(p: Path) -> bytes:
    """Only the convective section of this module defines the conv columns (other edits keep the cid)."""
    s = p.read_bytes().replace(b"\r\n", b"\n")
    a, b = s.find(b"# ==== conv-begin"), s.find(b"# ==== conv-end")
    return s[a:b] if 0 <= a < b else s


# ============================================================================ specs and the fit plan
def spec_name(hp: tuple[int, int], conv: str = "") -> str:
    return f"b{int(hp[0])}x{int(hp[1])}{conv}"


B0 = spec_name(DEFAULT_HP)


def spec_def(spec: str) -> dict:
    if spec == "v1b":
        return {"ring": False, "hp": DEFAULT_HP, "conv": ""}
    if spec == "orc":
        return {"ring": True, "hp": DEFAULT_HP, "conv": "", "oracle": True}
    if spec == "reg":
        return {"ring": True, "hp": DEFAULT_HP, "conv": "", "regional": True}
    m = re.fullmatch(r"b(\d+)x(\d+)(a|c|ac)?", spec)
    if m:
        return {"ring": True, "hp": (int(m.group(1)), int(m.group(2))), "conv": m.group(3) or ""}
    m = re.fullmatch(r"t(a|c|ac)", spec)
    if m:
        return {"ring": True, "hp": "tuned", "conv": m.group(1)}
    raise ValueError(spec)


def fit_id(spec: str, s: int, lead: int, thr: float) -> str:
    return PD.fit_id(spec, s, lead, thr)


def fit_plan(cfg: dict) -> list[tuple[str, int, float]]:
    """(spec, fold, threshold) of every fit, each at every frame of leads(cfg)."""
    thrs, pt, hv = thresholds(cfg), primary(cfg), heavy(cfg)
    sel = [t for t in (pt, hv) if t in thrs]
    out = []
    for s in folds(cfg):
        out += [("v1b", s, t) for t in thrs] + [(B0, s, t) for t in thrs]
        out += [(spec_name(tuple(h)), s, t) for h in cfg["hp_grid_points"] if tuple(h) != DEFAULT_HP for t in sel]
        out += [(spec_name(DEFAULT_HP, c), s, t) for c in ("a", "c", "ac") for t in sel]
        out += [(f"t{c}", s, t) for c in ("a", "c", "ac") for t in sel]
        out += [("orc", s, t) for t in sel] + [("reg", s, t) for t in sel]
    return out


def setting_specs(cfg: dict) -> list[str]:
    return [spec_name(tuple(h)) for h in cfg["hp_grid_points"]]


def unit_stage(spec: str) -> int:
    return STAGE5 if spec_def(spec)["hp"] == "tuned" else 3


# ============================================================================ units
def units(cfg: dict) -> list:
    from src.gold import build_table_v2 as BT2
    U = PB.Unit
    u = [U("preflight", "gate", 0)]
    feat = [U(f"feat-{y}-{m:02d}", "feat", 1, ("preflight",), {"y": y, "m": m}) for y, m in BT2.build_months(cfg)]
    nwp = [U(f"nwp-{y}-{m:02d}", "nwp", 1, (f"feat-{y}-{m:02d}",), {"y": y, "m": m}) for y, m in PE.nwp_months(cfg)]
    conv = [U(f"conv-{y}-{m:02d}", "conv", 1, (f"feat-{y}-{m:02d}",), {"y": y, "m": m}) for y, m in run_months(cfg)]
    orc = [U(f"orc-{y}-{m:02d}", "orc", 1, (f"feat-{y}-{m:02d}",), {"y": y, "m": m}) for y, m in run_months(cfg)]
    u += feat + nwp + conv + orc
    u.append(U("g2", "gate", 2, tuple(x.id for x in feat + nwp + conv)))
    base, s3, s5 = [], [], []
    for L in leads(cfg):
        for s in folds(cfg):
            base.append(U(PD.base_id(s, L), "base", 3, ("g2",), {"season": s, "lead": L}))
        for sp, s, t in fit_plan(cfg):
            p = {"spec": sp, "season": s, "lead": L, "thr": t}
            if unit_stage(sp) == STAGE5:
                dd = spec_def(sp)
                alias = fit_id(spec_name(DEFAULT_HP, dd["conv"]), s, L, t)
                s5.append(U(fit_id(sp, s, L, t), "fit", STAGE5, ("hp-select", alias), p))
            else:
                deps = ("g2",) + (tuple(f"orc-{y}-{m:02d}" for y, m in run_months(cfg)) if sp == "orc" else ())
                s3.append(U(fit_id(sp, s, L, t), "fit", 3, deps, p))
    dt = [U("dtcheck", "dtcheck", 3, ("preflight",))]
    fv = [U("fssv1", "fssv1", 3, tuple(b.id for b in base))]
    u += base + s3 + dt + fv
    pt = primary(cfg)
    grid25 = tuple(f.id for f in s3 if f.params["spec"] in setting_specs(cfg) and f.params["thr"] == pt)
    u.append(U("hp-select", "gate", 4, grid25))
    u += s5
    allfits = tuple(f.id for f in s3 + s5) + tuple(b.id for b in base)
    u.append(U("cmp", "gate", 6, allfits + ("hp-select", "dtcheck") + tuple(x.id for x in orc)))
    u.append(U("fss", "gate", 6, allfits + ("hp-select", "fssv1")))
    u.append(U("figs", "gate", 7, ("cmp", "fss")))
    closing = ["figs"]
    if cfg.get("smoke"):
        u.append(U("finalfit-smoke", "gate", 7, ("cmp",)))
        u.append(U("replay-smoke", "gate", 7, ("cmp", "finalfit-smoke")))
        closing += ["finalfit-smoke", "replay-smoke"]
    u.append(U("compare-done", "gate", 8, tuple(closing)))
    return u


def unit_minutes(cfg: dict, us: list) -> dict:
    """Planning estimate (minutes) per unit: P4E measured seconds (REPORT_P4E §6.3: Model 1-B 109 s mean, v1 (B)
    23 s) x 1.5 safety, scaled by the boosting cost of the hp setting; conv / orc / dtcheck from their data volume."""
    out = {}
    for u in us:
        if u.kind == "fit":
            dd = spec_def(u.params["spec"])
            if u.params["spec"] == "v1b":
                m = 23.4 / 60 * 1.5
            else:
                nl, md = dd["hp"] if dd["hp"] != "tuned" else (31, 500)
                m = 109.0 / 60 * 1.5 * (nl / 15) ** 0.6 * (500 / md) ** 0.1
                m *= 1.05 if dd["conv"] or dd.get("oracle") else 1.0
                m *= 1.3 if dd.get("regional") else 1.0
            if cfg.get("smoke"):
                m = 0.5 + m * 0.15
            out[u.id] = m
        elif u.kind in ("feat", "nwp"):
            out[u.id] = 0.2
        elif u.kind == "conv":
            out[u.id] = 3.0
        elif u.kind == "orc":
            out[u.id] = 4.0
        elif u.kind == "base":
            out[u.id] = 1.0
        elif u.kind == "dtcheck":
            out[u.id] = 20.0
        elif u.kind == "fssv1":
            out[u.id] = 8.0
        else:
            out[u.id] = {"preflight": 1.0, "g2": 6.0, "hp-select": 2.0, "cmp": 45.0, "fss": 25.0, "figs": 20.0,
                         "finalfit-smoke": 3.0, "replay-smoke": 5.0, "compare-done": 1.0}.get(u.id, 1.0)
            if cfg.get("smoke") and u.id in ("cmp", "fss", "figs"):
                out[u.id] *= 0.4
    return out


WORK_PHASES = (("feat", "nwp", "conv", "orc"), ("base", "fit3", "dtcheck", "fssv1"), ("fit5",))


def _phase_of(u) -> int | None:
    k = u.kind
    if k == "fit":
        k = "fit5" if u.stage == STAGE5 else "fit3"
    for i, ks in enumerate(WORK_PHASES):
        if k in ks:
            return i
    return None


def part_units(us: list, i: int, n: int, cfg: dict) -> list:
    """Work units of part i/n: LPT on the planned minutes, separately for the opening work, the stage-3 work and the
    stage-5 work (same assignment as plan_estimate)."""
    um = unit_minutes(cfg, us)
    out = []
    for ph in range(len(WORK_PHASES)):
        work = [u for u in us if _phase_of(u) == ph]
        a, _ = PE.lpt_assign([(u.id, um[u.id]) for u in work], n)
        mine = [u for u in work if a[u.id] == i - 1]
        mine.sort(key=lambda u: (u.stage, u.kind != "base", u.params.get("lead", 0), u.params.get("spec", ""),
                                 u.params.get("season", 0), u.params.get("thr", 0.0), u.id))
        out += mine
    return out


def plan_estimate(cfg: dict, us: list, n: int) -> dict:
    um = unit_minutes(cfg, us)
    open_g = sum(um[u.id] for u in us if u.kind == "gate" and u.stage <= LAST_OPENING_STAGE)
    mid_g = sum(um[u.id] for u in us if u.kind == "gate" and u.id in MID_GATES)
    close_g = sum(um[u.id] for u in us if u.kind == "gate" and u.stage > LAST_OPENING_STAGE and u.id not in MID_GATES)
    loads = []
    for ph in range(len(WORK_PHASES)):
        _, ld = PE.lpt_assign([(u.id, um[u.id]) for u in us if _phase_of(u) == ph], n)
        loads.append(ld)
    walls = [open_g + mid_g + sum(ld[k] for ld in loads) for k in range(n)]
    k = int(np.argmax(walls))
    walls[k] += close_g
    walls = [round(x, 1) for x in walls]
    pl = cfg["pre_registered"]["planning"]
    return {"parts": n, "part_minutes": walls, "max_part_minutes": max(walls), "runner_minutes": round(sum(walls), 0),
            "within_target": max(walls) <= float(pl["part_target_minutes"]),
            "unit_minutes_total": round(sum(um.values()), 1),
            "by_kind": {kk: round(sum(um[u.id] for u in us if u.kind == kk), 1) for kk in sorted({u.kind for u in us})}}


def choose_parts(cfg: dict, us: list, margin: float = 0.8) -> dict:
    tgt = float(cfg["pre_registered"]["planning"]["part_target_minutes"]) * margin
    for n in range(1, 41):
        e = plan_estimate(cfg, us, n)
        if e["max_part_minutes"] <= tgt:
            return e
    raise ValueError("no part count meets the part target")


def inner_cv_plan_check(cfg: dict) -> list[dict]:
    """Set-B structural check per fold: training days (other folds' seasons, training months, >= nwp_from, smoke
    days only in the smoke) must give >= 2 usable inner blocks (and >= 2 calendar months for month blocks)."""
    from src.gold import build_table_v2 as BT2
    days = BT2.build_days(cfg)
    sd = smoke_days(cfg)
    if sd is not None:
        days = [d for d in days if d in sd]
    buf = int(cfg["inner_cv"]["buffer_days"])
    blk = cfg["set_b"]["inner_cv_block"]
    rows = []
    for s in folds(cfg):
        tr = [d for d in days if d.year != s and BT2.is_training_month(cfg, d.year, d.month) and d >= PE.nwp_from(cfg)]
        ub = []
        for b in PE._blocks(tr, blk):
            a, e = PE._block_bounds(b, blk)
            if [d for d in tr if d <= a - timedelta(days=buf + 1) or d >= e + timedelta(days=buf)]:
                ub.append(b)
        cal = sorted({(b[1] if blk == "calendar_month" else b.month) for b in ub})
        ok = len(ub) >= 2 and (len(cal) >= 2 if blk == "calendar_month" else True)
        rows.append({"heldout_season": s, "block": blk, "usable_blocks": [str(b) for b in ub],
                     "calendar_months": cal, "ok": ok})
    return rows


# ============================================================================ run context
class Ctx(PE.Ctx):
    """p4e.Ctx (table_v2 gid, set-B NWP columns) + the convective columns (table_v3_conv)."""

    def conv_prefix(self) -> str:
        return f"{self.cfg['conv_store']['gold_prefix']}gid={self.gid}/cid={conv_id(self.cfg)}/"

    def conv_key(self, y: int, m: int, name: str = "part.parquet") -> str:
        return f"{self.conv_prefix()}season={y}/month={m:02d}/{name}"

    def conv_local(self) -> Path:
        return (root() / "data" / "gold" / "met_training" / "v3" / "table_v3_conv" / f"gid={self.gid}" /
                f"cid={conv_id(self.cfg)}")

    def fetch_conv(self) -> int:
        pre = self.conv_prefix()
        return sum(self._download(k, size, self.conv_local() / k[len(pre):])
                   for k, size in sorted(self.store.list(pre).items()))


def write_json(p: Path, obj) -> None:
    PB.write_json(p, obj)


def rel(p: Path) -> str:
    return Path(p).resolve().relative_to(root().resolve()).as_posix()


def out_dir(*parts: str) -> Path:
    p = root().joinpath(OUT, *parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def silver_frames(ctx, source: str, days: list, pixels: np.ndarray, cols: list[str]) -> pd.DataFrame:
    """Silver rows of `source` on `days` (lake/v1/data/silver/met/<source>/date=...), only `pixels` and `cols`."""
    rels = [PD._silver_rel(source, d) for d in days]
    ctx.fetch_lake(rels, f"data/silver/met/{source}/")
    parts = []
    for r in rels:
        p = root() / r
        if p.is_file():
            parts.append(pd.read_parquet(p, columns=["pixel_id"] + cols,
                                         filters=[("pixel_id", "in", [int(x) for x in pixels])]))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["pixel_id"] + cols)


# ==== conv-begin (P4G.1 convective columns; this section alone defines conv_id with the pre-registered block)
CONV_VARS = {"cape": "cape", "li": "lifted_index"}


def conv_feature_names(cfg: dict, groups: str) -> list[str]:
    f = cfg["pre_registered"]["convective"]["features"]
    out = []
    for g in groups:
        cols = f[CONV_GROUPS[g]]
        if not isinstance(cols, list):
            raise ValueError(f"convective group {g} is {cols}")
        out += cols
    return out


def compute_conv_features(rows: pd.DataFrame, cv: pd.DataFrame, pmap: dict, rain_now: np.ndarray) -> pd.DataFrame:
    """rows: (pixel_id, prediction_time, frame_start); cv: Silver nwp_conv rows (pixel_id = Open-Meteo point,
    time_utc = time label T, cape, lifted_index, forecast_issue_time_utc, available_time_utc).
    Value of the hour T = floor_1h(frame_start) at the nearest point (near) and averaged over the points within 0.3 deg
    (m03); a record is admissible only if issue <= prediction_time - 22 h AND available <= prediction_time (D214).
    cape_x_rain = cape_near x [rain_now >= 0.5 mm/h] (rain_now = latest admissible GSMaP GC at the pixel, NaN stays
    NaN). Group times = latest issue / availability among the records used."""
    NAT = np.iinfo("int64").min
    H = 3600 * 10**9
    n = len(rows)

    def ns(s):
        return s.dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]").to_numpy().view("int64")
    pt = ns(rows["prediction_time"])
    fs = ns(rows["frame_start"])
    h0 = fs - fs % H
    pix = rows["pixel_id"].to_numpy("int64")
    key = pd.MultiIndex.from_arrays([cv["pixel_id"].to_numpy("int64"), ns(cv["time_utc"])])
    tab = pd.DataFrame({"cape": cv["cape"].to_numpy("float64"), "li": cv["lifted_index"].to_numpy("float64"),
                        "iss": ns(cv["forecast_issue_time_utc"]), "av": ns(cv["available_time_utc"])}, index=key)
    tab = tab[~tab.index.duplicated()]
    out, iss_max, av_max = {}, np.full(n, NAT, "int64"), np.full(n, NAT, "int64")
    lists = {"near": [[pmap[int(p)]["near"]] for p in pix], "m03": [pmap[int(p)]["within"] for p in pix]}
    for agg, pts in lists.items():
        lens = np.array([len(x) for x in pts])
        starts = np.r_[0, np.cumsum(lens)[:-1]]
        flat = np.concatenate([np.asarray(x, "int64") for x in pts])
        r = tab.reindex(pd.MultiIndex.from_arrays([flat, np.repeat(h0, lens)]))
        iss, av = r["iss"].to_numpy("float64"), r["av"].to_numpy("float64")
        ptr = np.repeat(pt, lens).astype("float64")
        ok = np.isfinite(iss) & np.isfinite(av) & (iss <= ptr - 22 * H) & (av <= ptr)
        for v in ("cape", "li"):
            x = np.where(ok, r[v].to_numpy("float64"), np.nan)
            cnt = np.add.reduceat(np.isfinite(x).astype("int64"), starts)
            tot = np.add.reduceat(np.nan_to_num(x), starts)
            out[f"{CONV_PREFIX}gfs_{v}_{agg}"] = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan).astype("float32")
        ii = np.maximum.reduceat(np.where(ok, iss, -np.inf), starts)
        aa = np.maximum.reduceat(np.where(ok, av, -np.inf), starts)
        iss_max = np.where(np.isfinite(ii), np.maximum(iss_max, np.where(np.isfinite(ii), ii, 0).astype("int64")), iss_max)
        av_max = np.where(np.isfinite(aa), np.maximum(av_max, np.where(np.isfinite(aa), aa, 0).astype("int64")), av_max)
    rn = np.asarray(rain_now, "float64")
    wet = np.where(np.isnan(rn), np.nan, (rn >= 0.5).astype("float64"))
    out[f"{CONV_PREFIX}gfs_cape_x_rain"] = (out[f"{CONV_PREFIX}gfs_cape_near"] * wet).astype("float32")
    df = pd.DataFrame(out, index=rows.index)
    for c, v in ((f"{CONV_PREFIX}gfs__issue_time", iss_max), (f"{CONV_PREFIX}gfs__available_time", av_max)):
        df[c] = pd.to_datetime(np.where(v == NAT, np.datetime64("NaT", "ns"), v.view("datetime64[ns]")), utc=True)
    return df


def conv_g2(df: pd.DataFrame, cols: list[str]) -> list:
    """Frame-mode G2 (ADD-005 frames, forecast group nwp_conv_gfs: available <= prediction_time, issue <=
    prediction_time) with the frame rules of v1 (allowlist nwp_)."""
    from src.quality.temporal_checks import FeatureGroup, FrameConfig, check_frame_table
    g = [FeatureGroup(f"{CONV_PREFIX}gfs", "forecast", tuple(cols))]
    v = list(check_frame_table(df, g, cols, FrameConfig.from_yaml(PB.CFG_PATH)))
    H = pd.Timedelta(hours=22)
    present = df[cols].notna().any(axis=1)
    bad = present & ~(df[f"{CONV_PREFIX}gfs__issue_time"] <= df["prediction_time"] - H)
    if bad.any():
        v.append(f"forecast_issue_time > prediction_time - 22 h ({int(bad.sum())} rows)")
    return v
# ==== conv-end


def run_conv(unit, ctx: Ctx) -> tuple[list[str], dict]:
    """Convective columns for every table_v2 row of one month (city + ring, every frame), reused when present."""
    from src.gold import features as F
    cfg, y, m = ctx.cfg, unit.params["y"], unit.params["m"]
    keys = [ctx.conv_key(y, m, n) for n in ("part.parquet", "part.meta.json")]
    have = ctx.store.list(ctx.conv_key(y, m, ""))
    if all(k in have for k in keys):
        return [], {"reused": True, "conv_keys": keys}
    gk = ctx.gold_key(y, m)
    gl = ctx.store.list(gk)
    if gk not in gl:
        raise RuntimeError(f"{unit.id}: {gk} missing (Gold is reused, never built in P4FG)")
    gp = ctx.gold_local() / f"season={y}" / f"month={m:02d}" / "part.parquet"
    ctx._download(gk, gl[gk], gp)
    gcols = [f"gc_k{k}" for k in range(7)]
    A = pd.read_parquet(gp, columns=["pixel_id", "t0", "lead_min", "prediction_time", "frame_start", "target_time"]
                        + gcols)
    for c in ("t0", "prediction_time", "frame_start", "target_time"):
        A[c] = A[c].astype("datetime64[ns, UTC]")
    first = date(y, m, 1)
    days = [first - timedelta(days=2) + timedelta(days=k) for k in range(36)]
    days = [d for d in days if d < date(y + (m == 12), m % 12 + 1, 1) + timedelta(days=1)]
    sp = cfg["conv_store"]["silver_prefix"]
    rels = {d: f"{sp}date={d.isoformat()}/part.parquet" for d in days}
    ls = ctx.store.list(sp)
    cv_parts = []
    for d, k in rels.items():
        if k in ls:
            p = root() / "data" / "_scratch" / "p4fg" / "nwp_conv" / f"date={d.isoformat()}.parquet"
            ctx._download(k, ls[k], p)
            cv_parts.append(pd.read_parquet(p))
    month_days = [d for d in days if (d.year, d.month) == (y, m)]
    have_days = {d for d, k in rels.items() if k in ls}
    missing = [d.isoformat() for d in month_days if d not in have_days]
    if missing:
        raise RuntimeError(f"{unit.id}: Silver nwp_conv missing for {missing[:5]} ({len(missing)} days); run the job "
                           "nwpconv_backfill first (P4G.1)")
    cv = pd.concat(cv_parts, ignore_index=True)
    for c in ("time_utc", "forecast_issue_time_utc", "available_time_utc"):
        cv[c] = cv[c].astype("datetime64[ns, UTC]")
    pix = np.sort(A["pixel_id"].unique().astype("int64"))
    pts = cv[["pixel_id", "lat", "lon"]].drop_duplicates("pixel_id")
    pmap = F.nwp_point_map(pts, pix)
    from src.models import data as D
    rain_now = D.asof_latest(A)
    cols = conv_feature_names(cfg, "ac")
    nw = compute_conv_features(A[["pixel_id", "prediction_time", "frame_start"]], cv, pmap, rain_now)
    B = pd.concat([A[["pixel_id", "t0", "lead_min", "prediction_time", "frame_start", "target_time"]], nw[cols + [
        f"{CONV_PREFIX}gfs__issue_time", f"{CONV_PREFIX}gfs__available_time"]]], axis=1)
    v = conv_g2(B, cols)
    if v:
        raise RuntimeError(f"G2 violations in table_v3_conv {y}-{m:02d}: {[str(x) for x in v[:3]]} (P4G.1 stop)")
    for c in ("t0", f"{CONV_PREFIX}gfs__issue_time", f"{CONV_PREFIX}gfs__available_time"):
        B[c] = B[c].astype("datetime64[us, UTC]")
    lp = ctx.conv_local() / f"season={y}" / f"month={m:02d}" / "part.parquet"
    PB.write_parquet(B.drop(columns=["prediction_time", "frame_start", "target_time"]), lp)
    nan = {c: round(float(B[c].isna().mean()), 5) for c in cols}
    meta = {"table": "table_v3_conv", "gold_id": ctx.gid, "conv_id": conv_id(cfg), "season": y, "month": m,
            "rows": int(len(B)), "pixels": int(len(pix)), "points": int(len(pts)), "features": cols, "nan_share": nan,
            "g2_violations": 0, "source": cfg["pre_registered"]["convective"]["api"],
            "rule": cfg["pre_registered"]["convective"]["admissible"], "note": NOT_FLOOD}
    write_json(lp.with_name("part.meta.json"), meta)
    for name, k in zip(("part.parquet", "part.meta.json"), keys):
        ctx.put_own(k, (lp.parent / name).read_bytes())
    return [], {"rows": int(len(B)), "nan_share_max": max(nan.values()), "conv_keys": keys}


# ============================================================================ P4G.0 oracle series
def slot_rates(gs: pd.DataFrame, col: str) -> pd.DataFrame:
    """Mean rate of every half-hour slot [s, s+30) from the hourly GSMaP windows started every 30 min:
    0.5 x (r(window s-30) + r(window s)) (both windows cover the slot; NaN if either is missing). Index slot start,
    columns pixel_id."""
    piv = gs.pivot_table(index="window_start_utc", columns="pixel_id", values=col, aggfunc="mean")
    if not len(piv):
        return piv
    grid = pd.date_range(piv.index.min(), piv.index.max(), freq="30min")
    piv = piv.reindex(grid)
    return 0.5 * (piv.shift(1) + piv)


def nb3_mean(rate: pd.DataFrame, pixels: np.ndarray) -> pd.DataFrame:
    """3x3 neighbourhood mean (centre included, NaN neighbours and pixels outside the columns ignored) for `pixels`."""
    from src.gold import features as F
    G = F.Grid()
    cols = rate.columns.to_numpy("int64")
    arr = rate.to_numpy("float64")
    ci, cj = G.ij(cols)
    idx_of = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(ci, cj))}
    out = np.full((len(rate), len(pixels)), np.nan)
    pi, pj = G.ij(np.asarray(pixels, "int64"))
    for n, (a, b) in enumerate(zip(pi, pj)):
        ks = [idx_of[(a + di, b + dj)] for di in (-1, 0, 1) for dj in (-1, 0, 1) if (a + di, b + dj) in idx_of]
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                out[:, n] = np.nanmean(arr[:, ks], axis=1)
    return pd.DataFrame(out, index=rate.index, columns=pixels)


def run_orc(unit, ctx: Ctx) -> tuple[list[str], dict]:
    """P4G.0 oracle series of one month for every table pixel: target-frame GSMaP GC / raw mean rate of each slot
    [s, s+30) (ignoring availability) and its 3x3 mean. Run output (oracle = true), never a Gold table."""
    y, m = unit.params["y"], unit.params["m"]
    with env():
        ctx.fetch_gold()
        city, ring, _ = PD.pixel_sets(ctx)
    pix = np.sort(np.r_[city, ring].astype("int64"))
    readpix = np.array(sorted(set(_neighbour_ids(pix)) | set(pix.tolist())), "int64")
    first = date(y, m, 1)
    nxt = date(y + (m == 12), m % 12 + 1, 1)
    days = [first - timedelta(days=1) + timedelta(days=k) for k in range((nxt - first).days + 2)]
    gs = silver_frames(ctx, "gsmap_now", days, readpix, ["window_start_utc", "rain_rate_mm_h", "rain_rate_gc_mm_h"])
    if not len(gs):
        raise RuntimeError(f"{unit.id}: no GSMaP NOW Silver rows")
    gs["window_start_utc"] = gs["window_start_utc"].astype("datetime64[ns, UTC]")
    res = {}
    for name, col in (("gc", "rain_rate_gc_mm_h"), ("raw", "rain_rate_mm_h")):
        r = slot_rates(gs, col)
        res[f"{ORACLE_PREFIX}{name}_target"] = r.reindex(columns=pix)
        res[f"{ORACLE_PREFIX}{name}_target_nb3"] = nb3_mean(r, pix)
    idx = res[f"{ORACLE_PREFIX}gc_target"].index
    keep = (idx >= pd.Timestamp(first, tz="UTC")) & (idx < pd.Timestamp(nxt, tz="UTC") + pd.Timedelta(hours=2))
    T = int(keep.sum())
    df = pd.DataFrame({"pixel_id": np.tile(pix, T), "slot_start_utc": np.repeat(idx[keep], len(pix))})
    for k, v in res.items():
        df[k] = v.to_numpy("float32")[keep].ravel()
    df["slot_start_utc"] = df["slot_start_utc"].astype("datetime64[us, UTC]")
    df["oracle"] = True
    p = out_dir("oracle", "series") / f"{y}-{m:02d}.parquet"
    PB.write_parquet(df, p)
    return [rel(p)], {"rows": int(len(df)), "pixels": int(len(pix)),
                      "nan_share_gc": round(float(df[f"{ORACLE_PREFIX}gc_target"].isna().mean()), 4), "oracle": True}


def _neighbour_ids(pix: np.ndarray) -> list[int]:
    """Pixel ids of the 3x3 neighbourhoods (grid id = lat_idx * 3600 + lon_idx, as features.Grid)."""
    out = set()
    for p in pix:
        p = int(p)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                out.add(p + di * 3600 + dj)
    return sorted(out)


def oracle_cols(ctx: Ctx, meta: pd.DataFrame, L: int) -> np.ndarray:
    """Oracle features for the rows of `meta` (frame L): target slot start = t0 + L."""
    cols = [f"{ORACLE_PREFIX}gc_target", f"{ORACLE_PREFIX}raw_target", f"{ORACLE_PREFIX}gc_target_nb3",
            f"{ORACLE_PREFIX}raw_target_nb3"]
    parts = []
    for y, m in run_months(ctx.cfg):
        r = f"{OUT}/oracle/series/{y}-{m:02d}.parquet"
        if not (root() / r).is_file():
            ctx.fetch_run_files(r)
        parts.append(pd.read_parquet(root() / r, columns=["pixel_id", "slot_start_utc"] + cols))
    s = pd.concat(parts, ignore_index=True).drop_duplicates(["pixel_id", "slot_start_utc"])
    s["slot_start_utc"] = s["slot_start_utc"].astype("datetime64[ns, UTC]")
    k = pd.DataFrame({"pixel_id": meta["pixel_id"].to_numpy("int64"),
                      "slot_start_utc": (meta["t0"] + pd.Timedelta(minutes=int(L))).astype("datetime64[ns, UTC]")})
    j = k.merge(s, on=["pixel_id", "slot_start_utc"], how="left", validate="many_to_one")
    return j[cols].to_numpy("float32"), cols


def conv_cols(ctx: Ctx, meta: pd.DataFrame, L: int, cols: list[str]) -> np.ndarray:
    ctx.fetch_conv()
    parts = []
    for y, m in sorted(set(zip((meta["ym"] // 100).tolist(), (meta["ym"] % 100).tolist()))):
        p = ctx.conv_local() / f"season={y}" / f"month={m:02d}" / "part.parquet"
        if not p.is_file():
            raise RuntimeError(f"conv partition {y}-{m:02d} missing")
        parts.append(pd.read_parquet(p, columns=["pixel_id", "t0"] + cols, filters=[("lead_min", "==", int(L))]))
    c = pd.concat(parts, ignore_index=True)
    c["t0"] = c["t0"].astype("datetime64[ns, UTC]")
    j = meta[["pixel_id", "t0"]].merge(c, on=["pixel_id", "t0"], how="left", validate="one_to_one")
    if len(j) != len(meta):
        raise RuntimeError("conv join changed the row count")
    return j[cols].to_numpy("float32")


# ============================================================================ data frames of a fit
def base_features(ctx: Ctx, ring: bool) -> list[str]:
    with env():
        ctx.fetch_gold()
        f = PD.feature_list(ctx, ring)
    ctx.fetch_nwp()
    return f + PE.nwp_feature_list(ctx)


def spec_features(ctx: Ctx, spec: str) -> list[str]:
    dd = spec_def(spec)
    f = base_features(ctx, dd["ring"])
    if dd["conv"]:
        f = f + conv_feature_names(ctx.cfg, dd["conv"])
    if dd.get("oracle"):
        f = f + [f"{ORACLE_PREFIX}gc_target", f"{ORACLE_PREFIX}raw_target", f"{ORACLE_PREFIX}gc_target_nb3",
                 f"{ORACLE_PREFIX}raw_target_nb3"]
    return f


def assert_no_oracle(feats: list[str], spec: str) -> None:
    """P4G.0: no oracle feature may reach a non-oracle model (TASK stop condition)."""
    if spec != "orc" and any(ORACLE_PREFIX in c for c in feats):
        raise RuntimeError(f"oracle feature in non-oracle spec {spec}: stop (P4G.0)")


def _smoke_mask(meta: pd.DataFrame, cfg: dict) -> np.ndarray | None:
    sd = smoke_days(cfg)
    if sd is None:
        return None
    d = meta["t0"].dt.tz_convert("UTC").dt.date
    return d.isin(sd).to_numpy()


def load(ctx: Ctx, L: int, feats: list[str], months: list | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """Rows of frame L (set-B months; smoke: smoke days only) with `feats` in that column order: P4E load_frame for
    the Gold + NWP columns, then the convective / oracle columns joined on (pixel_id, t0)."""
    extra = [c for c in feats if c.startswith((CONV_PREFIX, ORACLE_PREFIX))]
    basef = [c for c in feats if c not in extra]
    if feats[:len(basef)] != basef:
        raise ValueError("feature order: Gold / NWP columns first, then conv / oracle")

    def loader():
        with env():
            meta, X = PE.load_frame(ctx, L, basef, months or PE.nwp_months(ctx.cfg))
        mk = _smoke_mask(meta, ctx.cfg)
        if mk is not None:
            meta, X = meta[mk].reset_index(drop=True), X[mk]
        return meta, X
    meta, X = ctx.cache.get((L, tuple(basef), tuple(months or ())), loader)
    if not extra:
        return meta, X
    blocks = []
    cc = [c for c in extra if c.startswith(CONV_PREFIX)]
    if cc:
        blocks.append(conv_cols(ctx, meta, L, cc))
    oc = [c for c in extra if c.startswith(ORACLE_PREFIX)]
    if oc:
        a, names = oracle_cols(ctx, meta, L)
        if names != oc:
            raise ValueError("oracle column order")
        blocks.append(a)
    return meta, np.hstack([X] + blocks)


# ============================================================================ regional target (P4G.2)
def grid_max(meta: pd.DataFrame, v: np.ndarray) -> np.ndarray:
    """Max of v over the 3x3 neighbourhood (same t0) among the loaded pixels; NaN neighbours ignored, all-NaN -> NaN."""
    from src.gold import features as F
    i, j = F.Grid().ij(meta["pixel_id"].to_numpy("int64"))
    i, j = i - i.min() + 1, j - j.min() + 1
    codes, uniq = pd.factorize(meta["t0"])
    g = np.full((len(uniq), i.max() + 2, j.max() + 2), -np.inf)
    g[codes, i, j] = np.where(np.isnan(v), -np.inf, v)
    nb = np.full_like(g, -np.inf)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            nb[:, 1:-1, 1:-1] = np.maximum(nb[:, 1:-1, 1:-1], g[:, 1 + di:g.shape[1] - 1 + di, 1 + dj:g.shape[2] - 1 + dj])
    out = nb[codes, i, j]
    return np.where(np.isfinite(out), out, np.nan)


def regional_label(meta: pd.DataFrame, thr: float) -> np.ndarray:
    lab = meta["label_rate_mm_h"].to_numpy("float64")
    m = grid_max(meta, lab)
    return (np.nan_to_num(m, nan=-1.0) >= thr).astype("int8")


def pers_value(X: np.ndarray, feats: list[str], L: int) -> np.ndarray:
    """Persistence value of every row from the gc_k columns (as p4b.persistence: latest for F0/F30, max k0-k2 F60)."""
    from src.models import data as D
    df = pd.DataFrame({f"gc_k{k}": X[:, feats.index(f"gc_k{k}")] for k in range(7)})
    return PB.persistence(df, L) if L in (0, 30) else D.max_avail_0_60(df)


def regional_clim(meta: pd.DataFrame, y: np.ndarray, train: np.ndarray, held: np.ndarray, L: int) -> np.ndarray:
    """Training-fold mean of the regional label by local month x local hour of the frame start (city rows), P4B
    climatology rule; cells without training rows -> overall training mean."""
    fs = (meta["t0"] + pd.Timedelta(minutes=int(L))).dt.tz_convert("Asia/Ho_Chi_Minh")
    key = (fs.dt.month * 100 + fs.dt.hour).to_numpy()
    tab = pd.Series(y[train].astype("float64")).groupby(key[train]).mean()
    pc = tab.reindex(key[held]).to_numpy("float64")
    return np.where(np.isnan(pc), float(y[train].mean()) if train.any() else np.nan, pc)


# ============================================================================ fits
def hp_choice(ctx: Ctx, s: int) -> tuple[int, int]:
    d = ctx.read_json(f"{OUT}/compare/hp_select.json")
    return tuple(d["folds"][str(int(s))]["setting"])


def run_fit(unit, ctx: Ctx) -> tuple[list[str], dict]:
    from src.gold import p4a_common as P
    from src.models import metrics as M
    cfg = ctx.cfg
    spec, s, L, thr = unit.params["spec"], unit.params["season"], unit.params["lead"], unit.params["thr"]
    dd = spec_def(spec)
    t1 = time.time()
    hp = dd["hp"]
    resolved = spec
    if hp == "tuned":
        hp = hp_choice(ctx, s)
        resolved = spec_name(hp, dd["conv"])
        if hp == DEFAULT_HP:                           # the fold chose g1: alias of the B0 conv fit (same model)
            info = {"unit": unit.id, "spec": spec, "alias": fit_id(resolved, s, L, thr), "resolved_spec": resolved,
                    "setting": list(hp), "heldout_season": s, "lead_min": L, "threshold": thr, "note": NOT_FLOOD}
            p = out_dir("fit") / f"{unit.id}.json"
            write_json(p, info)
            return [rel(p)], {"alias": info["alias"], "seconds": round(time.time() - t1, 1)}
    feats = spec_features(ctx, spec)
    assert_no_oracle(feats, spec)
    meta, X = load(ctx, L, feats)
    city = meta["is_city"].to_numpy("float32") == 1
    trainable = (meta["season"].to_numpy() != s) & meta["train_month"].to_numpy(bool)
    trainable &= (meta["t0"] >= pd.Timestamp(PE.nwp_from(cfg), tz="UTC")).to_numpy()
    held = (meta["season"].to_numpy() == s) & city
    keep = trainable & (city if not dd["ring"] else np.ones(len(city), bool))
    inner_eval = trainable & city
    if dd.get("regional"):
        y = regional_label(meta, thr)
    else:
        y = meta[thr_col(thr)].to_numpy("int8")
    params = PE.lgb_params(cfg) | {"num_leaves": int(hp[0]), "min_data_in_leaf": int(hp[1])}
    ic = cfg["inner_cv"]
    block = cfg["set_b"]["inner_cv_block"]
    cv = PE.cv_fit(X, y, None, keep, inner_eval, meta, params, ic, feats, block)
    o = PE.inner_oof(cv, X)
    iy = np.concatenate([y[ho] for ho, _ in o.values()]).astype(bool)
    ip = np.concatenate([p for _, p in o.values()])
    bst = PE.refit(X, y, None, keep, cv["median"], params, feats)
    p_out = bst.predict(X[held])
    work = cv["work"] + int(keep.sum()) * cv["median"]
    platt = PB.fit_calibrator("platt", ip, iy)
    ipc = PB.apply_calibrator(platt, ip)
    rule = "max_csi" if dd.get("regional") else cfg["evaluation"]["cut_rules"][f"{thr:g}"]
    cut = M.choose_cut(iy, ipc, rule)
    oof = pd.DataFrame({"pixel_id": meta["pixel_id"].to_numpy("int64")[held],
                        "t0": meta.loc[held, "t0"].reset_index(drop=True).astype("datetime64[us, UTC]"),
                        "p_raw": p_out.astype("float32")})
    if dd.get("regional"):
        pv = pers_value(X, feats, L)
        prg = grid_max(meta, pv)
        oof["y_reg"] = y[held]
        oof["pers_reg"] = prg[held].astype("float32")
        oof["pclim_reg"] = regional_clim(meta, y, inner_eval, held, L).astype("float32")
    if dd.get("oracle"):
        oof["oracle"] = True
    po = (out_dir("oracle", "oof") if dd.get("oracle") else out_dir("oof")) / f"{unit.id}.parquet"
    PB.write_parquet(oof, po)
    rels = [rel(po)]
    if dd["ring"] and not dd.get("oracle") and not dd.get("regional"):
        rn = ring_near_mask(meta, cfg) & (meta["season"].to_numpy() == s)
        if rn.any():                                   # REPLAY edge rows (ring pixels within Chebyshev 2), held out
            pr = pd.DataFrame({"pixel_id": meta["pixel_id"].to_numpy("int64")[rn],
                               "t0": meta.loc[rn, "t0"].reset_index(drop=True).astype("datetime64[us, UTC]"),
                               "p_raw": bst.predict(X[rn]).astype("float32")})
            pq_ = out_dir("oof_ring") / f"{unit.id}.parquet"
            PB.write_parquet(pr, pq_)
            rels.append(rel(pq_))
    used = [f for f in cv["folds"] if f.get("used")]
    info = {"unit": unit.id, "spec": spec, "resolved_spec": resolved, "dataset": {k: (list(v) if isinstance(v, tuple)
            else v) for k, v in dd.items()}, "setting": list(hp), "heldout_season": s, "lead_min": L,
            "threshold": thr, "oracle": bool(dd.get("oracle")), "regional": bool(dd.get("regional")),
            "labels": "training: IMERG Final V07 rows only; held-out: IMERG Final (Late for 2025-10, secondary)",
            "features": feats, "n_features": len(feats), "lgb_params": params,
            "inner_cv": {k: ic[k] for k in ("max_rounds", "early_stopping_rounds", "buffer_days")} | {"block": block},
            "inner_folds": cv["folds"], "inner_fold_best_iterations": cv["best"], "median_rounds": cv["median"],
            "platt": platt, "cut": cut, "cut_rule": rule, "train_rows": int(keep.sum()),
            "train_rows_ring": int((keep & ~city).sum()), "train_pos": int(y[keep].sum()),
            "held_rows": int(held.sum()), "smoke_days": bool(smoke_days(cfg)),
            "inner_oof": {"n": int(len(iy)), "base_rate": float(iy.mean()), "logloss_raw": PB.logloss(iy, ip),
                          "logloss_cal": PB.logloss(iy, ipc), "brier_raw": M.brier(iy, ip), "brier_cal": M.brier(iy, ipc)},
            "rounds": {"max_rounds": int(ic["max_rounds"]), "inner_at_cap": int(sum(f["at_cap"] for f in used)),
                       "inner_used": len(used), "median_share_of_cap": cv["median"] / int(ic["max_rounds"])},
            "work_grows": work / 1e9, "note": NOT_FLOOD}
    if dd["conv"]:
        info["conv"] = {"groups": dd["conv"], "conv_id": conv_id(cfg)}
    if dd.get("oracle"):
        info["note"] = "ORACLE (deliberately leaky diagnostic, never served); " + NOT_FLOOD
    ip_ = (out_dir("oracle", "fit") if dd.get("oracle") else out_dir("fit")) / f"{unit.id}.json"
    write_json(ip_, info)
    rss = P.rss_gb()
    if np.isfinite(rss) and rss > float(cfg["resources"]["max_rss_gb"]):
        raise RuntimeError(f"{unit.id}: RSS {rss:.2f} GB above resources.max_rss_gb")
    return rels + [rel(ip_)], {"median_rounds": cv["median"], "train_rows": int(keep.sum()), "rss_gb": round(rss, 2),
                               "work_grows": round(work / 1e9, 4), "seconds": round(time.time() - t1, 1)}


def ring_near_mask(meta: pd.DataFrame, cfg: dict) -> np.ndarray:
    """Ring pixels with Chebyshev distance <= replay.ring_chebyshev_max to the nearest city pixel."""
    from src.gold import features as F
    k = int(cfg["pre_registered"]["replay"]["ring_chebyshev_max"])
    pix = meta[["pixel_id", "is_city"]].drop_duplicates("pixel_id")
    city = pix.loc[pix["is_city"].to_numpy("float32") == 1, "pixel_id"].to_numpy("int64")
    ring = pix.loc[pix["is_city"].to_numpy("float32") != 1, "pixel_id"].to_numpy("int64")
    G = F.Grid()
    ci, cj = G.ij(city)
    ri, rj = G.ij(ring)
    d = np.maximum(np.abs(ri[:, None] - ci[None, :]), np.abs(rj[:, None] - cj[None, :])).min(1) if len(ring) else []
    near = set(ring[np.asarray(d) <= k].tolist()) if len(ring) else set()
    return meta["pixel_id"].isin(near).to_numpy()


# ============================================================================ base rows (smoke-day aware)
def run_base(unit, ctx: Ctx) -> tuple[list[str], dict]:
    """P4D base rows of one fold and frame (city, eval = training months, secondary = 2025-10 Late); smoke: smoke
    days only (climatology from the smoke training days)."""
    from src.gold import build_table_v2 as BT2
    cfg, s, L = ctx.cfg, unit.params["season"], unit.params["lead"]
    with env():
        ctx.fetch_gold()
        parts_ = PD.gold_parts(ctx)
    ycols = [thr_col(t) for t in PD.thresholds(cfg)]
    sd = smoke_days(cfg)
    parts = []
    for y, m, part in parts_:
        df = pd.read_parquet(part, columns=list(dict.fromkeys(PB.KEYS + ["is_city", "label_rate_mm_h"] + ycols +
                                                              PD.GK)), filters=[("lead_min", "==", L)])
        df = df[df["is_city"] == 1]
        if sd is not None:
            df = df[df["t0"].dt.tz_convert("UTC").dt.date.isin(sd)]
        df["train_month"] = BT2.is_training_month(cfg, y, m)
        df["label_source"] = BT2.label_source(cfg, y, m)
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    for c in ("t0", "frame_start"):
        df[c] = df[c].dt.tz_convert("UTC")
    train = df[(df["season"] != s) & df["train_month"]]
    held = df[df["season"] == s].reset_index(drop=True)
    utc = held["t0"]
    out = pd.DataFrame({"pixel_id": held["pixel_id"].to_numpy("int64"), "t0": utc,
                        "day": utc.dt.tz_convert(PD.LOCAL_TZ).dt.date.astype(str).to_numpy(),
                        "half": [f"{a:02d}-{'1' if b <= 15 else '2'}" for a, b in zip(utc.dt.month, utc.dt.day)],
                        "role": np.where(held["train_month"], "eval", "secondary"),
                        "label_source": held["label_source"].to_numpy(),
                        "label_rate_mm_h": held["label_rate_mm_h"].to_numpy("float32"),
                        "pers": PB.persistence(held, L).astype("float32")})
    for t in PD.thresholds(cfg):
        out[thr_col(t)] = held[thr_col(t)].to_numpy("int8")
        out[f"pclim_{kstr(t)}"] = PB.clim_for(train, held, thr_col(t)).astype("float32")
    p = out_dir("base") / f"s{s}_lead{L}.parquet"
    PB.write_parquet(out, p)
    return [rel(p)], {"rows": len(out), "rows_eval": int((out["role"] == "eval").sum()),
                      "rows_secondary": int((out["role"] == "secondary").sum()), "train_rows_clim": len(train)}


# ============================================================================ P4F.0 cloud dt-check
def dt_verdict(code_dt_h: float, ratio: float, rule: dict) -> str:
    """Pre-registered P4F.0 verdict (D214)."""
    lo, hi = rule["ok_ratio"]
    blo, bhi = rule["bug_ratio"]
    if code_dt_h != 0.5 or (np.isfinite(ratio) and blo <= ratio <= bhi):
        return "dt_bug"
    if np.isfinite(ratio) and lo <= ratio <= hi:
        return "ok"
    return "needs_review"


def run_dtcheck(unit, ctx: Ctx) -> tuple[list[str], dict]:
    """Season totals over the same half-hour slots (GC, raw and IMERG Final all present) at the 47 city pixels:
    IMERG = rate x 0.5 h; GSMaP slot = 0.25 x (r(s-30) + r(s)) mm (code Δt 0.5 h per file) and x 2 for Δt = 1 h."""
    cfg = ctx.cfg
    rule = cfg["pre_registered"]["dt_check"]
    with env():
        ctx.fetch_gold()
        city, _, _ = PD.pixel_sets(ctx)
    city = np.sort(np.asarray(city, "int64"))
    rows = []
    for a, b in rule["periods"]:
        season = int(a[:4])
        tot = {"slots": 0, "imerg": 0.0, "gc": 0.0, "raw": 0.0, "days": 0, "days_missing": 0}
        for d in pd.date_range(a, b, freq="D"):
            d = d.date()
            gs = silver_frames(ctx, "gsmap_now", [d - timedelta(days=1), d], city,
                               ["window_start_utc", "rain_rate_mm_h", "rain_rate_gc_mm_h"])
            im = silver_frames(ctx, "imerg_final", [d], city, ["interval_start_utc", "rain_rate_mm_h"])
            if not len(gs) or not len(im):
                tot["days_missing"] += 1
                continue
            gs["window_start_utc"] = gs["window_start_utc"].astype("datetime64[ns, UTC]")
            im["interval_start_utc"] = im["interval_start_utc"].astype("datetime64[ns, UTC]")
            g = slot_rates(gs, "rain_rate_gc_mm_h").reindex(columns=city)
            r = slot_rates(gs, "rain_rate_mm_h").reindex(columns=city)
            i = im.pivot_table(index="interval_start_utc", columns="pixel_id", values="rain_rate_mm_h",
                               aggfunc="mean").reindex(columns=city)
            day0 = pd.Timestamp(d, tz="UTC")
            slots = pd.date_range(day0, periods=48, freq="30min")
            g, r, i = g.reindex(slots), r.reindex(slots), i.reindex(slots)
            ok = g.notna().to_numpy() & r.notna().to_numpy() & i.notna().to_numpy()
            tot["slots"] += int(ok.sum())
            tot["imerg"] += float(np.where(ok, i.to_numpy(), 0).sum() * 0.5)
            tot["gc"] += float(np.where(ok, g.to_numpy(), 0).sum() * GSMAP_SLOT_DT_H)
            tot["raw"] += float(np.where(ok, r.to_numpy(), 0).sum() * GSMAP_SLOT_DT_H)
            tot["days"] += 1
        n = len(city)
        rows.append({"season": season, "period": f"{a}..{b}", "pixels": n, "same_slots": tot["slots"],
                     "days": tot["days"], "days_missing": tot["days_missing"],
                     "sum_imerg_final_mm": tot["imerg"] / n, "sum_gc_mm_dt0p5": tot["gc"] / n,
                     "sum_gc_mm_dt1": 2 * tot["gc"] / n, "sum_raw_mm_dt0p5": tot["raw"] / n,
                     "sum_raw_mm_dt1": 2 * tot["raw"] / n,
                     "ratio_gc_imerg_dt0p5": tot["gc"] / tot["imerg"] if tot["imerg"] else np.nan,
                     "ratio_gc_imerg_dt1": 2 * tot["gc"] / tot["imerg"] if tot["imerg"] else np.nan,
                     "ratio_raw_imerg_dt0p5": tot["raw"] / tot["imerg"] if tot["imerg"] else np.nan,
                     "_gc": tot["gc"], "_raw": tot["raw"], "_imerg": tot["imerg"]})
    t = pd.DataFrame(rows)
    pooled = float(t["_gc"].sum() / t["_imerg"].sum()) if t["_imerg"].sum() else float("nan")
    verdict = dt_verdict(GSMAP_SLOT_DT_H, pooled, rule)
    out = out_dir("dtcheck")
    t.drop(columns=["_gc", "_raw", "_imerg"]).to_csv(out / "dt_check_seasons.csv", index=False)
    res = {"code_dt_h_per_30min_file": GSMAP_SLOT_DT_H, "pooled_ratio_gc_imerg_code_dt": pooled,
           "pooled_ratio_gc_imerg_dt1": 2 * pooled, "pooled_ratio_raw_imerg_code_dt":
           float(t["_raw"].sum() / t["_imerg"].sum()) if t["_imerg"].sum() else None,
           "verdict": verdict, "rule": rule, "seasons": t.drop(columns=["_gc", "_raw", "_imerg"]).to_dict("records"),
           "note": "P4F.0 cloud check; " + NOT_FLOOD}
    write_json(out / "dt_check.json", res)
    return [rel(out / "dt_check_seasons.csv"), rel(out / "dt_check.json")], {"verdict": verdict,
                                                                             "pooled_ratio": round(pooled, 4)}


# ============================================================================ reuse-only Gold / NWP, gate g2, dispatch
def run_reuse(unit, ctx: Ctx) -> tuple[list[str], dict]:
    """feat / nwp: the partition must already exist on R2 (P4FG never builds table_v2*: writes there are not
    approved for P4F/P4G)."""
    y, m = unit.params["y"], unit.params["m"]
    if unit.kind == "feat":
        keys = [ctx.gold_key(y, m, n) for n in ("part.parquet", "part.meta.json")]
        have = ctx.store.list(ctx.gold_key(y, m, ""))
    else:
        keys = [ctx.nwp_key(y, m, n) for n in ("part.parquet", "part.meta.json")]
        have = ctx.store.list(ctx.nwp_key(y, m, ""))
    miss = [k for k in keys if k not in have]
    if miss:
        raise RuntimeError(f"{unit.id}: {miss[0]} missing; P4FG reuses table_v2 / table_v2_nwp and never builds them "
                           "(R2 writes approved only under table_v3*/): stop")
    return [], {"reused": True, "keys": keys, "bytes": int(sum(have[k] for k in keys))}


def setb_label_check(cfg: dict, lab: pd.DataFrame) -> list[dict]:
    """Every fit (fold, frame, threshold) needs >= 2 usable inner blocks: held-out block with a positive city row AND
    training rows (other blocks, buffer) with a positive city row (set B, >= nwp_from)."""
    buf = pd.Timedelta(days=int(cfg["inner_cv"]["buffer_days"]))
    blk = cfg["set_b"]["inner_cv_block"]
    t = lab["t0"].dt.tz_convert("UTC")
    key = ((t.dt.year * 100 + t.dt.month) if blk == "calendar_month" else
           (t.dt.year * 10000 + t.dt.month * 100 + t.dt.day)).to_numpy()
    nb = (lab["t0"] >= pd.Timestamp(PE.nwp_from(cfg), tz="UTC")).to_numpy()
    rows = []
    for s in folds(cfg):
        trainable = (lab["season"].to_numpy() != s) & lab["train_month"].to_numpy(bool) & nb
        for L in leads(cfg):
            lm = lab["lead_min"].to_numpy() == L
            for thr in thresholds(cfg):
                y = lab[thr_col(thr)].to_numpy() > 0
                usable = []
                for k in sorted(set(key[trainable & lm].tolist())):
                    a, b = PE._key_bounds(k, blk)
                    ho = trainable & lm & (key == k)
                    tr = trainable & lm & (key != k) & ((t < a - buf) | (t >= b + buf)).to_numpy()
                    if y[ho].any() and y[tr].any():
                        usable.append(int(k))
                rows.append({"heldout_season": s, "lead_min": L, "threshold": thr, "block": blk,
                             "usable_blocks": usable, "ok": len(usable) >= 2})
    return rows


def gate_g2(ctx: Ctx) -> tuple[list[str], dict]:
    """P4D G2 on the reused Gold (partitions, violations, NaN shares, A1 equality) + set-B NWP NaN shares, convective
    partitions (G2 = 0, NaN shares) and the set-B inner-CV label check of this run's fits, before any fit."""
    from src.gold import build_table_v2 as BT2
    cfg = ctx.cfg
    rels, m = PD.gate_g2(ctx)
    ctx.fetch_nwp()
    ctx.fetch_conv()
    out = out_dir("g2")
    nan, conv = [], []
    for y, mo in PE.nwp_months(cfg):
        mp = ctx.nwp_local() / f"season={y}" / f"month={mo:02d}" / "part.meta.json"
        if not mp.is_file():
            raise RuntimeError(f"G2 gate FAIL: set-B NWP partition {y}-{mo:02d} missing")
        d = json.loads(mp.read_text(encoding="utf-8"))
        nan.append({"partition": f"{y}-{mo:02d}", "rows": d["rows"], **d["nan_share"]})
    for y, mo in run_months(cfg):
        mp = ctx.conv_local() / f"season={y}" / f"month={mo:02d}" / "part.meta.json"
        if not mp.is_file():
            raise RuntimeError(f"G2 gate FAIL: convective partition {y}-{mo:02d} missing")
        d = json.loads(mp.read_text(encoding="utf-8"))
        if int(d["g2_violations"]) != int(cfg["pre_registered"]["convective"]["g2_violations_max"]):
            raise RuntimeError(f"G2 gate FAIL: convective {y}-{mo:02d} has G2 violations (P4G.1 stop)")
        conv.append({"partition": f"{y}-{mo:02d}", "rows": d["rows"], "points": d["points"], **d["nan_share"]})
    ycols = [thr_col(t) for t in PD.thresholds(cfg)]
    sd = smoke_days(cfg)
    labs = []
    for y, mo, part in PD.gold_parts(ctx):
        df = pd.read_parquet(part, columns=["season", "t0", "lead_min", "is_city"] + ycols)
        df = df[df["is_city"] == 1].drop(columns="is_city")
        if sd is not None:
            df = df[df["t0"].dt.tz_convert("UTC").dt.date.isin(sd)]
        df["train_month"] = BT2.is_training_month(cfg, y, mo)
        labs.append(df)
    lab = pd.concat(labs, ignore_index=True)
    lab["t0"] = lab["t0"].astype("datetime64[ns, UTC]")
    chk = setb_label_check(cfg, lab)
    pd.DataFrame(nan).to_csv(out / "nwp_nan_share.csv", index=False)
    pd.DataFrame(conv).to_csv(out / "conv_nan_share.csv", index=False)
    pd.DataFrame(chk).to_csv(out / "set_b_inner_cv_check.csv", index=False)
    bad = [r for r in chk if not r["ok"]]
    if bad:
        raise RuntimeError(f"G2 gate FAIL: set-B inner CV has < 2 usable blocks for {bad[:3]}; no fit was started")
    return rels + [rel(out / n) for n in ("nwp_nan_share.csv", "conv_nan_share.csv", "set_b_inner_cv_check.csv")], \
        m | {"conv_partitions": len(conv), "conv_g2_violations": 0, "set_b_inner_cv_ok": True}


def run_unit(unit, ctx: Ctx) -> tuple[list[str], dict]:
    from src.models import p4fg_gates as G
    from src.models import p4fg_replay as R
    t1 = time.time()
    with env():
        if unit.kind == "gate":
            if unit.id == "preflight":
                return PD.gate_preflight(ctx)
            if unit.id == "g2":
                return gate_g2(ctx)
            if unit.id == "finalfit-smoke":
                return R.gate_finalfit_smoke(ctx)
            if unit.id == "replay-smoke":
                return R.gate_replay_smoke(ctx)
            return G.GATES[unit.id](ctx)
        if unit.kind == "fit":
            return run_fit(unit, ctx)
        if unit.kind in ("feat", "nwp"):
            rels, met = run_reuse(unit, ctx)
        elif unit.kind == "conv":
            rels, met = run_conv(unit, ctx)
        elif unit.kind == "orc":
            rels, met = run_orc(unit, ctx)
        elif unit.kind == "base":
            rels, met = run_base(unit, ctx)
        elif unit.kind == "dtcheck":
            rels, met = run_dtcheck(unit, ctx)
        elif unit.kind == "fssv1":
            rels, met = G.run_fssv1(unit, ctx)
        else:
            raise ValueError(unit.kind)
    return rels, {**met, "seconds": round(time.time() - t1, 1)}
