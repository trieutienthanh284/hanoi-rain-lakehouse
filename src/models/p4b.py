"""P4B — rainfall nowcast v1: the units of the cloud jobs `p4b` and `p4b_smoke` (driver: src/models/cloud_train.py).

Not a flood model. Labels are IMERG (satellite estimate): Final V07 for training / CV / test, Late V07 only for the
secondary October-2025 check (D93). Evaluation definition: configs/model_v1.yaml (scope) and REPORT_P4B §0.

Stages (a unit runs only when all its dependencies have a done marker on R2; `--part i/n` splits every stage
round-robin; gate units are run by whichever part first sees their dependencies done):
  1  feat-YYYY-MM      Gold table_A1_v1 (+ table_B1_v1 from scope.nwp_from) for one month: labels (Final, or Late
                       for months in scope.label_source_by_month), v1 features (src/gold/features.py), NWP features;
                       frame-mode G2 on the partition before upload (a violation fails the unit).
     g2                gate: G2 again on every partition, duplicates, and v1 == pilot table_A on the common columns.
  3  cv-<spec>-...     blocked CV (one fold per calendar month, 1-day buffer), pooled OOF predictions:
                       a (pilot table_A, P4A parameters, all thresholds), b (no doy), c (+ motion, advection,
                       upwind), d (+ extent), dm (+ month) at the primary threshold.
     sel-features      gate: cal_month only if the mean OOF log-loss over the frames decreases (F* = d or dm).
  4  cv-g<n>-...       the 8 grid settings on F* at the primary threshold.
     sel-setting       gate: lowest mean pooled OOF log-loss.
  5  cv-e / cv-f0      F* with the chosen setting / with P4A parameters at the other thresholds;
     cv-A24 / cv-B24   F* and F* + NWP on the months with NWP history (same folds, same rows).
     oof-select        gate: parameters per threshold, calibration (cross-fitted raw / Platt / isotonic), cuts,
                       7.6 rule, B-vs-A ship rule.
  6  refit-<A|B>-...   final models on the whole CV period (rounds = median of the fold optima).
     oof-report        gate: ablation table (a)-(g) + 2023 ablation, fold spread, OOF summary. STOP POINT.
  7  final-test        P4B.6, only with --final (job p4b; guard: lake/v1/reports/p4b/OOF_DONE.json with
                       final_approved = true for this run id). In p4b_smoke it runs on 2024 pseudo-test days.

Gold layout on R2: lake/v1/gold/met_training/<table>/gid=<gold id>/season=YYYY/month=MM/part.parquet, gold id =
hash of the feature code and configuration (new code -> new prefix, nothing is ever overwritten).
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import os
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CODE_ROOT = Path(__file__).resolve().parents[2]   # code and configs
ROOT = CODE_ROOT                                    # workspace (data, outputs); tests may redirect it
CFG_PATH = CODE_ROOT / "configs" / "model_v1.yaml"
LAKE = "lake/v1/"
GOLD_REMOTE = LAKE + "gold/met_training/"
OOF_DONE_KEY = LAKE + "reports/p4b/OOF_DONE.json"
PILOT_REL = "data/gold/met_training/pilot_v0/"
GOLD_BUILD_VERSION = "p4b-gold-1"
TABLES = {"A1": "table_A1_v1", "B1": "table_B1_v1"}
LOCAL_TZ = "Asia/Ho_Chi_Minh"
KEYS = ["pixel_id", "t0", "lead_min", "frame_start", "season"]
LABELS = ["label_rate_mm_h", "y_0p5", "y_2p5", "y_7p6"]
V1_NEW = ("motion_", "adv_", "upwind_", "extent_", "cal_month")
MOTION_PREFIXES = ("motion_", "adv_", "upwind_")
CODE_FILES = ["src/models/cloud_train.py", "src/models/p4b.py", "src/gold/features.py", "src/gold/build_labels.py",
              "src/gold/build_training_table.py", "src/gold/build_table_B.py", "src/models/metrics.py",
              "src/models/baselines.py", "src/models/data.py", "src/quality/temporal_checks.py",
              "configs/model_v1.yaml"]
GOLD_CODE = ["src/gold/features.py", "src/gold/build_labels.py", "src/gold/build_training_table.py",
             "src/gold/build_table_B.py"]


# ============================================================================ configuration and periods
def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_cfg(smoke: bool = False, path: Path | None = None) -> dict:
    cfg = yaml.safe_load(Path(path or CFG_PATH).read_text(encoding="utf-8"))
    sm = cfg.pop("smoke", {})
    if smoke:
        cfg = deep_merge(cfg, sm)
        cfg["smoke"] = True
        for name in ("test", "secondary"):                    # a smoke run never reads the 2025 season
            assert pd.Timestamp(cfg["scope"][name][1]) < pd.Timestamp("2025-01-01"), name
        for a, b in cfg["scope"]["cv_periods"]:
            assert pd.Timestamp(b) < pd.Timestamp("2025-01-01")
    return cfg


def kstr(t: float) -> str:
    return f"{t:g}".replace(".", "p")


def thr_col(t: float) -> str:
    return "y_" + kstr(t)


def thresholds(cfg: dict) -> list[float]:
    return [float(t) for t in cfg["targets"]["thresholds_mm_h"]]


def primary(cfg: dict) -> float:
    return float(cfg["targets"]["primary_mm_h"])


def _ts(s) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def period(cfg: dict, name: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    a, b = cfg["scope"][name]
    return _ts(a), _ts(b) + pd.Timedelta("1D")                 # [start, end) on t0


def cv_folds(cfg: dict, restricted: bool = False) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Fold periods [start, end): explicit scope.cv_periods (smoke) or the season months of scope.cv."""
    if cfg["scope"].get("cv_periods"):
        folds = [(_ts(a), _ts(b) + pd.Timedelta("1D")) for a, b in cfg["scope"]["cv_periods"]]
    else:
        lo, hi = period(cfg, "cv")
        folds, m = [], lo.replace(day=1)
        while m < hi:
            nxt = m + pd.offsets.MonthBegin(1)
            if m.month in cfg["scope"]["season_months"]:
                folds.append((max(m, lo), min(nxt, hi)))
            m = nxt
    if restricted:
        b0 = _ts(cfg["cv"]["b_months_from"])
        folds = [f for f in folds if f[0] >= b0]
    return folds


def data_days(cfg: dict) -> list[date]:
    ps = cv_folds(cfg) + [period(cfg, "test"), period(cfg, "secondary")]
    days = set()
    for a, b in ps:
        d = a.date()
        while d < b.date():
            if d.month in cfg["scope"]["season_months"]:
                days.add(d)
            d += timedelta(days=1)
    n = cfg["scope"].get("max_days_per_month")
    out = sorted(days)
    if n:
        out = [d for d in out if sum(1 for x in out if (x.year, x.month) == (d.year, d.month) and x <= d) <= n]
    return out


def build_months(cfg: dict) -> list[tuple[int, int]]:
    return sorted({(d.year, d.month) for d in data_days(cfg)})


def label_source(cfg: dict, y: int, m: int) -> str:
    return cfg["scope"].get("label_source_by_month", {}).get(f"{y}-{m:02d}", cfg["scope"]["label_source_default"])


def has_nwp(cfg: dict, y: int, m: int) -> bool:
    return pd.Timestamp(year=y, month=m, day=1) >= pd.Timestamp(cfg["scope"]["nwp_from"])


def grid_settings(cfg: dict) -> list[dict]:
    g = cfg["grid"]
    return [{"num_leaves": a, "min_data_in_leaf": b, "lambda_l2": c}
            for a, b, c in itertools.product(g["num_leaves"], g["min_data_in_leaf"], g["lambda_l2"])]


def lgb_params(cfg: dict, setting: str) -> dict:
    """setting: 'p4a' or 'g<n>' (1-based grid index)."""
    if setting == "p4a":
        p = dict(cfg["lightgbm_p4a"])
    else:
        p = dict(cfg["grid"]["fixed"]) | grid_settings(cfg)[int(setting[1:]) - 1]
    p.update(seed=cfg["random_seed"], deterministic=True, force_row_wise=True)
    return p


def gold_id(cfg: dict) -> str:
    h = hashlib.sha256(GOLD_BUILD_VERSION.encode())
    for f in GOLD_CODE:
        h.update((CODE_ROOT / f).read_bytes().replace(b"\r\n", b"\n"))
    keep = {k: cfg[k] for k in ("frames", "features", "availability", "targets", "scope")}
    h.update(json.dumps(keep, sort_keys=True, default=str).encode())
    rule = CODE_ROOT / cfg["availability"]["openmeteo_rule_file"]
    if rule.is_file():
        h.update(rule.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:12]


# ============================================================================ units
@dataclass(frozen=True)
class Unit:
    id: str
    kind: str                     # feat | gate | cv | refit | final
    stage: int
    deps: tuple = ()
    params: dict = field(default_factory=dict, hash=False, compare=False)


def cv_id(spec: str, lead: int, thr: float) -> str:
    return f"cv-{spec}-lead{lead}-thr{kstr(thr)}"


def units(cfg: dict) -> list[Unit]:
    leads = [int(x) for x in cfg["frames"]["lead_min"]]
    thrs, pt = thresholds(cfg), primary(cfg)
    others = [t for t in thrs if t != pt]
    u: list[Unit] = []
    feat = [Unit(f"feat-{y}-{m:02d}", "feat", 1, (), {"y": y, "m": m}) for y, m in build_months(cfg)]
    u += feat
    u.append(Unit("g2", "gate", 2, tuple(x.id for x in feat)))
    s3 = [Unit(cv_id("a", L, t), "cv", 3, ("g2",), {"spec": "a", "lead": L, "thr": t}) for t in thrs for L in leads]
    s3 += [Unit(cv_id(s, L, pt), "cv", 3, ("g2",), {"spec": s, "lead": L, "thr": pt})
           for s in ("b", "c", "d", "dm") for L in leads]
    u += s3
    u.append(Unit("sel-features", "gate", 3, tuple(x.id for x in s3)))
    s4 = [Unit(cv_id(f"g{n}", L, pt), "cv", 4, ("sel-features",), {"spec": f"g{n}", "lead": L, "thr": pt})
          for n in range(1, len(grid_settings(cfg)) + 1) for L in leads]
    u += s4
    u.append(Unit("sel-setting", "gate", 4, tuple(x.id for x in s4)))
    s5 = [Unit(cv_id(s, L, t), "cv", 5, ("sel-setting",), {"spec": s, "lead": L, "thr": t})
          for s in ("e", "f0") for t in others for L in leads]
    s5 += [Unit(cv_id(s, L, t), "cv", 5, ("sel-setting",), {"spec": s, "lead": L, "thr": t})
           for s in ("A24", "B24") for t in thrs for L in leads]
    u += s5
    u.append(Unit("oof-select", "gate", 5, tuple(x.id for x in s5)))
    s6 = [Unit(f"refit-{v}-lead{L}-thr{kstr(t)}", "refit", 6, ("oof-select",), {"variant": v, "lead": L, "thr": t})
          for v in ("A", "B") for t in thrs for L in leads]
    u += s6
    u.append(Unit("oof-report", "gate", 6, tuple(x.id for x in s6)))
    u.append(Unit("final-test", "final", 7, ("oof-report",)))
    return u


def part_units(us: list[Unit], i: int, n: int) -> list[Unit]:
    """Non-gate, non-final units of part i/n: round-robin inside every stage (each part gets a share of every
    stage, so parallel parts progress stage by stage together)."""
    out = []
    for st in sorted({x.stage for x in us}):
        work = [x for x in us if x.stage == st and x.kind not in ("gate", "final")]
        out += [x for k, x in enumerate(work) if k % n == i - 1]
    return out


# ============================================================================ run context (store access)
class Ctx:
    """Everything a unit needs: config, store, run identity, local workspace and the remote listing cache."""

    def __init__(self, cfg: dict, store, job: str, run_id: str, mirror: Path, log):
        self.cfg, self.store, self.job, self.run_id, self.mirror, self.log = cfg, store, job, run_id, mirror, log
        self.gid = gold_id(cfg)
        self._ls: dict[str, dict] = {}

    # ---- remote helpers
    def listing(self, prefix: str) -> dict:
        if prefix not in self._ls:
            self._ls[prefix] = self.store.list(prefix)
        return self._ls[prefix]

    def _download(self, key: str, size: int, dst: Path) -> bool:
        if dst.is_file() and dst.stat().st_size == size:
            return False
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".part")
        tmp.write_bytes(self.store.get(key))
        os.replace(tmp, dst)
        return True

    def fetch_lake(self, rels: list[str], list_prefix: str) -> int:
        """Download lake/v1/<rel> -> <ROOT>/<rel> for the rels that exist remotely (listing of list_prefix)."""
        ls = self.listing(LAKE + list_prefix)
        n = 0
        for rel in rels:
            k = LAKE + rel
            if k in ls:
                n += self._download(k, ls[k], ROOT / rel)
        return n

    def fetch_lake_prefix(self, rel_prefix: str, keep=lambda rel: True) -> int:
        n = 0
        for k, size in sorted(self.store.list(LAKE + rel_prefix).items()):
            rel = k[len(LAKE):]
            if keep(rel):
                n += self._download(k, size, ROOT / rel)
        return n

    def gold_key(self, table: str, y: int, m: int, name: str = "part.parquet") -> str:
        return f"{GOLD_REMOTE}{TABLES[table]}/gid={self.gid}/season={y}/month={m:02d}/{name}"

    def gold_local(self, table: str) -> Path:
        return ROOT / "data" / "gold" / "met_training" / "v1" / TABLES[table] / f"gid={self.gid}"

    def fetch_gold(self, table: str) -> int:
        pre = f"{GOLD_REMOTE}{TABLES[table]}/gid={self.gid}/"
        n = 0
        for k, size in sorted(self.store.list(pre).items()):
            n += self._download(k, size, self.gold_local(table) / k[len(pre):])
        return n

    def put_own(self, key: str, data: bytes) -> None:
        """Upload an output of the running unit (overwritable while the unit has no marker)."""
        self.store.overwritable.add(key)
        try:
            self.store.put(key, data, overwrite=True)
        finally:
            self.store.overwritable.discard(key)

    def run_files_prefix(self) -> str:
        return f"{LAKE}runs/{self.job}/{self.run_id}/files/"

    def fetch_run_files(self, rel_prefix: str) -> int:
        pre = self.run_files_prefix()
        n = 0
        for k, size in sorted(self.store.list(pre + rel_prefix).items()):
            n += self._download(k, size, ROOT / k[len(pre):])
        return n

    def read_json(self, rel: str) -> dict:
        p = ROOT / rel
        if not p.is_file():
            self.fetch_run_files(rel)
        return json.loads(p.read_text(encoding="utf-8"))


def write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=_jsonable), encoding="utf-8")
    os.replace(tmp, p)


def _jsonable(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_parquet(df: pd.DataFrame, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, p)


def rel(p: Path) -> str:
    return Path(p).resolve().relative_to(ROOT.resolve()).as_posix()


# ============================================================================ stage 1: Gold v1 (features)
def _silver_rel(source: str, d: date) -> str:
    return f"data/silver/met/{source}/date={d.isoformat()}/part.parquet"


def _city_pixels(ctx: Ctx) -> np.ndarray:
    """The 47 city pixels = pixel set of the pilot labels (same definition); scope.pixel_limit keeps the first n."""
    lab = sorted(ctx.listing(LAKE + PILOT_REL + "labels/"))
    keys = [k for k in lab if k.endswith("part.parquet")]
    ctx.fetch_lake([keys[0][len(LAKE):]], PILOT_REL + "labels/")
    pix = np.sort(pd.read_parquet(ROOT / keys[0][len(LAKE):], columns=["pixel_id"])["pixel_id"].unique())
    n = ctx.cfg["scope"].get("pixel_limit")
    return pix[:n] if n else pix


def _labels_for_month(ctx: Ctx, y: int, m: int, days: list[date], pix: np.ndarray) -> pd.DataFrame:
    from src.gold import build_labels as BL
    src = label_source(ctx.cfg, y, m)
    if src == "imerg_final":                                   # reuse the pilot label partitions (same definition)
        r = f"{PILOT_REL}labels/season={y}/month={m:02d}/part.parquet"
        ctx.fetch_lake([r], PILOT_REL + "labels/")
        lab = pd.read_parquet(ROOT / r)
    else:                                                      # provisional labels (IMERG Late), built here
        need = days + [days[-1] + timedelta(days=1)]
        ctx.fetch_lake([_silver_rel(src, d) for d in need], f"data/silver/met/{src}/")
        lab, _ = BL.build_month(y, m, ctx.cfg, pix, source=src, days=days)
    lab["t0"] = lab["t0"].astype("datetime64[ns, UTC]")
    dd = lab["t0"].dt.date
    lab = lab[dd.isin(set(days)) & lab["pixel_id"].isin(pix)].reset_index(drop=True)
    return lab


def _groups(cols: list[str]):
    from src.gold import features as F
    from src.quality.temporal_checks import FeatureGroup
    by: dict[str, list[str]] = {}
    for c in cols:
        by.setdefault(F.group_of(c), []).append(c)
    return [FeatureGroup(g, F.GROUP_KIND[g], tuple(cs)) for g, cs in by.items()]


def table_features(cols) -> list[str]:
    from src.gold import features as F
    allow = tuple(F.FEATURE_GROUPS_A["gsmap"]) + ("gsmap_n_missing", "cal_", "lat", "lon", "nwp_")
    return [c for c in cols if "__" not in c and c.startswith(allow) and not c.startswith("label")]


def g2_violations(df: pd.DataFrame, fcols: list[str]) -> list[str]:
    from src.quality.temporal_checks import FrameConfig, check_frame_table
    fc = FrameConfig.from_yaml(CFG_PATH)
    return [str(v) for v in check_frame_table(df, _groups(fcols), fcols, fc)]


def run_feat(unit: Unit, ctx: Ctx) -> tuple[list[str], dict]:
    from src.gold import build_table_B as TB
    from src.gold import build_training_table as BT
    from src.gold import features as F
    cfg, y, m = ctx.cfg, unit.params["y"], unit.params["m"]
    days = [d for d in data_days(cfg) if (d.year, d.month) == (y, m)]
    pix = _city_pixels(ctx)
    lab = _labels_for_month(ctx, y, m, days, pix)
    need = sorted({d - timedelta(days=1) for d in days} | set(days))
    ctx.fetch_lake([_silver_rel("gsmap_now", d) for d in need], "data/silver/met/gsmap_now/")
    grid = F.Grid()
    p = F.feature_params(cfg)
    assert p["version"] == "v1"
    wins = BT.load_windows(need, grid)
    t0s = pd.DatetimeIndex(np.sort(lab["t0"].unique())).tz_convert("UTC").as_unit("ns")
    feats = [F.compute_features(*F.stack_windows(wins, t0s[s:s + 240], p, grid.n_lat, grid.n_lon), t0s[s:s + 240],
                                pix, p, grid) for s in range(0, len(t0s), 240)]
    fe = pd.concat(feats, ignore_index=True)
    fcols = F.feature_columns(fe)
    A = lab.merge(fe, on=["pixel_id", "t0"], how="left", validate="many_to_one")
    for c in ("t0", "prediction_time", "frame_start", "target_time", "gsmap__event_time", "gsmap__available_time"):
        A[c] = A[c].astype("datetime64[us, UTC]")
    out, metrics = [], {"rows_A1": len(A), "days": len(days), "label_source": label_source(cfg, y, m),
                        "windows": len(wins)}
    v = g2_violations(A, fcols)
    if v:
        raise RuntimeError(f"G2 violations in table_A1_v1 {y}-{m:02d}: {v[:3]}")
    tables = [("A1", A, fcols)]
    if has_nwp(cfg, y, m):
        dd = sorted({d + timedelta(days=k) for d in days for k in (-1, 0, 1)})
        ctx.fetch_lake([_silver_rel("openmeteo", d) for d in dd], "data/silver/met/openmeteo/")
        om = TB.load_om([d for d in dd if (ROOT / _silver_rel("openmeteo", d)).is_file()])
        pmap = F.nwp_point_map(om, pix)
        rows = A[["pixel_id", "prediction_time", "frame_start"]].copy()
        for c in ("prediction_time", "frame_start"):
            rows[c] = rows[c].astype("datetime64[ns, UTC]")
        nw = F.compute_nwp_features(rows.reset_index(drop=True), om, pmap)
        for c in [c for c in nw.columns if c.endswith("_time")]:
            nw[c] = nw[c].astype("datetime64[us, UTC]")
        B = pd.concat([A.reset_index(drop=True), nw], axis=1)
        bcols = fcols + [c for c in nw.columns if "__" not in c]
        v = g2_violations(B, bcols)
        if v:
            raise RuntimeError(f"G2 violations in table_B1_v1 {y}-{m:02d}: {v[:3]}")
        tables.append(("B1", B, bcols))
        metrics["rows_B1"] = len(B)
        metrics["nwp_nan_share_max"] = float(B[[c for c in nw.columns if "__" not in c]].isna().mean().max())
    for t, df, cols in tables:
        lp = ctx.gold_local(t) / f"season={y}" / f"month={m:02d}" / "part.parquet"
        write_parquet(df, lp)
        meta = {"table": TABLES[t], "gold_id": ctx.gid, "season": y, "month": m, "rows": len(df), "features": cols,
                "label_source": label_source(cfg, y, m), "days": [d.isoformat() for d in days],
                "nan_share": {c: round(float(df[c].isna().mean()), 5) for c in cols}, "g2_violations": 0}
        write_json(lp.with_name("part.meta.json"), meta)
        for name in ("part.parquet", "part.meta.json"):
            k = ctx.gold_key(t, y, m, name)
            ctx.put_own(k, (lp.parent / name).read_bytes())
            out.append(k)
    metrics["g2_violations"] = 0
    return [], metrics | {"gold_keys": out}


def gate_g2(ctx: Ctx) -> tuple[list[str], dict]:
    """G2 on every partition (re-read from R2), duplicates, label columns, v1 == pilot on common columns."""
    rows, nan = [], []
    for t in ("A1", "B1"):
        ctx.fetch_gold(t)
        for part in sorted(ctx.gold_local(t).glob("season=*/month=*/part.parquet")):
            df = pd.read_parquet(part)
            fcols = table_features(df.columns)
            v = g2_violations(df, fcols)
            dup = int(df.duplicated(["pixel_id", "t0", "lead_min"]).sum())
            rows.append({"table": TABLES[t], "partition": f"{part.parent.parent.name}/{part.parent.name}",
                         "rows": len(df), "features": len(fcols), "violations": len(v), "duplicates": dup,
                         "labels_present": all(c in df.columns for c in ["label_rate_mm_h"] + [thr_col(x) for x in thresholds(ctx.cfg)]),
                         "label_source": str(df["label_source"].iloc[0]) if len(df) else "", "detail": "; ".join(v)})
            nan.append({"table": TABLES[t], "partition": rows[-1]["partition"]} |
                       {c: float(df[c].isna().mean()) for c in fcols})
    # single code path: every pilot column except cal_doy_* is identical in v1 (first CV fold month)
    fs = cv_folds(ctx.cfg)[0][0]
    rp = f"{PILOT_REL}table_A/season={fs.year}/month={fs.month:02d}/part.parquet"
    ctx.fetch_lake([rp], PILOT_REL + "table_A/")
    eq = {"pilot_partition": rp}
    a1 = ctx.gold_local("A1") / f"season={fs.year}" / f"month={fs.month:02d}" / "part.parquet"
    if (ROOT / rp).is_file() and a1.is_file():
        pa, va = pd.read_parquet(ROOT / rp), pd.read_parquet(a1)
        common = [c for c in table_features(pa.columns) if c in va.columns]
        j = va[["pixel_id", "t0", "lead_min"] + common].merge(pa[["pixel_id", "t0", "lead_min"] + common],
                                                              on=["pixel_id", "t0", "lead_min"], suffixes=("_v1", "_v0"))
        diffs = {c: float(np.nanmax(np.abs(j[c + "_v1"].to_numpy("float64") - j[c + "_v0"].to_numpy("float64")),
                                    initial=0.0)) for c in common}
        nan_mismatch = {c: int((j[c + "_v1"].isna() != j[c + "_v0"].isna()).sum()) for c in common}
        eq |= {"rows_compared": len(j), "common_columns": len(common), "max_abs_diff": max(diffs.values(), default=0.0),
               "nan_mismatch": sum(nan_mismatch.values()),
               "pilot_only_columns": [c for c in table_features(pa.columns) if c not in va.columns]}
    else:
        eq |= {"rows_compared": 0, "note": "pilot partition or v1 partition missing"}
    out = ROOT / "reports" / "p4b" / "g2"
    out.mkdir(parents=True, exist_ok=True)
    r = pd.DataFrame(rows)
    r.to_csv(out / "partitions.csv", index=False)
    pd.DataFrame(nan).to_csv(out / "nan_share.csv", index=False)
    write_json(out / "pilot_equality.json", eq)
    ok = len(r) > 0 and (r["violations"] == 0).all() and (r["duplicates"] == 0).all() and r["labels_present"].all()
    ok_eq = eq.get("rows_compared", 0) > 0 and eq["max_abs_diff"] <= 1e-6 and eq["nan_mismatch"] == 0
    m = {"partitions": len(r), "rows": int(r["rows"].sum()) if len(r) else 0,
         "violations": int(r["violations"].sum()) if len(r) else -1,
         "duplicates": int(r["duplicates"].sum()) if len(r) else -1, "pilot_equal": bool(ok_eq),
         "pilot_max_abs_diff": eq.get("max_abs_diff")}
    if not (ok and ok_eq):
        raise RuntimeError(f"G2 gate FAIL: {m}")
    return [rel(p) for p in sorted(out.glob("*")) if p.is_file()], m


# ============================================================================ data access for training
def table_dir(ctx: Ctx, table: str) -> Path:
    return ROOT / PILOT_REL / "table_A" if table == "pilot_A" else ctx.gold_local(table)


def ensure_table(ctx: Ctx, table: str) -> None:
    if table == "pilot_A":
        months = {(a.year, a.month) for a, _ in cv_folds(ctx.cfg)}
        ctx.fetch_lake_prefix(PILOT_REL + "table_A/", keep=lambda r: r.endswith("schema.json") or any(
            f"season={y}/month={m:02d}/" in r for y, m in months))
    else:
        ctx.fetch_gold(table)


def load_rows(ctx: Ctx, table: str, periods: list, cols: list[str], lead: int | None = None) -> pd.DataFrame:
    d = table_dir(ctx, table)
    parts = []
    for part in sorted(d.glob("season=*/month=*/part.parquet")):
        y, m = int(part.parent.parent.name.split("=")[1]), int(part.parent.name.split("=")[1])
        ms = pd.Timestamp(year=y, month=m, day=1, tz="UTC")
        me = ms + pd.offsets.MonthBegin(1)
        if not any(ms < b and me > a for a, b in periods):
            continue
        have = set(_schema_names(part))
        need = [c for c in dict.fromkeys(KEYS + ["target_time"] + LABELS + list(cols)) if c in have]
        df = pd.read_parquet(part, columns=need)
        if lead is not None:
            df = df[df["lead_min"] == lead]
        t0 = df["t0"]
        sel = np.zeros(len(df), bool)
        for a, b in periods:
            sel |= ((t0 >= a) & (t0 < b)).to_numpy()
        parts.append(df[sel])
    if not parts:
        raise RuntimeError(f"no rows in {table} for {periods[:2]}…")
    out = pd.concat(parts, ignore_index=True)
    n = ctx.cfg["scope"].get("pixel_limit")
    if n:
        out = out[out["pixel_id"].isin(np.sort(out["pixel_id"].unique())[:n])].reset_index(drop=True)
    out["day"] = out["t0"].dt.tz_convert(LOCAL_TZ).dt.date.astype(str)
    return out


def _schema_names(part: Path) -> list[str]:
    import pyarrow.parquet as pq
    return pq.read_schema(part).names


def feature_set(ctx: Ctx, name: str) -> tuple[str, list[str]]:
    """Feature list of a spec family: pilot | b | c | d | dm | B (= F* + nwp). Returns (table, columns)."""
    if name == "pilot":
        s = json.loads((ROOT / PILOT_REL / "table_A" / "schema.json").read_text(encoding="utf-8"))
        return "pilot_A", [f["name"] for f in s["features"]]
    t = "B1" if name.startswith("B") else "A1"
    part = next(iter(sorted(ctx.gold_local(t).glob("season=*/month=*/part.parquet"))), None)
    if part is None:
        raise RuntimeError(f"no {TABLES[t]} partition for gold id {ctx.gid}")
    allc = table_features(_schema_names(part))
    base = [c for c in allc if not c.startswith(V1_NEW) and not c.startswith("nwp_")]
    sets = {"b": base}
    sets["c"] = base + [c for c in allc if c.startswith(MOTION_PREFIXES)]
    sets["d"] = sets["c"] + [c for c in allc if c.startswith("extent_")]
    sets["dm"] = sets["d"] + [c for c in allc if c == "cal_month"]
    if t == "B1":
        fstar = name.split(":")[1]
        return t, sets[fstar] + [c for c in allc if c.startswith("nwp_")]
    return t, sets[name]


def spec_def(ctx: Ctx, spec: str) -> dict:
    """spec -> table, feature family, parameter setting, restricted folds."""
    if spec == "a":
        return {"fam": "pilot", "setting": "p4a", "restricted": False}
    if spec in ("b", "c", "d", "dm"):
        return {"fam": spec, "setting": "p4a", "restricted": False}
    fstar = ctx.read_json("reports/p4b/select/features.json")["feature_set"]
    if spec.startswith("g"):
        return {"fam": fstar, "setting": spec, "restricted": False}
    chosen = ctx.read_json("reports/p4b/select/setting.json")["chosen"]
    return {"e": {"fam": fstar, "setting": chosen, "restricted": False},
            "f0": {"fam": fstar, "setting": "p4a", "restricted": False},
            "A24": {"fam": fstar, "setting": chosen, "restricted": True},
            "B24": {"fam": f"B:{fstar}", "setting": chosen, "restricted": True}}[spec]


# ============================================================================ metrics helpers
def logloss(y, p) -> float:
    y = np.asarray(y, "float64"); p = np.clip(np.asarray(p, "float64"), 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))) if len(y) else float("nan")


def persistence(df: pd.DataFrame, lead: int) -> np.ndarray:
    from src.models import data as D
    return D.asof_latest(df) if lead in (0, 30) else D.max_avail_0_60(df)


def clim_for(train: pd.DataFrame, held: pd.DataFrame, yc: str) -> np.ndarray:
    from src.models import baselines as BLN
    pc = BLN.clim_lookup(BLN.clim_table(train, yc), held)
    return np.where(np.isnan(pc), float(train[yc].mean()), pc)


def oof_metrics(y, p, pclim, pers, thr) -> dict:
    from src.models import metrics as M
    y = np.asarray(y).astype(bool)
    c = M.choose_cut(y, p, "max_csi") if y.any() else {"cut": np.nan, "csi": np.nan, "bias": np.nan}
    yp = np.nan_to_num(np.asarray(pers, "float64"), nan=-1.0) >= thr
    return {"n": int(len(y)), "base_rate": float(y.mean()) if len(y) else np.nan, "logloss": logloss(y, p),
            "brier": M.brier(y, p), "bss_clim": M.bss(y, p, pclim), "auc_pr": M.auc_pr(y, p),
            "csi_maxcut_insample": c["csi"], "bias_maxcut": c["bias"],
            "csi_pers": float(M.scores(**M.contingency(y, yp))["csi"])}


# ============================================================================ stage 3-5: blocked CV units
def run_cv(unit: Unit, ctx: Ctx) -> tuple[list[str], dict]:
    import lightgbm as lgb
    cfg, spec, lead, thr = ctx.cfg, unit.params["spec"], unit.params["lead"], unit.params["thr"]
    sd = spec_def(ctx, spec)
    ensure_table(ctx, "B1" if sd["fam"].startswith("B") else ("pilot_A" if sd["fam"] == "pilot" else "A1"))
    table, feats = feature_set(ctx, sd["fam"])
    folds = cv_folds(cfg, restricted=sd["restricted"])
    gk = [f"gc_k{k}" for k in range(7)]
    df = load_rows(ctx, table, folds, list(dict.fromkeys(feats + gk)), lead=lead)
    yc = thr_col(thr)
    params = lgb_params(cfg, sd["setting"])
    buf = pd.Timedelta(days=int(cfg["cv"]["buffer_days"]))
    t0 = df["t0"]
    fold_of = np.full(len(df), -1, "int16")
    for fi, (a, b) in enumerate(folds):
        fold_of[((t0 >= a) & (t0 < b)).to_numpy()] = fi
    X = df[feats].to_numpy("float32")
    y = df[yc].to_numpy().astype("int8")
    boosters, best, fold_rows = [], [], []
    t1 = time.time()
    for fi, (a, b) in enumerate(folds):
        held = fold_of == fi
        train = (fold_of >= 0) & ~held & ((t0 < a - buf) | (t0 >= b + buf)).to_numpy()
        if held.sum() == 0 or train.sum() == 0 or y[train].sum() == 0:
            boosters.append(None); best.append(0)
            continue
        dtr = lgb.Dataset(X[train], y[train], feature_name=feats, free_raw_data=True)
        dva = lgb.Dataset(X[held], y[held], reference=dtr)
        bst = lgb.train(params, dtr, num_boost_round=int(cfg["cv"]["max_rounds"]), valid_sets=[dva],
                        valid_names=["held"], callbacks=[lgb.early_stopping(int(cfg["cv"]["early_stopping_rounds"]),
                                                                            verbose=False)])
        boosters.append(bst)
        best.append(int(bst.best_iteration or bst.current_iteration()))
        fold_rows.append({"fold": fi, "start": a.isoformat(), "end": b.isoformat(), "train_rows": int(train.sum()),
                          "held_rows": int(held.sum()), "best_iteration": best[-1]})
    used = [b for b, bs in zip(best, boosters) if bs is not None]
    med = int(np.median(used)) if used else 0
    p_raw = np.full(len(df), np.nan); p_best = np.full(len(df), np.nan); pclim = np.full(len(df), np.nan)
    capped = 0
    for fi, (a, b) in enumerate(folds):
        held = fold_of == fi
        if boosters[fi] is None:
            continue
        bst = boosters[fi]
        it = min(med, bst.current_iteration())
        capped += it < med
        p_raw[held] = bst.predict(X[held], num_iteration=max(it, 1))
        p_best[held] = bst.predict(X[held], num_iteration=max(best[fi], 1))
        train = (fold_of >= 0) & ~held & ((t0 < a - buf) | (t0 >= b + buf)).to_numpy()
        pclim[held] = clim_for(df[train], df[held], yc)
    pers = persistence(df, lead)
    ok = np.isfinite(p_raw)
    oof = pd.DataFrame({"pixel_id": df["pixel_id"].to_numpy(), "t0": df["t0"].to_numpy(), "lead_min": np.int16(lead),
                        "day": df["day"].to_numpy(), "fold": fold_of, "y": y, "label_rate_mm_h":
                        df["label_rate_mm_h"].to_numpy("float32"), "p_raw": p_raw.astype("float32"),
                        "p_best": p_best.astype("float32"), "p_clim": pclim.astype("float32"),
                        "pers": pers.astype("float32")})[ok]
    for fr in fold_rows:
        sel = oof["fold"].to_numpy() == fr["fold"]
        fr |= oof_metrics(oof["y"].to_numpy()[sel], oof["p_raw"].to_numpy()[sel], oof["p_clim"].to_numpy()[sel],
                          oof["pers"].to_numpy()[sel], thr)
    pooled = oof_metrics(oof["y"].to_numpy(), oof["p_raw"].to_numpy(), oof["p_clim"].to_numpy(),
                         oof["pers"].to_numpy(), thr)
    base = ROOT / "reports" / "p4b"
    po = base / "oof" / f"{unit.id}.parquet"
    write_parquet(oof, po)
    info = {"unit": unit.id, "spec": spec, "spec_def": sd, "table": table, "lead_min": lead, "threshold": thr,
            "label": "IMERG Final V07 (satellite estimate)", "features": feats, "n_features": len(feats),
            "lgb_params": params, "folds": fold_rows, "fold_best_iterations": best, "median_rounds": med,
            "folds_capped_below_median": int(capped), "pooled": pooled,
            "pooled_logloss_at_fold_best": logloss(oof["y"], oof["p_best"]),
            "rows": int(len(df)), "seconds": round(time.time() - t1, 1)}
    pj = base / "cv" / f"{unit.id}.json"
    write_json(pj, info)
    return [rel(po), rel(pj)], {"pooled_logloss": pooled["logloss"], "median_rounds": med,
                                "bss_clim": pooled["bss_clim"], "seconds": info["seconds"]}


def _cv_info(ctx: Ctx, uid: str) -> dict:
    return ctx.read_json(f"reports/p4b/cv/{uid}.json")


def _oof(ctx: Ctx, uid: str) -> pd.DataFrame:
    r = f"reports/p4b/oof/{uid}.parquet"
    if not (ROOT / r).is_file():
        ctx.fetch_run_files(r)
    return pd.read_parquet(ROOT / r)


def gate_sel_features(ctx: Ctx) -> tuple[list[str], dict]:
    pt, leads = primary(ctx.cfg), ctx.cfg["frames"]["lead_min"]
    tab = {s: [_cv_info(ctx, cv_id(s, L, pt))["pooled"]["logloss"] for L in leads] for s in ("d", "dm")}
    use_month = float(np.mean(tab["dm"])) < float(np.mean(tab["d"]))
    out = {"feature_set": "dm" if use_month else "d", "use_cal_month": use_month, "pooled_logloss_by_lead": tab,
           "rule": ctx.cfg["selection"]["month_feature"], "leads": leads, "threshold": pt}
    p = ROOT / "reports/p4b/select/features.json"
    write_json(p, out)
    return [rel(p)], {"feature_set": out["feature_set"], "mean_logloss_d": float(np.mean(tab["d"])),
                      "mean_logloss_dm": float(np.mean(tab["dm"]))}


def gate_sel_setting(ctx: Ctx) -> tuple[list[str], dict]:
    pt, leads = primary(ctx.cfg), ctx.cfg["frames"]["lead_min"]
    rows = []
    for n, s in enumerate(grid_settings(ctx.cfg), start=1):
        ll = [_cv_info(ctx, cv_id(f"g{n}", L, pt))["pooled"]["logloss"] for L in leads]
        rows.append({"setting": f"g{n}", **s, **{f"logloss_F{L}": v for L, v in zip(leads, ll)},
                     "mean_logloss": float(np.mean(ll))})
    best = min(rows, key=lambda r: r["mean_logloss"])
    p = ROOT / "reports/p4b/select/setting.json"
    write_json(p, {"chosen": best["setting"], "params": lgb_params(ctx.cfg, best["setting"]), "table": rows,
                   "rule": "lowest mean pooled OOF log-loss over F0/F30/F60 at the primary threshold"})
    pd.DataFrame(rows).to_csv(p.with_name("grid.csv"), index=False)
    return [rel(p), rel(p.with_name("grid.csv"))], {"chosen": best["setting"], "mean_logloss": best["mean_logloss"]}


# ============================================================================ calibration and cuts (OOF)
def fit_calibrator(method: str, p: np.ndarray, y: np.ndarray) -> dict:
    p = np.asarray(p, "float64"); y = np.asarray(y, "float64")
    if method == "raw":
        return {"method": "raw"}
    if method == "platt":
        from sklearn.linear_model import LogisticRegression
        z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
        return {"method": "platt", "a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])}
    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
        return {"method": "isotonic", "x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}
    raise ValueError(method)


def apply_calibrator(cal: dict, p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, "float64")
    if cal["method"] == "raw":
        return p
    if cal["method"] == "platt":
        z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        return 1 / (1 + np.exp(-(cal["a"] * z + cal["b"])))
    return np.interp(p, np.asarray(cal["x"]), np.asarray(cal["y"]))


def crossfit(method: str, oof: pd.DataFrame) -> np.ndarray:
    """Leave-one-fold-out calibration: fold f is calibrated by a calibrator fitted on the other folds' OOF."""
    out = np.full(len(oof), np.nan)
    f = oof["fold"].to_numpy()
    for k in np.unique(f):
        tr, te = f != k, f == k
        if tr.sum() == 0 or oof["y"].to_numpy()[tr].sum() == 0:
            out[te] = oof["p_raw"].to_numpy()[te]
            continue
        cal = fit_calibrator(method, oof["p_raw"].to_numpy()[tr], oof["y"].to_numpy()[tr])
        out[te] = apply_calibrator(cal, oof["p_raw"].to_numpy()[te])
    return out


def _ds(df: pd.DataFrame, p, yhat=None):
    from src.models import metrics as M
    y = df["y"].to_numpy().astype(bool)
    yh = np.zeros(len(y), bool) if yhat is None else yhat
    return M.day_stats(df["day"].to_numpy(), y, yh, p, df["p_clim"].to_numpy("float64"))


def choose_calibration(ctx: Ctx, oof: pd.DataFrame) -> tuple[str, dict, np.ndarray]:
    from src.models import metrics as M
    ev = ctx.cfg["evaluation"]
    praw = oof["p_raw"].to_numpy("float64")
    res = {"raw": {"brier": M.brier(oof["y"], praw)}}
    probs = {"raw": praw}
    ds_raw = _ds(oof, praw)
    for m in ctx.cfg["selection"]["calibration"]:
        if m == "raw":
            continue
        pc = crossfit(m, oof)
        probs[m] = pc
        d = M.paired_bootstrap(_ds(oof, pc), ds_raw, ev["bootstrap_resamples"], ctx.cfg["random_seed"], ev["ci_level"])
        res[m] = {"brier": M.brier(oof["y"], pc), "brier_diff_vs_raw": d["brier"]["diff"],
                  "lo": d["brier"]["lo"], "hi": d["brier"]["hi"]}
    sig = [m for m in res if m != "raw" and res[m]["hi"] < 0]
    choice = min(sig, key=lambda m: res[m]["brier"]) if sig else "raw"
    return choice, res, probs[choice]


def choose_cut(ctx: Ctx, oof: pd.DataFrame, p: np.ndarray, thr: float) -> dict:
    from src.models import metrics as M
    ev, sel = ctx.cfg["evaluation"], ctx.cfg["selection"]
    y = oof["y"].to_numpy().astype(bool)
    if thr == float(ctx.cfg["targets"]["secondary_mm_h"]):
        c = M.choose_cut(y, p, f"freq_bias_{sel['heavy_bias']:g}")
        return {"rule": c["rule"], "cut": c["cut"], "csi": c["csi"], "bias": c["bias"]}
    mx = M.choose_cut(y, p, "max_csi")
    out = {"max_csi": mx, "rule": "max_csi", "cut": mx["cut"], "csi": mx["csi"], "bias": mx["bias"]}
    lo, hi = sel["cut_bias_band"]
    if not (lo <= mx["bias"] <= hi):
        b1 = M.choose_cut(y, p, "freq_bias_1.0")
        d = M.paired_bootstrap(_ds(oof, p, p >= mx["cut"]), _ds(oof, p, p >= b1["cut"]), ev["bootstrap_resamples"],
                               ctx.cfg["random_seed"], ev["ci_level"])
        out["bias_1"] = b1
        out["csi_diff_maxcsi_minus_bias1"] = d["csi"]
        if not d["csi"]["lo"] > 0:                       # max-CSI cut not significantly better -> honest bias 1.0
            out |= {"rule": "freq_bias_1.0", "cut": b1["cut"], "csi": b1["csi"], "bias": b1["bias"]}
    return out


def chosen_spec(ctx: Ctx, variant: str, thr: float, sel_cache: dict | None = None) -> str:
    if variant == "B":
        return "B24"
    if thr == primary(ctx.cfg):
        return ctx.read_json("reports/p4b/select/setting.json")["chosen"]
    return (sel_cache or ctx.read_json("reports/p4b/select/selection.json"))["spec_by_threshold"][kstr(thr)]


def gate_oof_select(ctx: Ctx) -> tuple[list[str], dict]:
    from src.models import metrics as M
    cfg = ctx.cfg
    leads, thrs, pt = cfg["frames"]["lead_min"], thresholds(cfg), primary(cfg)
    ev = cfg["evaluation"]
    chosen = ctx.read_json("reports/p4b/select/setting.json")["chosen"]
    spec_thr, gain = {kstr(pt): chosen}, []
    for t in [t for t in thrs if t != pt]:
        le = np.mean([_cv_info(ctx, cv_id("e", L, t))["pooled"]["logloss"] for L in leads])
        lf = np.mean([_cv_info(ctx, cv_id("f0", L, t))["pooled"]["logloss"] for L in leads])
        use_f0 = lf < le * (1 - cfg["selection"]["other_thresholds_clear_gain_rel"])
        spec_thr[kstr(t)] = "f0" if use_f0 else "e"
        gain.append({"threshold": t, "mean_logloss_chosen_setting": float(le), "mean_logloss_p4a": float(lf),
                     "use": spec_thr[kstr(t)]})
    sel = {"spec_by_threshold": spec_thr, "setting_check": gain, "models": {}}
    cal_rows, cut_rows = [], []
    for variant in ("A", "B"):
        for t in thrs:
            spec = "B24" if variant == "B" else spec_thr[kstr(t)]
            for L in leads:
                uid = cv_id(spec, L, t)
                oof = _oof(ctx, uid)
                info = _cv_info(ctx, uid)
                ch, res, p = choose_calibration(ctx, oof)
                cut = choose_cut(ctx, oof, p, t)
                key = f"{variant}-lead{L}-thr{kstr(t)}"
                sel["models"][key] = {"cv_unit": uid, "setting": info["spec_def"]["setting"],
                                      "median_rounds": info["median_rounds"], "calibration": ch, "cut": cut,
                                      "features": info["features"]}
                cal_rows.append({"variant": variant, "lead_min": L, "threshold": t, "choice": ch,
                                 **{f"brier_{m}": r["brier"] for m, r in res.items()},
                                 **{f"brier_diff_{m}_hi": r.get("hi") for m, r in res.items() if m != "raw"}})
                cut_rows.append({"variant": variant, "lead_min": L, "threshold": t, "rule": cut["rule"],
                                 "cut": cut["cut"], "csi": cut["csi"], "bias": cut["bias"],
                                 "maxcsi_bias": cut.get("max_csi", {}).get("bias")})
                sel["models"][key]["_p"] = p                       # in memory only (nested rule below)
                sel["models"][key]["_oof"] = oof
    # 7.6 mm/h: bias-1.5 cut alone vs nested (also above the 2.5 mm/h cut); keep the better OOF CSI
    hv = float(cfg["targets"]["secondary_mm_h"])
    nested = []
    if hv in thrs and pt in thrs:
        for variant in ("A", "B"):
            for L in leads:
                h, q = sel["models"][f"{variant}-lead{L}-thr{kstr(hv)}"], sel["models"][f"{variant}-lead{L}-thr{kstr(pt)}"]
                j = h["_oof"][["pixel_id", "t0", "y"]].assign(ph=h["_p"]).merge(
                    q["_oof"][["pixel_id", "t0"]].assign(pq=q["_p"]), on=["pixel_id", "t0"], how="left")
                y = j["y"].to_numpy().astype(bool)
                alone = j["ph"].to_numpy() >= h["cut"]["cut"]
                nest = alone & (np.nan_to_num(j["pq"].to_numpy(), nan=-1) >= q["cut"]["cut"])
                ca = float(M.scores(**M.contingency(y, alone))["csi"])
                cn = float(M.scores(**M.contingency(y, nest))["csi"])
                h["heavy_rule"] = "nested" if cn > ca else "bias_1.5"
                nested.append({"variant": variant, "lead_min": L, "csi_bias15": ca, "csi_nested": cn,
                               "bias_nested": float(M.scores(**M.contingency(y, nest))["bias"]),
                               "rule": h["heavy_rule"]})
    # B ships if not worse than A-restricted (same rows) at any frame for the primary threshold
    bvsa = []
    for L in leads:
        a, b = sel["models"][f"A-lead{L}-thr{kstr(pt)}"], sel["models"][f"B-lead{L}-thr{kstr(pt)}"]
        oa = _oof(ctx, cv_id("A24", L, pt))
        pa = oa["p_raw"].to_numpy("float64") if a["calibration"] == "raw" else crossfit(a["calibration"], oa)
        pb = b["_p"]
        ob = b["_oof"]
        j = oa[["pixel_id", "t0"]].assign(pa=pa).merge(ob[["pixel_id", "t0"]].assign(pb=pb), on=["pixel_id", "t0"])
        oaj = oa.merge(j[["pixel_id", "t0"]], on=["pixel_id", "t0"]).reset_index(drop=True)
        ya = oaj["y"].to_numpy().astype(bool)
        cut_a = M.choose_cut(ya, j["pa"].to_numpy(), "max_csi")["cut"]
        cut_b = b["cut"]["cut"]
        d = M.paired_bootstrap(_ds(oaj, j["pb"].to_numpy(), j["pb"].to_numpy() >= cut_b),
                               _ds(oaj, j["pa"].to_numpy(), j["pa"].to_numpy() >= cut_a),
                               ev["bootstrap_resamples"], cfg["random_seed"], ev["ci_level"])
        worse = d["brier"]["lo"] > 0 or d["csi"]["hi"] < 0
        bvsa.append({"lead_min": L, "rows": len(j), "brier_B_minus_A": d["brier"], "csi_B_minus_A": d["csi"],
                     "B_worse": bool(worse)})
    sel["ship_B"] = not any(r["B_worse"] for r in bvsa)
    sel["B_vs_A_restricted"] = bvsa
    sel["heavy_rule"] = nested
    sel["serving_rule"] = ("use the set-B model when every nwp_* input is present and built from the "
                           "previous_day1 run (issue <= prediction_time - 22 h, available_time <= prediction_time); "
                           "otherwise (NWP missing or stale) use the set-A model") if sel["ship_B"] else \
        "set A only (set B not shipped: worse than A-restricted on OOF at some frame)"
    for m in sel["models"].values():
        m.pop("_p", None); m.pop("_oof", None)
    base = ROOT / "reports/p4b/select"
    write_json(base / "selection.json", sel)
    pd.DataFrame(cal_rows).to_csv(base / "calibration.csv", index=False)
    pd.DataFrame(cut_rows).to_csv(base / "cuts.csv", index=False)
    pd.DataFrame(nested).to_csv(base / "heavy_rule.csv", index=False)
    rels = [rel(base / f) for f in ("selection.json", "calibration.csv", "cuts.csv", "heavy_rule.csv")]
    return rels, {"ship_B": sel["ship_B"], "spec_by_threshold": spec_thr,
                  "calibration_choices": pd.DataFrame(cal_rows)["choice"].value_counts().to_dict()}


# ============================================================================ stage 6: refit
def model_dir(variant: str, lead: int, thr: float) -> Path:
    return ROOT / "models" / "v1" / variant / f"lead{lead}_thr{kstr(thr)}"


def run_refit(unit: Unit, ctx: Ctx) -> tuple[list[str], dict]:
    import lightgbm as lgb
    cfg = ctx.cfg
    v, lead, thr = unit.params["variant"], unit.params["lead"], unit.params["thr"]
    sel = ctx.read_json("reports/p4b/select/selection.json")
    m = sel["models"][f"{v}-lead{lead}-thr{kstr(thr)}"]
    info = _cv_info(ctx, m["cv_unit"])
    sd = info["spec_def"]
    table = info["table"]
    ensure_table(ctx, table)
    folds = cv_folds(cfg, restricted=sd["restricted"])
    feats = info["features"]
    df = load_rows(ctx, table, folds, feats, lead=lead)
    yc = thr_col(thr)
    t1 = time.time()
    rounds = max(int(m["median_rounds"]), 1)
    bst = lgb.train(info["lgb_params"], lgb.Dataset(df[feats].to_numpy("float32"), df[yc].to_numpy(),
                                                    feature_name=feats), num_boost_round=rounds)
    oof = _oof(ctx, m["cv_unit"])
    cal = fit_calibrator(m["calibration"], oof["p_raw"].to_numpy(), oof["y"].to_numpy())
    d = model_dir(v, lead, thr)
    d.mkdir(parents=True, exist_ok=True)
    bst.save_model(str(d / "model.txt"))
    write_json(d / "calibration.json", cal)
    pd.DataFrame({"feature": feats, "gain": bst.feature_importance("gain"),
                  "split": bst.feature_importance("split")}).sort_values("gain", ascending=False).to_csv(
        d / "importance.csv", index=False)
    meta = {"model": f"v1/{v}/lead{lead}_thr{kstr(thr)}", "status": "v1 candidate — rainfall nowcast, not flood",
            "variant": v, "table": TABLES.get(table, table), "gold_id": ctx.gid, "lead_min": lead,
            "threshold_mm_h": thr, "label": "IMERG Final V07 (satellite estimate)", "features": feats,
            "n_features": len(feats), "lgb_params": info["lgb_params"], "num_boost_round": rounds,
            "rounds_rule": "median of the CV fold optima", "cv_unit": m["cv_unit"],
            "training_period": [[a.isoformat(), b.isoformat()] for a, b in folds], "rows": len(df),
            "calibration": m["calibration"], "cut": m["cut"], "heavy_rule": m.get("heavy_rule"),
            "serving_rule": sel["serving_rule"] if v == "B" else "set A (fallback when NWP inputs are missing or stale)",
            "ship_B": sel["ship_B"], "run_id": ctx.run_id, "seconds": round(time.time() - t1, 1)}
    write_json(d / "meta.json", meta)
    return [rel(d / f) for f in ("model.txt", "calibration.json", "importance.csv", "meta.json")], \
        {"rounds": rounds, "rows": len(df), "seconds": meta["seconds"]}


# ============================================================================ OOF report (stop point)
def gate_oof_report(ctx: Ctx) -> tuple[list[str], dict]:
    cfg = ctx.cfg
    leads, thrs, pt = cfg["frames"]["lead_min"], thresholds(cfg), primary(cfg)
    sel = ctx.read_json("reports/p4b/select/selection.json")
    chosen = ctx.read_json("reports/p4b/select/setting.json")["chosen"]
    feats = ctx.read_json("reports/p4b/select/features.json")
    out = ROOT / "reports/p4b/oof"
    out.mkdir(parents=True, exist_ok=True)
    abl = []
    steps = [("a", "a", "new CV scheme, P4A features + parameters"), ("b", "b", "without cal_doy"),
             ("c", "c", "+ motion / advection / upwind"), ("d", "d", "+ rain extent"),
             ("d+month", "dm", "+ cal_month (kept only if better)"), ("e", chosen, "regularised setting (grid)")]
    for L in leads:
        prev = None
        for name, spec, what in steps:
            pm = _cv_info(ctx, cv_id(spec, L, pt))["pooled"]
            row = {"lead_min": L, "step": name, "spec": spec, "what": what, **pm}
            if prev is not None and name != "d+month":
                row["d_logloss"] = pm["logloss"] - prev["logloss"]; row["d_brier"] = pm["brier"] - prev["brier"]
            abl.append(row)
            if name != "d+month" or feats["feature_set"] == "dm":
                prev = pm
        m = sel["models"][f"A-lead{L}-thr{kstr(pt)}"]
        o = _oof(ctx, m["cv_unit"])
        pf = crossfit(m["calibration"], o) if m["calibration"] != "raw" else o["p_raw"].to_numpy("float64")
        fm = oof_metrics(o["y"], pf, o["p_clim"], o["pers"], pt)
        abl.append({"lead_min": L, "step": "f", "spec": chosen, "what": f"calibration = {m['calibration']} "
                    "(cross-fitted)", **fm, "d_logloss": fm["logloss"] - prev["logloss"],
                    "d_brier": fm["brier"] - prev["brier"]})
        a24, b24 = _oof(ctx, cv_id("A24", L, pt)), _oof(ctx, cv_id("B24", L, pt))
        j = a24.merge(b24[["pixel_id", "t0", "p_raw"]], on=["pixel_id", "t0"], suffixes=("_A", "_B"))
        ma = oof_metrics(j["y"], j["p_raw_A"], j["p_clim"], j["pers"], pt)
        mb = oof_metrics(j["y"], j["p_raw_B"], j["p_clim"], j["pers"], pt)
        abl.append({"lead_min": L, "step": "g(A, NWP months)", "spec": "A24", "what": "set A on the NWP months", **ma})
        abl.append({"lead_min": L, "step": "g", "spec": "B24", "what": "set B (+ NWP) on the same rows", **mb,
                    "d_logloss": mb["logloss"] - ma["logloss"], "d_brier": mb["brier"] - ma["brier"]})
        # 2023 ablation: main model (trained with 2023) vs A24 (without 2023) on the same 2024 rows
        main = _oof(ctx, cv_id(chosen, L, pt))
        k = main.merge(a24[["pixel_id", "t0", "p_raw"]], on=["pixel_id", "t0"], suffixes=("_with", "_without"))
        mw = oof_metrics(k["y"], k["p_raw_with"], k["p_clim"], k["pers"], pt)
        mo = oof_metrics(k["y"], k["p_raw_without"], k["p_clim"], k["pers"], pt)
        abl.append({"lead_min": L, "step": "2023: with", "spec": chosen, "what": "trained with 2023 (2024 rows)", **mw})
        abl.append({"lead_min": L, "step": "2023: without", "spec": "A24", "what": "trained without 2023", **mo,
                    "d_logloss": mo["logloss"] - mw["logloss"], "d_brier": mo["brier"] - mw["brier"]})
    pd.DataFrame(abl).to_csv(out / "ablation.csv", index=False)
    # P4A variant under the new scheme (all thresholds), fold spread of every CV unit
    pa, folds, spread = [], [], []
    done = {k[len(ctx.run_files_prefix() + "reports/p4b/cv/"):-5] for k in
            ctx.store.list(ctx.run_files_prefix() + "reports/p4b/cv/") if k.endswith(".json")}
    for uid in sorted(done):
        info = _cv_info(ctx, uid)
        for f in info["folds"]:
            folds.append({"unit": uid, "spec": info["spec"], "lead_min": info["lead_min"],
                          "threshold": info["threshold"], **f})
        fl = pd.DataFrame(info["folds"])
        if len(fl):
            spread.append({"unit": uid, "spec": info["spec"], "lead_min": info["lead_min"],
                           "threshold": info["threshold"], "folds": len(fl), "median_rounds": info["median_rounds"],
                           "capped": info["folds_capped_below_median"],
                           "logloss_min": fl["logloss"].min(), "logloss_max": fl["logloss"].max(),
                           "bss_min": fl["bss_clim"].min(), "bss_max": fl["bss_clim"].max(),
                           "bss_std": fl["bss_clim"].std(), "pooled_logloss": info["pooled"]["logloss"],
                           "pooled_bss": info["pooled"]["bss_clim"]})
        if info["spec"] == "a":
            pa.append({"lead_min": info["lead_min"], "threshold": info["threshold"], **info["pooled"]})
    pd.DataFrame(pa).to_csv(out / "p4a_variant_new_cv.csv", index=False)
    pd.DataFrame(folds).to_csv(out / "folds.csv", index=False)
    pd.DataFrame(spread).to_csv(out / "fold_spread.csv", index=False)
    # chosen v1 models: OOF metrics at their calibration and cut
    from src.models import metrics as M
    fin = []
    for key, m in sel["models"].items():
        o = _oof(ctx, m["cv_unit"])
        p = crossfit(m["calibration"], o) if m["calibration"] != "raw" else o["p_raw"].to_numpy("float64")
        y = o["y"].to_numpy().astype(bool)
        sc = M.scores(**M.contingency(y, p >= m["cut"]["cut"]))
        th = float(key.split("thr")[1].replace("p", "."))
        yp = np.nan_to_num(o["pers"].to_numpy("float64"), nan=-1) >= th
        fin.append({"model": key, "cv_unit": m["cv_unit"], "calibration": m["calibration"], "cut_rule": m["cut"]["rule"],
                    "cut": m["cut"]["cut"], "csi": float(sc["csi"]), "pod": float(sc["pod"]), "far": float(sc["far"]),
                    "bias": float(sc["bias"]), "brier": M.brier(y, p), "bss_clim": M.bss(y, p, o["p_clim"]),
                    "auc_pr": M.auc_pr(y, p), "csi_pers": float(M.scores(**M.contingency(y, yp))["csi"]),
                    "rows": len(o)})
    pd.DataFrame(fin).to_csv(out / "v1_models_oof.csv", index=False)
    summary = {"run_id": ctx.run_id, "job": ctx.job, "gold_id": ctx.gid, "feature_set": feats["feature_set"],
               "setting": chosen, "ship_B": sel["ship_B"], "spec_by_threshold": sel["spec_by_threshold"],
               "evaluation": "OOF only (CV 2023-05..2024-10, IMERG Final); test season untouched",
               "ablation_primary": [{k: r.get(k) for k in ("lead_min", "step", "logloss", "brier", "bss_clim",
                                                           "csi_maxcut_insample", "csi_pers")} for r in abl]}
    write_json(ROOT / "reports/p4b/oof_summary.json", summary)
    rels = [rel(p) for p in sorted(out.glob("*.csv"))] + ["reports/p4b/oof_summary.json"]
    e = [r for r in abl if r["step"] == "e"]
    return rels, {"oof_complete": True, "feature_set": feats["feature_set"], "setting": chosen,
                  "ship_B": sel["ship_B"],
                  "primary_e": [{"lead": r["lead_min"], "logloss": round(r["logloss"], 5), "bss": round(r["bss_clim"], 4),
                                 "csi_pers": round(r["csi_pers"], 4)} for r in e]}


# ============================================================================ stage 7: final test (P4B.6)
def check_final_guard(store, run_id: str) -> tuple[bool, str]:
    """--final runs only if lake/v1/reports/p4b/OOF_DONE.json exists, final_approved is true and it names this run."""
    if not store.list(OOF_DONE_KEY):
        return False, f"{OOF_DONE_KEY} does not exist"
    try:
        o = json.loads(store.get(OOF_DONE_KEY))
    except Exception as exc:                                    # noqa: BLE001
        return False, f"{OOF_DONE_KEY} unreadable ({type(exc).__name__})"
    if o.get("final_approved") is not True:
        return False, "final_approved is not true (the user has not approved the final test)"
    if o.get("run_id") != run_id:
        return False, f"OOF_DONE.json names run {o.get('run_id')!r}, not this code's run {run_id!r}"
    return True, "approved"


def _calibrated(model_d: Path, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    import lightgbm as lgb
    bst = lgb.Booster(model_file=str(model_d / "model.txt"))
    cal = json.loads((model_d / "calibration.json").read_text(encoding="utf-8"))
    pr = bst.predict(X)
    return pr, apply_calibrator(cal, pr), json.loads((model_d / "meta.json").read_text(encoding="utf-8"))


def run_final(ctx: Ctx) -> tuple[list[str], dict]:
    from src.models import baselines as BLN
    from src.models import metrics as M
    cfg, ev = ctx.cfg, ctx.cfg["evaluation"]
    leads, thrs, pt = cfg["frames"]["lead_min"], thresholds(cfg), primary(cfg)
    sel = ctx.read_json("reports/p4b/select/selection.json")
    ctx.fetch_run_files("models/v1/")
    ensure_table(ctx, "A1"); ensure_table(ctx, "B1")
    out = ROOT / "reports/p4b/final"
    out.mkdir(parents=True, exist_ok=True)
    gk = [f"gc_k{k}" for k in range(7)]
    lf = cfg["baselines"]["logistic_features"]
    periods = final_periods(cfg)
    pilot_var = {}
    if pilot_comparison_enabled(cfg):            # pilot v0 test predictions exist only for the real 2025 test season
        man_p = ROOT / "models/pilot_v0/predict_manifest.json"
        ctx.fetch_lake(["models/pilot_v0/predict_manifest.json"], "models/pilot_v0/")
        pilot_var = json.loads(man_p.read_text(encoding="utf-8"))["threshold_variant"] if man_p.is_file() else {}
    rows, diffs, verd, months, rel_rows, wet = [], [], [], [], [], []
    names = {"test": "test (IMERG Final)", "secondary": "secondary (IMERG Late, provisional; never used for choices)"}
    for pname in ("test", "secondary"):
        per = [periods[pname]]
        for L in leads:
            featsA = json.loads((model_dir("A", L, pt) / "meta.json").read_text(encoding="utf-8"))["features"]
            featsB = json.loads((model_dir("B", L, pt) / "meta.json").read_text(encoding="utf-8"))["features"]
            A = load_rows(ctx, "A1", per, list(dict.fromkeys(featsA + gk + lf)), lead=L)
            B = load_rows(ctx, "B1", per, featsB, lead=L)[["pixel_id", "t0"] + [c for c in featsB if c.startswith("nwp_")]]
            df = A.merge(B, on=["pixel_id", "t0"], how="left")
            cvA = load_rows(ctx, "A1", cv_folds(cfg), list(dict.fromkeys(gk + lf)), lead=L)
            pers = persistence(df, L)
            for t in thrs:
                yc, k = thr_col(t), kstr(t)
                y = df[yc].to_numpy().astype(bool)
                pclim = clim_for(cvA, df, yc)
                preds = {}
                for v in ("A", "B"):
                    md = model_dir(v, L, t)
                    pr, pc, meta = _calibrated(md, df[meta_feats(md)].to_numpy("float32"))
                    yh = pc >= meta["cut"]["cut"]
                    if meta.get("heavy_rule") == "nested":
                        q = model_dir(v, L, pt)
                        _, pq, mq = _calibrated(q, df[meta_feats(q)].to_numpy("float32"))
                        yh &= pq >= mq["cut"]["cut"]
                    preds[f"v1_{v}"] = (pc, yh, pr)
                nwp_ok = df[[c for c in featsB if c.startswith("nwp_")]].notna().all(axis=1).to_numpy()
                if sel["ship_B"]:
                    pc = np.where(nwp_ok, preds["v1_B"][0], preds["v1_A"][0])
                    yh = np.where(nwp_ok, preds["v1_B"][1], preds["v1_A"][1])
                    pr = np.where(nwp_ok, preds["v1_B"][2], preds["v1_A"][2])
                    preds["v1"] = (pc, yh, pr)
                else:
                    preds["v1"] = preds["v1_A"]
                # baselines (fitted on the CV period only)
                base = {"clim": (pclim, pclim >= M.choose_cut(cvA[yc].to_numpy().astype(bool), clim_for(cvA, cvA, yc),
                                                              "max_csi" if t != float(cfg["targets"]["secondary_mm_h"])
                                                              else "freq_bias_1.5")["cut"])}
                yp = np.nan_to_num(pers, nan=-1.0) >= t
                base["pers"] = (yp.astype("float64"), yp)
                cp = BLN.calpers_cut(cvA["gc_k0"].to_numpy("float64"), cvA[yc].to_numpy().astype(bool))
                ycp = np.nan_to_num(df["gc_k0"].to_numpy("float64"), nan=-1.0) >= cp["cut_mm_h"]
                base["calpers"] = (ycp.astype("float64"), ycp)
                lg = BLN.logit_fit(cvA[lf].to_numpy("float64"), cvA[yc].to_numpy(), cfg["random_seed"])
                pl_cv = lg.predict_proba(np.nan_to_num(cvA[lf].to_numpy("float64")))[:, 1]
                rule = "max_csi" if t != float(cfg["targets"]["secondary_mm_h"]) else "freq_bias_1.5"
                cl = M.choose_cut(cvA[yc].to_numpy().astype(bool), pl_cv, rule)["cut"]
                pl = lg.predict_proba(np.nan_to_num(df[lf].to_numpy("float64")))[:, 1]
                base["logit"] = (pl, pl >= cl)
                if pname == "test" and f"{t:g}" in pilot_var:
                    r = f"{PILOT_REL}preds/lgbm_{pilot_var[f'{t:g}']}_lead{L}_thr{k}_test.parquet"
                    ctx.fetch_lake([r], PILOT_REL + "preds/")
                    if (ROOT / r).is_file():
                        pv = pd.read_parquet(ROOT / r, columns=["pixel_id", "t0", "p", "yhat"])
                        pv["t0"] = pv["t0"].astype("datetime64[ns, UTC]")
                        j = df[["pixel_id", "t0"]].assign(t0=df["t0"].astype("datetime64[ns, UTC]")).merge(
                            pv, on=["pixel_id", "t0"], how="left")
                        base["pilot_v0"] = (j["p"].to_numpy("float64"), j["yhat"].fillna(0).to_numpy().astype(bool))
                okb = np.ones(len(df), bool)
                for nm, (pb, _) in base.items():
                    okb &= np.isfinite(pb)
                ds = {}
                for nm, (p, yh, *_) in list(preds.items()) + list(base.items()):
                    p = np.asarray(p, "float64")
                    s = M.summary(y[okb], p[okb], np.asarray(yh)[okb], pclim[okb])
                    d_ = M.day_stats(df["day"].to_numpy()[okb], y[okb], np.asarray(yh)[okb], p[okb], pclim[okb])
                    ci = M.bootstrap_ci(d_, ev["bootstrap_resamples"], cfg["random_seed"], ev["ci_level"])
                    ds[nm] = d_
                    rows.append({"period": pname, "lead_min": L, "threshold": t, "model": nm, **s,
                                 "csi_lo": ci["csi"][0], "csi_hi": ci["csi"][1], "bss_lo": ci["bss_clim"][0],
                                 "bss_hi": ci["bss_clim"][1]})
                    if nm in ("v1", "v1_A", "v1_B"):
                        rel_rows.append(M.reliability(y[okb], p[okb], ev["reliability_bins"]).assign(
                            period=pname, lead_min=L, threshold=t, model=nm))
                wins, worse = [], []
                for nm in base:
                    d = M.paired_bootstrap(ds["v1"], ds[nm], ev["bootstrap_resamples"], cfg["random_seed"], ev["ci_level"])
                    w = d["csi"]["lo"] > 0 and d["brier"]["hi"] < 0
                    b = d["csi"]["hi"] < 0 or d["brier"]["lo"] > 0
                    wins.append((nm, w)); worse.append((nm, b))
                    diffs.append({"period": pname, "lead_min": L, "threshold": t, "baseline": nm,
                                  "csi_diff": d["csi"]["diff"], "csi_lo": d["csi"]["lo"], "csi_hi": d["csi"]["hi"],
                                  "brier_diff": d["brier"]["diff"], "brier_lo": d["brier"]["lo"],
                                  "brier_hi": d["brier"]["hi"], "win": w, "significantly_worse": b})
                verd.append({"period": pname, "lead_min": L, "threshold": t,
                             "verdict": "BEATS_ALL" if all(w for _, w in wins) else
                             ("LOSES" if any(b for _, b in worse) else "TIES"),
                             "wins_vs": ",".join(n for n, w in wins if w), "worse_than": ",".join(n for n, b in worse if b)})
                mon = df["t0"].dt.tz_convert(LOCAL_TZ).dt.month.to_numpy()
                for mm in np.unique(mon):
                    s_ = mon == mm
                    for nm in ("v1", "pers"):
                        p, yh = (preds[nm][0], preds[nm][1]) if nm == "v1" else base["pers"]
                        months.append({"period": pname, "lead_min": L, "threshold": t, "month": int(mm), "model": nm,
                                       **M.summary(y[s_], np.asarray(p, "float64")[s_], np.asarray(yh)[s_], pclim[s_])})
                if pname == "test" and t == pt:
                    lab0 = df.groupby("day")["label_rate_mm_h"].mean().sort_values(ascending=False)
                    for day in lab0.index[:int(ev["wettest_days"])]:
                        s_ = df["day"].to_numpy() == day
                        for nm in ("v1", "pers"):
                            p, yh = (preds[nm][0], preds[nm][1]) if nm == "v1" else base["pers"]
                            sc = M.scores(**M.contingency(y[s_], np.asarray(yh)[s_]))
                            wet.append({"day": day, "city_mean_label_mm_h": float(lab0[day]), "lead_min": L,
                                        "model": nm, "csi": float(sc["csi"]), "pod": float(sc["pod"]),
                                        "far": float(sc["far"]), "bias": float(sc["bias"]),
                                        "brier": M.brier(y[s_], np.asarray(p, "float64")[s_])})
                pp = df[["pixel_id", "t0"]].assign(lead_min=np.int16(L), y=y.astype("int8"),
                                                     **{f"p_{nm}": np.asarray(v_[0], "float32") for nm, v_ in preds.items()},
                                                     **{f"yhat_{nm}": np.asarray(v_[1]).astype("int8")
                                                        for nm, v_ in preds.items()})
                write_parquet(pp, out / "preds" / f"{pname}_lead{L}_thr{k}.parquet")
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(diffs).to_csv(out / "diffs.csv", index=False)
    pd.DataFrame(verd).to_csv(out / "verdict.csv", index=False)
    pd.DataFrame(months).to_csv(out / "by_month.csv", index=False)
    pd.DataFrame(wet).to_csv(out / "wettest_days.csv", index=False)
    if rel_rows:
        pd.concat(rel_rows, ignore_index=True).to_csv(out / "reliability.csv", index=False)
    write_json(out / "periods.json", {k: names[k] for k in names} | {
        "test": [str(x) for x in period(cfg, "test")], "secondary": [str(x) for x in period(cfg, "secondary")]})
    rels = [rel(p) for p in sorted(out.rglob("*")) if p.is_file()]
    v = pd.DataFrame(verd)
    return rels, {"verdicts": v.to_dict("records")[:30]}


def final_periods(cfg: dict) -> dict:
    """Periods read by the final unit. In p4b_smoke both lie in 2024 (never the 2025 test season nor October 2025)."""
    out = {name: period(cfg, name) for name in ("test", "secondary")}
    if cfg.get("smoke"):
        limit = pd.Timestamp("2025-01-01", tz="UTC")
        assert all(b <= limit for _, b in out.values()), "smoke final must not read 2025"
    return out


def pilot_comparison_enabled(cfg: dict) -> bool:
    """The pilot v0 baseline uses its stored 2025 test-season predictions: real p4b job only, never the smoke."""
    return not cfg.get("smoke")


def meta_feats(md: Path) -> list[str]:
    return json.loads((md / "meta.json").read_text(encoding="utf-8"))["features"]


# ============================================================================ dispatch
GATES = {"g2": gate_g2, "sel-features": gate_sel_features, "sel-setting": gate_sel_setting,
         "oof-select": gate_oof_select, "oof-report": gate_oof_report}


def run_unit(unit: Unit, ctx: Ctx) -> tuple[list[str], dict]:
    if unit.kind == "feat":
        return run_feat(unit, ctx)
    if unit.kind == "cv":
        return run_cv(unit, ctx)
    if unit.kind == "refit":
        return run_refit(unit, ctx)
    if unit.kind == "gate":
        return GATES[unit.id](ctx)
    if unit.kind == "final":
        return run_final(ctx)
    raise ValueError(unit.kind)
