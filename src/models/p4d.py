"""P4D — rainfall nowcast v2 candidates, "cut" (Model 1) vs "weight" (Model 2): units of the cloud jobs `p4d` and
`p4d_smoke` (driver: src/models/cloud_train.py). Design: configs/model_v2.yaml and REPORT_P4D §1 (frozen before any
training). Not a flood model; labels are IMERG (satellite estimate), not gauges. No 2022, no substitute, no --final.

Stages (a unit runs when all its dependencies have a done marker on R2; `--part i/n` splits every stage round-robin;
gates are run by whichever part first sees their dependencies done, committed race-safely):
  0  preflight        gate: R2 bucket total + projection <= r2_budget.max_total_gb; ring geometry vs feature footprint
  1  feat-YYYY-MM     Gold table_v2 for one month (city + ring pixels; src/gold/build_table_v2.py), frame-mode G2
  2  g2               gate: G2 on every partition re-read from R2, duplicates, NaN share city vs ring, pixel table,
                      city rows == P4B table_A1_v1 on the common columns (single feature code path)
  3  base-s<S>-lead<L> evaluation rows of held-out season S (city pixels): labels, climatology fitted on the
                      training seasons, persistence, day / half-month / role (eval | secondary = IMERG Late 2025-10)
     fit-m0-...       M0 = v1 configuration (set d, g1) on the 47 city pixels, every threshold
     fit-m1a-...      M0 data with the Model 1 cut (training rows of 1-15 May and 16-31 Oct dropped), 2.5 mm/h
     sel-m1a          gate: keep m1a only if the mean recalibrated pooled OOF log-loss (2.5, F0/F30/F60) decreases
  4  fit-m1b-...      previous kept step + ring pixels (+ dist_to_city_center_km, is_city), 2.5 mm/h
     sel-m1b          gate: same rule -> Model 1 = chain end; dataset of Model 2 = that dataset without the cut
  5  fit-m1-...       final Model 1 at 0.5 / 7.6 mm/h (alias of m0 when the chain end is m0)
     fit-m2-g<n>-...  Model 2 grid (k, alpha, beta) at 2.5 and 7.6 mm/h
     sel-m2           gate: lowest mean recalibrated pooled OOF log-loss at 2.5; 7.6 CSI/POD per setting
  6  fit-m2-...       chosen Model 2 setting at the remaining thresholds (0.5)
  7  cmp-main         gate: M0 / Model 1 / Model 2 per frame and threshold, bootstrap differences, reliability, per
                      fold, secondary (IMERG Late 2025-10, apart)
     cmp-detail       gate: per half-month, 10 wettest days per season, Yagi (2024-09) and Matmo (2025-10, Late)
  8  compare-done     gate: summary; its winner writes runs/<job>/<run_id>/summary.json and COMPARE_DONE.json (STOP)

Every fit unit: LOSO fold S (training = the other seasons, IMERG Final rows only), inner blocked CV by calendar
month inside the training seasons (early stopping on the held-out month: city pixels, all days, unweighted; rounds =
median of the fold optima), Platt fitted on the inner OOF (cross-fit), cut chosen on the recalibrated inner OOF,
refit on all training rows, prediction of held-out season S (city pixels). Output files carry no timings, so a
re-run produces the same bytes (race fix, REPORT_P4B §8.1).

Gold layout on R2: lake/v1/gold/met_training/table_v2/gid=<gold id>/season=YYYY/month=MM/part.parquet.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.models import p4b as PB

CODE_ROOT = Path(__file__).resolve().parents[2]
CFG_PATH = CODE_ROOT / "configs" / "model_v2.yaml"
LAKE = "lake/v1/"
TABLE = "table_v2"
GOLD_REMOTE = LAKE + "gold/met_training/" + TABLE + "/"
GOLD_BUILD_VERSION = "p4d-gold-1"
A1_REF = {"table": "table_A1_v1", "gold_id": "0803d9d8ed72"}      # P4B Gold (run c0df12477947), read only
CODE_FILES = ["src/models/cloud_train.py", "src/models/p4d.py", "src/models/p4b.py", "src/gold/build_table_v2.py",
              "src/gold/features.py", "src/gold/build_labels.py", "src/gold/build_training_table.py",
              "src/models/metrics.py", "src/models/baselines.py", "src/models/data.py",
              "src/quality/temporal_checks.py", "configs/model_v2.yaml"]
GOLD_CODE = ["src/gold/build_table_v2.py", "src/gold/features.py", "src/gold/build_labels.py",
             "src/gold/build_training_table.py"]
NOT_FLOOD = "rainfall nowcast, not a flood prediction; labels IMERG (satellite estimate), not rain gauges"
LOCAL_TZ = "Asia/Ho_Chi_Minh"
GK = [f"gc_k{k}" for k in range(7)]
OUT = "reports/p4d"


def root() -> Path:
    """Workspace (data, outputs) = p4b.ROOT, so tests redirect both modules at once."""
    return PB.ROOT


# ============================================================================ configuration
def load_cfg(smoke: bool = False, path: Path | None = None) -> dict:
    cfg = yaml.safe_load(Path(path or CFG_PATH).read_text(encoding="utf-8"))
    sm = cfg.pop("smoke", {})
    if smoke:
        cfg = PB.deep_merge(cfg, sm)
        cfg["smoke"] = True
    return cfg


kstr, thr_col = PB.kstr, PB.thr_col


def thresholds(cfg: dict) -> list[float]:
    return [float(t) for t in cfg["targets"]["thresholds_mm_h"]]


def primary(cfg: dict) -> float:
    return float(cfg["targets"]["primary_mm_h"])


def heavy(cfg: dict) -> float:
    return float(cfg["targets"]["secondary_mm_h"])


def leads(cfg: dict) -> list[int]:
    return [int(x) for x in cfg["frames"]["lead_min"]]


def seasons(cfg: dict) -> list[int]:
    return [int(s) for s in cfg["evaluation"]["loso_seasons"]]


def grid_settings(cfg: dict) -> list[dict]:
    g = cfg["model2"]["grid"]
    return [{"k": float(k), "alpha": float(a), "beta": float(b)} for k, a, b in itertools.product(g["k"], g["alpha"],
                                                                                                    g["beta"])]


def lgb_params(cfg: dict) -> dict:
    p = dict(cfg["lightgbm"])
    p.update(seed=cfg["random_seed"], deterministic=True, force_row_wise=True)
    return p


def gold_id(cfg: dict) -> str:
    h = hashlib.sha256(GOLD_BUILD_VERSION.encode())
    for f in GOLD_CODE:
        h.update((CODE_ROOT / f).read_bytes().replace(b"\r\n", b"\n"))
    keep = {k: cfg[k] for k in ("frames", "features", "availability", "targets", "scope", "pixels")}
    h.update(json.dumps(keep, sort_keys=True, default=str).encode())
    return h.hexdigest()[:12]


def _md(mmdd: str) -> int:
    m, d = mmdd.split("-")
    return int(m) * 100 + int(d)


def cut_mask(md: np.ndarray, cfg: dict) -> np.ndarray:
    """True for rows dropped by the Model 1 cut (UTC date of t0 as month*100 + day)."""
    out = np.zeros(len(md), bool)
    for a, b in cfg["model1"]["cut_utc"]:
        out |= (md >= _md(a)) & (md <= _md(b))
    return out


def sample_weights(md: np.ndarray, label: np.ndarray, w: dict, cfg: dict) -> np.ndarray:
    """Model 2: w_period x w_class; w_period = k inside model2.period_window_utc, w_class = 1 + alpha [label >= 2.5]
    + beta [label >= 7.6] (label = IMERG rate of the frame being trained)."""
    a, b = cfg["model2"]["period_window_utc"]
    wp = np.where((md >= _md(a)) & (md <= _md(b)), w["k"], 1.0)
    lab = np.nan_to_num(np.asarray(label, "float64"), nan=0.0)
    wc = 1.0 + w["alpha"] * (lab >= primary(cfg)) + w["beta"] * (lab >= heavy(cfg))
    return (wp * wc).astype("float32")


# ============================================================================ units
def fit_id(spec: str, s: int, lead: int, thr: float) -> str:
    return f"fit-{spec}-s{s}-lead{lead}-thr{kstr(thr)}"


def base_id(s: int, lead: int) -> str:
    return f"base-s{s}-lead{lead}"


def units(cfg: dict) -> list:
    from src.gold import build_table_v2 as BT2
    U = PB.Unit
    Ls, thrs, pt, Ss = leads(cfg), thresholds(cfg), primary(cfg), seasons(cfg)
    others = [t for t in thrs if t != pt]
    gthr = [float(t) for t in cfg["model2"]["grid_thresholds"] if float(t) in thrs]
    nset = len(grid_settings(cfg))
    u = [U("preflight", "gate", 0)]
    feat = [U(f"feat-{y}-{m:02d}", "feat", 1, ("preflight",), {"y": y, "m": m}) for y, m in BT2.build_months(cfg)]
    u += feat
    u.append(U("g2", "gate", 2, tuple(x.id for x in feat)))

    def fit(spec, s, L, t, stage, deps):
        return U(fit_id(spec, s, L, t), "fit", stage, deps, {"spec": spec, "season": s, "lead": L, "thr": t})
    s3, base = [], []
    for L in Ls:                       # ordered by lead: a part keeps one data slice while its units share a lead
        for s in Ss:
            base.append(U(base_id(s, L), "base", 3, ("g2",), {"season": s, "lead": L}))
            s3 += [fit("m0", s, L, t, 3, ("g2",)) for t in thrs] + [fit("m1a", s, L, pt, 3, ("g2",))]
    u += base + s3
    u.append(U("sel-m1a", "gate", 3, tuple(x.id for x in base) +
               tuple(x.id for x in s3 if x.params["thr"] == pt)))
    s4 = [fit("m1b", s, L, pt, 4, ("sel-m1a",)) for L in Ls for s in Ss]
    u += s4
    u.append(U("sel-m1b", "gate", 4, ("sel-m1a",) + tuple(x.id for x in s4)))
    s5 = []
    for L in Ls:
        for s in Ss:
            s5 += [fit("m1", s, L, t, 5, ("sel-m1b",)) for t in others]
            s5 += [fit(f"m2-g{n}", s, L, t, 5, ("sel-m1b",)) for n in range(1, nset + 1) for t in gthr]
    u += s5
    u.append(U("sel-m2", "gate", 5, ("sel-m1b",) + tuple(x.id for x in s5 if x.params["spec"].startswith("m2-"))))
    s6 = [fit("m2", s, L, t, 6, ("sel-m2",)) for L in Ls for s in Ss for t in thrs if t not in gthr]
    u += s6
    alldeps = tuple(x.id for x in base + s3 + s4 + s5 + s6) + ("sel-m1b", "sel-m2")
    u.append(U("cmp-main", "gate", 7, alldeps))
    u.append(U("cmp-detail", "gate", 7, alldeps))
    u.append(U("compare-done", "gate", 8, ("cmp-main", "cmp-detail")))
    return u


part_units = PB.part_units


# ============================================================================ run context
class DataCache:
    def __init__(self):
        self.key, self.val = None, None

    def get(self, key, loader):
        if key != self.key:
            self.key, self.val = None, None
            import gc
            gc.collect()
            self.val, self.key = loader(), key
        return self.val


class Ctx(PB.Ctx):
    """p4b.Ctx with the table_v2 gold id and layout."""

    def __init__(self, cfg: dict, store, job: str, run_id: str, mirror: Path, log):
        self.cfg, self.store, self.job, self.run_id, self.mirror, self.log = cfg, store, job, run_id, mirror, log
        self.gid = gold_id(cfg)
        self._ls: dict[str, dict] = {}
        self.cache = DataCache()
        self._pix = None
        self._base: dict = {}

    def gold_key(self, y: int, m: int, name: str = "part.parquet") -> str:          # noqa: D401 (override)
        return f"{GOLD_REMOTE}gid={self.gid}/season={y}/month={m:02d}/{name}"

    def gold_local(self) -> Path:                                                   # noqa: D401 (override)
        return root() / "data" / "gold" / "met_training" / "v2" / TABLE / f"gid={self.gid}"

    def fetch_gold(self) -> int:                                                    # noqa: D401 (override)
        pre = f"{GOLD_REMOTE}gid={self.gid}/"
        n = 0
        for k, size in sorted(self.store.list(pre).items()):
            n += self._download(k, size, self.gold_local() / k[len(pre):])
        return n


def write_json(p: Path, obj) -> None:
    PB.write_json(p, obj)


def rel(p: Path) -> str:
    return Path(p).resolve().relative_to(root().resolve()).as_posix()


def out_dir(*parts: str) -> Path:
    p = root().joinpath(OUT, *parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ============================================================================ stage 0: preflight
def bucket_total_bytes(store) -> int:
    if hasattr(store, "root"):                         # LocalStore (tests)
        return sum(p.stat().st_size for p in Path(store.root).rglob("*") if p.is_file())
    return int(sum(store.list("").values()))


def gate_preflight(ctx: Ctx) -> tuple[list[str], dict]:
    from src.gold import build_table_v2 as BT2
    from src.gold import features as F
    cfg, rb = ctx.cfg, ctx.cfg["r2_budget"]
    total = bucket_total_bytes(ctx.store)
    own = sum(ctx.store.list(f"{GOLD_REMOTE}gid={ctx.gid}/").values()) + \
        sum(ctx.store.list(f"{LAKE}runs/{ctx.job}/{ctx.run_id}/").values())
    proj = max(0.0, (float(rb["projected_gold_gb"]) + float(rb["projected_run_gb"])) * 1e9 - own)
    p = F.feature_params(cfg)
    city, ring, prec = pixel_sets(ctx)              # raises if min_margin_px is not the configured rule's margin
    rec = {"bucket_total_bytes": total, "this_run_bytes_already": own, "projected_new_bytes": proj,
           "projected_total_gb": round((total + proj) / 1e9, 3), "max_total_gb": float(rb["max_total_gb"]),
           "required_margin_px": BT2.required_margin_px(p), "ring_min_margin_px": cfg["pixels"]["ring"]["min_margin_px"],
           "ring": prec,
           "gold_id": ctx.gid, "days": len(BT2.build_days(cfg)), "months": len(BT2.build_months(cfg)),
           "smoke": bool(cfg.get("smoke"))}
    ok_geo = prec["ring_pixels"] > 0 and prec["min_margin_px"] == prec["largest_margin_meeting_target"]
    ok_budget = rec["projected_total_gb"] <= rec["max_total_gb"]
    rec |= {"ok_geometry": ok_geo, "ok_budget": ok_budget}
    pth = out_dir() / "preflight.json"
    write_json(pth, rec)
    if not (ok_geo and ok_budget):
        raise RuntimeError(f"preflight FAIL (nothing built): {rec}")
    return [rel(pth)], {"r2_total_gb": round(total / 1e9, 3), "projected_total_gb": rec["projected_total_gb"],
                        "ring_pixels": prec["ring_pixels"], "ring_margin_px": prec["min_margin_px"], "ok": True}


# ============================================================================ stage 1: Gold table_v2
def _silver_rel(source: str, d) -> str:
    return f"data/silver/met/{source}/date={d.isoformat()}/part.parquet"


def pixel_sets(ctx: Ctx) -> tuple[np.ndarray, np.ndarray, dict]:
    """City (is_city_pixel of IMERG Silver on the first build day, D05) and ring pixels (configured rule)."""
    if ctx._pix is None:
        from src.gold import build_table_v2 as BT2
        d0 = BT2.build_days(ctx.cfg)[0]
        r = _silver_rel("imerg_final", d0)
        ctx.fetch_lake([r], "data/silver/met/imerg_final/")
        t = pd.read_parquet(root() / r, columns=["pixel_id", "is_city_pixel"]).drop_duplicates("pixel_id")
        city = np.sort(t.loc[t["is_city_pixel"].astype(bool), "pixel_id"].to_numpy("int64"))
        ring, rec = BT2.ring_pixels(city, ctx.cfg)
        ctx._pix = (city, ring, rec | {"city_from": r})
    return ctx._pix


def run_feat(unit, ctx: Ctx) -> tuple[list[str], dict]:
    from src.gold import build_table_v2 as BT2
    from src.gold import build_training_table as BT
    from src.gold import features as F
    cfg, y, m = ctx.cfg, unit.params["y"], unit.params["m"]
    days = [d for d in BT2.build_days(cfg) if (d.year, d.month) == (y, m)]
    city, ring, prec = pixel_sets(ctx)
    pixels = np.sort(np.r_[city, ring])
    src = BT2.label_source(cfg, y, m)
    ctx.fetch_lake([_silver_rel(src, d) for d in days + [days[-1] + timedelta(days=1)]], f"data/silver/met/{src}/")
    need = sorted({d - timedelta(days=1) for d in days} | set(days))
    ctx.fetch_lake([_silver_rel("gsmap_now", d) for d in need], "data/silver/met/gsmap_now/")
    grid = F.Grid()
    wins = BT.load_windows(need, grid)
    df, fcols, st = BT2.build_partition(cfg, y, m, days, pixels, city, wins, grid)
    v = BT2.g2_violations(df, fcols, CFG_PATH)
    if v:
        raise RuntimeError(f"G2 violations in {TABLE} {y}-{m:02d}: {v[:3]}")
    lp = ctx.gold_local() / f"season={y}" / f"month={m:02d}" / "part.parquet"
    PB.write_parquet(df, lp)
    nan = BT2.nan_share_by_group(df, fcols)
    meta = {"table": TABLE, "gold_id": ctx.gid, "season": y, "month": m, "rows": len(df), "features": fcols,
            "label_source": src, "training_rows": BT2.is_training_month(cfg, y, m),
            "days": [d.isoformat() for d in days], "pixels": prec, "stats": st, "g2_violations": 0,
            "nan_share": {r["pixels"]: {c: round(r[c], 5) for c in fcols} for r in nan},
            "smoke": bool(cfg.get("smoke")), "note": NOT_FLOOD}
    write_json(lp.with_name("part.meta.json"), meta)
    keys = []
    for name in ("part.parquet", "part.meta.json"):
        k = ctx.gold_key(y, m, name)
        ctx.put_own(k, (lp.parent / name).read_bytes())
        keys.append(k)
    return [], {"rows": len(df), "rows_city": st["rows_city"], "rows_ring": st["rows_ring"], "days": len(days),
                "label_source": src, "g2_violations": 0, "gold_keys": keys}


def gold_parts(ctx: Ctx) -> list[tuple[int, int, Path]]:
    out = []
    for part in sorted(ctx.gold_local().glob("season=*/month=*/part.parquet")):
        out.append((int(part.parent.parent.name.split("=")[1]), int(part.parent.name.split("=")[1]), part))
    return out


def gate_g2(ctx: Ctx) -> tuple[list[str], dict]:
    from src.gold import build_table_v2 as BT2
    ctx.fetch_gold()
    rows, nan, labs = [], [], []
    for y, m, part in gold_parts(ctx):
        df = pd.read_parquet(part)
        fcols = BT2.feature_columns(df.columns)
        v = BT2.g2_violations(df, fcols, CFG_PATH)
        rows.append({"partition": f"season={y}/month={m:02d}", "rows": len(df), "features": len(fcols),
                     "violations": len(v), "duplicates": int(df.duplicated(["pixel_id", "t0", "lead_min"]).sum()),
                     "rows_city": int((df["is_city"] == 1).sum()), "rows_ring": int((df["is_city"] == 0).sum()),
                     "pixels_city": int(df.loc[df["is_city"] == 1, "pixel_id"].nunique()),
                     "pixels_ring": int(df.loc[df["is_city"] == 0, "pixel_id"].nunique()),
                     "labels_present": all(c in df.columns for c in ["label_rate_mm_h"] +
                                           [thr_col(t) for t in thresholds(ctx.cfg)]),
                     "label_source": str(df["label_source"].iloc[0]) if len(df) else "",
                     "bytes": int((part).stat().st_size), "detail": "; ".join(v[:3])})
        for r in BT2.nan_share_by_group(df, fcols):
            nan.append({"partition": rows[-1]["partition"], **r})
        c = df.loc[df["is_city"] == 1, ["season", "t0", "lead_min"] + [thr_col(t) for t in thresholds(ctx.cfg)]].copy()
        c["train_month"] = BT2.is_training_month(ctx.cfg, y, m)
        labs.append(c)
        del df
    r = pd.DataFrame(rows)
    nn = pd.DataFrame(nan)
    out = out_dir("g2")
    r.to_csv(out / "partitions.csv", index=False)
    nn.to_csv(out / "nan_share_by_partition.csv", index=False)
    fcols = [c for c in nn.columns if c not in ("partition", "pixels", "rows")]
    agg, nrows = [], {}
    for g, sub in nn.groupby("pixels"):
        w = sub["rows"].to_numpy("float64")
        nrows[g] = int(w.sum())
        agg.append({"pixels": g, **{c: float(np.nansum(sub[c].to_numpy("float64") * w) / max(w.sum(), 1))
                                    for c in fcols}})
    a = pd.DataFrame(agg).set_index("pixels").T.reset_index().rename(columns={"index": "feature"})
    a["ring_minus_city"] = a.get("ring", np.nan) - a.get("city", np.nan)
    a.to_csv(out / "nan_share_city_vs_ring.csv", index=False)
    city, ring, prec = pixel_sets(ctx)
    from src.gold import features as F
    pt = BT2.pixel_table(city, ring, F.feature_params(ctx.cfg))
    pt.to_csv(out / "pixels.csv", index=False)
    cov = [c for c in pt.columns if c.startswith("box_r") or c in ("ring_coverage", "ring_quadrant_min", "adv_inside")]
    tr = []
    for g, sub in pt.groupby(pt["is_city"].map({1.0: "city", 0.0: "ring"})):
        for c in cov:
            tr.append({"pixels": g, "footprint": c, "n_pixels": len(sub), "mean": float(sub[c].mean()),
                       "min": float(sub[c].min()), "share_truncated": float((sub[c] < 1 - 1e-9).mean())})
    pd.DataFrame(tr).to_csv(out / "truncation_city_vs_ring.csv", index=False)
    lab = pd.concat(labs, ignore_index=True)
    lab["t0"] = lab["t0"].astype("datetime64[ns, UTC]")
    icv = inner_cv_label_check(ctx.cfg, lab)
    del lab, labs
    pd.DataFrame(icv).to_csv(out / "inner_cv_check.csv", index=False)
    bad_icv = [r for r in icv if not r["ok"]]
    eq = a1_equality(ctx)
    write_json(out / "a1_equality.json", eq)
    write_json(out / "pixels.json", prec)
    ok = len(r) > 0 and (r["violations"] == 0).all() and (r["duplicates"] == 0).all() and r["labels_present"].all()
    ok_eq = eq.get("rows_compared", 0) > 0 and eq["max_abs_diff"] <= 1e-6 and eq["nan_mismatch"] == 0
    gold_bytes = int(sum(ctx.store.list(f"{GOLD_REMOTE}gid={ctx.gid}/").values()))
    m = {"partitions": len(r), "rows": int(r["rows"].sum()) if len(r) else 0,
         "violations": int(r["violations"].sum()) if len(r) else -1,
         "duplicates": int(r["duplicates"].sum()) if len(r) else -1, "a1_equal": bool(ok_eq),
         "a1_max_abs_diff": eq.get("max_abs_diff"), "ring_pixels": prec["ring_pixels"],
         "ring_margin_px": prec["min_margin_px"], "ring_pixels_truncated": prec["ring_pixels_truncated"],
         "ring_coverage_min": prec["ring_coverage_min"], "ring_coverage_mean": prec["ring_coverage_mean"],
         "city_pixels": prec["city_pixels"], "gold_bytes": gold_bytes,
         "nan_share_max_city": float(a["city"].max()) if "city" in a else None,
         "nan_share_max_ring": float(a["ring"].max()) if "ring" in a else None,
         "nan_share_rows": nrows, "inner_cv_ok": not bad_icv, "inner_cv_failing": bad_icv[:10]}
    write_json(out / "summary.json", m)
    if bad_icv:
        raise RuntimeError(f"G2 gate FAIL: inner CV has < 2 usable blocks (held-out month and training rows each "
                           f"need a positive city row) for {len(bad_icv)} fold/variant/frame/threshold cases, e.g. "
                           f"{bad_icv[:3]}; choose other days (scope.day_ranges) — no fit unit was started")
    if not (ok and ok_eq):
        raise RuntimeError(f"G2 gate FAIL: {m}")
    return [rel(p) for p in sorted(out.glob("*")) if p.is_file()], m


def a1_equality(ctx: Ctx) -> dict:
    """City rows of table_v2 == P4B table_A1_v1 (gold 0803d9d8ed72) on every common column, first month."""
    from src.gold import build_table_v2 as BT2
    parts = gold_parts(ctx)
    if not parts:
        return {"rows_compared": 0, "note": "no table_v2 partition"}
    y, m, part = parts[0]
    rp = f"data/gold/met_training/{A1_REF['table']}/gid={A1_REF['gold_id']}/season={y}/month={m:02d}/part.parquet"
    k = LAKE + "gold/met_training/" + rp.split("data/gold/met_training/")[1]
    ls = ctx.store.list(k)
    if k not in ls:
        return {"rows_compared": 0, "reference": k, "note": "reference partition missing"}
    dst = root() / "cloud_in" / "a1" / f"{y}-{m:02d}.parquet"
    ctx._download(k, ls[k], dst)
    ref, v2 = pd.read_parquet(dst), pd.read_parquet(part)
    v2 = v2[v2["is_city"] == 1]
    common = [c for c in BT2.feature_columns(ref.columns) if c in v2.columns] + ["label_rate_mm_h"]
    j = v2[["pixel_id", "t0", "lead_min"] + common].merge(ref[["pixel_id", "t0", "lead_min"] + common],
                                                          on=["pixel_id", "t0", "lead_min"], suffixes=("_v2", "_v1"))
    diffs = {c: float(np.nanmax(np.abs(j[c + "_v2"].to_numpy("float64") - j[c + "_v1"].to_numpy("float64")),
                                initial=0.0)) for c in common}
    nanm = {c: int((j[c + "_v2"].isna() != j[c + "_v1"].isna()).sum()) for c in common}
    return {"reference": k, "partition": f"{y}-{m:02d}", "rows_v2_city": int(len(v2)), "rows_compared": int(len(j)),
            "common_columns": len(common), "max_abs_diff": max(diffs.values(), default=0.0),
            "nan_mismatch": int(sum(nanm.values())),
            "v2_only_columns": [c for c in BT2.feature_columns(v2.columns) if c not in common]}


# ============================================================================ inner-CV feasibility (fail fast)
VARIANTS = {"plain": False, "cut": True}          # m0 / m1b-without-cut / m2 (no cut) and m1a / m1b-with-cut


def _train_mask_days(days, held: tuple[int, int], buf: int, use_cut: bool, cfg: dict) -> list:
    from datetime import date
    y, m = held
    a = date(y, m, 1)
    b = date(y + (m == 12), m % 12 + 1, 1)
    out = []
    for d in days:
        if (d.year, d.month) == held:
            continue
        if not (d <= a - timedelta(days=buf + 1) or d >= b + timedelta(days=buf)):
            continue
        if use_cut and cut_mask(np.array([d.month * 100 + d.day]), cfg)[0]:
            continue
        out.append(d)
    return out


def inner_cv_plan_check(cfg: dict) -> list[dict]:
    """Structural check from the configured days only (plan time, before any unit runs): for every LOSO fold and
    every training variant (without / with the Model 1 cut), the inner CV must have >= 2 usable blocks, a block
    (training-season year-month) being usable when training days remain outside it (1-day buffer, cut applied),
    and the usable blocks must span >= 2 distinct calendar months (D185: a training set of one calendar month, e.g.
    2023-05 + 2025-05 in fold 2024 of p4d_smoke 793004743201, gives no seasonal contrast for early stopping)."""
    from src.gold import build_table_v2 as BT2
    days = BT2.build_days(cfg)
    buf = int(cfg["inner_cv"]["buffer_days"])
    rows = []
    for s in seasons(cfg):
        tr_days = [d for d in days if d.year != s and BT2.is_training_month(cfg, d.year, d.month)]
        blocks = sorted({(d.year, d.month) for d in tr_days})
        for var, use_cut in VARIANTS.items():
            ub = [(y, m) for y, m in blocks if _train_mask_days(tr_days, (y, m), buf, use_cut, cfg)]
            cal = sorted({m for _, m in ub})
            rows.append({"heldout_season": s, "variant": var, "inner_blocks": [f"{y}-{m:02d}" for y, m in blocks],
                         "usable_blocks": [f"{y}-{m:02d}" for y, m in ub], "calendar_months": cal,
                         "ok": len(ub) >= 2 and len(cal) >= 2})
    return rows


def inner_cv_label_check(cfg: dict, lab: pd.DataFrame) -> list[dict]:
    """Data check on the Gold labels (gate g2, before any fit): for every fold x variant x frame x threshold, the
    inner CV must have >= 2 usable blocks, i.e. held-out month with a positive city row AND remaining training rows
    (buffer, cut) with a positive city row. lab: city rows with season, t0, lead_min, train_month, y_* columns.
    Catches dry smoke blocks (p4d_smoke 793004743201: 2023-05-13..18 has no city window >= 2.5 mm/h)."""
    buf = pd.Timedelta(days=int(cfg["inner_cv"]["buffer_days"]))
    t = lab["t0"].dt.tz_convert("UTC")
    ym = (t.dt.year * 100 + t.dt.month).to_numpy()
    md = (t.dt.month * 100 + t.dt.day).to_numpy()
    rows = []
    for s in seasons(cfg):
        trainable = (lab["season"].to_numpy() != s) & lab["train_month"].to_numpy(bool)
        for var, use_cut in VARIANTS.items():
            keep = trainable & (~cut_mask(md, cfg) if use_cut else True)
            for L in leads(cfg):
                lm = lab["lead_min"].to_numpy() == L
                for thr in thresholds(cfg):
                    y = lab[thr_col(thr)].to_numpy() > 0
                    usable = []
                    for mo in sorted(set(ym[trainable & lm].tolist())):
                        a, b = _month_bounds(mo)
                        ho = trainable & lm & (ym == mo)
                        tr = keep & lm & (ym != mo) & ((t < a - buf) | (t >= b + buf)).to_numpy()
                        if y[ho].any() and y[tr].any():
                            usable.append(int(mo))
                    rows.append({"heldout_season": s, "variant": var, "lead_min": L, "threshold": thr,
                                 "usable_blocks": usable, "ok": len(usable) >= 2})
    return rows


# ============================================================================ data access
META = ["pixel_id", "t0", "season", "is_city", "label_rate_mm_h"]


def all_features(ctx: Ctx) -> list[str]:
    from src.gold import build_table_v2 as BT2
    parts = gold_parts(ctx)
    if not parts:
        raise RuntimeError(f"no {TABLE} partition for gold id {ctx.gid}")
    return BT2.feature_columns(PB._schema_names(parts[0][2]))


def feature_list(ctx: Ctx, ring: bool) -> list[str]:
    """Set d of v1 (everything but cal_month and the static ring columns); + ring_model_extra for ring models."""
    from src.gold import build_table_v2 as BT2
    allc = all_features(ctx)
    d = [c for c in allc if c != "cal_month" and c not in BT2.STATIC]
    return d + ([c for c in ctx.cfg["features"]["ring_model_extra"] if c in allc] if ring else [])


def load_frame(ctx: Ctx, lead: int, feats: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """All table_v2 rows of one frame: metadata frame + float32 feature matrix (same row order)."""
    from src.gold import build_table_v2 as BT2
    cfg = ctx.cfg
    ycols = [thr_col(t) for t in thresholds(cfg)]
    parts = gold_parts(ctx)
    counts = [int((pd.read_parquet(part, columns=["lead_min"])["lead_min"] == lead).sum()) for _, _, part in parts]
    X = np.empty((sum(counts), len(feats)), "float32")        # filled in place: no second full copy (RAM)
    metas, at = [], 0
    for (y, m, part), n in zip(parts, counts):
        df = pd.read_parquet(part, columns=list(dict.fromkeys(META + ycols + feats)),
                             filters=[("lead_min", "==", lead)])
        assert len(df) == n, (part, len(df), n)
        meta = df[META + ycols].copy()
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


# ============================================================================ stage 3: evaluation base rows
def run_base(unit, ctx: Ctx) -> tuple[list[str], dict]:
    from src.gold import build_table_v2 as BT2
    cfg, s, L = ctx.cfg, unit.params["season"], unit.params["lead"]
    ctx.fetch_gold()
    ycols = [thr_col(t) for t in thresholds(cfg)]
    parts = []
    for y, m, part in gold_parts(ctx):
        df = pd.read_parquet(part, columns=list(dict.fromkeys(PB.KEYS + ["is_city", "label_rate_mm_h"] + ycols + GK)),
                             filters=[("lead_min", "==", L)])
        df = df[df["is_city"] == 1]
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
                        "day": utc.dt.tz_convert(LOCAL_TZ).dt.date.astype(str).to_numpy(),
                        "half": [f"{a:02d}-{'1' if b <= 15 else '2'}" for a, b in zip(utc.dt.month, utc.dt.day)],
                        "role": np.where(held["train_month"], "eval", "secondary"),
                        "label_source": held["label_source"].to_numpy(),
                        "label_rate_mm_h": held["label_rate_mm_h"].to_numpy("float32"),
                        "pers": PB.persistence(held, L).astype("float32")})
    for t in thresholds(cfg):
        out[thr_col(t)] = held[thr_col(t)].to_numpy("int8")
        out[f"pclim_{kstr(t)}"] = PB.clim_for(train, held, thr_col(t)).astype("float32")
    p = out_dir("base") / f"s{s}_lead{L}.parquet"
    PB.write_parquet(out, p)
    return [rel(p)], {"rows": len(out), "rows_eval": int((out["role"] == "eval").sum()),
                      "rows_secondary": int((out["role"] == "secondary").sum()), "train_rows_clim": len(train)}


def base_rows(ctx: Ctx, s: int, L: int) -> pd.DataFrame:
    if (s, L) not in ctx._base:
        r = f"{OUT}/base/s{s}_lead{L}.parquet"
        if not (root() / r).is_file():
            ctx.fetch_run_files(r)
        b = pd.read_parquet(root() / r)
        b["t0"] = b["t0"].astype("datetime64[ns, UTC]")
        ctx._base[(s, L)] = b
    return ctx._base[(s, L)]


# ============================================================================ fit units
def _read_sel(ctx: Ctx, name: str) -> dict:
    return ctx.read_json(f"{OUT}/select/{name}.json")


def dataset_def(ctx: Ctx, spec: str) -> dict:
    """Training data of a spec: ring pixels or not, Model 1 cut or not, Model 2 weights or none."""
    if spec == "m0":
        return {"ring": False, "cut": False, "weights": None}
    if spec == "m1a":
        return {"ring": False, "cut": True, "weights": None}
    if spec == "m1b":
        return {"ring": True, "cut": bool(_read_sel(ctx, "m1a")["kept"]), "weights": None}
    s1 = _read_sel(ctx, "m1b")
    if spec == "m1":
        if s1["chain_end"] == "m0":
            return {"alias": "m0"}
        return {"ring": bool(s1["ring"]), "cut": bool(s1["cut"]), "weights": None, "chain_end": s1["chain_end"]}
    if spec.startswith("m2-g"):
        n = int(spec[4:])
        return {"ring": bool(s1["ring"]), "cut": False, "weights": grid_settings(ctx.cfg)[n - 1], "setting": f"g{n}"}
    if spec == "m2":
        ch = _read_sel(ctx, "m2")["chosen"]
        return {"ring": bool(s1["ring"]), "cut": False, "weights": grid_settings(ctx.cfg)[int(ch[1:]) - 1],
                "setting": ch}
    raise ValueError(spec)


def _month_bounds(ym: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    a = pd.Timestamp(year=ym // 100, month=ym % 100, day=1, tz="UTC")
    return a, a + pd.offsets.MonthBegin(1)


def run_fit(unit, ctx: Ctx) -> tuple[list[str], dict]:
    import lightgbm as lgb
    from src.models import metrics as M
    cfg = ctx.cfg
    spec, s, L, thr = unit.params["spec"], unit.params["season"], unit.params["lead"], unit.params["thr"]
    dd = dataset_def(ctx, spec)
    info_p = out_dir("fit") / f"{unit.id}.json"
    if dd.get("alias"):
        write_json(info_p, {"unit": unit.id, "alias": fit_id(dd["alias"], s, L, thr),
                            "note": "Model 1 chain end is m0: the final Model 1 is M0 at this threshold"})
        return [rel(info_p)], {"alias": dd["alias"]}
    t1 = time.time()
    ctx.fetch_gold()
    feats = feature_list(ctx, dd["ring"])
    meta, X = ctx.cache.get((L, tuple(feats)), lambda: load_frame(ctx, L, feats))
    yc = thr_col(thr)
    y = meta[yc].to_numpy("int8")
    city = (meta["is_city"].to_numpy("float32") == 1)
    md = meta["md"].to_numpy()
    t0 = meta["t0"]
    trainable = (meta["season"].to_numpy() != s) & meta["train_month"].to_numpy(bool)
    keep = trainable & (city | dd["ring"])
    if dd["cut"]:
        keep &= ~cut_mask(md, cfg)
    held = (meta["season"].to_numpy() == s) & city
    w = sample_weights(md, meta["label_rate_mm_h"].to_numpy(), dd["weights"], cfg) if dd["weights"] else None
    params = lgb_params(cfg)
    ic = cfg["inner_cv"]
    buf = pd.Timedelta(days=int(ic["buffer_days"]))
    ym = meta["ym"].to_numpy()
    months = sorted(set(ym[trainable & city].tolist()))
    boosters, best, folds = [], [], []
    for mo in months:
        a, b = _month_bounds(mo)
        ho = trainable & city & (ym == mo)
        tr = keep & (ym != mo) & ((t0 < a - buf) | (t0 >= b + buf)).to_numpy()
        if ho.sum() == 0 or tr.sum() == 0 or y[tr].sum() == 0 or y[ho].sum() == 0:
            folds.append({"month": int(mo), "train_rows": int(tr.sum()), "held_rows": int(ho.sum()), "used": False})
            continue
        dtr = lgb.Dataset(X[tr], y[tr], weight=None if w is None else w[tr], feature_name=feats, free_raw_data=True)
        dva = lgb.Dataset(X[ho], y[ho], reference=dtr)                     # evaluation rows: never weighted
        bst = lgb.train(params, dtr, num_boost_round=int(ic["max_rounds"]), valid_sets=[dva], valid_names=["held"],
                        callbacks=[lgb.early_stopping(int(ic["early_stopping_rounds"]), verbose=False)])
        bi = int(bst.best_iteration or bst.current_iteration())
        boosters.append((mo, ho, bst)); best.append(bi)
        folds.append({"month": int(mo), "train_rows": int(tr.sum()), "held_rows": int(ho.sum()),
                      "train_pos": int(y[tr].sum()), "best_iteration": bi, "used": True})
        del dtr, dva
    if not boosters:
        raise RuntimeError(f"{unit.id}: no usable inner fold")
    med = max(int(np.median(best)), 1)
    iy, ip = [], []
    for mo, ho, bst in boosters:
        ip.append(bst.predict(X[ho], num_iteration=max(min(med, bst.current_iteration()), 1)))
        iy.append(y[ho])
    del boosters
    iy, ip = np.concatenate(iy).astype(bool), np.concatenate(ip)
    platt = PB.fit_calibrator("platt", ip, iy)
    ipc = PB.apply_calibrator(platt, ip)
    rule = cfg["evaluation"]["cut_rules"][f"{thr:g}"]
    cut = M.choose_cut(iy, ipc, rule)
    dfull = lgb.Dataset(X[keep], y[keep], weight=None if w is None else w[keep], feature_name=feats)
    bst = lgb.train(params, dfull, num_boost_round=med)
    p_out = bst.predict(X[held])
    oof = pd.DataFrame({"pixel_id": meta["pixel_id"].to_numpy("int64")[held],
                        "t0": meta.loc[held, "t0"].reset_index(drop=True).astype("datetime64[us, UTC]"),
                        "p_raw": p_out.astype("float32")})
    po = out_dir("oof") / f"{unit.id}.parquet"
    PB.write_parquet(oof, po)
    from src.gold import p4a_common as P
    rss = P.rss_gb()
    info = {"unit": unit.id, "spec": spec, "dataset": dd, "heldout_season": s, "lead_min": L, "threshold": thr,
            "labels": "training: IMERG Final V07 rows only; held-out: IMERG Final (Late for 2025-10, role secondary)",
            "features": feats, "n_features": len(feats), "lgb_params": params, "inner_folds": folds,
            "inner_fold_best_iterations": best, "median_rounds": med, "platt": platt, "cut": cut,
            "inner_oof": {"n": int(len(iy)), "base_rate": float(iy.mean()), "logloss_raw": PB.logloss(iy, ip),
                          "logloss_cal": PB.logloss(iy, ipc), "brier_raw": M.brier(iy, ip),
                          "brier_cal": M.brier(iy, ipc)},
            "train_rows": int(keep.sum()), "train_rows_ring": int((keep & ~city).sum()),
            "train_pos": int(y[keep].sum()), "train_weight_sum": float(w[keep].sum()) if w is not None else None,
            "held_rows": int(held.sum()), "note": NOT_FLOOD}
    write_json(info_p, info)
    if np.isfinite(rss) and rss > float(cfg["resources"]["max_rss_gb"]):
        raise RuntimeError(f"{unit.id}: RSS {rss:.2f} GB above resources.max_rss_gb")
    return [rel(po), rel(info_p)], {"median_rounds": med, "train_rows": int(keep.sum()), "rss_gb": round(rss, 2),
                                    "inner_logloss_cal": info["inner_oof"]["logloss_cal"],
                                    "seconds": round(time.time() - t1, 1)}


# ============================================================================ pooled OOF access (gates)
def fit_info(ctx: Ctx, uid: str) -> dict:
    return ctx.read_json(f"{OUT}/fit/{uid}.json")


def fold_pred(ctx: Ctx, spec: str, s: int, L: int, thr: float) -> tuple[pd.DataFrame, dict]:
    uid = fit_id(spec, s, L, thr)
    info = fit_info(ctx, uid)
    if info.get("alias"):
        uid = info["alias"]
        info = fit_info(ctx, uid)
    r = f"{OUT}/oof/{uid}.parquet"
    if not (root() / r).is_file():
        ctx.fetch_run_files(r)
    o = pd.read_parquet(root() / r)
    o["t0"] = o["t0"].astype("datetime64[ns, UTC]")
    return o, info


def pooled(ctx: Ctx, spec: str, L: int, thr: float, roles=("eval",), nested_spec: str | None = None) -> pd.DataFrame:
    """Base rows of the held-out seasons joined with the fold predictions of `spec`: y, p_raw, p_cal (fold Platt),
    yhat at the fold's inner cut; 7.6 mm/h also needs the 2.5 mm/h alarm of `nested_spec` (nested rule)."""
    out = []
    for s in seasons(ctx.cfg):
        b = base_rows(ctx, s, L)
        b = b[b["role"].isin(roles)]
        o, info = fold_pred(ctx, spec, s, L, thr)
        j = b.merge(o, on=["pixel_id", "t0"], how="inner", validate="one_to_one")
        if len(j) != len(b):
            raise RuntimeError(f"{spec} s{s} lead{L} thr{thr:g}: {len(b) - len(j)} base rows without prediction")
        j["y"] = j[thr_col(thr)].astype(bool)
        j["p_clim"] = j[f"pclim_{kstr(thr)}"].astype("float64")
        j["p_raw"] = j["p_raw"].astype("float64")
        j["p_cal"] = PB.apply_calibrator(info["platt"], j["p_raw"].to_numpy())
        j["yhat"] = j["p_cal"].to_numpy() >= info["cut"]["cut"]
        j["season_fold"] = s
        if thr == heavy(ctx.cfg) and nested_spec and ctx.cfg["evaluation"]["heavy_rule"] == "nested":
            o2, i2 = fold_pred(ctx, nested_spec, s, L, primary(ctx.cfg))
            q = j[["pixel_id", "t0"]].merge(o2, on=["pixel_id", "t0"], how="left")
            pq = PB.apply_calibrator(i2["platt"], q["p_raw"].to_numpy("float64"))
            j["yhat"] = j["yhat"].to_numpy() & (np.nan_to_num(pq, nan=-1.0) >= i2["cut"]["cut"])
        out.append(j)
    return pd.concat(out, ignore_index=True)


def ece(y, p, bins: int) -> float:
    from src.models import metrics as M
    r = M.reliability(y, p, bins)
    r = r[r["n"] > 0]
    return float((r["n"] * (r["mean_p"] - r["obs_freq"]).abs()).sum() / max(r["n"].sum(), 1))


def metric_row(j: pd.DataFrame, thr: float, bins: int) -> dict:
    from src.models import metrics as M
    y, pr, pc, pcl = j["y"].to_numpy(bool), j["p_raw"].to_numpy(), j["p_cal"].to_numpy(), j["p_clim"].to_numpy()
    sc = M.scores(**M.contingency(y, j["yhat"].to_numpy(bool)))
    yp = np.nan_to_num(j["pers"].to_numpy("float64"), nan=-1.0) >= thr
    mx = M.choose_cut(y, pc, "max_csi") if y.any() else {"csi": np.nan}
    return {"n": int(len(y)), "base_rate": float(y.mean()) if len(y) else np.nan,
            "logloss_raw": PB.logloss(y, pr), "logloss_cal": PB.logloss(y, pc),
            "brier_raw": M.brier(y, pr), "brier_cal": M.brier(y, pc), "bss_raw": M.bss(y, pr, pcl),
            "bss_cal": M.bss(y, pc, pcl), "auc_pr": M.auc_pr(y, pc), "ece_cal": ece(y, pc, bins),
            "ece_raw": ece(y, pr, bins), "csi": float(sc["csi"]), "pod": float(sc["pod"]), "far": float(sc["far"]),
            "bias": float(sc["bias"]), "csi_maxcut_insample": float(mx["csi"]),
            "csi_pers": float(M.scores(**M.contingency(y, yp))["csi"])}


def ll_day(j: pd.DataFrame, col: str = "p_cal") -> pd.DataFrame:
    y = j["y"].to_numpy("float64")
    p = np.clip(j[col].to_numpy("float64"), 1e-7, 1 - 1e-7)
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return pd.DataFrame({"day": j["day"].to_numpy(), "ll": ll, "n": 1.0}).groupby("day")[["ll", "n"]].sum()


def ll_paired(da: pd.DataFrame, db: pd.DataFrame, n: int, seed: int, level: float) -> dict:
    """Day-block bootstrap of the mean log-loss difference A - B (same days; per-day sums of several frames may be
    added before the call)."""
    from src.models import metrics as M
    days = da.index.intersection(db.index)
    A, B = da.loc[days], db.loc[days]
    W = M._resample_weights(len(days), n, seed)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (W @ A["ll"].to_numpy()) / (W @ A["n"].to_numpy()) - (W @ B["ll"].to_numpy()) / (W @ B["n"].to_numpy())
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    point = A["ll"].sum() / A["n"].sum() - B["ll"].sum() / B["n"].sum()
    return {"diff": float(point), "lo": float(np.nanquantile(d, lo)), "hi": float(np.nanquantile(d, hi)),
            "days": int(len(days))}


def _sum_days(frames: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(frames).groupby(level=0).sum()


def compare_steps(ctx: Ctx, new: str, old: str, thr: float) -> tuple[list[dict], dict, bool]:
    """Chain rule: per-frame metrics of both specs at `thr`, per-frame and frame-pooled bootstrap of the recalibrated
    log-loss difference new - old; kept = mean over frames of logloss_cal strictly lower."""
    ev = ctx.cfg["evaluation"]
    rows, dn, do = [], [], []
    for L in leads(ctx.cfg):
        jn, jo = pooled(ctx, new, L, thr), pooled(ctx, old, L, thr)
        mn, mo = metric_row(jn, thr, ev["reliability_bins"]), metric_row(jo, thr, ev["reliability_bins"])
        a, b = ll_day(jn), ll_day(jo)
        dn.append(a); do.append(b)
        d = ll_paired(a, b, ev["bootstrap_resamples"], ctx.cfg["random_seed"], ev["ci_level"])
        rows.append({"lead_min": L, "spec": new, **mn})
        rows.append({"lead_min": L, "spec": old, **mo})
        rows.append({"lead_min": L, "spec": f"{new} - {old}", "logloss_cal": d["diff"], "ci_lo": d["lo"],
                     "ci_hi": d["hi"], "brier_cal": mn["brier_cal"] - mo["brier_cal"], "csi": mn["csi"] - mo["csi"]})
    allf = ll_paired(_sum_days(dn), _sum_days(do), ev["bootstrap_resamples"], ctx.cfg["random_seed"], ev["ci_level"])
    mean_new = float(np.mean([r["logloss_cal"] for r in rows if r["spec"] == new]))
    mean_old = float(np.mean([r["logloss_cal"] for r in rows if r["spec"] == old]))
    return rows, {"mean_logloss_cal_new": mean_new, "mean_logloss_cal_old": mean_old, "frames_pooled_diff": allf}, \
        bool(mean_new < mean_old)


def gate_sel_m1a(ctx: Ctx) -> tuple[list[str], dict]:
    pt = primary(ctx.cfg)
    rows, summ, kept = compare_steps(ctx, "m1a", "m0", pt)
    out = out_dir("select")
    pd.DataFrame(rows).to_csv(out / "m1a.csv", index=False)
    obj = {"step": "m1a (cut)", "previous": "m0", "kept": kept, **summ, "rule": ctx.cfg["model1"]["keep_rule"],
           "threshold": pt, "evaluation": "pooled LOSO OOF, city pixels, full May-Oct, IMERG Final rows only"}
    write_json(out / "m1a.json", obj)
    return [rel(out / "m1a.csv"), rel(out / "m1a.json")], {"kept": kept, "mean_logloss_m1a": summ["mean_logloss_cal_new"],
                                                         "mean_logloss_m0": summ["mean_logloss_cal_old"]}


def gate_sel_m1b(ctx: Ctx) -> tuple[list[str], dict]:
    pt = primary(ctx.cfg)
    m1a_kept = bool(_read_sel(ctx, "m1a")["kept"])
    prev = "m1a" if m1a_kept else "m0"
    rows, summ, kept = compare_steps(ctx, "m1b", prev, pt)
    end = "m1b" if kept else prev
    model1 = {kstr(t): (end if (t == pt or end == "m0") else "m1") for t in thresholds(ctx.cfg)}
    obj = {"step": "m1b (+ ring)", "previous": prev, "kept": kept, "chain_end": end,
           "ring": end == "m1b", "cut": (end == "m1a") or (end == "m1b" and m1a_kept), "m1a_kept": m1a_kept,
           "model1_spec_by_threshold": model1, **summ, "rule": ctx.cfg["model1"]["keep_rule"], "threshold": pt}
    out = out_dir("select")
    pd.DataFrame(rows).to_csv(out / "m1b.csv", index=False)
    write_json(out / "m1b.json", obj)
    return [rel(out / "m1b.csv"), rel(out / "m1b.json")], {"kept": kept, "chain_end": end, "ring": obj["ring"],
                                                         "cut": obj["cut"]}


def gate_sel_m2(ctx: Ctx) -> tuple[list[str], dict]:
    cfg, ev = ctx.cfg, ctx.cfg["evaluation"]
    pt, hv = primary(cfg), heavy(cfg)
    gthr = [float(t) for t in cfg["model2"]["grid_thresholds"] if float(t) in thresholds(cfg)]
    rows = []
    for n, st in enumerate(grid_settings(cfg), start=1):
        spec = f"m2-g{n}"
        r = {"setting": f"g{n}", **st}
        for L in leads(cfg):
            m = metric_row(pooled(ctx, spec, L, pt), pt, ev["reliability_bins"])
            r |= {f"logloss_cal_F{L}": m["logloss_cal"], f"logloss_raw_F{L}": m["logloss_raw"],
                  f"brier_cal_F{L}": m["brier_cal"], f"csi_2p5_F{L}": m["csi"]}
            if hv in gthr:
                h = metric_row(pooled(ctx, spec, L, hv, nested_spec=spec), hv, ev["reliability_bins"])
                r |= {f"csi_7p6_F{L}": h["csi"], f"pod_7p6_F{L}": h["pod"], f"far_7p6_F{L}": h["far"],
                      f"bias_7p6_F{L}": h["bias"], f"brier_cal_7p6_F{L}": h["brier_cal"]}
        r["mean_logloss_cal"] = float(np.mean([r[f"logloss_cal_F{L}"] for L in leads(cfg)]))
        rows.append(r)
    best = min(rows, key=lambda x: x["mean_logloss_cal"])
    model2 = {kstr(t): (f"m2-{best['setting']}" if t in gthr else "m2") for t in thresholds(cfg)}
    out = out_dir("select")
    pd.DataFrame(rows).to_csv(out / "m2_grid.csv", index=False)
    write_json(out / "m2.json", {"chosen": best["setting"], "weights": {k: best[k] for k in ("k", "alpha", "beta")},
                                 "model2_spec_by_threshold": model2, "table": rows, "rule": cfg["model2"]["choose"],
                                 "dataset": dataset_def(ctx, "m2-g1") | {"weights": "per setting"}})
    return [rel(out / "m2_grid.csv"), rel(out / "m2.json")], {"chosen": best["setting"],
                                                            "mean_logloss": best["mean_logloss_cal"]}


# ============================================================================ comparison (stage 7-8)
def models_map(ctx: Ctx) -> dict:
    thrs = thresholds(ctx.cfg)
    return {"M0": {kstr(t): "m0" for t in thrs}, "Model1": _read_sel(ctx, "m1b")["model1_spec_by_threshold"],
            "Model2": _read_sel(ctx, "m2")["model2_spec_by_threshold"]}


def model_frame(ctx: Ctx, mm: dict, name: str, L: int, thr: float, roles) -> pd.DataFrame:
    spec = mm[name][kstr(thr)]
    return pooled(ctx, spec, L, thr, roles=roles, nested_spec=mm[name].get(kstr(primary(ctx.cfg))))


def _diff_rows(ctx: Ctx, frames: dict, L: int, thr: float, period: str) -> list[dict]:
    from src.models import metrics as M
    ev, seed = ctx.cfg["evaluation"], ctx.cfg["random_seed"]
    out = []
    for a, b in (("Model1", "M0"), ("Model2", "M0"), ("Model2", "Model1")):
        ja, jb = frames[a], frames[b]
        da = M.day_stats(ja["day"].to_numpy(), ja["y"], ja["yhat"], ja["p_cal"], ja["p_clim"])
        db = M.day_stats(jb["day"].to_numpy(), jb["y"], jb["yhat"], jb["p_cal"], jb["p_clim"])
        d = M.paired_bootstrap(da, db, ev["bootstrap_resamples"], seed, ev["ci_level"])
        ll = ll_paired(ll_day(ja), ll_day(jb), ev["bootstrap_resamples"], seed, ev["ci_level"])
        out.append({"period": period, "lead_min": L, "threshold": thr, "comparison": f"{a} - {b}",
                    "logloss_cal_diff": ll["diff"], "logloss_lo": ll["lo"], "logloss_hi": ll["hi"],
                    **{f"{k}_{x}": d[k][x] for k in ("csi", "brier", "bss_clim", "pod") for x in ("diff", "lo", "hi")}})
    return out


def gate_cmp_main(ctx: Ctx) -> tuple[list[str], dict]:
    from src.models import metrics as M
    cfg, ev = ctx.cfg, ctx.cfg["evaluation"]
    mm = models_map(ctx)
    rows, diffs, rel_rows, byfold, sec_rows, sec_diffs = [], [], [], [], [], []
    for L in leads(cfg):
        for t in thresholds(cfg):
            for period, roles, rr, dr in (("eval", ("eval",), rows, diffs), ("secondary", ("secondary",), sec_rows,
                                                                            sec_diffs)):
                frames = {nm: model_frame(ctx, mm, nm, L, t, roles) for nm in mm}
                if not len(frames["M0"]):
                    continue
                for nm, j in frames.items():
                    ds = M.day_stats(j["day"].to_numpy(), j["y"], j["yhat"], j["p_cal"], j["p_clim"])
                    ci = M.bootstrap_ci(ds, ev["bootstrap_resamples"], cfg["random_seed"], ev["ci_level"])
                    rr.append({"period": period, "lead_min": L, "threshold": t, "model": nm,
                               "spec": mm[nm][kstr(t)], **metric_row(j, t, ev["reliability_bins"]),
                               "csi_lo": ci["csi"][0], "csi_hi": ci["csi"][1], "bss_lo": ci["bss_clim"][0],
                               "bss_hi": ci["bss_clim"][1]})
                    if period == "eval":
                        rel_rows.append(M.reliability(j["y"], j["p_cal"], ev["reliability_bins"]).assign(
                            lead_min=L, threshold=t, model=nm, prob="cal"))
                        for s, g in j.groupby("season_fold"):
                            byfold.append({"heldout_season": s, "lead_min": L, "threshold": t, "model": nm,
                                           **metric_row(g, t, ev["reliability_bins"])})
                j0 = frames["M0"]                                            # reference rows: persistence, climatology
                yp = np.nan_to_num(j0["pers"].to_numpy("float64"), nan=-1.0) >= t
                sp = M.scores(**M.contingency(j0["y"], yp))
                rr.append({"period": period, "lead_min": L, "threshold": t, "model": "persistence", "n": len(j0),
                           "csi": float(sp["csi"]), "pod": float(sp["pod"]), "far": float(sp["far"]),
                           "bias": float(sp["bias"]), "brier_cal": M.brier(j0["y"], yp.astype(float))})
                dr += _diff_rows(ctx, frames, L, t, period)
    out = out_dir("compare")
    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(diffs).to_csv(out / "diffs.csv", index=False)
    pd.DataFrame(byfold).to_csv(out / "by_fold.csv", index=False)
    if rel_rows:
        pd.concat(rel_rows, ignore_index=True).to_csv(out / "reliability.csv", index=False)
    pd.DataFrame(sec_rows).to_csv(out / "secondary_metrics.csv", index=False)
    pd.DataFrame(sec_diffs).to_csv(out / "secondary_diffs.csv", index=False)
    write_json(out / "models.json", {"models": mm, "note": NOT_FLOOD,
                                     "secondary": "2025-10, IMERG Late labels (provisional), never used for a choice"})
    pt = primary(cfg)
    head = [{"lead": r["lead_min"], "model": r["model"], "logloss_cal": round(r["logloss_cal"], 5),
             "bss_cal": round(r["bss_cal"], 4), "csi": round(r["csi"], 4)}
            for r in rows if r["threshold"] == pt and r["model"] in mm]
    return [rel(p) for p in sorted(out.glob("*")) if p.is_file() and p.name in (
        "metrics.csv", "diffs.csv", "by_fold.csv", "reliability.csv", "secondary_metrics.csv", "secondary_diffs.csv",
        "models.json")], {"primary": head}


def gate_cmp_detail(ctx: Ctx) -> tuple[list[str], dict]:
    from src.models import metrics as M
    cfg, ev = ctx.cfg, ctx.cfg["evaluation"]
    mm = models_map(ctx)
    pt = primary(cfg)
    half, wet, events = [], [], []
    ev_days = {k: (pd.Timestamp(a).date(), pd.Timestamp(b).date()) for k, (a, b) in ev["events"].items()}

    def _cat(j, yhat, p):
        sc = M.scores(**M.contingency(j["y"], yhat))
        return {"n": int(len(j)), "base_rate": float(j["y"].mean()) if len(j) else np.nan, "csi": float(sc["csi"]),
                "pod": float(sc["pod"]), "far": float(sc["far"]), "bias": float(sc["bias"]),
                "brier": M.brier(j["y"], p) if len(j) else np.nan}
    for L in leads(cfg):
        for t in thresholds(cfg):
            frames = {nm: model_frame(ctx, mm, nm, L, t, ("eval", "secondary")) for nm in mm}
            j0 = frames["M0"]
            pers_yh = np.nan_to_num(j0["pers"].to_numpy("float64"), nan=-1.0) >= t
            for nm, j in list(frames.items()) + [("persistence", j0)]:
                yh = pers_yh if nm == "persistence" else j["yhat"].to_numpy(bool)
                p = pers_yh.astype(float) if nm == "persistence" else j["p_cal"].to_numpy()
                for (role, h), idx in j.groupby(["role", "half"]).indices.items():
                    g = j.iloc[idx]
                    r = {"role": role, "half_month": h, "lead_min": L, "threshold": t, "model": nm,
                         **_cat(g, yh[idx], p[idx])}
                    if nm != "persistence":
                        r["logloss_cal"] = PB.logloss(g["y"], p[idx])
                    half.append(r)
                dd = pd.to_datetime(j["day"]).dt.date
                for ename, (a, b) in ev_days.items():
                    for day in sorted(set(dd[(dd >= a) & (dd <= b)])):
                        sel = (dd == day).to_numpy()
                        events.append({"event": ename, "day": str(day), "role": str(j["role"][sel].iloc[0]),
                                       "label_source": str(j["label_source"][sel].iloc[0]), "lead_min": L,
                                       "threshold": t, "model": nm, **_cat(j[sel], yh[sel], p[sel])})
                if t == pt:
                    for s in seasons(cfg):
                        e = (j["season_fold"] == s).to_numpy() & (j["role"] == "eval").to_numpy()
                        lab = base_rows(ctx, s, leads(cfg)[0])
                        lab = lab[lab["role"] == "eval"].groupby("day")["label_rate_mm_h"].mean()
                        top = lab.sort_values(ascending=False).index[:int(ev["wettest_days"])]
                        for day in top:
                            sel = e & (j["day"] == day).to_numpy()
                            wet.append({"heldout_season": s, "day": day, "city_mean_label_mm_h_F0": float(lab[day]),
                                        "lead_min": L, "threshold": t, "model": nm, **_cat(j[sel], yh[sel], p[sel])})
    out = out_dir("compare")
    pd.DataFrame(half).to_csv(out / "halfmonth.csv", index=False)
    pd.DataFrame(wet).to_csv(out / "wettest_days.csv", index=False)
    pd.DataFrame(events).to_csv(out / "events.csv", index=False)
    return [rel(out / f) for f in ("halfmonth.csv", "wettest_days.csv", "events.csv")], \
        {"halfmonth_rows": len(half), "wettest_rows": len(wet), "event_rows": len(events)}


def gate_compare_done(ctx: Ctx) -> tuple[list[str], dict]:
    cfg = ctx.cfg
    s1, s2 = _read_sel(ctx, "m1b"), _read_sel(ctx, "m2")
    m1a = _read_sel(ctx, "m1a")
    met = ctx.read_json(f"{OUT}/compare/models.json")
    r = f"{OUT}/compare/diffs.csv"
    if not (root() / r).is_file():
        ctx.fetch_run_files(r)
    d = pd.read_csv(root() / r)
    d = d[d["threshold"] == primary(cfg)]
    summ = {"job": ctx.job, "run_id": ctx.run_id, "gold_id": ctx.gid, "smoke": bool(cfg.get("smoke")),
            "stage": "P4D.7 comparison complete (stop point: the user chooses after the report)",
            "model1": {"chain_end": s1["chain_end"], "m1a_kept": m1a["kept"], "m1b_kept": s1["kept"],
                       "cut": s1["cut"], "ring": s1["ring"]},
            "model2": {"chosen": s2["chosen"], "weights": s2["weights"]}, "models": met["models"],
            "primary_diffs": d[["lead_min", "comparison", "logloss_cal_diff", "logloss_lo", "logloss_hi", "csi_diff",
                                "csi_lo", "csi_hi", "brier_diff", "brier_lo", "brier_hi"]].to_dict("records"),
            "note": NOT_FLOOD}
    p = out_dir() / "compare_summary.json"
    write_json(p, summ)
    return [rel(p)], {"compare_complete": True, "model1_chain_end": s1["chain_end"], "model2_chosen": s2["chosen"]}


# ============================================================================ dispatch
GATES = {"preflight": gate_preflight, "g2": gate_g2, "sel-m1a": gate_sel_m1a, "sel-m1b": gate_sel_m1b,
         "sel-m2": gate_sel_m2, "cmp-main": gate_cmp_main, "cmp-detail": gate_cmp_detail,
         "compare-done": gate_compare_done}


def run_unit(unit, ctx: Ctx) -> tuple[list[str], dict]:
    if unit.kind == "feat":
        return run_feat(unit, ctx)
    if unit.kind == "base":
        return run_base(unit, ctx)
    if unit.kind == "fit":
        return run_fit(unit, ctx)
    if unit.kind == "gate":
        return GATES[unit.id](ctx)
    raise ValueError(unit.kind)


# ============================================================================ planning estimates (no data read)
def _unit_minutes(cfg: dict) -> dict:
    """Estimated runner minutes per unit id (P4B run c0df12477947: CV unit = 12 folds on 830,208 rows in 92-131 s;
    feat unit ~45 s for 47 pixels). Fit time scales with the training rows; + 25 s load / predict / upload."""
    from src.gold import build_table_v2 as BT2
    n_city = 47
    n_ring = cfg["pixels"]["ring"].get("max_pixels") or cfg["pixels"]["ring"]["expected_pixels"]
    npx = n_city + n_ring
    dl = BT2.build_days(cfg)
    md = np.array([d.month * 100 + d.day for d in dl])
    cut_frac = float(cut_mask(md, cfg).mean())
    rows_city = len(dl) * (len(seasons(cfg)) - 1) / len(seasons(cfg)) * 48 * n_city
    sec_per_row = 120.0 / 830208
    gscale = min(1.0, len(dl) / 552) if len(dl) else 1.0
    out = {}
    for u in units(cfg):
        if u.kind == "fit":
            sp = u.params["spec"]
            ring_like = sp in ("m1b", "m1") or sp.startswith("m2")      # worst case: the ring is kept (m1b)
            r = rows_city * (npx / n_city if ring_like else 1.0) * ((1 - cut_frac) if sp == "m1a" else 1.0)
            out[u.id] = (r * sec_per_row * (1.2 if sp.startswith("m2") else 1.0) + 25) / 60
        elif u.kind == "feat":
            out[u.id] = (45 * npx / n_city) * (len(dl) / 18 / 30.7) / 60 + 0.3
        elif u.kind == "base":
            out[u.id] = 0.7
        else:
            out[u.id] = max(0.5, {"preflight": 1, "g2": 6 * npx / 80, "sel-m1a": 3, "sel-m1b": 3, "sel-m2": 8,
                                  "cmp-main": 18, "cmp-detail": 7, "compare-done": 1}[u.id] * gscale)
    return out


def part_wall_minutes(cfg: dict, n: int) -> float:
    """Wall minutes of the slowest part when n parts run together: per stage the largest round-robin share of work
    units, plus the gates (run by one part while the others wait)."""
    um = _unit_minutes(cfg)
    us = units(cfg)
    wall = 0.0
    for st in sorted({u.stage for u in us}):
        work = [u for u in us if u.stage == st and u.kind != "gate"]
        if work:
            wall += max(sum(um[u.id] for k, u in enumerate(work) if k % n == i) for i in range(n))
        wall += sum(um[u.id] for u in us if u.stage == st and u.kind == "gate")
    return wall


def choose_parts(cfg: dict, limit: float = 320.0, margin: float = 0.8) -> int:
    """Smallest number of parts whose slowest part is <= margin x limit (20 % reserve for estimate error)."""
    for n in range(1, 41):
        if part_wall_minutes(cfg, n) <= margin * limit:
            return n
    raise ValueError("no part count fits the time limit")


def estimates(cfg: dict) -> dict:
    """Rows, sizes, RAM and runner minutes from the configuration and the P4B run c0df12477947 timings (no data read).
    Used by REPORT_P4D §2.3."""
    from src.gold import build_table_v2 as BT2
    days = len(BT2.build_days(cfg))
    n_city = 47
    n_ring = cfg["pixels"]["ring"].get("max_pixels") or cfg["pixels"]["ring"]["expected_pixels"]
    npx = n_city + n_ring
    per_frame = days * 48 * npx
    rows = per_frame * len(leads(cfg))
    n_feat = 89                                                   # set d (86) + 3 ring columns
    train_days = days * (len(seasons(cfg)) - 1) / len(seasons(cfg))
    largest_train = int(train_days * 48 * npx)                    # fold with the most training days, ring model
    x_all = per_frame * n_feat * 4                                # frame matrix, filled in place (load_frame)
    meta = per_frame * 48                                         # keys, labels, masks (~48 B / row)
    x_fold = largest_train * n_feat * 4                           # X[train] copy of one inner fold or the refit
    binned = largest_train * n_feat                               # LightGBM bins (uint8)
    peak = x_all + meta + 1.5 * x_fold + binned + 0.8e9           # + interpreter, pandas, boosters
    um = _unit_minutes(cfg)
    us = units(cfg)
    by_stage = {}
    for u in us:
        by_stage[u.stage] = by_stage.get(u.stage, 0.0) + um[u.id]
    n = choose_parts(cfg)
    return {"days": days, "pixels": npx, "ring_pixels": n_ring, "rows_per_frame": per_frame, "rows_total": rows,
            "gold_bytes_est": rows * 80, "fit_units": sum(u.kind == "fit" for u in us), "units": len(us),
            "largest_training_rows": largest_train, "peak_ram_gb_est": round(peak / 1e9, 2),
            "runner_minutes_by_stage": {k: round(v, 1) for k, v in sorted(by_stage.items())},
            "runner_minutes_total": round(sum(um.values()), 0), "parts": n,
            "part_wall_minutes": round(part_wall_minutes(cfg, n), 0),
            "part_wall_minutes_by_n": {k: round(part_wall_minutes(cfg, k), 0) for k in (max(1, n - 2), n - 1, n, n + 2)
                                       if k >= 1}}
