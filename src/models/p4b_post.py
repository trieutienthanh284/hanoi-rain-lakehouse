"""P4B post-final job `p4b_post` (plan step 4 of P4B_PROGRESS.md; user decisions D141 (4)-(6), D142).

Reads ONLY the stored outputs of the finished run p4b/c0df12477947 (fixed run id, sha256-checked against that run's
markers) and the Gold v1 label tables of that run (gold id 0803d9d8ed72). It never trains, never re-runs or
re-scores the test (no model is loaded; every prediction comes from the stored files), and writes only under
lake/v1/runs/p4b_post/<run_id>/ (store restriction of src/models/cloud_train.py: it can neither write the source run
nor lake/v1/reports/p4b/OOF_DONE.json).

Units (stage 1, then the report gate):
  labels-test / labels-secondary   preds/<period>_* + label_rate_mm_h from Gold table_A1_v1 by (pixel_id, t0,
                                   lead_min) + label_source per row; checks y == (label_rate >= threshold)
  reliability                      reliability diagrams from final/reliability.csv (test and secondary, 3 x 3)
  replay-test2025_wet{1,2,3}_*     v1 replay of the P4A.8 test-season wettest days from preds/test_*
  replay-yagi_2024-09-06_08        Yagi from the OOF predictions of the held-out fold 2024-09 (cv units named in
                                   models/v1/*/meta.json; yhat = p_raw >= cut, nested rule at 7.6 mm/h)
  replay-matmo_2025-10-06_07       Matmo from preds/secondary_* (IMERG Late, provisional) + per-day metric table
  post-report                      index of every output (sha256) and the source identity

Rain nowcasting against IMERG (satellite estimate), not flood prediction; every panel and table says so.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.models import p4b as PB

SOURCE_JOB = "p4b"
SOURCE_RUN = "c0df12477947"
SOURCE_GOLD = "0803d9d8ed72"
LAKE = PB.LAKE
OUT_REL = "reports/p4b/post"
SRC_DIR = "cloud_in"                 # local copies of the source files (never uploaded)
TZ = PB.LOCAL_TZ
CODE_FILES = ["src/models/p4b_post.py", "src/models/cloud_train.py", "src/models/p4b.py", "src/models/metrics.py",
              "src/models/data.py", "src/gold/features.py", "configs/model_v1.yaml"]
NOT_FLOOD = "Rain nowcasting against IMERG (satellite estimate, not gauges); not flood prediction."
MODELS_FINAL = ("v1", "v1_A", "v1_B")
EVENTS = {
    "test2025_wet1_2025-09-30": {"days": ["2025-09-30"], "source": "test",
                                 "note": "test 2025, wettest test day #1 of P4A.8; out-of-sample (run once, P4B.6)"},
    "test2025_wet2_2025-08-26": {"days": ["2025-08-26"], "source": "test",
                                 "note": "test 2025, wettest test day #2 of P4A.8; out-of-sample (run once, P4B.6)"},
    "test2025_wet3_2025-09-29": {"days": ["2025-09-29"], "source": "test",
                                 "note": "test 2025, wettest test day #3 of P4A.8; out-of-sample (run once, P4B.6)"},
    "yagi_2024-09-06_08": {"days": ["2024-09-06", "2024-09-07", "2024-09-08"], "source": "oof", "fold_month": "2024-09",
                           "note": ("Yagi from the OUT-OF-FOLD predictions of the held-out calendar month 2024-09 "
                                    "(D141 (6)): the fold models never saw 2024-09 (1-day buffer on both sides). The "
                                    "probability cut, the number of boosting rounds and the hyperparameters were chosen "
                                    "on the pooled OOF of all folds, which includes this fold, so the selection is not "
                                    "fully independent of these days. Not the refit model. Set A and set B are shown "
                                    "separately (OOF rows do not tell which NWP inputs were stale).")},
    "matmo_2025-10-06_07": {"days": ["2025-10-06", "2025-10-07"], "source": "secondary",
                            "note": ("Matmo, secondary out-of-sample check with IMERG Late labels (provisional); "
                                     "reported apart, never used for any choice (D93).")},
}


def kstr(t: float) -> str:
    return PB.kstr(t)


# ============================================================================ units
def units() -> list[PB.Unit]:
    s1 = [PB.Unit("labels-test", "post", 1, (), {"period": "test"}),
          PB.Unit("labels-secondary", "post", 1, (), {"period": "secondary"}),
          PB.Unit("reliability", "post", 1, (), {})]
    s1 += [PB.Unit(f"replay-{name}", "post", 1, (), {"event": name}) for name in EVENTS]
    return s1 + [PB.Unit("post-report", "gate", 2, tuple(u.id for u in s1))]


# ============================================================================ source run (read only, sha-checked)
class SourceRun:
    """Files of lake/v1/runs/p4b/c0df12477947/ (read only). Every file must be an output listed in one of the run's
    markers and its bytes must match that sha256; the run must be finished (done.json + final-test marker)."""

    def __init__(self, store):
        self.store = store
        self.pre = f"{LAKE}runs/{SOURCE_JOB}/{SOURCE_RUN}/"
        ls = store.list(self.pre)
        for need in ("done.json", "markers/final-test.done.json"):
            if self.pre + need not in ls:
                raise RuntimeError(f"source run {SOURCE_JOB}/{SOURCE_RUN} is not finished: {need} missing")
        self.sha: dict[str, str] = {}
        for k in sorted(ls):
            if k.startswith(self.pre + "markers/") and k.endswith(".done.json"):
                self.sha.update(json.loads(store.get(k)).get("outputs", {}))

    def local(self, rel: str) -> Path:
        return PB.ROOT / SRC_DIR / SOURCE_JOB / SOURCE_RUN / rel

    def fetch(self, rel: str) -> Path:
        key = self.pre + "files/" + rel
        if key not in self.sha:
            raise KeyError(f"{rel} is not an output of run {SOURCE_JOB}/{SOURCE_RUN}")
        dst = self.local(rel)
        if dst.is_file() and hashlib.sha256(dst.read_bytes()).hexdigest() == self.sha[key]:
            return dst
        data = self.store.get(key)
        if hashlib.sha256(data).hexdigest() != self.sha[key]:
            raise RuntimeError(f"sha256 of {key} differs from the run's marker")
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".part")
        tmp.write_bytes(data)
        tmp.replace(dst)
        return dst

    def json(self, rel: str) -> dict:
        return json.loads(self.fetch(rel).read_text(encoding="utf-8"))


class PostCtx:
    def __init__(self, store, run_id: str, log):
        self.store, self.run_id, self.log = store, run_id, log
        self.cfg = PB.load_cfg()
        self._src = None
        self.gctx = PB.Ctx(self.cfg, store, SOURCE_JOB, SOURCE_RUN, PB.ROOT / "cloud_out" / "p4b_post", log)
        self.gold_id_now = self.gctx.gid
        self.gctx.gid = SOURCE_GOLD                     # the Gold tables of the source run, whatever the code says

    @property
    def src(self) -> SourceRun:
        if self._src is None:
            self._src = SourceRun(self.store)
        return self._src

    def gold_rows(self, lo: pd.Timestamp, hi: pd.Timestamp, cols: list[str], lead: int | None = None) -> pd.DataFrame:
        """table_A1_v1 rows with lo <= t0 < hi (only the months needed are downloaded)."""
        months = pd.period_range(lo.tz_convert(None).to_period("M"),
                                 (hi - pd.Timedelta("1ns")).tz_convert(None).to_period("M"), freq="M")
        for m in months:
            for name in ("part.parquet",):
                k = self.gctx.gold_key("A1", m.year, m.month, name)
                ls = self.gctx.listing(k)
                if k not in ls:
                    raise RuntimeError(f"Gold partition missing: {k}")
                self.gctx._download(k, ls[k], self.gctx.gold_local("A1") / f"season={m.year}" /
                                    f"month={m.month:02d}" / name)
        return PB.load_rows(self.gctx, "A1", [(lo, hi)], cols, lead=lead)


def out_dir(*parts) -> Path:
    p = PB.ROOT / OUT_REL
    for x in parts:
        p = p / x
    p.mkdir(parents=True, exist_ok=True)
    return p


def label_of(cfg: dict, t0: pd.Series) -> np.ndarray:
    """Label source per row from the month of t0 (UTC), as in the Gold build (scope.label_source_by_month)."""
    ym = t0.dt.tz_convert("UTC").dt.strftime("%Y-%m")
    lut = {k: PB.label_source(cfg, int(k[:4]), int(k[5:])) for k in ym.unique()}
    return ym.map(lut).to_numpy()


def local_range(days: list[str]) -> tuple[pd.Timestamp, pd.Timestamp]:
    lo = pd.Timestamp(days[0]).tz_localize(TZ).tz_convert("UTC")
    hi = (pd.Timestamp(days[-1]) + pd.Timedelta("1D")).tz_localize(TZ).tz_convert("UTC")
    return lo, hi


def to_utc(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s)
    return (s.dt.tz_localize("UTC") if s.dt.tz is None else s.dt.tz_convert("UTC")).astype("datetime64[ns, UTC]")


# ============================================================================ labels join
def run_labels(unit: PB.Unit, pc: PostCtx) -> tuple[list[str], dict]:
    cfg, period = pc.cfg, unit.params["period"]
    lo, hi = PB.period(cfg, period)
    out = out_dir("preds_labelled")
    rels, stats = [], []
    for L in [int(x) for x in cfg["frames"]["lead_min"]]:
        g = pc.gold_rows(lo, hi, [], lead=L)[["pixel_id", "t0", "lead_min", "label_rate_mm_h"]]
        g["t0"] = to_utc(g["t0"])
        for t in PB.thresholds(cfg):
            rel = f"reports/p4b/final/preds/{period}_lead{L}_thr{kstr(t)}.parquet"
            pr = pd.read_parquet(pc.src.fetch(rel))
            pr["t0"] = to_utc(pr["t0"])
            j = pr.merge(g, on=["pixel_id", "t0", "lead_min"], how="left", validate="one_to_one")
            rate = j["label_rate_mm_h"].to_numpy("float64")
            fin = np.isfinite(rate)
            mism = int(((rate[fin] >= t) != j["y"].to_numpy()[fin].astype(bool)).sum())
            if mism:
                raise RuntimeError(f"{rel}: {mism} rows where y != (label_rate >= {t}): wrong join")
            j["label_rate_mm_h"] = j["label_rate_mm_h"].astype("float32")
            j["label_source"] = label_of(cfg, j["t0"])
            p = out / f"{period}_lead{L}_thr{kstr(t)}.parquet"
            PB.write_parquet(j, p)
            rels.append(PB.rel(p))
            stats.append({"file": p.name, "rows": len(j), "label_missing": int((~fin).sum()), "y_mismatch": mism,
                          "label_sources": sorted(set(j["label_source"]))})
    PB.write_json(out / f"{period}_join.json", {"period": period, "join_keys": ["pixel_id", "t0", "lead_min"],
                                                 "gold_table": f"table_A1_v1/gid={SOURCE_GOLD}",
                                                 "source_run": f"{SOURCE_JOB}/{SOURCE_RUN}", "files": stats,
                                                 "note": NOT_FLOOD})
    rels.append(PB.rel(out / f"{period}_join.json"))
    return rels, {"rows": int(sum(s["rows"] for s in stats)), "label_missing": int(sum(s["label_missing"]
                                                                                      for s in stats))}


# ============================================================================ reliability diagrams
def ece(df: pd.DataFrame) -> float:
    d = df[df["n"] > 0]
    return float((d["n"] * (d["mean_p"] - d["obs_freq"]).abs()).sum() / d["n"].sum()) if len(d) else float("nan")


def run_reliability(unit: PB.Unit, pc: PostCtx) -> tuple[list[str], dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cfg = pc.cfg
    r = pd.read_csv(pc.src.fetch("reports/p4b/final/reliability.csv"))
    out = out_dir("reliability")
    leads, thrs = [int(x) for x in cfg["frames"]["lead_min"]], PB.thresholds(cfg)
    rels, rows = [], []
    colors = {"v1": "#1b6ca8", "v1_A": "#d95f02", "v1_B": "#1b9e77"}
    for period in ("test", "secondary"):
        rec = PB.period_record(cfg, period)
        fig, ax = plt.subplots(len(thrs), len(leads), figsize=(4.2 * len(leads), 4.0 * len(thrs)),
                               constrained_layout=True, squeeze=False)
        for i, t in enumerate(thrs):
            for j, L in enumerate(leads):
                a = ax[i, j]
                a.plot([0, 1], [0, 1], color="0.6", lw=0.8, ls="--")
                cell = r[(r["period"] == period) & (r["lead_min"] == L) & np.isclose(r["threshold"], t)]
                for m in MODELS_FINAL:
                    d = cell[(cell["model"] == m) & (cell["n"] > 0)]
                    if len(d):
                        a.plot(d["mean_p"], d["obs_freq"], marker="o", ms=3, lw=1.2, color=colors[m], label=m)
                    rows.append({"period": period, "lead_min": L, "threshold": t, "model": m,
                                 "ece": ece(cell[cell["model"] == m]), "n": int(cell[cell["model"] == m]["n"].sum()),
                                 "label_source": rec["label_source"]})
                b = a.twinx()
                v = cell[cell["model"] == "v1"]
                b.bar((v["p_lo"] + v["p_hi"]) / 2, v["n"].clip(lower=1), width=0.09, color="0.85", zorder=0)
                b.set_yscale("log"); b.set_ylabel("n (v1)", fontsize=7); b.tick_params(labelsize=6)
                a.set_zorder(b.get_zorder() + 1); a.patch.set_visible(False)
                a.set_xlim(0, 1); a.set_ylim(0, 1)
                a.set_title(f">= {t:g} mm/h, F{L}", fontsize=9)
                a.set_xlabel("forecast probability", fontsize=8); a.set_ylabel("observed frequency", fontsize=8)
                if i == 0 and j == 0:
                    a.legend(fontsize=7, loc="upper left")
        fig.suptitle(f"P4B v1 reliability — {period} {rec['t0_from_utc'][:10]}..{rec['t0_to_utc_exclusive'][:10]} "
                     f"(t0 UTC, end exclusive); labels = {rec['label_name']}\n{NOT_FLOOD}", fontsize=9)
        p = out / f"reliability_{period}.png"
        fig.savefig(p, dpi=100); plt.close(fig)
        rels.append(PB.rel(p))
    e = pd.DataFrame(rows)
    e.insert(0, "note", NOT_FLOOD)
    e.to_csv(out / "ece.csv", index=False)
    rels.append(PB.rel(out / "ece.csv"))
    return rels, {"figures": 2, "cells": len(e)}


# ============================================================================ replay packages
def obs_class(rate: np.ndarray, thrs: list[float]) -> np.ndarray:
    c = np.zeros(len(rate), "int8")
    for ci, t in enumerate(sorted(thrs), start=1):
        c = np.where(rate >= t, ci, c)
    return np.where(np.isnan(rate), -1, c).astype("int8")


def pred_class(df: pd.DataFrame, model: str, thrs: list[float]) -> np.ndarray:
    c = np.zeros(len(df), "int8")
    for ci, t in enumerate(sorted(thrs), start=1):
        c = np.where(df[f"yhat_{model}_{kstr(t)}"].to_numpy() == 1, ci, c)
    return c


def day_metrics(df: pd.DataFrame, models: list[str], thrs: list[float], label: str) -> pd.DataFrame:
    """Per local day x frame x threshold x model: contingency, POD, FAR, CSI, bias, Brier (persistence: 0/1)."""
    from src.models import metrics as M
    rows = []
    for (day, L), g in df.groupby(["day", "lead_min"], sort=True):
        for t in thrs:
            k = kstr(t)
            ok = g[f"y_{k}"].notna().to_numpy()
            y = g[f"y_{k}"].to_numpy()[ok].astype(bool)
            for m in models:
                if f"yhat_{m}_{k}" not in g:
                    continue
                f = g[f"yhat_{m}_{k}"].to_numpy()[ok]
                p = g[f"p_{m}_{k}"].to_numpy("float64")[ok]
                fin = ~np.isnan(p)
                c = M.contingency(y[fin], f[fin] == 1)
                s = M.scores(**c)
                rows.append({"day_local": day, "lead_min": int(L), "threshold": t, "model": m, "n": int(fin.sum()),
                             "base_rate": float(y[fin].mean()) if fin.any() else float("nan"), **c,
                             **{x: float(v) for x, v in s.items()},
                             "brier": M.brier(y[fin], p[fin]) if fin.any() else float("nan"), "label_source": label})
    out = pd.DataFrame(rows)
    out["note"] = NOT_FLOOD
    return out


def panels(df: pd.DataFrame, name: str, label_name: str, models: list[str], thrs: list[float], out: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    ccmap = ListedColormap(["#f0f0f0", "#a6d96a", "#fdae61", "#d7191c"])
    cnorm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4)
    pt = 2.5 if 2.5 in thrs else thrs[0]
    f0 = df[df["lead_min"] == df["lead_min"].min()].groupby("t0")["obs_rate_mm_h"].mean()
    t_all = list(f0.index)
    c = t_all.index(f0.idxmax()) if f0.notna().any() else 0
    sel = t_all[max(0, min(c - 3, len(t_all) - 8)):][:8]
    spec = [("obs_rate_mm_h", dict(cmap="Blues", vmin=0, vmax=20), f"observed mm/h\n({label_name.split(' (')[0]})")]
    spec += [(f"p_{m}_{kstr(pt)}", dict(cmap="viridis", vmin=0, vmax=1), f"P(>={pt:g}) {m}") for m in models]
    spec += [(f"class_{m}", dict(cmap=ccmap, norm=cnorm), f"class {m}") for m in models]
    spec += [("class_pers", dict(cmap=ccmap, norm=cnorm), "class persistence")]
    paths = []
    for L in sorted(df["lead_min"].unique()):
        d = df[df["lead_min"] == L]
        fig, ax = plt.subplots(len(spec), len(sel), figsize=(2.1 * len(sel), 2.0 * len(spec) + 0.6),
                               constrained_layout=True, squeeze=False)
        for j, t in enumerate(sel):
            s = d[d["t0"] == t]
            loc = (t + pd.Timedelta(minutes=int(L))).tz_convert(TZ)
            for i, (col, kw, ttl) in enumerate(spec):
                a = ax[i, j]
                sc = a.scatter(s["lon"], s["lat"], c=s[col], marker="s", s=60, **kw)
                if j == 0:
                    a.set_ylabel(ttl, fontsize=7)
                if i == 0:
                    a.set_title(f"frame {loc:%m-%d %H:%M}", fontsize=8)
                a.set_xticks([]); a.set_yticks([]); a.set_aspect("equal")
                if j == len(sel) - 1:
                    fig.colorbar(sc, ax=a, shrink=0.8)
        fig.suptitle(f"P4B v1 replay {name} — F{L} (frame start, local time); observed = {label_name}\n{NOT_FLOOD}",
                     fontsize=8)
        p = out / f"panels_F{L}.png"
        fig.savefig(p, dpi=90); plt.close(fig)
        paths.append(p)
    return paths


def _final_event_rows(pc: PostCtx, ev: dict) -> tuple[pd.DataFrame, list[str]]:
    """Wide rows of one event from the stored final predictions (no model is loaded)."""
    cfg = pc.cfg
    lo, hi = local_range(ev["days"])
    plo, phi = PB.period(cfg, ev["source"])
    if lo < plo or hi > phi:
        raise ValueError(f"event days {ev['days']} are not inside the {ev['source']} period")
    gk = [f"gc_k{k}" for k in range(7)]
    frames = []
    for L in [int(x) for x in cfg["frames"]["lead_min"]]:
        g = pc.gold_rows(lo, hi, ["lat", "lon"] + gk, lead=L)
        g["t0"] = to_utc(g["t0"])
        w = g[["pixel_id", "t0", "lead_min", "frame_start", "label_rate_mm_h"]].copy()
        w["pers_rate_mm_h"] = PB.persistence(g, L).astype("float32")
        for t in PB.thresholds(cfg):
            k = kstr(t)
            pr = pd.read_parquet(pc.src.fetch(f"reports/p4b/final/preds/{ev['source']}_lead{L}_thr{k}.parquet"))
            pr["t0"] = to_utc(pr["t0"])
            pr = pr[(pr["t0"] >= lo) & (pr["t0"] < hi)]
            ren = {"y": f"y_{k}", **{f"p_{m}": f"p_{m}_{k}" for m in MODELS_FINAL},
                   **{f"yhat_{m}": f"yhat_{m}_{k}" for m in MODELS_FINAL}}
            w = w.merge(pr.rename(columns=ren)[["pixel_id", "t0"] + list(ren.values())], on=["pixel_id", "t0"],
                        how="inner", validate="one_to_one")
            w[f"p_pers_{k}"] = (np.nan_to_num(w["pers_rate_mm_h"].to_numpy("float64"), nan=-1.0) >= t).astype("float32")
            w[f"yhat_pers_{k}"] = w[f"p_pers_{k}"].astype("int8")
        frames.append(w)
    return pd.concat(frames, ignore_index=True), list(MODELS_FINAL) + ["pers"]


def _oof_event_rows(pc: PostCtx, ev: dict) -> tuple[pd.DataFrame, list[str]]:
    """Yagi: OOF predictions of the held-out fold (cv units of the shipped models), yhat rebuilt from p_raw and the
    cuts in meta.json; 7.6 mm/h nested rule (also above the 2.5 mm/h cut of the same variant)."""
    cfg = pc.cfg
    lo, hi = local_range(ev["days"])
    thrs, pt = PB.thresholds(cfg), PB.primary(cfg)
    hv = float(cfg["targets"]["secondary_mm_h"])
    frames, used = [], {}
    for L in [int(x) for x in cfg["frames"]["lead_min"]]:
        w = None
        for v in ("A", "B"):
            parts = {}
            for t in thrs:
                k = kstr(t)
                meta = pc.src.json(f"models/v1/{v}/lead{L}_thr{k}/meta.json")
                if meta.get("calibration") != "raw":
                    raise RuntimeError(f"{v} lead{L} thr{k}: calibration {meta.get('calibration')!r}; the OOF cut "
                                       "can only be rebuilt from p_raw for raw models")
                uid = meta["cv_unit"]
                info = pc.src.json(f"reports/p4b/cv/{uid}.json")
                fold_start = {int(f["fold"]): str(f["start"])[:7] for f in info["folds"]}
                o = pd.read_parquet(pc.src.fetch(f"reports/p4b/oof/{uid}.parquet"))
                o["t0"] = to_utc(o["t0"])
                o = o[(o["t0"] >= lo) & (o["t0"] < hi)]
                months = {fold_start.get(int(f)) for f in o["fold"].unique()}
                if months != {ev["fold_month"]}:
                    raise RuntimeError(f"{uid}: event rows belong to folds {sorted(map(str, months))}, "
                                       f"not only the held-out month {ev['fold_month']}")
                used[f"{v}-lead{L}-thr{k}"] = {"cv_unit": uid, "cut": meta["cut"]["cut"],
                                               "heavy_rule": meta.get("heavy_rule"), "rows": len(o)}
                parts[t] = (o, meta)
            q_o, q_meta = parts[pt]
            q = q_o[["pixel_id", "t0"]].assign(q=q_o["p_raw"].to_numpy("float64") >= q_meta["cut"]["cut"])
            for t, (o, meta) in parts.items():
                k = kstr(t)
                yh = o["p_raw"].to_numpy("float64") >= meta["cut"]["cut"]
                if t == hv and meta.get("heavy_rule") == "nested":
                    jq = o[["pixel_id", "t0"]].merge(q, on=["pixel_id", "t0"], how="left")
                    yh &= jq["q"].fillna(False).to_numpy(bool)
                cols = o[["pixel_id", "t0", "lead_min", "y", "label_rate_mm_h", "pers"]].copy()
                cols[f"p_v1_{v}_{k}"] = o["p_raw"].astype("float32").to_numpy()
                cols[f"yhat_v1_{v}_{k}"] = yh.astype("int8")
                cols = cols.rename(columns={"y": f"y_{k}"})
                if w is None:
                    w = cols.rename(columns={"pers": "pers_rate_mm_h"})
                else:
                    keep = ["pixel_id", "t0", f"p_v1_{v}_{k}", f"yhat_v1_{v}_{k}"] + ([f"y_{k}"] if f"y_{k}" not in w
                                                                                      else [])
                    w = w.merge(cols[keep], on=["pixel_id", "t0"], how="left" if v == "B" else "outer",
                                validate="one_to_one")
        w["lead_min"] = np.int16(L)
        w["frame_start"] = w["t0"] + pd.Timedelta(minutes=L)
        for t in thrs:
            k = kstr(t)
            w[f"p_pers_{k}"] = (np.nan_to_num(w["pers_rate_mm_h"].to_numpy("float64"), nan=-1.0) >= t).astype("float32")
            w[f"yhat_pers_{k}"] = w[f"p_pers_{k}"].astype("int8")
        frames.append(w)
    df = pd.concat(frames, ignore_index=True)
    df.attrs["used"] = used
    return df, ["v1_A", "v1_B", "pers"]


def run_replay(unit: PB.Unit, pc: PostCtx) -> tuple[list[str], dict]:
    from src.gold.features import Grid
    cfg = pc.cfg
    name = unit.params["event"]
    ev = EVENTS[name]
    thrs = PB.thresholds(cfg)
    df, models = (_oof_event_rows if ev["source"] == "oof" else _final_event_rows)(pc, ev)
    used = df.attrs.get("used")
    df = df.sort_values(["lead_min", "t0", "pixel_id"]).reset_index(drop=True)
    lat, lon = Grid().latlon(df["pixel_id"].to_numpy())
    df["lat"], df["lon"] = lat.astype("float32"), lon.astype("float32")
    df = df.rename(columns={"label_rate_mm_h": "obs_rate_mm_h"})
    df["obs_class"] = obs_class(df["obs_rate_mm_h"].to_numpy("float64"), thrs)
    df["label_source"] = label_of(cfg, df["t0"])
    srcs = sorted(set(df["label_source"]))
    label_name = " + ".join(PB.LABEL_NAMES.get(x, x) for x in srcs)
    df["frame_start_local"] = pd.to_datetime(df["frame_start"]).dt.tz_convert(TZ)
    df["day"] = df["t0"].dt.tz_convert(TZ).dt.date.astype(str)
    df = df[df["day"].isin(ev["days"])].reset_index(drop=True)
    for m in models:
        df[f"class_{m}"] = pred_class(df, m, thrs)
    out = out_dir("replay", name)
    pq = out / "replay.parquet"
    PB.write_parquet(df, pq)
    dm = day_metrics(df, models, thrs, "+".join(srcs))
    dm.to_csv(out / "per_day_metrics.csv", index=False)
    figs = panels(df, name, label_name, [m for m in models if m != "pers"], thrs, out)
    files = [pq, out / "per_day_metrics.csv", *figs]
    summ = {"event": name, "days_local": ev["days"], "source": ev["source"], "label_source": srcs,
            "label_name": label_name, "note": ev["note"], "status": NOT_FLOOD,
            "source_run": f"{SOURCE_JOB}/{SOURCE_RUN}", "models": models, "rows": len(df),
            "prediction_source": ("OOF parquet of the shipped models' cv units" if ev["source"] == "oof"
                                  else f"reports/p4b/final/preds/{ev['source']}_* (stored; no model loaded)"),
            "oof_units": used, "files": {PB.rel(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            "metrics_event": dm.groupby(["lead_min", "threshold", "model"])[["tp", "fp", "fn", "tn"]].sum()
            .reset_index().to_dict("records")}
    PB.write_json(out / "summary.json", summ)
    rels = [PB.rel(p) for p in files] + [PB.rel(out / "summary.json")]
    return rels, {"rows": len(df), "days": len(ev["days"]), "label_source": srcs}


# ============================================================================ index
def run_post_report(pc: PostCtx) -> tuple[list[str], dict]:
    from src.models import cloud_train as CT
    pre = CT.run_prefix("p4b_post", pc.run_id) + "markers/"
    outputs = {}
    for k in sorted(pc.store.list(pre)):
        if k.endswith(".done.json") and not k.endswith("post-report.done.json"):
            outputs.update(json.loads(pc.store.get(k)).get("outputs", {}))
    idx = {"job": "p4b_post", "run_id": pc.run_id, "source_run": f"{SOURCE_JOB}/{SOURCE_RUN}",
           "source_gold_id": SOURCE_GOLD, "gold_id_of_current_code": pc.gold_id_now,
           "status": NOT_FLOOD, "labels": {"test": PB.period_record(pc.cfg, "test"),
                                           "secondary": PB.period_record(pc.cfg, "secondary")},
           "rules": ["no training", "no test re-run or re-scoring: predictions and metrics come from the stored "
                     "files of the source run", "writes only under lake/v1/runs/p4b_post/<run_id>/"],
           "outputs": outputs}
    p = out_dir() / "index.json"
    PB.write_json(p, idx)
    return [PB.rel(p)], {"outputs": len(outputs), "source_run": SOURCE_RUN, "source_gold_id": SOURCE_GOLD,
                         "gold_id_matches_code": pc.gold_id_now == SOURCE_GOLD}


def run_unit(unit: PB.Unit, pc: PostCtx) -> tuple[list[str], dict]:
    if unit.id.startswith("labels-"):
        return run_labels(unit, pc)
    if unit.id == "reliability":
        return run_reliability(unit, pc)
    if unit.id.startswith("replay-"):
        return run_replay(unit, pc)
    if unit.id == "post-report":
        return run_post_report(pc)
    raise ValueError(unit.id)
