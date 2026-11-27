"""P4F.1 final fit + P4F.6 REPLAY package (job p4fg_final; smoke code paths in p4fg_smoke: finalfit-smoke,
replay-smoke). Not a flood model; labels are IMERG (satellite estimate). v1 stays the official model; the REPLAY package
is for the web demo only (lake/v1/serving/replay/pkg=2026-11/, put-if-absent), never live serving.

p4fg_final reads the SELECTION.json of a finished p4fg run (source: configs/model_p4fg.yaml final_job.source_run_id,
else lake/v1/runs/p4fg/latest.json) and REFUSES to run without it. REPLAY is not exported when the P4F.0 dt-check
verdict of the source run is not "ok" (TASK: do not export REPLAY if P4F.0 stopped).

Units of p4fg_final (stage):
  0 sel                         gate: source run id, SELECTION.json, Model 2 per fold, dt verdict (refuses if missing)
  1 loso05-<spec>-s<fold>-lead<L> Model 2 LOSO fit at 0.5 mm/h when Model 2 != Model 1-B (REPLAY 0p5 columns)
    final-<model>-lead<L>-thr<t>  final fit of Model 2 and Model 1-B (P4F.1) -> models/v2_candidate/<model>/...
  2 replay-<event>              REPLAY rows of one event (yagi, wet0930, wettest3, matmo)
  3 replay-publish              gate: _manifest.json, upload to the serving prefix (put-if-absent), re-download check
  4 final-done                  gate: summary
"""
from __future__ import annotations

import gzip
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.models import p4b as PB
from src.models import p4d as PD
from src.models import p4e as PE
from src.models import p4fg as FG_
from src.models.p4fg import (B0, NOT_FLOOD, env, heavy, kstr, leads, out_dir, primary, rel, root, spec_def,
                             spec_name, thr_col, write_json)

SERVING_PREFIX = "lake/v1/serving/replay/pkg=2026-11/"
MODEL_DIRS = {"model2": "model2", "model1b": "model1b"}
FI_CFG = "configs/flood_indicator.yaml"
EVENTS = ("yagi", "wet0930", "wettest3", "matmo")


# ============================================================================ source run (p4fg) and selection
def source_run_id(store, cfg: dict) -> str:
    rid = (cfg.get("final_job") or {}).get("source_run_id")
    if rid:
        return str(rid)
    k = "lake/v1/runs/p4fg/latest.json"
    if k not in store.list(k):
        raise RuntimeError("p4fg_final: no source p4fg run (latest.json missing); run p4fg first")
    return json.loads(store.get(k))["run_id"]


def read_selection(store, cfg: dict) -> dict:
    rid = source_run_id(store, cfg)
    k = f"lake/v1/runs/p4fg/{rid}/SELECTION.json"
    if k not in store.list(k):
        raise RuntimeError(f"p4fg_final refuses to run: selection marker {k} does not exist (TASK P4F.1)")
    sel = json.loads(store.get(k))
    sel["source_run_id"] = rid
    return sel


def resolved_specs(sel: dict) -> dict:
    """{model: {fold: resolved spec b<nl>x<md><conv> or v1b}} from the selection marker (tuned aliases resolved)."""
    hp = {int(s): tuple(v["setting"]) for s, v in sel["hp_select"].items()}
    out = {}
    for name, fs in (("model2", sel["model2_specs"]), ("model1b", {s: B0 for s in sel["model2_specs"]})):
        r = {}
        for s, sp in fs.items():
            s = int(s)
            dd = spec_def(sp)
            r[s] = spec_name(hp[s] if dd["hp"] == "tuned" else dd["hp"], dd["conv"])
        out[name] = r
    return out


def final_setting(sel: dict, model: str) -> str:
    """P4F.1 (D214): the per-fold setting when both folds agree, else the setting with the lowest summed inner-OOF
    log-loss over the two LOSO folds; conv groups of Model 2 kept."""
    rs = resolved_specs(sel)[model]
    specs = set(rs.values())
    if len(specs) == 1:
        return specs.pop()
    conv = spec_def(next(iter(rs.values())))["conv"]
    tot = {}
    for s, v in sel["hp_select"].items():
        for k, x in v["totals"].items():
            tot[k] = tot.get(k, 0.0) + float(x)
    best = min(tot, key=lambda k: (tot[k], int(k.split("x")[0]), -int(k.split("x")[1])))
    nl, md = best.split("x")
    return spec_name((int(nl), int(md)), conv)


# ============================================================================ P4F.1 final fit
def final_rounds(infos: list[dict]) -> int:
    """Median of the LOSO fits' median_rounds (round half up), D214."""
    v = sorted(int(i["median_rounds"]) for i in infos)
    m = float(np.median(v))
    return max(int(np.floor(m + 0.5)), 1)


def final_mask(meta: pd.DataFrame, cfg: dict) -> np.ndarray:
    d = meta["t0"].dt.tz_convert("UTC").dt.date
    m = np.zeros(len(meta), bool)
    for a, b in cfg["pre_registered"]["final_fit"]["data"]:
        m |= ((d >= pd.Timestamp(a).date()) & (d <= pd.Timestamp(b).date())).to_numpy()
    return m & meta["train_month"].to_numpy(bool)


def code_hash() -> str:
    h = hashlib.sha256()
    for f in FG_.CODE_FILES:
        h.update((FG_.CODE_ROOT / f).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def final_fit(ctx, model: str, spec: str, L: int, thr: float, loso_infos: list[dict], out_root: str) -> tuple[list[str], dict]:
    """Fit `spec` on the final-fit data (decision 6; smoke: the smoke days inside those ranges), rounds = median of the
    LOSO fits' rounds; Platt + cut from an inner CV on the same rows (calendar-month blocks, predictions at the final
    rounds); package model.txt, meta.json, importance.csv under <out_root>/<model>/lead<L>_thr<t>/."""
    from src.models import metrics as M
    cfg = ctx.cfg
    dd = spec_def(spec)
    if dd.get("oracle") or dd.get("regional"):
        raise RuntimeError("final fit of an oracle / regional spec refused")
    feats = FG_.spec_features(ctx, spec)
    FG_.assert_no_oracle(feats, spec)
    meta, X = FG_.load(ctx, L, feats)
    city = meta["is_city"].to_numpy("float32") == 1
    keep = final_mask(meta, cfg) & (meta["t0"] >= pd.Timestamp(PE.nwp_from(cfg), tz="UTC")).to_numpy()
    if not dd["ring"]:
        keep &= city
    y = meta[thr_col(thr)].to_numpy("int8")
    rounds = final_rounds(loso_infos)
    hp = dd["hp"]
    params = PE.lgb_params(cfg) | {"num_leaves": int(hp[0]), "min_data_in_leaf": int(hp[1])}
    cv = PE.cv_fit(X, y, None, keep, keep & city, meta, params, cfg["inner_cv"], feats, cfg["set_b"]["inner_cv_block"])
    iy, ip = [], []
    for mo, ho, bst in cv["boosters"]:
        iy.append(y[ho])
        ip.append(bst.predict(X[ho], num_iteration=max(min(rounds, bst.current_iteration()), 1)))
    iy, ip = np.concatenate(iy).astype(bool), np.concatenate(ip)
    platt = PB.fit_calibrator("platt", ip, iy)
    cut = M.choose_cut(iy, PB.apply_calibrator(platt, ip), cfg["evaluation"]["cut_rules"][f"{thr:g}"])
    bst = PE.refit(X, y, None, keep, rounds, params, feats)
    d = root() / out_root / MODEL_DIRS[model] / f"lead{L}_thr{kstr(thr)}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.txt").write_text(bst.model_to_string(), encoding="utf-8")
    imp = pd.DataFrame({"feature": feats, "gain": bst.feature_importance("gain"),
                        "split": bst.feature_importance("split")}).sort_values("gain", ascending=False)
    imp.to_csv(d / "importance.csv", index=False)
    t = meta.loc[keep, "t0"]
    info = {"model": model, "spec": spec, "setting": list(hp), "conv_groups": dd["conv"], "lead_min": L,
            "threshold": thr, "features": feats, "rounds": rounds, "rounds_rule": cfg["pre_registered"]["final_fit"]
            ["rounds"], "loso_rounds": [int(i["median_rounds"]) for i in loso_infos],
            "data_range_utc": [str(t.min()), str(t.max())], "train_rows": int(keep.sum()),
            "label_source": "imerg_final_v07 (2025-10 excluded: Matmo out of sample)", "platt": platt, "cut": cut,
            "cut_rule": cfg["evaluation"]["cut_rules"][f"{thr:g}"], "heavy_rule": "nested (7.6 alarm needs the 2.5 "
            "alarm of the same model)", "probabilities": "p_raw from model.txt; p_cal = Platt(p_raw); alarm p_cal >= cut",
            "run_id": ctx.run_id, "job": ctx.job, "gold_id": ctx.gid, "nwp_id": PE.nwp_id(cfg),
            "conv_id": FG_.conv_id(cfg) if dd["conv"] else None, "code_sha256": code_hash(),
            "smoke": bool(cfg.get("smoke")), "note": "v2 candidate, NOT the official model (v1 stays official); "
            + NOT_FLOOD}
    write_json(d / "meta.json", info)
    return [rel(d / n) for n in ("model.txt", "importance.csv", "meta.json")], {"rounds": rounds,
                                                                                 "train_rows": int(keep.sum())}


# ============================================================================ REPLAY
def wettest_days(base: pd.DataFrame, n: int = 3, exclude: str = "2025-09-30") -> list[str]:
    """D214: 3 local days of 2025-05..09 (eval role) with the highest city-mean IMERG Final label at F0, 09-30
    excluded, ties -> earlier day."""
    b = base[(base["role"] == "eval") & (base["day"] != exclude)]
    m = b.groupby("day")["label_rate_mm_h"].mean().reset_index()
    m = m.sort_values(["label_rate_mm_h", "day"], ascending=[False, True])
    return [str(x) for x in m["day"].head(n)]


def slot_mm_gc(ctx, days_utc: list, pixels: np.ndarray) -> pd.DataFrame:
    """GSMaP NOW GC mm per half-hour slot (index slot END, columns pixels): 0.25 x (r(s-30) + r(s)) — the U1 rule
    (indicator_analysis.py:88, Δt 0.5 h per 30-min file)."""
    gs = FG_.silver_frames(ctx, "gsmap_now", days_utc, pixels, ["window_start_utc", "rain_rate_gc_mm_h"])
    gs["window_start_utc"] = gs["window_start_utc"].astype("datetime64[ns, UTC]")
    r = FG_.slot_rates(gs, "rain_rate_gc_mm_h").reindex(columns=pixels)
    mm = r * FG_.GSMAP_SLOT_DT_H
    mm.index = mm.index + pd.Timedelta(minutes=30)
    return mm


def slot_mm_imerg(ctx, source: str, days_utc: list, pixels: np.ndarray) -> pd.DataFrame:
    im = FG_.silver_frames(ctx, source, days_utc, pixels, ["interval_start_utc", "rain_rate_mm_h"])
    im["interval_start_utc"] = im["interval_start_utc"].astype("datetime64[ns, UTC]")
    piv = im.pivot_table(index="interval_start_utc", columns="pixel_id", values="rain_rate_mm_h", aggfunc="mean")
    piv = piv.reindex(pd.date_range(piv.index.min(), piv.index.max(), freq="30min")).reindex(columns=pixels) * 0.5
    piv.index = piv.index + pd.Timedelta(minutes=30)
    return piv


def icon_columns(t0: pd.Series, pix: np.ndarray, gc_mm: pd.DataFrame, im_mm: pd.DataFrame, gc_rate_now: np.ndarray,
                 im_product: str, fi: dict) -> pd.DataFrame:
    """Operational GC icon (product available at t0): A6h = 10 observed GC slots ending at the newest fully available
    slot end (t0 - 30 min: windows are available 65 min after their start, ADD-005) + 1-hour persistence extension
    (latest admissible GC rate x 1 h, configs/flood_indicator.yaml serving_mode_P5); A24h = 46 slots + 1 h.
    Hindsight IMERG icon: history mode, A6h / A24h = 12 / 48 slots ending at t0. Thresholds: product_thresholds."""
    from src.events import flood_indicator as FI
    a, b, lm = FI.product_thresholds(fi, "gsmap_now_gc")
    ai, bi, li = FI.product_thresholds(fi, im_product)
    gc6 = gc_mm.rolling(10, min_periods=10).sum()
    gc24 = gc_mm.rolling(46, min_periods=46).sum()
    im6 = im_mm.rolling(12, min_periods=12).sum()
    im24 = im_mm.rolling(48, min_periods=48).sum()

    def look(tab, times):
        k = pd.DataFrame({"t": times, "p": pix})
        st = tab.stack(future_stack=True).rename("v").reset_index()
        st.columns = ["t", "p", "v"]
        return k.merge(st, on=["t", "p"], how="left")["v"].to_numpy("float64")
    t_op = (t0 - pd.Timedelta(minutes=30)).to_numpy()
    ext = np.nan_to_num(np.asarray(gc_rate_now, "float64"), nan=np.nan) * 1.0
    a6 = look(gc6, t_op) + ext
    a24 = look(gc24, t_op) + ext
    h6, h24 = look(im6, t0.to_numpy()), look(im24, t0.to_numpy())
    lev = FI.indicator_levels(a6, a24, a, b, lm)
    levh = FI.indicator_levels(h6, h24, ai, bi, li)
    names = {0: "none", 1: "possible", 2: "high", -1: "no_data"}
    return pd.DataFrame({"a6h_mm": a6.astype("float32"), "a24h_mm": a24.astype("float32"),
                         "icon": [names[int(x)] for x in lev], "icon_product": "gsmap_now_gc",
                         "a6h_mm_hindsight": h6.astype("float32"), "a24h_mm_hindsight": h24.astype("float32"),
                         "icon_hindsight": [names[int(x)] for x in levh], "icon_product_hindsight": im_product})


RAIN_CLASSES = ("none", "light", "moderate", "heavy")


def rain_class(a05, a25, a76) -> np.ndarray:
    """Highest alarm (0.5 light, 2.5 moderate, 7.6 heavy: the 7.6 alarm is already nested with 2.5)."""
    out = np.full(len(a25), "none", dtype=object)
    if a05 is not None:
        out[np.asarray(a05, bool)] = "light"
    out[np.asarray(a25, bool)] = "moderate"
    out[np.asarray(a76, bool)] = "heavy"
    return out


def model_preds(ctx_src, specs_by_fold: dict, fold: int, L: int, thr: float, ring: bool) -> pd.DataFrame | None:
    """LOSO OOF (+ ring rows within Chebyshev 2) of a p4fg run: pixel_id, t0, p_raw, p_cal, alarm (fold cut)."""
    from src.models import p4fg_gates as G
    spec = specs_by_fold[fold]
    try:
        o, info = G.fold_pred(ctx_src, spec, fold, L, thr)
    except FileNotFoundError:
        return None
    uid = info["unit"]
    parts = [o[["pixel_id", "t0", "p_raw"]]]
    if ring:
        r = f"{FG_.OUT}/oof_ring/{uid}.parquet"
        if not (root() / r).is_file():
            ctx_src.fetch_run_files(r)
        if (root() / r).is_file():
            q = pd.read_parquet(root() / r)
            q["t0"] = q["t0"].astype("datetime64[ns, UTC]")
            parts.append(q[["pixel_id", "t0", "p_raw"]])
    j = pd.concat(parts, ignore_index=True)
    j["p_cal"] = PB.apply_calibrator(info["platt"], j["p_raw"].to_numpy("float64"))
    j["alarm"] = j["p_cal"].to_numpy() >= info["cut"]["cut"]
    return j


def v1_preds(ctx, kind: str, L: int, thr: float) -> pd.DataFrame:
    """v1 (shipped set B) row-level predictions of the P4B run: kind 'oof2024' (cv-B24, raw, v1 cut) or 'test' /
    'secondary' (final/preds, p_v1 / yhat_v1)."""
    from src.models import p4fg_gates as G
    if kind == "oof2024":
        idx = G._p4b_store_file(ctx, "reports/p4b/oof/v1_models_oof.csv").set_index("model")
        k = f"B-lead{L}-thr{kstr(thr)}"
        o = G._p4b_store_file(ctx, f"reports/p4b/oof/{idx.loc[k, 'cv_unit']}.parquet")
        o["alarm"] = o["p_raw"].to_numpy("float64") >= float(idx.loc[k, "cut"])
        o = o.rename(columns={"p_raw": "p"})
    else:
        o = G._p4b_store_file(ctx, f"reports/p4b/final/preds/{kind}_lead{L}_thr{kstr(thr)}.parquet")
        o = o.rename(columns={"p_v1": "p", "yhat_v1": "alarm"})
    o["t0"] = pd.to_datetime(o["t0"], utc=True).astype("datetime64[ns, UTC]")
    return o[["pixel_id", "t0", "p", "alarm"]]


def build_replay(ctx, event: str, days_local: list[str], plan: dict) -> pd.DataFrame:
    """One row per (half-hour t0, pixel): city 47 + ring within Chebyshev 2 (is_city flag). plan: {"label_source",
    "v1_kind", "oos": {model: (bool, why)}, "models": {name: callable(L, thr) -> DataFrame(pixel_id, t0, p_raw|p,
    alarm)}, "display_union": bool}."""
    from src.gold import features as F
    from src.models import data as D
    cfg = ctx.cfg
    tz = PD.LOCAL_TZ
    t_local = pd.DatetimeIndex(np.concatenate([pd.date_range(pd.Timestamp(d, tz=tz), periods=48, freq="30min")
                                               .tz_convert("UTC").values for d in sorted(days_local)]), tz="UTC")
    t0 = t_local
    months = sorted({(t.year, t.month) for t in t0})
    with env():
        ctx.fetch_gold()
        city, ring, _ = PD.pixel_sets(ctx)
    gparts = []
    for y, m in months:
        p = ctx.gold_local() / f"season={y}" / f"month={m:02d}" / "part.parquet"
        g = pd.read_parquet(p, columns=["pixel_id", "t0", "lead_min", "is_city", "label_rate_mm_h", "label_source"]
                            + [f"gc_k{k}" for k in range(7)], filters=[("lead_min", "==", 0)])
        g["t0"] = g["t0"].astype("datetime64[ns, UTC]")
        gparts.append(g[g["t0"].isin(t0)])
    g = pd.concat(gparts, ignore_index=True)
    meta = g[["pixel_id", "is_city"]].drop_duplicates("pixel_id")
    near = FG_.ring_near_mask(meta.assign(t0=pd.Timestamp("2000-01-01", tz="UTC")), cfg)
    keep_pix = set(meta.loc[(meta["is_city"].to_numpy("float32") == 1) | near, "pixel_id"].astype(int))
    g = g[g["pixel_id"].isin(keep_pix)].sort_values(["t0", "pixel_id"]).reset_index(drop=True)
    pix = g["pixel_id"].to_numpy("int64")
    lat, lon = F.Grid().latlon(pix)
    gcols = [f"gc_k{k}" for k in range(7)]
    arr = np.vstack([g[c].to_numpy("float64") for c in gcols])
    kfirst = np.argmax(~np.isnan(arr), axis=0).astype("float64")
    kfirst[~(~np.isnan(arr)).any(0)] = np.nan
    issue = int(cfg["frames"]["issue_offset_min"])
    out = pd.DataFrame({"t0_utc": g["t0"], "t0_local": g["t0"].dt.tz_convert(tz).dt.strftime("%Y-%m-%d %H:%M"),
                        "pixel_id": pix, "lat": lat.astype("float32"), "lon": lon.astype("float32"),
                        "is_city": g["is_city"].to_numpy("float32") == 1,
                        "gsmap_gc_mmh": D.asof_latest(g).astype("float32"),
                        "gsmap_age_min": (issue + 30.0 * kfirst).astype("float32"),
                        "label_mmh": g["label_rate_mm_h"].to_numpy("float32"),
                        "label_product": g["label_source"].astype(str).to_numpy()})
    key = out[["pixel_id", "t0_utc"]].rename(columns={"t0_utc": "t0"})
    for name, fn in plan["models"].items():
        for L in leads(cfg):
            al = {}
            for thr in FG_.thresholds(cfg):
                pr = fn(L, thr)
                if pr is None:
                    continue
                pcol = "p_raw" if "p_raw" in pr.columns else "p"
                j = key.merge(pr[["pixel_id", "t0", pcol, "alarm"]], on=["pixel_id", "t0"], how="left")
                out[f"p_{name}_{kstr(thr)}_F{L}"] = j[pcol].to_numpy("float32")
                al[thr] = j["alarm"].fillna(False).to_numpy(bool)
            if primary(cfg) in al and heavy(cfg) in al:
                out[f"rain_class_{name}_F{L}"] = rain_class(al.get(0.5), al[primary(cfg)], al[heavy(cfg)])
                if name == "m2":
                    pv = (D.asof_latest(g) if L in (0, 30) else D.max_avail_0_60(g))
                    union = al[heavy(cfg)] | (np.nan_to_num(pv, nan=-1.0) >= heavy(cfg))
                    use_u = bool(plan.get("display_union")) and L in (30, 60)
                    out[f"display_heavy_F{L}"] = union if use_u else al[heavy(cfg)]
        out[f"oos_{name}"] = bool(plan["oos"][name][0])
    out["display_rule"] = "union(model, persistence) F30/F60 (P4G.4 adopted)" if plan.get("display_union") \
        else "model-only (P4G.4 not adopted)"
    fi = yaml.safe_load((FG_.CODE_ROOT / FI_CFG).read_text(encoding="utf-8"))
    days_utc = sorted({(t - pd.Timedelta(days=k)).date() for t in t0 for k in (0, 1, 2)})
    up = np.array(sorted(keep_pix), "int64")
    gc_mm = slot_mm_gc(ctx, days_utc, up)
    im_mm = slot_mm_imerg(ctx, plan["label_source"], days_utc, up)
    ic = icon_columns(out["t0_utc"], pix, gc_mm, im_mm, out["gsmap_gc_mmh"].to_numpy(),
                      "imerg_late" if plan["label_source"] == "imerg_late" else "imerg_final", fi)
    out = pd.concat([out.reset_index(drop=True), ic], axis=1)
    out["t0_utc"] = out["t0_utc"].astype("datetime64[us, UTC]")
    out["event"] = event
    return out


def write_replay(df: pd.DataFrame, d: Path) -> dict:
    d.mkdir(parents=True, exist_ok=True)
    PB.write_parquet(df, d / "replay.parquet")
    cols = {}
    for c in df.columns:
        v = df[c]
        if pd.api.types.is_float_dtype(v):
            cols[c] = [None if not np.isfinite(x) else round(float(x), 2) for x in v.to_numpy("float32")]
        elif pd.api.types.is_datetime64_any_dtype(v):
            cols[c] = v.dt.strftime("%Y-%m-%dT%H:%M:%SZ").tolist()
        elif pd.api.types.is_bool_dtype(v):
            cols[c] = [bool(x) for x in v]
        else:
            cols[c] = [None if (isinstance(x, float) and np.isnan(x)) else (int(x) if isinstance(x, (np.integer,))
                       else x) for x in v.tolist()]
    raw = json.dumps({"rows": int(len(df)), "columns": cols}, separators=(",", ":"), ensure_ascii=False).encode()
    (d / "replay.json.gz").write_bytes(gzip.compress(raw, mtime=0))
    return {n: hashlib.sha256((d / n).read_bytes()).hexdigest() for n in ("replay.parquet", "replay.json.gz")}


# D217-5 (accepted by the user, CLAUDE.md §8 P4F/P4G addendum 2026-11-27): the operational GC windows end 30 min
# before t0; the REPLAY manifest records this shift (S2 fix D218-2)
ICON_WINDOWS = {
    "operational": {"product": "gsmap_now_gc", "a6h": "10 observed GC half-hour slots ending at t0 - 30 min "
                    "+ latest admissible GC rate x 1 h (persistence extension, serving_mode_P5)",
                    "a24h": "46 observed GC half-hour slots ending at t0 - 30 min + latest admissible GC rate x 1 h",
                    "window_end": "t0 - 30 min", "window_shift_min": 30,
                    "why": "GSMaP NOW windows become available 65 min after their start (ADD-005): at t0 the newest "
                           "fully available half-hour slot ends at t0 - 30 min",
                    "slot_rule": "slot [s, s+30) = 0.25 x (r(window s-30) + r(window s)) mm (code dt 0.5 h per "
                                 "30-min file, P4F.0)",
                    "thresholds": "configs/flood_indicator.yaml product_thresholds.gsmap_now_gc",
                    "decision": "D217-5, accepted by the user 2026-11-27 (CLAUDE.md §8 P4F/P4G addendum)"},
    "hindsight": {"product": "imerg_final (imerg_late for Matmo)", "a6h": "12 half-hour slots ending at t0",
                  "a24h": "48 half-hour slots ending at t0", "window_end": "t0", "window_shift_min": 0,
                  "thresholds": "configs/flood_indicator.yaml product_thresholds of the IMERG product"},
    "never": "GSMaP NOW raw is never used for icons",
}


def column_dictionary(df: pd.DataFrame, plan: dict) -> dict:
    out = {}
    for c in df.columns:
        e = {"dtype": str(df[c].dtype)}
        for name, (o, why) in plan["oos"].items():
            if f"_{name}_" in c or c == f"oos_{name}":
                e |= {"model": name, "oos": bool(o), "why": why}
        if c.startswith(("a6h", "a24h", "icon")):
            e["product"] = "imerg (hindsight)" if "hindsight" in c else "gsmap_now_gc (operational, available at t0)"
        if c.startswith("label"):
            e["product"] = plan["label_source"]
        if c.startswith("gsmap"):
            e["product"] = "gsmap_now_gc (available at t0 + 10 min, ADD-005)"
        out[c] = e
    return out


def publish(ctx, files: dict, prefix: str, scratch: Path) -> dict:
    """Upload {relative key: local path} under `prefix` put-if-absent (identical bytes = ok, other bytes = error),
    then re-download every object to `scratch`, compare sha256, and remove the scratch files."""
    res = {"uploaded": [], "identical": [], "verified": [], "mismatch": []}
    for k, p in sorted(files.items()):
        data = Path(p).read_bytes()
        st = ctx.store.put(prefix + k, data)
        res["uploaded" if st == "uploaded" else "identical"].append(prefix + k)
    scratch.mkdir(parents=True, exist_ok=True)
    ls = ctx.store.list(prefix)
    for k, p in sorted(files.items()):
        key = prefix + k
        if key not in ls:
            res["mismatch"].append(key)
            continue
        dst = scratch / k.replace("/", "__")
        dst.write_bytes(ctx.store.get(key))
        ok = hashlib.sha256(dst.read_bytes()).hexdigest() == hashlib.sha256(Path(p).read_bytes()).hexdigest()
        (res["verified"] if ok else res["mismatch"]).append(key)
        dst.unlink()
    res["listed"] = sorted(k for k in ls)
    return res


def fi_hash() -> str:
    return hashlib.sha256((FG_.CODE_ROOT / FI_CFG).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# ============================================================================ smoke units (p4fg_smoke)
def gate_finalfit_smoke(ctx) -> tuple[list[str], dict]:
    from src.models import p4fg_gates as G
    cfg = ctx.cfg
    L, thr = leads(cfg)[0], primary(cfg)
    infos = [G.fit_info(ctx, FG_.fit_id(B0, s, L, thr)) for s in FG_.folds(cfg)]
    rels, m = final_fit(ctx, "model1b", B0, L, thr, infos, f"{FG_.OUT}/smoke_models")
    return rels, m | {"note": "smoke: code path of p4fg_final only"}


def gate_replay_smoke(ctx) -> tuple[list[str], dict]:
    """One event day (smoke_days) exported under the run prefix only (never the serving prefix): v1 from the P4B test
    predictions, Model 1-B and Model 2 (selection of this smoke run) from the fold-2025 LOSO OOF, IMERG Final labels."""
    cfg = ctx.cfg
    day = cfg.get("replay_smoke_day", "2025-07-03")
    sel = ctx.read_json(f"{FG_.OUT}/compare/selection.json")
    specs_m2 = {int(k): v for k, v in sel["selection"]["model2_specs"].items()}
    specs_b0 = {s: B0 for s in specs_m2}
    fold = 2025
    plan = {"label_source": "imerg_final", "display_union": bool(sel["union_adopted_for_model2"]),
            "oos": {"v1": (True, "P4B test 2025 (v1 trained on 2023-2024)"),
                    "m2": (True, "LOSO fold 2025 OOF"), "m1b": (True, "LOSO fold 2025 OOF")},
            "models": {"v1": lambda L, t: v1_preds(ctx, "test", L, t),
                       "m2": lambda L, t: model_preds(ctx, specs_m2, fold, L, t, True),
                       "m1b": lambda L, t: model_preds(ctx, specs_b0, fold, L, t, True)}}
    df = build_replay(ctx, f"smoke_{day}", [day], plan)
    d = out_dir("replay_smoke", day)
    sha = write_replay(df, d)
    man = {"package": "smoke (run prefix only)", "events": {day: {"rows": int(len(df)), "sha256": sha}},
           "columns": column_dictionary(df, plan), "icon_windows": ICON_WINDOWS, "flood_indicator_sha256": fi_hash(),
           "p4f0_verdict": ctx.read_json(f"{FG_.OUT}/dtcheck/dt_check.json")["verdict"], "note": NOT_FLOOD}
    write_json(d / "_manifest.json", man)
    files = {f"{day}/{n}": d / n for n in ("replay.parquet", "replay.json.gz", "_manifest.json")}
    pub = publish(ctx, files, f"lake/v1/runs/{ctx.job}/{ctx.run_id}/replay_pkg/",
                  root() / "data" / "_scratch" / "p4fg" / "replay_verify")
    write_json(d / "publish_check.json", pub)
    if pub["mismatch"]:
        raise RuntimeError(f"replay smoke publish check failed: {pub['mismatch']}")
    return [rel(d / n) for n in ("replay.parquet", "replay.json.gz", "_manifest.json", "publish_check.json")], {
        "rows": int(len(df)), "verified": len(pub["verified"])}


# ============================================================================ p4fg_final job
OUT_FINAL = "reports/p4fg_final"


def final_units(cfg: dict, sel: dict | None) -> list:
    """Units of p4fg_final. Without a selection marker only `sel` exists (it refuses)."""
    U = PB.Unit
    u = [U("sel", "gate", 0)]
    if sel is None:
        return u
    rs = resolved_specs(sel)
    same = rs["model2"] == rs["model1b"]
    loso = []
    if not same:
        for L in leads(cfg):
            for s, sp in sorted(rs["model2"].items()):
                loso.append(U(FG_.fit_id(sp, s, L, 0.5), "loso05", 1, ("sel",),
                              {"spec": sp, "season": s, "lead": L, "thr": 0.5}))
    fin = []
    for model in (("model2", "model1b") if not same else ("model1b",)):
        sp = final_setting(sel, model)
        for L in leads(cfg):
            for t in PD.thresholds(cfg):
                deps = ("sel",) + tuple(x.id for x in loso if x.params["lead"] == L) if (t == 0.5 and model ==
                                                                                       "model2") else ("sel",)
                fin.append(U(f"final-{model}-lead{L}-thr{kstr(t)}", "final", 1, deps,
                             {"model": model, "spec": sp, "lead": L, "thr": t}))
    rep = [U(f"replay-{e}", "replay", 2, ("sel",) + tuple(x.id for x in loso + fin), {"event": e}) for e in EVENTS]
    return u + loso + fin + rep + [U("replay-publish", "gate", 3, tuple(x.id for x in rep)),
                                   U("final-done", "gate", 4, ("replay-publish",))]


def src_ctx(ctx, sel: dict):
    """Read-only context on the source p4fg run (OOF, base rows, fit records)."""
    return FG_.Ctx(ctx.cfg, ctx.store, "p4fg", sel["source_run_id"], ctx.mirror, ctx.log)


def loso_info(cs, spec: str, s: int, L: int, thr: float) -> dict:
    from src.models import p4fg_gates as G
    info = G.fit_info(cs, FG_.fit_id(spec, s, L, thr))
    return G.fit_info(cs, info["alias"]) if info.get("alias") else info


def final_preds(ctx, model: str, L: int, thr: float, months: list) -> pd.DataFrame:
    """Predictions of a packaged final model (models/v2_candidate/<model>/lead<L>_thr<t>/) on the Gold rows of
    `months` (city + ring): p_raw, alarm (Platt + cut of meta.json; 7.6 nested with the 2.5 alarm of the model)."""
    import lightgbm as lgb

    def one(t):
        r = f"models/v2_candidate/{MODEL_DIRS[model]}/lead{L}_thr{kstr(t)}"
        d = root() / r
        if not (d / "model.txt").is_file() or not (d / "meta.json").is_file():
            ctx.fetch_run_files(r + "/")                 # written by a final unit of another part
        meta_ = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        bst = lgb.Booster(model_str=(d / "model.txt").read_text(encoding="utf-8"))
        m, X = FG_.load(ctx, L, meta_["features"], months)
        p = bst.predict(X)
        pc = PB.apply_calibrator(meta_["platt"], p)
        return pd.DataFrame({"pixel_id": m["pixel_id"].to_numpy("int64"), "t0": m["t0"].to_numpy(),
                             "p_raw": p, "alarm": pc >= meta_["cut"]["cut"]})
    o = one(thr)
    if thr == heavy(ctx.cfg):
        o2 = one(primary(ctx.cfg))
        o["alarm"] = o["alarm"].to_numpy() & o2["alarm"].to_numpy()
    o["t0"] = pd.to_datetime(o["t0"], utc=True)
    return o


def event_plan(ctx, sel: dict, event: str) -> tuple[list[str], dict]:
    cs = src_ctx(ctx, sel)
    rs = resolved_specs(sel)
    same = rs["model2"] == rs["model1b"]
    m2_orig = {int(k): v for k, v in sel["model2_specs"].items()}
    b0 = {s: B0 for s in m2_orig}
    union = bool(sel.get("union_adopted_for_model2"))

    def loso(specs, fold, cur05):
        def f(L, t):
            if t == 0.5 and cur05:                        # Model 2 at 0.5: LOSO fit of this final run
                return model_preds(ctx, {fold: rs["model2"][fold]}, fold, L, t, True)
            return model_preds(cs, specs, fold, L, t, True)
        return f
    if event == "matmo":
        days, lab, vk = ["2025-10-06", "2025-10-07"], "imerg_late", "secondary"
        oos = {"v1": (True, "P4B secondary check 2025-10 (v1 trained on 2023-2024)"),
               "m2": (True, "final fit; 2025-10 excluded from training (decision 6)"),
               "m1b": (True, "final fit; 2025-10 excluded from training (decision 6)")}
        models = {"v1": lambda L, t: v1_preds(ctx, vk, L, t),
                  "m2": lambda L, t: final_preds(ctx, "model1b" if same else "model2", L, t, [(2025, 10)])}
        if not same:
            models["m1b"] = lambda L, t: final_preds(ctx, "model1b", L, t, [(2025, 10)])
    else:
        if event == "yagi":
            a, b = ctx.cfg["pre_registered"]["replay"]["yagi"]
            days, fold, vk = [str(d.date()) for d in pd.date_range(a, b, freq="D")], 2024, "oof2024"
            v1why = "P4B OOF fold 2024-09; not fully independent (MODEL_CARD v1 §6.8)"
        elif event == "wet0930":
            days, fold, vk = [ctx.cfg["pre_registered"]["replay"]["wet_test_day"]], 2025, "test"
            v1why = "P4B test 2025"
        else:
            with env():
                b0rows = PD.base_rows(cs, 2025, 0)
            days, fold, vk = wettest_days(b0rows), 2025, "test"
            v1why = "P4B test 2025"
        lab = "imerg_final"
        oos = {"v1": (True, v1why), "m2": (True, f"LOSO fold {fold} OOF"), "m1b": (True, f"LOSO fold {fold} OOF")}
        models = {"v1": lambda L, t: v1_preds(ctx, vk, L, t), "m2": loso(m2_orig, fold, not same)}
        if not same:
            models["m1b"] = loso(b0, fold, False)
    if same:
        oos.pop("m1b")
    return days, {"label_source": lab, "display_union": union, "oos": oos, "models": models}


def run_final_unit(unit, ctx) -> tuple[list[str], dict]:
    from src.models import p4fg_gates as G
    t1 = time.time()
    sel = read_selection(ctx.store, ctx.cfg)
    if unit.id == "sel":
        rs = resolved_specs(sel)
        rec = {"source_run_id": sel["source_run_id"], "model2": sel["model2"],
               "resolved": {k: {str(s): v for s, v in d.items()} for k, d in rs.items()},
               "final_settings": {m: final_setting(sel, m) for m in rs}, "dt_verdict": sel["dt_check"]["verdict"],
               "union_adopted_for_model2": sel["union_adopted_for_model2"], "note": NOT_FLOOD}
        p = root() / OUT_FINAL / "selection_source.json"
        write_json(p, rec)
        return [rel(p)], {"model2": sel["model2"], "dt_verdict": rec["dt_verdict"]}
    if unit.kind == "loso05":
        return FG_.run_fit(unit, ctx)
    if unit.kind == "final":
        cs = src_ctx(ctx, sel)
        rs = resolved_specs(sel)
        model, L, t = unit.params["model"], unit.params["lead"], unit.params["thr"]
        same = rs["model2"] == rs["model1b"]
        orig = ({int(k): v for k, v in sel["model2_specs"].items()} if model == "model2" else
                {s: B0 for s in rs["model1b"]})
        infos = []
        for s in sorted(orig):
            if model == "model2" and t == 0.5 and not same:
                infos.append(G.fit_info(ctx, FG_.fit_id(rs["model2"][s], s, L, t)))
            else:
                infos.append(loso_info(cs, orig[s], s, L, t))
        rels, m = final_fit(ctx, model, unit.params["spec"], L, t, infos, "models/v2_candidate")
        return rels, m | {"seconds": round(time.time() - t1, 1)}
    if unit.kind == "replay":
        if sel["dt_check"]["verdict"] != "ok":
            raise RuntimeError("REPLAY refused: P4F.0 dt-check verdict of the source run is not ok (TASK P4F.6)")
        ev = unit.params["event"]
        days, plan = event_plan(ctx, sel, ev)
        df = build_replay(ctx, ev, days, plan)
        d = root() / OUT_FINAL / "replay" / ev
        sha = write_replay(df, d)
        rec = {"event": ev, "days_local": days, "rows": int(len(df)), "sha256": sha,
               "label_product": plan["label_source"],
               "oos": {k: {"oos": v[0], "why": v[1]} for k, v in plan["oos"].items()},
               "columns": column_dictionary(df, plan), "display_rule": str(df["display_rule"].iloc[0])}
        write_json(d / "event.json", rec)
        return [rel(d / n) for n in ("replay.parquet", "replay.json.gz", "event.json")], {
            "rows": int(len(df)), "seconds": round(time.time() - t1, 1)}
    if unit.id == "replay-publish":
        if sel["dt_check"]["verdict"] != "ok":
            raise RuntimeError("REPLAY refused: P4F.0 verdict not ok")
        evs, files = {}, {}
        for e in EVENTS:
            rec = ctx.read_json(f"{OUT_FINAL}/replay/{e}/event.json")
            for n in ("replay.parquet", "replay.json.gz"):
                if not (root() / OUT_FINAL / "replay" / e / n).is_file():
                    ctx.fetch_run_files(f"{OUT_FINAL}/replay/{e}/{n}")
                files[f"{e}/{n}"] = root() / OUT_FINAL / "replay" / e / n
            evs[e] = {k: rec[k] for k in ("days_local", "rows", "sha256", "label_product", "oos", "display_rule")}
        cols = ctx.read_json(f"{OUT_FINAL}/replay/{EVENTS[0]}/event.json")["columns"]
        man = {"package": "pkg=2026-11", "events": evs, "column_dictionary": cols,
               "wettest_definition": ctx.cfg["pre_registered"]["replay"]["wettest_test_days"],
               "model_runs": {"p4fg": sel["source_run_id"], "p4fg_final": ctx.run_id},
               "model2": sel["model2"], "flood_indicator_sha256": fi_hash(),
               "p4f0_verdict": sel["dt_check"]["verdict"],
               "icons": "operational = GSMaP NOW GC (product available at t0, product_thresholds.gsmap_now_gc, "
                        "1-hour persistence extension); hindsight = IMERG Final (Late for Matmo); GSMaP NOW raw is "
                        "never used for icons",
               "icon_windows": ICON_WINDOWS,
               "notes": ["v1 stays the official model; Model 2 / Model 1-B are candidates (demo only)",
                         "Yagi v1 predictions are P4B OOF, not fully independent (MODEL_CARD v1 §6.8)",
                         "ring rows: v1 columns NaN (P4B predicted city pixels only)", NOT_FLOOD]}
        mp = root() / OUT_FINAL / "replay" / "_manifest.json"
        write_json(mp, man)
        files["_manifest.json"] = mp
        pub = publish(ctx, files, SERVING_PREFIX, root() / "data" / "_scratch" / "p4fg" / "replay_verify")
        pc = root() / OUT_FINAL / "replay" / "publish_check.json"
        write_json(pc, pub)
        if pub["mismatch"]:
            raise RuntimeError(f"REPLAY publish check failed: {pub['mismatch']}")
        return [rel(mp), rel(pc)], {"verified": len(pub["verified"]), "uploaded": len(pub["uploaded"])}
    if unit.id == "final-done":
        rec = {"job": ctx.job, "run_id": ctx.run_id, "source_run_id": sel["source_run_id"], "model2": sel["model2"],
               "stage": "P4F.1 final fit + P4F.6 REPLAY complete (stop point S3)", "note": NOT_FLOOD}
        p = root() / OUT_FINAL / "final_summary.json"
        write_json(p, rec)
        return [rel(p)], {"final_complete": True}
    raise ValueError(unit.id)
