"""P4F + P4G gates (job p4fg / p4fg_smoke): hp-select (P4G.3), cmp (P4G.0 oracle, P4G.2 regional, P4G.4 union rule,
P4G.5 selection chain, P4E shadow gate for Model 2, comparison table), fssv1 (P4F.2), fss, figs (P4F.3), compare-done.
Pre-registered rules (configs/model_p4fg.yaml `pre_registered`, DECISIONS D214) are pure functions below so that the
tests check them directly. Events, Matmo and 2026 data are never inputs of a selection. Not a flood model; labels are
IMERG (satellite estimate)."""
from __future__ import annotations

import contextlib
import json

import numpy as np
import pandas as pd

from src.models import p4b as PB
from src.models import p4d as PD
from src.models import p4e as PE
from src.models import p4fg as FG_
from src.models.p4fg import (B0, DEFAULT_HP, NOT_FLOOD, OUT, env, fit_id, heavy, kstr, leads, out_dir, primary, rel,
                             root, spec_name, thr_col, write_json)


# ============================================================================ pure pre-registered rules
def hp_select_rule(totals: dict, default=DEFAULT_HP) -> tuple:
    """totals: {(num_leaves, min_data): summed inner-OOF log-loss}. Lowest wins; tie (|diff| < 1e-9) -> fewer leaves,
    then larger min_data_in_leaf (D214)."""
    best = min(totals.values())
    tied = [k for k, v in totals.items() if abs(v - best) < 1e-9]
    return sorted(tied, key=lambda k: (k[0], -k[1]))[0]


def accept_step(per_frame_25: list[dict], pooled_ll: dict, per_frame_76: list[dict], rule: dict) -> dict:
    """P4G.5 (D214). per_frame_25: {lead_min, logloss_hi, csi_lo}; pooled_ll: {diff, lo, hi} of (new - current) day
    sums pooled over frames; per_frame_76: {lead_min, pod_diff, csi_diff} (point estimates)."""
    a = rule["accept_all"]
    checks = {"pooled_logloss_hi_lt": bool(pooled_ll["hi"] < a["pooled_logloss_2p5_ci90_upper_lt"])}
    for r in per_frame_25:
        checks[f"F{r['lead_min']}_logloss_hi"] = bool(r["logloss_hi"] <= a["every_frame_logloss_2p5_ci90_upper_max"])
        checks[f"F{r['lead_min']}_csi_lo"] = bool(r["csi_lo"] >= a["every_frame_csi_2p5_ci90_lower_min"])
    for r in per_frame_76:
        checks[f"F{r['lead_min']}_pod76"] = bool(r["pod_diff"] >= a["every_frame_pod_7p6_diff_min"])
        checks[f"F{r['lead_min']}_csi76"] = bool(r["csi_diff"] >= a["every_frame_csi_7p6_diff_min"])
    complete = bool(per_frame_25) and bool(per_frame_76) and len(per_frame_25) == len(per_frame_76)
    return {"accept": bool(complete and all(checks.values())), "complete": complete, "checks": checks}


def union_adopt(rows: list[dict], rule: dict) -> dict:
    """P4G.4 (D214). rows: {lead_min, csi_diff, csi_lo, bias_union} for (union - model-only) at 7.6."""
    need = [int(x) for x in rule["adopt_all_frames"]]
    by = {int(r["lead_min"]): r for r in rows}
    checks = {}
    for L in need:
        r = by.get(L)
        if r is None:
            checks[f"F{L}_present"] = False
            continue
        checks[f"F{L}_csi_lo"] = bool(r["csi_lo"] >= rule["csi_diff_ci90_lower_min"])
        checks[f"F{L}_csi_point"] = bool(r["csi_diff"] > rule["csi_diff_point_gt"])
        checks[f"F{L}_bias"] = bool(r["bias_union"] <= rule["bias_max"])
    return {"adopt": bool(checks) and all(checks.values()), "complete": all(L in by for L in need), "checks": checks}


def regional_skill(rows: list[dict], rule: dict) -> dict:
    """P4G.2 (D214). rows: {lead_min, bss, bss_lo, bss_hi, dbss, dbss_lo, dbss_hi} at one threshold."""
    by = {int(r["lead_min"]): r for r in rows}
    checks = {}
    for L in [int(x) for x in rule["skill_frames"]]:
        r = by.get(L)
        if r is None:
            checks[f"F{L}_present"] = False
            continue
        checks[f"F{L}_bss"] = bool(r["bss"] > 0 and r["bss_lo"] > 0)
        checks[f"F{L}_dbss_pers"] = bool(r["dbss"] > 0 and r["dbss_lo"] > 0)
    return {"skill": bool(checks) and all(checks.values()), "checks": checks}


def choose_cut_csi(x: np.ndarray, y: np.ndarray) -> float:
    """Oracle A: threshold on a rate (mm/h) maximising CSI (candidates 0.1..40 mm/h)."""
    from src.models import metrics as M
    x = np.nan_to_num(np.asarray(x, "float64"), nan=-1.0)
    y = np.asarray(y, bool)
    cands = np.unique(np.r_[np.round(np.arange(0.1, 2.0, 0.1), 2), np.round(np.arange(2.0, 40.01, 0.2), 2)])
    best, bc = -1.0, float(cands[0])
    for c in cands:
        v = float(M.scores(**M.contingency(y, x >= c))["csi"])
        if np.isfinite(v) and v > best:
            best, bc = v, float(c)
    return bc


# ============================================================================ OOF access
def fit_info(ctx, uid: str, oracle: bool = False) -> dict:
    return ctx.read_json(f"{OUT}/{'oracle/' if oracle else ''}fit/{uid}.json")


def fold_pred(ctx, spec: str, s: int, L: int, thr: float) -> tuple[pd.DataFrame, dict]:
    orc = spec == "orc"
    uid = fit_id(spec, s, L, thr)
    info = fit_info(ctx, uid, orc)
    if info.get("alias"):
        uid = info["alias"]
        info = fit_info(ctx, uid, orc)
    r = f"{OUT}/{'oracle/' if orc else ''}oof/{uid}.parquet"
    if not (root() / r).is_file():
        ctx.fetch_run_files(r)
    o = pd.read_parquet(root() / r)
    o["t0"] = o["t0"].astype("datetime64[ns, UTC]")
    return o, info


def base_rows(ctx, s: int, L: int) -> pd.DataFrame:
    with env():
        return PD.base_rows(ctx, s, L)


def pooled(ctx, fold_specs: dict, L: int, thr: float, roles=("eval",)) -> pd.DataFrame:
    """Base rows of the held-out folds joined with the predictions of the model {fold: spec}: y, p_raw, p_cal (fold
    Platt), yhat at the fold cut; 7.6 nested with the same spec's 2.5 alarm (heavy_rule nested, as v1)."""
    cfg = ctx.cfg
    out = []
    for s, spec in sorted(fold_specs.items()):
        b = base_rows(ctx, s, L)
        b = b[b["role"].isin(roles)]
        o, info = fold_pred(ctx, spec, s, L, thr)
        j = b.merge(o[["pixel_id", "t0", "p_raw"]], on=["pixel_id", "t0"], how="inner", validate="one_to_one")
        if len(j) != len(b):
            raise RuntimeError(f"{spec} s{s} lead{L} thr{thr:g}: {len(b) - len(j)} base rows without prediction")
        j["y"] = j[thr_col(thr)].astype(bool)
        j["p_clim"] = j[f"pclim_{kstr(thr)}"].astype("float64")
        j["p_raw"] = j["p_raw"].astype("float64")
        j["p_cal"] = PB.apply_calibrator(info["platt"], j["p_raw"].to_numpy())
        j["yhat"] = j["p_cal"].to_numpy() >= info["cut"]["cut"]
        j["season_fold"] = s
        if thr == heavy(cfg) and cfg["evaluation"]["heavy_rule"] == "nested":
            o2, i2 = fold_pred(ctx, spec, s, L, primary(cfg))
            q = j[["pixel_id", "t0"]].merge(o2[["pixel_id", "t0", "p_raw"]], on=["pixel_id", "t0"], how="left")
            pq = PB.apply_calibrator(i2["platt"], q["p_raw"].to_numpy("float64"))
            j["yhat"] = j["yhat"].to_numpy() & (np.nan_to_num(pq, nan=-1.0) >= i2["cut"]["cut"])
        out.append(j)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def model_specs(ctx) -> dict:
    """{model name: {fold: spec}} of every candidate (P4G.5 list): B0, tuned, conv steps on B0 and on tuned."""
    cfg = ctx.cfg
    hp = {s: FG_.hp_choice(ctx, s) for s in FG_.folds(cfg)}
    out = {"v1B": {s: "v1b" for s in hp}, "Model1-B": {s: B0 for s in hp},
           "tuned": {s: spec_name(h) for s, h in hp.items()}}
    for c in ("a", "c", "ac"):
        out[f"B0+{c}"] = {s: spec_name(DEFAULT_HP, c) for s in hp}
        out[f"tuned+{c}"] = {s: f"t{c}" for s in hp}
    return out


def chain_model(hp_mode: str, conv: str) -> str:
    base = "Model1-B" if hp_mode == "default" else "tuned"
    if not conv:
        return base
    return f"{'B0' if hp_mode == 'default' else 'tuned'}+{conv}"


# ============================================================================ hp-select (P4G.3)
def gate_hp_select(ctx) -> tuple[list[str], dict]:
    cfg = ctx.cfg
    pt = primary(cfg)
    rows, res = [], {}
    cap = int(cfg["inner_cv"]["max_rounds"])
    share = float(cfg["pre_registered"]["hp_grid"]["report_rounds_share_above"])
    for s in FG_.folds(cfg):
        totals = {}
        for h in cfg["hp_grid_points"]:
            h = tuple(h)
            tot = 0.0
            for L in leads(cfg):
                f = fit_info(ctx, fit_id(spec_name(h), s, L, pt))
                n, ll = f["inner_oof"]["n"], f["inner_oof"]["logloss_raw"]
                tot += n * ll
                rows.append({"heldout_season": s, "num_leaves": h[0], "min_data_in_leaf": h[1], "lead_min": L,
                             "inner_oof_n": n, "inner_oof_logloss_raw": ll, "median_rounds": f["median_rounds"],
                             "rounds_share_of_cap": f["median_rounds"] / cap,
                             "rounds_above_share": f["median_rounds"] > share * cap})
            totals[h] = tot
        ch = hp_select_rule(totals)
        res[str(s)] = {"setting": list(ch), "spec": spec_name(ch), "is_default": ch == DEFAULT_HP,
                       "totals": {f"{k[0]}x{k[1]}": v for k, v in totals.items()}}
    out = out_dir("compare")
    t = pd.DataFrame(rows)
    t["chosen"] = [res[str(r.heldout_season)]["setting"] == [r.num_leaves, r.min_data_in_leaf] for r in t.itertuples()]
    t.to_csv(out / "hp_grid_inner.csv", index=False)
    summ = {"folds": res, "rule": cfg["pre_registered"]["hp_grid"]["select_metric"],
            "rounds_above_share_units": t.loc[t["rounds_above_share"], ["heldout_season", "num_leaves",
                                                                       "min_data_in_leaf", "lead_min"]]
            .to_dict("records"), "note": NOT_FLOOD}
    write_json(out / "hp_select.json", summ)
    return [rel(out / "hp_grid_inner.csv"), rel(out / "hp_select.json")], {
        "chosen": {k: v["setting"] for k, v in res.items()}}


# ============================================================================ cmp (P4G.0, P4G.2, P4G.4, P4G.5, shadow)
def _stats(j: pd.DataFrame, yhat: np.ndarray) -> dict:
    from src.models import metrics as M
    sc = M.scores(**M.contingency(j["y"].to_numpy(bool), np.asarray(yhat, bool)))
    return {"n": int(len(j)), "positives": int(j["y"].sum()), "csi": float(sc["csi"]), "pod": float(sc["pod"]),
            "far": float(sc["far"]), "bias": float(sc["bias"])}


def _boot_csi_diff(ctx, j: pd.DataFrame, ya: np.ndarray, yb: np.ndarray) -> dict:
    from src.models import metrics as M
    ev = ctx.cfg["evaluation"]
    p = j["p_cal"].to_numpy() if "p_cal" in j else np.zeros(len(j))
    da = M.day_stats(j["day"].to_numpy(), j["y"], np.asarray(ya, bool), p, j["p_clim"])
    db = M.day_stats(j["day"].to_numpy(), j["y"], np.asarray(yb, bool), p, j["p_clim"])
    d = M.paired_bootstrap(da, db, ev["bootstrap_resamples"], ctx.cfg["random_seed"], ev["ci_level"])
    return {"csi_diff": d["csi"]["diff"], "csi_lo": d["csi"]["lo"], "csi_hi": d["csi"]["hi"]}


def selection_chain(ctx, specs: dict) -> dict:
    cfg = ctx.cfg
    rule = cfg["pre_registered"]["selection"]
    pt, hv = primary(cfg), heavy(cfg)
    cur = {"hp": "default", "conv": ""}
    steps = []
    cache: dict = {}

    def fr(name, L, t):
        k = (name, L, t)
        if k not in cache:
            cache[k] = pooled(ctx, specs[name], L, t)
        return cache[k]
    for step in rule["order"]:
        if step == "conv_g_b":
            steps.append({"step": step, "status": "dropped",
                          "reason": "TCWV not available from the Previous Runs API (conv probe)"})
            continue
        if step == "hp_tuned":
            cand = {"hp": "tuned", "conv": cur["conv"]}
        elif step == "conv_g_a":
            cand = {"hp": cur["hp"], "conv": "a"}
        elif step == "conv_g_c":
            cand = {"hp": cur["hp"], "conv": ("ac" if "a" in cur["conv"] else "c")}
        else:
            raise ValueError(step)
        cn, kn = chain_model(cand["hp"], cand["conv"]), chain_model(cur["hp"], cur["conv"])
        same = specs[cn] == specs[kn]
        p25, p76, la, lb = [], [], [], []
        for L in leads(cfg):
            a, b = fr(cn, L, pt), fr(kn, L, pt)
            d = PE.paired(ctx, a, b, pt)
            p25.append({"lead_min": L, "logloss_diff": d["logloss_cal_diff"], "logloss_lo": d["logloss_lo"],
                        "logloss_hi": d["logloss_hi"], "csi_diff": d["csi_diff"], "csi_lo": d["csi_lo"],
                        "csi_hi": d["csi_hi"]})
            la.append(a); lb.append(b)
            a7, b7 = fr(cn, L, hv), fr(kn, L, hv)
            sa, sb = _stats(a7, a7["yhat"]), _stats(b7, b7["yhat"])
            p76.append({"lead_min": L, "pod_diff": sa["pod"] - sb["pod"], "csi_diff": sa["csi"] - sb["csi"],
                        "pod_new": sa["pod"], "csi_new": sa["csi"], "pod_cur": sb["pod"], "csi_cur": sb["csi"]})
        pl = PE._ll_pooled_frames(ctx, la, lb)
        v = accept_step(p25, pl, p76, rule)
        if same:
            v = {"accept": False, "complete": True, "checks": {"identical_to_current": True}}
        steps.append({"step": step, "candidate": cn, "current": kn, "candidate_specs": specs[cn],
                      "pooled_logloss": {k: pl[k] for k in ("diff", "lo", "hi")}, "per_frame_2p5": p25,
                      "per_frame_7p6": p76, **v, "status": "accepted" if v["accept"] else "rejected"})
        if v["accept"]:
            cur = cand
    m2 = chain_model(cur["hp"], cur["conv"])
    return {"steps": steps, "model2": m2, "model2_specs": specs[m2], "model2_is_model1b": m2 == "Model1-B",
            "rule": rule, "smoke": bool(cfg.get("smoke")), "note": "events, Matmo, 2026 data never used; " + NOT_FLOOD}


def shadow_for_model2(ctx, specs: dict, m2: str) -> dict:
    """P4E shadow gate (pre-registered D189) for Model 2 vs v1 (B): same function as P4E, Model 2 in the Model 1-B
    slot."""
    cfg = ctx.cfg
    pt, hv = primary(cfg), heavy(cfg)
    days = event_day_sets(ctx, FG_.folds(cfg))
    pool_days = set(days.get("yagi", set())).union(*[v for k, v in days.items() if k.startswith("wettest")])
    met, diffs, ev7, mat = [], [], [], []
    for L in leads(cfg):
        a, b = pooled(ctx, specs[m2], L, pt), pooled(ctx, specs["v1B"], L, pt)
        met.append({"period": "eval", "lead_min": L, "threshold": pt, "model": "Model1-B",
                    **PE.metric_ci(ctx, a, pt)})
        diffs.append({"period": "eval", "lead_min": L, "threshold": pt, "comparison": "Model1-B - v1B",
                      **PE.paired(ctx, a, b, pt)})
        for n, sp in (("Model1-B", specs[m2]), ("v1B", specs["v1B"])):
            j = pooled(ctx, sp, L, hv, ("eval", "secondary"))
            sel = j["day"].isin(pool_days).to_numpy() & (j["role"] == "eval").to_numpy()
            ev7.append({"lead_min": L, "model": n, **PE._cat(j[sel], j["yhat"].to_numpy(bool)[sel],
                                                             j["p_cal"].to_numpy()[sel])})
            if L == 0:
                sm = j["day"].isin(days.get("matmo", set())).to_numpy() & (j["role"] == "secondary").to_numpy()
                if sm.any():
                    mat.append({"lead_min": L, "model": n, **PE._cat(j[sm], j["yhat"].to_numpy(bool)[sm],
                                                                     j["p_cal"].to_numpy()[sm])})
    c2 = {**cfg, "frames": {**cfg["frames"], "lead_min": leads(cfg)}}     # frames of this run (smoke: F30 only)
    g = PE.shadow_gate(c2, pd.DataFrame(diffs), pd.DataFrame(met), pd.DataFrame(ev7), pd.DataFrame(mat))
    return g | {"model2": m2, "frames_evaluated": leads(cfg),
                "note": "P4E shadow gate (D189) applied to Model 2 vs v1 (B); reported only"}


def event_day_sets(ctx, folds) -> dict:
    """P4E event days (evaluation.events, local days) + the wettest days of every fold at the first frame of this run
    (P4E definition). Descriptive / shadow gate only, never a selection input."""
    cfg = ctx.cfg
    out = {n: {d.strftime("%Y-%m-%d") for d in pd.date_range(a, b, freq="D")}
           for n, (a, b) in cfg["evaluation"]["events"].items()}
    for s in folds:
        lab = base_rows(ctx, s, leads(cfg)[0])
        lab = lab[lab["role"] == "eval"].groupby("day")["label_rate_mm_h"].mean()
        out[f"wettest{s}"] = set(lab.sort_values(ascending=False).index[:int(cfg["evaluation"]["wettest_days"])])
    return out


def union_rule(ctx, specs: dict, names: list[str]) -> dict:
    cfg = ctx.cfg
    hv = heavy(cfg)
    rule = cfg["pre_registered"]["union_rule"]
    res, rows = {}, []
    for n in names:
        fr_rows = []
        for L in [x for x in leads(cfg) if x in [int(v) for v in rule["adopt_all_frames"]]]:
            j = pooled(ctx, specs[n], L, hv)
            ym = j["yhat"].to_numpy(bool)
            yp = np.nan_to_num(j["pers"].to_numpy("float64"), nan=-1.0) >= hv
            yu = ym | yp
            sm, sp_, su = _stats(j, ym), _stats(j, yp), _stats(j, yu)
            d = _boot_csi_diff(ctx, j, yu, ym)
            r = {"model": n, "lead_min": L, **{f"{k}_model": v for k, v in sm.items()},
                 **{f"{k}_pers": v for k, v in sp_.items()}, **{f"{k}_union": v for k, v in su.items()},
                 "pod_diff": su["pod"] - sm["pod"], "far_diff": su["far"] - sm["far"], **d}
            rows.append(r)
            fr_rows.append({"lead_min": L, "csi_diff": d["csi_diff"], "csi_lo": d["csi_lo"], "bias_union": su["bias"]})
        res[n] = union_adopt(fr_rows, rule)
    return {"verdicts": res, "rows": rows, "rule": rule}


def oracle_a(ctx) -> pd.DataFrame:
    cfg = ctx.cfg
    rows = []
    fl = FG_.folds(cfg)
    for L in leads(cfg):
        series = {}
        for s in fl:
            b = base_rows(ctx, s, L)
            b = b[b["role"] == "eval"].reset_index(drop=True)
            a, cols = FG_.oracle_cols(ctx, b.assign(ym=0), L)
            series[s] = (b, dict(zip(cols, a.T)))
        for t in (primary(cfg), heavy(cfg)):
            for src in ("gc", "raw"):
                key = f"oracle_{src}_target"
                cut = {}
                for s in fl:
                    others = [o for o in fl if o != s] or [s]
                    x = np.concatenate([series[o][1][key] for o in others])
                    yy = np.concatenate([series[o][0][thr_col(t)].to_numpy(bool) for o in others])
                    cut[s] = choose_cut_csi(x, yy)
                for mode in ("at_threshold", "oof_cut"):
                    allj, ally = [], []
                    for s in fl:
                        b, v = series[s]
                        c = t if mode == "at_threshold" else cut[s]
                        yh = np.nan_to_num(v[key], nan=-1.0) >= c
                        j = b.assign(y=b[thr_col(t)].astype(bool))
                        rows.append({"oracle": True, "kind": f"A_{src}", "mode": mode, "fold": s, "lead_min": L,
                                     "threshold": t, "cut_mm_h": c, "cut_in_sample": len(fl) == 1 and mode == "oof_cut",
                                     **_stats(j, yh), "nan_share": float(np.isnan(v[key]).mean())})
                        allj.append(j); ally.append(yh)
                    jj = pd.concat(allj, ignore_index=True)
                    rows.append({"oracle": True, "kind": f"A_{src}", "mode": mode, "fold": "pooled", "lead_min": L,
                                 "threshold": t, **_stats(jj, np.concatenate(ally))})
    return pd.DataFrame(rows)


def regional(ctx, specs: dict) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    from src.models import metrics as M
    cfg, ev = ctx.cfg, ctx.cfg["evaluation"]
    rule = cfg["pre_registered"]["regional"]
    rows, rel_, verdict = [], [], {}
    for t in [float(x) for x in rule["thresholds"]]:
        skill_rows = []
        for L in leads(cfg):
            parts = []
            for s in FG_.folds(cfg):
                b = base_rows(ctx, s, L)
                b = b[b["role"] == "eval"]
                o, info = fold_pred(ctx, "reg", s, L, t)
                j = b.merge(o, on=["pixel_id", "t0"], how="inner", validate="one_to_one")
                j["p_cal"] = PB.apply_calibrator(info["platt"], j["p_raw"].to_numpy("float64"))
                j["yhat"] = j["p_cal"].to_numpy() >= info["cut"]["cut"]
                parts.append(j)
            j = pd.concat(parts, ignore_index=True)
            y = j["y_reg"].to_numpy(bool)
            pc = j["pclim_reg"].to_numpy("float64")
            pers = np.nan_to_num(j["pers_reg"].to_numpy("float64"), nan=-1.0) >= t
            day = j["day"].to_numpy()
            da = M.day_stats(day, y, j["yhat"].to_numpy(bool), j["p_cal"].to_numpy(), pc)
            db = M.day_stats(day, y, pers, pers.astype("float64"), pc)
            ci = M.bootstrap_ci(da, ev["bootstrap_resamples"], cfg["random_seed"], ev["ci_level"])
            d = M.paired_bootstrap(da, db, ev["bootstrap_resamples"], cfg["random_seed"], ev["ci_level"])
            bss = float(M.bss(y, j["p_cal"].to_numpy(), pc))
            bss_p = float(M.bss(y, pers.astype("float64"), pc))
            sc = M.scores(**M.contingency(y, j["yhat"].to_numpy(bool)))
            scp = M.scores(**M.contingency(y, pers))
            px = pooled(ctx, specs["Model1-B"], L, t)            # pixel target, same pixels and rows
            r = {"threshold": t, "lead_min": L, "n": int(len(j)), "base_rate": float(y.mean()), "bss": bss,
                 "bss_lo": ci["bss_clim"][0], "bss_hi": ci["bss_clim"][1], "bss_pers": bss_p,
                 "dbss": d["bss_clim"]["diff"], "dbss_lo": d["bss_clim"]["lo"], "dbss_hi": d["bss_clim"]["hi"],
                 "csi": float(sc["csi"]), "pod": float(sc["pod"]), "far": float(sc["far"]), "bias": float(sc["bias"]),
                 "csi_pers": float(scp["csi"]), "pixel_target_csi_model1b": _stats(px, px["yhat"])["csi"],
                 "pixel_target_base_rate": float(px["y"].mean()), "experiment_only": True}
            rows.append(r)
            skill_rows.append(r)
            rel_.append(M.reliability(y, j["p_cal"].to_numpy("float64"), ev["reliability_bins"])
                        .assign(threshold=t, lead_min=L, model="regional"))
        verdict[f"{t:g}"] = regional_skill(skill_rows, rule)
    return pd.DataFrame(rows), verdict, pd.concat(rel_, ignore_index=True) if rel_ else pd.DataFrame()


def oracle_b(ctx, specs: dict) -> pd.DataFrame:
    cfg = ctx.cfg
    rows = []
    for t in (primary(cfg), heavy(cfg)):
        for L in leads(cfg):
            a = pooled(ctx, {s: "orc" for s in FG_.folds(cfg)}, L, t)
            b = pooled(ctx, specs["Model1-B"], L, t)
            ma, mb = PE.metric_ci(ctx, a, t), PE.metric_ci(ctx, b, t)
            d = PE.paired(ctx, a, b, t)
            rows.append({"oracle": True, "kind": "B", "threshold": t, "lead_min": L,
                         **{f"{k}_oracle_b": ma.get(k) for k in ("logloss_cal", "bss_cal", "csi", "pod", "far", "bias")},
                         **{f"{k}_model1b": mb.get(k) for k in ("logloss_cal", "bss_cal", "csi", "pod", "far", "bias")},
                         "logloss_diff": d["logloss_cal_diff"], "logloss_lo": d["logloss_lo"],
                         "logloss_hi": d["logloss_hi"], "csi_diff": d["csi_diff"], "csi_lo": d["csi_lo"],
                         "csi_hi": d["csi_hi"],
                         "heavy_gap_remaining_share": ((1 - ma["csi"]) / (1 - mb["csi"]) if t == heavy(cfg) and
                                                       mb["csi"] < 1 else None)})
    return pd.DataFrame(rows)


def gate_cmp(ctx) -> tuple[list[str], dict]:
    cfg = ctx.cfg
    specs = model_specs(ctx)
    sel = selection_chain(ctx, specs)
    m2 = sel["model2"]
    sh = shadow_for_model2(ctx, specs, m2)
    un = union_rule(ctx, specs, list(dict.fromkeys(["Model1-B", m2])))
    oa = oracle_a(ctx)
    ob = oracle_b(ctx, specs)
    rg, rv, rgrel = regional(ctx, specs)
    dt = ctx.read_json(f"{OUT}/dtcheck/dt_check.json")
    table = comparison_table(ctx, specs, sel, oa, ob)
    out = out_dir("compare")
    files = {"comparison_table.csv": table, "union_rule.csv": pd.DataFrame(un["rows"]),
             "regional_metrics.csv": rg, "regional_reliability.csv": rgrel}
    for k, v in files.items():
        v.to_csv(out / k, index=False)
    oo = out_dir("oracle")
    oa.assign(oracle=True).to_csv(oo / "oracle_a.csv", index=False)
    ob.assign(oracle=True).to_csv(oo / "oracle_b.csv", index=False)
    write_json(oo / "oracle_summary.json", {"oracle": True, "note": "deliberately leaky diagnostic, never served",
                                            "heavy_gap_remaining_share": ob.loc[ob["threshold"] == heavy(cfg),
                                                                                ["lead_min", "heavy_gap_remaining_share"]]
                                            .to_dict("records")})
    summ = {"selection": sel, "shadow_gate_model2": sh,
            "union_rule": {k: v for k, v in un.items() if k != "rows"},
            "union_adopted_for_model2": bool(un["verdicts"].get(m2, {}).get("adopt", False)),
            "regional_skill": rv, "dt_check": {"verdict": dt["verdict"], "pooled_ratio": dt[
                "pooled_ratio_gc_imerg_code_dt"]}, "smoke": bool(cfg.get("smoke")), "note": NOT_FLOOD}
    write_json(out / "selection.json", summ)
    rels = [rel(out / k) for k in files] + [rel(out / "selection.json"), rel(oo / "oracle_a.csv"),
                                            rel(oo / "oracle_b.csv"), rel(oo / "oracle_summary.json")]
    return rels, {"model2": m2, "steps": {s["step"]: s["status"] for s in sel["steps"]},
                  "shadow_gate_pass": sh["pass"], "union_adopted": summ["union_adopted_for_model2"],
                  "dt_verdict": dt["verdict"]}


def comparison_table(ctx, specs: dict, sel: dict, oa: pd.DataFrame, ob: pd.DataFrame) -> pd.DataFrame:
    from src.models import metrics as M
    cfg = ctx.cfg
    names = ["v1B", "Model1-B", "tuned"] + [s["candidate"] for s in sel["steps"] if s.get("candidate")]
    names = list(dict.fromkeys(names))
    rows = []
    for t in (primary(cfg), heavy(cfg)):
        for L in leads(cfg):
            ref = pooled(ctx, specs["v1B"], L, t)
            for n in names:
                j = ref if n == "v1B" else pooled(ctx, specs[n], L, t)
                r = {"block": "models", "model": n, "is_model2": n == sel["model2"], "threshold": t, "lead_min": L,
                     **{k: v for k, v in PE.metric_ci(ctx, j, t).items() if k in (
                         "logloss_cal", "bss_cal", "csi", "pod", "far", "bias", "ece_cal", "csi_lo", "csi_hi")}}
                if n != "v1B":
                    d = PE.paired(ctx, j, ref, t)
                    r |= {"dlogloss_vs_v1B": d["logloss_cal_diff"], "dlogloss_lo": d["logloss_lo"],
                          "dlogloss_hi": d["logloss_hi"], "dcsi_vs_v1B": d["csi_diff"], "dcsi_lo": d["csi_lo"],
                          "dcsi_hi": d["csi_hi"]}
                rows.append(r)
            yp = np.nan_to_num(ref["pers"].to_numpy("float64"), nan=-1.0) >= t
            sc = M.scores(**M.contingency(ref["y"].to_numpy(bool), yp))
            rows.append({"block": "models", "model": "persistence", "threshold": t, "lead_min": L,
                         "csi": float(sc["csi"]), "pod": float(sc["pod"]), "far": float(sc["far"]),
                         "bias": float(sc["bias"])})
    for _, r in ob.iterrows():
        rows.append({"block": "oracle (diagnostic)", "model": "oracle_b", "threshold": r["threshold"],
                     "lead_min": r["lead_min"], "logloss_cal": r["logloss_cal_oracle_b"], "bss_cal": r["bss_cal_oracle_b"],
                     "csi": r["csi_oracle_b"], "pod": r["pod_oracle_b"], "far": r["far_oracle_b"],
                     "bias": r["bias_oracle_b"]})
    for _, r in oa[(oa["fold"] == "pooled")].iterrows():
        rows.append({"block": "oracle (diagnostic)", "model": f"oracle_{r['kind'].lower()}_{r['mode']}",
                     "threshold": r["threshold"], "lead_min": r["lead_min"], "csi": r["csi"], "pod": r["pod"],
                     "far": r["far"], "bias": r["bias"]})
    t = pd.DataFrame(rows)
    t["display_name"] = [display(cfg, m) for m in t["model"]]
    return t


def display(cfg: dict, name: str) -> str:
    dn = cfg.get("display_names", {})
    if name in dn:
        return dn[name]
    if name.startswith("B0+") or name.startswith("tuned+"):
        base, c = name.split("+")
        return f"{dn.get('m1b' if base == 'B0' else 'tuned', base)} {' '.join(dn.get(f'conv_g_{x}', x) for x in c)}"
    if name.startswith("oracle_"):
        return dn.get("oracle_b", name) if name == "oracle_b" else f"{name} (chẩn đoán, không phục vụ)"
    return name


# ============================================================================ FSS (P4F.2 v1 on P4B; all candidates)
def _fss_rows(j: pd.DataFrame, obs: np.ndarray, fc: np.ndarray, sizes, n, seed, level, **key) -> list[dict]:
    from src.models import p4e_figs as FGX
    out = []
    for sz in sizes:
        tr = FGX.fss_terms(j["pixel_id"].to_numpy(), j["t0"], j["day"].to_numpy(), obs, fc, sz)
        v, lo, hi = FGX.fss_ci(tr, n, seed, level)
        out.append({**key, "size": f"{sz}x{sz}", "fss": v, "lo": lo, "hi": hi, "days": int(len(tr))})
    return out


def _p4b_store_file(ctx, rel_in_run: str) -> pd.DataFrame | dict:
    src = ctx.cfg["sources"]["p4b_run"]
    key = f"lake/v1/runs/{src['job']}/{src['run_id']}/files/{rel_in_run}"
    dst = root() / "data" / "_scratch" / "p4fg" / "p4b" / rel_in_run
    ls = ctx.store.list(key)
    if key not in ls:
        raise FileNotFoundError(key)
    ctx._download(key, ls[key], dst)
    return pd.read_parquet(dst) if dst.suffix == ".parquet" else pd.read_csv(dst)


def run_fssv1(unit, ctx) -> tuple[list[str], dict]:
    """P4F.2: FSS of v1 (set B, the shipped v1) on the P4B row-level predictions on R2 (OOF 2024 cv-B24, cut and
    raw calibration from v1_models_oof.csv; test 2025 final/preds yhat_v1) + persistence, 47-pixel mask, truncated
    windows, pooled sums, day-block bootstrap."""
    cfg, ev = ctx.cfg, ctx.cfg["evaluation"]
    n, seed, level = int(ev["bootstrap_resamples"]), int(cfg["random_seed"]), float(ev["ci_level"])
    pt, hv = primary(cfg), heavy(cfg)
    idx = _p4b_store_file(ctx, "reports/p4b/oof/v1_models_oof.csv").set_index("model")
    rows = []
    sd = FG_.smoke_days(cfg)
    for L in leads(cfg):
        oof = {}
        for t in (pt, hv):
            o = _p4b_store_file(ctx, f"reports/p4b/oof/{idx.loc[f'B-lead{L}-thr{kstr(t)}', 'cv_unit']}.parquet")
            o["t0"] = pd.to_datetime(o["t0"], utc=True)
            o["yhat"] = o["p_raw"].to_numpy("float64") >= float(idx.loc[f"B-lead{L}-thr{kstr(t)}", "cut"])
            oof[t] = o
        o76 = oof[hv].merge(oof[pt][["pixel_id", "t0", "yhat"]].rename(columns={"yhat": "y25"}), on=["pixel_id", "t0"])
        o76["yhat"] = o76["yhat"] & o76["y25"]
        for t, o in ((pt, oof[pt]), (hv, o76)):
            if sd is not None:
                o = o[o["t0"].dt.date.isin(sd)]
            o = o.assign(day=o["t0"].dt.tz_convert(PD.LOCAL_TZ).dt.date.astype(str))
            if not len(o):
                continue
            obs = (o["label_rate_mm_h"].to_numpy("float64") >= t)
            rows += _fss_rows(o, obs, o["yhat"].to_numpy(bool), (3, 5), n, seed, level, model="v1",
                              data="P4B OOF 2024 (cv-B24)", lead_min=L, threshold=t)
            rows += _fss_rows(o, obs, np.nan_to_num(o["pers"].to_numpy("float64"), nan=-1.0) >= t, (3, 5), n, seed,
                              level, model="persistence", data="P4B OOF 2024 (cv-B24)", lead_min=L, threshold=t)
        b = base_rows(ctx, 2025, L) if 2025 in FG_.folds(cfg) else None
        for t in (pt, hv):
            te = _p4b_store_file(ctx, f"reports/p4b/final/preds/test_lead{L}_thr{kstr(t)}.parquet")
            te["t0"] = pd.to_datetime(te["t0"], utc=True)
            if b is not None:
                te = te.merge(b[["pixel_id", "t0", "day", "pers", "label_rate_mm_h"]], on=["pixel_id", "t0"],
                              how="inner")
            else:
                te = te.assign(day=te["t0"].dt.tz_convert(PD.LOCAL_TZ).dt.date.astype(str), pers=np.nan)
            if not len(te):
                continue
            obs = te["y"].to_numpy(bool)
            rows += _fss_rows(te, obs, te["yhat_v1"].to_numpy(bool), (3, 5), n, seed, level, model="v1",
                              data="P4B test 2025 (final/preds)", lead_min=L, threshold=t)
            rows += _fss_rows(te, obs, np.nan_to_num(te["pers"].to_numpy("float64"), nan=-1.0) >= t, (3, 5), n,
                              seed, level, model="persistence", data="P4B test 2025 (final/preds)", lead_min=L,
                              threshold=t)
    out = out_dir("fss")
    t = pd.DataFrame(rows)
    t["path"] = "row-level P4B predictions on R2 (no recomputation)"
    t["note"] = NOT_FLOOD
    t.to_csv(out / "fss_v1_p4b.csv", index=False)
    return [rel(out / "fss_v1_p4b.csv")], {"rows": int(len(t))}


def gate_fss(ctx) -> tuple[list[str], dict]:
    cfg, ev = ctx.cfg, ctx.cfg["evaluation"]
    n, seed, level = int(ev["bootstrap_resamples"]), int(cfg["random_seed"]), float(ev["ci_level"])
    specs = model_specs(ctx)
    seen, rows = {}, []
    for name, sp in specs.items():
        key = json.dumps(sp, sort_keys=True)
        if key in seen:
            rows_alias = seen[key]
            rows += [r | {"model": name, "same_as": rows_alias} for r in rows if r["model"] == rows_alias]
            continue
        seen[key] = name
        for t in (primary(cfg), heavy(cfg)):
            for L in leads(cfg):
                j = pooled(ctx, sp, L, t)
                obs = j["y"].to_numpy(bool)
                rows += _fss_rows(j, obs, j["yhat"].to_numpy(bool), (3, 5), n, seed, level, model=name,
                                  lead_min=L, threshold=t)
                if name == "v1B":
                    rows += _fss_rows(j, obs, np.nan_to_num(j["pers"].to_numpy("float64"), nan=-1.0) >= t, (3, 5),
                                      n, seed, level, model="persistence", lead_min=L, threshold=t)
    out = out_dir("fss")
    t = pd.DataFrame(rows)
    t["note"] = NOT_FLOOD
    t.to_csv(out / "fss_models.csv", index=False)
    return [rel(out / "fss_models.csv")], {"rows": int(len(t))}


# ============================================================================ figures (P4F.3)
@contextlib.contextmanager
def fig_text(cfg: dict):
    from src.models import p4e_figs as FGX
    old = dict(FGX.DISPLAY), dict(FGX.TEXT)
    FGX.DISPLAY.clear(); FGX.TEXT.clear()
    FGX.DISPLAY.update({k: v for k, v in cfg.get("display_names", {}).items()})
    FGX.TEXT.update(cfg.get("axis_labels", {}))
    try:
        yield FGX
    finally:
        FGX.DISPLAY.clear(); FGX.DISPLAY.update(old[0])
        FGX.TEXT.clear(); FGX.TEXT.update(old[1])


def gate_figs(ctx) -> tuple[list[str], dict]:
    cfg, ev = ctx.cfg, ctx.cfg["evaluation"]
    n, seed, lvl = int(ev["bootstrap_resamples"]), int(cfg["random_seed"]), float(ev["ci_level"])
    sel = ctx.read_json(f"{OUT}/compare/selection.json")["selection"]
    specs = model_specs(ctx)
    m2 = sel["model2"]
    models = list(dict.fromkeys(["v1B", "Model1-B"] + ([m2] if m2 != "Model1-B" else [])))
    label = {"v1B": "v1B", "Model1-B": "Model1-B", m2: "m2"} if m2 != "Model1-B" else {"v1B": "v1B",
                                                                                     "Model1-B": "Model1-B"}
    thrs, Ls = [primary(cfg), heavy(cfg)], leads(cfg)
    title = "P4FG SMOKE" if cfg.get("smoke") else "P4FG"
    paths, tables = [], {}
    with fig_text(cfg) as FGX:
        F = {}
        for mname in models:
            for t in thrs:
                for L in Ls:
                    j = pooled(ctx, specs[mname], L, t)
                    F[(label[mname], float(t), int(L))] = j[[c for c in PE.FIG_COLS if c in j.columns]].copy()
        mlist = [label[m] for m in models]
        out = out_dir("figs")
        p, tables["reliability"] = FGX.reliability(F, mlist, thrs, Ls, out, title, ev["reliability_bins"])
        paths += p
        p, tables["scores_ci"] = FGX.scores_ci(F, mlist, thrs, Ls, out, title, n, seed, lvl)
        paths += p
        p, tables["fss"] = FGX.fss(F, mlist, thrs, Ls, out, title, [1, 3, 5], n, seed, lvl)
        paths += p
        p, tables["halfmonth_csi"] = FGX.halfmonth_csi(F, mlist, thrs, Ls, out, title)
        paths += p
        del F
        try:                                           # P4F.3 re-render of stored P4E outputs: reported, never blocking
            rp, rt = relabel_p4e(ctx, FGX, n, seed, lvl)
            paths += rp
            tables.update(rt)
            relabel = {"ok": True, "png": len(rp)}
        except Exception as e:                         # noqa: BLE001
            relabel = {"ok": False, "error": repr(e)[:300]}
    write_json(out_dir("figs") / "p4e_relabel_status.json", relabel)
    for k, t in tables.items():
        t = t.copy()
        t["note"] = NOT_FLOOD
        t.to_csv(out_dir("figs") / f"{k}.csv", index=False)
    return [rel(x) for x in paths] + [rel(out_dir("figs") / f"{k}.csv") for k in tables] + \
        [rel(out_dir("figs") / "p4e_relabel_status.json")], {"png": len(paths), "p4e_relabel_ok": relabel["ok"]}


P4E_OUT = "reports/p4e"            # folder of the stored P4E outputs (runs/p4e/<run>/files/reports/p4e/)


@contextlib.contextmanager
def p4e_out():
    """Inside FG.env() the module global PE.OUT points at reports/p4fg; p4e.pooled -> p4d_env() copies PE.OUT into
    PD.OUT, so the P4E reads must see P4E's own folder (S2 fix D218-1: the smoke relabel looked for
    reports/p4fg/base/... under the P4E run and failed with FileNotFoundError)."""
    old = PE.OUT
    PE.OUT = P4E_OUT
    try:
        yield
    finally:
        PE.OUT = old


def relabel_p4e(ctx, FGX, n, seed, lvl) -> tuple[list, dict]:
    """P4F.3: the P4E figures re-rendered from the stored P4E outputs (run sources.p4e_run, read only) with the display
    names; output under reports/p4fg/figs/p4e_relabel/."""
    with p4e_out():
        F, fs, thrs, Ls, src, cfg_e = _p4e_frames(ctx, n)
    models = [m for m, _ in fs["models"]]
    out = out_dir("figs", "p4e_relabel")
    paths, tables = [], {}
    title = f"P4E {src['run_id']} (vẽ lại, nhãn P4F.3)"
    p, tables["p4e_relabel_reliability"] = FGX.reliability(F, models, thrs, Ls, out, title,
                                                           cfg_e["evaluation"]["reliability_bins"])
    paths += p
    p, tables["p4e_relabel_scores_ci"] = FGX.scores_ci(F, models, thrs, Ls, out, title, n, seed, lvl)
    paths += p
    p, tables["p4e_relabel_halfmonth_csi"] = FGX.halfmonth_csi(F, models, thrs, Ls, out, title)
    paths += p
    return paths, tables


def _p4e_frames(ctx, n):
    """Pooled eval frames of the P4E figure models, read from the stored P4E run (call inside p4e_out())."""
    from src.models import p4e as PEm
    src = ctx.cfg["sources"]["p4e_run"]
    cfg_e = PEm.load_cfg(smoke=False)
    cfg_e["evaluation"]["bootstrap_resamples"] = n
    ce = PEm.Ctx(cfg_e, ctx.store, src["job"], src["run_id"], ctx.mirror, ctx.log)
    fs = PEm.fig_spec(cfg_e)
    thrs, Ls = [PEm.primary(cfg_e), PEm.heavy(cfg_e)], PEm.leads(cfg_e)
    F = {}
    for name, spec in fs["models"]:
        for t in thrs:
            if not PEm.has_fit(cfg_e, spec, t, fs["folds"]):
                continue
            for L in Ls:
                j = PEm.pooled(ce, spec, L, t, ("eval",), fs["folds"])
                F[(name, float(t), int(L))] = j[[c for c in PEm.FIG_COLS if c in j.columns]].copy()
    return F, fs, thrs, Ls, src, cfg_e


# ============================================================================ compare-done
def gate_compare_done(ctx) -> tuple[list[str], dict]:
    sel = ctx.read_json(f"{OUT}/compare/selection.json")
    hp = ctx.read_json(f"{OUT}/compare/hp_select.json")
    tab = pd.read_csv(root() / OUT / "compare" / "comparison_table.csv") if (root() / OUT / "compare" /
                                                                          "comparison_table.csv").is_file() else None
    if tab is None:
        ctx.fetch_run_files(f"{OUT}/compare/comparison_table.csv")
        tab = pd.read_csv(root() / OUT / "compare" / "comparison_table.csv")
    fssr = f"{OUT}/fss/fss_models.csv"
    if not (root() / fssr).is_file():
        ctx.fetch_run_files(fssr)
    fs = pd.read_csv(root() / fssr)
    piv = fs.pivot_table(index=["model", "threshold", "lead_min"], columns="size", values="fss").reset_index()
    piv.columns = [c if c in ("model", "threshold", "lead_min") else f"fss_{c}" for c in piv.columns]
    full = tab.merge(piv, on=["model", "threshold", "lead_min"], how="left")
    out = out_dir("compare")
    full.to_csv(out / "comparison_table_full.csv", index=False)
    s = sel["selection"]
    summ = {"job": ctx.job, "run_id": ctx.run_id, "gold_id": ctx.gid, "conv_id": FG_.conv_id(ctx.cfg),
            "prereg_sha": FG_.PREREG_SHA, "smoke": bool(ctx.cfg.get("smoke")), "model2": s["model2"],
            "model2_specs": s["model2_specs"], "hp_select": hp["folds"], "steps": [
                {k: st.get(k) for k in ("step", "status", "candidate")} for st in s["steps"]],
            "shadow_gate_model2": {k: sel["shadow_gate_model2"].get(k) for k in ("pass", "complete")},
            "union_adopted_for_model2": sel["union_adopted_for_model2"], "regional_skill": sel["regional_skill"],
            "dt_check": sel["dt_check"], "stage": "P4FG comparison + selection complete (stop point S2/S3)",
            "note": NOT_FLOOD}
    write_json(out / "compare_summary.json", summ)
    return [rel(out / "comparison_table_full.csv"), rel(out / "compare_summary.json")], {
        "compare_complete": True, "model2": s["model2"], "dt_verdict": sel["dt_check"]["verdict"]}


GATES = {"hp-select": gate_hp_select, "cmp": gate_cmp, "fss": gate_fss, "figs": gate_figs,
         "compare-done": gate_compare_done}
