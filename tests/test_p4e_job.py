"""P4E.0 job definition (offline, no data, no LightGBM): configuration merge and the pre-registration guard (D189),
learning rate / round cap (D190), smoke units, dry-scene downsampling (inputs only), rain scene of the two-stage
model, the pre-registered equivalence rule, the cost model reproducing P4D (D191) and part planning."""
import copy
import json
import shutil

import numpy as np
import pandas as pd
import pytest
import yaml

from src.gold import features as F
from src.models import cloud_train as CT
from src.models import p4d as PD
from src.models import p4e as PE
from src.realtime import common as C

CFG = PE.load_cfg()
SMOKE = PE.load_cfg(smoke=True)


# ---------------------------------------------------------------- configuration
def test_merge_on_model_v2_and_p4e_settings():
    assert CFG["lightgbm"]["learning_rate"] == 0.06 and CFG["inner_cv"]["max_rounds"] == 1500
    assert CFG["inner_cv"]["early_stopping_rounds"] == 75 and CFG["lightgbm"]["num_leaves"] == 15
    assert "model2" not in CFG and "smoke" not in CFG
    v2 = PD.load_cfg()
    for k in ("frames", "targets", "scope", "availability", "features", "pixels", "evaluation"):
        assert CFG[k] == v2[k], k                                   # evaluation design unchanged
    assert PE.gold_id(CFG) == PD.gold_id(v2) == "a4a6329bcf13"      # full job reuses the P4D Gold table_v2


def test_smoke_scope():
    assert SMOKE["smoke"] and SMOKE["frames"]["lead_min"] == [0] and SMOKE["evaluation"]["loso_seasons"] == [2024, 2025]
    assert SMOKE["targets"]["thresholds_mm_h"] == [2.5, 7.6] and SMOKE["inner_cv"]["max_rounds"] == 1500
    assert SMOKE["pixels"]["ring"]["max_pixels"] is None                # D198: full ring (224 px) in the smoke
    assert PE.gold_id(SMOKE) != PE.gold_id(CFG)


# ---------------------------------------------------------------- D198 (user, 2026-11-23, after smoke 048a2def9cf8)
def test_d198_config():
    ds = CFG["downsampling"]
    assert ds["status"] == "off" and ds["equivalence_status"] == "not_applied"
    raw = yaml.safe_load(PE.CFG_PATH.read_text(encoding="utf-8"))
    assert PE.prereg_sha(raw) == PE.PREREG_SHA                        # pre-registered thresholds unchanged
    assert not [sp for sp, _, _ in PE.fit_plan(CFG) + PE.fit_plan(SMOKE) if PE.dataset_def(sp).get("ds_p")]
    dg = CFG["diagnostics"]
    assert dg["r_near_count"] == len(dg["r_near_pixels"]) == len(set(dg["r_near_pixels"])) == 90
    assert CFG["set_b"]["option"] == "A" and CFG["set_b"]["inner_cv_block"] == "calendar_month"
    assert SMOKE["set_b"]["inner_cv_block"] == "utc_day"
    assert CFG["downsampling"]["share_units"]["gold_id"] == PE.gold_id(CFG) == "a4a6329bcf13"


def test_d198_full_units():
    us = PE.units(CFG)
    ids = [u.id for u in us]
    assert len(ids) == len(set(ids))
    fits = [u for u in us if u.kind == "fit"]
    specs = {(u.params["spec"], u.params["season"], u.params["thr"]) for u in fits}
    assert {u.params["lead"] for u in fits} == {0, 30, 60} and len(fits) == 3 * len(PE.fit_plan(CFG)) == 129
    for sp in ("m0", "m1", "rnear", "wout"):
        assert {(sp, s, t) for s in (2023, 2024, 2025) for t in (2.5, 7.6)} <= specs
    for sp in ("v1b", "m1b"):
        assert {(sp, s, t) for s in (2024, 2025) for t in (0.5, 2.5, 7.6)} <= specs
        assert not {(sp, 2023, t) for t in (0.5, 2.5, 7.6)} & specs          # NWP history starts 2024-05
    assert {("m1ts", s, 7.6) for s in (2023, 2024, 2025)} | {("m1bts", s, 7.6) for s in (2024, 2025)} <= specs
    assert not [s for s in specs if s[0].startswith("m1ds")]                # D198: no downsampled unit
    nwp = [u for u in us if u.kind == "nwp"]
    assert [u.id for u in nwp] == [f"nwp-{y}-{m:02d}" for y in (2024, 2025) for m in range(5, 11)]
    assert all(u.deps == (f"feat-{u.id[4:]}",) for u in nwp)
    dry = [u for u in us if u.kind == "dry"]
    assert len(dry) == 18 and all(u.params["gid"] == "a4a6329bcf13" for u in dry)     # every month (finding)
    g = {u.id: u for u in us if u.kind == "gate"}
    assert list(g) == ["preflight", "g2", "cmp-diag", "cmp-2stage", "cmp", "figs", "compare-done"]
    assert set(g["g2"].deps) == {u.id for u in us if u.kind in ("feat", "nwp")}
    assert set(g["cmp"].deps) >= {u.id for u in us if u.kind in ("fit", "base", "dry")} | {"cmp-diag", "cmp-2stage"}
    m0 = {u.id for u in fits if u.params["spec"] in ("m0", "m1")}
    assert m0 <= set(g["cmp-diag"].deps) and m0 <= set(g["cmp-2stage"].deps)   # rounds check reads them
    assert {u.id for u in fits if u.params["spec"] == "rnear"} <= set(g["cmp-diag"].deps)
    assert {u.id for u in fits if u.params["spec"] == "m1bts"} <= set(g["cmp-2stage"].deps)
    icv = PE.inner_cv_plan_check(CFG)
    assert all(r["ok"] for r in icv) and {r["variant"] for r in icv} == {"plain", "set_b"}
    assert PE.nwp_id(CFG) != PE.nwp_id(SMOKE)


def test_d198_parts_lpt_and_estimate():
    us = PE.units(CFG)
    e = PE.choose_parts(CFG, us, margin=PE.PART_MARGIN)
    assert e["max_part_minutes"] <= PE.PART_MARGIN * CFG["planning"]["part_target_minutes"]
    assert e["max_part_minutes"] <= CFG["planning"]["part_hard_limit_minutes"]
    n = e["parts"]
    parts = [PE.part_units(us, i, n, CFG) for i in range(1, n + 1)]
    work = [u.id for u in us if u.kind != "gate"]
    got = [u.id for p in parts for u in p]
    assert sorted(got) == sorted(work)                                     # every work unit in exactly one part
    um = PE.unit_minutes(CFG, us)
    loads = [sum(um[u.id] for u in p if u.kind in ("fit", "base", "dry")) for p in parts]
    _, wl = PE.lpt_assign([(u.id, um[u.id]) for u in us if u.kind in ("fit", "base", "dry")], n)
    assert np.allclose(loads, wl)                                          # same assignment as plan_estimate
    for p in parts:                                                        # fits grouped by frame / feature set
        f = [u for u in p if u.kind == "fit"]
        keys = [PE._work_sort_key(u)[:4] for u in f]
        assert keys == sorted(keys)
    assert PE.part_units(us, 1, n) == PD.part_units(us, 1, n)              # without cfg: P4B round-robin
    for k in ("cmp", "compare-done"):
        assert um[k] >= CFG["planning"]["gate_minutes"]["done" if k == "compare-done" else k]   # floor


def test_d198_r_near_from_data():
    G = F.Grid()
    city = np.array([(G.lat_idx_min + 20 + i) * 3600 + G.lon_idx_min + 20 + j for i in range(3) for j in range(3)])
    ring = np.array([(G.lat_idx_min + 20 + i) * 3600 + G.lon_idx_min + 20 + j for i in range(-4, 7)
                     for j in range(-4, 7) if not (0 <= i < 3 and 0 <= j < 3)])
    meta = pd.DataFrame({"pixel_id": np.r_[city, ring], "is_city": np.r_[np.ones(9), np.zeros(len(ring))]})
    ci, cj = G.ij(city)
    ri, rj = G.ij(ring)
    ri, rj = ri - ci.min(), rj - cj.min()                                  # city block = rows / cols 0..2
    d = np.maximum(np.maximum(np.maximum(-ri, ri - 2), 0), np.maximum(np.maximum(-rj, rj - 2), 0))
    want = sorted(int(x) for x in ring[d <= 2])
    assert len(want) == 7 * 7 - 9
    c = copy.deepcopy(CFG)
    c["diagnostics"] |= {"r_near_pixels": want, "r_near_count": len(want)}
    assert [int(x) for x in PE.r_near_pixels(meta, c)] == want
    c["diagnostics"]["r_near_pixels"] = want[:-1]
    with pytest.raises(RuntimeError, match="R-near"):
        PE.r_near_pixels(meta, c)


def test_d198_setb_label_check():
    c = copy.deepcopy(CFG)
    t0 = pd.date_range("2024-05-01", "2025-09-30 23:30", freq="30min", tz="UTC")
    t0 = t0[t0.month.isin([5, 6, 7, 8, 9, 10])]
    lab = pd.DataFrame({"t0": t0, "season": t0.year, "lead_min": 0, "train_month": True})
    for t in PE.thresholds(c):
        lab[PE.thr_col(t)] = ((t0.day == 10) & (t0.hour == 6)).astype("int8")
    c["frames"]["lead_min"] = [0]
    rows = PE.setb_label_check(c, lab)
    assert rows and all(r["ok"] for r in rows) and {r["heldout_season"] for r in rows} == {2024, 2025}
    lab2 = lab.copy()
    for t in PE.thresholds(c):
        lab2.loc[lab2["season"] == 2024, PE.thr_col(t)] = 0               # no positive training row for fold 2025
    bad = [r for r in PE.setb_label_check(c, lab2) if not r["ok"]]
    assert bad and {r["heldout_season"] for r in bad} == {2025}


def _diffs(L, hi=0.0003, lo_csi=-0.002):
    return {"period": "eval", "lead_min": L, "threshold": 2.5, "comparison": "Model1-B - v1B", "logloss_hi": hi,
            "csi_lo": lo_csi}


@pytest.mark.parametrize("change,ok", [({}, True), ({"hi": 0.0006}, False), ({"csi": -0.006}, False),
                                       ({"bss": 0.0}, False), ({"pod": -0.021}, False), ({"c76": -0.011}, False)])
def test_shadow_gate_rule(change, ok):
    c = copy.deepcopy(CFG)
    d = pd.DataFrame([_diffs(L, change.get("hi", 0.0003), change.get("csi", -0.002)) for L in (0, 30, 60)])
    m = pd.DataFrame([{"period": "eval", "lead_min": L, "threshold": 2.5, "model": "Model1-B",
                       "bss_cal": change.get("bss", 0.05)} for L in (0, 30, 60)])
    e = pd.DataFrame([r for L in (0, 30, 60) for r in (
        {"lead_min": L, "model": "v1B", "pod": 0.5, "csi": 0.3},
        {"lead_min": L, "model": "Model1-B", "pod": 0.5 + (change.get("pod", 0.0) if L == 60 else 0),
         "csi": 0.3 + (change.get("c76", 0.0) if L == 60 else 0)})])
    mat = pd.DataFrame([{"lead_min": 0, "model": "v1B", "csi": 0.4}, {"lead_min": 0, "model": "Model1-B", "csi": 0.3}])
    g = PE.shadow_gate(c, d, m, e, mat)
    assert g["complete"] and g["pass"] is ok
    assert g["matmo"]["flag"] is True                                      # drop 0.1 > 0.05: flag, never a fail
    g2 = PE.shadow_gate(c, d[d["lead_min"] != 60], m, e, mat)
    assert not g2["complete"] and not g2["pass"]


def test_ci_rules():
    assert PE.ci_rule({"diff": -0.001, "lo": -0.002, "hi": -0.0001}, 0.0)["pass"] is True
    assert PE.ci_rule({"diff": -0.001, "lo": -0.002, "hi": 0.0}, 0.0)["pass"] is False


def test_prereg_guard(tmp_path):
    raw = yaml.safe_load(PE.CFG_PATH.read_text(encoding="utf-8"))
    assert PE.prereg_sha(raw) == PE.PREREG_SHA
    for path, val in ((("downsampling", "equivalence", "abs_csi_2p5_max"), 0.004),
                      (("shadow_gate", "events_7p6_every_frame", "pod_diff_min"), -0.03),
                      (("rounds_check", "max_share_of_cap"), 0.7), (("planning", "part_target_minutes"), 300)):
        r = copy.deepcopy(raw)
        d = r
        for k in path[:-1]:
            d = d[k]
        d[path[-1]] = val
        p = tmp_path / "model_p4e.yaml"
        p.write_text(yaml.safe_dump(r), encoding="utf-8")
        shutil.copy(PE.CODE_ROOT / "configs" / "model_v2.yaml", tmp_path / "model_v2.yaml")
        with pytest.raises(RuntimeError, match="pre-registered"):
            PE.load_cfg(path=p)
    r = copy.deepcopy(raw)
    r["race"]["claim_poll_seconds"] = 5                             # outside the pre-registered sections: allowed
    p = tmp_path / "ok.yaml"
    p.write_text(yaml.safe_dump(r), encoding="utf-8")
    assert PE.prereg_sha(r) == PE.PREREG_SHA


def test_smoke_units_and_plan_check():
    us = PE.units(SMOKE)
    ids = [u.id for u in us]
    assert ids[0] == "preflight" and ids[-3:] == ["cmp", "figs", "compare-done"]
    dry = [u for u in us if u.kind == "dry"]
    assert [u.id for u in dry] == ["dry-full-2024-07"] and dry[0].deps == ("preflight",)
    assert [u.id for u in us if u.kind == "nwp"] == ["nwp-2024-10", "nwp-2025-07", "nwp-2025-10"]
    fits = [u for u in us if u.kind == "fit"]
    got = {(u.params["spec"], u.params["season"], u.params["thr"]) for u in fits}
    assert got == {("m0", 2024, 2.5), ("m0", 2024, 7.6), ("m1", 2024, 2.5), ("m1", 2024, 7.6), ("m1", 2025, 2.5),
                   ("m1", 2025, 7.6), ("rnear", 2024, 2.5), ("rnear", 2024, 7.6), ("wout", 2024, 2.5),
                   ("wout", 2024, 7.6), ("m1ts", 2024, 7.6), ("v1b", 2025, 2.5), ("m1b", 2025, 2.5)}
    assert all(u.params["lead"] == 0 for u in fits)
    assert [u.id for u in us if u.kind == "base"] == ["base-s2024-lead0", "base-s2025-lead0"]
    cmp = {u.id: u for u in us}["cmp"]
    assert set(cmp.deps) == {u.id for u in us if u.kind in ("fit", "base", "dry")} | {"cmp-diag", "cmp-2stage"}
    icv = PE.inner_cv_plan_check(SMOKE)
    assert [r["variant"] for r in icv] == ["plain", "plain", "set_b"] and all(r["ok"] for r in icv)
    assert icv[2]["heldout_season"] == 2025 and icv[2]["block"] == "utc_day" and len(icv[2]["usable_blocks"]) == 6
    assert PD.VARIANTS == {"plain": False, "cut": True} and PD.OUT == "reports/p4d"    # env restored


def test_dataset_defs():
    assert PE.dataset_def("m0") == {"ring": False} and PE.dataset_def("m1") == {"ring": True}
    assert PE.dataset_def("m1ds10") == {"ring": True, "ds_p": 0.1}
    assert PE.dataset_def("m1ts") == {"ring": True, "two_stage": True}
    assert PE.dataset_def("rnear") == {"ring": True, "ring_subset": "r_near"}
    assert PE.dataset_def("wout") == {"ring": True, "w_out": True}
    assert PE.dataset_def("v1b") == {"ring": False, "set_b": True} and PE.dataset_def("m1b")["set_b"]
    assert PE.dataset_def("m1bts") == {"ring": True, "set_b": True, "two_stage": True}
    with pytest.raises(ValueError):
        PE.dataset_def("m2")


# ---------------------------------------------------------------- downsampling (inputs only)
def _frame(seed=1, T=40, dry_every=3):
    G = F.Grid()
    pix = np.array([(G.lat_idx_min + 10 + i) * 3600 + G.lon_idx_min + 10 + j for i in range(3) for j in range(3)])
    t0 = pd.date_range("2024-07-01", periods=T, freq="30min", tz="UTC")
    rng = np.random.default_rng(seed)
    feats = ["gc_k0", "gc_k1", "raw_k0", "nb_r1_max_k0", "ring_N_mean_k0", "upwind_mean", "cal_hour_sin", "lat"]
    meta = pd.DataFrame({"pixel_id": np.tile(pix, T), "t0": np.repeat(t0, len(pix))})
    X = rng.gamma(0.5, 2.0, (len(meta), len(feats))).astype("float32")
    dry_t = t0[::dry_every]
    rows = meta["t0"].isin(dry_t).to_numpy()
    X[np.ix_(rows, [0, 1, 2, 3, 4, 5])] = 0.0                    # every rain column 0 at the dry scenes
    meta["label_rate_mm_h"] = rng.gamma(0.5, 2.0, len(meta)).astype("float32")
    return meta, X, feats, set(dry_t)


def test_dry_scene_uses_inputs_only_and_whole_scenes():
    meta, X, feats, dry_t = _frame()
    assert PE.rain_columns(feats, CFG) == ["gc_k0", "gc_k1", "raw_k0", "nb_r1_max_k0", "ring_N_mean_k0", "upwind_mean"]
    d = PE.dry_scene_rows(meta, X, feats, CFG)
    assert set(meta.loc[d, "t0"]) == dry_t
    meta2 = meta.assign(label_rate_mm_h=99.0)                      # labels never change the dry definition
    assert (PE.dry_scene_rows(meta2, X, feats, CFG) == d).all()
    X2 = X.copy()                                                  # one pixel with rain makes the scene wet
    first = np.flatnonzero(meta["t0"] == sorted(dry_t)[0])[0]
    X2[first, 0] = 0.1
    assert not PE.dry_scene_rows(meta, X2, feats, CFG)[meta["t0"] == sorted(dry_t)[0]].any()
    X3 = X.copy()
    X3[first, 0] = np.nan                                          # missing input is never "dry"
    assert not PE.dry_scene_rows(meta, X3, feats, CFG)[first]


# D196 (user, 2026-11-23): NaN in motion-derived columns (adv_*, upwind_*) is dry-compatible ONLY when every raw rain
# input window of the row is non-NaN and exactly 0; NaN in any raw rain column keeps the scene "not dry".
MOTION = 5                                                         # feats[5] == "upwind_mean"
RAW = 2                                                            # feats[2] == "raw_k0"


def test_d196_motion_columns():
    _, _, feats, _ = _frame()
    assert PE.motion_columns(PE.rain_columns(feats, CFG), CFG) == ["upwind_mean"]


def test_d196_dry_scene_with_nan_motion_is_dropped():
    meta, X, feats, dry_t = _frame(T=400)
    rows = meta["t0"].isin(dry_t).to_numpy()
    X[rows, MOTION] = np.nan                                       # motion undefined when nothing rains
    drop, w, dry, st = PE.downsample(meta, X, feats, CFG, 0.1)
    assert set(meta.loc[dry, "t0"]) == dry_t                       # still dry scenes
    u = PE.keep_draw(meta["t0"], str(CFG["downsampling"]["seed"]))
    gone = rows & (u >= 0.1)
    assert gone.any() and drop[gone].all() and not drop[~rows].any()
    r, s = PE.dry_rows(X, feats, CFG)
    assert s["rows_dry_with_motion_nan"] == int(rows.sum()) and s["motion_columns"] == 1


def test_d196_one_nan_raw_value_keeps_scene():
    meta, X, feats, dry_t = _frame(T=400)
    rows = meta["t0"].isin(dry_t).to_numpy()
    X[rows, MOTION] = np.nan
    t = sorted(dry_t)[0]
    sc = (meta["t0"] == t).to_numpy()
    X[np.flatnonzero(sc)[0], RAW] = np.nan                         # one missing raw rain value in the scene
    drop, w, dry, _ = PE.downsample(meta, X, feats, CFG, 0.1)
    assert not dry[sc].any() and not drop[sc].any() and np.allclose(w[sc], 1.0)
    assert set(meta.loc[dry, "t0"]) == dry_t - {t}                 # other dry scenes unaffected


def test_d196_wet_scene_is_kept():
    meta, X, feats, dry_t = _frame(T=400)
    wet = (~meta["t0"].isin(dry_t)).to_numpy()
    X[wet, MOTION] = np.nan                                        # NaN motion does not make a wet scene dry
    drop, w, dry, _ = PE.downsample(meta, X, feats, CFG, 0.1)
    assert not dry[wet].any() and not drop[wet].any() and np.allclose(w[wet], 1.0)
    X2 = X.copy()                                                  # raw windows all 0 but motion > 0: not dry
    t = sorted(dry_t)[0]
    sc = (meta["t0"] == t).to_numpy()
    X2[np.flatnonzero(sc)[0], MOTION] = 0.3
    assert not PE.dry_scene_rows(meta, X2, feats, CFG)[sc].any()


def test_d196_dry_share_stats():
    meta, X, feats, dry_t = _frame(T=30, dry_every=3)
    st = PE.dry_share_stats(meta, X, feats, CFG)
    assert st["scenes"] == 30 and st["dry_scenes"] == len(dry_t) == 10
    assert st["dry_scene_share"] == pytest.approx(1 / 3) and st["dry_scene_row_share"] == pytest.approx(1 / 3)
    assert st["raw_columns"] == 5 and st["rows_dry"] == st["rows_raw_all_zero"] == 10 * 9


def test_downsample_keeps_whole_scenes_with_inverse_weight():
    meta, X, feats, dry_t = _frame(T=400)
    drop, w, dry, st = PE.downsample(meta, X, feats, CFG, 0.2)
    assert not (drop & ~dry).any()                                 # only dry scenes are dropped
    for t, g in meta.assign(drop=drop, w=w).groupby("t0"):
        assert g["drop"].nunique() == 1 and g["w"].nunique() == 1  # all pixels of a scene together
    kept_dry = dry & ~drop
    assert np.allclose(w[kept_dry], 5.0) and np.allclose(w[~dry], 1.0)
    share = kept_dry.sum() / dry.sum()
    assert 0.1 < share < 0.32 and st["dry_rows"] == int(dry.sum())
    d2, w2, _, _ = PE.downsample(meta, X, feats, CFG, 0.2)         # deterministic
    assert (d2 == drop).all() and (w2 == w).all()
    u1 = PE.keep_draw(meta["t0"], "p4e-dry-scene-v1")
    u2 = PE.keep_draw(meta["t0"].iloc[::-1].reset_index(drop=True), "p4e-dry-scene-v1")[::-1]
    assert np.allclose(u1, u2)                                     # depends on t0 only, not on row order


# ---------------------------------------------------------------- two-stage rain scene
def test_rain_scene_neighbourhood():
    G = F.Grid()
    pix = np.array([(G.lat_idx_min + 10 + i) * 3600 + G.lon_idx_min + 10 + j for i in range(5) for j in range(5)])
    t0 = pd.date_range("2024-07-01", periods=2, freq="30min", tz="UTC")
    meta = pd.DataFrame({"pixel_id": np.tile(pix, 2), "t0": np.repeat(t0, 25), "label_rate_mm_h": 0.0})
    centre = pix[12]                                               # (2, 2)
    meta.loc[(meta["pixel_id"] == centre) & (meta["t0"] == t0[0]), "label_rate_mm_h"] = 0.6
    r = PE.rain_scene(meta, CFG)
    first = meta["t0"] == t0[0]
    i, j = G.ij(meta["pixel_id"].to_numpy())
    i, j = i - i.min(), j - j.min()
    expect = first & (np.abs(i - 2) <= 1) & (np.abs(j - 2) <= 1)
    assert (r == expect.to_numpy()).all() and r.sum() == 9         # 3 x 3 around the wet pixel, same t0 only
    lab = meta["label_rate_mm_h"].to_numpy()
    assert ((lab >= 7.6) <= r).all()                               # {>= 7.6} is a subset of R (exact factorisation)


# ---------------------------------------------------------------- pre-registered rules
def _row(L, lo=-0.0002, hi=0.0003, bss=0.001, csi=-0.002, ece=0.001, c76=-0.001):
    return {"lead_min": L, "logloss_lo": lo, "logloss_hi": hi, "bss_clim_diff": bss, "csi_diff": csi,
            "ece_cal_diff": ece, "csi76_diff": c76}


@pytest.mark.parametrize("change,ok", [({}, True), ({"hi": 0.0006}, False), ({"lo": -0.00051}, False),
                                       ({"bss": -0.0031}, False), ({"csi": 0.0031}, False), ({"ece": 0.0021}, False),
                                       ({"c76": -0.0051}, False), ({"ece": -0.01}, True)])
def test_equivalence_rule(change, ok):
    eq = CFG["downsampling"]["equivalence"]
    rows = [_row(0), _row(30), _row(60, **change)]
    v = PE.equivalence_verdict(rows, eq)
    assert v["complete"] and v["pass"] is ok
    v = PE.equivalence_verdict([_row(0, c76=None)], eq)
    assert not v["complete"] and not v["pass"]


# ---------------------------------------------------------------- cost model and planning
def test_cost_model_reproduces_p4d():
    """lr 0.03 / ES 150 (P4D): ring 2.5 fit (3.62 M rows, 11.3 inner folds) ~ 738 s; measured mean 756 s."""
    c = copy.deepcopy(CFG)
    c["lightgbm"]["learning_rate"], c["inner_cv"]["early_stopping_rounds"], c["inner_cv"]["max_rounds"] = 0.03, 150, 3000
    m = PE.fit_minutes(c, "ring", 2.5, 3_623_424, 11.3) * 60
    assert 700 < m < 780
    new = PE.fit_minutes(CFG, "ring", 2.5, 3_623_424, 11.3) * 60
    assert new < 0.65 * m                                          # lr 0.06: about half the boosting work


def test_plan_estimate_and_parts():
    us = PE.units(SMOKE)
    e1, e2 = PE.plan_estimate(SMOKE, us, 1), PE.plan_estimate(SMOKE, us, 2)
    assert e1["parts"] == 1 and len(e2["part_minutes"]) == 2 and e2["max_part_minutes"] < e1["max_part_minutes"]
    assert e2["runner_minutes"] >= e1["runner_minutes"] - 1          # opening gates are waited for by every part
    for run, ds in (("A", None), ("B", 0.1), ("B", None)):
        u, um = PE.full_run_plan(CFG, run, ds)
        e = PE.choose_parts(CFG, u, um)
        assert e["max_part_minutes"] <= CFG["planning"]["part_target_minutes"]
    ua, uma = PE.full_run_plan(CFG, "A")
    assert sum(x.kind == "fit" for x in ua) == 54
    ub0, umb0 = PE.full_run_plan(CFG, "B")
    ub1, umb1 = PE.full_run_plan(CFG, "B", 0.1)
    assert sum(umb1.values()) < sum(umb0.values())


def test_lpt_balances():
    load = PE.lpt([("a", 10), ("b", 9), ("c", 8), ("d", 2), ("e", 1)], 2)
    assert sorted(load) == [13, 17]                                # greedy LPT (makespan <= 4/3 optimum)


# ---------------------------------------------------------------- driver
def test_run_id_depends_on_p4e_code():
    assert CT.run_id_for("p4e_smoke") != CT.run_id_for("p4d_smoke") != CT.run_id_for("p4e")
    assert "p4e" in CT.P4E_JOBS and "p4e" in CT.JOBS
    assert "src/models/p4e.py" in PE.CODE_FILES and "configs/model_p4e.yaml" in PE.CODE_FILES


def test_driver_plan_and_refusals(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HFL_P2B_EXCLUDE_SOURCES", "none")
    monkeypatch.setattr(CT, "ROOT", tmp_path / "ws")
    spec = f"local:{tmp_path / 's'}"
    assert C.cli(CT.main, ["--job", "p4e_smoke", "--store", spec, "--run-id", "x", "--plan"]) == 0
    ev = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.startswith("{")]
    plan = [e for e in ev if e["event"] == "plan"][0]
    est = [e for e in ev if e["event"] == "estimate"][0]
    assert plan["prereg_sha"] == PE.PREREG_SHA and est["within_target"]
    for extra in (["--final"], ["--write-oof-done"]):
        assert C.cli(CT.main, ["--job", "p4e_smoke", "--store", spec, "--run-id", "x", *extra]) == 2
    capsys.readouterr()
    assert C.cli(CT.main, ["--job", "p4e", "--store", spec, "--run-id", "y", "--part", "1/7", "--plan"]) == 0
    ev = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.startswith("{")]
    est = [e for e in ev if e["event"] == "estimate"][0]
    assert est["parts"] == 7 and est["proposed_parts"] == 7 and est["within_target"]
    assert C.cli(CT.main, ["--job", "p4e", "--store", spec, "--run-id", "y", "--race-drill"]) == 2
    st = CT.open_run_store(spec, "p4e_smoke")
    st.put("lake/v1/gold/met_training/table_v2_ext/x", b"1")               # table_v2*/ allowed
    with pytest.raises(C.KeyRefused):
        st.put("lake/v1/gold/met_training/table_A1_v1/x", b"1")
    with pytest.raises(C.KeyRefused):
        st.put("lake/v1/runs/p4d/x", b"1")
