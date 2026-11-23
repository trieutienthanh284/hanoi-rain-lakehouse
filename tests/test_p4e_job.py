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
    assert SMOKE["smoke"] and SMOKE["frames"]["lead_min"] == [0] and SMOKE["evaluation"]["loso_seasons"] == [2024]
    assert SMOKE["targets"]["thresholds_mm_h"] == [2.5, 7.6] and SMOKE["inner_cv"]["max_rounds"] == 1500
    assert PE.gold_id(SMOKE) != PE.gold_id(CFG)


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
    assert ids[0] == "preflight" and ids[-2:] == ["cmp", "compare-done"]
    fits = [u for u in us if u.kind == "fit"]
    assert {(u.params["spec"], u.params["thr"]) for u in fits} == {("m0", 2.5), ("m0", 7.6), ("m1", 2.5), ("m1", 7.6),
                                                                   ("m1ds10", 2.5), ("m1ts", 7.6)}
    assert all(u.params["season"] == 2024 and u.params["lead"] == 0 for u in fits)
    cmp = {u.id: u for u in us}["cmp"]
    assert set(cmp.deps) == {u.id for u in us if u.kind in ("fit", "base")}
    icv = PE.inner_cv_plan_check(SMOKE)
    assert len(icv) == 1 and icv[0]["ok"] and icv[0]["variant"] == "plain"
    assert PD.VARIANTS == {"plain": False, "cut": True} and PD.OUT == "reports/p4d"    # env restored
    with pytest.raises(NotImplementedError):
        PE.units(CFG)


def test_dataset_defs():
    assert PE.dataset_def("m0") == {"ring": False} and PE.dataset_def("m1") == {"ring": True}
    assert PE.dataset_def("m1ds10") == {"ring": True, "ds_p": 0.1}
    assert PE.dataset_def("m1ts") == {"ring": True, "two_stage": True}
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
    assert CT.run_id_for("p4e_smoke") != CT.run_id_for("p4d_smoke")
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
    st = CT.open_run_store(spec, "p4e_smoke")
    st.put("lake/v1/gold/met_training/table_v2_ext/x", b"1")               # table_v2*/ allowed
    with pytest.raises(C.KeyRefused):
        st.put("lake/v1/gold/met_training/table_A1_v1/x", b"1")
    with pytest.raises(C.KeyRefused):
        st.put("lake/v1/runs/p4d/x", b"1")
