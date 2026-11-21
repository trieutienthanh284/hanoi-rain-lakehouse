"""P4B.7 — src/models/predict.py v1 loader / predictor on synthetic models (stub boosters, no LightGBM training,
no real Gold data; D94). Two optional checks read the installed models/v1 and models/pilot_v0 files (metadata,
and model loading + inference on random feature rows) and are skipped when those files are absent (e.g. CI)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models import predict as P

ROOT = Path(__file__).resolve().parents[1]
RULE = ("use the set-B model when every nwp_* input is present and built from the previous_day1 run "
        "(issue <= prediction_time - 22 h, available_time <= prediction_time); otherwise (NWP missing or stale) "
        "use the set-A model")
FA = ["gc_k0", "gc_k1"]
FB = FA + ["nwp_gfs_near_h0", "nwp_ecmwf_near_h0"]
CUT = {0.5: 0.30, 2.5: 0.50, 7.6: 0.70}


class Stub:
    """p = clip(gc_k0 / 10 + offset): set A offset 0, set B offset +0.05 (so the set used is visible)."""
    def __init__(self, path: Path):
        self.off = json.loads((path.parent / "stub.json").read_text())["offset"]

    def predict(self, X):
        return np.clip(np.asarray(X)[:, 0] / 10.0 + self.off, 0.0, 1.0)


def _write_models(root: Path, ship_B=True, heavy="nested", run_id="r1", skip=()):
    for v in ("A", "B"):
        for L in P.V1_LEADS:
            for t in P.V1_THRESHOLDS:
                if (v, L, t) in skip:
                    continue
                d = root / v / f"lead{L}_thr{P.kstr(t)}"
                d.mkdir(parents=True)
                meta = {"variant": v, "lead_min": L, "threshold_mm_h": t, "features": FA if v == "A" else FB,
                        "cut": {"rule": "max_csi", "cut": CUT[t]}, "heavy_rule": heavy if t == 7.6 else None,
                        "ship_B": ship_B, "run_id": run_id, "calibration": "raw",
                        "serving_rule": RULE if v == "B" else "set A (fallback when NWP inputs are missing or stale)"}
                (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
                (d / "calibration.json").write_text(json.dumps({"method": "raw"}), encoding="utf-8")
                (d / "stub.json").write_text(json.dumps({"offset": 0.0 if v == "A" else 0.05}))
                (d / "model.txt").write_text("stub")
    return root


def _rows(gc0, t0="2025-07-21 06:00"):
    n = len(gc0)
    return pd.DataFrame({"pixel_id": np.arange(n), "t0": pd.Timestamp(t0, tz="UTC"),
                         "gc_k0": np.asarray(gc0, "float32"), "gc_k1": np.zeros(n, "float32")})


def _nwp(f, leads=P.V1_LEADS, gfs=1.0, ecmwf=1.0, issue_age_h=30.0, avail_age_h=1.0):
    out = []
    for L in leads:
        n = f[["pixel_id", "t0"]].copy()
        n["lead_min"] = L
        n["nwp_gfs_near_h0"], n["nwp_ecmwf_near_h0"] = gfs, ecmwf
        pt = n["t0"] + pd.Timedelta(minutes=10)
        for s in ("gfs", "ecmwf"):
            n[f"nwp_{s}__issue_time"] = pt - pd.Timedelta(hours=issue_age_h)
            n[f"nwp_{s}__available_time"] = pt - pd.Timedelta(hours=avail_age_h)
        out.append(n)
    return pd.concat(out, ignore_index=True)


@pytest.fixture
def mv(tmp_path):
    return P.load_v1(_write_models(tmp_path / "v1"), booster_loader=Stub)


def test_load_v1_reads_serving_rule(mv):
    assert mv.ship_B and mv.run_id == "r1" and mv.nwp_min_age_h == 22.0
    assert len(mv.models) == 18 and mv.models[("B", 60, 7.6)].heavy_rule == "nested"


def test_load_v1_refuses_inconsistent_models(tmp_path):
    root = _write_models(tmp_path / "a")
    m = root / "B" / "lead30_thr2p5" / "meta.json"
    d = json.loads(m.read_text()); d["run_id"] = "other"; m.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="mixed runs"):
        P.load_v1(root, booster_loader=Stub)
    root2 = _write_models(tmp_path / "b", skip={("B", 0, 0.5)})
    with pytest.raises(FileNotFoundError, match="ship_B"):
        P.load_v1(root2, booster_loader=Stub)
    root3 = _write_models(tmp_path / "c", skip={("A", 0, 0.5)})
    with pytest.raises(FileNotFoundError, match="set-A"):
        P.load_v1(root3, booster_loader=Stub)


def test_set_A_only_when_not_shipped(tmp_path):
    mv = P.load_v1(_write_models(tmp_path / "v1", ship_B=False), booster_loader=Stub)
    f = _rows([5.0, 5.0])
    out = P.predict_v1(f, mv, nwp=_nwp(f))
    assert (out["model_set"] == "A").all() and np.allclose(out["p_2p5"], 0.5)


def test_set_B_when_nwp_present_and_fallback_when_missing(mv):
    f = _rows([5.0, 5.0, 5.0])
    n = _nwp(f)
    n.loc[(n["pixel_id"] == 1), "nwp_gfs_near_h0"] = np.nan            # one input missing -> A
    out = P.predict_v1(f, mv, nwp=n)
    assert len(out) == 3 * len(f)
    for L in P.V1_LEADS:
        o = out[out["lead_min"] == L].set_index("pixel_id")
        assert list(o["model_set"]) == ["B", "A", "B"]
        np.testing.assert_allclose(o["p_2p5"].to_numpy(), [0.55, 0.50, 0.55], rtol=1e-6)
    # no NWP frame at all -> set A everywhere
    assert (P.predict_v1(f, mv)["model_set"] == "A").all()


def test_stale_or_early_nwp_falls_back_to_A(mv):
    f = _rows([5.0])
    assert (P.predict_v1(f, mv, nwp=_nwp(f, issue_age_h=22.0))["model_set"] == "B").all()   # boundary: allowed
    assert (P.predict_v1(f, mv, nwp=_nwp(f, issue_age_h=21.9))["model_set"] == "A").all()   # newer than previous day
    assert (P.predict_v1(f, mv, nwp=_nwp(f, avail_age_h=-0.1))["model_set"] == "A").all()   # not yet available
    n = _nwp(f); n["nwp_ecmwf__issue_time"] = pd.NaT
    assert (P.predict_v1(f, mv, nwp=n)["model_set"] == "A").all()
    n = _nwp(f).drop(columns=[c for c in _nwp(f).columns if "__" in c])                     # no group times
    assert (P.predict_v1(f, mv, nwp=n)["model_set"] == "B").all()


def test_nwp_joined_per_lead(mv):
    f = _rows([5.0])
    n = _nwp(f)
    n.loc[n["lead_min"] == 30, "nwp_ecmwf_near_h0"] = np.nan
    out = P.predict_v1(f, mv, nwp=n).set_index("lead_min")
    assert out.loc[0, "model_set"] == "B" and out.loc[30, "model_set"] == "A" and out.loc[60, "model_set"] == "B"


def test_cuts_nested_heavy_rule_and_classes(mv):
    # set A: p = gc_k0/10 for every threshold (stub); cuts 0.30 / 0.50 / 0.70
    f = _rows([0.0, 3.5, 6.0, 8.0])
    out = P.predict_v1(f, mv, leads=[0])
    assert list(out["yhat_0p5"]) == [0, 1, 1, 1]
    assert list(out["yhat_2p5"]) == [0, 0, 1, 1]
    assert list(out["yhat_7p6"]) == [0, 0, 0, 1]
    assert list(out["rain_class"]) == [0, 1, 2, 3]
    assert list(out["rain_class_name"]) == ["no_rain", "light", "moderate", "heavy"]
    assert out.attrs["note"].endswith("not flood prediction.")


def test_nested_rule_blocks_heavy_below_primary_cut(tmp_path):
    root = _write_models(tmp_path / "v1")
    for L in P.V1_LEADS:                                  # 2.5 cut above the 7.6 cut: nested must veto
        m = root / "A" / f"lead{L}_thr2p5" / "meta.json"
        d = json.loads(m.read_text()); d["cut"]["cut"] = 0.9; m.write_text(json.dumps(d))
    mv = P.load_v1(root, booster_loader=Stub)
    out = P.predict_v1(_rows([8.0]), mv, leads=[0])
    assert out.loc[0, "yhat_7p6"] == 0 and out.loc[0, "rain_class"] == 1
    root2 = _write_models(tmp_path / "v1b", heavy="bias_1.5")
    for L in P.V1_LEADS:
        m = root2 / "A" / f"lead{L}_thr2p5" / "meta.json"
        d = json.loads(m.read_text()); d["cut"]["cut"] = 0.9; m.write_text(json.dumps(d))
    out2 = P.predict_v1(_rows([8.0]), P.load_v1(root2, booster_loader=Stub), leads=[0])
    assert out2.loc[0, "yhat_7p6"] == 1 and out2.loc[0, "rain_class"] == 3


def test_gold_style_rows_with_lead_column(mv):
    f = pd.concat([_rows([5.0, 9.0]).assign(lead_min=L) for L in P.V1_LEADS], ignore_index=True)
    n = _nwp(_rows([5.0, 9.0]))
    g = f.merge(n, on=["pixel_id", "t0", "lead_min"])
    out = P.predict_v1(g, mv)
    assert len(out) == 6 and (out["model_set"] == "B").all()
    assert (out["frame_end"] - out["frame_start"] == pd.Timedelta(minutes=30)).all()


def test_missing_feature_raises(mv):
    with pytest.raises(KeyError, match="features missing"):
        P.predict_v1(_rows([1.0]).drop(columns=["gc_k1"]), mv)


def test_apply_calibration_matches_p4b():
    from src.models import p4b as PB
    p = np.linspace(0, 1, 11)
    for cal in ({"method": "raw"}, {"method": "platt", "a": 1.3, "b": -0.2},
                {"method": "isotonic", "x": [0, 0.5, 1], "y": [0.1, 0.4, 0.9]}):
        np.testing.assert_allclose(P.apply_calibration(cal, p), PB.apply_calibrator(cal, p))


def test_load_switch():
    with pytest.raises(ValueError):
        P.load("v2")


V1_REAL = ROOT / "models" / "v1"


@pytest.mark.skipif(not (V1_REAL / "B" / "lead0_thr2p5" / "meta.json").is_file(), reason="models/v1 not installed")
def test_installed_v1_metadata():
    mv = P.load_v1(booster_loader=lambda p: None)          # metadata only
    assert mv.run_id == "c0df12477947" and mv.ship_B and mv.nwp_min_age_h == 22.0
    assert all(mv.models[("A", L, 7.6)].heavy_rule == "nested" for L in P.V1_LEADS)
    assert all(m.calibration["method"] == "raw" for m in mv.models.values())
    for L in P.V1_LEADS:
        a, b = set(mv.models[("A", L, 2.5)].features), set(mv.models[("B", L, 2.5)].features)
        assert a < b and all(c.startswith("nwp_") for c in b - a)


@pytest.mark.skipif(not (V1_REAL / "B" / "lead0_thr2p5" / "model.txt").is_file(), reason="models/v1 not installed")
def test_installed_v1_and_v0_load_and_predict_on_random_rows():
    pytest.importorskip("lightgbm")
    mv = P.load("v1")
    feats = sorted({c for m in mv.models.values() for c in m.features})
    rng = np.random.default_rng(0)
    f = pd.DataFrame(rng.gamma(0.5, 2.0, size=(4, len(feats))).astype("float32"), columns=feats)
    f.insert(0, "pixel_id", np.arange(4)); f.insert(1, "t0", pd.Timestamp("2025-07-21 06:00", tz="UTC"))
    f.loc[2:, [c for c in feats if c.startswith("nwp_")]] = np.nan     # rows 2, 3 -> set-A fallback
    out = P.predict_v1(f, mv)
    assert len(out) == 12 and list(out[out["lead_min"] == 0]["model_set"]) == ["B", "B", "A", "A"]
    for k in ("0p5", "2p5", "7p6"):
        assert out[f"p_{k}"].between(0, 1).all()
    if (ROOT / "models" / "pilot_v0" / "predict_manifest.json").is_file():
        v0 = P.load("v0")
        assert v0.name == "pilot_v0_A_no2023" and len(v0.models) == 9
