"""P4F.1 serving path: load_v2_candidate() / predict_v2_candidate() on synthetic rows and tiny boosters; v1 functions
unchanged; oracle features can never be served."""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from src.models import predict as P

FEATS = ["gc_k0", "nwp_gfs_near_h0", "nwp_conv_gfs_cape_near"]


def _pack(root, model="model2", feats=FEATS, run="r1", leads=(0, 30), thrs=(0.5, 2.5, 7.6)):
    rng = np.random.default_rng(0)
    X = rng.random((400, len(feats))).astype("float32")
    for L in leads:
        for t in thrs:
            y = (X[:, 0] * 10 >= t).astype(int)
            b = lgb.train({"objective": "binary", "verbose": -1, "num_threads": 1, "min_data_in_leaf": 10},
                          lgb.Dataset(X, y), num_boost_round=5)
            d = root / model / f"lead{L}_thr{P.kstr(t)}"
            d.mkdir(parents=True)
            (d / "model.txt").write_text(b.model_to_string(), encoding="utf-8")
            meta = {"model": model, "lead_min": L, "threshold": t, "features": feats, "run_id": run,
                    "platt": {"method": "platt", "a": 1.0, "b": 0.0}, "cut": {"cut": 0.5}}
            (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def _rows(n=50):
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"pixel_id": np.arange(n), "t0": pd.Timestamp("2025-07-01", tz="UTC"), "lead_min": 0})
    for c in FEATS:
        df[c] = rng.random(n).astype("float32")
    return pd.concat([df, df.assign(lead_min=30)], ignore_index=True)


def test_load_and_predict(tmp_path):
    _pack(tmp_path)
    mv = P.load_v2_candidate("model2", root=tmp_path)
    assert mv.leads == [0, 30] and mv.thresholds == [0.5, 2.5, 7.6] and mv.run_id == "r1"
    out = P.predict_v2_candidate(_rows(), mv)
    assert len(out) == 100 and {"p_raw_2p5", "p_2p5", "yhat_7p6", "rain_class_name"} <= set(out.columns)
    assert ((out["yhat_7p6"] == 1) <= (out["yhat_2p5"] == 1)).all()          # nested heavy alarm
    assert "candidate" in out.attrs["model_version"]
    with pytest.raises(KeyError):
        P.predict_v2_candidate(_rows().drop(columns=["nwp_conv_gfs_cape_near"]), mv)


def test_refusals(tmp_path):
    with pytest.raises(ValueError):
        P.load_v2_candidate("v1", root=tmp_path)
    with pytest.raises(FileNotFoundError):
        P.load_v2_candidate("model1b", root=tmp_path)
    _pack(tmp_path / "a", feats=FEATS + ["oracle_gc_target"])
    with pytest.raises(ValueError, match="oracle"):
        P.load_v2_candidate("model2", root=tmp_path / "a")
    _pack(tmp_path / "b")
    d = tmp_path / "b" / "model2" / "lead0_thr2p5" / "meta.json"
    m = json.loads(d.read_text()); m["lead_min"] = 30; d.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="does not match"):
        P.load_v2_candidate("model2", root=tmp_path / "b")


def test_v1_default_unchanged():
    assert P.load.__doc__ and "v2_candidate" in P.load.__doc__
    with pytest.raises(ValueError):
        P.load("v3")
