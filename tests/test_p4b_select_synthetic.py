"""P4B — selection and OOF-report gates on SYNTHETIC CV outputs (no training, D94): sel-features, sel-setting,
oof-select (calibration, cuts, B-vs-A ship rule) and oof-report (ablation table, fold spread)."""
import json

import numpy as np
import pandas as pd
import pytest

from src.models import cloud_train as CT
from src.models import p4b as PB


def _oof(rng, n_days, lead, quality, shift=0.0):
    t0 = pd.date_range("2024-07-01", periods=n_days * 48, freq="30min", tz="UTC")
    pix = [1, 2, 3]
    T = np.repeat(t0, len(pix)); Pp = np.tile(pix, len(t0))
    z = rng.normal(size=len(T))
    y = (rng.random(len(T)) < 1 / (1 + np.exp(-(2 * z - 2.8)))).astype("int8")
    p = 1 / (1 + np.exp(-(quality * (2 * z - 2.8) + shift)))
    day = pd.DatetimeIndex(T).tz_convert("Asia/Ho_Chi_Minh").date.astype(str)
    fold = (pd.DatetimeIndex(T).day > n_days // 2).astype("int16")
    return pd.DataFrame({"pixel_id": Pp, "t0": T, "lead_min": np.int16(lead), "day": day, "fold": fold, "y": y,
                         "label_rate_mm_h": (y * 5).astype("float32"), "p_raw": p.astype("float32"),
                         "p_best": p.astype("float32"), "p_clim": np.full(len(T), y.mean(), "float32"),
                         "pers": np.where(rng.random(len(T)) < 0.5, y * 5.0, 0).astype("float32")})


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    monkeypatch.setattr(PB, "ROOT", tmp_path / "ws")
    cfg = PB.load_cfg(smoke=True)
    st = CT.open_run_store(f"local:{tmp_path / 'store'}", "p4b_smoke")
    c = PB.Ctx(cfg, st, "p4b_smoke", "t1", tmp_path / "m", print)
    rng = np.random.default_rng(5)
    quality = {"a": 0.5, "b": 0.55, "c": 0.7, "d": 0.75, "dm": 0.74, "g1": 0.8, "A24": 0.8, "B24": 0.85}
    sd = {"a": ("pilot", "p4a", False), "b": ("b", "p4a", False), "c": ("c", "p4a", False), "d": ("d", "p4a", False),
          "dm": ("dm", "p4a", False), "g1": ("d", "g1", False), "A24": ("d", "g1", True), "B24": ("B:d", "g1", True)}
    pre = c.run_files_prefix()
    for spec, q in quality.items():
        for L in (0, 30, 60):
            uid = PB.cv_id(spec, L, 2.5)
            o = _oof(np.random.default_rng(L), 20, L, q, shift=0.7)   # same rows/labels for every spec of a lead
            info = {"unit": uid, "spec": spec, "spec_def": {"fam": sd[spec][0], "setting": sd[spec][1],
                                                            "restricted": sd[spec][2]},
                    "table": "B1" if spec == "B24" else "A1", "lead_min": L, "threshold": 2.5, "features": ["gc_k0"],
                    "lgb_params": {}, "median_rounds": 100, "folds_capped_below_median": 0,
                    "folds": [{"fold": f, **PB.oof_metrics(o.y[o.fold == f], o.p_raw[o.fold == f], o.p_clim[o.fold == f],
                                                          o.pers[o.fold == f], 2.5)} for f in (0, 1)],
                    "pooled": PB.oof_metrics(o.y, o.p_raw, o.p_clim, o.pers, 2.5)}
            st.put(pre + f"reports/p4b/oof/{uid}.parquet", _bytes(o))
            st.put(pre + f"reports/p4b/cv/{uid}.json", json.dumps(info).encode())
    return c


def _bytes(df):
    import io
    b = io.BytesIO(); df.to_parquet(b, index=False)
    return b.getvalue()


def test_selection_and_report_gates(ctx):
    _, m1 = PB.gate_sel_features(ctx)
    assert m1["feature_set"] == "d"                         # dm is worse (quality 0.74 < 0.75)
    _, m2 = PB.gate_sel_setting(ctx)
    assert m2["chosen"] == "g1"
    rels, m3 = PB.gate_oof_select(ctx)
    sel = json.loads((PB.ROOT / "reports/p4b/select/selection.json").read_text(encoding="utf-8"))
    assert set(sel["models"]) == {f"{v}-lead{L}-thr2p5" for v in "AB" for L in (0, 30, 60)}
    assert all(m["calibration"] in ("raw", "platt", "isotonic") for m in sel["models"].values())
    assert sel["models"]["A-lead0-thr2p5"]["calibration"] != "raw"         # shift 0.7: mis-calibrated
    assert sel["ship_B"] is True and "otherwise" in sel["serving_rule"]
    for m in sel["models"].values():
        assert m["cut"]["rule"] in ("max_csi", "freq_bias_1.0") and 0 < m["cut"]["cut"] < 1
    rels, m4 = PB.gate_oof_report(ctx)
    abl = pd.read_csv(PB.ROOT / "reports/p4b/oof/ablation.csv")
    steps = abl[abl.lead_min == 0]["step"].tolist()
    assert steps == ["a", "b", "c", "d", "d+month", "e", "f", "g(A, NWP months)", "g", "2023: with", "2023: without"]
    assert (abl.loc[abl.step == "c", "d_logloss"] < 0).all()               # better synthetic quality -> lower
    spread = pd.read_csv(PB.ROOT / "reports/p4b/oof/fold_spread.csv")
    assert len(spread) == 8 * 3 and (spread["folds"] == 2).all()
    assert m4["oof_complete"] and m4["setting"] == "g1"
