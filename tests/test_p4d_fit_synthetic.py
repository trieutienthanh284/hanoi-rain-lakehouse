"""P4D.4-P4D.7 end to end on a SYNTHETIC table_v2 (no real data; tiny LightGBM, 2 threads): base rows, LOSO fits
with inner CV + Platt cross-fit, the selection gates, the comparison gates and COMPARE_DONE.json, driven by
cloud_train.main exactly as on the runner. Also: evaluation rows are never cut, 2025-10 (Late) is secondary only,
a fit unit re-run gives byte-identical outputs, a second invocation skips everything."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from src.gold import build_table_v2 as BT2
from src.gold import features as F
from src.models import cloud_train as CT
from src.models import p4b as PB
from src.models import p4d as PD
from src.realtime import common as C

G = F.Grid()
BOX = np.array([(G.lat_idx_min + i) * 3600 + G.lon_idx_min + j for i in range(10, 20) for j in range(10, 18)])
CITY = np.sort(BOX[:47])
RING = np.sort(BOX[47:51])
RUN = "fx1"


def _cfg():
    c = copy.deepcopy(PD.load_cfg(smoke=True))
    c["scope"]["day_ranges"] = [["2023-05-14", "2023-05-17"], ["2024-09-06", "2024-09-08"],
                                ["2024-10-15", "2024-10-17"], ["2025-05-14", "2025-05-17"],
                                ["2025-07-14", "2025-07-16"], ["2025-10-06", "2025-10-07"]]   # D185: >= 2 calendar months per LOSO training set
    c["targets"]["thresholds_mm_h"] = [0.5, 2.5, 7.6]
    c["model2"] = {**c["model2"], "grid": {"k": [2], "alpha": [0, 1], "beta": [4]}, "grid_thresholds": [2.5, 7.6]}
    c["lightgbm"] |= {"num_threads": 2, "min_data_in_leaf": 50}
    c["inner_cv"] |= {"max_rounds": 60, "early_stopping_rounds": 10}
    c["evaluation"] |= {"bootstrap_resamples": 50, "wettest_days": 2}
    return c


def _partition(rng, y, m, days, cfg):
    pix = np.r_[CITY, RING]
    t0 = pd.DatetimeIndex(np.concatenate([pd.date_range(pd.Timestamp(d, tz="UTC"), periods=48, freq="30min").values
                                          for d in days]), tz="UTC")
    rows = []
    lat0 = F.city_centre(CITY)
    st = F.pixel_static_columns(pix, CITY, lat0, ring=(0.3, 1.0)).set_index("pixel_id")
    lat, lon = G.latlon(pix)
    for L in (0, 30, 60):
        T, Pn = len(t0), len(pix)
        r = rng.gamma(0.35, 4.0, (T, Pn)).astype("float32")                    # latent rain at the frame
        gc = {k: np.clip(r * (1 - 0.1 * k) + rng.normal(0, 0.8, (T, Pn)), 0, None).astype("float32") for k in range(7)}
        lab = np.clip(r * rng.lognormal(0, 0.35, (T, Pn)), 0, None).astype("float32")
        df = pd.DataFrame({"pixel_id": np.tile(pix, T), "t0": np.repeat(t0, Pn), "lead_min": np.int16(L)})
        df["t0"] = df["t0"].astype("datetime64[us, UTC]")
        df["prediction_time"] = df["t0"] + pd.Timedelta(minutes=10)
        df["frame_start"] = df["t0"] + pd.Timedelta(minutes=L)
        df["target_time"] = df["frame_start"] + pd.Timedelta(minutes=30)
        df["label_rate_mm_h"] = lab.reshape(-1)
        for t in (0.5, 2.5, 7.6):
            df[PB.thr_col(t)] = (df["label_rate_mm_h"] >= t).astype("int8")
        df["label_source"] = "imerg_late_v07" if (y, m) == (2025, 10) else "imerg_final_v07"
        df["season"] = np.int16(y)
        for k in range(7):
            df[f"gc_k{k}"] = gc[k].reshape(-1)
        df["nb_r1_mean_k0"] = (gc[0] * 0.8 + 0.1).reshape(-1)
        hour = df["t0"].dt.tz_convert("Asia/Ho_Chi_Minh").dt.hour
        df["cal_hour_sin"] = np.sin(2 * np.pi * hour / 24).astype("float32")
        df["cal_month"] = np.float32(m)
        df["lat"] = np.tile(lat, T).astype("float32"); df["lon"] = np.tile(lon, T).astype("float32")
        df["dist_to_city_center_km"] = st.loc[df["pixel_id"], "dist_to_city_center_km"].to_numpy()
        df["is_city"] = st.loc[df["pixel_id"], "is_city"].to_numpy()
        df["ring_coverage"] = st.loc[df["pixel_id"], "ring_coverage"].to_numpy()
        df["gsmap__event_time"] = df["t0"]
        df["gsmap__available_time"] = df["t0"] + pd.Timedelta(minutes=5)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setenv("HFL_P2B_EXCLUDE_SOURCES", "none")
    work, store = tmp_path / "ws", tmp_path / "store"
    monkeypatch.setattr(PB, "ROOT", work)
    monkeypatch.setattr(CT, "ROOT", work)
    cfg = _cfg()
    monkeypatch.setattr(PD, "load_cfg", lambda smoke=False, path=None: copy.deepcopy(cfg))
    gid = PD.gold_id(cfg)
    rng = np.random.default_rng(11)
    by_month = {}
    for d in BT2.build_days(cfg):
        by_month.setdefault((d.year, d.month), []).append(d)
    st = CT.open_run_store(f"local:{store}", "p4d_smoke")
    for (y, m), days in by_month.items():
        df = _partition(rng, y, m, days, cfg)
        v = BT2.g2_violations(df, BT2.feature_columns(df.columns), PD.CFG_PATH)
        assert not v, v[:2]
        p = store / f"lake/v1/gold/met_training/table_v2/gid={gid}/season={y}/month={m:02d}/part.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p, index=False)
    for uid in ["preflight", "g2"] + [f"feat-{y}-{m:02d}" for y, m in by_month]:  # Gold stages done (other test)
        st.put_json(CT.marker_key("p4d_smoke", RUN, uid), {"unit": uid, "outputs": {}, "synthetic": True})
    return cfg, store, gid


def _r(rel):
    return PB.ROOT / rel


def test_end_to_end(run, tmp_path, capsys):
    cfg, store, gid = run
    spec = f"local:{store}"
    assert C.cli(CT.main, ["--job", "p4d_smoke", "--store", spec, "--run-id", RUN, "--part", "1/1"]) == 0
    pre = f"lake/v1/runs/p4d_smoke/{RUN}/"
    st = CT.open_run_store(spec, "p4d_smoke")
    done = CT.done_units(st, "p4d_smoke", RUN)
    assert {u.id for u in PD.units(cfg)} <= done
    cd = json.loads(st.get(pre + "COMPARE_DONE.json"))
    assert cd["compare_marker"].endswith("compare-done.done.json")
    assert json.loads(st.get(pre + "summary.json"))["stage"] == "P4D.7 comparison complete"
    # selection records
    s1a = json.loads(_r("reports/p4d/select/m1a.json").read_text(encoding="utf-8"))
    s1b = json.loads(_r("reports/p4d/select/m1b.json").read_text(encoding="utf-8"))
    s2 = json.loads(_r("reports/p4d/select/m2.json").read_text(encoding="utf-8"))
    assert s1a["kept"] == (s1a["mean_logloss_cal_new"] < s1a["mean_logloss_cal_old"])
    assert s1b["previous"] == ("m1a" if s1a["kept"] else "m0") and s1b["chain_end"] in ("m0", "m1a", "m1b")
    assert s2["chosen"] in ("g1", "g2") and set(s2["model2_spec_by_threshold"]) == {"0p5", "2p5", "7p6"}
    assert s2["model2_spec_by_threshold"]["0p5"] == "m2"
    lo, hi = s1a["frames_pooled_diff"]["lo"], s1a["frames_pooled_diff"]["hi"]
    assert lo <= s1a["frames_pooled_diff"]["diff"] <= hi
    # fit records: Platt + inner cut, training rows Final only, never 2025-10, cut honoured
    f = json.loads(_r("reports/p4d/fit/fit-m1a-s2024-lead0-thr2p5.json").read_text(encoding="utf-8"))
    assert f["platt"]["method"] == "platt" and f["cut"]["rule"] == "max_csi" and f["dataset"]["cut"]
    assert [x["month"] for x in f["inner_folds"]] == [202305, 202505, 202507]
    f0 = json.loads(_r("reports/p4d/fit/fit-m0-s2024-lead0-thr2p5.json").read_text(encoding="utf-8"))
    assert f["train_rows"] < f0["train_rows"] and f0["train_rows_ring"] == 0
    h = json.loads(_r("reports/p4d/fit/fit-m0-s2025-lead0-thr7p6.json").read_text(encoding="utf-8"))
    assert h["cut"]["rule"] == "freq_bias_1.5" and 202510 not in [x["month"] for x in h["inner_folds"]]
    g = json.loads(_r("reports/p4d/fit/fit-m2-g2-s2023-lead30-thr2p5.json").read_text(encoding="utf-8"))
    assert g["dataset"]["weights"] == {"k": 2.0, "alpha": 1.0, "beta": 4.0} and not g["dataset"]["cut"]
    assert g["train_weight_sum"] > g["train_rows"]
    # evaluation rows: full blocks (cut days included), city pixels only; 2025-10 = secondary
    b = pd.read_parquet(_r("reports/p4d/base/s2025_lead0.parquet"))
    assert set(b["role"]) == {"eval", "secondary"} and set(b.loc[b["role"] == "secondary", "label_source"]) == {"imerg_late"}
    assert b["pixel_id"].isin(CITY).all() and b["pixel_id"].nunique() == 47
    assert (pd.to_datetime(b["t0"]).dt.day.isin([14, 15]) & (b["role"] == "eval")).any()   # cut days evaluated
    o = pd.read_parquet(_r("reports/p4d/oof/fit-m1a-s2025-lead0-thr2p5.parquet"))
    assert len(o) == len(b)
    # comparison tables
    m = pd.read_csv(_r("reports/p4d/compare/metrics.csv"))
    ev = m[(m["period"] == "eval") & m["model"].isin(["M0", "Model1", "Model2"])]
    assert len(ev) == 3 * 3 * 3 and ev["logloss_cal"].notna().all() and ev["ece_cal"].notna().all()
    assert (m["model"] == "persistence").any()
    sec = pd.read_csv(_r("reports/p4d/compare/secondary_metrics.csv"))
    assert len(sec) > 0 and (sec["period"] == "secondary").all()
    d = pd.read_csv(_r("reports/p4d/compare/diffs.csv"))
    assert set(d["comparison"]) == {"Model1 - M0", "Model2 - M0", "Model2 - Model1"}
    assert (d["logloss_lo"] <= d["logloss_hi"]).all()
    hm = pd.read_csv(_r("reports/p4d/compare/halfmonth.csv"))
    assert {"05-1", "05-2", "09-1", "10-1", "10-2"} <= set(hm["half_month"])
    evs = pd.read_csv(_r("reports/p4d/compare/events.csv"))
    assert set(evs["event"]) == {"yagi", "matmo"}
    assert set(evs.loc[evs["event"] == "matmo", "label_source"]) == {"imerg_late"}
    assert set(evs.loc[evs["event"] == "matmo", "role"]) == {"secondary"}
    wd = pd.read_csv(_r("reports/p4d/compare/wettest_days.csv"))
    assert set(wd["heldout_season"]) == {2023, 2024, 2025} and not wd["day"].str.startswith("2025-10").any()
    # outputs carry no timings: a fresh re-run of one fit unit gives identical bytes
    u = {x.id: x for x in PD.units(cfg)}["fit-m1b-s2023-lead30-thr2p5"]
    before = {k: st.get(k) for k in st.list(pre + "files/reports/p4d/") if "fit-m1b-s2023-lead30-thr2p5" in k}
    for p_ in (_r("reports/p4d/oof/fit-m1b-s2023-lead30-thr2p5.parquet"),
               _r("reports/p4d/fit/fit-m1b-s2023-lead30-thr2p5.json")):
        p_.unlink()
    ctx = PD.Ctx(cfg, st, "p4d_smoke", RUN, tmp_path / "m2", print)
    PD.run_fit(u, ctx)
    assert len(before) == 2
    for k, v in before.items():
        assert _r(k.split("/files/", 1)[1]).read_bytes() == v
    # a second invocation finds everything done: nothing re-run, nothing uploaded
    capsys.readouterr()
    keys_before = st.list(pre)
    assert C.cli(CT.main, ["--job", "p4d_smoke", "--store", spec, "--run-id", RUN, "--part", "1/1"]) == 0
    end = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.startswith('{"event": "end"')][0]
    assert end["finished_now"] == 0 and st.list(pre) == keys_before        # only latest.json (outside pre) rewritten
    assert C.cli(CT.main, ["--job", "p4d_smoke", "--store", spec, "--run-id", RUN, "--part", "1/2"]) == 2
