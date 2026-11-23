"""P4E.0 smoke end to end on a SYNTHETIC table_v2 (no real data; tiny LightGBM, 2 threads), driven by
cloud_train.main exactly as on the runner: 2 parts (part 1 exits when its work is done, part 2 closes the run),
Gold reuse, single-writer commits, M0 / Model 1 / downsampled / two-stage fits, the comparison gate (rounds check,
equivalence, two-stage violation, dry share, cost) and COMPARE_DONE.json, then the extended race drill."""
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
from src.models import p4e as PE
from src.realtime import common as C

G = F.Grid()
BOX = np.array([(G.lat_idx_min + i) * 3600 + G.lon_idx_min + j for i in range(10, 20) for j in range(10, 18)])
CITY = np.sort(BOX[:47])
RING = np.sort(BOX[47:51])
RUN, JOB = "fx1", "p4e_smoke"


def _cfg():
    c = copy.deepcopy(PE.load_cfg(smoke=True))
    c["scope"]["day_ranges"] = [["2023-05-14", "2023-05-17"], ["2024-09-06", "2024-09-08"],
                                ["2024-10-15", "2024-10-17"], ["2025-07-14", "2025-07-16"], ["2025-10-06", "2025-10-07"]]
    c["lightgbm"] |= {"num_threads": 2, "min_data_in_leaf": 50}
    c["inner_cv"] |= {"max_rounds": 200, "early_stopping_rounds": 10}
    c["evaluation"] |= {"bootstrap_resamples": 50, "wettest_days": 2}
    c["race"] |= {"claim_poll_seconds": 0.05}
    return c


def _partition(rng, y, m, days):
    pix = np.r_[CITY, RING]
    t0 = pd.DatetimeIndex(np.concatenate([pd.date_range(pd.Timestamp(d, tz="UTC"), periods=48, freq="30min").values
                                          for d in days]), tz="UTC")
    lat0 = F.city_centre(CITY)
    st = F.pixel_static_columns(pix, CITY, lat0, ring=(0.3, 1.0)).set_index("pixel_id")
    lat, lon = G.latlon(pix)
    T, Pn = len(t0), len(pix)
    dry = rng.random(T) < 0.35                                                 # whole-domain dry scenes
    rows = []
    for L in (0,):
        r = rng.gamma(0.35, 5.0, (T, Pn)).astype("float32")
        r[dry] = 0.0
        gc = {k: np.clip(r * (1 - 0.1 * k) + rng.normal(0, 0.8, (T, Pn)), 0, None).astype("float32") for k in range(7)}
        for k in range(7):
            gc[k][dry] = 0.0
        lab = np.clip(r * rng.lognormal(0, 0.3, (T, Pn)), 0, None).astype("float32")
        df = pd.DataFrame({"pixel_id": np.tile(pix, T), "t0": np.repeat(t0, Pn), "lead_min": np.int16(L)})
        df["t0"] = df["t0"].astype("datetime64[us, UTC]")
        df["prediction_time"] = df["t0"] + pd.Timedelta(minutes=10)
        df["frame_start"] = df["t0"] + pd.Timedelta(minutes=L)
        df["target_time"] = df["frame_start"] + pd.Timedelta(minutes=30)
        df["label_rate_mm_h"] = lab.reshape(-1)
        for t in (2.5, 7.6):
            df[PB.thr_col(t)] = (df["label_rate_mm_h"] >= t).astype("int8")
        df["label_source"] = "imerg_late_v07" if (y, m) == (2025, 10) else "imerg_final_v07"
        df["season"] = np.int16(y)
        for k in range(7):
            df[f"gc_k{k}"] = gc[k].reshape(-1)
        df["nb_r1_mean_k0"] = (gc[0] * 0.8).reshape(-1)
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
    monkeypatch.setattr(PE, "load_cfg", lambda smoke=False, path=None, check_prereg=True: copy.deepcopy(cfg))
    gid = PE.gold_id(cfg)
    rng = np.random.default_rng(7)
    by_month = {}
    for d in BT2.build_days(cfg):
        by_month.setdefault((d.year, d.month), []).append(d)
    st = CT.open_run_store(f"local:{store}", JOB)
    for (y, m), days in by_month.items():                 # Gold already on R2 -> feat units must reuse it
        df = _partition(rng, y, m, days)
        assert not BT2.g2_violations(df, BT2.feature_columns(df.columns), PD.CFG_PATH)
        p = store / f"lake/v1/gold/met_training/table_v2/gid={gid}/season={y}/month={m:02d}/part.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p, index=False)
        p.with_name("part.meta.json").write_text("{}", encoding="utf-8")
    for uid in ("preflight", "g2"):                       # need real Silver / A1 Gold: covered by the cloud smoke
        st.put_json(CT.marker_key(JOB, RUN, uid), {"unit": uid, "outputs": {}, "synthetic": True})
    return cfg, store


def _r(rel):
    return PB.ROOT / rel


def _events(out):
    return [json.loads(x) for x in out.splitlines() if x.startswith("{")]


def test_smoke_end_to_end(run, tmp_path, capsys):
    cfg, store = run
    spec = f"local:{store}"
    base = ["--job", JOB, "--store", spec, "--run-id", RUN, "--wait-minutes", "0"]
    pre = f"lake/v1/runs/{JOB}/{RUN}/"
    st = CT.open_run_store(spec, JOB)
    # part 1: its work units, then it exits (closing gates not ready, no idle wait)
    assert C.cli(CT.main, base + ["--part", "1/2"]) == 0
    ev = _events(capsys.readouterr().out)
    assert [e for e in ev if e["event"] == "end"][0]["compare_complete"] is False
    assert "cmp" not in CT.done_units(st, JOB, RUN)
    # part 2: the rest, then cmp and compare-done
    assert C.cli(CT.main, base + ["--part", "2/2"]) == 0
    capsys.readouterr()
    done = CT.done_units(st, JOB, RUN)
    assert {u.id for u in PE.units(cfg)} <= done
    assert json.loads(st.get(pre + "COMPARE_DONE.json"))["compare_marker"].endswith("compare-done.done.json")
    assert json.loads(st.get(f"lake/v1/runs/{JOB}/latest.json")) == {"job": JOB, "run_id": RUN}
    # Gold reused, nothing written under the gold prefix by the run
    fm = json.loads(st.get(CT.marker_key(JOB, RUN, "feat-2023-05")))
    assert fm["metrics"]["reused"] is True and fm["outputs"] == {}
    # single writer: one claim generation per unit, no losers in a sequential run
    claims = [k for k in st.list(pre + "claims/")]
    assert claims and all(k.endswith("/g001.json") for k in claims)
    assert not st.list(pre + "scratch/")
    # fits
    f0 = json.loads(_r("reports/p4e/fit/fit-m0-s2024-lead0-thr2p5.json").read_text(encoding="utf-8"))
    f1 = json.loads(_r("reports/p4e/fit/fit-m1-s2024-lead0-thr2p5.json").read_text(encoding="utf-8"))
    assert f0["lgb_params"]["learning_rate"] == 0.06 and f0["inner_cv"]["max_rounds"] == 200
    assert f0["train_rows_ring"] == 0 and f1["train_rows_ring"] > 0
    assert [x["month"] for x in f1["inner_folds"]] == [202305, 202507]   # 2025-10 (Late) never trains
    ds = json.loads(_r("reports/p4e/fit/fit-m1ds10-s2024-lead0-thr2p5.json").read_text(encoding="utf-8"))
    d = ds["downsampling"]
    assert d["p"] == 0.1 and 0.2 < d["dry_share_training"] < 0.5 and d["rows_saved"] > 0
    assert ds["train_rows"] == d["training_rows_kept"] < f1["train_rows"]
    assert ds["train_weight_sum"] == pytest.approx(f1["train_rows"], rel=0.15)   # 1/p weights restore the mass
    ts = json.loads(_r("reports/p4e/fit/fit-m1ts-s2024-lead0-thr7p6.json").read_text(encoding="utf-8"))
    assert ts["two_stage"]["stage2_training_rows"] < ts["train_rows"] and ts["cut"]["rule"] == "freq_bias_1.5"
    o = pd.read_parquet(_r("reports/p4e/oof/fit-m1ts-s2024-lead0-thr7p6.parquet"))
    assert np.allclose(o["p_raw"], o["p_stage1"] * o["p_stage2"], atol=1e-6)
    b = pd.read_parquet(_r("reports/p4e/base/s2024_lead0.parquet"))
    assert len(o) == len(b) and b["pixel_id"].isin(CITY).all()                 # evaluation: city, never sampled
    # comparison gate
    ch = json.loads(_r("reports/p4e/compare/checks.json").read_text(encoding="utf-8"))
    assert ch["rounds_check_ok"] and len(ch["two_stage_violation"]) == 1
    assert 0 <= ch["two_stage_violation"][0]["violation_rate"] <= 1
    assert ch["equivalence_smoke_informative"]["complete"] is False             # no 7.6 downsampled unit in the smoke
    m = pd.read_csv(_r("reports/p4e/compare/metrics.csv"))
    assert set(m["model"]) == {"M0", "Model1", "Model1-ds10", "Model1-twostage"}
    df = pd.read_csv(_r("reports/p4e/compare/diffs.csv"))
    assert set(df["comparison"]) == {"Model1 - M0", "Model1-ds10 - Model1", "Model1-twostage - Model1"}
    assert (df["logloss_lo"] <= df["logloss_hi"]).all() and df["ece_cal_diff"].notna().all()
    cost = pd.read_csv(_r("reports/p4e/compare/cost.csv"))
    assert len(cost) == 6 and cost["seconds_actual"].notna().all() and (cost["work_grows"] > 0).all()
    # a fit re-run gives identical bytes (outputs carry no timings)
    u = {x.id: x for x in PE.units(cfg)}["fit-m1ds10-s2024-lead0-thr2p5"]
    before = {k: st.get(k) for k in st.list(pre + "files/reports/p4e/") if u.id in k}
    ctx = PE.Ctx(cfg, st, JOB, RUN, tmp_path / "m2", print)
    PE.run_fit(u, ctx)
    assert len(before) == 2
    for k, v in before.items():
        assert _r(k.split("/files/", 1)[1]).read_bytes() == v
    # extended race drill on the finished run (part 1 again, everything done): exactly one writer
    keys_before = {k: st.get(k) for k in st.list(pre + "files/") if "/drill/" not in k}
    assert C.cli(CT.main, base + ["--part", "1/2", "--race-drill"]) == 0
    end = [e for e in _events(capsys.readouterr().out) if e["event"] == "end"][0]
    dr = end["race_drill"]
    assert dr["ok"] and dr["loser_exit_code"] == 0 and sorted(dr["states"].values()) == ["won", "yielded"]
    assert {k: st.get(k) for k in st.list(pre + "files/") if "/drill/" not in k} == keys_before
    # a different part count is refused
    assert C.cli(CT.main, base + ["--part", "1/3"]) == 2
