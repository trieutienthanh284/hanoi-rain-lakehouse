"""P4E smoke end to end on a SYNTHETIC table_v2 + synthetic Open-Meteo Silver (no real data; tiny LightGBM, 2 threads),
driven by cloud_train.main exactly as on the runner: 2 parts (LPT assignment), Gold reuse, set-B NWP columns built
from the Open-Meteo points (option A, D198), single-writer commits, M0 / Model 1 / R-near / W-out / two-stage /
v1 (B) / Model 1-B / set-B two-stage fits, the gates cmp-diag / cmp-2stage / cmp (rounds check, propose / use rules,
shadow gate, cost, dry share of every listed month), figures and COMPARE_DONE.json, then the extended race drill."""
import copy
import json
from datetime import timedelta

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
RING = np.sort(BOX[47:])                     # 33 ring pixels: some within Chebyshev 2 of the city, some not
POINTS = BOX[::7]                            # synthetic Open-Meteo points (subset of the box)
RUN, JOB = "fx1", "p4e_smoke"


def _r_near():
    ci, cj = G.ij(CITY)
    ri, rj = G.ij(RING)
    d = np.maximum(np.abs(ri[:, None] - ci[None, :]), np.abs(rj[:, None] - cj[None, :])).min(1)
    return [int(x) for x in RING[d <= 2]]


def _cfg():
    c = copy.deepcopy(PE.load_cfg(smoke=True))
    c["scope"]["day_ranges"] = [["2023-05-14", "2023-05-17"], ["2024-09-06", "2024-09-08"],
                                ["2024-10-15", "2024-10-17"], ["2025-07-14", "2025-07-16"], ["2025-10-06", "2025-10-07"]]
    c["lightgbm"] |= {"num_threads": 2, "min_data_in_leaf": 50}
    c["inner_cv"] |= {"max_rounds": 200, "early_stopping_rounds": 10}
    c["evaluation"] |= {"bootstrap_resamples": 50, "wettest_days": 2}
    c["race"] |= {"claim_poll_seconds": 0.05}
    rn = _r_near()
    c["diagnostics"] |= {"r_near_pixels": rn, "r_near_count": len(rn)}
    c["fits"] = [{"spec": "m0", "folds": [2024], "thr": [2.5, 7.6]},
                 {"spec": "m1", "folds": [2024, 2025], "thr": [2.5, 7.6]},
                 {"spec": "rnear", "folds": [2024], "thr": [2.5, 7.6]},
                 {"spec": "wout", "folds": [2024], "thr": [2.5, 7.6]},
                 {"spec": "m1ts", "folds": [2024], "thr": [7.6]},
                 {"spec": "v1b", "folds": [2025], "thr": [2.5, 7.6]},
                 {"spec": "m1b", "folds": [2025], "thr": [2.5, 7.6]},
                 {"spec": "m1bts", "folds": [2025], "thr": [7.6]}]
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
    r = rng.gamma(0.35, 5.0, (T, Pn)).astype("float32")
    r[dry] = 0.0
    gc = {k: np.clip(r * (1 - 0.1 * k) + rng.normal(0, 0.8, (T, Pn)), 0, None).astype("float32") for k in range(7)}
    for k in range(7):
        gc[k][dry] = 0.0
    lab = np.clip(r * rng.lognormal(0, 0.3, (T, Pn)), 0, None).astype("float32")
    df = pd.DataFrame({"pixel_id": np.tile(pix, T), "t0": np.repeat(t0, Pn), "lead_min": np.int16(0)})
    df["t0"] = df["t0"].astype("datetime64[us, UTC]")
    df["prediction_time"] = df["t0"] + pd.Timedelta(minutes=10)
    df["frame_start"] = df["t0"]
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
    return df


def _openmeteo(rng, d):
    """One Silver day of synthetic Open-Meteo previous_day1 runs (hourly, two models) at the POINTS."""
    lat, lon = G.latlon(POINTS)
    hours = pd.date_range(pd.Timestamp(d, tz="UTC"), periods=24, freq="h")
    rows = []
    for model in ("gfs_global", "ecmwf_ifs025"):
        for p, a, b in zip(POINTS, lat, lon):
            iss = hours - pd.Timedelta(hours=30)
            rows.append(pd.DataFrame({"pixel_id": np.int64(p), "model": model, "interval_start_utc": hours,
                                      "precip_mm": rng.gamma(0.4, 2.0, 24).astype("float32"),
                                      "forecast_issue_time_utc": iss, "available_time_utc": iss + pd.Timedelta(hours=6),
                                      "lat": float(a), "lon": float(b)}))
    df = pd.concat(rows, ignore_index=True)
    for c in ("interval_start_utc", "forecast_issue_time_utc", "available_time_utc"):
        df[c] = df[c].astype("datetime64[us, UTC]")
    return df


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setenv("HFL_P2B_EXCLUDE_SOURCES", "none")
    work, store = tmp_path / "ws", tmp_path / "store"
    monkeypatch.setattr(PB, "ROOT", work)
    monkeypatch.setattr(CT, "ROOT", work)
    cfg = _cfg()
    monkeypatch.setattr(PE, "load_cfg", lambda smoke=False, path=None, check_prereg=True: copy.deepcopy(cfg))
    # preflight and the P4D part of g2 need real Silver / A1 Gold (covered by the cloud smoke); the set-B part of g2
    # (NWP partitions, set-B inner-CV label check) runs for real
    monkeypatch.setattr(PD, "gate_preflight", lambda ctx: ([], {"synthetic": True}))
    monkeypatch.setattr(PD, "gate_g2", lambda ctx: ([], {"synthetic": True}))
    gid = PE.gold_id(cfg)
    rng = np.random.default_rng(7)
    by_month = {}
    for d in BT2.build_days(cfg):
        by_month.setdefault((d.year, d.month), []).append(d)
    for (y, m), days in by_month.items():                 # Gold already on R2 -> feat units must reuse it
        df = _partition(rng, y, m, days)
        assert not BT2.g2_violations(df, BT2.feature_columns(df.columns), PD.CFG_PATH)
        p = store / f"lake/v1/gold/met_training/table_v2/gid={gid}/season={y}/month={m:02d}/part.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p, index=False)
        p.with_name("part.meta.json").write_text("{}", encoding="utf-8")
        if (y, m) in PE.nwp_months(cfg):
            for d in sorted({x + timedelta(days=k) for x in days for k in (-1, 0, 1)}):
                q = store / f"lake/v1/data/silver/met/openmeteo/date={d.isoformat()}/part.parquet"
                q.parent.mkdir(parents=True, exist_ok=True)
                _openmeteo(rng, d).to_parquet(q, index=False)
    for y, m in PE.share_months(cfg):                    # one "full" Gold month (synthetic, 3 days), read only
        df = _partition(rng, y, m, [pd.Timestamp(f"{y}-{m:02d}-{d:02d}").date() for d in (3, 4, 5)])
        p = store / f"lake/v1/gold/met_training/table_v2/gid=a4a6329bcf13/season={y}/month={m:02d}/part.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p, index=False)
    return cfg, store


def _r(rel):
    return PB.ROOT / rel


def _j(rel):
    return json.loads(_r(rel).read_text(encoding="utf-8"))


def _events(out):
    return [json.loads(x) for x in out.splitlines() if x.startswith("{")]


def test_smoke_end_to_end(run, tmp_path, capsys):
    cfg, store = run
    spec = f"local:{store}"
    base = ["--job", JOB, "--store", spec, "--run-id", RUN, "--wait-minutes", "0"]
    pre = f"lake/v1/runs/{JOB}/{RUN}/"
    st = CT.open_run_store(spec, JOB)
    us = PE.units(cfg)
    assert {u.kind for u in us} >= {"nwp", "fit", "dry", "base"}
    # sequential parts with no waiting: a part exits when what it needs is not done yet, and resumes when started again
    for k in range(6):
        assert C.cli(CT.main, base + ["--part", f"{k % 2 + 1}/2"]) == 0
        if "compare-done" in CT.done_units(st, JOB, RUN):
            break
    capsys.readouterr()
    done = CT.done_units(st, JOB, RUN)
    assert {u.id for u in us} <= done
    assert json.loads(st.get(pre + "COMPARE_DONE.json"))["compare_marker"].endswith("compare-done.done.json")
    fm = json.loads(st.get(CT.marker_key(JOB, RUN, "feat-2023-05")))
    assert fm["metrics"]["reused"] is True and fm["outputs"] == {}
    claims = [k for k in st.list(pre + "claims/")]
    assert claims and all(k.endswith("/g001.json") for k in claims)
    # set B: NWP columns for every Gold row (city + ring), option A, under table_v2_nwp/ only
    nk = [k for k in st.list("lake/v1/gold/met_training/table_v2_nwp/") if k.endswith("part.parquet")]
    assert len(nk) == len(PE.nwp_months(cfg)) == 4                      # 2024-09, 2024-10, 2025-07, 2025-10
    nmeta = json.loads(st.get(nk[0].replace("part.parquet", "part.meta.json")))
    assert nmeta["pixels"] == len(CITY) + len(RING) and nmeta["option"] == "A" and nmeta["g2_violations"] == 0
    assert len(nmeta["features"]) == 16 and max(nmeta["nan_share"].values()) < 0.5
    assert _r("reports/p4e/g2/set_b_inner_cv_check.csv").is_file()
    # fits
    f = {s: _j(f"reports/p4e/fit/fit-{s}-s{2025 if s in ('v1b', 'm1b') else 2024}-lead0-thr2p5.json")
         for s in ("m0", "m1", "rnear", "wout", "v1b", "m1b")}
    assert f["m0"]["lgb_params"]["learning_rate"] == 0.06 and f["m0"]["train_rows_ring"] == 0
    assert 0 < f["rnear"]["train_rows_ring"] < f["m1"]["train_rows_ring"]
    assert f["rnear"]["r_near"]["pixels"] == len(_r_near()) < len(RING)
    assert f["wout"]["ring_weight"] == 0.5 and f["wout"]["train_rows"] == f["m1"]["train_rows"]
    assert f["wout"]["train_weight_sum"] == pytest.approx(f["m1"]["train_rows"] - 0.5 * f["m1"]["train_rows_ring"])
    assert f["v1b"]["train_rows_ring"] == 0 and f["m1b"]["train_rows_ring"] > 0
    nwpf = [c for c in f["m1b"]["features"] if c.startswith("nwp_")]
    assert len(nwpf) == 16 and set(nwpf) <= set(f["v1b"]["features"])
    assert not [c for c in f["m1"]["features"] if c.startswith("nwp_")]
    blocks = [x["month"] for x in f["v1b"]["inner_folds"]]
    assert f["v1b"]["inner_cv"]["block"] == "utc_day" and min(blocks) >= 20240906 and max(blocks) < 20241100
    assert [x["month"] for x in f["m1"]["inner_folds"]] == [202305, 202507]   # 2025-10 (Late) never trains
    for s in ("m1ts", "m1bts"):
        fold = 2025 if s == "m1bts" else 2024
        ts = _j(f"reports/p4e/fit/fit-{s}-s{fold}-lead0-thr7p6.json")
        assert ts["two_stage"]["stage2_training_rows"] < ts["train_rows"]
        o = pd.read_parquet(_r(f"reports/p4e/oof/fit-{s}-s{fold}-lead0-thr7p6.parquet"))
        assert np.allclose(o["p_raw"], o["p_stage1"] * o["p_stage2"], atol=1e-6)
        b = pd.read_parquet(_r(f"reports/p4e/base/s{fold}_lead0.parquet"))
        assert len(o) == len(b) and b["pixel_id"].isin(CITY).all()              # evaluation: city, never sampled
    # P4E.2 gate
    dg = _j("reports/p4e/compare/diag_summary.json")
    assert set(dg["propose"]) == {"R-near", "W-out"} and dg["folds"] == [2024]
    assert all(set(r) >= {"diff", "lo", "hi", "pass"} for r in dg["propose"].values())
    dd = pd.read_csv(_r("reports/p4e/compare/diag_diffs.csv"))
    assert set(dd["comparison"]) == {"Model1 - M0", "R-near - Model1", "W-out - Model1"}
    dist = pd.read_csv(_r("reports/p4e/compare/diag_event_dist.csv"))
    assert {"final_events", "matmo_late"} <= set(dist["group"]) and {">=7.6", ">=2.5", "<0.5"} <= set(dist["label_class"])
    assert "lead_min" in pd.read_csv(_r("reports/p4e/compare/diag_matmo_missed.csv")).columns
    # P4E.3 gate
    tu = _j("reports/p4e/compare/twostage_use.json")
    assert tu["complete"] and set(tu["rules"]) == {"A", "B"} and tu["rules"]["A"]["pre_registered"]
    assert all(0 <= v["violation_rate_cal"] <= 1 for v in tu["violation"])
    # P4E.4 + P4E.6 gate
    ch = _j("reports/p4e/compare/checks.json")
    assert ch["rounds_check_ok"] and ch["p4e1_downsampling"]["status"] == "off"
    assert ch["p4e1_downsampling"]["equivalence_rule"] == "not_applied" and ch["folds_set_b"] == [2025]
    sg = ch["shadow_gate"]
    assert sg["complete"] and isinstance(sg["pass"], bool) and sg["frames"][0]["lead_min"] == 0
    assert ch["model1b_7p6_spec"] == ("m1bts" if tu["use_two_stage"] else "m1b")
    m = pd.read_csv(_r("reports/p4e/compare/metrics.csv"))
    assert {"v1B", "Model1-B", "Model1-A", "persistence"} <= set(m["model"])
    assert "M0" not in set(m["model"])                                          # M0 not fitted on fold 2025 here
    df = pd.read_csv(_r("reports/p4e/compare/diffs.csv"))
    assert "Model1-B - v1B" in set(df["comparison"]) and (df["logloss_lo"] <= df["logloss_hi"]).all()
    tab = pd.read_csv(_r("reports/p4e/compare/comparison_table.csv"))
    assert {"P4E.1", "P4E.5"} <= set(tab["part"]) and "P4E.3 use rule (set A)" in set(tab["part"])
    cost = pd.read_csv(_r("reports/p4e/compare/cost.csv"))
    assert (cost["kind"] == "fit").sum() == sum(u.kind == "fit" for u in us)
    assert cost.loc[cost["kind"].isin(["fit", "nwp"]), "seconds_actual"].notna().all()
    dry = pd.read_csv(_r("reports/p4e/compare/dry_share_all.csv"))
    assert list(dry["month"].unique()) == [7] and (dry["gold_id"] == "a4a6329bcf13").all()
    figs = {k.rsplit("/", 1)[1] for k in st.list(pre + "files/reports/p4e/figs/")}
    assert {"reliability.png", "scores_ci.png", "fss.png", "halfmonth_csi.png", "diagnosis_p_on_heavy.png"} <= figs
    sc = pd.read_csv(_r("reports/p4e/figs/scores_ci.csv"))
    assert set(sc["model"]) == {"v1B", "Model1-B", "Model1-A", "persistence"}
    # a fit re-run gives identical bytes (outputs carry no timings)
    u = {x.id: x for x in us}["fit-rnear-s2024-lead0-thr2p5"]
    before = {k: st.get(k) for k in st.list(pre + "files/reports/p4e/") if u.id in k}
    ctx = PE.Ctx(cfg, st, JOB, RUN, tmp_path / "m2", print)
    PE.run_fit(u, ctx)
    assert len(before) == 2
    for k, v in before.items():
        assert _r(k.split("/files/", 1)[1]).read_bytes() == v
    # extended race drill on the finished run: exactly one writer
    keys_before = {k: st.get(k) for k in st.list(pre + "files/") if "/drill/" not in k}
    assert C.cli(CT.main, base + ["--part", "1/2", "--race-drill"]) == 0
    end = [e for e in _events(capsys.readouterr().out) if e["event"] == "end"][0]
    dr = end["race_drill"]
    assert dr["ok"] and dr["loser_exit_code"] == 0 and sorted(dr["states"].values()) == ["won", "yielded"]
    assert {k: st.get(k) for k in st.list(pre + "files/") if "/drill/" not in k} == keys_before
    assert C.cli(CT.main, base + ["--part", "1/3"]) == 2                          # a different part count is refused
