"""P4F/P4G end to end on a SYNTHETIC lake (no real data; tiny LightGBM, 2 threads), driven by cloud_train.main as on
the runner: p4fg (2 parts, LPT; reuse-only Gold / NWP, convective columns from Silver nwp_conv, oracle series, dt-check,
fits incl. hp grid, conv steps, tuned aliases, oracle B, regional target; hp-select, cmp (selection chain, union rule,
oracle A/B, regional skill, shadow gate), fss, figs, compare-done -> SELECTION.json), then p4fg_final (final fits,
REPLAY events, publish under the serving prefix with re-download check), and p4fg_smoke (smoke gates, race drill)."""
import copy
import io
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
from src.models import p4fg as FG
from src.models import p4fg_gates as G
from src.models import p4fg_replay as R
from src.realtime import common as C

Gr = F.Grid()
BOX = np.array([(Gr.lat_idx_min + i) * 3600 + Gr.lon_idx_min + j for i in range(10, 20) for j in range(10, 18)])
CITY = np.sort(BOX[:47])
RING = np.sort(BOX[47:])
POINTS = BOX[::7]
BORDER = np.array(sorted({int(p) + di * 3600 + dj for p in BOX for di in (-1, 0, 1) for dj in (-1, 0, 1)}))
DAYS = [["2023-05-14", "2023-05-16"], ["2024-09-06", "2024-09-08"], ["2024-10-15", "2024-10-17"],
        ["2025-07-14", "2025-07-17"], ["2025-09-29", "2025-09-30"], ["2025-10-06", "2025-10-07"]]
NWP16 = [f"nwp_{m}_{a}_{h}" for m in ("gfs", "ecmwf") for a in ("near", "m03") for h in ("hm1", "h0", "hp1", "sum3")]
P4B = "lake/v1/runs/p4b/c0df12477947/files/reports/p4b/"


def _cfg(smoke: bool) -> dict:
    c = copy.deepcopy(FG.load_cfg(smoke=False))
    c["scope"]["day_ranges"] = DAYS
    c["frames"]["lead_min"] = [0, 30]
    c["lightgbm"] |= {"num_threads": 2}
    c["inner_cv"] |= {"max_rounds": 60, "early_stopping_rounds": 5}
    c["evaluation"] |= {"bootstrap_resamples": 30, "wettest_days": 2}
    c["race"] |= {"claim_poll_seconds": 0.05}
    c["set_b"]["inner_cv_block"] = "utc_day"
    c["hp_grid_points"] = [[15, 500], [31, 200]]
    if smoke:
        c |= {"smoke": True, "smoke_days": DAYS, "run_leads": [30], "run_thresholds": [2.5, 7.6], "folds": [2025],
              "replay_smoke_day": "2025-07-15"}
        c["evaluation"]["loso_seasons"] = [2025]
    return c


def _days(c):
    return BT2.build_days(c)


def _partition(rng, y, m, days):
    pix = np.r_[CITY, RING]
    t0 = pd.DatetimeIndex(np.concatenate([pd.date_range(pd.Timestamp(d, tz="UTC"), periods=48, freq="30min").values
                                          for d in days]), tz="UTC")
    st = F.pixel_static_columns(pix, CITY, F.city_centre(CITY), ring=(0.3, 1.0)).set_index("pixel_id")
    lat, lon = Gr.latlon(pix)
    T, Pn = len(t0), len(pix)
    dry = rng.random(T) < 0.3
    r = rng.gamma(0.45, 6.0, (T, Pn)).astype("float32")
    r[dry] = 0.0
    gc = {k: np.clip(r * (1 - 0.1 * k) + rng.normal(0, 0.8, (T, Pn)), 0, None).astype("float32") for k in range(7)}
    out = []
    for L in (0, 30):
        sh = np.roll(r, -L // 30, axis=0)
        lab = np.clip(sh * rng.lognormal(0, 0.3, (T, Pn)), 0, None).astype("float32")
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
        df["nb_r1_mean_k0"] = (gc[0] * 0.8).reshape(-1)
        hour = df["t0"].dt.tz_convert("Asia/Ho_Chi_Minh").dt.hour
        df["cal_hour_sin"] = np.sin(2 * np.pi * hour / 24).astype("float32")
        df["cal_month"] = np.float32(m)
        df["lat"] = np.tile(lat, T).astype("float32"); df["lon"] = np.tile(lon, T).astype("float32")
        for c in ("dist_to_city_center_km", "is_city", "ring_coverage"):
            df[c] = st.loc[df["pixel_id"], c].to_numpy()
        df["gsmap__event_time"] = df["t0"]
        df["gsmap__available_time"] = df["t0"] + pd.Timedelta(minutes=5)
        out.append(df)
    return pd.concat(out, ignore_index=True)


def _put(store, key, df):
    p = store / key
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)


def _write_world(store, c, rng):
    gid = FG.gold_id(c)
    by = {}
    for d in _days(c):
        by.setdefault((d.year, d.month), []).append(d)
    nid = PE.nwp_id(c)
    for (y, m), days in by.items():
        g = _partition(rng, y, m, days)
        k = f"lake/v1/gold/met_training/table_v2/gid={gid}/season={y}/month={m:02d}/"
        _put(store, k + "part.parquet", g)
        (store / k / "part.meta.json").write_text("{}", encoding="utf-8")
        if (y, m) in PE.nwp_months(c):
            n = g[["pixel_id", "t0", "lead_min"]].copy()
            for col in NWP16:
                n[col] = rng.gamma(0.5, 1.0, len(n)).astype("float32")
            kn = f"lake/v1/gold/met_training/table_v2_nwp/gid={gid}/nid={nid}/season={y}/month={m:02d}/"
            _put(store, kn + "part.parquet", n)
            (store / kn / "part.meta.json").write_text(json.dumps({"rows": len(n), "pixels": 80, "points": 12,
                                                                   "nearest_point_deg": {"max": 0.2},
                                                                   "nan_share": {x: 0.0 for x in NWP16}}))
            first = pd.Timestamp(y, m, 1)
            for d in pd.date_range(first - pd.Timedelta(days=2), first + pd.offsets.MonthEnd(0), freq="D"):
                T = pd.date_range(d.tz_localize("UTC"), periods=24, freq="h")
                lat, lon = Gr.latlon(POINTS)
                rows = []
                for p, a, b in zip(POINTS, lat, lon):
                    iss = T.floor("6h") - pd.Timedelta(hours=24)
                    rows.append(pd.DataFrame({"pixel_id": np.int64(p), "lat": float(a), "lon": float(b),
                                              "model": "gfs_global", "time_utc": T,
                                              "cape": rng.gamma(1.0, 600, 24), "lifted_index": rng.normal(-1, 2, 24),
                                              "forecast_issue_time_utc": iss,
                                              "available_time_utc": iss + pd.Timedelta(minutes=420)}))
                _put(store, f"lake/v1/silver/nwp_conv/date={d.date().isoformat()}/part.parquet",
                     pd.concat(rows, ignore_index=True))
    sil = sorted({d + timedelta(days=k) for d in _days(c) for k in (-2, -1, 0, 1)})
    for d in sil:
        w = pd.date_range(pd.Timestamp(d, tz="UTC"), periods=48, freq="30min")
        n = len(BORDER)
        imr = rng.gamma(0.4, 4.0, (48, n))
        gs = pd.DataFrame({"pixel_id": np.tile(BORDER, 48), "window_start_utc": np.repeat(w, n),
                           "rain_rate_gc_mm_h": (imr * 1.1).ravel(), "rain_rate_mm_h": (imr * 0.9).ravel()})
        _put(store, f"lake/v1/data/silver/met/gsmap_now/date={d.isoformat()}/part.parquet", gs)
        im = pd.DataFrame({"pixel_id": np.tile(BORDER, 48), "interval_start_utc": np.repeat(w, n),
                           "rain_rate_mm_h": imr.ravel(), "is_city_pixel": np.tile(np.isin(BORDER, CITY), 48)})
        src = "imerg_late" if (d.year, d.month) == (2025, 10) else "imerg_final"
        _put(store, f"lake/v1/data/silver/met/{src}/date={d.isoformat()}/part.parquet", im)
    # P4B row-level v1 predictions (OOF 2024 set B, test 2025, secondary 2025-10)
    idx = []
    for L in (0, 30):
        for t in (0.5, 2.5, 7.6):
            k = PB.kstr(t)
            idx.append({"model": f"B-lead{L}-thr{k}", "cv_unit": f"cv-B24-lead{L}-thr{k}", "calibration": "raw",
                        "cut": 0.3})
            for kind, yy in (("oof", 2024), ("test", 2025), ("secondary", 2025)):
                ds = [d for d in _days(c) if d.year == yy and ((d.month == 10) == (kind == "secondary"))]
                t0 = pd.DatetimeIndex(np.concatenate([pd.date_range(pd.Timestamp(d, tz="UTC"), periods=48,
                                                                    freq="30min").values for d in ds]), tz="UTC")
                df = pd.DataFrame({"pixel_id": np.tile(CITY, len(t0)), "t0": np.repeat(t0, len(CITY)),
                                   "lead_min": L})
                p = rng.random(len(df))
                df["y"] = (rng.random(len(df)) < 0.2).astype("int8")
                if kind == "oof":
                    df["day"] = df["t0"].dt.tz_convert("Asia/Ho_Chi_Minh").dt.date.astype(str)
                    df["label_rate_mm_h"] = rng.gamma(0.4, 5.0, len(df))
                    df = df.assign(fold="2024-09", p_raw=p, p_best=p, p_clim=0.1, pers=rng.gamma(0.4, 5.0, len(df)))
                    _put(store, P4B + f"oof/cv-B24-lead{L}-thr{k}.parquet", df)
                else:
                    _put(store, P4B + f"final/preds/{kind}_lead{L}_thr{k}.parquet",
                         df.assign(p_v1=p, yhat_v1=(p > 0.6).astype("int8")))
    (store / P4B / "oof").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(idx).to_csv(store / P4B / "oof" / "v1_models_oof.csv", index=False)


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setenv("HFL_P2B_EXCLUDE_SOURCES", "none")
    work, store = tmp_path / "ws", tmp_path / "store"
    monkeypatch.setattr(PB, "ROOT", work)
    monkeypatch.setattr(CT, "ROOT", work)
    monkeypatch.setattr(PD, "gate_preflight", lambda ctx: ([], {"synthetic": True}))
    monkeypatch.setattr(PD, "gate_g2", lambda ctx: ([], {"synthetic": True}))
    monkeypatch.setattr(PD, "pixel_sets", lambda ctx: (CITY, RING, {"ring_pixels": len(RING)}))
    monkeypatch.setattr(G, "relabel_p4e", lambda ctx, FGX, n, seed, lvl: ([], {}))
    full, smoke = _cfg(False), _cfg(True)
    _write_world(store, full, np.random.default_rng(11))
    return full, smoke, store, monkeypatch


def _drive(job, store, parts=2, rounds=6, extra=()):
    base = ["--job", job, "--store", f"local:{store}", "--run-id", "fx", "--wait-minutes", "0"]
    for k in range(rounds):
        rc = C.cli(CT.main, base + ["--part", f"{k % parts + 1}/{parts}"] + list(extra))
        assert rc == 0
        st = CT.open_run_store(f"local:{store}", job)
        done = CT.done_units(st, job, "fx")
        if ("compare-done" in done) or ("final-done" in done):
            return st
    raise AssertionError(f"{job} did not finish: done {len(done)}")


def _j(rel):
    return json.loads((PB.ROOT / rel).read_text(encoding="utf-8"))


def test_full_and_final(world, capsys):
    full, smoke, store, mp = world
    mp.setattr(FG, "load_cfg", lambda smoke=False, path=None, check_prereg=True: copy.deepcopy(full))
    st = _drive("p4fg", store)
    pre = "lake/v1/runs/p4fg/fx/"
    sel = json.loads(st.get(pre + "SELECTION.json"))
    assert not sel["smoke"] and set(sel["model2_specs"]) == {"2024", "2025"} and sel["prereg_sha"] == FG.PREREG_SHA
    assert sel["dt_check"]["verdict"] == "ok" and 0.6 <= sel["dt_check"]["pooled_ratio"] <= 1.5
    # conv columns written under table_v3_conv only; Gold / NWP never rebuilt
    ck = [k for k in st.list("lake/v1/gold/met_training/table_v3_conv/") if k.endswith("part.parquet")]
    assert len(ck) == len(FG.run_months(full)) and json.loads(st.get(ck[0].replace("part.parquet", "part.meta.json")))[
        "g2_violations"] == 0
    feats = _j("reports/p4fg/fit/fit-b15x500ac-s2025-lead30-thr2p5.json")["features"]
    assert {"nwp_conv_gfs_cape_near", "nwp_conv_gfs_cape_x_rain"} <= set(feats) and not [f for f in feats if "oracle" in f]
    of = _j("reports/p4fg/oracle/fit/fit-orc-s2025-lead30-thr2p5.json")
    assert of["oracle"] and "oracle_gc_target" in of["features"]
    assert pd.read_parquet(PB.ROOT / "reports/p4fg/oracle/oof/fit-orc-s2025-lead30-thr2p5.parquet")["oracle"].all()
    hp = _j("reports/p4fg/compare/hp_select.json")
    assert set(hp["folds"]) == {"2024", "2025"}
    for s in (2024, 2025):
        t = _j(f"reports/p4fg/fit/fit-ta-s{s}-lead0-thr2p5.json")
        assert ("alias" in t) == (hp["folds"][str(s)]["setting"] == [15, 500])
    sj = _j("reports/p4fg/compare/selection.json")
    steps = {x["step"]: x["status"] for x in sj["selection"]["steps"]}
    assert steps["conv_g_b"] == "dropped" and set(steps) == {"hp_tuned", "conv_g_a", "conv_g_b", "conv_g_c"}
    assert {"pass", "complete"} <= set(sj["shadow_gate_model2"]) and set(sj["regional_skill"]) == {"2.5", "7.6"}
    oa = pd.read_csv(PB.ROOT / "reports/p4fg/oracle/oracle_a.csv")
    assert oa["oracle"].all() and {"A_gc", "A_raw"} == set(oa["kind"]) and {"at_threshold", "oof_cut"} == set(oa["mode"])
    ob = pd.read_csv(PB.ROOT / "reports/p4fg/oracle/oracle_b.csv")
    assert ob["oracle"].all() and ob["heavy_gap_remaining_share"].notna().any()
    un = pd.read_csv(PB.ROOT / "reports/p4fg/compare/union_rule.csv")
    assert set(un["lead_min"]) == {30} and {"csi_diff", "csi_lo", "bias_union"} <= set(un.columns)
    full_tab = pd.read_csv(PB.ROOT / "reports/p4fg/compare/comparison_table_full.csv")
    assert {"fss_3x3", "fss_5x5", "display_name"} <= set(full_tab.columns) and "persistence" in set(full_tab["model"])
    fv = pd.read_csv(PB.ROOT / "reports/p4fg/fss/fss_v1_p4b.csv")
    assert set(fv["data"]) == {"P4B OOF 2024 (cv-B24)", "P4B test 2025 (final/preds)"} and set(fv["size"]) == {"3x3", "5x5"}
    figs = {k.rsplit("/", 1)[1] for k in st.list(pre + "files/reports/p4fg/figs/")}
    assert {"reliability.png", "scores_ci.png", "fss.png", "halfmonth_csi.png"} <= figs
    dt = _j("reports/p4fg/dtcheck/dt_check.json")
    assert dt["code_dt_h_per_30min_file"] == 0.5 and dt["pooled_ratio_gc_imerg_dt1"] == pytest.approx(
        2 * dt["pooled_ratio_gc_imerg_code_dt"])
    rn = list((PB.ROOT / "reports/p4fg/oof_ring").glob("fit-b15x500-*.parquet"))
    assert rn and pd.read_parquet(rn[0])["pixel_id"].isin(RING).all()
    # ---- p4fg_final: final fits, REPLAY, publish
    st2 = _drive("p4fg_final", store, parts=1, rounds=3)
    mods = [k for k in st2.list("lake/v1/runs/p4fg_final/fx/files/models/v2_candidate/") if k.endswith("meta.json")]
    assert mods and all("/models/v2_candidate/" in k for k in mods)
    m0 = json.loads(st2.get(mods[0]))
    assert m0["rounds"] >= 1 and "2025-10" not in m0["data_range_utc"][1][:7] and not [f for f in m0["features"]
                                                                                       if "oracle" in f]
    srv = st2.list("lake/v1/serving/replay/pkg=2026-11/")
    assert {f"lake/v1/serving/replay/pkg=2026-11/{e}/replay.parquet" for e in ("yagi", "wet0930", "wettest3",
                                                                              "matmo")} <= set(srv)
    man = json.loads(st2.get("lake/v1/serving/replay/pkg=2026-11/_manifest.json"))
    assert man["p4f0_verdict"] == "ok" and man["events"]["matmo"]["label_product"] == "imerg_late"
    iw = man["icon_windows"]                                           # D217-5 window shift recorded (S2, D218-2)
    assert iw["operational"]["window_shift_min"] == 30 and iw["operational"]["window_end"] == "t0 - 30 min"
    assert iw["hindsight"]["window_shift_min"] == 0
    assert len(man["events"]["wettest3"]["days_local"]) == 3 and "2025-09-30" not in man["events"]["wettest3"]["days_local"]
    pc = _j("reports/p4fg_final/replay/publish_check.json")
    assert not pc["mismatch"] and len(pc["verified"]) == 9
    assert not list((PB.ROOT / "data/_scratch/p4fg/replay_verify").glob("*"))           # scratch files removed
    rp = pd.read_parquet(io.BytesIO(st2.get("lake/v1/serving/replay/pkg=2026-11/yagi/replay.parquet")))
    assert {"t0_utc", "t0_local", "pixel_id", "lat", "lon", "is_city", "gsmap_gc_mmh", "gsmap_age_min", "label_mmh",
            "label_product", "p_v1_2p5_F0", "p_m2_7p6_F30", "oos_v1", "oos_m2", "rain_class_m2_F30",
            "display_heavy_F30", "display_rule", "a6h_mm", "icon", "icon_product", "a6h_mm_hindsight",
            "icon_hindsight", "icon_product_hindsight"} <= set(rp.columns)
    assert (~rp["is_city"]).any() and rp.loc[~rp["is_city"], "p_v1_2p5_F0"].isna().all()
    assert rp.loc[~rp["is_city"], "p_m2_2p5_F0"].notna().any()
    assert set(rp["icon_product"]) == {"gsmap_now_gc"} and set(rp["icon_product_hindsight"]) == {"imerg_final"}
    # D220: probability columns on the evaluated scale; p_raw_* for m1b/m2 only; scale per column in the manifest
    ps = man["events"]["yagi"]["prob_scale"]
    assert ps["p_v1_2p5_F0"] == "v1_serving_raw" and ps["p_m2_2p5_F0"] == "platt_fold"
    assert ps["p_raw_m2_2p5_F0"] == "raw" and "p_raw_v1_2p5_F0" not in rp.columns
    assert man["events"]["matmo"]["prob_scale"]["p_m2_2p5_F0"] == "platt_final"
    assert set(man["prob_scales"]) == {"platt_fold", "platt_final", "v1_serving_raw", "raw"}
    src_sel = _j("reports/p4fg_final/selection_source.json")
    same = src_sel["resolved"]["model2"] == src_sel["resolved"]["model1b"]
    mm = json.loads(st2.get("lake/v1/runs/p4fg_final/fx/files/models/v2_candidate/"
                            f"{R.MODEL_DIRS['model1b' if same else 'model2']}/lead0_thr2p5/meta.json"))
    rm = pd.read_parquet(io.BytesIO(st2.get("lake/v1/serving/replay/pkg=2026-11/matmo/replay.parquet")))
    ok = rm["p_raw_m2_2p5_F0"].notna()
    assert ok.any() and np.allclose(rm.loc[ok, "p_m2_2p5_F0"], PB.apply_calibrator(
        mm["platt"], rm.loc[ok, "p_raw_m2_2p5_F0"].to_numpy("float64")), atol=1e-5)
    assert not np.allclose(rm.loc[ok, "p_m2_2p5_F0"], rm.loc[ok, "p_raw_m2_2p5_F0"], atol=1e-4)
    # the store refuses a second package with other bytes (put-if-absent): re-running publish is a no-op
    assert C.cli(CT.main, ["--job", "p4fg_final", "--store", f"local:{store}", "--run-id", "fx", "--part", "1/1"]) == 0


def test_smoke(world, capsys):
    full, scfg, store, mp = world
    mp.setattr(FG, "load_cfg", lambda smoke=False, path=None, check_prereg=True: copy.deepcopy(scfg))
    us = FG.units(scfg)
    assert FG.gold_id(scfg) == FG.gold_id(full)
    st = _drive("p4fg_smoke", store, parts=2, rounds=8)
    pre = "lake/v1/runs/p4fg_smoke/fx/"
    assert {u.id for u in us} <= CT.done_units(st, "p4fg_smoke", "fx")
    sel = json.loads(st.get(pre + "SELECTION.json"))
    assert sel["smoke"] and set(sel["model2_specs"]) == {"2025"}
    sm = [k for k in st.list(pre + "files/reports/p4fg/smoke_models/") if k.endswith("model.txt")]
    assert len(sm) == 1
    rk = st.list(pre + "replay_pkg/")
    assert {pre + "replay_pkg/2025-07-15/replay.parquet", pre + "replay_pkg/2025-07-15/_manifest.json"} <= set(rk)
    assert not st.list("lake/v1/serving/")                                    # smoke never writes the serving prefix
    sman = json.loads(st.get(pre + "replay_pkg/2025-07-15/_manifest.json"))
    assert sman["icon_windows"]["operational"]["window_shift_min"] == 30
    assert sman["prob_scale"]["p_m1b_2p5_F30"] == "platt_fold" and sman["prob_scale"]["p_v1_2p5_F30"] == "v1_serving_raw"
    srp = pd.read_parquet(io.BytesIO(st.get(pre + "replay_pkg/2025-07-15/replay.parquet")))
    info = json.loads(st.get(pre + f"files/reports/p4fg/fit/{FG.fit_id(FG.B0, 2025, 30, 2.5)}.json"))
    ok = srp["p_raw_m1b_2p5_F30"].notna()
    assert ok.any() and np.allclose(srp.loc[ok, "p_m1b_2p5_F30"], PB.apply_calibrator(
        info["platt"], srp.loc[ok, "p_raw_m1b_2p5_F30"].to_numpy("float64")), atol=1e-5)   # fold Platt (D220)
    capsys.readouterr()
    keys_before = {k: st.get(k) for k in st.list(pre + "files/") if "/drill/" not in k}
    assert C.cli(CT.main, ["--job", "p4fg_smoke", "--store", f"local:{store}", "--run-id", "fx", "--wait-minutes", "0",
                           "--part", "1/2", "--race-drill"]) == 0
    end = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.startswith("{") and '"end"' in x][-1]
    assert end["race_drill"]["ok"]
    assert {k: st.get(k) for k in st.list(pre + "files/") if "/drill/" not in k} == keys_before
