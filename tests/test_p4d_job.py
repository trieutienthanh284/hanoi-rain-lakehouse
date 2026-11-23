"""P4D job structure (offline, no data, no LightGBM): frozen design in configs/model_v2.yaml, unit graph and parts,
Model 1 cut / Model 2 weights, ring geometry, static pixel columns, cloud_train guards (no --final, parts lock),
race-safe commit for deterministic outputs, planning estimates."""
import json

import numpy as np
import pandas as pd
import pytest

from src.gold import build_table_v2 as BT2
from src.gold import features as F
from src.models import cloud_train as CT
from src.models import p4d as PD
from src.realtime import common as C

CFG = PD.load_cfg()
SMOKE = PD.load_cfg(smoke=True)
G = F.Grid()


def _bbox_city(n=47):
    """47 synthetic city pixels inside the 10 x 8 margin-10 box (= the real city bounding box)."""
    box = [(G.lat_idx_min + i) * 3600 + G.lon_idx_min + j for i in range(10, 20) for j in range(10, 18)]
    return np.sort(np.array(box[:n], dtype="int64"))


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("HFL_P2B_EXCLUDE_SOURCES", "none")
    monkeypatch.setattr(CT, "ROOT", tmp_path / "ws")


# ---------------------------------------------------------------- frozen design
def test_design_frozen_in_config():
    assert CFG["evaluation"]["loso_seasons"] == [2023, 2024, 2025]
    assert CFG["scope"]["label_source_by_month"] == {"2025-10": "imerg_late"}
    assert CFG["scope"]["training_label_source"] == "imerg_final"
    assert CFG["evaluation"]["secondary_months"] == ["2025-10"]
    assert CFG["model1"]["cut_utc"] == [["05-01", "05-15"], ["10-16", "10-31"]]
    assert CFG["model2"]["grid"] == {"k": [2, 3], "alpha": [0, 1], "beta": [0, 4]} and len(PD.grid_settings(CFG)) == 8
    lg = CFG["lightgbm"]                                      # v1 setting g1
    assert (lg["num_leaves"], lg["min_data_in_leaf"], lg["lambda_l2"], lg["learning_rate"]) == (15, 500, 1, 0.03)
    assert CFG["evaluation"]["recalibration"] == "platt_crossfit" and CFG["evaluation"]["heavy_rule"] == "nested"
    assert not any(str(y).startswith("2022") for y in CFG["scope"]["seasons"])           # hub decision 0: no 2022
    assert BT2.build_months(CFG)[0] == (2023, 5) and BT2.build_months(CFG)[-1] == (2025, 10)
    assert len(BT2.build_days(CFG)) == 3 * 184


def test_training_months_exclude_late():
    assert BT2.is_training_month(CFG, 2025, 9) and not BT2.is_training_month(CFG, 2025, 10)
    assert BT2.label_source(CFG, 2025, 10) == "imerg_late" and BT2.label_source(CFG, 2024, 10) == "imerg_final"


def test_cut_mask_by_utc_date():
    md = np.array([430, 501, 515, 516, 1015, 1016, 1031], dtype="int16")
    assert PD.cut_mask(md, CFG).tolist() == [False, True, True, False, False, True, True]


def test_sample_weights_grid_formula():
    md = np.array([510, 520, 1010, 1020], dtype="int16")
    lab = np.array([3.0, 8.0, 0.1, np.nan])
    w = PD.sample_weights(md, lab, {"k": 3.0, "alpha": 1.0, "beta": 4.0}, CFG)
    # w_period: 1 (May 10), 3 (May 20), 3 (Oct 10), 1 (Oct 20); w_class: 2, 6, 1, 1 (NaN label -> 0)
    assert w.tolist() == [2.0, 18.0, 3.0, 1.0]
    assert PD.sample_weights(md, lab, {"k": 2.0, "alpha": 0.0, "beta": 0.0}, CFG).tolist() == [1.0, 2.0, 2.0, 1.0]


# ---------------------------------------------------------------- units
def test_unit_graph_full_run():
    us = PD.units(CFG)
    ids = [u.id for u in us]
    assert len(ids) == len(set(ids))
    by = {u.id: u for u in us}
    fits = [u for u in us if u.kind == "fit"]
    count = lambda sp: sum(u.params["spec"] == sp for u in fits)              # noqa: E731
    assert count("m0") == 27 and count("m1a") == 9 and count("m1b") == 9 and count("m1") == 18 and count("m2") == 9
    assert sum(u.params["spec"].startswith("m2-g") for u in fits) == 8 * 9 * 2
    assert len(fits) == 216 and sum(u.kind == "feat" for u in us) == 18 and sum(u.kind == "base" for u in us) == 9
    assert all(u.params["thr"] == 2.5 for u in fits if u.params["spec"] in ("m1a", "m1b"))
    assert by["sel-m1a"].deps and all(d.startswith(("base-", "fit-m0-", "fit-m1a-")) for d in by["sel-m1a"].deps)
    assert all("thr2p5" in d for d in by["sel-m1a"].deps if d.startswith("fit-"))
    assert all(u.deps == ("sel-m1a",) for u in fits if u.params["spec"] == "m1b")
    assert all(u.deps == ("sel-m1b",) for u in fits if u.params["spec"] == "m1" or u.params["spec"].startswith("m2-g"))
    assert all(u.deps == ("sel-m2",) for u in fits if u.params["spec"] == "m2")
    assert by["compare-done"].deps == ("cmp-main", "cmp-detail") and by["compare-done"].stage == 8
    assert not any(u.kind == "final" for u in us)                             # no test-season unit in this phase
    for u in us:                                                              # dependencies point backwards
        assert all(by[d].stage <= u.stage for d in u.deps)


def test_parts_cover_every_work_unit_once():
    us = PD.units(CFG)
    work = [u.id for u in us if u.kind != "gate"]
    n = PD.choose_parts(CFG)
    assert n == 13
    got = [u.id for i in range(1, n + 1) for u in PD.part_units(us, i, n)]
    assert sorted(got) == sorted(work)
    st5 = [len([u for u in PD.part_units(us, i, n) if u.stage == 5]) for i in range(1, n + 1)]
    assert max(st5) - min(st5) <= 1                                           # stage 5 balanced


def test_smoke_units_and_ids(monkeypatch):
    monkeypatch.setattr(CT, "ROOT", PD.CODE_ROOT)                             # run ids hash the code files
    us = PD.units(SMOKE)
    fits = [u for u in us if u.kind == "fit"]
    assert {u.params["thr"] for u in fits} == {2.5}
    assert sorted({u.params["spec"] for u in fits}) == ["m0", "m1a", "m1b", "m2-g1"]
    assert [u.id for u in us if u.kind == "feat"] == ["feat-2023-05", "feat-2024-10", "feat-2025-07", "feat-2025-10"]
    assert PD.gold_id(SMOKE) != PD.gold_id(CFG)
    assert CT.run_id_for("p4d") != CT.run_id_for("p4d_smoke")
    assert CT.run_id_for("p4d") not in (CT.run_id_for("p4b"), CT.run_id_for("p4b_smoke"))
    days = BT2.build_days(SMOKE)
    md = np.array([d.month * 100 + d.day for d in days])
    assert PD.cut_mask(md, SMOKE).any() and (~PD.cut_mask(md, SMOKE)).any()  # every smoke block exercises the cut


# ---------------------------------------------------------------- ring geometry and static columns
def test_ring_rule_largest_margin_with_150():
    p = F.feature_params(CFG)
    assert BT2.required_margin_px(p) == 10                                    # full set-d footprint
    city = _bbox_city()
    assert [len(BT2.ring_candidates(city, m)) for m in (10, 9, 8, 7, 6)] == [33, 73, 121, 177, 241]
    assert BT2.largest_margin_for(city, 150) == 7 == CFG["pixels"]["ring"]["min_margin_px"]
    ring, rec = BT2.ring_pixels(city, CFG)
    assert len(ring) == 177 == CFG["pixels"]["ring"]["expected_pixels"] and not rec["subsampled"]
    assert rec["ring_pixels_truncated"] == 177 - 33 and 0.8 < rec["ring_coverage_min"] < 1.0
    assert (BT2.margin_px(ring) >= 7).all() and not np.isin(ring, city).any()
    r4, rec4 = BT2.ring_pixels(city, SMOKE)
    assert len(r4) == 4 and rec4["subsampled"] and np.isin(r4, ring).all()
    assert np.array_equal(r4, BT2.ring_pixels(city, SMOKE)[0])               # fixed seed
    bad = json.loads(json.dumps(CFG))
    bad["pixels"]["ring"]["min_margin_px"] = 8
    with pytest.raises(ValueError, match="not the largest margin"):
        BT2.ring_pixels(city, bad)


def test_truncation_table_and_coverage():
    p = F.feature_params(CFG)
    city = _bbox_city()
    ring, _ = BT2.ring_pixels(city, CFG)
    pt = BT2.pixel_table(city, ring, p)
    c, r = pt[pt["is_city"] == 1], pt[pt["is_city"] == 0]
    assert (c[["ring_coverage", "ring_quadrant_min", "box_r5", "adv_inside"]] == 1).all().all()
    assert (r["box_r5"] == 1).all()                                           # margin 7 >= box radius 5
    edge = r["margin_px"] < 10
    assert (r.loc[edge, "ring_coverage"] < 1).all() and (r.loc[~edge, "ring_coverage"] == 1).all()
    assert (r.loc[edge, "ring_quadrant_min"] < 1).all() and (r["ring_quadrant_min"] > 0).all()
    assert (r["adv_inside"] == (r["margin_px"] >= 8)).all()


def test_truncated_ring_feature_uses_in_domain_part():
    """A margin-7 pixel: ring quadrant means are computed over the in-domain cells only (never NaN)."""
    p = F.feature_params(CFG)
    city = _bbox_city()
    ring, _ = BT2.ring_pixels(city, CFG)
    pid = ring[BT2.margin_px(ring) == 7][:1]
    rng = np.random.default_rng(0)
    T, K = 1, len(p["K"])
    gc = rng.gamma(0.5, 2.0, (T, K, G.n_lat, G.n_lon)).astype("float32")
    t0 = pd.DatetimeIndex([pd.Timestamp("2024-07-01 06:00", tz="UTC")])
    df = F.compute_features(gc, gc.copy(), t0, pid, p, G)
    ii, jj = G.ij(pid)
    rq = F.ring_quadrants(ii, jj, G, p["ring"])
    a = np.minimum(gc[0, 0], p["cap"])
    for q in "NESW":
        ri, rj = rq[q][0]
        assert np.isclose(df[f"ring_{q}_mean_k0"].iloc[0], a[ri, rj].mean(), rtol=1e-5)
    assert df[[c for c in df.columns if c.startswith(("nb_", "ring_"))]].notna().all().all()


def test_pixel_static_columns():
    city = _bbox_city()
    pix = np.r_[city[:2], BT2.ring_pixels(city, CFG)[0][:2]]
    st = F.pixel_static_columns(pix, city, F.city_centre(city), ring=(0.3, 1.0))
    assert st["is_city"].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert (st["ring_coverage"].iloc[:2] == 1).all() and ((st["ring_coverage"] > 0) & (st["ring_coverage"] <= 1)).all()
    assert "ring_coverage" not in F.pixel_static_columns(pix, city, F.city_centre(city)).columns
    lat0, lon0 = F.city_centre(city)
    lat, lon = G.latlon(pix)
    approx = np.hypot((lat - lat0) * 111.2, (lon - lon0) * 111.2 * np.cos(np.radians(lat0)))
    assert np.allclose(st["dist_to_city_center_km"], approx, rtol=0.01)
    assert F.group_of("dist_to_city_center_km") == "pixel_static" and F.GROUP_KIND["pixel_static"] == "static"
    assert F.group_of("ring_coverage") == "pixel_static" and F.group_of("ring_N_mean_k0") == "gsmap"
    assert F.group_of("gsmap_n_missing") == "gsmap_meta" and F.group_of("lat") == "static"


# ---------------------------------------------------------------- driver guards
@pytest.mark.parametrize("flag", ["--final", "--write-oof-done", "--approve-final", "--race-drill"])
def test_p4d_refuses_final_and_friends(tmp_path, flag):
    assert C.cli(CT.main, ["--job", "p4d", "--store", f"local:{tmp_path / 's'}", flag]) == 2


def test_parts_lock(tmp_path):
    st = CT.open_run_store(f"local:{tmp_path / 's'}", "p4d")
    assert CT.parts_lock(st, "p4d", "r", 5) == (True, 5)
    assert CT.parts_lock(st, "p4d", "r", 5) == (True, 5)
    assert CT.parts_lock(st, "p4d", "r", 1) == (False, 5)


def test_store_restricted_to_run_and_table_v2(tmp_path):
    st = CT.open_run_store(f"local:{tmp_path / 's'}", "p4d")
    st.put("lake/v1/gold/met_training/table_v2/gid=x/a.bin", b"1")
    st.put("lake/v1/runs/p4d/r/files/a.bin", b"1")
    for k in ("lake/v1/gold/met_training/table_A1_v1/gid=0803d9d8ed72/x", "lake/v1/runs/p4b/x", "lake/v1/reports/x",
              "lake/v1/runs/p4d_smoke/x"):
        with pytest.raises(C.KeyRefused):
            st.put(k, b"1")


def _out(rel, data: bytes):
    p = CT.ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return rel


def test_commit_safe_won_skipped_identical(tmp_path):
    st = CT.open_run_store(f"local:{tmp_path / 's'}", "p4d")
    u = CT.Unit("fit-x", "fit")
    rel = _out("reports/p4d/oof/x.parquet", b"same bytes")
    assert CT.commit_unit_safe(st, "p4d", "r", u, [rel], {}, tmp_path / "m")["state"] == "won"
    assert CT.commit_unit_safe(st, "p4d", "r", u, [rel], {}, tmp_path / "m")["state"] == "skipped"


def test_commit_safe_never_overwrites_after_marker(tmp_path, monkeypatch):
    st = CT.open_run_store(f"local:{tmp_path / 's'}", "p4d")
    u = CT.Unit("g", "gate")
    rel = _out("reports/p4d/x.csv", b"winner")
    CT.commit_unit_safe(st, "p4d", "r", u, [rel], {}, tmp_path / "m1")
    key = CT.file_key("p4d", "r", rel)
    calls = {"n": 0}
    real = CT.marker_exists

    def first_false(*a):                     # our first check ran before the winner's marker existed
        calls["n"] += 1
        return False if calls["n"] == 1 else real(*a)
    monkeypatch.setattr(CT, "marker_exists", first_false)
    _out(rel, b"loser, other bytes")
    c = CT.commit_unit_safe(st, "p4d", "r", u, [rel], {}, tmp_path / "m2")
    assert c["state"] == "lost" and c["mismatch"] == [key]
    assert st.get(key) == b"winner"                                           # nothing overwritten


def test_commit_safe_replaces_unmarked_leftover(tmp_path):
    st = CT.open_run_store(f"local:{tmp_path / 's'}", "p4d")
    rel = "reports/p4d/x.csv"
    key = CT.file_key("p4d", "r", rel)
    st.put(key, b"interrupted attempt")                                       # output without a marker
    _out(rel, b"complete attempt")
    c = CT.commit_unit_safe(st, "p4d", "r", CT.Unit("g", "gate"), [rel], {}, tmp_path / "m")
    assert c["state"] == "won" and st.get(key) == b"complete attempt"
    mk = json.loads(st.get(CT.marker_key("p4d", "r", "g")))
    assert mk["outputs"][key] == C.sha256_bytes(b"complete attempt")


def test_fit_outputs_carry_no_timing():
    """Race fix: files listed in markers must be byte-identical on a re-run (no 'seconds' inside them)."""
    import inspect
    src = inspect.getsource(PD.run_fit)
    info_block = src.split("info = {")[1].split("write_json(info_p, info)")[0]
    assert "seconds" not in info_block and "rss" not in info_block


def test_estimates_within_limits():
    e = PD.estimates(CFG)
    assert e["pixels"] == 224 and e["rows_per_frame"] == 552 * 48 * 224 and e["rows_total"] == 17_805_312
    assert e["peak_ram_gb_est"] <= CFG["resources"]["max_rss_gb"]
    assert (e["gold_bytes_est"] / 1e9) <= CFG["r2_budget"]["projected_gold_gb"]
    assert e["part_wall_minutes"] <= 0.8 * 320 and e["parts"] == 13
    assert PD.part_wall_minutes(CFG, 12) > 0.8 * 320                           # 13 is the smallest count


def test_workflow_lists_p4d():
    wf = (PD.CODE_ROOT / ".github" / "workflows" / "train.yml").read_text(encoding="utf-8")
    import re
    opts = [x.strip() for x in re.search(r"options: \[([^\]]*)\]", wf).group(1).split(",")]
    assert "p4d_smoke" in opts and "p4d" in opts and "--wait-minutes 150" in wf
    assert "tests/test_p4d_job.py" in wf and "inputs.job == 'p4d'" in wf


# ---------------------------------------------------------------- inner-CV feasibility (fail fast)
OLD_SMOKE = [["2023-05-13", "2023-05-18"], ["2024-10-14", "2024-10-19"], ["2025-05-13", "2025-05-18"],
             ["2025-10-06", "2025-10-07"]]                                    # p4d_smoke 793004743201


def _with_days(ranges):
    c = json.loads(json.dumps(SMOKE))
    c["scope"]["day_ranges"] = ranges
    return c


def test_inner_cv_plan_check_configs():
    assert all(r["ok"] for r in PD.inner_cv_plan_check(CFG))
    sm = PD.inner_cv_plan_check(SMOKE)
    assert all(r["ok"] and len(r["usable_blocks"]) == 2 for r in sm) and len(sm) == 6
    assert SMOKE["scope"]["day_ranges"][2] == ["2025-07-01", "2025-07-06"]
    full = {(r["heldout_season"], r["variant"]): r for r in PD.inner_cv_plan_check(CFG)}
    assert len(full[(2023, "cut")]["usable_blocks"]) == 11 and len(full[(2025, "plain")]["usable_blocks"]) == 12


def test_inner_cv_plan_check_failures():
    # old smoke set: fold 2024 trains on 2023-05 + 2025-05 = 2 year-month blocks but one calendar month -> refused
    old = {(r["heldout_season"], r["variant"]): r for r in PD.inner_cv_plan_check(_with_days(OLD_SMOKE))}
    assert not old[(2024, "plain")]["ok"] and not old[(2024, "cut")]["ok"]
    assert old[(2024, "plain")]["usable_blocks"] == ["2023-05", "2025-05"] and old[(2024, "plain")]["calendar_months"] == [5]
    assert all(old[(s, v)]["ok"] for s in (2023, 2025) for v in ("plain", "cut"))     # May + October
    one = PD.inner_cv_plan_check(_with_days([["2023-05-20", "2023-05-22"], ["2024-07-20", "2024-07-22"]]))
    bad = {(r["heldout_season"], r["variant"]) for r in one if not r["ok"]}
    assert {(2023, "plain"), (2024, "plain")} <= bad and (2025, "plain") not in bad   # 1 block vs 2 blocks
    same = PD.inner_cv_plan_check(_with_days([["2023-05-20", "2023-05-22"], ["2024-05-20", "2024-05-22"]]))
    assert not {(r["heldout_season"], r["variant"]): r["ok"] for r in same}[(2025, "plain")]  # 2 blocks, 1 month
    allcut = PD.inner_cv_plan_check(_with_days([["2023-05-01", "2023-05-05"], ["2024-10-20", "2024-10-25"],
                                                ["2025-07-01", "2025-07-02"]]))
    st = {(r["heldout_season"], r["variant"]): r["ok"] for r in allcut}
    assert st[(2025, "plain")] and not st[(2025, "cut")]                      # 2023-05 / 2024-10 rows all cut


def _labels(ranges, dry_month=None, thrs=(2.5,)):
    rows = []
    for a, b in ranges:
        for d in pd.date_range(a, b, freq="D", tz="UTC"):
            t0 = pd.date_range(d, periods=48, freq="30min")
            y = np.zeros(48, "int8") if (d.year * 100 + d.month) == dry_month else (np.arange(48) % 5 == 0).astype("int8")
            for L in (0, 30, 60):
                rows.append(pd.DataFrame({"season": d.year, "t0": t0, "lead_min": L, "y_2p5": y,
                                          "train_month": not (d.year == 2025 and d.month == 10)}))
    return pd.concat(rows, ignore_index=True)


def test_inner_cv_label_check_catches_old_dry_block():
    c = _with_days(OLD_SMOKE)
    lab = _labels(OLD_SMOKE, dry_month=202305)      # 2023-05-13..18: no city window >= 2.5 mm/h (P2B events)
    r = PD.inner_cv_label_check(c, lab)
    bad = {(x["heldout_season"], x["variant"]) for x in r if not x["ok"]}
    assert (2024, "plain") in bad and (2024, "cut") in bad                     # the failing fit-m0-s2024-...
    new = SMOKE["scope"]["day_ranges"]
    assert all(x["ok"] for x in PD.inner_cv_label_check(SMOKE, _labels(new)))


@pytest.mark.parametrize("ranges", [OLD_SMOKE, [["2023-05-20", "2023-05-22"], ["2024-05-20", "2024-05-22"]]])
def test_main_refuses_infeasible_days(tmp_path, monkeypatch, ranges):
    bad = _with_days(ranges)
    monkeypatch.setattr(PD, "load_cfg", lambda smoke=False, path=None: json.loads(json.dumps(bad)))
    assert C.cli(CT.main, ["--job", "p4d_smoke", "--store", f"local:{tmp_path / 's'}", "--run-id", "x",
                           "--plan"]) == 2
    assert not (tmp_path / "s").exists() or not any((tmp_path / "s").rglob("parts.json"))
