"""P4B — offline tests of the p4b / p4b_smoke cloud jobs (no network, no LightGBM, no Gold data, no training):
unit catalogue and dependencies, --part split, stage gating and resume, --final guard, smoke isolation (D96),
evaluation periods (D93), calibrators and cuts on synthetic OOF."""
import json

import numpy as np
import pandas as pd
import pytest

from src.models import cloud_train as CT
from src.models import p4b as PB
from src.realtime import common as C

CFG = PB.load_cfg()
SMOKE = PB.load_cfg(smoke=True)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("HFL_P2B_EXCLUDE_SOURCES", "none")


# ---------------------------------------------------------------- evaluation definition (D93)
def test_periods_follow_d93():
    folds = PB.cv_folds(CFG)
    assert len(folds) == 12 and folds[0][0] == pd.Timestamp("2023-05-01", tz="UTC")
    assert folds[-1][1] == pd.Timestamp("2024-11-01", tz="UTC")
    assert [f[0].month for f in folds] == [5, 6, 7, 8, 9, 10] * 2
    assert PB.period(CFG, "test") == (pd.Timestamp("2025-05-01", tz="UTC"), pd.Timestamp("2025-10-01", tz="UTC"))
    assert PB.period(CFG, "secondary") == (pd.Timestamp("2025-10-01", tz="UTC"), pd.Timestamp("2025-11-01", tz="UTC"))
    assert PB.label_source(CFG, 2025, 10) == "imerg_late"
    assert all(PB.label_source(CFG, y, m) == "imerg_final" for y, m in PB.build_months(CFG) if (y, m) != (2025, 10))
    assert len(PB.cv_folds(CFG, restricted=True)) == 6
    assert PB.thresholds(CFG) == [0.5, 2.5, 7.6] and PB.primary(CFG) == 2.5
    assert len(PB.build_months(CFG)) == 18          # 2023-05 .. 2025-10 season months


def test_smoke_never_touches_2025():
    days = PB.data_days(SMOKE)
    assert days and max(days).year == 2024
    assert len(PB.cv_folds(SMOKE)) == 2 and PB.thresholds(SMOKE) == [2.5]
    assert len(PB.grid_settings(SMOKE)) == 1


# ---------------------------------------------------------------- units, dependencies, parts
@pytest.mark.parametrize("cfg", [CFG, SMOKE], ids=["full", "smoke"])
def test_unit_catalogue(cfg):
    us = PB.units(cfg)
    ids = [u.id for u in us]
    assert len(ids) == len(set(ids))
    pos = {u.id: k for k, u in enumerate(us)}
    for u in us:
        assert all(d in pos and pos[d] < pos[u.id] for d in u.deps), u.id
    assert us[-1].id == "final-test" and us[-1].deps == ("oof-report",)
    assert [u.id for u in us if u.kind == "gate"] == ["g2", "sel-features", "sel-setting", "oof-select", "oof-report"]


def test_full_catalogue_counts():
    us = PB.units(CFG)
    kinds = pd.Series([u.kind for u in us]).value_counts().to_dict()
    assert kinds == {"feat": 18, "gate": 5, "cv": 9 + 12 + 24 + 12 + 18, "refit": 18, "final": 1}
    assert sum(u.id.startswith("cv-g") for u in us) == 8 * 3


@pytest.mark.parametrize("n", [1, 2, 3, 4, 7])
def test_parts_cover_every_work_unit_once(n):
    us = PB.units(CFG)
    work = [u.id for u in us if u.kind not in ("gate", "final")]
    got = [u.id for i in range(1, n + 1) for u in PB.part_units(us, i, n)]
    assert sorted(got) == sorted(work) and len(got) == len(set(got))
    if n == 3:                                      # every part gets a share of every stage
        for i in range(1, 4):
            assert {u.stage for u in PB.part_units(us, i, 3)} == {1, 3, 4, 5, 6}


# ---------------------------------------------------------------- driver with fake units
def _fake(monkeypatch, tmp_path, calls):
    monkeypatch.setattr(CT, "ROOT", tmp_path / "ws")

    def run_unit(unit, ctx):
        calls.append(unit.id)
        return [], {"fake": True}
    monkeypatch.setattr(PB, "run_unit", run_unit)


def _log(capsys, event):
    out = capsys.readouterr().out
    return [json.loads(x) for x in out.splitlines() if x.startswith("{") and f'"{event}"' in x], out


def test_smoke_runs_everything_in_its_own_prefix(tmp_path, capsys, monkeypatch):
    calls = []
    _fake(monkeypatch, tmp_path, calls)
    store = f"local:{tmp_path / 'store'}"
    assert C.cli(CT.main, ["--job", "p4b_smoke", "--store", store, "--run-id", "s1"]) == 0
    us = PB.units(SMOKE)
    assert calls == [u.id for u in us]              # stage order, final unit included (2024 pseudo-test)
    st = CT.open_run_store(store, "p4b_smoke")
    assert CT.done_units(st, "p4b_smoke", "s1") == {u.id for u in us}
    keys = st.list("lake/v1/")
    assert keys and all(k.startswith(("lake/v1/runs/p4b_smoke/", "lake/v1/gold/met_training/")) for k in keys)
    assert PB.OOF_DONE_KEY not in keys
    # the full job sees none of the smoke markers (separate prefix and run id)
    full = CT.open_run_store(store, "p4b")
    assert CT.done_units(full, "p4b", "s1") == set()
    calls.clear()
    assert C.cli(CT.main, ["--job", "p4b_smoke", "--store", store, "--run-id", "s1"]) == 0
    assert calls == []                              # resumed: nothing again


def test_smoke_and_full_run_ids_differ():
    assert CT.run_id_for("p4b") != CT.run_id_for("p4b_smoke")
    assert CT.run_prefix("p4b_smoke", "x") == "lake/v1/runs/p4b_smoke/x/"


def test_smoke_can_never_write_oof_done(tmp_path, capsys):
    st = CT.open_run_store(f"local:{tmp_path}", "p4b_smoke")
    for key in (PB.OOF_DONE_KEY, "lake/v1/runs/p4b/r/markers/x.done.json", "lake/v1/data/gold/x"):
        with pytest.raises(C.KeyRefused):
            st.put(key, b"{}")
    rc = C.cli(CT.main, ["--job", "p4b_smoke", "--store", f"local:{tmp_path}", "--write-oof-done", "--approve-final"])
    assert rc == 2 and "refused" in capsys.readouterr().out
    assert not (tmp_path / PB.OOF_DONE_KEY).exists()


def test_full_job_stops_at_oof_report_without_final(tmp_path, capsys, monkeypatch):
    calls = []
    _fake(monkeypatch, tmp_path, calls)
    store = f"local:{tmp_path / 'store'}"
    assert C.cli(CT.main, ["--job", "p4b", "--store", store, "--run-id", "r1"]) == 0
    assert calls[-1] == "oof-report" and "final-test" not in calls
    st = CT.open_run_store(store, "p4b")
    assert json.loads(st.get("lake/v1/runs/p4b/r1/summary.json"))["stage"].startswith("OOF complete")


def test_parallel_parts_wait_for_each_other(tmp_path, capsys, monkeypatch):
    calls = []
    _fake(monkeypatch, tmp_path, calls)
    store = f"local:{tmp_path / 'store'}"
    us = PB.units(CFG)
    assert C.cli(CT.main, ["--job", "p4b", "--part", "1/2", "--store", store, "--run-id", "r1"]) == 0
    assert calls == [u.id for u in PB.part_units(us, 1, 2) if u.stage == 1]   # stops: part 2 not done
    ev, _ = _log(capsys, "waiting_for_other_parts")
    assert ev and "feat-2023-06" in ev[0]["missing"]
    calls.clear()
    assert C.cli(CT.main, ["--job", "p4b", "--part", "2/2", "--store", store, "--run-id", "r1"]) == 0
    assert calls[:9] == [u.id for u in PB.part_units(us, 2, 2) if u.stage == 1] and "g2" in calls
    assert C.cli(CT.main, ["--job", "p4b", "--part", "1/2", "--store", store, "--run-id", "r1"]) == 0


# ---------------------------------------------------------------- --final guard
def _oof_done(tmp_path, obj):
    st = CT.oof_done_store(f"local:{tmp_path}")
    st.put_json(PB.OOF_DONE_KEY, obj, overwrite=True)


@pytest.mark.parametrize("obj,reason", [
    (None, "does not exist"),
    ({"run_id": "r1", "final_approved": False}, "final_approved is not true"),
    ({"run_id": "r1", "final_approved": "yes"}, "final_approved is not true"),
    ({"run_id": "other", "final_approved": True}, "not this code's run"),
])
def test_final_refused_without_approval(tmp_path, capsys, monkeypatch, obj, reason):
    calls = []
    _fake(monkeypatch, tmp_path, calls)
    if obj is not None:
        _oof_done(tmp_path, obj)
    rc = C.cli(CT.main, ["--job", "p4b", "--final", "--store", f"local:{tmp_path}", "--run-id", "r1"])
    ev, _ = _log(capsys, "final_refused")
    assert rc == 2 and calls == [] and reason in ev[0]["reason"]


def test_write_oof_done_then_final_runs_once(tmp_path, capsys, monkeypatch):
    calls = []
    _fake(monkeypatch, tmp_path, calls)
    store = f"local:{tmp_path}"
    with pytest.raises(SystemExit):                 # no OOF report yet
        CT.main(["--job", "p4b", "--store", store, "--run-id", "r1", "--write-oof-done"])
    assert C.cli(CT.main, ["--job", "p4b", "--store", store, "--run-id", "r1"]) == 0
    assert C.cli(CT.main, ["--job", "p4b", "--store", store, "--run-id", "r1", "--write-oof-done"]) == 0
    st = CT.open_run_store(store, "p4b")
    assert PB.check_final_guard(st, "r1") == (False, "final_approved is not true (the user has not approved the final test)")
    assert C.cli(CT.main, ["--job", "p4b", "--store", store, "--run-id", "r1", "--write-oof-done",
                           "--approve-final"]) == 0
    assert PB.check_final_guard(st, "r1")[0]
    calls.clear()
    assert C.cli(CT.main, ["--job", "p4b", "--final", "--store", store, "--run-id", "r1"]) == 0
    assert calls == ["final-test"]
    calls.clear()
    assert C.cli(CT.main, ["--job", "p4b", "--final", "--store", store, "--run-id", "r1"]) == 0
    assert calls == []                              # once


# ---------------------------------------------------------------- calibration and cuts on synthetic OOF
def _synthetic_oof(n=6000, seed=1):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    p_true = 1 / (1 + np.exp(-(2 * z - 2.5)))
    y = (rng.random(n) < p_true).astype("int8")
    p_raw = 1 / (1 + np.exp(-(0.6 * (2 * z - 2.5) + 0.8)))          # mis-calibrated but monotone
    days = np.repeat([f"2024-07-{d:02d}" for d in range(1, 31)], n // 30)
    return pd.DataFrame({"y": y, "p_raw": p_raw.astype("float32"), "fold": np.arange(n) % 4, "day": days,
                         "p_clim": np.full(n, y.mean(), "float32")})


def test_calibrators_and_crossfit():
    o = _synthetic_oof()
    for m in ("platt", "isotonic"):
        cal = PB.fit_calibrator(m, o["p_raw"], o["y"])
        json.dumps(cal)                             # serialisable (calibration.json)
        pc = PB.apply_calibrator(cal, o["p_raw"].to_numpy())
        assert ((pc >= 0) & (pc <= 1)).all()
        cf = PB.crossfit(m, o)
        assert np.isfinite(cf).all()
        assert ((cf - o["y"]) ** 2).mean() < ((o["p_raw"] - o["y"]) ** 2).mean()
    assert np.array_equal(PB.apply_calibrator({"method": "raw"}, o["p_raw"].to_numpy()), o["p_raw"].to_numpy())


def test_calibration_choice_and_cut_rules():
    o = _synthetic_oof()
    ctx = type("X", (), {"cfg": CFG})()
    ch, res, p = PB.choose_calibration(ctx, o)
    assert ch in ("platt", "isotonic") and res[ch]["hi"] < 0       # clearly mis-calibrated -> not raw
    c = PB.choose_cut(ctx, o, p, 2.5)
    assert c["rule"] in ("max_csi", "freq_bias_1.0") and 0 < c["cut"] < 1
    if not (0.8 <= c["max_csi"]["bias"] <= 1.5):
        assert "bias_1" in c
    h = PB.choose_cut(ctx, o, p, 7.6)
    assert h["rule"] == "freq_bias_1.5" and abs(h["bias"] - 1.5) < 0.1


# ---------------------------------------------------------------- the smoke final never reads 2025
def test_smoke_final_reads_only_2024_rows(tmp_path, monkeypatch):
    """Runs the whole final unit of p4b_smoke with fake models and a fake row loader that records every period
    requested; every row read must lie in 2024 and no pilot (2025 test-season) prediction file may be fetched."""
    ws = tmp_path / "ws"
    monkeypatch.setattr(PB, "ROOT", ws)
    cfg = PB.load_cfg(smoke=True)
    lim = pd.Timestamp("2025-01-01", tz="UTC")
    for _, b in PB.final_periods(cfg).values():
        assert b <= lim
    assert not PB.pilot_comparison_enabled(cfg) and PB.pilot_comparison_enabled(CFG)
    PB.write_json(ws / "reports/p4b/select/selection.json", {"ship_B": True})
    for v in "AB":
        for L in (0, 30, 60):
            PB.write_json(PB.model_dir(v, L, 2.5) / "meta.json",
                          {"features": ["gc_k0"] + (["nwp_gfs_near_h0"] if v == "B" else []), "cut": {"cut": 0.5}})
    seen, fetched = [], []
    rng = np.random.default_rng(0)

    def fake_rows(ctx, table, periods, cols, lead=None):
        seen.extend(periods)
        t0 = pd.DatetimeIndex(np.concatenate([pd.date_range(a, b, freq="30min", inclusive="left").values
                                              for a, b in periods]), tz="UTC")
        n = len(t0) * 2
        df = pd.DataFrame({"pixel_id": np.tile([1, 2], len(t0)), "t0": np.repeat(t0, 2), "lead_min": lead,
                           "season": 2024, "label_rate_mm_h": rng.gamma(0.5, 3.0, n)})
        df["frame_start"] = df["t0"] + pd.to_timedelta(df["lead_min"], unit="min")
        df["target_time"] = df["frame_start"] + pd.Timedelta(minutes=30)
        df["y_2p5"] = (df["label_rate_mm_h"] >= 2.5).astype("int8")
        for c in cols:
            if c not in df:
                df[c] = rng.gamma(0.5, 2.0, n).astype("float32")
        df["day"] = df["t0"].dt.tz_convert("Asia/Ho_Chi_Minh").dt.date.astype(str)
        return df

    def fake_cal(md, X):
        pr = rng.random(len(X))
        return pr, pr, json.loads((md / "meta.json").read_text(encoding="utf-8"))

    monkeypatch.setattr(PB, "load_rows", fake_rows)
    monkeypatch.setattr(PB, "_calibrated", fake_cal)
    monkeypatch.setattr(PB, "ensure_table", lambda ctx, t: None)
    st = CT.open_run_store(f"local:{tmp_path / 'store'}", "p4b_smoke")
    ctx = PB.Ctx(cfg, st, "p4b_smoke", "s1", tmp_path / "m", print)
    monkeypatch.setattr(ctx, "fetch_lake", lambda rels, pre: fetched.extend(rels) or 0)
    rels, m = PB.run_final(ctx)
    assert seen and all(b <= lim for _, b in seen)                         # 2024 rows only
    assert not any("pilot_v0" in r or "2025" in r for r in fetched)          # no pilot / 2025 file
    met = pd.read_csv(ws / "reports/p4b/final/metrics.csv")
    assert "pilot_v0" not in set(met["model"]) and set(met["period"]) == {"test", "secondary"}
    assert {"v1", "v1_A", "v1_B", "pers", "calpers", "clim", "logit"} <= set(met["model"])


# ---------------------------------------------------------------- NaN intervals never pass a decision (D101)
def _d(csi=(0.0, -0.01, 0.01), brier=(0.0, -0.001, 0.001)):
    return {"csi": dict(zip(("diff", "lo", "hi"), csi)), "brier": dict(zip(("diff", "lo", "hi"), brier))}


def test_ship_rule_fails_loudly_on_nan():
    assert PB.b_worse_than_a(_d(), 0) is False                                   # interval contains 0
    assert PB.b_worse_than_a(_d(brier=(0.002, 0.001, 0.003)), 0) is True          # Brier significantly higher
    assert PB.b_worse_than_a(_d(csi=(-0.02, -0.03, -0.01)), 0) is True            # CSI significantly lower
    for bad in (_d(csi=(0.0, float("nan"), 0.01)), _d(brier=(0.0, -0.001, float("nan"))),
                _d(csi=(float("nan"),) * 3)):
        with pytest.raises(RuntimeError, match="NaN interval"):
            PB.b_worse_than_a(bad, 30)


def test_final_verdict_marks_nan_intervals():
    w, b, n = PB.win_worse(_d(csi=(0.0, float("nan"), float("nan"))))
    assert (w, b, n) == (False, False, True)
    assert PB.win_worse(_d(csi=(0.05, 0.02, 0.08), brier=(-0.002, -0.003, -0.001))) == (True, False, False)
    assert PB.verdict_of([("a", True), ("b", False)], [("a", False), ("b", False)], [("a", False), ("b", True)]) \
        == "UNDETERMINED"
    assert PB.verdict_of([("a", True)], [("a", False)], [("a", False)]) == "BEATS_ALL"
    assert PB.verdict_of([("a", False), ("b", False)], [("a", True), ("b", False)], [("a", False), ("b", True)]) \
        == "LOSES"


def test_fetch_smoke_is_mirror_only(tmp_path):
    from src.models import fetch_run as FR
    st = CT.open_run_store(f"local:{tmp_path / 'store'}", "p4b_smoke")
    pre = CT.run_prefix("p4b_smoke", "s9")
    st.put(pre + "files/models/v1/A/lead0_thr2p5/model.txt", b"smoke-model")
    st.put(pre + "markers/final-test.done.json", b"{}")
    dest = tmp_path / "repo"
    log = FR.fetch(st, "p4b_smoke", "s9", dest)
    assert log["installed"] == [] and not (dest / "models").exists()
    assert (dest / "reports/p4b_smoke/cloud/s9/files/models/v1/A/lead0_thr2p5/model.txt").read_bytes() == b"smoke-model"
