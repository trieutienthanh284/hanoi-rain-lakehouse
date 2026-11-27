"""P4F/P4G job structure (no data): pre-registration hash guard (D214), unit graph, oracle isolation (P4G.0), R2 write
prefixes exactly as approved (CLAUDE.md §8), install rules of fetch_run, planning limits, final job refusal."""
import json
import re
from pathlib import Path

import pytest
import yaml

from src.models import cloud_train as CT
from src.models import fetch_run as FR
from src.models import p4fg as FG
from src.models import p4fg_replay as R

ROOT = Path(__file__).resolve().parents[1]
FULL = FG.load_cfg(smoke=False)
SMOKE = FG.load_cfg(smoke=True)


def test_prereg_hash_matches_decisions_and_guards(tmp_path):
    raw = yaml.safe_load(FG.CFG_PATH.read_text(encoding="utf-8"))
    assert FG.prereg_sha(raw) == FG.PREREG_SHA
    dec = (ROOT / "docs/agent_runs/DECISIONS.md")
    if dec.is_file():
        assert FG.PREREG_SHA in dec.read_text(encoding="utf-8")
    raw["pre_registered"]["selection"]["accept_all"]["every_frame_csi_7p6_diff_min"] = -0.02
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(RuntimeError, match="stop and ask"):
        FG.load_cfg(path=p)


def test_smoke_reuses_the_full_gold_and_never_changes_gold_keys():
    assert FG.gold_id(SMOKE) == FG.gold_id(FULL) == "a4a6329bcf13"
    assert FG.leads(SMOKE) == [30] and FG.thresholds(SMOKE) == [2.5, 7.6] and FG.folds(SMOKE) == [2025]
    assert FG.folds(FULL) == [2024, 2025] and FG.leads(FULL) == [0, 30, 60]
    assert FULL["lightgbm"]["learning_rate"] == 0.06 and FULL["inner_cv"]["max_rounds"] == 1500
    assert FULL["inner_cv"]["early_stopping_rounds"] == 75
    assert len(FULL["hp_grid_points"]) == 9 and [15, 500] in FULL["hp_grid_points"]


def test_unit_graph():
    us = {u.id: u for u in FG.units(FULL)}
    fits = [u for u in us.values() if u.kind == "fit"]
    assert len(fits) == 228
    t5 = [u for u in fits if u.stage == FG.STAGE5]
    assert t5 and all("hp-select" in u.deps for u in t5)
    assert all(re.fullmatch(r"fit-b15x500(a|c|ac)-s\d+-lead\d+-thr\dp\d", u.deps[1]) for u in t5)
    assert all(d in us for u in us.values() for d in u.deps)
    hp = us["hp-select"]
    assert len(hp.deps) == 9 * 2 * 3 and all("thr2p5" in d for d in hp.deps)
    assert all(d.startswith("orc-") or d == "g2" for u in fits if u.params["spec"] == "orc" for d in u.deps)
    assert {"cmp", "fss", "figs", "compare-done", "dtcheck", "fssv1"} <= set(us)
    s = {u.id for u in FG.units(SMOKE)}
    assert {"finalfit-smoke", "replay-smoke"} <= s and "finalfit-smoke" not in us
    specs = {u.params["spec"] for u in fits}
    assert specs >= {"v1b", "b15x500", "b31x200", "b63x2000", "b15x500a", "b15x500c", "b15x500ac", "ta", "tc", "tac",
                     "orc", "reg"}
    assert not [u for u in fits if u.params["thr"] == 0.5 and u.params["spec"] not in ("v1b", "b15x500")]


def test_oracle_isolation():
    oracle_names = ["oracle_gc_target", "oracle_raw_target", "oracle_gc_target_nb3", "oracle_raw_target_nb3"]
    for sp, _, _ in FG.fit_plan(FULL):
        dd = FG.spec_def(sp)
        assert bool(dd.get("oracle")) == (sp == "orc")
        if dd["conv"]:
            assert not [c for c in FG.conv_feature_names(FULL, dd["conv"]) if "oracle" in c]
    with pytest.raises(RuntimeError, match="oracle"):
        FG.assert_no_oracle(["gc_k0", "oracle_gc_target"], "b15x500")
    FG.assert_no_oracle(["gc_k0", "oracle_gc_target"], "orc")
    src = (ROOT / "src/models/predict.py").read_text(encoding="utf-8")
    assert not [n for n in oracle_names if n in src]
    rep = (ROOT / "src/models/p4fg_replay.py").read_text(encoding="utf-8")
    assert "oracle_gc_target" not in rep and "final fit of an oracle" in rep


def test_r2_prefixes_exactly_approved(tmp_path):
    exp = {"p4fg": ("lake/v1/runs/p4fg/", "lake/v1/gold/met_training/table_v3"),
           "p4fg_smoke": ("lake/v1/runs/p4fg_smoke/", "lake/v1/gold/met_training/table_v3"),
           "p4fg_final": ("lake/v1/runs/p4fg_final/", "lake/v1/serving/replay/pkg=2026-11/"),
           "nwpconv_backfill": ("lake/v1/runs/nwpconv_backfill/", "lake/v1/bronze/openmeteo_conv/",
                                "lake/v1/silver/nwp_conv/")}
    for job, pre in exp.items():
        st = CT.open_run_store(f"local:{tmp_path / job}", job)
        assert set(st.allowed) == set(pre)
        for bad in ("lake/v1/gold/met_training/table_v2/gid=x/a.parquet",
                    "lake/v1/gold/met_training/table_v2_nwp/gid=x/a.parquet", "lake/v1/models/v1/x.txt",
                    "lake/v1/serving/replay/pkg=2026-12/x.json"):
            with pytest.raises(Exception):
                st.put(bad, b"x")
    assert FULL["conv_store"]["gold_prefix"].startswith("lake/v1/gold/met_training/table_v3")


def test_fetch_run_installs_only_candidate_models():
    assert {"p4fg", "p4fg_smoke", "nwpconv_backfill"} <= set(FR.MIRROR_ONLY_JOBS)
    assert FR.JOB_INSTALL_PREFIXES["p4fg_final"] == ("models/v2_candidate/",)


def test_planning_within_limits():
    us = FG.units(FULL)
    pl = FULL["pre_registered"]["planning"]
    rec = FG.choose_parts(FULL, us, margin=1.0)
    assert rec["max_part_minutes"] <= pl["part_target_minutes"] <= pl["part_hard_limit_minutes"]
    e = FG.plan_estimate(SMOKE, FG.units(SMOKE), 2)
    assert e["max_part_minutes"] <= pl["part_target_minutes"]
    for i in (1, 2):
        mine = FG.part_units(us, i, 2, FULL)
        assert mine and not [u for u in mine if u.kind == "gate"]
    allw = [u.id for i in range(1, 5) for u in FG.part_units(us, i, 4, FULL)]
    assert sorted(allw) == sorted(u.id for u in us if u.kind != "gate") and len(set(allw)) == len(allw)


def test_inner_cv_plan_ok():
    assert all(r["ok"] for r in FG.inner_cv_plan_check(FULL))
    assert all(r["ok"] for r in FG.inner_cv_plan_check(SMOKE))


def test_final_job_refuses_without_selection(tmp_path):
    st = CT.open_run_store(f"local:{tmp_path / 's'}", "p4fg_final")
    with pytest.raises(RuntimeError, match="latest.json missing"):
        R.read_selection(st, FULL)
    p = tmp_path / "s" / "lake/v1/runs/p4fg/latest.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"job": "p4fg", "run_id": "abc"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="refuses to run"):
        R.read_selection(st, FULL)
    assert [u.id for u in R.final_units(FULL, None)] == ["sel"]
    rc = CT.main(["--job", "p4fg_final", "--store", f"local:{tmp_path / 's'}"])
    assert rc == 2


def test_final_units_with_selection():
    sel = {"model2": "tuned+a", "model2_specs": {"2024": "ta", "2025": "ta"},
           "hp_select": {"2024": {"setting": [31, 200], "totals": {"15x500": 2.0, "31x200": 1.0}},
                         "2025": {"setting": [31, 200], "totals": {"15x500": 2.0, "31x200": 1.0}}}}
    us = R.final_units(FULL, sel)
    ids = [u.id for u in us]
    assert sum(i.startswith("fit-b31x200a-") and i.endswith("thr0p5") for i in ids) == 6
    assert sum(i.startswith("final-model2") for i in ids) == 9 and sum(i.startswith("final-model1b") for i in ids) == 9
    assert {"replay-yagi", "replay-wet0930", "replay-wettest3", "replay-matmo", "replay-publish", "final-done"} <= set(ids)
    same = R.final_units(FULL, sel | {"model2_specs": {"2024": "b15x500", "2025": "b15x500"}})
    assert not [u for u in same if u.kind == "loso05"] and not [u for u in same if "model2" in u.id]


def test_reuse_unit_refuses_missing_gold(tmp_path):
    st = CT.open_run_store(f"local:{tmp_path / 's'}", "p4fg_smoke")
    ctx = FG.Ctx(SMOKE, st, "p4fg_smoke", "x", tmp_path / "m", print)
    u = [x for x in FG.units(SMOKE) if x.kind == "feat"][0]
    with pytest.raises(RuntimeError, match="never builds"):
        FG.run_reuse(u, ctx)


def test_workflow_lists_jobs_and_tests():
    wf = yaml.safe_load((ROOT / ".github/workflows/train.yml").read_text(encoding="utf-8"))
    opts = wf[True]["workflow_dispatch"]["inputs"]["job"]["options"]
    assert {"p4fg_smoke", "p4fg", "p4fg_final", "nwpconv_backfill"} <= set(opts)
    txt = (ROOT / ".github/workflows/train.yml").read_text(encoding="utf-8")
    for t in ("test_rss_gate.py", "test_p4fg_rules.py", "test_p4fg_job.py", "test_nwpconv_backfill.py",
              "test_predict_v2_candidate.py", "test_p4fg_fit_synthetic.py"):
        assert t in txt


def test_display_names_cover_the_figures():
    dn = FULL["display_names"]
    assert dn["v1"] == "v1 (chính thức)" and dn["m1b"] == "Model 1-B" and dn["m2"] == "Model 2 (ứng viên)"
    assert dn["persistence"] == "Persistence" and "không phục vụ" in dn["oracle_b"]
    assert set(FULL["axis_labels"]) >= {"probability", "observed_frequency", "neighbourhood"}


def test_relabel_p4e_reads_the_p4e_folder_inside_env(tmp_path, monkeypatch):
    """S2 fix D218-1: inside FG.env() (PE.OUT = reports/p4fg) the P4E re-render must read reports/p4e/... of the P4E
    run; the smoke failed with FileNotFoundError because p4d_env() copied reports/p4fg into PD.OUT."""
    from types import SimpleNamespace
    from src.models import p4d as PD
    from src.models import p4e as PE
    from src.models import p4fg_gates as G

    class Stop(Exception):
        pass
    seen = []

    def fake_base_rows(ctx, s, L):
        seen.append((PD.OUT, ctx.job, ctx.run_id))
        raise Stop
    monkeypatch.setattr(PD, "base_rows", fake_base_rows)
    cfg = FG.load_cfg(smoke=True)
    ctx = SimpleNamespace(cfg=cfg, store=None, mirror=tmp_path, log=lambda *a, **k: None)
    with FG.env():
        with pytest.raises(Stop):
            G.relabel_p4e(ctx, None, 10, 0, 0.9)
        assert PE.OUT == FG.OUT and PD.OUT == FG.OUT                    # env() state restored after the P4E reads
    src = cfg["sources"]["p4e_run"]
    assert seen == [("reports/p4e", src["job"], src["run_id"])]
    assert PE.OUT == "reports/p4e"


def test_replay_manifests_record_the_d217_5_window_shift():
    import inspect
    iw = R.ICON_WINDOWS
    assert iw["operational"]["window_shift_min"] == 30 and iw["operational"]["product"] == "gsmap_now_gc"
    assert "10 observed" in iw["operational"]["a6h"] and "46 observed" in iw["operational"]["a24h"]
    assert iw["hindsight"]["window_shift_min"] == 0 and "raw is never used" in iw["never"]
    src = inspect.getsource(R)
    assert src.count('"icon_windows": ICON_WINDOWS') == 2                # smoke manifest and serving manifest
