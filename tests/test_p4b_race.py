"""P4B post-final change (plan step 4): race fix of REPORT_P4B §3.5, job-specific step summary, label source in
periods.json, --race-drill. Offline: local store, fake units, no LightGBM, no Gold data."""
import json

import pandas as pd
import pytest

from src.models import cloud_train as CT
from src.models import p4b as PB
from src.realtime import common as C

CFG = PB.load_cfg()
SMOKE = PB.load_cfg(smoke=True)


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("HFL_P2B_EXCLUDE_SOURCES", "none")
    monkeypatch.setattr(CT, "ROOT", tmp_path / "ws")


def _unit(uid="g", kind="gate"):
    return CT.Unit(uid, kind)


def _out(tmp_path, rel, data: bytes):
    p = CT.ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return rel


# ---------------------------------------------------------------- commit_unit
def test_commit_won_then_skipped_without_upload(tmp_path):
    st = CT.open_run_store(f"local:{tmp_path / 's'}", "p4b")
    mirror = tmp_path / "m"
    rel = _out(tmp_path, "reports/p4b/x.csv", b"first")
    assert CT.commit_unit(st, "p4b", "r", _unit(), [rel], {}, mirror)["state"] == "won"
    key = CT.file_key("p4b", "r", rel)
    _out(tmp_path, rel, b"second part, other bytes")
    c = CT.commit_unit(st, "p4b", "r", _unit(), [rel], {}, mirror)
    assert c == {"state": "skipped", "mismatch": []}
    assert st.get(key) == b"first"                    # the loser never overwrites the winner's outputs


@pytest.mark.parametrize("same,expect", [(True, []), (False, ["lake/v1/runs/p4b/r/files/reports/p4b/x.csv"])])
def test_commit_lost_race_reports_mismatch(tmp_path, monkeypatch, same, expect):
    """The other part writes the marker between our existence check and our marker write."""
    st = CT.open_run_store(f"local:{tmp_path / 's'}", "p4b")
    rel = _out(tmp_path, "reports/p4b/x.csv", b"A")
    CT.commit_unit(st, "p4b", "r", _unit(), [rel], {}, tmp_path / "m1")      # the winner
    monkeypatch.setattr(CT, "marker_exists", lambda *a: False)              # our check ran before it existed
    _out(tmp_path, rel, b"A" if same else b"B")
    c = CT.commit_unit(st, "p4b", "r", _unit(), [rel], {}, tmp_path / "m2")
    assert c["state"] == "lost" and c["mismatch"] == expect


# ---------------------------------------------------------------- driver: summary.json / done.json only by the winner
def _fake(monkeypatch, calls, on_unit=None):
    def run_unit(unit, ctx):
        calls.append(unit.id)
        if on_unit:
            on_unit(unit, ctx)
        rel = f"reports/p4b/out/{unit.id}.txt"
        p = CT.ROOT / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(unit.id)
        return [rel], {"fake": True}
    monkeypatch.setattr(PB, "run_unit", run_unit)


def _other_part_commits(store_spec, uid):
    """Simulate a parallel part that finishes the same gate while this part is computing it."""
    def hook(unit, ctx):
        if unit.id == uid:
            st = CT.open_run_store(store_spec, "p4b")
            st.put_json(CT.marker_key("p4b", "r1", uid), {"unit": uid, "outputs": {}, "by": "other part"})
    return hook


def test_oof_report_lost_to_other_part_writes_no_summary(tmp_path, capsys, monkeypatch):
    store = f"local:{tmp_path / 'store'}"
    calls = []
    _fake(monkeypatch, calls, _other_part_commits(store, "oof-report"))
    assert C.cli(CT.main, ["--job", "p4b", "--store", store, "--run-id", "r1"]) == 0
    st = CT.open_run_store(store, "p4b")
    assert "oof-report" in calls
    assert not st.list("lake/v1/runs/p4b/r1/summary.json")                 # the other part writes it, not us
    assert not st.list(CT.file_key("p4b", "r1", "reports/p4b/out/oof-report.txt"))   # no upload after its marker
    out = capsys.readouterr().out
    assert '"marker_exists_skip_upload"' in out and '"unit": "oof-report"' in out


def test_final_lost_race_identical_outputs_exit0_no_done(tmp_path, capsys, monkeypatch):
    store = f"local:{tmp_path / 'store'}"
    calls = []
    _fake(monkeypatch, calls)
    assert C.cli(CT.main, ["--job", "p4b", "--store", store, "--run-id", "r1"]) == 0
    st = CT.open_run_store(store, "p4b")
    assert json.loads(st.get("lake/v1/runs/p4b/r1/summary.json"))["stage"].startswith("OOF complete")
    CT.oof_done_store(f"local:{tmp_path / 'store'}").put_json(PB.OOF_DONE_KEY, {"run_id": "r1",
                                                                                "final_approved": True}, overwrite=True)

    def hook(unit, ctx):                       # the other part commits final-test with the same output bytes
        rel = "reports/p4b/out/final-test.txt"
        (CT.ROOT / rel).parent.mkdir(parents=True, exist_ok=True)
        (CT.ROOT / rel).write_text("final-test")
        o = CT.open_run_store(store, "p4b")
        outs = CT.upload_outputs(o, "p4b", "r1", [rel], tmp_path / "other")
        CT.write_marker(o, "p4b", "r1", CT.Unit("final-test", "final"), outs, {}, tmp_path / "other")
    calls.clear()
    _fake(monkeypatch, calls, lambda u, c: hook(u, c) if u.id == "final-test" else None)
    real = CT.marker_exists
    monkeypatch.setattr(CT, "marker_exists", lambda s, j, r, uid: False if uid == "final-test" else real(s, j, r, uid))
    assert C.cli(CT.main, ["--job", "p4b", "--final", "--store", store, "--run-id", "r1"]) == 0
    assert calls == ["final-test"]
    assert not st.list("lake/v1/runs/p4b/r1/done.json")                   # only the marker's writer writes it
    assert '"marker_lost"' in capsys.readouterr().out


def test_lost_race_mismatch_exit_code(tmp_path, capsys, monkeypatch):
    store = f"local:{tmp_path / 'store'}"
    calls = []

    def hook(unit, ctx):
        if unit.id == "g2":
            rel = "reports/p4b/out/g2.txt"
            (CT.ROOT / rel).parent.mkdir(parents=True, exist_ok=True)
            (CT.ROOT / rel).write_text("bytes of the other part")
            o = CT.open_run_store(store, "p4b")
            outs = CT.upload_outputs(o, "p4b", "r1", [rel], tmp_path / "other")
            CT.write_marker(o, "p4b", "r1", CT.Unit("g2", "gate"), outs, {}, tmp_path / "other")
    _fake(monkeypatch, calls, hook)
    real = CT.marker_exists
    monkeypatch.setattr(CT, "marker_exists", lambda s, j, r, uid: False if uid == "g2" else real(s, j, r, uid))
    rc = C.cli(CT.main, ["--job", "p4b", "--store", store, "--run-id", "r1"])
    out = capsys.readouterr().out
    assert rc == 3 and '"output_mismatch": 1' in out


# ---------------------------------------------------------------- race drill (p4b_smoke only)
def test_race_drill_on_finished_smoke_run(tmp_path, capsys, monkeypatch):
    store = f"local:{tmp_path / 'store'}"
    calls = []
    _fake(monkeypatch, calls)
    assert C.cli(CT.main, ["--job", "p4b_smoke", "--store", store, "--run-id", "s1"]) == 0
    st = CT.open_run_store(store, "p4b_smoke")
    before = {k: st.get(k) for k in st.list("lake/v1/runs/p4b_smoke/s1/") if not k.endswith("latest.json")}
    calls.clear()
    assert C.cli(CT.main, ["--job", "p4b_smoke", "--store", store, "--run-id", "s1", "--race-drill"]) == 0
    gates = [u.id for u in PB.units(SMOKE) if u.kind in ("gate", "final")]
    assert calls == gates                                    # every gate + final unit ran again ...
    after = {k: st.get(k) for k in st.list("lake/v1/runs/p4b_smoke/s1/") if not k.endswith("latest.json")}
    assert after == before                                   # ... and nothing on the store changed
    ev = [e for e in (json.loads(x) for x in capsys.readouterr().out.splitlines() if x.startswith("{"))
          if e["event"] == "race_drill"]
    assert len(ev) == 1 and ev[0]["ok"] is True and set(ev[0]["states"].values()) == {"skipped"}


def test_race_drill_refused_outside_smoke(tmp_path, capsys):
    for job in ("p4b", "p4b_post", "p4a6"):
        assert C.cli(CT.main, ["--job", job, "--store", f"local:{tmp_path}", "--race-drill"]) == 2


def test_race_drill_fails_on_unfinished_run(tmp_path, capsys, monkeypatch):
    store = f"local:{tmp_path / 'store'}"
    _fake(monkeypatch, [])
    rc = C.cli(CT.main, ["--job", "p4b_smoke", "--store", store, "--run-id", "s2", "--race-drill",
                         "--max-minutes", "0"])
    assert rc == 3                                           # nothing done: the drill cannot pass


# ---------------------------------------------------------------- step summary
@pytest.mark.parametrize("job,title", [("p4b", "v1 (P4B)"), ("p4b_smoke", "v1 smoke"), ("p4b_post", "post-final"),
                                       ("p4a7", "PILOT v0")])
def test_step_summary_title_per_job(job, title):
    md = CT.step_summary(job, "r9", ["u"], 0, None)
    assert md.startswith(f"### cloud_train `{job}` run `r9`") and title in md
    if job != "p4a7":
        assert "PILOT" not in md


def test_step_summary_goes_to_job_file_not_github(tmp_path, capsys, monkeypatch):
    gh = tmp_path / "gh_summary.md"
    gh.write_text("")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(gh))
    _fake(monkeypatch, [])
    store = f"local:{tmp_path / 'store'}"
    assert C.cli(CT.main, ["--job", "p4b_smoke", "--store", store, "--run-id", "s3"]) == 0
    f = CT.ROOT / "cloud_out" / "step_summary_p4b_smoke.md"
    assert f.is_file() and "`p4b_smoke` run `s3`" in f.read_text(encoding="utf-8")
    assert gh.read_text() == ""                              # only the workflow step appends it (on success)
    assert not (CT.ROOT / "cloud_out" / "step_summary_p4b.md").exists()


def test_workflow_appends_summary_only_on_success():
    wf = (PB.CODE_ROOT / ".github" / "workflows" / "train.yml").read_text(encoding="utf-8")
    assert 'rm -f "cloud_out/step_summary_${JOB}.md"' in wf
    assert 'if [ "$OUTCOME" = "success" ] && [ -f "$f" ]; then' in wf and "no summary" in wf
    assert "OUTCOME: ${{ steps.train.outcome }}" in wf
    code = (PB.CODE_ROOT / "src" / "models" / "cloud_train.py").read_text(encoding="utf-8")
    assert 'os.environ["GITHUB_STEP_SUMMARY"]' not in code and "os.environ.get(\"GITHUB_STEP_SUMMARY\")" not in code


# ---------------------------------------------------------------- periods.json label source
def test_period_record_label_source():
    t = PB.period_record(CFG, "test")
    assert t["label_source"] == "imerg_final" and list(t["label_source_by_month"]) == [
        "2025-05", "2025-06", "2025-07", "2025-08", "2025-09"]
    assert t["t0_from_utc"].startswith("2025-05-01") and t["t0_to_utc_exclusive"].startswith("2025-10-01")
    s = PB.period_record(CFG, "secondary")
    assert s["label_source"] == "imerg_late" and s["label_source_by_month"] == {"2025-10": "imerg_late"}
    assert "never used" in s["role"] and "Late" in s["label_name"]
    assert PB.period_record(SMOKE, "test")["label_source"] == "imerg_final"
    assert PB.period_record(SMOKE, "secondary")["label_source"] == "imerg_late"
    pd.Timestamp(t["t0_from_utc"])                             # parseable


# ---------------------------------------------------------------- fetch_run --verify (cloud check of the race fix)
def test_fetch_verify_detects_marker_mismatch(tmp_path, capsys):
    from src.models import fetch_run as FR
    store = f"local:{tmp_path / 'store'}"
    st = CT.open_run_store(store, "p4b_smoke")
    rel = _out(tmp_path, "reports/p4b/x.csv", b"good")
    CT.commit_unit(st, "p4b_smoke", "v1r", _unit("g2"), [rel], {}, tmp_path / "m")
    dest = tmp_path / "dest"
    args = ["--job", "p4b_smoke", "--run-id", "v1r", "--store", store, "--dest-root", str(dest), "--verify"]
    assert C.cli(FR.main, args) == 0
    key = CT.file_key("p4b_smoke", "v1r", rel)
    st.overwritable.add(key)
    st.put(key, b"overwritten by a losing part", overwrite=True)
    assert C.cli(FR.main, args) == 1
    ev = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.startswith('{"event": "verified"')]
    assert ev[0]["ok"] is True and ev[1]["ok"] is False and ev[1]["sha_mismatch"] == 1
