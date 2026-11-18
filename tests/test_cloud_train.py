"""P4C — offline tests of the cloud training entry point (no network, no LightGBM, no Gold data needed)."""
import json
from pathlib import Path

import pytest

from src.models import cloud_train as CT
from src.realtime import common as C

CFG = {"frames": {"lead_min": [0, 30, 60]}, "targets": {"thresholds_mm_h": [0.5, 2.5, 7.6]}}
SECRET = "Sup3rSecretValue-for-test-123"


@pytest.fixture(autouse=True)
def _guard(monkeypatch):
    monkeypatch.setenv("HFL_P2B_EXCLUDE_SOURCES", "imerg_late")


# ---------------------------------------------------------------- units and parts
def test_units_per_job():
    u6 = CT.job_units("p4a6", CFG)
    assert [u.kind for u in u6].count("train") == 12 and u6[-1].id == "final"
    assert sum(u.params.get("variant") == "B" for u in u6) == 9
    assert {u.id for u in u6 if u.params.get("variant") == "A_no2023"} == {
        "train-A_no2023-lead0-thr0p5", "train-A_no2023-lead30-thr0p5", "train-A_no2023-lead60-thr0p5"}
    u7 = CT.job_units("p4a7", CFG)
    assert [u.kind for u in u7].count("loso") == 27 and u7[-1].id == "final"
    assert CT.job_units("p4b", CFG) == []
    assert len({u.id for u in u6}) == len(u6) and len({u.id for u in u7}) == len(u7)


@pytest.mark.parametrize("job", ["p4a6", "p4a7"])
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 7, 12, 30])
def test_parts_cover_all_units_exactly_once(job, n):
    units = CT.job_units(job, CFG)
    trainable = [u.id for u in units if u.kind != "final"]
    got = [u.id for i in range(1, n + 1) for u in CT.part_units(units, i, n)]
    assert sorted(got) == sorted(trainable) and len(got) == len(set(got))
    assert all(u.kind != "final" for i in range(1, n + 1) for u in CT.part_units(units, i, n))


@pytest.mark.parametrize("bad", ["0/2", "3/2", "1/0", "x/2"])
def test_bad_part_refused(bad):
    with pytest.raises(ValueError):
        CT.parse_part(bad)


# ---------------------------------------------------------------- R2 key layout
def test_key_layout():
    assert CT.run_prefix("p4a7", "abc") == "lake/v1/runs/p4a7/abc/"
    assert CT.file_key("p4a7", "abc", "models/pilot_v0/LOSO_2024/lead0_thr2p5/model.txt") == \
        "lake/v1/runs/p4a7/abc/files/models/pilot_v0/LOSO_2024/lead0_thr2p5/model.txt"
    assert CT.marker_key("p4a6", "r1", "train-B-lead0-thr2p5") == "lake/v1/runs/p4a6/r1/markers/train-B-lead0-thr2p5.done.json"
    assert CT.latest_key("p4a6") == "lake/v1/runs/p4a6/latest.json"
    for bad in ("../x", "a/../../b", "/etc/x"):
        with pytest.raises(ValueError):
            CT.file_key("p4a6", "r1", bad)
    rid = CT.run_id_for("p4a6")
    assert len(rid) == 12 and rid == CT.run_id_for("p4a6") and rid != CT.run_id_for("p4a7")


def test_store_refuses_writes_outside_job_prefix(tmp_path):
    st = CT.open_run_store(f"local:{tmp_path}", "p4a6")
    st.put("lake/v1/runs/p4a6/r/x.json", b"{}")
    for key in ("lake/v1/runs/p4a7/r/x.json", "lake/v1/data/gold/x.parquet", "lake/v1/realtime/x", "other/x"):
        with pytest.raises(C.KeyRefused):
            st.put(key, b"{}")
    with pytest.raises(C.KeyRefused):                      # only latest.json may be overwritten
        st.put("lake/v1/runs/p4a6/r/x.json", b"[]", overwrite=True)
    st.put(CT.latest_key("p4a6"), b"1", overwrite=True)
    st.put(CT.latest_key("p4a6"), b"2", overwrite=True)


# ---------------------------------------------------------------- skip logic with fake markers
def _plan(capsys, tmp_path, job, part="1/1", extra=()):
    rc = C.cli(CT.main, ["--job", job, "--part", part, "--store", f"local:{tmp_path}", "--run-id", "r1", "--plan", *extra])
    out = capsys.readouterr().out
    plan = [json.loads(line) for line in out.splitlines() if line.startswith("{") and '"plan"' in line][0]
    return rc, plan, out


def test_finished_units_are_skipped(tmp_path, capsys):
    st = CT.open_run_store(f"local:{tmp_path}", "p4a7")
    for uid in ("loso-2023-lead0-thr0p5", "loso-2023-lead0-thr2p5"):
        st.put_json(CT.marker_key("p4a7", "r1", uid), {"unit": uid})
    assert CT.done_units(st, "p4a7", "r1") == {"loso-2023-lead0-thr0p5", "loso-2023-lead0-thr2p5"}
    rc, plan, _ = _plan(capsys, tmp_path, "p4a7", "1/3")
    assert rc == 0 and plan["units_in_part"] == 9 and plan["done"] == 2
    assert "loso-2023-lead0-thr0p5" not in plan["todo"] and "loso-2023-lead0-thr7p6" in plan["todo"]


def test_driver_resumes_and_writes_markers(tmp_path, capsys, monkeypatch):
    """Fake training: the driver uploads outputs under the run prefix, writes one marker per unit, and a second
    invocation trains nothing."""
    ws = tmp_path / "ws"; ws.mkdir()
    monkeypatch.setattr(CT, "ROOT", ws)
    calls = []

    def fake_train(unit, cfg, cache, lg):
        rel = f"models/pilot_v0/{unit.params['variant']}/lead{unit.params['lead']}_thr{CT.kstr(unit.params['thr'])}/model.txt"
        (ws / rel).parent.mkdir(parents=True, exist_ok=True)
        (ws / rel).write_text(unit.id)
        calls.append(unit.id)
        return [rel, "does/not/exist.txt"], {"best_iteration": 7, "cut": 0.3}

    monkeypatch.setattr(CT, "run_train", fake_train)
    store = f"local:{tmp_path / 'store'}"
    args = ["--job", "p4a6", "--part", "2/6", "--store", store, "--run-id", "r1"]
    assert C.cli(CT.main, args) == 0
    assert len(calls) == 2
    st = CT.open_run_store(store, "p4a6")
    done = CT.done_units(st, "p4a6", "r1")
    assert done == set(calls)
    files = st.list("lake/v1/runs/p4a6/r1/files/")
    assert len(files) == 2 and all(k.endswith("/model.txt") for k in files)
    assert json.loads(st.get(CT.latest_key("p4a6")))["run_id"] == "r1"
    assert (ws / "cloud_out" / "p4a6" / "r1" / "markers").is_dir()
    calls.clear()
    assert C.cli(CT.main, args) == 0
    assert calls == []                                     # resumed: nothing trained again
    out = capsys.readouterr().out
    assert '"final": "waiting"' in out                     # other parts still missing -> no final unit


def test_reserved_job_exits_2(tmp_path, capsys):
    rc = C.cli(CT.main, ["--job", "p4b", "--store", f"local:{tmp_path}"])
    assert rc == 2 and "job_reserved" in capsys.readouterr().out


# ---------------------------------------------------------------- secrets never reach the log
def test_no_secret_in_logs(tmp_path, capsys, monkeypatch):
    for name in ("R2_SECRET_ACCESS_KEY", "R2_ACCESS_KEY_ID", "GSMAP_FTP_PASSWORD"):
        monkeypatch.setenv(name, SECRET + name)
    # the run id is logged: make it contain a secret value on purpose
    rc = C.cli(CT.main, ["--job", "p4a7", "--store", f"local:{tmp_path}", "--run-id",
                         SECRET + "R2_SECRET_ACCESS_KEY", "--plan"])
    out = capsys.readouterr()
    assert rc == 0
    assert SECRET not in out.out and SECRET not in out.err and C.MASK in out.out
    md = CT.step_summary("p4a7", "r1", ["u1"], 0, {"k": "v"})
    assert SECRET not in C.redact(md)


# ---------------------------------------------------------------- P4C.4 read-back
def test_fetch_run_installs_and_archives(tmp_path):
    from src.models import fetch_run as FR
    st = CT.open_run_store(f"local:{tmp_path / 'store'}", "p4a6")
    pre = CT.run_prefix("p4a6", "r9")
    st.put(pre + "files/models/pilot_v0/B/lead30_thr2p5/model.txt", b"new-model")
    st.put(pre + "files/models/pilot_v0/B/lead0_thr2p5/model.txt", b"same")
    st.put(pre + "files/reports/p4a/compare_A_B/test_metrics.csv", b"a,b\n")
    st.put(pre + "markers/final.done.json", b"{}")
    st.put(CT.latest_key("p4a6"), json.dumps({"run_id": "r9"}).encode(), overwrite=True)
    dest = tmp_path / "repo"
    (dest / "models/pilot_v0/B/lead0_thr2p5").mkdir(parents=True)
    (dest / "models/pilot_v0/B/lead0_thr2p5/model.txt").write_bytes(b"same")
    (dest / "models/pilot_v0/B/lead30_thr2p5").mkdir(parents=True)
    (dest / "models/pilot_v0/B/lead30_thr2p5/model.txt").write_bytes(b"old-model")
    assert FR.resolve_run_id(st, "p4a6", "latest") == "r9"
    log = FR.fetch(st, "p4a6", "r9", dest)
    assert log["identical"] == 1 and log["archived"] == ["models/pilot_v0/B/lead30_thr2p5/model.txt"]
    assert (dest / "models/pilot_v0/B/lead30_thr2p5/model.txt").read_bytes() == b"new-model"
    assert (dest / "_archive/fetch_run/p4a6/r9/models/pilot_v0/B/lead30_thr2p5/model.txt").read_bytes() == b"old-model"
    assert not (dest / "reports/p4a/compare_A_B").exists()        # cloud reports stay in the mirror
    assert (dest / "reports/p4a6/cloud/r9/files/reports/p4a/compare_A_B/test_metrics.csv").is_file()
    assert (dest / "reports/p4a6/cloud/r9/markers/final.done.json").is_file()
