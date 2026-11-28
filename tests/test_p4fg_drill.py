"""D219 (user decision 2026-11-28): race-drill pass condition evaluated over ALL parts. (a) the S2b false negative
(part 2 drills the unit part 1 already claimed; both part-2 contenders yield) now passes with committed_by_other_part;
(b) a real second writer to a final key (other bytes, same bytes after the marker, or a second claim) still fails.
Offline: local store, threads, no LightGBM."""
import json
import threading
import time
from pathlib import Path

import pytest

from src.models import cloud_train as CT
from src.realtime import common as C

JOB, RUN = "p4fg_smoke", "r1"
RELS = ["reports/p4fg/compare/metrics.csv", "reports/p4fg/compare/checks.json"]


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("HFL_P2B_EXCLUDE_SOURCES", "none")
    monkeypatch.setattr(CT, "ROOT", tmp_path / "ws")


def _store(tmp_path):
    return CT.open_run_store(f"local:{tmp_path / 's'}", JOB)


def _run_unit(u):
    for r in RELS:
        p = CT.ROOT / r
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(f"content of {r}".encode())
    return RELS, {"x": 1}


def _base(du):
    return {f"reports/{JOB}/drill/{du}/{Path(r).name}": f"content of {r}".encode() for r in RELS}


def test_single_part_drill_still_passes(tmp_path):
    st = _store(tmp_path)
    out = CT.race_drill_single(st, JOB, RUN, CT.Unit("cmp", "gate"), _run_unit, tmp_path / "m", poll_seconds=0.05)
    assert out["ok"] and sorted(out["states"].values()) == ["won", "yielded"]
    assert out["claimant"] and out["winner"] and not out["committed_by_other_part"]
    assert out["claims"] == ["claims/drill-cmp-r1/g001.json"] and out["no_final_after_marker"]
    assert out["other_contenders_checked"] == 1 and out["loser_exit_code"] == 0


def test_s2b_false_negative_now_passes(tmp_path):
    """REPORT_P4FG §S2b: part 1 contender B holds the claim on drill-cmp-r1 and is still uploading when part 2 picks
    the first drill unit without a marker (r1). Both part-2 contenders wait for the marker and yield. Pre-D219 this
    was ok false (no 'won' in part 2's own pair); single-writer held, so it must pass now."""
    st = _store(tmp_path)
    du = "drill-cmp-r1"
    base = _base(du)
    p1b = {r: b + b"\n" for r, b in base.items()}                       # part 1 B bytes (the claimant)
    CT.drill_register(st, JOB, RUN, du, "p1-drillA", base)
    CT.drill_register(st, JOB, RUN, du, "p1-drillB", p1b)
    assert CT.take_claim(st, JOB, RUN, du, "p1-drillB", 30) == 1

    def part1_finishes():
        outs = {}
        for r, b in p1b.items():
            st.put(CT.file_key(JOB, RUN, r), b)
            outs[CT.file_key(JOB, RUN, r)] = C.sha256_bytes(b)
        CT.write_marker(st, JOB, RUN, CT.Unit(du, "gate"), outs, {}, tmp_path / "p1")
        for r, b in base.items():                                         # part 1 A yielded: scratch only
            st.put(CT.run_prefix(JOB, RUN) + f"scratch/p1-drillA/files/{r}", b)
    t = threading.Timer(0.6, part1_finishes)
    t.start()
    out = CT.race_drill_single(st, JOB, RUN, CT.Unit("cmp", "gate"), _run_unit, tmp_path / "m", poll_seconds=0.05)
    t.join()
    assert out["drill_unit"] == du and out["states"] == {"A": "yielded", "B": "yielded"}   # the S2b situation
    assert out["winner"] is None and out["claimant"] == "p1-drillB"
    assert out["ok"] and out["committed_by_other_part"]
    assert out["claims"] == [f"claims/{du}/g001.json"]
    assert out["final_equals_marker"] and out["final_equals_claimant"] and out["no_final_after_marker"]
    assert out["loser_bytes_in_scratch_only"] and out["other_contenders_checked"] == 3   # p1 A + own A, B
    assert out["loser_exit_code"] == 0
    for r, b in p1b.items():
        assert st.get(CT.file_key(JOB, RUN, r)) == b


def test_claimant_payload_must_match_final_bytes(tmp_path):
    """committed_by_other_part is no blanket pass: the other part's claimant must have written its own bytes."""
    st = _store(tmp_path)
    du = "drill-cmp-r1"
    base = _base(du)
    CT.drill_register(st, JOB, RUN, du, "p1-drillB", {r: b + b"\n" for r, b in base.items()})
    assert CT.take_claim(st, JOB, RUN, du, "p1-drillB", 30) == 1

    def part1_writes_other_bytes():
        outs = {}
        for r in base:
            st.put(CT.file_key(JOB, RUN, r), b"not the registered payload")
            outs[CT.file_key(JOB, RUN, r)] = C.sha256_bytes(b"not the registered payload")
        CT.write_marker(st, JOB, RUN, CT.Unit(du, "gate"), outs, {}, tmp_path / "p1")
    t = threading.Timer(0.4, part1_writes_other_bytes)
    t.start()
    out = CT.race_drill_single(st, JOB, RUN, CT.Unit("cmp", "gate"), _run_unit, tmp_path / "m", poll_seconds=0.05)
    t.join()
    assert out["committed_by_other_part"] and out["final_equals_marker"]
    assert not out["final_equals_claimant"] and not out["ok"]


@pytest.mark.parametrize("rogue", ["other_bytes", "same_bytes_after_marker", "second_claim"])
def test_real_second_writer_still_fails(tmp_path, monkeypatch, rogue):
    """A writer other than the claimant touches a final key (or takes a second claim) after the marker: fail."""
    st = _store(tmp_path)
    orig = CT.write_marker
    done = []

    def marker_then_rogue(store, job, run_id, unit, outputs, metrics, mirror):
        orig(store, job, run_id, unit, outputs, metrics, mirror)
        if done or not unit.id.startswith("drill-"):
            return
        done.append(1)
        time.sleep(0.1)                                                   # file-time resolution on Windows
        key = sorted(outputs)[0]
        if rogue == "second_claim":
            assert CT.put_if_absent(store, CT.claim_key(job, run_id, unit.id, 2), b'{"attempt": "rogue"}')
            return
        data = b"rogue bytes" if rogue == "other_bytes" else store.get(key)
        store.overwritable.add(key)
        store.put(key, data, overwrite=True)
        store.overwritable.discard(key)
    monkeypatch.setattr(CT, "write_marker", marker_then_rogue)
    out = CT.race_drill_single(st, JOB, RUN, CT.Unit("cmp", "gate"), _run_unit, tmp_path / "m", poll_seconds=0.05)
    assert done and not out["ok"]
    if rogue == "other_bytes":
        assert not out["final_equals_marker"] and not out["no_final_after_marker"]
    elif rogue == "same_bytes_after_marker":
        assert out["final_equals_marker"] and not out["no_final_after_marker"]
    else:
        assert out["claims"] == ["claims/drill-cmp-r1/g001.json", "claims/drill-cmp-r1/g002.json"]


def test_non_exclusive_claims_still_fail(tmp_path, monkeypatch):
    """Same check as test_p4e_race.test_drill_detects_a_second_writer, on the p4fg job."""
    st = _store(tmp_path)
    monkeypatch.setattr(CT, "put_if_absent", lambda store, key, data: True)
    out = CT.race_drill_single(st, JOB, RUN, CT.Unit("cmp", "gate"), _run_unit, tmp_path / "m", poll_seconds=0.05)
    assert not out["ok"] and out["claims"] == []


def test_payload_records_only_under_scratch(tmp_path):
    st = _store(tmp_path)
    CT.race_drill_single(st, JOB, RUN, CT.Unit("cmp", "gate"), _run_unit, tmp_path / "m", poll_seconds=0.05)
    keys = st.list(CT.run_prefix(JOB, RUN))
    recs = [k for k in keys if "/drill_payload/" in k]
    assert len(recs) == 2 and all(k.startswith(CT.run_prefix(JOB, RUN) + "scratch/") for k in recs)
    assert all(json.loads(st.get(k))["unit"] == "drill-cmp-r1" for k in recs)
