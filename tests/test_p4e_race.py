"""P4E.0-1 (D188 fix): exactly one writer per shared output. commit_unit_single (atomic claim, then outputs and
marker by the claim holder only), loser = 'yielded' with exit code 0 and bytes under its scratch prefix, stale-claim
takeover, adoption of the winner's bytes, extended race drill. Offline: local store, threads, no LightGBM."""
import json
import threading
import time

import pytest

from src.models import cloud_train as CT
from src.realtime import common as C

JOB, RUN = "p4e_smoke", "r1"


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("HFL_P2B_EXCLUDE_SOURCES", "none")
    monkeypatch.setattr(CT, "ROOT", tmp_path / "ws")


def _store(tmp_path):
    return CT.open_run_store(f"local:{tmp_path / 's'}", JOB)


def _write(rel, data: bytes):
    p = CT.ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return rel


def _race(st, tmp_path, payloads: dict, unit_id="cmp", kind="gate"):
    """Contenders with different bytes, released at the same instant."""
    u = CT.Unit(unit_id, kind)
    bar, res = threading.Barrier(len(payloads)), {}
    ids = {n: f"att-{n}" for n in payloads}

    def go(n):
        bar.wait()
        res[n] = CT.commit_unit_single(st, JOB, RUN, u, list(payloads[n]), {"m": n}, tmp_path / f"m{n}",
                                       payload=payloads[n], me=ids[n], poll_seconds=0.05)
    th = [threading.Thread(target=go, args=(n,)) for n in payloads]
    for t in th:
        t.start()
    for t in th:
        t.join()
    return res, ids


def test_put_if_absent_is_atomic(tmp_path):
    st = _store(tmp_path)
    key = CT.claim_key(JOB, RUN, "u", 1)
    bar, wins = threading.Barrier(8), []

    def go(i):
        bar.wait()
        if CT.put_if_absent(st, key, f"{i}".encode()):
            wins.append(i)
    th = [threading.Thread(target=go, args=(i,)) for i in range(8)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    assert len(wins) == 1 and st.get(key) == f"{wins[0]}".encode()


@pytest.mark.parametrize("trial", range(6))
def test_single_writer_under_simultaneous_commit(tmp_path, trial):
    st = _store(tmp_path)
    rels = ["reports/p4e/compare/metrics.csv", "reports/p4e/compare/diffs.csv"]
    pay = {n: {r: f"{n}:{r}:{trial}".encode() for r in rels} for n in ("A", "B", "C")}
    res, ids = _race(st, tmp_path, pay)
    states = sorted(r["state"] for r in res.values())
    assert states == ["won", "yielded", "yielded"]
    w = [n for n, r in res.items() if r["state"] == "won"][0]
    mk = json.loads(st.get(CT.marker_key(JOB, RUN, "cmp")))
    for r in rels:
        key = CT.file_key(JOB, RUN, r)
        assert st.get(key) == pay[w][r]                                     # final bytes = the winner's only
        assert mk["outputs"][key] == C.sha256_bytes(pay[w][r])              # and match the marker
    for n in res:
        if n != w:                                                           # losers: scratch only, exit code 0
            assert res[n]["mismatch"] == [] and CT._exit_code({"mismatch": res[n]["mismatch"]}) == 0
            sp = CT.run_prefix(JOB, RUN) + f"scratch/{ids[n]}/files/"
            assert {k[len(sp):] for k in st.list(sp)} == set(rels)
    assert CT.latest_claim(st, JOB, RUN, "cmp")[0] == 1                      # one claim generation


def test_fit_units_keep_no_scratch(tmp_path):
    st = _store(tmp_path)
    pay = {n: {"reports/p4e/oof/x.parquet": n.encode()} for n in ("A", "B")}
    res, ids = _race(st, tmp_path, pay, unit_id="fit-x", kind="fit")
    assert sorted(r["state"] for r in res.values()) == ["won", "yielded"]
    assert not st.list(CT.run_prefix(JOB, RUN) + "scratch/")


def test_marker_exists_skipped_and_adopted(tmp_path):
    st = _store(tmp_path)
    rel = _write("reports/p4e/compare/metrics.csv", b"winner")
    u = CT.Unit("cmp", "gate")
    assert CT.commit_unit_single(st, JOB, RUN, u, [rel], {}, tmp_path / "m")["state"] == "won"
    _write(rel, b"own last-digit variant")
    c = CT.commit_unit_single(st, JOB, RUN, u, [rel], {}, tmp_path / "m")
    assert c == {"state": "skipped", "mismatch": []}
    assert (CT.ROOT / rel).read_bytes() == b"winner"                          # local copy replaced by R2 bytes
    assert st.get(CT.file_key(JOB, RUN, rel)) == b"winner"


def test_live_claim_waits_for_marker_then_yields_and_adopts(tmp_path):
    st = _store(tmp_path)
    rel = _write("reports/p4e/compare/metrics.csv", b"mine")
    u = CT.Unit("cmp", "gate")
    assert CT.take_claim(st, JOB, RUN, "cmp", "holder", 30) == 1             # another part holds a live claim

    def holder():
        time.sleep(0.3)
        st.put(CT.file_key(JOB, RUN, rel), b"holder")
        CT.write_marker(st, JOB, RUN, u, {CT.file_key(JOB, RUN, rel): C.sha256_bytes(b"holder")}, {}, tmp_path / "h")
    t = threading.Thread(target=holder)
    t.start()
    c = CT.commit_unit_single(st, JOB, RUN, u, [rel], {}, tmp_path / "m", poll_seconds=0.05)
    t.join()
    assert c["state"] == "yielded"
    assert st.get(CT.file_key(JOB, RUN, rel)) == b"holder" and (CT.ROOT / rel).read_bytes() == b"holder"


def test_stale_claim_is_superseded_and_leftovers_replaced(tmp_path):
    st = _store(tmp_path)
    rel = "reports/p4e/compare/metrics.csv"
    old = {"unit": "cmp", "gen": 1, "attempt": "dead", "written_utc": "2026-01-01T00:00:00Z"}
    assert CT.put_if_absent(st, CT.claim_key(JOB, RUN, "cmp", 1), json.dumps(old).encode())
    st.put(CT.file_key(JOB, RUN, rel), b"partial upload of the dead holder")
    c = CT.commit_unit_single(st, JOB, RUN, CT.Unit("cmp", "gate"), [rel], {}, tmp_path / "m",
                              payload={rel: b"complete"}, stale_minutes=30)
    assert c["state"] == "won" and CT.latest_claim(st, JOB, RUN, "cmp")[0] == 2
    assert st.get(CT.file_key(JOB, RUN, rel)) == b"complete"


def test_first_generation_never_overwrites(tmp_path):
    """A final key that exists although nobody held a claim is a breach: refused, nothing overwritten."""
    st = _store(tmp_path)
    rel = "reports/p4e/compare/metrics.csv"
    st.put(CT.file_key(JOB, RUN, rel), b"foreign")
    with pytest.raises(RuntimeError, match="single-writer breach"):
        CT.commit_unit_single(st, JOB, RUN, CT.Unit("cmp", "gate"), [rel], {}, tmp_path / "m", payload={rel: b"x"})
    assert st.get(CT.file_key(JOB, RUN, rel)) == b"foreign"
    assert not CT.marker_exists(st, JOB, RUN, "cmp")


def test_race_drill_single(tmp_path):
    st = _store(tmp_path)
    rels = ["reports/p4e/compare/metrics.csv", "reports/p4e/compare/checks.json"]

    def run_unit(u):
        for r in rels:
            _write(r, f"content of {r}".encode())
        return rels, {"x": 1}
    u = CT.Unit("cmp", "gate")
    out = CT.race_drill_single(st, JOB, RUN, u, run_unit, tmp_path / "m", poll_seconds=0.05)
    assert out["ok"] and out["drill_unit"] == "drill-cmp-r1" and out["loser_exit_code"] == 0
    assert sorted(out["states"].values()) == ["won", "yielded"]
    out2 = CT.race_drill_single(st, JOB, RUN, u, run_unit, tmp_path / "m", poll_seconds=0.05)
    assert out2["ok"] and out2["drill_unit"] == "drill-cmp-r2"               # next free drill unit


def test_drill_detects_a_second_writer(tmp_path, monkeypatch):
    """If the claim were not exclusive (both 'win'), the drill must fail."""
    st = _store(tmp_path)
    monkeypatch.setattr(CT, "put_if_absent", lambda store, key, data: True)   # claims no longer exclusive
    rels = ["reports/p4e/compare/metrics.csv"]

    def run_unit(u):
        _write(rels[0], b"c")
        return rels, {}
    out = CT.race_drill_single(st, JOB, RUN, CT.Unit("cmp", "gate"), run_unit, tmp_path / "m", poll_seconds=0.05)
    assert not out["ok"]


def test_drive_counts_yielded(tmp_path):
    st = _store(tmp_path)

    class A:
        max_minutes, wait_minutes = 10.0, 0.0
    rel = _write("reports/p4e/compare/metrics.csv", b"m")
    u = CT.Unit("cmp", "gate")
    from src.models import p4b as PB
    pu = PB.Unit("cmp", "gate", 4, ())
    CT.take_claim(st, JOB, RUN, "cmp", "other", 30)
    threading.Timer(0.3, lambda: CT.write_marker(st, JOB, RUN, u, {}, {}, tmp_path / "h")).start()
    res = CT.drive(A(), time.time(), st, JOB, RUN, [pu], lambda x: ([rel], {}), tmp_path / "m",
                   commit=lambda *a: CT.commit_unit_single(*a, poll_seconds=0.05))
    assert res["yielded"] == ["cmp"] and res["won"] == [] and CT._exit_code(res) == 0
