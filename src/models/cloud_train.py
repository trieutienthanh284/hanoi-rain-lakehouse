"""P4C.2 / P4B — cloud training entry point (GitHub Actions runner or local): PILOT v0 jobs and the v1 job.

  python -m src.models.cloud_train --job p4a6 [--part 1/2] [--store r2|local:<dir>] [--plan]
  python -m src.models.cloud_train --job p4a7 --part 2/3
  python -m src.models.cloud_train --job p4b_smoke              # P4B: first cloud run on real data (D94)
  python -m src.models.cloud_train --job p4b [--part 1/3] [--wait-minutes 90]
  python -m src.models.cloud_train --job p4b --final            # P4B.6, only after the user approved (OOF_DONE.json)
  python -m src.models.cloud_train --job p4b --write-oof-done [--approve-final]   # local agent step, no training
  python -m src.models.cloud_train --job p4b_smoke --race-drill  # re-run the gates of a finished smoke run (race check)
  python -m src.models.cloud_train --job p4b_post                # post-final outputs of run c0df12477947 (no training)
  python -m src.models.cloud_train --job p4d_smoke               # P4D: minutes-long check of every p4d code path
  python -m src.models.cloud_train --job p4d --part 2/5 --wait-minutes 150   # P4D.3-P4D.7 (stops at COMPARE_DONE)

Jobs (definitions exactly as in TASK_P4A_PILOT_MODEL.md, code imported, never copied):
  p4a6  remaining set-B models (all 9 are (re)trained: the 2 local ones serve as a determinism check),
        the A_no2023 0.5 mm/h models (3), then the final unit: compare_A_B (A-2024 vs B-2024, raw NWP
        baseline) + evaluate_lgbm for B and for A_no2023 0.5.
  p4a7  leave-one-season-out for set A: 3 held-out seasons x 3 frames x 3 thresholds = 27 fits, then the final
        unit: per-season out-of-sample metrics and spread (fold definitions: D76).

Remote layout (R2, everything under lake/v1/runs/<job>/; nothing is ever deleted):
  lake/v1/runs/<job>/<run_id>/files/<repo-relative path>     outputs (models, preds, reports, run records)
  lake/v1/runs/<job>/<run_id>/markers/<unit>.done.json       one per finished unit (commit marker)
  lake/v1/runs/<job>/<run_id>/summary.json, done.json        written by the final unit (p4b: only by the part
                                                             that wrote the unit's marker, REPORT_P4B §3.5)
  lake/v1/runs/<job>/latest.json                             pointer to the newest run id (only overwritable key)
Inputs are read from lake/v1/<repo-relative path> (synced by src/lake/r2_sync.py, P4C.1).
run_id = hash of the job, configs/model_pilot.yaml and the training code: re-running the same code resumes the
same run (finished units are skipped); changed code starts a new run. A local mirror of every uploaded file is
kept in cloud_out/<job>/<run_id>/ (GitHub Actions artifact, 7 days). The GitHub step summary is written to
cloud_out/step_summary_<job>.md and appended by the workflow only when the Train step succeeded.
Secrets: only from the environment; all output goes through src.realtime.common's redacting streams.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from src.realtime import common as C  # noqa: E402

LAKE = "lake/v1/"
RUNS = LAKE + "runs/"
JOBS = ("p4a6", "p4a7", "p4b", "p4b_smoke", "p4b_post", "p4d", "p4d_smoke", "p4e_smoke")
P4B_JOBS = ("p4b", "p4b_smoke")
POST_JOBS = ("p4b_post",)        # post-final outputs; reads a fixed p4b run, never trains (src/models/p4b_post.py)
P4D_JOBS = ("p4d", "p4d_smoke")  # v2 candidates cut vs weight (src/models/p4d.py); no --final in this phase
P4E_JOBS = ("p4e_smoke",)       # v2 challenger (src/models/p4e.py); the full job p4e is added after the smoke
SUMMARY_TITLE = {"p4a6": "PILOT v0", "p4a7": "PILOT v0", "p4b": "v1 (P4B)", "p4b_smoke": "v1 smoke (P4B, 2024 only)",
                 "p4b_post": "P4B post-final (no training, no test re-scoring)",
                 "p4d": "v2 candidates cut vs weight (P4D, LOSO OOF only)",
                 "p4d_smoke": "v2 smoke (P4D, short blocks, 2.5 mm/h only)",
                 "p4e_smoke": "v2 challenger smoke (P4E.0: single writer, rounds cap, downsampling, two-stage)"}
GOLD_REL = "data/gold/met_training/pilot_v0/"
CODE_FILES = ["src/models/cloud_train.py", "src/models/train_lgbm.py", "src/models/data.py", "src/models/metrics.py",
              "src/models/compare_A_B.py", "src/models/evaluate_lgbm.py", "src/gold/p4a_common.py",
              "configs/model_pilot.yaml"]
# leave-one-season-out folds (D76): inner validation = last two months of the most recent training season
LOSO = {
    2023: {"tune": [("2024-05-01", "2024-10-31"), ("2025-05-01", "2025-07-31")], "validation": ("2025-08-01", "2025-09-30"),
           "heldout": ("2023-05-01", "2023-10-31")},
    2024: {"tune": [("2023-05-01", "2023-10-31"), ("2025-05-01", "2025-07-31")], "validation": ("2025-08-01", "2025-09-30"),
           "heldout": ("2024-05-01", "2024-10-31")},
    2025: {"tune": [("2023-05-01", "2024-08-31")], "validation": ("2024-09-01", "2024-10-31"),
           "heldout": ("2025-05-01", "2025-09-30")},
}
# inputs per job: repo-relative prefixes downloaded from lake/v1/
INPUTS = {
    "p4a6": [GOLD_REL + "table_B/", GOLD_REL + "table_A/schema.json", GOLD_REL + "table_A/season=2024/",
             GOLD_REL + "table_A/season=2025/", GOLD_REL + "preds/baselines_", GOLD_REL + "preds/lgbm_A_no2023_",
             "models/pilot_v0/A_no2023/"],
    "p4a7": [GOLD_REL + "table_A/", GOLD_REL + "preds/lgbm_A_lead"],
}


# ------------------------------------------------------------------------------------------- units
@dataclass(frozen=True)
class Unit:
    id: str
    kind: str                     # train | loso | final
    params: dict = field(default_factory=dict, hash=False, compare=False)


def kstr(t: float) -> str:
    return f"{t:g}".replace(".", "p")


def job_units(job: str, cfg: dict) -> list[Unit]:
    leads = list(cfg["frames"]["lead_min"])
    thrs = [float(t) for t in cfg["targets"]["thresholds_mm_h"]]
    u: list[Unit] = []
    if job == "p4a6":
        for lead in leads:                                   # ordered by lead: one data load per (variant, lead)
            for t in thrs:
                u.append(Unit(f"train-B-lead{lead}-thr{kstr(t)}", "train", {"variant": "B", "lead": lead, "thr": t}))
            u.append(Unit(f"train-A_no2023-lead{lead}-thr0p5", "train", {"variant": "A_no2023", "lead": lead, "thr": 0.5}))
    elif job == "p4a7":
        for s in sorted(LOSO):
            for lead in leads:
                for t in thrs:
                    u.append(Unit(f"loso-{s}-lead{lead}-thr{kstr(t)}", "loso", {"season": s, "lead": lead, "thr": t}))
    u.append(Unit("final", "final", {}))
    return u


def parse_part(part: str | None) -> tuple[int, int]:
    if not part:
        return 1, 1
    i, n = (int(x) for x in part.split("/"))
    if not (n >= 1 and 1 <= i <= n):
        raise ValueError(f"bad --part {part!r} (expected i/n with 1 <= i <= n)")
    return i, n


def part_units(units: list[Unit], i: int, n: int) -> list[Unit]:
    """Trainable units of part i/n: contiguous blocks (keeps units of one (variant, lead) together, so a part loads
    each data slice once). The final unit is not assigned to a part: any part runs it once every trainable unit
    has its marker."""
    tr = [x for x in units if x.kind != "final"]
    lo, hi = (len(tr) * (i - 1)) // n, (len(tr) * i) // n
    return tr[lo:hi]


# ------------------------------------------------------------------------------------------- keys / store
def run_id_for(job: str) -> str:
    import hashlib
    h = hashlib.sha256(job.encode())
    files = CODE_FILES
    if job in P4B_JOBS:
        from src.models import p4b as PB
        files = PB.CODE_FILES
    elif job in POST_JOBS:
        from src.models import p4b_post as PP
        files = PP.CODE_FILES
    elif job in P4D_JOBS:
        from src.models import p4d as PD
        files = PD.CODE_FILES
    elif job in P4E_JOBS:
        from src.models import p4e as PE
        files = PE.CODE_FILES
    for f in files:
        h.update((ROOT / f).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:12]


def run_prefix(job: str, run_id: str) -> str:
    p = f"{RUNS}{job}/{run_id}/"
    assert p.startswith(RUNS) and ".." not in p
    return p


def file_key(job: str, run_id: str, rel: str) -> str:
    rel = Path(rel).as_posix()
    if rel.startswith("/") or ".." in rel.split("/"):
        raise ValueError(f"bad relative path {rel}")
    return run_prefix(job, run_id) + "files/" + rel


def marker_key(job: str, run_id: str, unit_id: str) -> str:
    return run_prefix(job, run_id) + f"markers/{unit_id}.done.json"


def latest_key(job: str) -> str:
    return f"{RUNS}{job}/latest.json"


def open_run_store(spec: str, job: str):
    """Store restricted to lake/v1/runs/<job>/ (puts only; reads anywhere). Only latest.json is overwritable."""
    second = f"{RUNS}{job}/"
    if job in P4B_JOBS:                           # p4b also writes the v1 Gold tables (never overwritten: gold id)
        from src.models import p4b as PB
        second = PB.GOLD_REMOTE
    elif job in P4D_JOBS:                         # p4d writes Gold table_v2 (lake/v1/gold/met_training/table_v2/)
        from src.models import p4d as PD
        second = PD.GOLD_REMOTE
    elif job in P4E_JOBS:                         # p4e: lake/v1/gold/met_training/table_v2*/ (CLAUDE.md §8, P4E)
        from src.models import p4e as PE
        second = PE.GOLD_REMOTE_PREFIX
    cfg = {"remote": {"realtime_prefix": f"{RUNS}{job}/", "status_prefix": second,
                      "overwritable_keys": [latest_key(job)], "bucket_env": "R2_BUCKET"}}
    return C.open_store(spec, cfg)


def done_units(store, job: str, run_id: str) -> set[str]:
    pre = run_prefix(job, run_id) + "markers/"
    return {k[len(pre):-len(".done.json")] for k in store.list(pre) if k.endswith(".done.json")}


# ------------------------------------------------------------------------------------------- transfer
def download_inputs(store, job: str) -> dict:
    n = b = 0
    for rel in INPUTS[job]:
        for key, size in sorted(store.list(LAKE + rel).items()):
            dst = ROOT / key[len(LAKE):]
            if dst.is_file() and dst.stat().st_size == size:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".part")
            tmp.write_bytes(store.get(key))
            os.replace(tmp, dst)
            n, b = n + 1, b + size
    return {"downloaded_objects": n, "downloaded_bytes": b}


def overlay_run_files(store, job: str, run_id: str) -> int:
    """Bring every output of earlier units of this run (any part) into the workspace."""
    pre = run_prefix(job, run_id) + "files/"
    n = 0
    for key, size in sorted(store.list(pre).items()):
        dst = ROOT / key[len(pre):]
        if dst.is_file() and dst.stat().st_size == size:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(store.get(key))
        n += 1
    return n


def upload_outputs(store, job: str, run_id: str, rels: list[str], mirror: Path) -> dict:
    """Upload unit outputs. A unit without marker may be re-run after an interruption: its own output keys are
    then overwritable (never anything else, never markers)."""
    out = {}
    for rel in rels:
        p = ROOT / rel
        if not p.is_file():
            continue
        key = file_key(job, run_id, rel)
        data = p.read_bytes()
        store.overwritable.add(key)
        store.put(key, data, overwrite=True)
        store.overwritable.discard(key)
        out[key] = C.sha256_bytes(data)
        m = mirror / "files" / rel
        m.parent.mkdir(parents=True, exist_ok=True)
        m.write_bytes(data)
    return out


def write_marker(store, job: str, run_id: str, unit: Unit, outputs: dict, metrics: dict, mirror: Path) -> None:
    obj = {"job": job, "run_id": run_id, "unit": unit.id, "finished_utc": C.iso(C.utcnow()), "outputs": outputs,
           "metrics": metrics, "context": C.run_context()}
    store.put_json(marker_key(job, run_id, unit.id), obj)
    m = mirror / "markers" / f"{unit.id}.done.json"
    m.parent.mkdir(parents=True, exist_ok=True)
    m.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")


def marker_exists(store, job: str, run_id: str, unit_id: str) -> bool:
    k = marker_key(job, run_id, unit_id)
    return k in store.list(k)


def commit_unit(store, job: str, run_id: str, unit: Unit, rels: list[str], metrics: dict, mirror: Path) -> dict:
    """Upload a finished unit and write its marker, safely when parallel parts run the same gate (REPORT_P4B §3.5).

    won      this part wrote the marker (only then may it write summary.json / done.json);
    skipped  the marker already existed: nothing is uploaded (the other part's outputs stay untouched);
    lost     the marker appeared between the check and the write: outputs were uploaded, the marker is the other
             part's; `mismatch` lists output keys whose sha256 differs from that marker (R2 then disagrees with it).
    """
    if marker_exists(store, job, run_id, unit.id):
        C.log("marker_exists_skip_upload", unit=unit.id)
        return {"state": "skipped", "mismatch": []}
    outs = upload_outputs(store, job, run_id, rels, mirror)
    try:
        write_marker(store, job, run_id, unit, outs, metrics, mirror)
        return {"state": "won", "mismatch": []}
    except C.ObjectConflict:
        other = json.loads(store.get(marker_key(job, run_id, unit.id))).get("outputs", {})
        mismatch = sorted(k for k, v in outs.items() if other.get(k) != v)
        C.log("marker_lost", unit=unit.id, outputs=len(outs), mismatch=len(mismatch))
        return {"state": "lost", "mismatch": mismatch}


def _mirror(mirror: Path, rel: str, data: bytes) -> None:
    m = mirror / "files" / rel
    m.parent.mkdir(parents=True, exist_ok=True)
    m.write_bytes(data)


def commit_unit_safe(store, job: str, run_id: str, unit: Unit, rels: list[str], metrics: dict, mirror: Path) -> dict:
    """commit_unit for jobs whose output files are deterministic (p4d: no timings inside outputs). Outputs are put
    without overwrite first (identical bytes -> 'exists_identical'); a key that exists with other bytes is replaced
    only while the unit has no marker (leftover of an interrupted attempt). If the marker appeared meanwhile, nothing
    is overwritten: state 'lost' with the keys whose sha256 differs from the winning marker (REPORT_P4B §8.1)."""
    if marker_exists(store, job, run_id, unit.id):
        C.log("marker_exists_skip_upload", unit=unit.id)
        return {"state": "skipped", "mismatch": []}
    outs, conflicts = {}, []
    for rel in rels:
        p = ROOT / rel
        if not p.is_file():
            continue
        key, data = file_key(job, run_id, rel), p.read_bytes()
        outs[key] = C.sha256_bytes(data)
        try:
            store.put(key, data)
            _mirror(mirror, rel, data)
        except C.ObjectConflict:
            conflicts.append((key, rel, data))

    def _lost() -> dict:
        other = json.loads(store.get(marker_key(job, run_id, unit.id))).get("outputs", {})
        mismatch = sorted(k for k, v in outs.items() if other.get(k) != v)
        C.log("marker_lost", unit=unit.id, outputs=len(outs), mismatch=len(mismatch))
        return {"state": "lost", "mismatch": mismatch}
    if conflicts:
        if marker_exists(store, job, run_id, unit.id):
            return _lost()
        for key, rel, data in conflicts:                 # own leftovers of an attempt without marker
            store.overwritable.add(key)
            try:
                store.put(key, data, overwrite=True)
            finally:
                store.overwritable.discard(key)
            _mirror(mirror, rel, data)
        C.log("replaced_unmarked_outputs", unit=unit.id, n=len(conflicts))
    try:
        write_marker(store, job, run_id, unit, outs, metrics, mirror)
        return {"state": "won", "mismatch": []}
    except C.ObjectConflict:
        return _lost()


# ------------------------------------------------------------------------------------------- single writer (D188)
def attempt_id(tag: str = "") -> str:
    """Identity of one contender (runner attempt, part, process): written into claims, used for scratch keys."""
    import uuid
    gh = os.environ.get("GITHUB_RUN_ID", "local")
    at = os.environ.get("GITHUB_RUN_ATTEMPT", "0")
    return f"{gh}-{at}-{os.getpid()}-{uuid.uuid4().hex[:8]}{('-' + tag) if tag else ''}"


def claim_key(job: str, run_id: str, unit_id: str, gen: int) -> str:
    return run_prefix(job, run_id) + f"claims/{unit_id}/g{gen:03d}.json"


def put_if_absent(store, key: str, data: bytes) -> bool:
    """Atomic create: True if this call created the object, False if it already existed. R2: PutObject with
    If-None-Match (server side); LocalStore: O_CREAT | O_EXCL (the store's own put is check-then-write)."""
    store.check_key(key)
    if isinstance(store, C.LocalStore):                # tests: content complete before the name appears
        import uuid
        p = store._p(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f"{p.name}.{uuid.uuid4().hex}.part")
        tmp.write_bytes(data)
        try:
            os.link(tmp, p)                            # atomic, fails if the name exists
            ok = True
        except FileExistsError:
            ok = False
        finally:
            tmp.unlink()                               # own temporary file of this call only
        store.ops["class_a"] += 1
        return ok
    try:
        store._put(key, data, C.sha256_bytes(data), False)      # IfNoneMatch="*": never replaces an object
    except C.ObjectConflict:
        return False
    return True


def latest_claim(store, job: str, run_id: str, unit_id: str) -> tuple[int, dict | None]:
    pre = run_prefix(job, run_id) + f"claims/{unit_id}/"
    keys = sorted(k for k in store.list(pre) if k.endswith(".json"))
    if not keys:
        return 0, None
    k = keys[-1]
    try:
        obj = json.loads(store.get(k))
    except ValueError:                                  # defensive: unreadable claim counts as fresh (never stale)
        obj = {"written_utc": C.iso(C.utcnow())}
    return int(k.rsplit("/g", 1)[1][:-5]), obj


def take_claim(store, job: str, run_id: str, unit_id: str, me: str, stale_minutes: float) -> int | None:
    """Generation number of the claim this contender now holds, or None (another contender holds a live claim).
    Generation 1 is the first claim; a claim older than stale_minutes without a marker may be superseded by the next
    generation (its holder died between claim and marker). Each generation is created at most once (atomic)."""
    g, obj = latest_claim(store, job, run_id, unit_id)
    if obj is not None:
        age = (C.utcnow() - C.parse_iso(obj["written_utc"])).total_seconds() / 60
        if age <= stale_minutes or marker_exists(store, job, run_id, unit_id):
            return None
    nxt = g + 1
    body = json.dumps({"unit": unit_id, "gen": nxt, "attempt": me, "written_utc": C.iso(C.utcnow()),
                       "context": C.run_context()}, indent=1).encode()
    return nxt if put_if_absent(store, claim_key(job, run_id, unit_id, nxt), body) else None


def adopt_outputs(store, job: str, run_id: str, unit_id: str) -> int:
    """Replace this part's local copies of a unit's outputs with the bytes of the winning marker, so that later
    units of this part read exactly what R2 holds (P4D: a part's own last-digit variants were read locally)."""
    mk = json.loads(store.get(marker_key(job, run_id, unit_id)))
    pre = run_prefix(job, run_id) + "files/"
    n = 0
    for key, sha in mk.get("outputs", {}).items():
        dst = ROOT / key[len(pre):]
        if dst.is_file() and C.sha256_bytes(dst.read_bytes()) == sha:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".adopt")
        tmp.write_bytes(store.get(key))
        os.replace(tmp, dst)
        n += 1
    return n


def commit_unit_single(store, job: str, run_id: str, unit: Unit, rels: list[str], metrics: dict, mirror: Path,
                       payload: dict | None = None, me: str | None = None, stale_minutes: float = 30.0,
                       poll_seconds: float = 15.0, scratch_kinds=("gate",)) -> dict:
    """Exactly one writer per unit (D188 fix, P4E.0-1): a contender first wins an atomic claim (If-None-Match); only
    the claim holder writes the final output keys and the marker. Every other contender writes nothing under the
    final keys: it waits for the marker, adopts the winner's bytes locally and returns 'yielded' (exit code 0); for
    gates it keeps its own bytes under runs/<job>/<run_id>/scratch/<attempt>/ for the record.
    payload: {rel: bytes} instead of reading ROOT/rel (race drill)."""
    if marker_exists(store, job, run_id, unit.id):
        adopt_outputs(store, job, run_id, unit.id)
        C.log("marker_exists_skip_upload", unit=unit.id)
        return {"state": "skipped", "mismatch": []}
    me = me or attempt_id()
    data = payload if payload is not None else {r: (ROOT / r).read_bytes() for r in rels if (ROOT / r).is_file()}
    waited = 0.0
    while True:
        g = take_claim(store, job, run_id, unit.id, me, stale_minutes)
        if g is not None:
            break
        if marker_exists(store, job, run_id, unit.id):
            if unit.kind in scratch_kinds:
                for r, b in data.items():
                    store.put(run_prefix(job, run_id) + f"scratch/{me}/files/{Path(r).as_posix()}", b)
            if payload is None:
                adopt_outputs(store, job, run_id, unit.id)
            C.log("claim_yielded", unit=unit.id, attempt=me, waited_s=round(waited, 1))
            return {"state": "yielded", "mismatch": []}
        time.sleep(poll_seconds)
        waited += poll_seconds
    outs = {}
    for r, b in data.items():
        key = file_key(job, run_id, r)
        outs[key] = C.sha256_bytes(b)
        try:
            store.put(key, b)                                   # if-absent; identical bytes -> exists_identical
        except C.ObjectConflict:
            if g == 1:                                          # nobody may write a final key without a claim
                raise RuntimeError(f"{key} exists although {unit.id} had no earlier claim (single-writer breach)")
            store.overwritable.add(key)                         # leftover of a dead holder of an older generation
            try:
                store.put(key, b, overwrite=True)
            finally:
                store.overwritable.discard(key)
        _mirror(mirror, r, b)
    if latest_claim(store, job, run_id, unit.id)[0] != g:     # superseded while uploading (holder seen as stale)
        C.log("claim_superseded", unit=unit.id, gen=g)
        return {"state": "yielded", "mismatch": []}
    try:
        write_marker(store, job, run_id, unit, outs, metrics, mirror)
    except C.ObjectConflict:
        other = json.loads(store.get(marker_key(job, run_id, unit.id))).get("outputs", {})
        mismatch = sorted(k for k, v in outs.items() if other.get(k) != v)
        C.log("marker_lost", unit=unit.id, outputs=len(outs), mismatch=len(mismatch))
        return {"state": "lost", "mismatch": mismatch}
    C.log("claim_won", unit=unit.id, gen=g, attempt=me, outputs=len(outs))
    return {"state": "won", "mismatch": []}


def race_drill_single(store, job: str, run_id: str, unit: Unit, run_unit, mirror: Path, poll_seconds: float = 2.0,
                      stale_minutes: float = 2.0) -> dict:
    """Extended race drill (P4E.0-1): the outputs of gate `unit` are committed under a fresh drill unit id by two
    contenders started at the same instant (threads, barrier) with DIFFERENT bytes. Pass = exactly one writer (one
    'won', one 'yielded'), every final key byte-identical to the winner's payload and to its marker sha, the loser's
    bytes only under its scratch prefix, and the loser's exit code 0. Two parts running this drill at the same time
    race on the same drill unit id as well (the first free drill-<gate>-r<k>)."""
    import threading
    rels, metrics = run_unit(unit)
    k = 1
    while marker_exists(store, job, run_id, f"drill-{unit.id}-r{k}"):
        k += 1
    du = Unit(f"drill-{unit.id}-r{k}", "gate")
    base = {f"reports/{job}/drill/{du.id}/{Path(r).name}": (ROOT / r).read_bytes() for r in rels
            if (ROOT / r).is_file()}
    pay = {"A": base, "B": {r: b + b"\n" for r, b in base.items()}}       # B differs in every file
    ids = {n: attempt_id(f"drill{n}") for n in pay}
    res, bar = {}, threading.Barrier(2)

    def go(n):
        bar.wait()
        try:
            res[n] = commit_unit_single(store, job, run_id, du, list(base), metrics, mirror, payload=pay[n],
                                        me=ids[n], poll_seconds=poll_seconds, stale_minutes=stale_minutes)
        except Exception as e:                          # noqa: BLE001 — a crashing contender fails the drill
            res[n] = {"state": f"error:{type(e).__name__}", "mismatch": []}
    th = [threading.Thread(target=go, args=(n,)) for n in pay]
    for t in th:
        t.start()
    for t in th:
        t.join()
    states = {n: r["state"] for n, r in res.items()}
    winners = [n for n, s in states.items() if s == "won"]
    mk = json.loads(store.get(marker_key(job, run_id, du.id))) if marker_exists(store, job, run_id, du.id) else {}
    have = store.list(file_key(job, run_id, f"reports/{job}/drill/{du.id}/"))
    final = {r: store.get(file_key(job, run_id, r)) if file_key(job, run_id, r) in have else b"" for r in base}
    w = winners[0] if len(winners) == 1 else None
    same_as_winner = w is not None and all(final[r] == pay[w][r] for r in base)
    same_as_marker = bool(mk) and all(mk["outputs"].get(file_key(job, run_id, r)) == C.sha256_bytes(final[r])
                                      for r in base)
    loser = [n for n in pay if n != w]
    scratch = run_prefix(job, run_id) + "scratch/"
    loser_scratch = bool(loser) and all(
        f"{scratch}{ids[loser[0]]}/files/{r}" in store.list(f"{scratch}{ids[loser[0]]}/") for r in base)
    loser_exit = _exit_code({"mismatch": res[loser[0]]["mismatch"]}) if loser else None
    ok = (len(winners) == 1 and sorted(states.values()) == ["won", "yielded"] and same_as_winner and same_as_marker
          and loser_scratch and loser_exit == 0)
    out = {"ok": ok, "drill_unit": du.id, "states": states, "winner": w, "final_equals_winner": same_as_winner,
           "final_equals_marker": same_as_marker, "loser_bytes_in_scratch_only": loser_scratch,
           "loser_exit_code": loser_exit, "files": len(base)}
    C.log("race_drill_single", **out)
    return out


def parts_lock(store, job: str, run_id: str, n: int) -> tuple[bool, int]:
    """A run is split into one fixed number of parts: the first invocation records n; any later invocation with
    another n is refused (prevents 1/1 and 1/2 of the same run from running on top of each other, REPORT_P4B §8.1)."""
    key = run_prefix(job, run_id) + "parts.json"
    if key not in store.list(key):
        try:
            store.put_json(key, {"job": job, "run_id": run_id, "parts": n, "written_utc": C.iso(C.utcnow())})
        except C.ObjectConflict:
            pass
    n0 = int(json.loads(store.get(key))["parts"])
    return n0 == n, n0


def write_step_summary(job: str, md: str) -> Path:
    """The step summary goes to a file named after the job; the workflow appends it to $GITHUB_STEP_SUMMARY only
    when the Train step succeeded (never a file of another job or of an earlier run)."""
    p = ROOT / "cloud_out" / f"step_summary_{job}.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(C.redact(md), encoding="utf-8")
    return p


# ------------------------------------------------------------------------------------------- training units
class DataCache:
    def __init__(self):
        self.key, self.val = None, None

    def get(self, key, loader):
        if key != self.key:
            self.key, self.val = None, None
            import gc; gc.collect()
            self.val, self.key = loader(), key
        return self.val


def _rels_for(variant: str, lead: int, thr: float) -> list[str]:
    k = kstr(thr)
    d = f"models/pilot_v0/{variant}/lead{lead}_thr{k}/"
    return [d + f for f in ("model.txt", "model_prerefit.txt", "calibrator.pkl", "meta.json", "importance.csv")] + \
        [GOLD_REL + f"preds/lgbm_{variant}_lead{lead}_thr{k}_{s}.parquet" for s in ("validation", "test")] + \
        [f"reports/p4a/runs/{variant}_lead{lead}_thr{k}.json"]


def run_train(unit: Unit, cfg: dict, cache: DataCache, lg) -> tuple[list[str], dict]:
    from src.gold import p4a_common as P
    from src.models import data as D
    from src.models import train_lgbm as TL
    v, lead, thr = unit.params["variant"], unit.params["lead"], unit.params["thr"]
    feats = D.feature_list(TL.TABLES[v])
    # same config hash as src/models/train_lgbm.py main()
    ch = P.cfg_hash({"lightgbm": cfg["lightgbm"], "splits": cfg["splits"], "evaluation": cfg["evaluation"],
                     "seed": cfg["random_seed"], "variant": v, "features": feats, "code": P.code_hash(*TL.CODE)})
    tb = TL.TABLES[v]
    tune, val, test = cache.get((v, lead), lambda: (D.load_split(cfg, TL.VARIANTS[v], feats, table=tb, lead=lead),
                                                     D.load_split(cfg, "validation", feats, table=tb, lead=lead),
                                                     D.load_split(cfg, "test", feats, table=tb, lead=lead)))
    m = TL.train_one(cfg, v, lead, thr, tune, val, test, feats, ch, lg)
    return _rels_for(v, lead, thr), {"best_iteration": m["best_iteration"], "val_logloss": m["val_logloss_best"],
                                     "cut": m["cut"]["cut"], "val_csi_at_cut": m["cut"]["csi"], "seconds": m["seconds"]}


def _load_ranges(cfg, ranges, feats, lead):
    import pandas as pd
    from src.models import data as D
    return pd.concat([D.load_split(cfg, r, feats, table="table_A", lead=lead) for r in ranges], ignore_index=True)


def run_loso(unit: Unit, cfg: dict, cache: DataCache, lg) -> tuple[list[str], dict]:
    from src.gold import p4a_common as P
    from src.models import data as D
    from src.models import train_lgbm as TL
    s, lead, thr = unit.params["season"], unit.params["lead"], unit.params["thr"]
    f = LOSO[s]
    v = f"LOSO_{s}"
    feats = D.feature_list("table_A")
    # register the fold as a variant of the unchanged training code (in memory only)
    TL.VARIANTS[v], TL.TABLES[v] = f"loso_{s}_tune", "table_A"
    cfg = json.loads(json.dumps(cfg))
    cfg["splits"][f"loso_{s}_tune"] = [f["tune"][0][0], f["tune"][-1][1]]
    ch = P.cfg_hash({"lightgbm": cfg["lightgbm"], "evaluation": cfg["evaluation"], "seed": cfg["random_seed"],
                     "fold": {k: f[k] for k in ("tune", "validation", "heldout")}, "features": feats,
                     "code": P.code_hash(*TL.CODE)})
    tune, val, test = cache.get((v, lead), lambda: (_load_ranges(cfg, f["tune"], feats, lead),
                                                     D.load_split(cfg, f["validation"], feats, table="table_A", lead=lead),
                                                     D.load_split(cfg, f["heldout"], feats, table="table_A", lead=lead)))
    m = TL.train_one(cfg, v, lead, thr, tune, val, test, feats, ch, lg)
    # correct the training-period block of meta.json (train_one writes the P4A split names)
    mp = TL.MODELS / v / f"lead{lead}_thr{kstr(thr)}" / "meta.json"
    meta = json.loads(mp.read_text(encoding="utf-8"))
    meta["training_period"] = {"tune": f["tune"], "validation": list(f["validation"]),
                               "refit": f["tune"] + [list(f["validation"])], "heldout_season": list(f["heldout"])}
    meta["loso"] = {"heldout_season": s, "note": "leave-one-season-out fold (D76); not a forward-in-time test"}
    P.write_json(mp, meta)
    rels = _rels_for(v, lead, thr)
    return rels, {"best_iteration": m["best_iteration"], "val_logloss": m["val_logloss_best"], "cut": m["cut"]["cut"],
                  "seconds": m["seconds"]}


# ------------------------------------------------------------------------------------------- final units
def final_p4a6(cfg: dict) -> tuple[list[str], dict]:
    import pandas as pd
    from src.models import compare_A_B, evaluate_lgbm
    compare_A_B.main()
    evaluate_lgbm.main(["--variant", "B"])
    evaluate_lgbm.main(["--variant", "A_no2023", "--thresholds", "0.5"])
    rels = []
    for d in ("reports/p4a/compare_A_B", "reports/p4a/lgbm_B", "reports/p4a/lgbm_A_no2023"):
        rels += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / d).glob("*")) if p.is_file()]
    rels += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "reports/p4a").glob("importance_*.png"))]
    t = pd.read_csv(ROOT / "reports/p4a/compare_A_B/test_diffs.csv")
    t = t[t["comparison"] == "B_2024 - A_2024"]
    table = [{"threshold": r.threshold, "lead_min": int(r.lead_min), "csi_diff_B_minus_A": round(r.csi_diff, 4),
              "csi_ci90": [round(r.csi_lo, 4), round(r.csi_hi, 4)], "brier_diff": round(r.brier_diff, 5),
              "brier_ci90": [round(r.brier_lo, 5), round(r.brier_hi, 5)]} for r in t.itertuples()]
    v = pd.read_csv(ROOT / "reports/p4a/lgbm_B/verdict.csv")
    return rels, {"B_minus_A_2024_test": table,
                  "verdict_B": [{"threshold": r.threshold, "lead_min": int(r.lead_min), "verdict": r.verdict}
                                for r in v.itertuples()]}


def loso_metrics(cfg: dict) -> "pd.DataFrame":
    import numpy as np
    import pandas as pd
    from src.models import baselines as BL
    from src.models import data as D
    from src.models import metrics as M
    from src.gold import p4a_common as P
    rows = []
    gk = [f"gc_k{k}" for k in range(7)]
    for s, f in sorted(LOSO.items()):
        train_lab = pd.concat([D.load_split(cfg, r, [], table="table_A") for r in f["tune"] + [f["validation"]]],
                              ignore_index=True)
        held = D.load_split(cfg, f["heldout"], gk, table="table_A")
        for lead in cfg["frames"]["lead_min"]:
            h = held[held["lead_min"] == lead].reset_index(drop=True)
            tr = train_lab[train_lab["lead_min"] == lead]
            pers_x = D.asof_latest(h) if lead in (0, 30) else D.max_avail_0_60(h)
            for thr in cfg["targets"]["thresholds_mm_h"]:
                k = kstr(thr); yc = D.thr_col(thr)
                pr = pd.read_parquet(P.GOLD / "preds" / f"lgbm_LOSO_{s}_lead{lead}_thr{k}_test.parquet")
                j = pr.merge(h[["pixel_id", "t0", "frame_start"]].assign(pers=pers_x), on=["pixel_id", "t0"],
                             how="inner", validate="one_to_one")
                y = j[yc].to_numpy().astype(bool)
                pc = BL.clim_lookup(BL.clim_table(tr, yc), j)
                pc = np.where(np.isnan(pc), tr[yc].mean(), pc)
                yp = np.nan_to_num(j["pers"].to_numpy("float64"), nan=-1.0) >= thr
                sm = M.summary(y, j["p"].to_numpy("float64"), j["yhat"].to_numpy().astype(bool), pc)
                rows.append({"heldout_season": s, "lead_min": lead, "threshold": float(thr), "n": len(j),
                             "base_rate": sm["base_rate"], "csi": sm["csi"], "pod": sm["pod"], "far": sm["far"],
                             "bias": sm["bias"], "csi_pers": float(M.scores(**M.contingency(y, yp))["csi"]),
                             "brier_cal": sm["brier"], "bss_cal": sm["bss_clim"], "auc_pr_cal": sm["auc_pr"],
                             "brier_raw": M.brier(y, j["p_raw"].to_numpy("float64")),
                             "bss_raw": M.bss(y, j["p_raw"].to_numpy("float64"), pc),
                             "auc_pr_raw": M.auc_pr(y, j["p_raw"].to_numpy("float64"))})
    return pd.DataFrame(rows)


def final_p4a7(cfg: dict) -> tuple[list[str], dict]:
    import numpy as np
    import pandas as pd
    from src.gold import p4a_common as P
    out = ROOT / "reports" / "p4a" / "loso"
    out.mkdir(parents=True, exist_ok=True)
    t = loso_metrics(cfg)
    t.to_csv(out / "loso_metrics.csv", index=False)
    spread = t.groupby(["threshold", "lead_min"]).agg(
        csi_min=("csi", "min"), csi_max=("csi", "max"), csi_mean=("csi", "mean"),
        bss_raw_min=("bss_raw", "min"), bss_raw_max=("bss_raw", "max"),
        csi_minus_pers_min=("csi", lambda s: float((s - t.loc[s.index, "csi_pers"]).min())),
        csi_minus_pers_max=("csi", lambda s: float((s - t.loc[s.index, "csi_pers"]).max()))).reset_index()
    spread.to_csv(out / "loso_spread.csv", index=False)
    # determinism / code-path check: fold 2025 has exactly the P4A set-A data, splits and parameters
    rep = {}
    for lead in cfg["frames"]["lead_min"]:
        for thr in cfg["targets"]["thresholds_mm_h"]:
            k = kstr(thr)
            a = P.GOLD / "preds" / f"lgbm_A_lead{lead}_thr{k}_test.parquet"
            b = P.GOLD / "preds" / f"lgbm_LOSO_2025_lead{lead}_thr{k}_test.parquet"
            if a.is_file() and b.is_file():
                x = pd.read_parquet(a, columns=["pixel_id", "t0", "p_raw"]).merge(
                    pd.read_parquet(b, columns=["pixel_id", "t0", "p_raw"]), on=["pixel_id", "t0"], suffixes=("_A", "_L"))
                rep[f"lead{lead}_thr{k}"] = float(np.abs(x["p_raw_A"] - x["p_raw_L"]).max())
    P.write_json(out / "repro_fold2025_vs_A.json", {"max_abs_diff_p_raw": rep,
                                                   "note": "fold 2025 = P4A set A (same data, splits, parameters)"})
    rels = [p.relative_to(ROOT).as_posix() for p in sorted(out.glob("*")) if p.is_file()]
    x = t[t["threshold"] == 2.5][["heldout_season", "lead_min", "csi", "csi_pers", "bss_raw", "bss_cal"]].round(3)
    return rels, {"csi_2p5": x.to_dict("records"), "repro_fold2025_max_abs_diff": max(rep.values()) if rep else None}


FINALS = {"p4a6": final_p4a6, "p4a7": final_p4a7}


# ------------------------------------------------------------------------------------------- driver
def step_summary(job: str, run_id: str, done_now: list[str], todo_left: int, final_metrics: dict | None,
                 extra: list[str] | None = None) -> str:
    lines = [f"### cloud_train `{job}` run `{run_id}` ({SUMMARY_TITLE.get(job, job)})", "",
             f"- units finished in this invocation: {len(done_now)}; units still missing: {todo_left}"]
    lines += [f"- {x}" for x in (extra or [])]
    if done_now:
        lines.append(f"- {', '.join(done_now[:30])}{' …' if len(done_now) > 30 else ''}")
    if final_metrics:
        lines += ["", "| key | value |", "|---|---|"]
        for k, v in final_metrics.items():
            if isinstance(v, list):
                for r in v[:30]:
                    lines.append(f"| {k} | {json.dumps(r, default=str)} |")
            else:
                lines.append(f"| {k} | {v} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, choices=JOBS)
    ap.add_argument("--part", default="1/1")
    ap.add_argument("--final", action="store_true", help="p4b only: run the P4B.6 test evaluation")
    ap.add_argument("--store", default="r2", help="r2 (default) or local:<dir>")
    ap.add_argument("--run-id", default=None, help="override the code-derived run id")
    ap.add_argument("--max-minutes", type=float, default=320.0, help="stop starting new units after this time")
    ap.add_argument("--plan", action="store_true", help="print the units of this part and their state; train nothing")
    ap.add_argument("--wait-minutes", type=float, default=0.0,
                    help="p4b: wait up to this long for other parts to finish a stage (polls R2 markers every 60 s)")
    ap.add_argument("--write-oof-done", action="store_true",
                    help="p4b, local agent step: write lake/v1/reports/p4b/OOF_DONE.json for this run (no training)")
    ap.add_argument("--approve-final", action="store_true",
                    help="with --write-oof-done: set final_approved = true (only after the user said so)")
    ap.add_argument("--race-drill", action="store_true",
                    help="p4b_smoke: after the normal loop, run every finished gate and the final unit again; each "
                         "must be skipped (no upload, summary.json / done.json unchanged). p4e_smoke: two "
                         "contenders commit the same gate at the same instant (single-writer drill)")
    a = ap.parse_args(argv)
    t_start = time.time()
    if a.race_drill and a.job not in ("p4b_smoke", "p4e_smoke"):
        C.log("refused", reason="--race-drill applies to jobs p4b_smoke and p4e_smoke only")
        return 2
    if a.job in P4B_JOBS:
        return main_p4b(a, t_start)
    if a.job in POST_JOBS:
        return main_post(a, t_start)
    if a.job in P4D_JOBS:
        return main_p4d(a, t_start)
    if a.job in P4E_JOBS:
        return main_p4e(a, t_start)
    if a.final:
        C.log("final_refused", job=a.job, reason="--final applies to job p4b only")
        return 2
    from src.gold import p4a_common as P
    P.require_guard()
    cfg = P.load_cfg()
    i, n = parse_part(a.part)
    units = job_units(a.job, cfg)
    mine = part_units(units, i, n)
    run_id = a.run_id or run_id_for(a.job)
    store = open_run_store(a.store, a.job)
    done = done_units(store, a.job, run_id)
    todo = [u for u in mine if u.id not in done]
    C.log("plan", job=a.job, run_id=run_id, part=f"{i}/{n}", units_in_part=len(mine), done=len(mine) - len(todo),
          todo=[u.id for u in todo][:40], final_done="final" in done)
    if a.plan:
        return 0
    mirror = ROOT / "cloud_out" / a.job / run_id
    store.put_json(latest_key(a.job), {"job": a.job, "run_id": run_id, "updated_utc": C.iso(C.utcnow()),
                                       "context": C.run_context()}, overwrite=True)
    if todo or "final" not in done:
        C.log("inputs", **download_inputs(store, a.job))
    lg = P.get_logger(f"cloud_{a.job}")
    cache = DataCache()
    done_now = []
    for u in todo:
        if (time.time() - t_start) / 60 > a.max_minutes:
            C.log("time_budget_reached", remaining=len(todo) - len(done_now), hint="run the job again: it resumes")
            break
        t1 = time.time()
        rels, metrics = (run_train if u.kind == "train" else run_loso)(u, cfg, cache, lg)
        outs = upload_outputs(store, a.job, run_id, rels, mirror)
        write_marker(store, a.job, run_id, u, outs, metrics, mirror)
        done_now.append(u.id)
        C.log("unit_done", unit=u.id, seconds=round(time.time() - t1, 1), rss_gb=round(P.rss_gb(), 2), **{
            k: v for k, v in metrics.items() if k in ("best_iteration", "val_logloss", "cut")})
    done = done_units(store, a.job, run_id)
    missing = [u.id for u in units if u.kind != "final" and u.id not in done]
    final_metrics = None
    if not missing and "final" not in done and (time.time() - t_start) / 60 <= a.max_minutes:
        C.log("final_start", job=a.job, overlay_files=overlay_run_files(store, a.job, run_id))
        rels, final_metrics = FINALS[a.job](cfg)
        outs = upload_outputs(store, a.job, run_id, rels, mirror)
        summary = {"job": a.job, "run_id": run_id, "status": "PILOT v0 — labels IMERG Final (satellite estimate)",
                   "units": len(units), "metrics": final_metrics, "finished_utc": C.iso(C.utcnow()),
                   "context": C.run_context()}
        store.put_json(run_prefix(a.job, run_id) + "summary.json", summary)
        write_marker(store, a.job, run_id, Unit("final", "final"), outs, final_metrics, mirror)
        store.put_json(run_prefix(a.job, run_id) + "done.json", {"job": a.job, "run_id": run_id,
                                                                   "finished_utc": summary["finished_utc"]})
        (mirror / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    elif "final" in done:
        C.log("final_already_done", job=a.job, run_id=run_id)
    write_step_summary(a.job, step_summary(a.job, run_id, done_now, len(missing), final_metrics))
    C.log("end", job=a.job, run_id=run_id, finished_now=len(done_now), missing_trainable=len(missing),
          final=("done" if final_metrics or "final" in done else "waiting"), store=store.stats(),
          minutes=round((time.time() - t_start) / 60, 1))
    return 0


# ------------------------------------------------------------------------------------------- P4B driver
def oof_done_store(spec: str):
    """Store that may write exactly lake/v1/reports/p4b/OOF_DONE.json (local agent step)."""
    from src.models import p4b as PB
    pre = PB.OOF_DONE_KEY.rsplit("/", 1)[0] + "/"
    return C.open_store(spec, {"remote": {"realtime_prefix": pre, "status_prefix": pre,
                                          "overwritable_keys": [PB.OOF_DONE_KEY], "bucket_env": "R2_BUCKET"}})


def write_oof_done(spec: str, run_id: str, approve: bool, run_store) -> dict:
    from src.models import p4b as PB
    if "oof-report" not in done_units(run_store, "p4b", run_id):
        raise SystemExit(f"oof-report of run {run_id} is not done; OOF_DONE.json not written")
    obj = {"job": "p4b", "run_id": run_id, "oof_report_marker": marker_key("p4b", run_id, "oof-report"),
           "written_utc": C.iso(C.utcnow()), "final_approved": bool(approve),
           "note": "final_approved is set to true only after the user read the OOF report and said to run the "
                   "final test (TASK_P4B_MODEL_V1.md, stop point after P4B.5)"}
    oof_done_store(spec).put_json(PB.OOF_DONE_KEY, obj, overwrite=True)
    return obj


def main_p4b(a, t_start: float) -> int:
    from src.models import p4b as PB
    smoke = a.job == "p4b_smoke"
    cfg = PB.load_cfg(smoke=smoke)
    i, n = parse_part(a.part)
    units = PB.units(cfg)
    mine = PB.part_units(units, i, n)
    gates = [u for u in units if u.kind == "gate"]
    final_u = [u for u in units if u.kind == "final"]
    run_id = a.run_id or run_id_for(a.job)
    store = open_run_store(a.store, a.job)
    if a.write_oof_done:
        if a.job != "p4b":
            C.log("refused", reason="--write-oof-done applies to job p4b only")
            return 2
        C.log("oof_done", **write_oof_done(a.store, run_id, a.approve_final, store))
        return 0
    if a.final and not smoke:
        ok, why = PB.check_final_guard(store, run_id)
        if not ok:
            C.log("final_refused", job=a.job, run_id=run_id, reason=why)
            return 2
    elif a.final and smoke:
        C.log("final_ignored", reason="p4b_smoke runs its final unit on 2024 pseudo-test days automatically")
    # final unit: p4b only with an approved --final; the smoke job always (2024 pseudo-test days, never 2025)
    allowed = mine + gates + (final_u if (smoke or a.final) else [])
    done = done_units(store, a.job, run_id)
    C.log("plan", job=a.job, run_id=run_id, part=f"{i}/{n}", units_total=len(units), units_in_part=len(mine),
          done=len(done), todo=[u.id for u in allowed if u.id not in done][:40], final=a.final, smoke=smoke,
          gold_id=PB.gold_id(cfg))
    if a.plan:
        return 0
    mirror = ROOT / "cloud_out" / a.job / run_id
    store.put_json(latest_key(a.job), {"job": a.job, "run_id": run_id, "updated_utc": C.iso(C.utcnow()),
                                       "context": C.run_context()}, overwrite=True)
    ctx = PB.Ctx(cfg, store, a.job, run_id, mirror, C.log)
    res = drive(a, t_start, store, a.job, run_id, allowed, lambda u: PB.run_unit(u, ctx), mirror)
    won = res["won"]
    if a.race_drill:                                  # smoke only (checked in main)
        res["drill"] = race_drill(store, a.job, run_id, gates + final_u, lambda u: PB.run_unit(u, ctx), mirror)
    done = done_units(store, a.job, run_id)
    missing = [u.id for u in units if u.kind != "final" and u.id not in done]
    if "oof-report" in won:
        store.put_json(run_prefix(a.job, run_id) + "summary.json",
                       {"job": a.job, "run_id": run_id, "stage": "OOF complete (stop point after P4B.5)",
                        "metrics": res["metrics"].get("oof-report"), "finished_utc": C.iso(C.utcnow()),
                        "context": C.run_context()})
    if "final-test" in won:
        store.put_json(run_prefix(a.job, run_id) + "done.json", {"job": a.job, "run_id": run_id,
                                                                   "finished_utc": C.iso(C.utcnow())})
    extra = [f"part {i}/{n}; final = {a.final}; race drill = {a.race_drill}", *_race_lines(res)]
    write_step_summary(a.job, step_summary(a.job, run_id, won, len(missing), res["metrics"] or None, extra))
    C.log("end", job=a.job, run_id=run_id, part=f"{i}/{n}", finished_now=len(won), missing_oof_units=len(missing),
          oof_complete="oof-report" in done, final=("done" if "final-test" in done else "not run"),
          skipped=len(res["skipped"]), lost=len(res["lost"]), output_mismatch=len(res["mismatch"]),
          race_drill=res.get("drill"), store=store.stats(), minutes=round((time.time() - t_start) / 60, 1))
    return _exit_code(res)


def _race_lines(res: dict) -> list[str]:
    out = []
    if res["skipped"] or res["lost"]:
        out.append(f"units run here but committed by another part: skipped {res['skipped']}, lost {res['lost']}")
    if res["mismatch"]:
        out.append(f"**output mismatch with the winning marker: {res['mismatch'][:10]}**")
    if res.get("drill"):
        out.append(f"race drill: {json.dumps(res['drill'])}")
    return out


def _exit_code(res: dict) -> int:
    """3 = R2 outputs disagree with a marker written by another part (lost race, different bytes) or a failed
    race drill; the run is then not trustworthy until checked (fetch_run --verify)."""
    if res["mismatch"] or (res.get("drill") and not res["drill"]["ok"]):
        return 3
    return 0


def drive(a, t_start: float, store, job: str, run_id: str, allowed: list, run_unit, mirror: Path,
          commit=None) -> dict:
    """Stage loop shared by p4b, p4b_smoke, p4b_post and p4d: run every allowed unit whose dependencies have markers,
    wait for other parts up to --wait-minutes, commit each unit with commit_unit() (p4d: commit_unit_safe())."""
    commit = commit or commit_unit
    res = {"won": [], "skipped": [], "lost": [], "yielded": [], "mismatch": [], "metrics": {}}
    waited = 0.0
    while True:
        if (time.time() - t_start) / 60 > a.max_minutes:
            C.log("time_budget_reached", hint="run the job again (same inputs): it resumes")
            break
        done = done_units(store, job, run_id)
        todo = [u for u in allowed if u.id not in done]
        if not todo:
            break
        ready = [u for u in todo if all(d in done for d in u.deps)]
        if not ready:
            low = min(u.stage for u in todo)          # report what blocks the lowest pending stage
            blocked = sorted({d for u in todo if u.stage == low for d in u.deps if d not in done})
            if waited >= a.wait_minutes:
                C.log("waiting_for_other_parts", missing=blocked[:20], n_missing=len(blocked),
                      hint="start this part again after the other parts have finished; it resumes")
                break
            time.sleep(60)
            waited += 1.0
            continue
        u = ready[0]
        t1 = time.time()
        C.log("unit_start", unit=u.id)
        rels, metrics = run_unit(u)
        c = commit(store, job, run_id, u, rels, metrics, mirror)
        res[c["state"]].append(u.id)
        res["mismatch"] += c["mismatch"]
        if c["state"] == "won" and u.kind in ("gate", "final"):
            res["metrics"][u.id] = metrics
        C.log("unit_done", unit=u.id, state=c["state"], seconds=round(time.time() - t1, 1),
              **{k: v for k, v in metrics.items() if k in ("pooled_logloss", "median_rounds", "rows_A1", "chosen",
                                                           "feature_set", "ship_B", "violations", "rows", "kept",
                                                           "chain_end", "ring", "cut", "rss_gb", "train_rows",
                                                           "r2_total_gb", "projected_total_gb", "a1_equal",
                                                           "ring_pixels", "gold_bytes", "alias", "rows_ring")})
    return res


def race_drill(store, job: str, run_id: str, units: list, run_unit, mirror: Path) -> dict:
    """Cloud check of the race fix on a finished smoke run: every gate / final unit is run again and committed
    through commit_unit(); each must come back 'skipped' (marker exists: no upload, no marker, no summary)."""
    pre = run_prefix(job, run_id)
    done = done_units(store, job, run_id)

    def _sums() -> dict:
        return {k: C.sha256_bytes(store.get(pre + k)) for k in ("summary.json", "done.json")
                if pre + k in store.list(pre + k)}

    def _outputs() -> dict:
        return {k: C.sha256_bytes(store.get(k)) for k in sorted(store.list(pre + "files/"))}

    before, files_before = _sums(), _outputs()
    states = {}
    for u in units:
        if u.id not in done:
            states[u.id] = "not_done"
            continue
        rels, metrics = run_unit(u)
        states[u.id] = commit_unit(store, job, run_id, u, rels, metrics, mirror)["state"]
    after, files_after = _sums(), _outputs()
    ok = (bool(states) and all(v == "skipped" for v in states.values()) and before == after
          and len(before) == 2 and files_before == files_after)
    C.log("race_drill", ok=ok, states=states, summary_and_done_unchanged=before == after,
          outputs_unchanged=files_before == files_after, present=sorted(before))
    return {"ok": ok, "states": states}


def main_post(a, t_start: float) -> int:
    """p4b_post: post-final outputs of the fixed run p4b/c0df12477947 (no training, no test re-scoring).
    Writes only under lake/v1/runs/p4b_post/<run_id>/ (store restriction)."""
    from src.models import p4b_post as PP
    if a.final or a.write_oof_done or a.approve_final:
        C.log("refused", reason="p4b_post never runs --final / --write-oof-done / --approve-final")
        return 2
    run_id = a.run_id or run_id_for(a.job)
    store = open_run_store(a.store, a.job)
    units = PP.units()
    C.log("plan", job=a.job, run_id=run_id, source_job=PP.SOURCE_JOB, source_run_id=PP.SOURCE_RUN,
          source_gold_id=PP.SOURCE_GOLD, units=len(units), done=len(done_units(store, a.job, run_id)))
    if a.plan:
        return 0
    mirror = ROOT / "cloud_out" / a.job / run_id
    store.put_json(latest_key(a.job), {"job": a.job, "run_id": run_id, "updated_utc": C.iso(C.utcnow()),
                                       "context": C.run_context()}, overwrite=True)
    pctx = PP.PostCtx(store, run_id, C.log)
    res = drive(a, t_start, store, a.job, run_id, units, lambda u: PP.run_unit(u, pctx), mirror)
    done = done_units(store, a.job, run_id)
    missing = [u.id for u in units if u.id not in done]
    if "post-report" in res["won"]:
        store.put_json(run_prefix(a.job, run_id) + "summary.json",
                       {"job": a.job, "run_id": run_id, "source": f"{PP.SOURCE_JOB}/{PP.SOURCE_RUN}",
                        "stage": "post-final outputs complete", "note": PP.NOT_FLOOD,
                        "metrics": res["metrics"].get("post-report"), "finished_utc": C.iso(C.utcnow()),
                        "context": C.run_context()})
        store.put_json(run_prefix(a.job, run_id) + "done.json", {"job": a.job, "run_id": run_id,
                                                                   "finished_utc": C.iso(C.utcnow())})
    extra = [f"source run `{PP.SOURCE_JOB}/{PP.SOURCE_RUN}` (read only), gold `{PP.SOURCE_GOLD}`", PP.NOT_FLOOD,
             *_race_lines(res)]
    write_step_summary(a.job, step_summary(a.job, run_id, res["won"], len(missing), res["metrics"] or None, extra))
    C.log("end", job=a.job, run_id=run_id, finished_now=len(res["won"]), missing=len(missing),
          done="post-report" in done, store=store.stats(), minutes=round((time.time() - t_start) / 60, 1))
    return _exit_code(res)


def main_p4d(a, t_start: float) -> int:
    """p4d / p4d_smoke: P4D.3-P4D.7 in one run (Gold table_v2, LOSO fits, selection gates, comparison). Stops at
    compare-done (COMPARE_DONE.json); no --final, no OOF_DONE, no race drill in this phase."""
    from src.models import p4d as PD
    if a.final or a.write_oof_done or a.approve_final or a.race_drill:
        C.log("refused", job=a.job, reason="p4d has no --final / --write-oof-done / --approve-final / --race-drill")
        return 2
    smoke = a.job == "p4d_smoke"
    cfg = PD.load_cfg(smoke=smoke)
    i, n = parse_part(a.part)
    units = PD.units(cfg)
    mine = PD.part_units(units, i, n)
    gates = [u for u in units if u.kind == "gate"]
    run_id = a.run_id or run_id_for(a.job)
    store = open_run_store(a.store, a.job)
    done = done_units(store, a.job, run_id)
    C.log("plan", job=a.job, run_id=run_id, part=f"{i}/{n}", units_total=len(units), units_in_part=len(mine),
          done=len(done), todo=[u.id for u in mine + gates if u.id not in done][:40], smoke=smoke,
          gold_id=PD.gold_id(cfg), note=PD.NOT_FLOOD)
    icv = PD.inner_cv_plan_check(cfg)                 # fail fast: every fold / variant needs >= 2 inner blocks
    bad = [r for r in icv if not r["ok"]]
    C.log("inner_cv_check", ok=not bad, cases=len(icv), failing=bad[:6])
    if bad:
        C.log("refused", job=a.job, reason="inner CV would have < 2 usable blocks or < 2 distinct calendar months "
                                           "for some fold / variant (see inner_cv_check); fix scope.day_ranges in "
                                           "configs/model_v2.yaml")
        return 2
    if a.plan:
        return 0
    ok, n0 = parts_lock(store, a.job, run_id, n)
    if not ok:
        C.log("refused", job=a.job, run_id=run_id, reason=f"this run is split into {n0} parts; got --part {i}/{n}")
        return 2
    mirror = ROOT / "cloud_out" / a.job / run_id
    store.put_json(latest_key(a.job), {"job": a.job, "run_id": run_id, "updated_utc": C.iso(C.utcnow()),
                                       "context": C.run_context()}, overwrite=True)
    ctx = PD.Ctx(cfg, store, a.job, run_id, mirror, C.log)
    res = drive(a, t_start, store, a.job, run_id, mine + gates, lambda u: PD.run_unit(u, ctx), mirror,
                commit=commit_unit_safe)
    done = done_units(store, a.job, run_id)
    missing = [u.id for u in units if u.id not in done]
    if "compare-done" in res["won"]:
        summ = {"job": a.job, "run_id": run_id, "gold_id": PD.gold_id(cfg), "stage": "P4D.7 comparison complete",
                "metrics": res["metrics"].get("compare-done"), "finished_utc": C.iso(C.utcnow()),
                "note": PD.NOT_FLOOD, "context": C.run_context()}
        store.put_json(run_prefix(a.job, run_id) + "summary.json", summ)
        store.put_json(run_prefix(a.job, run_id) + "COMPARE_DONE.json",
                       {"job": a.job, "run_id": run_id, "compare_marker": marker_key(a.job, run_id, "compare-done"),
                        "finished_utc": summ["finished_utc"],
                        "note": "stop point after P4D.7: the user chooses Model 1 or Model 2 after the report"})
    extra = [f"part {i}/{n}; gold `{PD.gold_id(cfg)}`", PD.NOT_FLOOD, *_race_lines(res)]
    write_step_summary(a.job, step_summary(a.job, run_id, res["won"], len(missing), res["metrics"] or None, extra))
    C.log("end", job=a.job, run_id=run_id, part=f"{i}/{n}", finished_now=len(res["won"]), missing=len(missing),
          compare_complete="compare-done" in done, skipped=len(res["skipped"]), lost=len(res["lost"]),
          output_mismatch=len(res["mismatch"]), store=store.stats(), minutes=round((time.time() - t_start) / 60, 1))
    return _exit_code(res)


def main_p4e(a, t_start: float) -> int:
    """p4e_smoke (P4E.0): single-writer commits (commit_unit_single, D188 fix), P4E learning rate / round cap, one
    downsampled and one two-stage unit, comparison gate, cost record; --race-drill runs the extended drill. Stops at
    compare-done (COMPARE_DONE.json). No --final, no OOF_DONE."""
    import functools
    from src.models import p4e as PE
    if a.final or a.write_oof_done or a.approve_final:
        C.log("refused", job=a.job, reason="p4e has no --final / --write-oof-done / --approve-final")
        return 2
    smoke = a.job == "p4e_smoke"
    cfg = PE.load_cfg(smoke=smoke)                     # raises if the pre-registered sections changed (D189)
    i, n = parse_part(a.part)
    units = PE.units(cfg)
    mine = PE.part_units(units, i, n)
    gates = [u for u in units if u.kind == "gate"]
    run_id = a.run_id or run_id_for(a.job)
    store = open_run_store(a.store, a.job)
    done = done_units(store, a.job, run_id)
    est = PE.plan_estimate(cfg, units, n)
    C.log("plan", job=a.job, run_id=run_id, part=f"{i}/{n}", units_total=len(units), units_in_part=len(mine),
          done=len(done), todo=[u.id for u in mine + gates if u.id not in done][:40], smoke=smoke,
          gold_id=PE.gold_id(cfg), prereg_sha=PE.PREREG_SHA, note=PE.NOT_FLOOD)
    C.log("estimate", parts=n, part_minutes=est["part_minutes"], runner_minutes=est["runner_minutes"],
          max_part_minutes=est["max_part_minutes"], within_target=est["within_target"])
    icv = PE.inner_cv_plan_check(cfg)
    bad = [r for r in icv if not r["ok"]]
    C.log("inner_cv_check", ok=not bad, cases=len(icv), failing=bad[:6])
    if bad:
        C.log("refused", job=a.job, reason="inner CV would have < 2 usable blocks or < 2 calendar months")
        return 2
    if est["max_part_minutes"] > float(cfg["planning"]["part_hard_limit_minutes"]):
        C.log("refused", job=a.job, reason=f"slowest part estimated at {est['max_part_minutes']} min > hard limit")
        return 2
    if a.plan:
        return 0
    ok, n0 = parts_lock(store, a.job, run_id, n)
    if not ok:
        C.log("refused", job=a.job, run_id=run_id, reason=f"this run is split into {n0} parts; got --part {i}/{n}")
        return 2
    mirror = ROOT / "cloud_out" / a.job / run_id
    lk = latest_key(a.job)                             # deterministic content: parts of one run write equal bytes
    body = {"job": a.job, "run_id": run_id}
    if lk not in store.list(lk) or json.loads(store.get(lk)) != body:
        store.put_json(lk, body, overwrite=True)
    ctx = PE.Ctx(cfg, store, a.job, run_id, mirror, C.log)
    rc = cfg["race"]
    commit = functools.partial(commit_unit_single, stale_minutes=float(rc["claim_stale_minutes"]),
                               poll_seconds=float(rc["claim_poll_seconds"]),
                               scratch_kinds=tuple(rc["loser_scratch_kinds"]))
    # phase 1: opening gates + this part's work units (waits up to --wait-minutes for the opening gates of other
    # parts); phase 2: closing gates with a short wait, so a part whose work is done exits instead of idling (P4D:
    # idle waiting at gates was ~1,300 of ~6,700 runner minutes, D191). The part that finishes last runs them.
    run = lambda u: PE.run_unit(u, ctx)                # noqa: E731
    opening = [u for u in gates if u.stage <= PE.LAST_OPENING_STAGE]
    closing = [u for u in gates if u.stage > PE.LAST_OPENING_STAGE]
    res = drive(a, t_start, store, a.job, run_id, mine + opening, run, mirror, commit=commit)
    import copy
    a2 = copy.copy(a)
    a2.wait_minutes = min(a.wait_minutes, float(rc["close_gate_wait_minutes"]))
    r2 = drive(a2, t_start, store, a.job, run_id, closing, run, mirror, commit=commit)
    for k, v in r2.items():
        if isinstance(v, list):
            res[k] += v
        else:
            res[k].update(v)
    if a.race_drill:
        g = {u.id: u for u in gates}[PE.DRILL_GATE]
        if g.id in done_units(store, a.job, run_id):
            res["drill"] = race_drill_single(store, a.job, run_id, g, lambda u: PE.run_unit(u, ctx), mirror)
        else:
            res["drill"] = {"ok": False, "reason": f"{g.id} not done yet in this run"}
    done = done_units(store, a.job, run_id)
    missing = [u.id for u in units if u.id not in done]
    if "compare-done" in res["won"]:                   # single writer: only the winner of compare-done
        summ = {"job": a.job, "run_id": run_id, "gold_id": PE.gold_id(cfg),
                "stage": "P4E.0 smoke complete" if smoke else "P4E comparison complete",
                "metrics": res["metrics"].get("compare-done"), "note": PE.NOT_FLOOD}
        store.put_json(run_prefix(a.job, run_id) + "summary.json", summ)
        store.put_json(run_prefix(a.job, run_id) + "COMPARE_DONE.json",
                       {"job": a.job, "run_id": run_id, "compare_marker": marker_key(a.job, run_id, "compare-done"),
                        "note": "stop point: the hub decides (P4E)"})
    extra = [f"part {i}/{n}; gold `{PE.gold_id(cfg)}`; estimate {est['part_minutes']} min per part",
             PE.NOT_FLOOD, *_race_lines(res)]
    write_step_summary(a.job, step_summary(a.job, run_id, res["won"], len(missing), res["metrics"] or None, extra))
    C.log("end", job=a.job, run_id=run_id, part=f"{i}/{n}", finished_now=len(res["won"]), missing=len(missing),
          compare_complete="compare-done" in done, skipped=len(res["skipped"]), yielded=len(res["yielded"]),
          lost=len(res["lost"]), output_mismatch=len(res["mismatch"]), race_drill=res.get("drill"),
          store=store.stats(), minutes=round((time.time() - t_start) / 60, 1))
    return _exit_code(res)


if __name__ == "__main__":
    raise SystemExit(C.cli(main))
