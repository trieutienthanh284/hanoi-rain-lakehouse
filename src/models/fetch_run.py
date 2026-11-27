"""P4C.4 — read back a cloud training run from R2 (read-only on R2).

  python -m src.models.fetch_run --job p4a6 [--run-id latest] [--store r2|local:<dir>] [--verify]

* Everything under lake/v1/runs/<job>/<run_id>/ is mirrored to reports/<job>/cloud/<run_id>/ (files/, markers/,
  summary.json, done.json).
* Run outputs that belong in the working tree are installed at their repository path:
  models/** , data/gold/met_training/pilot_v0/preds/** and the JSON run records reports/p4a/runs/*.json.
  An identical local file is left alone; a different local file is first moved to
  _archive/fetch_run/<job>/<run_id>/<path> (never deleted), then replaced. Every action is listed in
  reports/<job>/cloud/<run_id>/fetch_log.json.
* Reports (reports/p4a/...) produced in the cloud stay in the mirror; the local REPORT is written from them.
* --verify (after the fetch, read only): every output listed in every marker must exist in the mirror with the
  marker's sha256, no mirrored file may be unlisted, and summary.json / done.json are reported; exit 1 otherwise
  (race-fix check, REPORT_P4B §3.5).
* p4e jobs (D196): the cloud figures (files/reports/p4e/figs/) are also copied to reports/p4e/figs/<job>_<run_id>/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import cloud_train as CT  # noqa: E402
from src.realtime import common as C  # noqa: E402

MIRROR_ONLY_JOBS = ("p4b_smoke", "p4b_post", "p4d", "p4d_smoke", "p4e_smoke", "p4fg_smoke", "p4fg", "nwpconv_backfill")   # outputs stay in the mirror; never installed (D101)
INSTALL_PREFIXES = ("models/", "data/gold/met_training/pilot_v0/preds/", "reports/p4a/runs/")
# P4F/P4G: p4fg_final installs only the v2 candidate models (models/v1/ and everything else stay untouched)
JOB_INSTALL_PREFIXES = {"p4fg_final": ("models/v2_candidate/",)}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def resolve_run_id(store, job: str, run_id: str) -> str:
    if run_id != "latest":
        return run_id
    return json.loads(store.get(CT.latest_key(job)))["run_id"]


def fetch(store, job: str, run_id: str, dest: Path) -> dict:
    pre = CT.run_prefix(job, run_id)
    keys = store.list(pre)
    if not keys:
        raise SystemExit(f"no objects under {pre}")
    mirror = dest / "reports" / job / "cloud" / run_id
    log = {"job": job, "run_id": run_id, "objects": len(keys), "bytes": sum(keys.values()), "installed": [],
           "identical": 0, "archived": []}
    for key in sorted(keys):
        rel_run = key[len(pre):]
        data = store.get(key)
        m = mirror / rel_run
        m.parent.mkdir(parents=True, exist_ok=True)
        m.write_bytes(data)
        if not rel_run.startswith("files/"):
            continue
        rel = rel_run[len("files/"):]
        if job in MIRROR_ONLY_JOBS or not rel.startswith(JOB_INSTALL_PREFIXES.get(job, INSTALL_PREFIXES))                 or ".." in rel.split("/"):
            continue
        dst = dest / rel
        if dst.is_file():
            if _sha(dst.read_bytes()) == _sha(data):
                log["identical"] += 1
                continue
            arc = dest / "_archive" / "fetch_run" / job / run_id / rel
            arc.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(dst), str(arc))
            log["archived"].append(rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".part")
        tmp.write_bytes(data)
        os.replace(tmp, dst)
        log["installed"].append(rel)
    (mirror / "fetch_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    return log


def verify(mirror: Path, job: str, run_id: str) -> dict:
    """Markers vs mirrored bytes (the mirror is an exact copy of the run prefix right after fetch())."""
    pre = CT.run_prefix(job, run_id) + "files/"
    listed, bad, missing = {}, [], []
    markers = sorted((mirror / "markers").glob("*.done.json"))
    for m in markers:
        for key, sha in json.loads(m.read_text(encoding="utf-8")).get("outputs", {}).items():
            rel = key[len(pre):] if key.startswith(pre) else None
            if rel is None:
                bad.append(key)
                continue
            listed[rel] = sha
            f = mirror / "files" / rel
            if not f.is_file():
                missing.append(rel)
            elif _sha(f.read_bytes()) != sha:
                bad.append(rel)
    have = {p.relative_to(mirror / "files").as_posix() for p in (mirror / "files").rglob("*") if p.is_file()}         if (mirror / "files").is_dir() else set()
    unlisted = sorted(have - set(listed))
    out = {"markers": len(markers), "outputs": len(listed), "sha_mismatch": bad, "missing": missing,
           "unlisted": unlisted, "summary_json": (mirror / "summary.json").is_file(),
           "done_json": (mirror / "done.json").is_file(), "compare_done_json": (mirror / "COMPARE_DONE.json").is_file()}
    out["ok"] = not (bad or missing or unlisted)
    # single-writer evidence (p4e, D188 fix): claim generations per unit and losing contenders' scratch prefixes
    cl = sorted((mirror / "claims").glob("*/g*.json")) if (mirror / "claims").is_dir() else []
    out["claims"] = len(cl)
    out["claims_superseded"] = sorted({p.parent.name for p in cl if p.name != "g001.json"})
    out["scratch_attempts"] = sorted(p.name for p in (mirror / "scratch").iterdir()) \
        if (mirror / "scratch").is_dir() else []
    return out


def copy_figs(dest: Path, job: str, run_id: str, phase: str = "p4e") -> list[str]:
    """p4e (D196): copy the cloud figures of the mirror (files/reports/p4e/figs/) to reports/p4e/figs/<job>_<run_id>/
    (new folder per run; an existing file with other bytes is never overwritten: error)."""
    src = dest / "reports" / job / "cloud" / run_id / "files" / "reports" / phase / "figs"
    out = dest / "reports" / phase / "figs" / f"{job}_{run_id}"
    done = []
    if not src.is_dir():
        return done
    for f in sorted(src.rglob("*")):
        if not f.is_file():
            continue
        t = out / f.relative_to(src)
        b = f.read_bytes()
        if t.is_file():
            if t.read_bytes() != b:
                raise RuntimeError(f"{t} exists with other bytes; not overwritten")
        else:
            t.parent.mkdir(parents=True, exist_ok=True)
            t.write_bytes(b)
        done.append(t.relative_to(dest).as_posix())
    return done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, choices=CT.JOBS)
    ap.add_argument("--run-id", default="latest")
    ap.add_argument("--store", default="r2")
    ap.add_argument("--dest-root", default=None, help="tests only: install below this directory instead of the repo")
    ap.add_argument("--verify", action="store_true", help="after the fetch: check every marker output (sha256)")
    a = ap.parse_args(argv)
    store = CT.open_run_store(a.store, a.job)          # reads only; puts would be restricted to runs/<job>/
    run_id = resolve_run_id(store, a.job, a.run_id)
    log = fetch(store, a.job, run_id, Path(a.dest_root) if a.dest_root else ROOT)
    C.log("fetched", job=a.job, run_id=run_id, objects=log["objects"], bytes=log["bytes"],
          installed=len(log["installed"]), identical=log["identical"], archived=len(log["archived"]))
    if a.job.startswith(("p4e", "p4fg_smoke")) or a.job == "p4fg":
        figs = copy_figs(Path(a.dest_root) if a.dest_root else ROOT, a.job, run_id,
                         "p4e" if a.job.startswith("p4e") else "p4fg")
        C.log("figs_copied", job=a.job, run_id=run_id, files=len(figs))
    if a.verify:
        dest = Path(a.dest_root) if a.dest_root else ROOT
        v = verify(dest / "reports" / a.job / "cloud" / run_id, a.job, run_id)
        C.log("verified", job=a.job, run_id=run_id, ok=v["ok"], markers=v["markers"], outputs=v["outputs"],
              sha_mismatch=len(v["sha_mismatch"]), missing=len(v["missing"]), unlisted=len(v["unlisted"]),
              summary_json=v["summary_json"], done_json=v["done_json"], compare_done_json=v["compare_done_json"],
              claims=v["claims"], claims_superseded=v["claims_superseded"], scratch_attempts=v["scratch_attempts"],
              first_problems=(v["sha_mismatch"] +
                                                                                        v["missing"] + v["unlisted"])[:10])
        return 0 if v["ok"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(C.cli(main))
