"""P4C.4 — read back a cloud training run from R2 (read-only on R2).

  python -m src.models.fetch_run --job p4a6 [--run-id latest] [--store r2|local:<dir>]

* Everything under lake/v1/runs/<job>/<run_id>/ is mirrored to reports/<job>/cloud/<run_id>/ (files/, markers/,
  summary.json, done.json).
* Run outputs that belong in the working tree are installed at their repository path:
  models/** , data/gold/met_training/pilot_v0/preds/** and the JSON run records reports/p4a/runs/*.json.
  An identical local file is left alone; a different local file is first moved to
  _archive/fetch_run/<job>/<run_id>/<path> (never deleted), then replaced. Every action is listed in
  reports/<job>/cloud/<run_id>/fetch_log.json.
* Reports (reports/p4a/...) produced in the cloud stay in the mirror; the local REPORT is written from them.
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

MIRROR_ONLY_JOBS = ("p4b_smoke",)   # smoke outputs stay in the mirror; never installed into models/ (D101)
INSTALL_PREFIXES = ("models/", "data/gold/met_training/pilot_v0/preds/", "reports/p4a/runs/")


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
        if job in MIRROR_ONLY_JOBS or not rel.startswith(INSTALL_PREFIXES) or ".." in rel.split("/"):
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, choices=CT.JOBS)
    ap.add_argument("--run-id", default="latest")
    ap.add_argument("--store", default="r2")
    ap.add_argument("--dest-root", default=None, help="tests only: install below this directory instead of the repo")
    a = ap.parse_args(argv)
    store = CT.open_run_store(a.store, a.job)          # reads only; puts would be restricted to runs/<job>/
    run_id = resolve_run_id(store, a.job, a.run_id)
    log = fetch(store, a.job, run_id, Path(a.dest_root) if a.dest_root else ROOT)
    C.log("fetched", job=a.job, run_id=run_id, objects=log["objects"], bytes=log["bytes"],
          installed=len(log["installed"]), identical=log["identical"], archived=len(log["archived"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(C.cli(main))
