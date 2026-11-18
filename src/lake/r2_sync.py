"""Upload the Phase-1 lake to Cloudflare R2 under lake/v1/ (never deletes).

  .venv/Scripts/python.exe -m src.lake.r2_sync --dry-run
  .venv/Scripts/python.exe -m src.lake.r2_sync
  .venv/Scripts/python.exe -m src.lake.r2_sync --verify
  ... --exclude data/bronze_regional/imerg_late --exclude data/silver/met/imerg_late
  ... --only data/gold/met_training --only models      (P4C.1: restrict the run to these directories)

--exclude DIR (repo-relative, repeatable): the directory is pruned before it is walked,
so no file below it is stat'ed, hashed or uploaded (used while a backfill still writes
there). The same prefixes are excluded from --verify.

Credentials: R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET from
the environment (loaded from .env at runtime). Values are never printed.
Objects with identical size and sha256 (stored as x-amz-meta-sha256) are skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
PREFIX = "lake/v1/"
REQUIRED = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")
INCLUDE = ("data/bronze_regional/**/*", "data/silver/met/**/*", "reports/p2b/**/*", "data/reference/*hanoi_city*",
           "data/reference/rain_pixels_hanoi_city.parquet", "configs/met_dataset.yaml",
           # P4C.1: Gold training tables and pilot models for cloud training (same relative layout)
           "data/gold/met_training/**/*", "models/**/*")
EXCLUDE_SUFFIX = (".part",)


def _norm_excludes(exclude) -> list[Path]:
    out = []
    for e in exclude or ():
        p = Path(os.path.abspath(ROOT / e))
        if p == ROOT or ROOT not in p.parents:
            raise ValueError(f"--exclude must be a sub-directory of the repo: {e}")
        out.append(p)
    return out


def _excluded(p: Path, ex: list[Path]) -> bool:
    return any(p == e or e in p.parents for e in ex)


def _norm_only(only) -> list[Path]:
    return _norm_excludes(only)


def local_files(exclude=None, only=None) -> list[Path]:
    ex = _norm_excludes(exclude)
    keep = _norm_only(only)
    out = set()
    for pat in INCLUDE:
        if pat.endswith("/**/*"):
            base = ROOT / pat[: -len("/**/*")]
            for dirpath, dirnames, filenames in os.walk(base):
                d = Path(dirpath)
                dirnames[:] = [n for n in dirnames if not _excluded(d / n, ex)]
                if _excluded(d, ex):
                    continue
                out |= {d / n for n in filenames if not n.endswith(EXCLUDE_SUFFIX)}
        else:
            out |= {p for p in ROOT.glob(pat) if not _excluded(p, ex) and p.is_file()
                    and not p.name.endswith(EXCLUDE_SUFFIX)}
    if keep:
        out = {p for p in out if _excluded(p, keep)}
    return sorted(out)


def key_for(p: Path) -> str:
    k = PREFIX + p.relative_to(ROOT).as_posix()
    if not k.startswith(PREFIX) or ".." in k.split("/"):
        raise ValueError(f"refusing key outside {PREFIX}: {k}")
    return k


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def env_status() -> tuple[bool, list[str]]:
    load_dotenv(ROOT / ".env", override=False)
    missing = [n for n in REQUIRED if not os.getenv(n)]
    return not missing, missing


def client():
    import boto3
    from botocore.config import Config
    return boto3.client(
        "s3", endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto", config=Config(retries={"max_attempts": 3, "mode": "standard"}))


def remote_listing(s3, bucket: str) -> dict[str, int]:
    out, token = {}, None
    while True:
        kw = {"Bucket": bucket, "Prefix": PREFIX}
        if token:
            kw["ContinuationToken"] = token
        r = s3.list_objects_v2(**kw)
        for o in r.get("Contents", []):
            out[o["Key"]] = o["Size"]
        if not r.get("IsTruncated"):
            return out
        token = r["NextContinuationToken"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--exclude", action="append", default=[], metavar="DIR",
                    help="repo-relative directory to skip entirely (repeatable)")
    ap.add_argument("--only", action="append", default=[], metavar="DIR",
                    help="repo-relative directory: restrict the run (and --verify) to files below it (repeatable)")
    ap.add_argument("--workers", type=int, default=1, help="parallel uploads (uploads only; downloads stay sequential)")
    ap.add_argument("--max-files", type=int, default=0,
                    help="upload at most N files in this run (0 = all); re-run to continue (identical files are skipped)")
    a = ap.parse_args(argv)
    files = local_files(a.exclude, a.only)
    plan = [(p, key_for(p), p.stat().st_size) for p in files]
    local_bytes = sum(s for _, _, s in plan)
    ok, missing = env_status()
    if not ok:
        print(json.dumps({"status": "NEEDS_REVIEW", "missing_env_vars": missing,
                          "local_files": len(plan), "local_bytes": local_bytes}))
        return 3
    bucket = os.environ["R2_BUCKET"]
    s3 = client()
    remote = remote_listing(s3, bucket)
    if a.exclude:
        ex_keys = tuple(PREFIX + Path(os.path.relpath(p, ROOT)).as_posix() + "/" for p in _norm_excludes(a.exclude))
        remote = {k: v for k, v in remote.items() if not k.startswith(ex_keys)}
    if a.only:
        only_keys = tuple(PREFIX + Path(os.path.relpath(p, ROOT)).as_posix() + "/" for p in _norm_only(a.only))
        remote = {k: v for k, v in remote.items() if k.startswith(only_keys)}
    if a.verify:
        want = {k: s for _, k, s in plan}
        missing_keys = [k for k in want if k not in remote]
        size_diff = [k for k in want if k in remote and remote[k] != want[k]]
        res = {"local_files": len(want), "local_bytes": local_bytes,
               "remote_objects_under_prefix": len(remote), "remote_bytes_under_prefix": sum(remote.values()),
               "missing_remote": len(missing_keys), "size_mismatch": len(size_diff),
               "examples": (missing_keys + size_diff)[:5]}
        res["match"] = not missing_keys and not size_diff
        print(json.dumps(res, indent=2))
        return 0 if res["match"] else 1
    up = skip = 0
    up_bytes = 0
    def check(item):
        p, k, size = item
        digest = sha256(p)
        if k in remote and remote[k] == size:
            meta = s3.head_object(Bucket=bucket, Key=k).get("Metadata", {})
            if meta.get("sha256") == digest:
                return None
        return (p, k, size, digest)

    if a.workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            checked = list(ex.map(check, plan))
    else:
        checked = [check(i) for i in plan]
    todo = [c for c in checked if c is not None]
    skip = len(checked) - len(todo)
    remaining = len(todo)
    if a.max_files:
        todo = todo[: a.max_files]

    def put(item):
        p, k, size, digest = item
        assert k.startswith(PREFIX)
        s3.upload_file(str(p), bucket, k, ExtraArgs={"Metadata": {"sha256": digest}})
        return size

    if a.dry_run:
        for p, k, size, _ in todo:
            print(json.dumps({"would_upload": k, "bytes": size}))
            up, up_bytes = up + 1, up_bytes + size
    elif a.workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            for size in ex.map(put, todo):
                up, up_bytes = up + 1, up_bytes + size
    else:
        for item in todo:
            up, up_bytes = up + 1, up_bytes + put(item)
    remaining -= up if not a.dry_run else 0
    print(json.dumps({"dry_run": a.dry_run, "uploaded" if not a.dry_run else "to_upload": up,
                      "bytes": up_bytes, "skipped_identical": skip, "remaining_after_run": remaining, "local_files": len(plan),
                      "local_bytes": local_bytes}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
