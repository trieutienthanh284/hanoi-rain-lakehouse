"""Shared plumbing for the realtime entry points (Phase 3A).

* Remote layout (R2, all state lives there; runners are ephemeral):
    lake/v1/realtime/{source}/date=YYYY-MM-DD/{id}.{h5|parquet}
    lake/v1/realtime/{source}/manifest/date=YYYY-MM-DD/{id}.json
    lake/v1/status/freshness.json
  Writes outside ``lake/v1/realtime/`` and ``lake/v1/status/`` are refused.
  Nothing is ever deleted. Objects are put-if-absent; the only overwritable key
  is the status snapshot listed in ``configs/realtime.yaml``.
* Idempotency = "skip if the manifest object already exists". The manifest is
  written after its data object, so it is the commit marker.
* Secrets come only from environment variables (``.env`` is loaded at runtime
  locally by the reused P1 helpers). ``redacted_output`` wraps stdout/stderr so
  that no secret value can reach a log line, even inside a traceback.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG = ROOT / "configs" / "realtime.yaml"
SECRET_ENV = ("EARTHDATA_TOKEN", "GSMAP_FTP_HOST", "GSMAP_FTP_USER", "GSMAP_FTP_PASSWORD",
              "R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")
MASK = "***"
MIN_SECRET_LEN = 4   # never redact 1-3 character values (would mangle ordinary text)


# ---------------------------------------------------------------- config / time
def load_config(path: Path = CONFIG) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def workdir(cfg: dict, sub: str) -> Path:
    """Local working copy: $RUNNER_TEMP/hfl on GitHub runners, else data/_ci_test/work."""
    base = Path(os.environ["RUNNER_TEMP"]) / "hfl" if os.environ.get("RUNNER_TEMP") \
        else ROOT / cfg["workdir"]["local_default"]
    p = base / sub
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_context() -> dict:
    """Non-secret provenance of the process that ingested a record."""
    return {k.lower(): os.environ.get(k) for k in ("GITHUB_RUN_ID", "GITHUB_SHA", "GITHUB_WORKFLOW")
            if os.environ.get(k)} or {"runner": "local"}


# ---------------------------------------------------------------- secret redaction
def secret_values() -> list[str]:
    vals = {os.environ.get(n, "") for n in SECRET_ENV}
    vals = {v.strip() for v in vals if v and len(v.strip()) >= MIN_SECRET_LEN}
    return sorted(vals, key=len, reverse=True)


def redact(text: str, secrets: Iterable[str] | None = None) -> str:
    for v in (secret_values() if secrets is None else secrets):
        if v and v in text:
            text = text.replace(v, MASK)
    return text


class RedactingStream(io.TextIOBase):
    """Text stream wrapper that masks secret values before they are written.
    Secret values are re-read from the environment on every write, so values
    loaded from .env after start-up are covered too."""

    def __init__(self, inner):
        self.inner = inner

    def write(self, s):
        return self.inner.write(redact(str(s)))

    def flush(self):
        self.inner.flush()

    @property
    def encoding(self):
        return getattr(self.inner, "encoding", "utf-8")


@contextlib.contextmanager
def redacted_output():
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = RedactingStream(old_out), RedactingStream(old_err)
    try:
        yield
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            sys.stdout, sys.stderr = old_out, old_err


def log(event: str, **fields) -> None:
    print(json.dumps({"event": event, **fields}, default=str, ensure_ascii=False), flush=True)


def cli(main: Callable[[list[str] | None], int], argv=None) -> int:
    """Run ``main`` with redacted stdout/stderr; uncaught errors -> exit 1 (redacted traceback)."""
    with redacted_output():
        try:
            return main(argv)
        except SystemExit:
            raise
        except Exception:
            traceback.print_exc()
            return 1


# ---------------------------------------------------------------- keys
def data_key(source: str, date: str, obj_id: str, ext: str) -> str:
    return f"lake/v1/realtime/{source}/date={date}/{obj_id}.{ext}"


def manifest_prefix(source: str, date: str) -> str:
    return f"lake/v1/realtime/{source}/manifest/date={date}/"


def manifest_key(source: str, date: str, obj_id: str) -> str:
    return manifest_prefix(source, date) + f"{obj_id}.json"


def id_from_manifest_key(key: str) -> str:
    return key.rsplit("/", 1)[-1][: -len(".json")]


# ---------------------------------------------------------------- stores
class KeyRefused(ValueError):
    pass


class ObjectConflict(RuntimeError):
    """An object exists with different content; it is never overwritten."""


class Store:
    """put-if-absent object store restricted to the realtime/status prefixes."""

    def __init__(self, cfg: dict):
        r = cfg["remote"]
        self.allowed = (r["realtime_prefix"], r["status_prefix"])
        self.overwritable = set(r.get("overwritable_keys", []))
        self.ops = {"class_a": 0, "class_b": 0}      # R2 pricing classes (A: list/put, B: get/head)
        self.uploaded_objects = 0
        self.uploaded_bytes = 0

    def check_key(self, key: str, overwrite: bool = False) -> None:
        if not key.startswith(self.allowed) or ".." in key.split("/") or "//" in key:
            raise KeyRefused(f"refusing key outside {self.allowed}: {key}")
        if overwrite and key not in self.overwritable:
            raise KeyRefused(f"overwrite not allowed for {key}")

    def put(self, key: str, data: bytes, overwrite: bool = False) -> str:
        """Returns 'uploaded', 'exists_identical'. Raises ObjectConflict if different content exists."""
        self.check_key(key, overwrite)
        digest = sha256_bytes(data)
        if not overwrite:
            have = self._head_sha(key)
            if have is not None:
                if have == digest:
                    return "exists_identical"
                raise ObjectConflict(f"{key} exists with different sha256; not overwritten")
        status = self._put(key, data, digest, overwrite)
        if status == "uploaded":
            self.uploaded_objects += 1
            self.uploaded_bytes += len(data)
        return status

    def put_json(self, key: str, obj: dict, overwrite: bool = False) -> str:
        return self.put(key, json.dumps(obj, indent=1, default=str, ensure_ascii=False).encode("utf-8"), overwrite)

    def get_json(self, key: str) -> dict:
        return json.loads(self.get(key).decode("utf-8"))

    # backend hooks
    def list(self, prefix: str) -> dict[str, int]: ...
    def get(self, key: str) -> bytes: ...
    def _head_sha(self, key: str) -> str | None: ...
    def _put(self, key: str, data: bytes, digest: str, overwrite: bool) -> str: ...

    def stats(self) -> dict:
        return {"uploaded_objects": self.uploaded_objects, "uploaded_bytes": self.uploaded_bytes, **self.ops}


class LocalStore(Store):
    """Directory-backed store with the same rules (tests, local dry runs)."""

    def __init__(self, cfg: dict, root: Path):
        super().__init__(cfg)
        self.root = Path(root)

    def _p(self, key: str) -> Path:
        return self.root / key

    def list(self, prefix: str) -> dict[str, int]:
        self.ops["class_a"] += 1
        base = self._p(prefix)
        folder = base if prefix.endswith("/") else base.parent
        if not folder.is_dir():
            return {}
        out = {}
        for p in folder.rglob("*"):
            if p.is_file() and not p.name.endswith(".part"):
                k = p.relative_to(self.root).as_posix()
                if k.startswith(prefix):
                    out[k] = p.stat().st_size
        return out

    def get(self, key: str) -> bytes:
        self.ops["class_b"] += 1
        return self._p(key).read_bytes()

    def _head_sha(self, key: str) -> str | None:
        self.ops["class_b"] += 1
        p = self._p(key)
        return sha256_bytes(p.read_bytes()) if p.is_file() else None

    def _put(self, key: str, data: bytes, digest: str, overwrite: bool) -> str:
        self.ops["class_a"] += 1
        p = self._p(key)
        if p.exists() and not overwrite:
            raise ObjectConflict(f"{key} appeared concurrently; not overwritten")
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".part")
        tmp.write_bytes(data)
        os.replace(tmp, p)
        return "uploaded"


class R2Store(Store):
    """Cloudflare R2 through the S3 API (client from src/lake/r2_sync.py)."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        from src.lake import r2_sync
        ok, missing = r2_sync.env_status()       # loads .env at runtime when present; never prints values
        if not ok:
            raise RuntimeError(f"missing environment variables: {missing}")
        self.bucket = os.environ[cfg["remote"]["bucket_env"]]
        self.s3 = r2_sync.client()

    def list(self, prefix: str) -> dict[str, int]:
        out, token = {}, None
        while True:
            kw = {"Bucket": self.bucket, "Prefix": prefix}
            if token:
                kw["ContinuationToken"] = token
            self.ops["class_a"] += 1
            r = self.s3.list_objects_v2(**kw)
            for o in r.get("Contents", []):
                out[o["Key"]] = o["Size"]
            if not r.get("IsTruncated"):
                return out
            token = r["NextContinuationToken"]

    def get(self, key: str) -> bytes:
        self.ops["class_b"] += 1
        return self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def _head_sha(self, key: str) -> str | None:
        from botocore.exceptions import ClientError
        self.ops["class_b"] += 1
        try:
            meta = self.s3.head_object(Bucket=self.bucket, Key=key).get("Metadata", {})
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return None
            raise
        return meta.get("sha256", "")

    def _put(self, key: str, data: bytes, digest: str, overwrite: bool) -> str:
        from botocore.exceptions import ClientError
        self.ops["class_a"] += 1
        kw = {"Bucket": self.bucket, "Key": key, "Body": data, "Metadata": {"sha256": digest}}
        if not overwrite:
            kw["IfNoneMatch"] = "*"           # conditional write: never replaces an object created meanwhile
        try:
            self.s3.put_object(**kw)
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("PreconditionFailed", "412"):
                raise ObjectConflict(f"{key} appeared concurrently; not overwritten") from e
            raise
        return "uploaded"


def open_store(spec: str, cfg: dict) -> Store:
    """``r2`` (default) or ``local:<dir>`` (relative paths are under the repository)."""
    if spec == "r2":
        return R2Store(cfg)
    if spec.startswith("local:"):
        p = Path(spec[len("local:"):])
        return LocalStore(cfg, p if p.is_absolute() else ROOT / p)
    raise ValueError(f"unknown store {spec!r}")


def existing_ids(store: Store, source: str, dates: Iterable[str]) -> set[str]:
    out = set()
    for d in sorted(set(dates)):
        out |= {id_from_manifest_key(k) for k in store.list(manifest_prefix(source, d)) if k.endswith(".json")}
    return out


def publish(store: Store, source: str, date: str, obj_id: str, ext: str, data: bytes, manifest: dict) -> dict:
    """Upload the data object, then its manifest (commit marker)."""
    dkey = data_key(source, date, obj_id, ext)
    s1 = store.put(dkey, data)
    manifest = {**manifest, "object_key": dkey, "object_bytes": len(data), "object_sha256": sha256_bytes(data)}
    s2 = store.put_json(manifest_key(source, date, obj_id), manifest)
    return {"id": obj_id, "data": s1, "manifest": s2, "bytes": len(data)}


# ---------------------------------------------------------------- reuse of P1 regional crop
@contextlib.contextmanager
def redirected(module, **attrs):
    """Temporarily point module globals (e.g. regional_crop.REGIONAL_ROOT) at the
    realtime working directory, so the P1/P2A writers are reused unchanged and
    never touch data/bronze_regional or data/_scratch (where the backfill runs)."""
    old = {k: getattr(module, k) for k in attrs}
    for k, v in attrs.items():
        setattr(module, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(module, k, v)
