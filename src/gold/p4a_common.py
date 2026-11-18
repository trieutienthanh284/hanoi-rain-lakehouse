"""Shared helpers for Phase 4A (pilot v0): config, hashes, done markers, logging, guards."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CFG_PATH = ROOT / "configs" / "model_pilot.yaml"
MARKERS = ROOT / "reports" / "p4a" / "markers"
GOLD = ROOT / "data" / "gold" / "met_training" / "pilot_v0"
SILVER = ROOT / "data" / "silver" / "met"
LOGS = ROOT / "logs" / "p4a"


def load_cfg(path: Path = CFG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cfg_hash(cfg: dict | None = None, extra: dict | None = None) -> str:
    cfg = cfg if cfg is not None else load_cfg()
    s = json.dumps({"cfg": cfg, "extra": extra or {}}, sort_keys=True, default=str)
    return sha256_bytes(s.encode())[:16]


def code_hash(*files: Path) -> str:
    h = hashlib.sha256()
    for f in files:
        h.update(Path(f).read_bytes())
    return h.hexdigest()[:16]


def excluded_sources() -> set[str]:
    return {s.strip() for s in os.environ.get("HFL_P2B_EXCLUDE_SOURCES", "").split(",") if s.strip()}


def guard_source(source: str) -> None:
    """Refuse to touch an excluded source (imerg_late while its backfill runs)."""
    if source in excluded_sources():
        raise PermissionError(f"source {source} is excluded by HFL_P2B_EXCLUDE_SOURCES")


def require_guard() -> None:
    if "imerg_late" not in excluded_sources():
        raise SystemExit("set HFL_P2B_EXCLUDE_SOURCES=imerg_late (imerg_late backfill still running)")


def rel(p: Path) -> str:
    try:
        return Path(p).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def write_atomic_bytes(p: Path, data: bytes) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, p)


def write_json(p: Path, obj) -> None:
    write_atomic_bytes(p, (json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n").encode("utf-8"))


def write_parquet(df, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, p)


def write_marker(step: str, outputs: list[Path], metrics: dict, inputs: dict | None = None,
                 cfg: dict | None = None) -> Path:
    m = {"step": step, "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         "config_hash": cfg_hash(cfg), "inputs": inputs or {},
         "outputs": {rel(o): sha256_file(o) for o in outputs if Path(o).is_file()},
         "metrics": metrics}
    p = MARKERS / f"{step}.done.json"
    write_json(p, m)
    return p


def read_marker(step: str) -> dict | None:
    p = MARKERS / f"{step}.done.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def get_logger(name: str) -> logging.Logger:
    LOGS.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger(name)
    if not lg.handlers:
        lg.setLevel(logging.INFO)
        fh = logging.FileHandler(LOGS / f"{name}.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        lg.addHandler(fh)
    return lg


def rss_gb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**30
    except ImportError:
        return float("nan")


def check_rss(limit_gb: float = 5.0) -> None:
    r = rss_gb()
    if r == r and r > limit_gb:
        print(f"STOP: RSS {r:.2f} GB > {limit_gb} GB", file=sys.stderr)
        raise SystemExit(3)
