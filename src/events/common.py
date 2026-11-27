"""Shared helpers for Phase U1 (event catalog): paths, markers, RAM guard.

Agent code. Imports the existing flood-evidence modules read-only; never edits them.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src" / "ingestion", ROOT / "src" / "processing"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

INBOX_DIR = ROOT / "data" / "_inbox" / "evidence"
INBOX_CSV = INBOX_DIR / "inbox.csv"
INBOX_AGENT_CSV = INBOX_DIR / "inbox_agent.csv"
BRONZE_FE = ROOT / "data" / "bronze" / "flood_evidence"
SILVER_FE = ROOT / "data" / "silver" / "flood_evidence" / "records"
CATALOG_DIR = ROOT / "data" / "gold" / "event_catalog"
REPORTS = ROOT / "reports" / "u1"
MARKERS = REPORTS / "markers"
SCRATCH = ROOT / "data" / "_scratch" / "u1"
LOGS = ROOT / "logs" / "u1"
PENDING_MARK = "[agent – chờ duyệt]"
LOCAL_TZ = "Asia/Ho_Chi_Minh"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rel(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def write_marker(step: str, outputs: list[Path], metrics: dict[str, Any],
                 inputs: list[Path] | None = None, config: dict[str, Any] | None = None) -> Path:
    """Done marker: config hash, input fingerprints, outputs with sha256, key metrics."""
    marker = {
        "step": step,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha256_text(json.dumps(config or {}, sort_keys=True, default=str)),
        "inputs": {rel(p): sha256_file(p) for p in (inputs or []) if Path(p).is_file()},
        "outputs": {rel(p): sha256_file(p) for p in outputs if Path(p).is_file()},
        "metrics": metrics,
    }
    path = MARKERS / f"{step}.done.json"
    write_json_atomic(path, marker)
    return path


def free_ram_gb() -> float:
    import psutil
    return psutil.virtual_memory().available / 2**30


def rss_gb() -> float:
    import psutil
    return psutil.Process().memory_info().rss / 2**30


class RamGuard:
    """Track peak RSS; report if free RAM stays < floor for `window_s` seconds."""

    def __init__(self, floor_gb: float = 0.5, window_s: float = 300.0, rss_cap_gb: float = 0.4):
        self.floor_gb, self.window_s, self.rss_cap_gb = floor_gb, window_s, rss_cap_gb
        self.low_since: float | None = None
        self.peak_rss = 0.0
        self.min_free = 99.0

    def check(self) -> str:
        """Return 'ok', 'low' (below floor, still waiting) or 'abort' (low for window_s)."""
        r, fr = rss_gb(), free_ram_gb()
        self.peak_rss = max(self.peak_rss, r)
        self.min_free = min(self.min_free, fr)
        if fr >= self.floor_gb:
            self.low_since = None
            return "ok"
        now = time.monotonic()
        if self.low_since is None:
            self.low_since = now
        return "abort" if now - self.low_since >= self.window_s else "low"

    def wait_ok(self, poll_s: float = 15.0) -> bool:
        """Block until free RAM >= floor; False if it stayed low for the whole window."""
        while True:
            st = self.check()
            if st == "ok":
                return True
            if st == "abort":
                return False
            time.sleep(poll_s)


def log_line(name: str, msg: str) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(LOGS / f"{name}.log", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()} {msg}\n")
