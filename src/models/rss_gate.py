"""RSS gate for local post-processing scripts (P4F.3, fixes D200).

Every local script of TASK_P4F_P4G calls `check()` at its main loop points (or wraps work in `RssGate`). Above the
limit (default 0.4 GB resident set) the process prints a STOP line and exits with code 4. Cloud jobs do not use it
(runners have 16 GB); bootstraps and row-level work run on the cloud.
"""
from __future__ import annotations

import sys

EXIT_CODE = 4
DEFAULT_LIMIT_GB = 0.4


def rss_gb() -> float:
    """Current resident set size in GB (psutil; peak working set on Windows when available)."""
    import psutil
    mi = psutil.Process().memory_info()
    return float(getattr(mi, "rss", 0)) / 2**30


def peak_gb() -> float:
    import psutil
    mi = psutil.Process().memory_info()
    return float(getattr(mi, "peak_wset", getattr(mi, "rss", 0))) / 2**30


class RssExceeded(SystemExit):
    pass


class RssGate:
    """`gate = RssGate(0.4); gate.check("step")` → SystemExit(4) above the limit; `gate.peak` = max seen."""

    def __init__(self, limit_gb: float = DEFAULT_LIMIT_GB, reader=rss_gb):
        self.limit_gb = float(limit_gb)
        self.reader = reader
        self.peak = 0.0

    def check(self, where: str = "") -> float:
        v = float(self.reader())
        self.peak = max(self.peak, v)
        if v > self.limit_gb:
            print(f"STOP rss_gate: RSS {v:.3f} GB > {self.limit_gb:.3f} GB at {where or '?'} (exit {EXIT_CODE})",
                  file=sys.stderr, flush=True)
            raise RssExceeded(EXIT_CODE)
        return v

    def __enter__(self):
        self.check("enter")
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.check("exit")
        return False


def check(where: str = "", limit_gb: float = DEFAULT_LIMIT_GB) -> float:
    return RssGate(limit_gb).check(where)
