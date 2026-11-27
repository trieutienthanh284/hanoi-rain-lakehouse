"""P4FG S2: run src.models.fetch_run under the 0.4 GB RSS gate (P4F.3).

  python scripts/p4fg/fetch_gated.py --job nwpconv_backfill --run-id 9900e8d4f8af --verify

A watcher thread samples RSS every 0.1 s; above the limit it prints a STOP line and exits the process with code 4
(fetch_run itself holds one object at a time). Prints the peak working set at the end.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import rss_gate as RG  # noqa: E402


def _watch(gate: RG.RssGate, stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            gate.check("fetch_run")
        except RG.RssExceeded:
            os._exit(RG.EXIT_CODE)
        time.sleep(0.1)


def main(argv=None) -> int:
    gate = RG.RssGate(RG.DEFAULT_LIMIT_GB)
    stop = threading.Event()
    t = threading.Thread(target=_watch, args=(gate, stop), daemon=True)
    t.start()
    from src.models import fetch_run as FR
    try:
        rc = FR.main(argv)
    finally:
        stop.set()
        t.join(1)
        print(f"rss_gate peak_rss_gb={gate.peak:.3f} peak_wset_gb={RG.peak_gb():.3f} limit_gb={gate.limit_gb}",
              flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
