"""P4F.3: the RSS gate of local scripts aborts above the limit with exit code 4 (fixes D200)."""
import subprocess
import sys

import pytest

from src.models import rss_gate as RG


def test_below_limit_passes_and_records_peak():
    g = RG.RssGate(0.4, reader=lambda: 0.1)
    assert g.check("a") == 0.1 and g.peak == 0.1
    with RG.RssGate(0.4, reader=lambda: 0.2) as h:
        assert h.peak == 0.2


def test_above_limit_exits_4():
    vals = iter([0.1, 0.5])
    g = RG.RssGate(0.4, reader=lambda: next(vals))
    g.check("start")
    with pytest.raises(SystemExit) as e:
        g.check("loop")
    assert e.value.code == 4 and g.peak == 0.5


def test_real_process_abort_path():
    code = ("from src.models import rss_gate as RG\n"
            "x = bytearray(80 * 2**20)\n"                     # ~80 MB resident
            "RG.RssGate(0.01).check('alloc')\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert r.returncode == 4 and "STOP rss_gate" in r.stderr


def test_real_rss_is_measured():
    assert 0 < RG.rss_gb() < 4 and RG.peak_gb() >= RG.rss_gb() * 0.5
