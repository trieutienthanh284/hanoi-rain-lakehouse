"""P4G.1 nwpconv_backfill on a local store with a fake HTTP client (no network): request shape, Bronze / Silver under the
approved prefixes only, put-if-absent, resume without a second request, one retry after a non-429 error, stop at 429,
and the issue / availability mapping (D214)."""
import gzip
import io
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.ingestion import openmeteo_conv_backfill as OB
from src.models import cloud_train as CT
from src.models import p4fg as FG

CFG = FG.load_cfg(smoke=False)


class Resp:
    def __init__(self, status, body):
        self.status_code, self.content, self.url = status, body, OB.API
        self.text = body.decode() if isinstance(body, bytes) else str(body)


def _pts(n=81):
    return pd.DataFrame({"pixel_id": np.arange(n, dtype="int64") + 4000000, "lat": 20.5 + 0.05 * np.arange(n),
                         "lon": 105.3 + 0.05 * np.arange(n)})


def _body(params):
    t0 = pd.Timestamp(params["start_date"], tz="UTC")
    t1 = pd.Timestamp(params["end_date"], tz="UTC") + pd.Timedelta(hours=23)
    times = pd.date_range(t0, t1, freq="h")
    n = len(params["latitude"].split(","))
    locs = [{"latitude": 21.0, "longitude": 105.8,
             "hourly": {"time": [int(t.timestamp()) for t in times],
                        "cape_previous_day1": [float(k % 7) * 100 for k in range(len(times))],
                        "lifted_index_previous_day1": [None if k == 3 else -1.0 for k in range(len(times))]}}
            for _ in range(n)]
    return json.dumps(locs).encode()


@pytest.fixture
def store(tmp_path):
    return CT.open_run_store(f"local:{tmp_path / 'store'}", "nwpconv_backfill")


def test_store_prefixes_are_exactly_the_approved_ones(store):
    assert set(store.allowed) == {"lake/v1/runs/nwpconv_backfill/", "lake/v1/bronze/openmeteo_conv/",
                                  "lake/v1/silver/nwp_conv/"}
    with pytest.raises(Exception):
        store.put("lake/v1/data/silver/met/openmeteo/x.json", b"{}")


def test_blocks_cover_the_period():
    b = OB.blocks(date(2024, 5, 1), date(2025, 10, 31), 13)
    assert b[0] == ("2024-05-01", "2024-05-13") and b[-1][1] == "2025-10-31"
    days = sum((date.fromisoformat(y) - date.fromisoformat(x)).days + 1 for x, y in b)
    assert days == (date(2025, 10, 31) - date(2024, 5, 1)).days + 1


def test_block_roundtrip_resume_and_mapping(store):
    calls = []

    def get(url, params, timeout, headers):
        calls.append(params)
        assert url == OB.API and "previous-runs-api" in url and "historical" not in url
        assert params["hourly"] == "cape_previous_day1,lifted_index_previous_day1" and params["models"] == "gfs_global"
        return Resp(200, _body(params))
    pts = _pts()
    state = {}
    r = OB.run_block(store, CFG, "2024-05-01", "2024-05-02", pts, get, state, sleep=lambda s: None)
    assert r["fetched"] and len(calls) == 1 and set(r["days"]) == {"2024-05-01", "2024-05-02"}
    assert all(d["dq_pass"] and d["points"] == 81 and d["hours"] == 24 for d in r["days"].values())
    keys = set(store.list("lake/v1/"))
    assert any(k.startswith("lake/v1/bronze/openmeteo_conv/block_start=2024-05-01/") for k in keys)
    sv = pd.read_parquet(io.BytesIO(store.get("lake/v1/silver/nwp_conv/date=2024-05-02/part.parquet")))
    T = sv["time_utc"]
    assert (sv["forecast_issue_time_utc"] == T.dt.floor("6h") - pd.Timedelta(hours=24)).all()
    assert (sv["available_time_utc"] - sv["forecast_issue_time_utc"] == pd.Timedelta(minutes=420)).all()
    assert sv["cape"].notna().all() and sv["lifted_index"].notna().all()
    s1 = pd.read_parquet(io.BytesIO(store.get("lake/v1/silver/nwp_conv/date=2024-05-01/part.parquet")))
    assert s1["lifted_index"].isna().sum() == 81                # the null at 03:00 UTC stays NaN (never filled)
    raw = json.loads(gzip.decompress(store.get("lake/v1/bronze/openmeteo_conv/block_start=2024-05-01/response.json.gz")))
    assert len(raw) == 81
    # resume: Bronze exists -> no request, identical Silver bytes accepted (put-if-absent)
    r2 = OB.run_block(store, CFG, "2024-05-01", "2024-05-02", pts, get, state, sleep=lambda s: None)
    assert not r2["fetched"] and len(calls) == 1


def test_politeness_retry_and_429(store):
    pts = _pts()
    slept = []
    seq = iter([Resp(500, b"oops"), None])

    def get(url, params, timeout, headers):
        x = next(seq)
        return Resp(200, _body(params)) if x is None else x
    state = {"last": __import__("time").time()}
    r = OB.run_block(store, CFG, "2024-06-01", "2024-06-01", pts, get, state, sleep=slept.append)
    assert r["fetched"] and slept and slept[0] > 100 and 60.0 in slept     # min interval wait, then one 60 s retry

    def bad(url, params, timeout, headers):
        return Resp(429, b"Daily API request limit exceeded")
    with pytest.raises(OB.RateLimited):
        OB.run_block(store, CFG, "2024-06-02", "2024-06-02", pts, bad, {}, sleep=lambda s: None)
    assert not store.list("lake/v1/bronze/openmeteo_conv/block_start=2024-06-02/")

    def twice(url, params, timeout, headers):
        return Resp(500, b"down")
    with pytest.raises(RuntimeError):
        OB.run_block(store, CFG, "2024-06-03", "2024-06-03", pts, twice, {}, sleep=lambda s: None)


def test_points_come_from_openmeteo_silver(tmp_path):
    st = CT.open_run_store(f"local:{tmp_path / 's'}", "nwpconv_backfill")
    p = tmp_path / "s" / f"lake/v1/data/silver/met/openmeteo/date={OB.POINTS_DAY}/part.parquet"
    p.parent.mkdir(parents=True)
    pts = _pts()
    pd.concat([pts.assign(model="gfs_global"), pts.assign(model="ecmwf_ifs025")]).to_parquet(p, index=False)
    got = OB.points(st)
    assert len(got) == 81 and got["pixel_id"].is_monotonic_increasing
