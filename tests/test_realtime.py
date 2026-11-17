"""Phase 3A realtime entry points — offline tests (no network, local object store).

Covers: idempotency, catch-up selection, manifest fields, staleness logic,
remote-key guard, compaction format, and that secret values never reach output.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.lake import grid  # noqa: E402
from src.lake import regional_crop as rc  # noqa: E402
from src.realtime import common as C  # noqa: E402
from src.realtime import compact_daily as cd  # noqa: E402
from src.realtime import freshness_monitor as fm  # noqa: E402
from src.realtime import gsmap_now_realtime as gr  # noqa: E402
from src.realtime import imerg_late_daily as il  # noqa: E402
from src.realtime import openmeteo_latest as om  # noqa: E402

FILL = np.float32(-9999.9)
DAY = "2025-09-30"
FAKE_SECRETS = {
    "EARTHDATA_TOKEN": "eyFAKEtokenVALUE0123456789abcdefXYZ",
    "GSMAP_FTP_HOST": "ftp.fake-host-for-tests.example",
    "GSMAP_FTP_USER": "fakeuser_rt_42",
    "GSMAP_FTP_PASSWORD": "Pa55w0rd!fake#secret",
    "R2_ACCOUNT_ID": "0123456789abcdeffedcba9876543210",
    "R2_ACCESS_KEY_ID": "AKFAKEACCESSKEYID000",
    "R2_SECRET_ACCESS_KEY": "fakeSecretAccessKey/abcdefghijklmnop+12345",
    "R2_BUCKET": "fake-bucket-name-xyz",
}


# ---------------------------------------------------------------- fixtures
def _field(t: int) -> np.ndarray:
    ii, jj = np.meshgrid(np.arange(grid.N_LAT), np.arange(grid.N_LON), indexing="ij")
    v = ((ii * 7 + jj * 13 + t * 101) % 5000).astype(np.float32) / 100
    v[(ii + jj + t) % 97 == 0] = FILL
    return v


def make_gsmap(folder: Path, t: int) -> Path:
    hh, mm = t // 2, (t % 2) * 30
    p = folder / f"gsmap_now_rain.20250930.{hh:02d}{mm:02d}.nc"
    with h5py.File(p, "w") as h:
        h["Latitude"] = grid.lat_centre(np.arange(grid.N_LAT)).astype(np.float32)
        h["Latitude"].attrs["units"] = np.bytes_("degrees_north")
        h["Longitude"] = grid.lon_centre(np.arange(grid.N_LON)).astype(np.float32)
        h["Longitude"].attrs["units"] = np.bytes_("degrees_east")
        h["Time"] = np.array([0], dtype=np.int32)
        h["Time"].attrs["units"] = np.bytes_(f"1hour since 2025-09-30 {hh:02d}:{mm:02d}:00.0")
        for n, ln in (("hourlyPrecipRate", "precip_now"), ("hourlyPrecipRateGC", "precip_gauge_now")):
            ds = h.create_dataset(n, data=_field(t)[None], compression="gzip")
            ds.attrs["_FillValue"] = np.array([FILL])
            ds.attrs["units"] = np.bytes_("mm/h")
            ds.attrs["long_name"] = np.bytes_(ln)
    return p


class FakeFTP:
    """Minimal ftplib.FTP stand-in serving local files as /now/latest."""

    def __init__(self, files: dict[str, Path], fail_with: str | None = None):
        self.files, self.fail_with = files, fail_with
        self.retrieved: list[str] = []

    def cwd(self, d):
        if self.fail_with:
            raise ConnectionError(self.fail_with)

    def nlst(self):
        extra = ["gsmap_now_flag.20250930.0000.nc", "gsmap_now.20250930.0000.dat.gz"]
        return sorted(self.files) + extra

    def voidcmd(self, c):
        return "200"

    def size(self, n):
        return self.files[n].stat().st_size

    def sendcmd(self, c):
        n = c.split()[1]
        start = datetime.strptime(n.split(".")[1] + n.split(".")[2], "%Y%m%d%H%M")
        return "213 " + (start + timedelta(hours=1, minutes=1)).strftime("%Y%m%d%H%M%S")

    def retrbinary(self, cmd, cb, blocksize=8192):
        n = cmd.split()[1]
        self.retrieved.append(n)
        cb(self.files[n].read_bytes())

    def quit(self):
        pass


@pytest.fixture()
def cfg():
    return C.load_config()


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("RUNNER_TEMP", str(tmp_path / "runner"))
    monkeypatch.setattr(gr, "RETRY_WAIT_S", 0)
    monkeypatch.setattr(om, "RETRY_WAIT_S", 0)
    return tmp_path


@pytest.fixture(scope="module")
def originals(tmp_path_factory):
    folder = tmp_path_factory.mktemp("latest")
    return {p.name: p for p in (make_gsmap(folder, t) for t in range(3))}


def store_of(cfg, tmp_path):
    return C.LocalStore(cfg, tmp_path / "r2")


# ---------------------------------------------------------------- catch-up selection
def test_catch_up_selects_all_missing_newest_first():
    t0 = datetime(2026, 11, 15, 5, 0, tzinfo=timezone.utc)
    wins = [{"name": f"n{k}", "start": t0 + timedelta(minutes=30 * k), "id": gr.window_id(t0 + timedelta(minutes=30 * k))}
            for k in range(48)]
    existing = {w["id"] for w in wins[10:40]}       # a 9 h outage at the start and the newest 4 h missing
    todo = gr.select_pending(wins, existing, limit=0)
    assert len(todo) == 18
    assert [w["start"] for w in todo] == sorted((w["start"] for w in todo), reverse=True)
    assert {w["id"] for w in todo}.isdisjoint(existing)
    assert len(gr.select_pending(wins, existing, limit=3)) == 3
    assert gr.select_pending(wins, {w["id"] for w in wins}, limit=0) == []


def test_rain_windows_ignores_flag_and_dat_files():
    from src.ingestion import gsmap_now_collector as gc
    names = ["gsmap_now_rain.20261116.0430.nc", "gsmap_now_flag.20261116.0430.nc",
             "gsmap_now.20261116.0430.dat.gz", "gsmap_gauge_now.20261116.0430_0530.dat.gz"]
    ws = gr.rain_windows(names, gc.load_config())
    assert [w["id"] for w in ws] == ["gsmap_now_20261116T0430"]
    assert ws[0]["end"] - ws[0]["start"] == timedelta(hours=1)


def test_imerg_late_selects_newest_complete_missing_days():
    cands = il.candidate_days(date(2026, 11, 16), 7)
    assert cands[0] == "2026-11-15" and cands[-1] == "2026-11-09"
    counts = {"2026-11-15": 30, "2026-11-14": 48, "2026-11-13": 48, "2026-11-12": 48}
    calls = []

    def counter(d):
        calls.append(d)
        return counts.get(d, 48)
    chosen, notes = il.select_days(cands, {il.day_id("2026-11-14")}, counter, 48, 2)
    assert chosen == ["2026-11-13", "2026-11-12"]
    assert notes["2026-11-15"].startswith("incomplete") and notes["2026-11-14"] == "exists"
    assert "2026-11-14" not in calls                 # no CMR query for days already in R2


# ---------------------------------------------------------------- GSMaP end to end (offline)
def test_gsmap_idempotent_manifest_and_crop(cfg, env, originals):
    store = store_of(cfg, env)
    ftp = FakeFTP(originals)
    r1 = gr.run(store, cfg, limit=None, ftp_factory=lambda: ftp)
    assert r1["listed"] == 3 and r1["published"] == 3 and not r1["failed"]
    assert store.uploaded_objects == 6

    store2 = store_of(cfg, env)
    ftp2 = FakeFTP(originals)
    r2 = gr.run(store2, cfg, limit=None, ftp_factory=lambda: ftp2)
    assert r2["published"] == 0 and r2["pending"] == 0 and r2["skipped_existing"] == 3
    assert store2.uploaded_objects == 0 and ftp2.retrieved == []

    keys = sorted(store.list("lake/v1/realtime/gsmap_now/manifest/"))
    assert len(keys) == 3 and all(k.startswith(f"lake/v1/realtime/gsmap_now/manifest/date={DAY}/") for k in keys)
    m = store.get_json(keys[0])
    required = {"source", "id", "original_name", "remote_dir", "original_size", "original_sha256",
                "remote_mdtm_utc", "window_start_utc", "window_end_utc", "ingestion_time_utc",
                "retrieval_started_utc", "available_time_utc", "crop_sha256", "validation",
                "object_key", "object_sha256", "object_bytes", "data_time_utc",
                "mdtm_minus_window_end_min", "ingestion_minus_window_end_min"}
    assert required <= set(m)
    assert m["validation"]["ok"] is True and len(m["original_sha256"]) == 64
    assert m["mdtm_minus_window_end_min"] == 1.0
    assert C.parse_iso(m["available_time_utc"]) >= C.parse_iso(m["window_end_utc"])
    assert C.parse_iso(m["ingestion_time_utc"]) >= C.parse_iso(m["retrieval_started_utc"])
    blob = store.get(m["object_key"])
    assert C.sha256_bytes(blob) == m["object_sha256"] == m["crop_sha256"]

    # crop equals the P1 crop of the original, bit for bit
    dom = rc.load_domain()
    w = rc.gsmap_window(originals[m["original_name"]])
    ref = rc.read_crop(w, dom)
    got = cd.read_window_object(blob)
    for n in cd.VARS:
        assert np.array_equal(got["vars"][n][0], ref["vars"][n][0], equal_nan=True)
        assert got["vars"][n][0].dtype == ref["vars"][n][0].dtype


def test_gsmap_limit_and_catch_up(cfg, env, originals):
    store = store_of(cfg, env)
    r1 = gr.run(store, cfg, limit=1, ftp_factory=lambda: FakeFTP(originals))
    assert r1["published"] == 1
    newest = max(originals)                         # names sort by time
    assert gr.window_id(gr.rain_windows([newest], __import__("src.ingestion.gsmap_now_collector",
                                                             fromlist=["x"]).load_config())[0]["start"]) \
        == r1["latency"][0]["id"]
    r2 = gr.run(store, cfg, limit=None, ftp_factory=lambda: FakeFTP(originals))
    assert r2["published"] == 2                     # the two older windows are caught up


def test_compaction_matches_backfill_format(cfg, env, originals, monkeypatch):
    store = store_of(cfg, env)
    gr.run(store, cfg, limit=None, ftp_factory=lambda: FakeFTP(originals))
    now = datetime(2025, 10, 1, 2, 0, tzinfo=timezone.utc)
    res = cd.run(store, cfg, now=now)
    assert res["days"][DAY] == "compacted 3 windows"
    res2 = cd.run(store, cfg, now=now)
    assert res2["days"][DAY] == "exists" and not res2["published"]

    key = C.data_key("compacted/gsmap_now", DAY, cd.day_id(DAY), "h5")
    out = env / "day.h5"
    out.write_bytes(store.get(key))
    # reference: the P1 backfill writer on the same originals
    monkeypatch.setattr(rc, "REGIONAL_ROOT", env / "ref_regional")
    dom = rc.load_domain()
    ws = rc.windows_for_date("gsmap_now", DAY, files=list(originals.values()))
    ref_path, _, _ = rc.write_day_file("gsmap_now", DAY, dom, ws, [rc.read_crop(w, dom) for w in ws])
    with h5py.File(out, "r") as a, h5py.File(ref_path, "r") as b:
        assert sorted(a.keys()) == sorted(b.keys())
        for k in a.keys():
            assert np.array_equal(a[k][...], b[k][...]), k
        for k in ("source", "axis_order", "n_windows", "domain_lat_idx_min", "domain_lon_idx_max"):
            assert a.attrs[k] == b.attrs[k]
    man = store.get_json(C.manifest_key("compacted/gsmap_now", DAY, cd.day_id(DAY)))
    assert man["n_windows"] == 3 and [l["time_index"] for l in man["lines"]] == [0, 1, 2]
    assert all("ingestion_time_utc" in l and "original_sha256" in l for l in man["lines"])


def test_due_days_waits_for_day_end():
    assert cd.due_days(datetime(2026, 11, 16, 0, 40, tzinfo=timezone.utc), 3, 1) == ["2026-11-14", "2026-11-13"]
    assert cd.due_days(datetime(2026, 11, 16, 1, 15, tzinfo=timezone.utc), 3, 1)[0] == "2026-11-15"


# ---------------------------------------------------------------- IMERG Late orchestration
def test_imerg_late_run_idempotent_and_provisional(cfg, env):
    store = store_of(cfg, env)
    lines = [{"window_start_utc": f"2026-11-14T{h:02d}:00:00Z", "window_end_utc": f"2026-11-14T{h:02d}:30:00Z"}
             for h in range(24)] * 2
    calls = []

    def proc(d):
        calls.append(d)
        return b"HDF5-day-bytes", lines, {"verification": None, "download_bytes": 1}
    r1 = il.run(store, cfg, today=date(2026, 11, 15), counter=lambda d: 48, processor=proc, max_days=1)
    assert [p["id"] for p in r1["published"]] == ["imerg_late_20261114"]
    m = store.get_json(C.manifest_key("imerg_late", "2026-11-14", "imerg_late_20261114"))
    assert m["label_status"] == "provisional" and m["n_windows"] == 48 and "ingestion_time_utc" in m
    r2 = il.run(store, cfg, today=date(2026, 11, 15), counter=lambda d: 30, processor=proc, max_days=1)
    assert not r2["published"] and calls == ["2026-11-14"]


def test_imerg_regional_env_works_outside_repository(tmp_path):
    """On runners $RUNNER_TEMP is outside the checkout: rc.rel must still work and
    the P2A writers must write only into the working directory."""
    work = tmp_path / "runner_temp"
    with il.regional_env(work):
        p = rc.crop_path("imerg_late", "2026-11-14")
        p.parent.mkdir(parents=True)
        p.write_bytes(b"x")
        assert rc.rel(p) == "regional/imerg_late/date=2026-11-14/imerg_late_20261114.h5"
        assert str(il.isc.SCRATCH).startswith(str(work))
    assert rc.REGIONAL_ROOT == ROOT / "data" / "bronze_regional" and rc.ROOT == ROOT


# ---------------------------------------------------------------- Open-Meteo orchestration
class FakeOM:
    def __init__(self, init_unix: dict[str, list[int]], fail_with: str | None = None):
        self.init, self.calls, self.forecasts, self.fail_with = init_unix, {}, 0, fail_with

    def __call__(self, url, params, ua, timeout=None):
        if self.fail_with:
            raise RuntimeError(self.fail_with)
        m = re.search(r"/data/([a-z0-9_]+)/static/meta.json", url)
        if m:
            seq = self.init[m[1]]
            k = self.calls.get(m[1], 0)
            self.calls[m[1]] = k + 1
            v = seq[min(k, len(seq) - 1)]
            return _Resp({"last_run_initialisation_time": v, "last_run_availability_time": v + 5 * 3600,
                          "last_run_modification_time": v + 5 * 3600, "update_interval_seconds": 21600})
        self.forecasts += 1
        models = params["models"].split(",")
        n = len(params["latitude"].split(","))
        t0 = 1794787200 - 3 * 3600
        times = [t0 + 3600 * k for k in range(1, 73)]
        out = []
        for i in range(n):
            h = {"time": times}
            for mo in models:
                key = f"precipitation_{mo}" if len(models) > 1 else "precipitation"
                h[key] = [0.1 * (k % 5) for k in range(72)]
            out.append({"latitude": 21.0, "longitude": 105.8, "hourly": h})
        return _Resp(out)


class _Resp:
    def __init__(self, obj):
        self.obj, self.content = obj, json.dumps(obj).encode()

    def json(self):
        return self.obj


def test_openmeteo_run_rows_and_idempotency(cfg, env):
    store = store_of(cfg, env)
    init = 1794787200                                   # 2026-11-16T00:00Z
    fake = FakeOM({"ncep_gfs013": [init], "ncep_gfs025": [init - 21600], "ecmwf_ifs025": [init - 21600]})
    r1 = om.run(store, cfg, getter=fake)
    assert fake.forecasts == 1 and len(r1["published"]) == 2
    assert r1["runs"]["gfs_global"]["id"] == "gfs_global_run20261116T0000"   # newest component (upper bound)
    assert r1["runs"]["ecmwf_ifs025"]["id"] == "ecmwf_ifs025_run20261115T1800"
    key = C.data_key("openmeteo", "2026-11-16", "gfs_global_run20261116T0000", "parquet")
    df = pd.read_parquet(__import__("io").BytesIO(store.get(key)))
    assert df["pixel_id"].nunique() == 81
    assert (df["interval_start_utc"] >= df["forecast_issue_time_utc"]).all()
    assert (df["lead_end_min"] >= 60).all()
    assert (df["available_time_utc"] == df["ingestion_time_utc"]).all()
    m = store.get_json(C.manifest_key("openmeteo", "2026-11-16", "gfs_global_run20261116T0000"))
    assert {"forecast_issue_time_utc", "ingestion_time_utc", "metadata", "raw_response_sha256",
            "lead_end_min_range", "request_url"} <= set(m)
    r2 = om.run(store, cfg, getter=fake)
    assert fake.forecasts == 1 and not r2["published"] and len(r2["skipped_existing"]) == 2


def test_openmeteo_skips_run_that_changes_during_request(cfg, env):
    store = store_of(cfg, env)
    init = 1794787200
    fake = FakeOM({"ncep_gfs013": [init, init + 21600], "ncep_gfs025": [init], "ecmwf_ifs025": [init]})
    r = om.run(store, cfg, getter=fake)
    assert r["skipped_changed"] == ["gfs_global"] and [p["id"] for p in r["published"]] == ["ecmwf_ifs025_run20261116T0000"]


# ---------------------------------------------------------------- Open-Meteo P3B hotfix (slow metadata)
class FaultyOM(FakeOM):
    """FakeOM with injectable failures and a simulated clock (no real waiting).
    meta_timeout: meta models whose metadata request times out; forecast_fail: forecast raises;
    rate_limit: 'meta' or 'forecast' -> HTTP 429. Forecast hours span [now - 24 h, now + 72 h]."""

    def __init__(self, init_unix, meta_timeout=(), forecast_fail=False, rate_limit=None, now=None):
        super().__init__(init_unix)
        self.meta_timeout, self.forecast_fail, self.rate_limit = set(meta_timeout), forecast_fail, rate_limit
        self.clock_s, self.timeouts, self.urls = 0.0, [], []
        self.base = int((now or datetime.now(timezone.utc)).timestamp()) // 3600 * 3600 - 24 * 3600

    def __call__(self, url, params, ua, timeout=None):
        self.urls.append(url)
        self.timeouts.append(timeout)
        m = re.search(r"/data/([a-z0-9_]+)/static/meta.json", url)
        if self.rate_limit == ("meta" if m else "forecast"):
            raise om.RateLimited("HTTP 429: Too many requests")
        if m and m[1] in self.meta_timeout:
            self.clock_s += timeout[1]                       # the read timeout elapses
            raise requests.exceptions.ReadTimeout(f"Read timed out. (read timeout={timeout[1]})")
        if not m and self.forecast_fail:
            self.clock_s += timeout[1]
            raise requests.exceptions.ConnectionError("forecast endpoint unreachable")
        if m:
            return super().__call__(url, params, ua, timeout)
        self.forecasts += 1
        models = params["models"].split(",")
        times = [self.base + 3600 * k for k in range(1, 97)]
        out = []
        for _ in range(len(params["latitude"].split(","))):
            h = {"time": times}
            for mo in models:
                h[f"precipitation_{mo}" if len(models) > 1 else "precipitation"] = [0.2] * len(times)
            out.append({"latitude": 21.0, "longitude": 105.8, "hourly": h})
        return _Resp(out)


@pytest.fixture()
def waits(monkeypatch):
    rec = []
    monkeypatch.setattr(om, "_sleep", rec.append)
    return rec


def _om_manifests(store):
    return {C.id_from_manifest_key(k).rsplit("_run", 1)[0]: store.get_json(k)
            for k in store.list("lake/v1/realtime/openmeteo/manifest/") if k.endswith(".json")}


def _recent_init():
    t = datetime.now(timezone.utc) - timedelta(hours=8)
    return int(t.replace(hour=t.hour - t.hour % 6, minute=0, second=0, microsecond=0).timestamp())


def test_rule_run_is_conservative():
    now = datetime(2026, 11, 17, 1, 20, tzinfo=timezone.utc)          # 08:20 Hanoi, run #16
    for delay in (360, 420, 530):
        r = om.rule_run(now, delay)
        assert r["issue"].hour % 6 == 0 and r["issue"].minute == 0
        assert r["availability"] == r["issue"] + timedelta(minutes=delay) <= now
        assert r["availability"] + timedelta(hours=6) > now                # newest run satisfying the rule
    assert om.rule_run(now, 420)["issue"] == datetime(2026, 11, 16, 18, tzinfo=timezone.utc)
    assert om.rule_run(now, 530)["issue"] == datetime(2026, 11, 16, 12, tzinfo=timezone.utc)
    assert set(om.load_rule_delays()) == {"gfs_global", "ecmwf_ifs025"}


def test_openmeteo_gfs013_metadata_timeout_falls_back(cfg, env, waits, capsys):
    init = _recent_init()
    fake = FaultyOM({"ncep_gfs025": [init], "ecmwf_ifs025": [init]}, meta_timeout={"ncep_gfs013"})
    code = om.main(["--store", f"local:{env / 'r2'}"], getter=fake)
    assert code == 0 and fake.forecasts == 1
    man = _om_manifests(C.LocalStore(cfg, env / "r2"))
    assert set(man) == {"gfs_global", "ecmwf_ifs025"}
    assert man["ecmwf_ifs025"]["availability_source"] == "metadata"
    assert man["ecmwf_ifs025"]["metadata_after_check"] == "same_run"
    g = man["gfs_global"]
    delay = om.load_rule_delays()["gfs_global"]
    issue, avail, ing = (C.parse_iso(g[k]) for k in ("forecast_issue_time_utc", "model_availability_time_utc",
                                                       "ingestion_time_utc"))
    assert g["availability_source"] == "rule_fallback" and g["rule_delay_min"] == delay
    assert avail == issue + timedelta(minutes=delay) <= ing                # never earlier than the rule
    assert "ReadTimeout" in g["metadata_error"] and g["rows"] > 0 and g["points"] == 81
    summary = [json.loads(l) for l in capsys.readouterr().out.splitlines() if '"summary"' in l][-1]
    assert summary["exit_code"] == 0
    assert summary["models"]["gfs_global"] == {"status": "published", "availability_source": "rule_fallback",
                                               "id": g["id"]}
    assert summary["models"]["ecmwf_ifs025"]["availability_source"] == "metadata"
    # ncep_gfs025 is not requested once ncep_gfs013 failed (bounded job time); waits: one 15 s retry wait
    assert waits == [om.META_RETRY_WAIT_S] and not any("ncep_gfs025" in u for u in fake.urls)


def test_openmeteo_all_metadata_timeout_both_fallback(cfg, env, waits):
    fake = FaultyOM({}, meta_timeout={"ncep_gfs013", "ncep_gfs025", "ecmwf_ifs025"})
    code = om.main(["--store", f"local:{env / 'r2'}"], getter=fake)
    assert code == 0 and fake.forecasts == 1
    store = C.LocalStore(cfg, env / "r2")
    man = _om_manifests(store)
    assert {m["availability_source"] for m in man.values()} == {"rule_fallback"} and len(man) == 2
    for m in man.values():
        df = pd.read_parquet(__import__("io").BytesIO(store.get(m["object_key"])))
        assert (df["availability_source"] == "rule_fallback").all()
        assert (df["interval_start_utc"] >= df["forecast_issue_time_utc"]).all()
    # freshness monitor accepts rule_fallback manifests as valid data
    st = fm.run(store, cfg, now=C.utcnow())
    om_st = st["sources"]["openmeteo"]
    assert om_st["stale"] is False and "openmeteo" not in st["stale_sources"]
    assert {s["availability_source"] for s in om_st["streams"].values()} == {"rule_fallback"}
    # next run: metadata is back and reports the same runs -> nothing new is requested
    runs = {m: int(C.parse_iso(v["forecast_issue_time_utc"]).timestamp()) for m, v in man.items()}
    ok = FaultyOM({"ncep_gfs013": [runs["gfs_global"]], "ncep_gfs025": [runs["gfs_global"]],
                   "ecmwf_ifs025": [runs["ecmwf_ifs025"]]})
    r = om.run(store, cfg, getter=ok)
    assert ok.forecasts == 0 and len(r["skipped_existing"]) == 2 and om.exit_code(r) == 0
    # metadata down again: a stored run at least as new as the rule run -> skipped, no duplicate
    again = FaultyOM({}, meta_timeout={"ncep_gfs013", "ecmwf_ifs025"})
    r = om.run(store, cfg, getter=again)
    assert again.forecasts == 0 and {v["status"] for v in r["models"].values()} == {"skipped_existing"}


def test_openmeteo_forecast_failure_for_both_exits_3(cfg, env, waits):
    init = _recent_init()
    fake = FaultyOM({"ncep_gfs013": [init], "ncep_gfs025": [init], "ecmwf_ifs025": [init]}, forecast_fail=True)
    code = om.main(["--store", f"local:{env / 'r2'}"], getter=fake)
    assert code == 3 and not _om_manifests(C.LocalStore(cfg, env / "r2"))
    # also with metadata failing for both models
    fake = FaultyOM({}, meta_timeout={"ncep_gfs013", "ecmwf_ifs025"}, forecast_fail=True)
    assert om.main(["--store", f"local:{env / 'r2'}"], getter=fake) == 3


@pytest.mark.parametrize("where", ["meta", "forecast"])
def test_openmeteo_429_exits_4_without_retry(cfg, env, waits, where):
    init = _recent_init()
    fake = FaultyOM({"ncep_gfs013": [init], "ncep_gfs025": [init], "ecmwf_ifs025": [init]}, rate_limit=where)
    code = om.main(["--store", f"local:{env / 'r2'}"], getter=fake)
    assert code == 4 and waits == []
    target = [u for u in fake.urls if ("meta.json" in u) == (where == "meta")]
    assert len(target) == 1                                    # the 429 request was not retried
    assert not _om_manifests(C.LocalStore(cfg, env / "r2"))


def test_openmeteo_one_failing_metadata_call_waits_at_most_80s(cfg, waits):
    fake = FaultyOM({}, meta_timeout={"ecmwf_ifs025"})
    with pytest.raises(requests.exceptions.ReadTimeout):
        om.model_meta(cfg["openmeteo"], "ecmwf_ifs025", getter=fake)
    assert fake.timeouts == [(10, 30), (10, 30)] and waits == [15]
    assert fake.clock_s + sum(waits) == 75 <= 80
    assert om.FORECAST_TIMEOUT[1] <= 60


# ---------------------------------------------------------------- staleness
def _rec(dt):
    return {"data_time_utc": C.iso(dt), "ingestion_time_utc": C.iso(dt), "key": "k"}


def test_staleness_logic(cfg):
    now = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)
    fresh = {"gsmap_now": {"gsmap_now": _rec(now - timedelta(minutes=50))},
             "openmeteo": {"gfs_global": _rec(now - timedelta(hours=8)),
                           "ecmwf_ifs025": _rec(now - timedelta(hours=11))},
             "imerg_late": {"imerg_late": _rec(now - timedelta(hours=36))}}
    s = fm.evaluate(now, fresh, cfg)
    assert s["overall"] == "OK" and s["stale_sources"] == []
    assert s["sources"]["openmeteo"]["age_min"] == 660.0

    stale = json.loads(json.dumps(fresh))
    stale["gsmap_now"]["gsmap_now"] = _rec(now - timedelta(minutes=121))
    stale["openmeteo"]["ecmwf_ifs025"] = _rec(now - timedelta(hours=13))
    del stale["imerg_late"]
    s = fm.evaluate(now, stale, cfg)
    assert s["overall"] == "STALE" and sorted(s["stale_sources"]) == ["gsmap_now", "imerg_late", "openmeteo"]
    assert s["sources"]["openmeteo"]["streams"]["gfs_global"]["stale"] is False
    # boundary: exactly at the threshold is not stale
    edge = {"gsmap_now": {"gsmap_now": _rec(now - timedelta(minutes=120))}}
    assert fm.evaluate(now, edge, cfg)["sources"]["gsmap_now"]["stale"] is False


def test_freshness_monitor_end_to_end_exit_codes(cfg, env, originals):
    store = store_of(cfg, env)
    gr.run(store, cfg, limit=None, ftp_factory=lambda: FakeFTP(originals))
    now = datetime(2025, 9, 30, 2, 30, tzinfo=timezone.utc)    # newest window 01:00-02:00 ends 30 min ago
    st = fm.run(store, cfg, now=now)
    assert st["sources"]["gsmap_now"]["stale"] is False
    assert st["sources"]["gsmap_now"]["streams"]["gsmap_now"]["key"].endswith("gsmap_now_20250930T0100.json")
    assert set(st["stale_sources"]) == {"openmeteo", "imerg_late"}
    snap = store.get_json(fm.STATUS_KEY)
    assert snap["overall"] == "STALE"
    fm.run(store, cfg, now=now + timedelta(minutes=10))          # status object may be overwritten
    code = fm.main(["--store", f"local:{env / 'r2'}"])           # real clock: everything is old -> stale
    assert code == 1


# ---------------------------------------------------------------- remote-key guard
def test_store_refuses_keys_outside_realtime_and_status(cfg, tmp_path):
    s = store_of(cfg, tmp_path)
    for bad in ("lake/v1/data/bronze_regional/x.h5", "lake/v2/realtime/x", "other/x",
                "lake/v1/realtime/../silver/x", "lake/v1/realtime//x"):
        with pytest.raises(C.KeyRefused):
            s.put(bad, b"x")
    s.put("lake/v1/realtime/gsmap_now/date=2026-01-01/a.h5", b"one")
    assert s.put("lake/v1/realtime/gsmap_now/date=2026-01-01/a.h5", b"one") == "exists_identical"
    with pytest.raises(C.ObjectConflict):
        s.put("lake/v1/realtime/gsmap_now/date=2026-01-01/a.h5", b"two")
    with pytest.raises(C.KeyRefused):
        s.put("lake/v1/realtime/gsmap_now/date=2026-01-01/a.h5", b"two", overwrite=True)
    s.put(fm.STATUS_KEY, b"{}", overwrite=True)
    s.put(fm.STATUS_KEY, b"{\"x\":1}", overwrite=True)
    assert s.get(fm.STATUS_KEY) == b"{\"x\":1}"


# ---------------------------------------------------------------- secrets never in output
@pytest.fixture()
def fake_secrets(monkeypatch):
    for k, v in FAKE_SECRETS.items():
        monkeypatch.setenv(k, v)
    return FAKE_SECRETS


def _assert_clean(text: str):
    for name, v in FAKE_SECRETS.items():
        assert v not in text, f"{name} leaked"


def test_no_secret_in_logs_on_ftp_failure(cfg, env, fake_secrets, capsys):
    msg = (f"530 Login incorrect for {fake_secrets['GSMAP_FTP_USER']}:{fake_secrets['GSMAP_FTP_PASSWORD']}"
           f"@{fake_secrets['GSMAP_FTP_HOST']}")
    store = store_of(cfg, env)
    code = C.cli(lambda argv: gr.run(store, cfg, ftp_factory=lambda: FakeFTP({}, fail_with=msg)) and 0)
    out = capsys.readouterr()
    assert code == 1 and "SourceStopped" in out.err and C.MASK in out.err + out.out
    _assert_clean(out.out + out.err)


def test_no_secret_in_logs_on_http_and_r2_failure(cfg, env, fake_secrets, capsys):
    store = store_of(cfg, env)
    url_err = (f"401 for https://example.invalid/?token={fake_secrets['EARTHDATA_TOKEN']} "
               f"endpoint https://{fake_secrets['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com/"
               f"{fake_secrets['R2_BUCKET']} key={fake_secrets['R2_ACCESS_KEY_ID']} "
               f"secret={fake_secrets['R2_SECRET_ACCESS_KEY']}")
    code = C.cli(lambda argv: om.run(store, cfg, getter=FakeOM({}, fail_with=url_err)) and 0)
    C.cli(lambda argv: C.log("event", detail=url_err) or 0)
    with C.redacted_output():
        print(url_err)
        print(url_err, file=sys.stderr)
    out = capsys.readouterr()
    assert code == 1
    _assert_clean(out.out + out.err)
    assert out.out.count(C.MASK) >= 5


def test_secret_loaded_after_start_is_still_redacted(monkeypatch, capsys):
    with C.redacted_output():
        monkeypatch.setenv("GSMAP_FTP_PASSWORD", "late-loaded-secret-123")
        print("pw=late-loaded-secret-123")
    assert "late-loaded-secret-123" not in capsys.readouterr().out


def test_manifests_contain_no_secret(cfg, env, originals, fake_secrets):
    store = store_of(cfg, env)
    gr.run(store, cfg, limit=None, ftp_factory=lambda: FakeFTP(originals))
    for k in store.list("lake/v1/"):
        if k.endswith(".json"):
            _assert_clean(store.get(k).decode("utf-8"))


def test_workflows_never_echo_secrets():
    wf = ROOT / ".github" / "workflows"
    files = sorted(wf.glob("*.yml"))
    assert files, "no workflow files"
    for p in files:
        txt = p.read_text(encoding="utf-8")
        for line in txt.splitlines():
            s = line.strip()
            if "secrets." in s:
                assert not re.search(r"\b(echo|printf|cat|env\b|printenv|set -x)", s), f"{p.name}: {s}"
        assert "set -x" not in txt and "printenv" not in txt, p.name
        assert "permissions:" in txt and "contents: read" in txt, p.name
