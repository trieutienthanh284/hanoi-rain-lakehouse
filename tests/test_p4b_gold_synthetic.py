"""P4B — the Gold v1 build units of the p4b job on SYNTHETIC Silver data (no real data, no training, D94):
feat units (Final labels from the pilot partition, Late labels built in the unit, set-B NWP join), frame-mode G2,
upload under the gold id, and the g2 gate including the v1 == pilot check on the common columns."""
import copy
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.gold import build_labels as BL
from src.gold import features as F
from src.gold import p4a_common as P
from src.models import cloud_train as CT
from src.models import p4b as PB

G = F.Grid()
PIX_ALL = np.array([(G.lat_idx_min + i) * 3600 + G.lon_idx_min + j for i in range(G.n_lat) for j in range(G.n_lon)])
CITY = np.sort(PIX_ALL[[15 * 28 + 14, 16 * 28 + 13, 14 * 28 + 15]])


def _cfg():
    c = copy.deepcopy(PB.load_cfg(smoke=True))
    c["scope"] |= {"cv_periods": [["2024-07-01", "2024-07-01"], ["2024-07-02", "2024-07-02"]],
                   "test": ["2024-08-01", "2024-08-01"], "secondary": ["2024-09-01", "2024-09-01"],
                   "label_source_by_month": {"2024-09": "imerg_late"}}
    return c


def _days(d0: str, n: int):
    d = date.fromisoformat(d0)
    return [d + timedelta(days=k) for k in range(n)]


def _put(store_root, rel, df):
    p = store_root / "lake/v1" / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)


def _field(rng, t):
    """Moving blob plus light noise: a reproducible synthetic GC field for window end t."""
    h = (t - pd.Timestamp("2024-06-29", tz="UTC")).total_seconds() / 3600
    gi, gj = np.meshgrid(np.arange(G.n_lat), np.arange(G.n_lon), indexing="ij")
    ci, cj = 5 + (1.5 * h) % 20, 4 + (2.0 * h) % 20
    return (12 * np.exp(-((gi - ci) ** 2 + (gj - cj) ** 2) / 8) + rng.gamma(0.3, 0.2, gi.shape)).astype("float32")


@pytest.fixture
def ws(tmp_path, monkeypatch):
    monkeypatch.setenv("HFL_P2B_EXCLUDE_SOURCES", "none")
    work, store = tmp_path / "ws", tmp_path / "store"
    monkeypatch.setattr(PB, "ROOT", work)
    monkeypatch.setattr(P, "SILVER", work / "data/silver/met")
    rng = np.random.default_rng(3)
    city = np.isin(PIX_ALL, CITY)
    for d in _days("2024-06-30", 3) + _days("2024-07-31", 2) + _days("2024-08-31", 2):
        rows = []
        for k in range(48):
            we = pd.Timestamp(d, tz="UTC") + pd.Timedelta(minutes=30 * k)
            f = _field(rng, we).reshape(-1)
            rows.append(pd.DataFrame({"pixel_id": PIX_ALL, "window_end_utc": we, "rain_rate_mm_h": f * 1.1,
                                      "rain_rate_gc_mm_h": f, "is_city_pixel": city}))
        _put(store, f"data/silver/met/gsmap_now/date={d}/part.parquet", pd.concat(rows, ignore_index=True))
    for src, days in (("imerg_final", _days("2024-07-01", 3) + _days("2024-08-01", 2)),
                      ("imerg_late", _days("2024-09-01", 2))):
        for d in days:
            t = pd.date_range(pd.Timestamp(d, tz="UTC"), periods=48, freq="30min")
            df = pd.DataFrame({"pixel_id": np.repeat(CITY, 48), "interval_start_utc": np.tile(t, len(CITY)),
                               "rain_rate_mm_h": rng.gamma(0.4, 3.0, 48 * len(CITY)), "is_city_pixel": True})
            _put(store, f"data/silver/met/{src}/date={d}/part.parquet", df)
    lat, lon = G.latlon(CITY)
    for d in _days("2024-06-30", 4) + _days("2024-07-31", 3) + _days("2024-08-31", 3):
        h = pd.date_range(pd.Timestamp(d, tz="UTC"), periods=24, freq="1h")
        parts = []
        for model in ("gfs_global", "ecmwf_ifs025"):
            iss = h.floor("6h") - pd.Timedelta("24h")
            parts.append(pd.DataFrame({"pixel_id": np.repeat(CITY, 24), "model": model,
                                       "interval_start_utc": np.tile(h, len(CITY)),
                                       "precip_mm": rng.gamma(0.5, 1.0, 24 * len(CITY)),
                                       "forecast_issue_time_utc": np.tile(iss, len(CITY)),
                                       "available_time_utc": np.tile(iss + pd.Timedelta("8h"), len(CITY)),
                                       "lat": np.repeat(lat, 24), "lon": np.repeat(lon, 24)}))
        _put(store, f"data/silver/met/openmeteo/date={d}/part.parquet", pd.concat(parts, ignore_index=True))
    # pilot labels (Final) built by the unchanged pilot function on the synthetic Final Silver
    cfg = _cfg()
    for (y, m), days in (((2024, 7), _days("2024-07-01", 2)), ((2024, 8), _days("2024-08-01", 1))):
        for d in days + [days[-1] + timedelta(days=1)]:
            src = store / "lake/v1" / f"data/silver/met/imerg_final/date={d}/part.parquet"
            dst = work / f"data/silver/met/imerg_final/date={d}/part.parquet"
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(src.read_bytes())
        lab, _ = BL.build_month(y, m, cfg, CITY, days=days)
        _put(store, f"data/gold/met_training/pilot_v0/labels/season={y}/month={m:02d}/part.parquet", lab)
    # pilot table_A for July with the v0 feature function (for the v1 == pilot check)
    lab = pd.read_parquet(store / "lake/v1/data/gold/met_training/pilot_v0/labels/season=2024/month=07/part.parquet")
    lab["t0"] = lab["t0"].astype("datetime64[ns, UTC]")
    import yaml
    p0 = F.feature_params(yaml.safe_load((PB.CODE_ROOT / "configs/model_pilot.yaml").read_text(encoding="utf-8")))
    wins = {}
    for d in _days("2024-06-30", 3):
        g = pd.read_parquet(store / "lake/v1" / f"data/silver/met/gsmap_now/date={d}/part.parquet")
        for we, x in g.groupby("window_end_utc"):
            a = np.full((G.n_lat, G.n_lon), np.nan, "float32")
            i, j = G.ij(x["pixel_id"].to_numpy())
            a[i, j] = x["rain_rate_gc_mm_h"].to_numpy(); r = a.copy(); r[i, j] = x["rain_rate_mm_h"].to_numpy()
            wins[pd.Timestamp(we)] = (a, r)
    t0s = pd.DatetimeIndex(np.sort(lab["t0"].unique())).tz_convert("UTC").as_unit("ns")
    fe = F.compute_features(*F.stack_windows(wins, t0s, p0, G.n_lat, G.n_lon), t0s, CITY, p0, G)
    _put(store, "data/gold/met_training/pilot_v0/table_A/season=2024/month=07/part.parquet",
         lab.merge(fe, on=["pixel_id", "t0"]))
    st = CT.open_run_store(f"local:{store}", "p4b_smoke")
    return PB.Ctx(cfg, st, "p4b_smoke", "t1", tmp_path / "mirror", print), store


def test_feat_units_and_g2_gate(ws):
    ctx, store = ws
    us = {u.id: u for u in PB.units(ctx.cfg)}
    assert [u for u in us if u.startswith("feat-")] == ["feat-2024-07", "feat-2024-08", "feat-2024-09"]
    res = {k: PB.run_feat(us[k], ctx)[1] for k in ("feat-2024-07", "feat-2024-08", "feat-2024-09")}
    assert res["feat-2024-07"]["rows_A1"] == 2 * 48 * 3 * len(CITY) and res["feat-2024-07"]["rows_B1"] > 0
    assert res["feat-2024-09"]["label_source"] == "imerg_late"
    keys = st_keys = ctx.store.list(PB.GOLD_REMOTE)
    assert all(f"/gid={ctx.gid}/" in k for k in st_keys) and len(keys) == 12      # 3 months x (A1, B1) x 2 files
    a = pd.read_parquet(ctx.gold_local("A1") / "season=2024/month=09/part.parquet")
    assert set(a["label_source"]) == {"imerg_late_v07"} and not any(c.startswith("cal_doy") for c in a.columns)
    assert a["motion_valid"].notna().all() and a["motion_valid"].mean() > 0.5     # the blob moves: mostly valid
    rels, m = PB.gate_g2(ctx)
    assert m["violations"] == 0 and m["duplicates"] == 0 and m["pilot_equal"] and m["pilot_max_abs_diff"] == 0.0
    eq = json.loads((PB.ROOT / "reports/p4b/g2/pilot_equality.json").read_text(encoding="utf-8"))
    assert eq["pilot_only_columns"] == ["cal_doy_sin", "cal_doy_cos"] and eq["rows_compared"] > 0
    # feature families used by the CV units
    tA, b = PB.feature_set(ctx, "b")
    _, d = PB.feature_set(ctx, "d")
    tB, bd = PB.feature_set(ctx, "B:dm")
    assert tA == "A1" and tB == "B1" and "cal_month" not in d and "cal_month" in bd
    assert not any(c.startswith(PB.V1_NEW) for c in b) and any(c.startswith("extent_") for c in d)
    assert sum(c.startswith("nwp_") for c in bd) == 16


def test_g2_gate_fails_on_violation(ws, monkeypatch):
    ctx, _ = ws
    u = {x.id: x for x in PB.units(ctx.cfg)}["feat-2024-08"]
    real = F.compute_features

    def leaky(*a, **k):                              # a feature that peeks 30 min into the future
        df = real(*a, **k)
        df["gsmap__event_time"] = df["t0"] + pd.Timedelta(minutes=30)
        return df
    monkeypatch.setattr(F, "compute_features", leaky)
    with pytest.raises(RuntimeError, match="G2 violations"):
        PB.run_feat(u, ctx)
    assert not ctx.store.list(PB.GOLD_REMOTE)        # nothing uploaded
