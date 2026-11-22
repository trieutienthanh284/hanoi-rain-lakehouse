"""P4D.3 — Gold table_v2 units on SYNTHETIC Silver (no real data, no training): preflight, feat units (city + ring
labels from IMERG Final / Late Silver, v1 features + static columns), frame-mode G2, upload under the gold id, the g2
gate (NaN share city vs ring, pixel table, city rows == P4B table_A1_v1 reference)."""
import copy
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.gold import build_table_v2 as BT2
from src.gold import features as F
from src.gold import p4a_common as P
from src.models import cloud_train as CT
from src.models import p4b as PB
from src.models import p4d as PD

G = F.Grid()
PIX_ALL = BT2.domain_pixels()
BOX = np.array([(G.lat_idx_min + i) * 3600 + G.lon_idx_min + j for i in range(10, 20) for j in range(10, 18)])
CITY = np.sort(BOX[:47])


def _cfg():
    c = copy.deepcopy(PD.load_cfg(smoke=True))
    c["scope"]["day_ranges"] = [["2023-05-15", "2023-05-16"], ["2024-10-15", "2024-10-16"],
                                ["2025-07-01", "2025-07-01"], ["2025-10-06", "2025-10-06"]]
    return c


def _days(d0, n):
    d = date.fromisoformat(d0)
    return [d + timedelta(days=k) for k in range(n)]


def _put(store_root, rel, df):
    p = store_root / "lake/v1" / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)


def _field(rng, t):
    h = (t - pd.Timestamp("2023-05-13", tz="UTC")).total_seconds() / 3600
    gi, gj = np.meshgrid(np.arange(G.n_lat), np.arange(G.n_lon), indexing="ij")
    ci, cj = 5 + (1.5 * h) % 20, 4 + (2.0 * h) % 20
    return (12 * np.exp(-((gi - ci) ** 2 + (gj - cj) ** 2) / 8) + rng.gamma(0.3, 0.2, gi.shape)).astype("float32")


@pytest.fixture
def ws(tmp_path, monkeypatch):
    monkeypatch.setenv("HFL_P2B_EXCLUDE_SOURCES", "none")
    work, store = tmp_path / "ws", tmp_path / "store"
    monkeypatch.setattr(PB, "ROOT", work)
    monkeypatch.setattr(P, "SILVER", work / "data/silver/met")
    rng = np.random.default_rng(5)
    city = np.isin(PIX_ALL, CITY)
    for d in _days("2023-05-14", 3) + _days("2024-10-14", 3) + _days("2025-06-30", 2) + _days("2025-10-05", 2):
        rows = []
        for k in range(48):
            we = pd.Timestamp(d, tz="UTC") + pd.Timedelta(minutes=30 * k)
            f = _field(rng, we).reshape(-1)
            rows.append(pd.DataFrame({"pixel_id": PIX_ALL, "window_end_utc": we, "rain_rate_mm_h": f * 1.1,
                                      "rain_rate_gc_mm_h": f, "is_city_pixel": city}))
        _put(store, f"data/silver/met/gsmap_now/date={d}/part.parquet", pd.concat(rows, ignore_index=True))
    for src, days in (("imerg_final", _days("2023-05-15", 3) + _days("2024-10-15", 3) + _days("2025-07-01", 2)),
                      ("imerg_late", _days("2025-10-06", 2))):
        for d in days:
            t = pd.date_range(pd.Timestamp(d, tz="UTC"), periods=48, freq="30min")
            _put(store, f"data/silver/met/{src}/date={d}/part.parquet",
                 pd.DataFrame({"pixel_id": np.repeat(PIX_ALL, 48), "interval_start_utc": np.tile(t, len(PIX_ALL)),
                               "rain_rate_mm_h": rng.gamma(0.4, 3.0, 48 * len(PIX_ALL)),
                               "is_city_pixel": np.repeat(city, 48)}))
    cfg = _cfg()
    st = CT.open_run_store(f"local:{store}", "p4d_smoke")
    return PD.Ctx(cfg, st, "p4d_smoke", "t1", tmp_path / "mirror", print), store


def _feat_all(ctx):
    us = {u.id: u for u in PD.units(ctx.cfg)}
    return {k: PD.run_feat(us[k], ctx)[1] for k in sorted(us) if k.startswith("feat-")}


def _reference_a1(ctx, store):
    """A P4B-like table_A1_v1 partition for 2023-05: the v1 feature path on the city pixels only."""
    v2 = pd.read_parquet(ctx.gold_local() / "season=2023/month=05/part.parquet")
    ref = v2[v2["is_city"] == 1].drop(columns=list(BT2.STATIC))
    _put(store, f"gold/met_training/{PD.A1_REF['table']}/gid={PD.A1_REF['gold_id']}/season=2023/month=05/part.parquet",
         ref)


def test_preflight_and_budget(ws):
    ctx, _ = ws
    rels, m = PD.gate_preflight(ctx)
    rec = json.loads((PB.ROOT / rels[0]).read_text(encoding="utf-8"))
    assert m["ok"] and rec["required_margin_px"] == 10 and rec["ok_geometry"] and rec["days"] == 6
    assert rec["ring"]["min_margin_px"] == 7 and rec["ring"]["ring_candidates"] == 177 and m["ring_pixels"] == 4
    ctx.cfg["r2_budget"]["max_total_gb"] = 0.0
    with pytest.raises(RuntimeError, match="preflight FAIL"):
        PD.gate_preflight(ctx)


def test_feat_units_and_g2_gate(ws):
    ctx, store = ws
    res = _feat_all(ctx)
    assert list(res) == ["feat-2023-05", "feat-2024-10", "feat-2025-07", "feat-2025-10"]
    npx = 47 + 4
    assert res["feat-2023-05"]["rows"] == 2 * 48 * 3 * npx and res["feat-2023-05"]["rows_ring"] == 2 * 48 * 3 * 4
    assert res["feat-2025-10"]["label_source"] == "imerg_late"
    keys = ctx.store.list(PD.GOLD_REMOTE)
    assert len(keys) == 8 and all(f"/table_v2/gid={ctx.gid}/" in k for k in keys)
    a = pd.read_parquet(ctx.gold_local() / "season=2023/month=05/part.parquet")
    assert {"dist_to_city_center_km", "is_city", "ring_coverage", "cal_month"} <= set(a.columns)
    assert not any(c.startswith("cal_doy") for c in a.columns)
    assert set(a.loc[a["is_city"] == 1, "pixel_id"]) == set(CITY) and a.loc[a["is_city"] == 0, "pixel_id"].nunique() == 4
    st = F.pixel_static_columns(a["pixel_id"].to_numpy(), CITY, F.city_centre(CITY), ring=(0.3, 1.0))
    assert np.allclose(st["dist_to_city_center_km"], a["dist_to_city_center_km"]) and (st["is_city"] == a["is_city"]).all()
    assert np.allclose(st["ring_coverage"], a["ring_coverage"]) and (a.loc[a["is_city"] == 1, "ring_coverage"] == 1).all()
    meta = json.loads((ctx.gold_local() / "season=2025/month=10/part.meta.json").read_text(encoding="utf-8"))
    assert meta["label_source"] == "imerg_late" and meta["training_rows"] is False and meta["g2_violations"] == 0
    _reference_a1(ctx, store)
    rels, m = PD.gate_g2(ctx)
    assert m["violations"] == 0 and m["duplicates"] == 0 and m["a1_equal"] and m["a1_max_abs_diff"] == 0.0
    assert m["ring_pixels"] == 4 and m["city_pixels"] == 47
    nan = pd.read_csv(PB.ROOT / "reports/p4d/g2/nan_share_city_vs_ring.csv").set_index("feature")
    assert {"city", "ring", "ring_minus_city"} <= set(nan.columns) and "ring_coverage" in nan.index
    nb_ring = [c for c in nan.index if c.startswith(("nb_", "ring_"))]
    assert (nan.loc[nb_ring, "ring"] <= nan.loc[nb_ring, "city"] + 1e-9).all()   # truncated, never NaN
    px = pd.read_csv(PB.ROOT / "reports/p4d/g2/pixels.csv")
    assert len(px) == 51 and (px.loc[px["is_city"] == 0, "margin_px"] >= 7).all()
    tr = pd.read_csv(PB.ROOT / "reports/p4d/g2/truncation_city_vs_ring.csv")
    assert set(tr["pixels"]) == {"city", "ring"} and (tr.loc[tr["pixels"] == "city", "share_truncated"] == 0).all()
    assert m["ring_margin_px"] == 7 and 0 < m["ring_coverage_min"] <= 1
    eq = json.loads((PB.ROOT / "reports/p4d/g2/a1_equality.json").read_text(encoding="utf-8"))
    assert eq["rows_compared"] == 2 * 48 * 3 * 47 and set(eq["v2_only_columns"]) == set(BT2.STATIC)


def test_g2_fails_without_reference(ws):
    ctx, _ = ws
    _feat_all(ctx)
    with pytest.raises(RuntimeError, match="G2 gate FAIL"):
        PD.gate_g2(ctx)


def test_feat_refuses_leaky_feature(ws, monkeypatch):
    ctx, _ = ws
    u = {x.id: x for x in PD.units(ctx.cfg)}["feat-2023-05"]
    real = F.compute_features

    def leaky(*a, **k):
        df = real(*a, **k)
        df["gsmap__event_time"] = df["t0"] + pd.Timedelta(minutes=30)
        return df
    monkeypatch.setattr(F, "compute_features", leaky)
    with pytest.raises(RuntimeError, match="G2 violations"):
        PD.run_feat(u, ctx)
    assert not ctx.store.list(PD.GOLD_REMOTE)


def test_g2_stops_on_dry_inner_block(ws, monkeypatch):
    """A training block without any city row >= 2.5 mm/h (the first p4d_smoke failure) stops g2 before any fit."""
    ctx, store = ws
    _feat_all(ctx)
    _reference_a1(ctx, store)
    p = ctx.gold_local() / "season=2025/month=07/part.parquet"
    df = pd.read_parquet(p)
    for c in ("y_0p5", "y_2p5", "y_7p6"):
        if c in df:
            df[c] = np.int8(0)
    df.to_parquet(p, index=False)                    # local copy only: g2 re-reads it (fetch skips same size)
    monkeypatch.setattr(PD.Ctx, "fetch_gold", lambda self: 0)
    with pytest.raises(RuntimeError, match="inner CV has < 2 usable blocks"):
        PD.gate_g2(ctx)
    chk = pd.read_csv(PB.ROOT / "reports/p4d/g2/inner_cv_check.csv")
    assert not chk.loc[chk["heldout_season"].isin([2023, 2024]), "ok"].any()
    assert chk.loc[chk["heldout_season"] == 2025, "ok"].all()
