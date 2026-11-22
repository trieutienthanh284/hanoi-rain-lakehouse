"""P4D.3 — Gold table_v2: city pixels + buffer ring, 2023-05 .. 2025-10 (one partition per month).

Pure building blocks used by the cloud units of src/models/p4d.py (the only caller):
  * pixel sets: the 47 city pixels (is_city_pixel of IMERG Silver, D05) and the buffer ring = every domain pixel whose
    margin to the domain edge is >= pixels.ring.min_margin_px, minus the city, where min_margin_px is the LARGEST
    margin giving >= pixels.ring.min_ring_pixels (D181; optional seeded subsample). Neighbourhood / ring / upwind /
    advection features of ring pixels near the domain edge are computed over the in-domain part only (features.py
    behaviour); ring_coverage records the in-domain share of the 0.3-1.0 deg ring (1.0 for city pixels);
  * labels: src/gold/build_labels.build_month on city + ring pixels (IMERG Final; IMERG Late for 2025-10);
  * features: src/gold/features.compute_features (feature_version v1, the single feature code path) on the same
    pixels, plus features.pixel_static_columns (dist_to_city_center_km, is_city, ring_coverage);
  * frame-mode G2 on the partition (src/quality/temporal_checks.check_frame_table) before anything is uploaded.

Never reads or writes anything itself except through the paths it is given; no network.
"""
from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.gold import features as F

KEYS = ["pixel_id", "t0", "lead_min", "frame_start", "season"]
STATIC = ("dist_to_city_center_km", "is_city", "ring_coverage")


# ------------------------------------------------------------------ periods
def build_days(cfg: dict) -> list[date]:
    """UTC days of the table: explicit scope.day_ranges (smoke) or the season months of scope.seasons."""
    sc = cfg["scope"]
    out: set[date] = set()
    if sc.get("day_ranges"):
        for a, b in sc["day_ranges"]:
            d, e = date.fromisoformat(a), date.fromisoformat(b)
            while d <= e:
                out.add(d)
                d += timedelta(days=1)
    else:
        for y in sc["seasons"]:
            for m in sc["season_months"]:
                d = date(y, m, 1)
                while d.month == m:
                    out.add(d)
                    d += timedelta(days=1)
    return sorted(x for x in out if x.month in sc["season_months"])


def build_months(cfg: dict) -> list[tuple[int, int]]:
    return sorted({(d.year, d.month) for d in build_days(cfg)})


def label_source(cfg: dict, y: int, m: int) -> str:
    return cfg["scope"].get("label_source_by_month", {}).get(f"{y}-{m:02d}", cfg["scope"]["label_source_default"])


def is_training_month(cfg: dict, y: int, m: int) -> bool:
    return label_source(cfg, y, m) == cfg["scope"]["training_label_source"]


# ------------------------------------------------------------------ pixel sets
def required_margin_px(p: dict, grid: F.Grid = F.Grid()) -> int:
    """Smallest margin (pixels) to the domain edge at which every input of feature set d lies inside the domain:
    neighbourhood boxes (Chebyshev radius), ring quadrants and upwind sector (outer radius in deg), and the
    advected-persistence sample (max shift x advection time of the longest lead, + 1 px for adv_max_r1)."""
    m = p["motion"]
    box = max(p["radii"])
    ring = math.ceil(p["ring"][1] / grid.res - 1e-9)
    upw = math.ceil(m["upwind_deg"][1] / grid.res - 1e-9)
    adv = math.ceil(m["max_shift_px_per_h"] * F.advect_hours(max(p["leads"]), p) - 1e-9) + 1
    return int(max(box, ring, upw, adv))


def domain_pixels(grid: F.Grid = F.Grid()) -> np.ndarray:
    return np.array([(grid.lat_idx_min + i) * 3600 + grid.lon_idx_min + j
                     for i in range(grid.n_lat) for j in range(grid.n_lon)], dtype="int64")


def margin_px(pixel_ids: np.ndarray, grid: F.Grid = F.Grid()) -> np.ndarray:
    i, j = grid.ij(pixel_ids)
    return np.minimum.reduce([i, grid.n_lat - 1 - i, j, grid.n_lon - 1 - j])


def ring_candidates(city: np.ndarray, margin: int, grid: F.Grid = F.Grid()) -> np.ndarray:
    allp = domain_pixels(grid)
    return np.setdiff1d(allp[margin_px(allp, grid) >= int(margin)], np.asarray(city, "int64"))


def largest_margin_for(city: np.ndarray, min_pixels: int, grid: F.Grid = F.Grid()) -> int:
    """Largest margin (px) whose ring (margin box minus city) has at least min_pixels pixels."""
    for m in range(min(grid.n_lat, grid.n_lon) // 2, -1, -1):
        if len(ring_candidates(city, m, grid)) >= int(min_pixels):
            return m
    raise ValueError(f"no margin gives {min_pixels} ring pixels")


def ring_pixels(city: np.ndarray, cfg: dict, grid: F.Grid = F.Grid()) -> tuple[np.ndarray, dict]:
    """Ring set by the configured rule; returns (sorted pixel ids, record for the gold meta / report).
    Rule largest_margin_min_pixels: the configured min_margin_px must be the largest margin giving
    >= min_ring_pixels ring pixels (checked here, so the config cannot drift from the rule)."""
    rc = cfg["pixels"]["ring"]
    p = F.feature_params(cfg)
    need = required_margin_px(p, grid)
    m = int(rc["min_margin_px"])
    best = largest_margin_for(city, int(rc["min_ring_pixels"]), grid)
    if rc["rule"] == "largest_margin_min_pixels" and m != best:
        raise ValueError(f"pixels.ring.min_margin_px {m} is not the largest margin with >= "
                         f"{rc['min_ring_pixels']} ring pixels ({best})")
    cand = ring_candidates(city, m, grid)
    full = len(cand)
    if rc.get("max_pixels") and full > int(rc["max_pixels"]):
        rng = np.random.default_rng(int(rc["subsample_seed"]))
        cand = np.sort(rng.choice(cand, size=int(rc["max_pixels"]), replace=False))
    city_margin = margin_px(np.asarray(city, "int64"), grid)
    cov = F.ring_coverage(cand, p["ring"], grid) if len(cand) else np.array([])
    rec = {"rule": rc["rule"], "min_ring_pixels": int(rc["min_ring_pixels"]), "min_margin_px": m,
           "largest_margin_meeting_target": best, "footprint_margin_px": need,
           "ring_candidates": int(full), "ring_pixels": int(len(cand)), "subsampled": bool(len(cand) < full),
           "expected_pixels": rc.get("expected_pixels"), "city_pixels": int(len(city)),
           "ring_pixels_truncated": int((margin_px(cand, grid) < need).sum()) if len(cand) else 0,
           "ring_coverage_min": float(cov.min()) if len(cov) else None,
           "ring_coverage_mean": float(cov.mean()) if len(cov) else None,
           "city_min_margin_px": int(city_margin.min()) if len(city) else None,
           "city_pixels_below_footprint": int((city_margin < need).sum())}
    return np.sort(cand), rec


def footprint_coverage(pixel_ids: np.ndarray, p: dict, grid: F.Grid = F.Grid()) -> pd.DataFrame:
    """In-domain share of every spatial footprint of set d per pixel: Chebyshev boxes (box_r<r>), the ring annulus
    (ring_coverage; the upwind sector lies in the same 0.3-1.0 deg annulus), the smallest quadrant of the ring
    (ring_quadrant_min) and whether the longest advection reach stays inside (adv_inside)."""
    pid = np.asarray(pixel_ids, dtype="int64")
    i, j = grid.ij(pid)
    out = {"pixel_id": pid}
    for r in p["radii"]:
        h = np.minimum(i + r, grid.n_lat - 1) - np.maximum(i - r, 0) + 1
        w = np.minimum(j + r, grid.n_lon - 1) - np.maximum(j - r, 0) + 1
        out[f"box_r{r}"] = h * w / (2 * r + 1) ** 2
    out["ring_coverage"] = F.ring_coverage(pid, p["ring"], grid)
    big = F.Grid(lat_idx_min=0, lon_idx_min=0, n_lat=200, n_lon=200)
    full = F.ring_quadrants(np.array([100]), np.array([100]), big, p["ring"])
    rq = F.ring_quadrants(i, j, grid, p["ring"])
    out["ring_quadrant_min"] = np.min([[len(rq[q][n][0]) / len(full[q][0][0]) for q in "NESW"]
                                       for n in range(len(pid))], axis=1) if len(pid) else np.array([])
    m = p["motion"]
    reach = math.ceil(m["max_shift_px_per_h"] * F.advect_hours(max(p["leads"]), p) - 1e-9) + 1
    out["adv_inside"] = (margin_px(pid, grid) >= reach).astype(float)
    return pd.DataFrame(out)


def pixel_table(city: np.ndarray, ring: np.ndarray, p: dict, grid: F.Grid = F.Grid()) -> pd.DataFrame:
    pix = np.sort(np.r_[np.asarray(city, "int64"), np.asarray(ring, "int64")])
    st = F.pixel_static_columns(pix, city, F.city_centre(city, grid), grid, ring=p["ring"])
    lat, lon = grid.latlon(pix)
    i, j = grid.ij(pix)
    fc = footprint_coverage(pix, p, grid).drop(columns=["pixel_id", "ring_coverage"])
    return pd.concat([st.assign(lat=lat, lon=lon, row=i, col=j, margin_px=margin_px(pix, grid)).reset_index(drop=True),
                      fc.reset_index(drop=True)], axis=1)


# ------------------------------------------------------------------ partition
def feature_columns(cols) -> list[str]:
    allow = tuple(F.FEATURE_GROUPS_A["gsmap"]) + ("gsmap_n_missing", "cal_", "lat", "lon") + STATIC
    return [c for c in cols if "__" not in c and c.startswith(allow) and not c.startswith("label")]


def groups(cols: list[str]):
    from src.quality.temporal_checks import FeatureGroup
    by: dict[str, list[str]] = {}
    for c in cols:
        by.setdefault(F.group_of(c), []).append(c)
    return [FeatureGroup(g, F.GROUP_KIND[g], tuple(cs)) for g, cs in by.items()]


def g2_violations(df: pd.DataFrame, fcols: list[str], cfg_path: Path) -> list[str]:
    from src.quality.temporal_checks import FrameConfig, check_frame_table
    return [str(v) for v in check_frame_table(df, groups(fcols), fcols, FrameConfig.from_yaml(cfg_path))]


def build_partition(cfg: dict, y: int, m: int, days: list[date], pixels: np.ndarray, city: np.ndarray,
                    windows: dict, grid: F.Grid = F.Grid()) -> tuple[pd.DataFrame, list[str], dict]:
    """Labels (city + ring) x features for one month. windows: GSMaP NOW windows (build_training_table.load_windows)
    of the days and the day before. Returns (table, feature columns, stats)."""
    from src.gold import build_labels as BL
    src = label_source(cfg, y, m)
    lab, st = BL.build_month(y, m, cfg, pixels, source=src, days=days, city_only=False)
    lab["t0"] = lab["t0"].astype("datetime64[ns, UTC]")
    lab = lab[lab["t0"].dt.date.isin(set(days)) & lab["pixel_id"].isin(pixels)].reset_index(drop=True)
    p = F.feature_params(cfg)
    assert p["version"] == "v1"
    t0s = pd.DatetimeIndex(np.sort(lab["t0"].unique())).tz_convert("UTC").as_unit("ns")
    fe = pd.concat([F.compute_features(*F.stack_windows(windows, t0s[s:s + 240], p, grid.n_lat, grid.n_lon),
                                       t0s[s:s + 240], pixels, p, grid) for s in range(0, len(t0s), 240)],
                   ignore_index=True)
    fe = fe.merge(F.pixel_static_columns(pixels, city, F.city_centre(city, grid), grid, ring=p["ring"]), on="pixel_id",
                  how="left", validate="many_to_one")
    fcols = [c for c in F.feature_columns(fe)]
    out = lab.merge(fe, on=["pixel_id", "t0"], how="left", validate="many_to_one")
    for c in ("t0", "prediction_time", "frame_start", "target_time", "gsmap__event_time", "gsmap__available_time"):
        out[c] = out[c].astype("datetime64[us, UTC]")
    stats = {"rows": len(out), "days": len(days), "label_source": src, "pixels": int(len(pixels)),
             "city_pixels": int(np.isin(pixels, city).sum()), "windows": len(windows),
             "label_rows_expected": st["expected_rows"], "label_rows_missing": st["missing_label_rows"],
             "rows_city": int((out["is_city"] == 1).sum()), "rows_ring": int((out["is_city"] == 0).sum())}
    return out, fcols, stats


def nan_share_by_group(df: pd.DataFrame, fcols: list[str]) -> list[dict]:
    """NaN share of every feature for city rows and for ring rows (P4D.3 report)."""
    rows = []
    for name, sel in (("city", df["is_city"] == 1), ("ring", df["is_city"] == 0)):
        sub = df.loc[sel, fcols]
        rows.append({"pixels": name, "rows": int(sel.sum()), **{c: float(sub[c].isna().mean()) if len(sub) else
                                                               float("nan") for c in fcols}})
    return rows
