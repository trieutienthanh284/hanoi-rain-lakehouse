"""Feature set A (satellite only) — the SINGLE feature code path for training and serving (pilot v0).

Pure functions. Inputs are GSMaP NOW windows on the regional domain grid (30 x 28 pixels of 0.1 deg,
see data/reference/hanoi_city_domain.json); outputs are feature columns for the requested pixels.

Batch shape convention: T analysis times; K = len(history_k) windows per t0 (k = 0 is the window
ending at t0, k ends at t0 - 30k); H x W domain grid. Serving calls the same functions with T = 1.

Admissibility (ADD-005, P4A.1): a window may be used at prediction_time = t0 + issue_offset only if
window_end <= t0 and window_end + available_after_window_end_min <= prediction_time.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

LOCAL_TZ = "Asia/Ho_Chi_Minh"


@dataclass(frozen=True)
class Grid:
    lat_idx_min: int = 1095
    lon_idx_min: int = 2843
    n_lat: int = 30
    n_lon: int = 28
    res: float = 0.1

    def ij(self, pixel_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pid = np.asarray(pixel_id, dtype="int64")
        return pid // 3600 - self.lat_idx_min, pid % 3600 - self.lon_idx_min

    def latlon(self, pixel_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pid = np.asarray(pixel_id, dtype="int64")
        return (pid // 3600 + 0.5) * self.res - 90.0, (pid % 3600 + 0.5) * self.res - 180.0


def feature_params(cfg: dict) -> dict:
    f, fr = cfg["features"], cfg["frames"]
    return {"K": list(f["history_k"]), "wet": float(f["wet_mm_h"]), "cap": float(f["cap_mm_h"]),
            "since_cap": float(f["minutes_since_wet_cap"]), "nb_k": list(f["neighbourhood_k"]),
            "radii": list(f["box_radii_px"]), "ring": tuple(f["ring_deg"]),
            "issue": int(fr["issue_offset_min"]),
            "avail_after": int(cfg["availability"]["gsmap_now"]["available_after_window_end_min"])}


# ------------------------------------------------------------------ window selection (admissibility)
def admissible(window_end: pd.Timestamp, t0: pd.Timestamp, p: dict) -> bool:
    pt = t0 + pd.Timedelta(minutes=p["issue"])
    return window_end <= t0 and window_end + pd.Timedelta(minutes=p["avail_after"]) <= pt


def stack_windows(windows: dict, t0s: pd.DatetimeIndex, p: dict, H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    """windows: {window_end (UTC Timestamp): (gc[H,W], raw[H,W])}. Returns gc, raw arrays [T,K,H,W]
    (NaN where the window is missing or not admissible at t0 + issue)."""
    T, K = len(t0s), len(p["K"])
    gc = np.full((T, K, H, W), np.nan, dtype="float32")
    raw = np.full((T, K, H, W), np.nan, dtype="float32")
    for ti, t0 in enumerate(t0s):
        for ki, k in enumerate(p["K"]):
            we = t0 - pd.Timedelta(minutes=30 * k)
            w = windows.get(we)
            if w is not None and admissible(we, t0, p):
                gc[ti, ki], raw[ti, ki] = w
    return gc, raw


# ------------------------------------------------------------------ geometry helpers
def ring_quadrants(ii: np.ndarray, jj: np.ndarray, grid: Grid, ring: tuple[float, float]) -> dict:
    """For each target pixel, (rows, cols) of domain pixels in the ring [r0, r1] deg per quadrant N/E/S/W.
    Quadrant by bearing from the target: E (-45, 45], N (45, 135], W (135, 180] U (-180, -135], S (-135, -45]."""
    H, W = grid.n_lat, grid.n_lon
    gi, gj = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    out = {q: [] for q in "NESW"}
    for i, j in zip(ii, jj):
        dlat, dlon = (gi - i) * grid.res, (gj - j) * grid.res
        d = np.hypot(dlat, dlon)
        ang = np.degrees(np.arctan2(dlat, dlon))
        inring = (d >= ring[0] - 1e-9) & (d <= ring[1] + 1e-9)
        sel = {"E": (ang > -45) & (ang <= 45), "N": (ang > 45) & (ang <= 135),
               "S": (ang > -135) & (ang <= -45), "W": (ang > 135) | (ang <= -135)}
        for q in "NESW":
            m = inring & sel[q]
            out[q].append((gi[m], gj[m]))
    return out


def _nanstat(fn, a, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(a, axis=axis)


def _wet_frac(a: np.ndarray, wet: float, axis) -> np.ndarray:
    valid = np.isfinite(a).sum(axis=axis)
    w = (a >= wet).sum(axis=axis)
    return np.where(valid > 0, w / np.maximum(valid, 1), np.nan)


# ------------------------------------------------------------------ main feature function
def compute_features(gc: np.ndarray, raw: np.ndarray, t0s: pd.DatetimeIndex, pixel_ids: np.ndarray,
                     p: dict, grid: Grid = Grid()) -> pd.DataFrame:
    """gc, raw: [T,K,H,W] admissible stacks (from stack_windows). Returns one row per (t0, pixel)
    with feature columns and the gsmap group times (event = newest admissible window end)."""
    T, K = gc.shape[:2]
    ii, jj = grid.ij(pixel_ids)
    Pn = len(pixel_ids)
    cap, wet = p["cap"], p["wet"]
    gcp = gc[:, :, ii, jj]                          # [T,K,P]
    rwp = raw[:, :, ii, jj]
    gcc = np.minimum(gcp, cap)                      # derived stats never exceed the product cap
    gcd = np.minimum(gc, cap)
    cols: dict[str, np.ndarray] = {}
    for ki, k in enumerate(p["K"]):
        cols[f"gc_k{k}"] = gcp[:, ki]
    for ki, k in enumerate(p["K"]):
        cols[f"raw_k{k}"] = rwp[:, ki]
    kpos = {k: i for i, k in enumerate(p["K"])}
    cols["gc_d01"] = gcc[:, kpos[0]] - gcc[:, kpos[1]]
    cols["gc_d02"] = gcc[:, kpos[0]] - gcc[:, kpos[2]]
    cols["gc_max_k06"] = _nanstat(np.nanmax, gcc, 1)
    cols["gc_mean_k06"] = _nanstat(np.nanmean, gcc, 1)
    present = np.isfinite(gcp)                      # [T,K,P]
    wetm = (gcp >= wet) & present
    first_wet = np.argmax(wetm, axis=1)
    any_wet = wetm.any(axis=1)
    kmin = np.asarray(p["K"])
    since = np.where(any_wet, 30.0 * kmin[first_wet], p["since_cap"])
    cols["gc_min_since_wet"] = np.where(present.any(axis=1), np.minimum(since, p["since_cap"]), np.nan)
    cols["raw_capped"] = (rwp >= cap).any(axis=1).astype("float32")
    cols["raw_capped"][~np.isfinite(rwp).any(axis=1)] = np.nan
    cols["gsmap_n_missing"] = (~present).sum(axis=1).astype("float32")
    newest = np.argmax(present, axis=1)             # first present k
    anyp = present.any(axis=1)
    age = p["issue"] + 30.0 * kmin[newest]
    cols["gsmap_age_min"] = np.where(anyp, age, np.nan)

    # neighbourhood boxes (Chebyshev radius r), windows k in nb_k
    H, W = gc.shape[2:]
    for k in p["nb_k"]:
        a = gcd[:, kpos[k]]                         # [T,H,W]
        for r in p["radii"]:
            mean = np.full((T, Pn), np.nan, "float32"); mx = mean.copy(); wf = mean.copy()
            for n, (i, j) in enumerate(zip(ii, jj)):
                box = a[:, max(i - r, 0):i + r + 1, max(j - r, 0):j + r + 1].reshape(T, -1)
                mean[:, n] = _nanstat(np.nanmean, box, 1)
                mx[:, n] = _nanstat(np.nanmax, box, 1)
                wf[:, n] = _wet_frac(box, wet, 1)
            cols[f"nb_r{r}_mean_k{k}"], cols[f"nb_r{r}_max_k{k}"], cols[f"nb_r{r}_wet_k{k}"] = mean, mx, wf
    rmax = max(p["radii"])
    cols[f"nb_r{rmax}_mean_change"] = cols[f"nb_r{rmax}_mean_k{p['nb_k'][0]}"] - cols[f"nb_r{rmax}_mean_k{p['nb_k'][-1]}"]

    # ring quadrants
    rq = ring_quadrants(ii, jj, grid, p["ring"])
    for k in p["nb_k"]:
        a = gcd[:, kpos[k]]
        for q in "NESW":
            mean = np.full((T, Pn), np.nan, "float32"); wf = mean.copy()
            for n, (ri, rj) in enumerate(rq[q]):
                if len(ri):
                    v = a[:, ri, rj]
                    mean[:, n] = _nanstat(np.nanmean, v, 1)
                    wf[:, n] = _wet_frac(v, wet, 1)
            cols[f"ring_{q}_mean_k{k}"], cols[f"ring_{q}_wet_k{k}"] = mean, wf

    # assemble rows (t0-major)
    df = pd.DataFrame({c: np.asarray(v, dtype="float32").reshape(-1) for c, v in cols.items()})
    t0v = np.repeat(t0s.values, Pn)
    df.insert(0, "t0", pd.DatetimeIndex(t0v).tz_localize("UTC") if t0s.tz is None else pd.DatetimeIndex(t0v, tz="UTC"))
    df.insert(0, "pixel_id", np.tile(np.asarray(pixel_ids, dtype="int64"), T))
    # calendar (local time) and static
    loc = df["t0"].dt.tz_convert(LOCAL_TZ)
    hour = loc.dt.hour + loc.dt.minute / 60.0
    doy = loc.dt.dayofyear.astype("float64")
    df["cal_hour_sin"] = np.sin(2 * np.pi * hour / 24).astype("float32")
    df["cal_hour_cos"] = np.cos(2 * np.pi * hour / 24).astype("float32")
    df["cal_doy_sin"] = np.sin(2 * np.pi * doy / 365.25).astype("float32")
    df["cal_doy_cos"] = np.cos(2 * np.pi * doy / 365.25).astype("float32")
    lat, lon = grid.latlon(df["pixel_id"].to_numpy())
    df["lat"], df["lon"] = lat.astype("float32"), lon.astype("float32")
    # group times for G2: newest admissible window used
    ev = df["t0"] - pd.to_timedelta(np.where(anyp, 30 * kmin[newest], 0).reshape(-1), unit="min")
    ev = ev.where(anyp.reshape(-1))
    df["gsmap__event_time"] = ev
    df["gsmap__available_time"] = ev + pd.Timedelta(minutes=p["avail_after"])
    return df


FEATURE_GROUPS_A = {
    "gsmap": ("gc_", "raw_", "nb_", "ring_", "gsmap_age_min"),
    "gsmap_meta": ("gsmap_n_missing",),
    "calendar": ("cal_",),
    "static": ("lat", "lon"),
}
GROUP_KIND = {"gsmap": "observation", "gsmap_meta": "static", "calendar": "static", "static": "static"}


def feature_columns(df: pd.DataFrame) -> list[str]:
    meta = {"pixel_id", "t0", "gsmap__event_time", "gsmap__available_time"}
    return [c for c in df.columns if c not in meta]


def group_of(col: str) -> str:
    for g, prefixes in FEATURE_GROUPS_A.items():
        if col in prefixes or any(col.startswith(pr) for pr in prefixes if pr.endswith("_")):
            return g
    raise KeyError(col)


# ------------------------------------------------------------------ feature set B: NWP (Open-Meteo), P4A.6
NWP_MODELS = {"gfs": "gfs_global", "ecmwf": "ecmwf_ifs025"}
NWP_RADIUS_DEG = 0.3


def nwp_point_map(points: pd.DataFrame, pixel_ids: np.ndarray, grid: Grid = Grid(),
                  radius: float = NWP_RADIUS_DEG) -> dict:
    """points: unique (pixel_id, lat, lon) of the Open-Meteo points. Returns per target pixel the nearest
    point and the points within `radius` degrees (centre to centre; the nearest is always included)."""
    pl = points[["pixel_id", "lat", "lon"]].drop_duplicates("pixel_id").reset_index(drop=True)
    lat, lon = grid.latlon(pixel_ids)
    out = {}
    for pid, a, b in zip(pixel_ids, lat, lon):
        d = np.hypot(pl["lat"].to_numpy() - a, pl["lon"].to_numpy() - b)
        near = int(pl["pixel_id"].iloc[int(np.argmin(d))])
        within = pl["pixel_id"].to_numpy()[d <= radius + 1e-9].tolist()
        out[int(pid)] = {"near": near, "within": sorted(set(within) | {near})}
    return out


def compute_nwp_features(rows: pd.DataFrame, om: pd.DataFrame, pmap: dict) -> pd.DataFrame:
    """rows: (pixel_id, prediction_time, frame_start). om: Open-Meteo Silver rows (model, pixel_id,
    interval_start_utc, precip_mm, forecast_issue_time_utc, available_time_utc).
    For each model: precipitation of the hour containing the frame start (h0), the hour before (hm1) and
    after (hp1), and their 3-hour sum, at the nearest point (near) and averaged over points within 0.3 deg
    (m03). Only records with forecast_issue_time <= prediction_time and available_time <= prediction_time
    are used (others -> NaN). Group times = latest issue / availability among the records used."""
    NAT = np.iinfo("int64").min
    n = len(rows)
    out = {}
    pt = rows["prediction_time"].dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]").to_numpy().view("int64")
    fs = rows["frame_start"].dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]").to_numpy().view("int64")
    h0 = fs - fs % (3600 * 10**9)
    pix = rows["pixel_id"].to_numpy("int64")
    lists = {"near": [[pmap[int(p)]["near"]] for p in pix], "m03": [pmap[int(p)]["within"] for p in pix]}
    def ns(col):
        return om[col].dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]").to_numpy().view("int64")
    times = {}
    for short, model in NWP_MODELS.items():
        sel = (om["model"] == model).to_numpy()
        key = pd.MultiIndex.from_arrays([om["pixel_id"].to_numpy("int64")[sel], ns("interval_start_utc")[sel]])
        tab = pd.DataFrame({"v": om["precip_mm"].to_numpy("float64")[sel], "iss": ns("forecast_issue_time_utc")[sel],
                            "av": ns("available_time_utc")[sel]}, index=key)
        tab = tab[~tab.index.duplicated()]
        iss_max = np.full(n, NAT, "int64"); av_max = np.full(n, NAT, "int64")
        for agg, pts in lists.items():
            lens = np.array([len(x) for x in pts]); starts = np.r_[0, np.cumsum(lens)[:-1]]
            flat_p = np.concatenate([np.asarray(x, dtype="int64") for x in pts])
            vals = {}
            for name, off in (("hm1", -1), ("h0", 0), ("hp1", 1)):
                hh = np.repeat(h0 + off * 3600 * 10**9, lens)
                r = tab.reindex(pd.MultiIndex.from_arrays([flat_p, hh]))
                iss = r["iss"].to_numpy("float64"); av = r["av"].to_numpy("float64")
                ptr = np.repeat(pt, lens).astype("float64")
                ok = np.isfinite(iss) & np.isfinite(av) & (iss <= ptr) & (av <= ptr)
                v = np.where(ok, r["v"].to_numpy("float64"), np.nan)
                cnt = np.add.reduceat(np.isfinite(v).astype("int64"), starts)
                tot = np.add.reduceat(np.nan_to_num(v), starts)
                vals[name] = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
                ii = np.where(ok, iss, -np.inf); aa = np.where(ok, av, -np.inf)
                im = np.maximum.reduceat(ii, starts); am = np.maximum.reduceat(aa, starts)
                iss_max = np.where(np.isfinite(im), np.maximum(iss_max, np.where(np.isfinite(im), im, 0).astype("int64")), iss_max)
                av_max = np.where(np.isfinite(am), np.maximum(av_max, np.where(np.isfinite(am), am, 0).astype("int64")), av_max)
            for name in ("hm1", "h0", "hp1"):
                out[f"nwp_{short}_{agg}_{name}"] = vals[name].astype("float32")
            out[f"nwp_{short}_{agg}_sum3"] = (vals["hm1"] + vals["h0"] + vals["hp1"]).astype("float32")
        times[f"nwp_{short}__issue_time"] = iss_max
        times[f"nwp_{short}__available_time"] = av_max
    df = pd.DataFrame(out, index=rows.index)
    for c, v in times.items():
        t = pd.to_datetime(np.where(v == NAT, np.datetime64("NaT"), v.view("datetime64[ns]")), utc=True)
        df[c] = t
    return df


FEATURE_GROUPS_A.update({"nwp_gfs": ("nwp_gfs_",), "nwp_ecmwf": ("nwp_ecmwf_",)})
GROUP_KIND.update({"nwp_gfs": "forecast", "nwp_ecmwf": "forecast"})
