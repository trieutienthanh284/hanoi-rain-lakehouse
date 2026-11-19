"""Feature set A (satellite only) — the SINGLE feature code path for training and serving (pilot v0 and v1).

feature_version (configs): v0 = pilot set A (with cal_doy_*); v1 = P4B.2: no cal_doy_*, + cal_month and the
motion (cross-correlation of the k = 2 -> 0 and k = 1 -> 0 GC fields), advected-persistence, upwind and
rain-extent columns, all computed from the same admissible stack (no other input).

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


MOTION_DEFAULTS = {"max_shift_px_per_h": 4, "min_corr": 0.3, "min_wet_frac": 0.01, "min_overlap_px": 100,
                   "window_centre_offset_min": -30, "upwind_deg": [0.3, 1.0], "upwind_half_angle_deg": 45.0,
                   "extent_mm_h": [0.5, 2.5]}


def feature_params(cfg: dict) -> dict:
    """feature_version 'v0' = pilot v0 set A (with cal_doy_*; keeps the v0 models loadable);
    'v1' = P4B.2 (no cal_doy_*; + cal_month, motion, advected persistence, upwind, rain extent)."""
    f, fr = cfg["features"], cfg["frames"]
    return {"K": list(f["history_k"]), "wet": float(f["wet_mm_h"]), "cap": float(f["cap_mm_h"]),
            "since_cap": float(f["minutes_since_wet_cap"]), "nb_k": list(f["neighbourhood_k"]),
            "radii": list(f["box_radii_px"]), "ring": tuple(f["ring_deg"]),
            "issue": int(fr["issue_offset_min"]), "leads": [int(x) for x in fr["lead_min"]],
            "frame_len": int(fr["frame_length_min"]),
            "avail_after": int(cfg["availability"]["gsmap_now"]["available_after_window_end_min"]),
            "version": str(f.get("feature_version", "v0")),
            "motion": {**MOTION_DEFAULTS, **(f.get("motion") or {})}}


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


# ------------------------------------------------------------------ P4B.2 motion / advection (pure; admissible windows only)
def _corr_at_shift(a: np.ndarray, b: np.ndarray, di: int, dj: int, region: tuple, min_overlap: int) -> np.ndarray:
    """Pearson correlation per t between a[t, i, j] and b[t, i - di, j - dj] over the cells (i, j) of `region`
    (i0, i1, j0, j1 of a) whose displaced cell lies inside the grid. NaN-aware; a, b: [T,H,W]."""
    T, H, W = a.shape
    i0, i1, j0, j1 = region
    ai0, ai1, aj0, aj1 = max(i0, di), min(i1, H + di), max(j0, dj), min(j1, W + dj)
    if ai1 <= ai0 or aj1 <= aj0:
        return np.full(T, np.nan)
    x = a[:, ai0:ai1, aj0:aj1].astype("float64")
    y = b[:, ai0 - di:ai1 - di, aj0 - dj:aj1 - dj].astype("float64")
    v = np.isfinite(x) & np.isfinite(y)
    n = v.sum(axis=(1, 2)).astype("float64")
    x0, y0 = np.where(v, x, 0.0), np.where(v, y, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mx, my = x0.sum((1, 2)) / n, y0.sum((1, 2)) / n
        vx = (x0 * x0).sum((1, 2)) / n - mx * mx
        vy = (y0 * y0).sum((1, 2)) / n - my * my
        cov = (x0 * y0).sum((1, 2)) / n - mx * my
        r = cov / np.sqrt(vx * vy)
    return np.where((n >= min_overlap) & (vx > 1e-12) & (vy > 1e-12), r, np.nan)


def motion_search(a: np.ndarray, b: np.ndarray, max_shift: int, region: tuple | None = None,
                  min_overlap: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Displacement (di, dj) in pixels from field b (earlier) to field a (later) maximising the cross-correlation
    over |di|, |dj| <= max_shift, i.e. rain at b[i - di, j - dj] is found at a[i, j]; parabolic sub-pixel
    refinement per axis. a, b: [T,H,W]. Returns di (rows, northward), dj (cols, eastward), peak corr; NaN if none."""
    T, H, W = a.shape
    region = region or (0, H, 0, W)
    s = np.arange(-max_shift, max_shift + 1)
    C = np.full((T, len(s), len(s)), np.nan)
    for ii, di in enumerate(s):
        for jj, dj in enumerate(s):
            C[:, ii, jj] = _corr_at_shift(a, b, int(di), int(dj), region, min_overlap)
    flat = np.where(np.isfinite(C), C, -np.inf).reshape(T, -1)
    best = flat.argmax(axis=1)
    peak = flat[np.arange(T), best]
    ok = np.isfinite(peak)
    bi, bj = np.divmod(best, len(s))
    tt = np.arange(T)

    def _off(cm, c0, cp):
        den = cm - 2 * c0 + cp
        with np.errstate(divide="ignore", invalid="ignore"):
            o = np.where(np.isfinite(cm) & np.isfinite(cp) & (den < 0), 0.5 * (cm - cp) / den, 0.0)
        return np.clip(np.nan_to_num(o), -0.5, 0.5)

    def _nb(i, j):
        inside = (i >= 0) & (i < len(s)) & (j >= 0) & (j < len(s))
        return np.where(inside, C[tt, np.clip(i, 0, len(s) - 1), np.clip(j, 0, len(s) - 1)], np.nan)

    c0 = C[tt, bi, bj]
    di = s[bi] + _off(_nb(bi - 1, bj), c0, _nb(bi + 1, bj))
    dj = s[bj] + _off(_nb(bi, bj - 1), c0, _nb(bi, bj + 1))
    return np.where(ok, di, np.nan), np.where(ok, dj, np.nan), np.where(ok, peak, np.nan)


def motion_vector(later: np.ndarray, earlier: np.ndarray, dt_h: float, p: dict,
                  region: tuple | None = None) -> dict:
    """Motion (u east, v north; pixels/hour) between two capped rain fields [T,H,W] dt_h hours apart.
    valid = peak corr >= min_corr and both fields have a wet fraction >= min_wet_frac (zero-rain and pure-noise
    fields are invalid). Correlation is computed on log1p(rate)."""
    m = p["motion"]
    T = later.shape[0]
    ms = max(1, int(round(m["max_shift_px_per_h"] * dt_h)))
    di, dj, r = motion_search(np.log1p(later), np.log1p(earlier), ms, region, int(m["min_overlap_px"]))
    sl = (slice(None),) if region is None else (slice(None), slice(region[0], region[1]), slice(region[2], region[3]))
    wa = _wet_frac(later[sl].reshape(T, -1), p["wet"], 1)
    wb = _wet_frac(earlier[sl].reshape(T, -1), p["wet"], 1)
    valid = np.isfinite(r) & (r >= m["min_corr"]) & (np.nan_to_num(wa) >= m["min_wet_frac"]) & \
        (np.nan_to_num(wb) >= m["min_wet_frac"])
    return {"u": dj / dt_h, "v": di / dt_h, "corr": r, "valid": valid}


def bilinear(field: np.ndarray, fi: np.ndarray, fj: np.ndarray) -> np.ndarray:
    """Bilinear sample of field [T,H,W] at fractional (row, col) positions [T,P]; NaN outside the grid or where a
    corner is NaN."""
    T, H, W = field.shape
    inside = (fi >= 0) & (fi <= H - 1) & (fj >= 0) & (fj <= W - 1)
    fi, fj = np.where(inside, fi, 0.0), np.where(inside, fj, 0.0)
    i0, j0 = np.floor(fi).astype("int64"), np.floor(fj).astype("int64")
    i1, j1 = np.minimum(i0 + 1, H - 1), np.minimum(j0 + 1, W - 1)
    wi, wj = fi - i0, fj - j0
    t = np.arange(T)[:, None]
    v = ((1 - wi) * (1 - wj) * field[t, i0, j0] + (1 - wi) * wj * field[t, i0, j1] +
         wi * (1 - wj) * field[t, i1, j0] + wi * wj * field[t, i1, j1])
    return np.where(inside, v, np.nan)


def advect_hours(lead_min: int, p: dict) -> float:
    """Time from the centre of window k = 0 to the centre of frame F<lead> (hours)."""
    return (lead_min + p["frame_len"] / 2 - p["motion"]["window_centre_offset_min"]) / 60.0


def upwind_stats(field: np.ndarray, ii: np.ndarray, jj: np.ndarray, u: np.ndarray, v: np.ndarray,
                 valid: np.ndarray, grid: Grid, p: dict) -> tuple[np.ndarray, np.ndarray]:
    """Mean rate and wet fraction of field [T,H,W] in the sector against the motion direction: distance
    upwind_deg[0]..[1] from the target, bearing within +-upwind_half_angle_deg of the upwind bearing. NaN where
    the motion is invalid. Returns [T,P] arrays."""
    m = p["motion"]
    H, W = field.shape[1:]
    gi, gj = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    dlat = (gi[None] - ii[:, None, None]) * grid.res                       # [P,H,W]
    dlon = (gj[None] - jj[:, None, None]) * grid.res
    d = np.hypot(dlat, dlon)
    ang = np.degrees(np.arctan2(dlat, dlon))
    ring = (d >= m["upwind_deg"][0] - 1e-9) & (d <= m["upwind_deg"][1] + 1e-9)
    ub = np.degrees(np.arctan2(-np.nan_to_num(v), -np.nan_to_num(u)))   # bearing the rain comes from
    diff = np.abs((ang[None] - ub[:, None, None, None] + 180.0) % 360.0 - 180.0)
    mask = ring[None] & (diff <= m["upwind_half_angle_deg"] + 1e-9)    # [T,P,H,W]
    f = field[:, None]
    fin = mask & np.isfinite(f)
    cnt = fin.sum(axis=(2, 3))
    tot = np.where(fin, f, 0.0).sum(axis=(2, 3))
    wetc = (fin & (np.nan_to_num(f) >= p["wet"])).sum(axis=(2, 3))
    with np.errstate(divide="ignore", invalid="ignore"):
        mean, wf = tot / cnt, wetc / cnt
    ok = valid[:, None] & (cnt > 0)
    return np.where(ok, mean, np.nan), np.where(ok, wf, np.nan)


def quadrant_of(ii: np.ndarray, jj: np.ndarray, H: int, W: int) -> np.ndarray:
    """0 = SW, 1 = SE, 2 = NW, 3 = NE (rows grow northward)."""
    return (np.asarray(ii) >= H // 2).astype(int) * 2 + (np.asarray(jj) >= W // 2).astype(int)


def v1_columns(gcd: np.ndarray, kpos: dict, ii: np.ndarray, jj: np.ndarray, anyp: np.ndarray, p: dict,
               grid: Grid) -> dict:
    """P4B.2 feature columns ([T,P] arrays) from the capped admissible GC stack gcd [T,K,H,W]."""
    T, _, H, W = gcd.shape
    f0, f2 = gcd[:, kpos[0]], gcd[:, kpos[2]]
    m20 = motion_vector(f0, f2, 1.0, p)
    m10 = motion_vector(f0, gcd[:, kpos[1]], 0.5, p) if 1 in kpos else None
    use20 = m20["valid"]
    use10 = (~use20 & m10["valid"]) if m10 is not None else np.zeros(T, bool)
    valid = use20 | use10
    u = np.where(use20, m20["u"], np.where(use10, m10["u"] if m10 is not None else np.nan, np.nan))
    v = np.where(use20, m20["v"], np.where(use10, m10["v"] if m10 is not None else np.nan, np.nan))
    corr = np.where(use10, m10["corr"] if m10 is not None else np.nan, m20["corr"])
    P = len(ii)
    rep = lambda a: np.repeat(np.asarray(a, "float64")[:, None], P, axis=1)  # noqa: E731
    cols: dict[str, np.ndarray] = {
        "motion_u": rep(u), "motion_v": rep(v), "motion_corr": rep(corr),
        "motion_valid": np.where(anyp, rep(valid.astype("float64")), np.nan)}
    # coarse local vector: 4 quadrants of the domain (1-hour displacement k = 2 -> 0)
    q = quadrant_of(ii, jj, H, W)
    qu, qv, qok = np.full((T, P), np.nan), np.full((T, P), np.nan), np.zeros((T, P))
    regions = [(0, H // 2, 0, W // 2), (0, H // 2, W // 2, W), (H // 2, H, 0, W // 2), (H // 2, H, W // 2, W)]
    for qi, reg in enumerate(regions):
        sel = q == qi
        if not sel.any():
            continue
        mq = motion_vector(f0, f2, 1.0, p, region=reg)
        qu[:, sel] = np.where(mq["valid"], mq["u"], np.nan)[:, None]
        qv[:, sel] = np.where(mq["valid"], mq["v"], np.nan)[:, None]
        qok[:, sel] = mq["valid"].astype("float64")[:, None]
    cols["motion_q_u"], cols["motion_q_v"] = qu, qv
    cols["motion_q_valid"] = np.where(anyp, qok, np.nan)
    # advected persistence (zero vector when the motion is invalid, i.e. plain persistence of k = 0)
    ua, va = np.where(valid, u, 0.0), np.where(valid, v, 0.0)
    for L in p["leads"]:
        h = advect_hours(L, p)
        si = ii[None, :] - (va * h)[:, None]
        sj = jj[None, :] - (ua * h)[:, None]
        cols[f"adv_rate_F{L}"] = bilinear(f0, si, sj)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cols[f"adv_max_r1_F{L}"] = np.nanmax(np.stack([bilinear(f0, si + a, sj + b)
                                                           for a in (-1, 0, 1) for b in (-1, 0, 1)]), axis=0)
    cols["upwind_mean"], cols["upwind_wet"] = upwind_stats(f0, ii, jj, u, v, valid, grid, p)
    # rain extent over the whole regional domain
    for thr in p["motion"]["extent_mm_h"]:
        k = f"{thr:g}".replace(".", "p")
        e0 = _wet_frac(f0.reshape(T, -1), thr, 1)
        e2 = _wet_frac(f2.reshape(T, -1), thr, 1)
        cols[f"extent_wet{k}_k0"] = rep(e0)
        cols[f"extent_wet{k}_d20"] = rep(e0 - e2)
    return cols


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

    if p.get("version", "v0") == "v1":                  # P4B.2: motion, advected persistence, upwind, extent
        for c, val in v1_columns(gcd, kpos, ii, jj, anyp, p, grid).items():
            cols[c] = np.where(anyp, val, np.nan)       # never a value without an admissible window

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
    if p.get("version", "v0") == "v0":                  # pilot v0 only; removed in v1 (P4B.2)
        df["cal_doy_sin"] = np.sin(2 * np.pi * doy / 365.25).astype("float32")
        df["cal_doy_cos"] = np.cos(2 * np.pi * doy / 365.25).astype("float32")
    else:                                               # v1: local month (used only if OOF log-loss improves)
        df["cal_month"] = loc.dt.month.astype("float32")
    lat, lon = grid.latlon(df["pixel_id"].to_numpy())
    df["lat"], df["lon"] = lat.astype("float32"), lon.astype("float32")
    # group times for G2: newest admissible window used
    ev = df["t0"] - pd.to_timedelta(np.where(anyp, 30 * kmin[newest], 0).reshape(-1), unit="min")
    ev = ev.where(anyp.reshape(-1))
    df["gsmap__event_time"] = ev
    df["gsmap__available_time"] = ev + pd.Timedelta(minutes=p["avail_after"])
    return df


FEATURE_GROUPS_A = {
    "gsmap": ("gc_", "raw_", "nb_", "ring_", "gsmap_age_min", "motion_", "adv_", "upwind_", "extent_"),
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
