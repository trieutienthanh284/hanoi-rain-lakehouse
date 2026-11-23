"""P4E.6 figures (D196), generated ON THE CLOUD inside the comparison stage of the p4e jobs (gate `figs`), written under
lake/v1/runs/<job>/<run_id>/files/reports/p4e/figs/ and copied by fetch_run to reports/p4e/figs/<job>_<run_id>/.
Plot style as P4B.6 / P4B.7 (src/models/p4b_post.py: reliability with a log-count twin axis, replay panels of
pixel squares, Agg backend, dpi 90-100, NOT_FLOOD in every title).

Pure functions over pooled OOF frames: F[(model, threshold, lead)] -> DataFrame with the columns of p4d.pooled
(pixel_id, t0, day, half, role, label_source, label_rate_mm_h, pers, y, p_raw, p_cal, yhat). Persistence is taken from
the `pers` column of any frame. Not a flood model; labels are IMERG (satellite estimate), not rain gauges.

  reliability      v1 (B), Model 1-B, Model 1 (A) | 2.5 and 7.6 | F0 / F30 / F60 | raw and recalibrated
  scores_ci        CSI / POD / FAR per frame with 90 % day-block bootstrap CIs, persistence included
  fss              FSS vs neighbourhood (1x1, 3x3, 5x5) at 2.5 and 7.6, inside the city mask (truncated windows
                   normalised by the in-mask pixel count), pooled sum-numerator / sum-denominator, day-block CI
  halfmonth_csi    CSI per half-month (UTC date of t0)
  replay           label IMERG vs P(>=2.5) of two models at F0 / F30 / F60 for named days (Late labels marked)
  diagnosis        P4E.2: distribution of P(>=2.5) and P(>=7.6) on city pixels with label >= 7.6, M0 vs Model 1
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

NOT_FLOOD = "Rain nowcast only (IMERG labels, satellite estimate); not a flood prediction."
TZ = "Asia/Ho_Chi_Minh"
PERS = "persistence"
COLORS = ["#1b6ca8", "#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
PERS_COLOR = "0.35"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _color(models: list[str], m: str) -> str:
    return PERS_COLOR if m == PERS else COLORS[models.index(m) % len(COLORS)]


def _get(F: dict, m: str, t: float, L: int):
    return F.get((m, float(t), int(L)))


def _pers_frame(F: dict, models: list[str], t: float, L: int) -> pd.DataFrame | None:
    for m in models:
        j = _get(F, m, t, L)
        if j is not None and len(j):
            return j
    return None


def _binary(F: dict, models: list[str], m: str, t: float, L: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray] | None:
    """(frame, y, forecast yes/no) for a model or persistence (pers >= threshold, NaN = no)."""
    if m == PERS:
        j = _pers_frame(F, models, t, L)
        if j is None:
            return None
        return j, j["y"].to_numpy(bool), np.nan_to_num(j["pers"].to_numpy("float64"), nan=-1.0) >= t
    j = _get(F, m, t, L)
    if j is None or not len(j):
        return None
    return j, j["y"].to_numpy(bool), j["yhat"].to_numpy(bool)


# ============================================================================ reliability
def reliability(F: dict, models: list[str], thrs: list[float], leads: list[int], out: Path, title: str,
                bins: int = 10) -> tuple[list[Path], pd.DataFrame]:
    from src.models import metrics as M
    plt = _plt()
    rows = []
    spec = [(t, kind) for t in thrs for kind in ("raw", "cal")]
    fig, ax = plt.subplots(len(spec), len(leads), figsize=(4.2 * len(leads), 4.0 * len(spec)),
                           constrained_layout=True, squeeze=False)
    for i, (t, kind) in enumerate(spec):
        for jx, L in enumerate(leads):
            a = ax[i, jx]
            a.plot([0, 1], [0, 1], color="0.6", lw=0.8, ls="--")
            ref = None
            for m in models:
                j = _get(F, m, t, L)
                if j is None or not len(j):
                    continue
                r = M.reliability(j["y"].to_numpy(bool), j[f"p_{kind}"].to_numpy("float64"), bins)
                d = r[r["n"] > 0]
                ece = float((d["n"] * (d["mean_p"] - d["obs_freq"]).abs()).sum() / max(d["n"].sum(), 1))
                a.plot(d["mean_p"], d["obs_freq"], marker="o", ms=3, lw=1.2, color=_color(models, m),
                       label=f"{m} (ECE {ece:.4f})")
                for _, x in r.iterrows():
                    rows.append({"threshold": t, "lead_min": L, "prob": kind, "model": m, **x.to_dict()})
                ref = r if ref is None else ref
            if ref is not None:
                b = a.twinx()
                b.bar((ref["p_lo"] + ref["p_hi"]) / 2, ref["n"].clip(lower=1), width=0.9 / bins, color="0.85",
                      zorder=0)
                b.set_yscale("log"); b.set_ylabel(f"n ({models[0]})", fontsize=7); b.tick_params(labelsize=6)
                a.set_zorder(b.get_zorder() + 1); a.patch.set_visible(False)
            a.set_xlim(0, 1); a.set_ylim(0, 1)
            a.set_title(f">= {t:g} mm/h, F{L}, {'recalibrated (Platt)' if kind == 'cal' else 'raw'}", fontsize=9)
            a.set_xlabel("forecast probability", fontsize=8); a.set_ylabel("observed frequency", fontsize=8)
            a.legend(fontsize=6, loc="upper left")
    fig.suptitle(f"{title} — reliability (pooled OOF, city pixels)\n{NOT_FLOOD}", fontsize=9)
    p = out / "reliability.png"
    fig.savefig(p, dpi=100); plt.close(fig)
    return [p], pd.DataFrame(rows)


# ============================================================================ CSI / POD / FAR with CIs
def scores_ci(F: dict, models: list[str], thrs: list[float], leads: list[int], out: Path, title: str,
              n_boot: int, seed: int, level: float) -> tuple[list[Path], pd.DataFrame]:
    from src.models import metrics as M
    plt = _plt()
    names = models + [PERS]
    rows = []
    for t in thrs:
        for L in leads:
            for m in names:
                b = _binary(F, models, m, t, L)
                if b is None:
                    continue
                j, y, f = b
                ds = M.day_stats(j["day"].to_numpy(), y, f, f.astype("float64"), np.zeros(len(y)))
                ci = M.bootstrap_ci(ds, n_boot, seed, level)
                sc = M.scores(**M.contingency(y, f))
                for k in ("csi", "pod", "far"):
                    rows.append({"threshold": t, "lead_min": L, "model": m, "metric": k, "value": float(sc[k]),
                                 "lo": ci[k][0], "hi": ci[k][1], "n": int(len(y)), "days": int(len(ds))})
    r = pd.DataFrame(rows)
    mets = ["csi", "pod", "far"]
    fig, ax = plt.subplots(len(thrs), 3, figsize=(4.4 * 3, 3.6 * len(thrs)), constrained_layout=True, squeeze=False)
    for i, t in enumerate(thrs):
        for jx, k in enumerate(mets):
            a = ax[i, jx]
            for q, m in enumerate(names):
                d = r[(r["threshold"] == t) & (r["model"] == m) & (r["metric"] == k)].sort_values("lead_min")
                if not len(d):
                    continue
                x = np.arange(len(d)) + (q - (len(names) - 1) / 2) * 0.12
                a.errorbar(x, d["value"], yerr=[d["value"] - d["lo"], d["hi"] - d["value"]], fmt="o", ms=4,
                           capsize=3, color=_color(models, m), label=m)
                a.set_xticks(np.arange(len(d)), [f"F{L}" for L in d["lead_min"]])
            a.set_title(f"{k.upper()} >= {t:g} mm/h (OOF cut; {int(level * 100)} % day-block CI)", fontsize=9)
            a.set_ylim(0, 1); a.grid(alpha=0.3)
            if i == 0 and jx == 0:
                a.legend(fontsize=7)
    fig.suptitle(f"{title} — CSI / POD / FAR per frame\n{NOT_FLOOD}", fontsize=9)
    p = out / "scores_ci.png"
    fig.savefig(p, dpi=100); plt.close(fig)
    return [p], r


# ============================================================================ Fractions Skill Score
def _box_sum(a: np.ndarray, r: int) -> np.ndarray:
    """Sum over the (2r+1)x(2r+1) window of every cell of a stack (T, H, W), zero outside the grid."""
    if r == 0:
        return a.astype("float64")
    T, H, W = a.shape
    pad = np.zeros((T, H + 2 * r + 1, W + 2 * r + 1), "float64")
    pad[:, r + 1:r + 1 + H, r + 1:r + 1 + W] = a
    c = pad.cumsum(1).cumsum(2)
    k = 2 * r + 1
    return c[:, k:, k:] - c[:, :-k, k:] - c[:, k:, :-k] + c[:, :-k, :-k]


def fss_terms(pixel_id: np.ndarray, t0: pd.Series, day: np.ndarray, obs: np.ndarray, fc: np.ndarray,
              size: int) -> pd.DataFrame:
    """Per-day sums of the FSS numerator sum((Pf - Po)^2) and denominator sum(Pf^2 + Po^2) over the city mask (the
    pixels present in the frame); window fractions use only in-mask pixels (truncated windows, normalised by the in-mask
    count)."""
    from src.gold import features as Fe
    i, j = Fe.Grid().ij(np.asarray(pixel_id, "int64"))
    i, j = i - i.min(), j - j.min()
    codes, uniq = pd.factorize(pd.Series(t0).reset_index(drop=True))
    H, W = int(i.max()) + 1, int(j.max()) + 1
    mask = np.zeros((H, W), "float64")
    mask[i, j] = 1.0
    o = np.zeros((len(uniq), H, W)); f = np.zeros_like(o)
    o[codes, i, j] = np.asarray(obs, "float64"); f[codes, i, j] = np.asarray(fc, "float64")
    r = size // 2
    cnt = _box_sum(mask[None], r)[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        po, pf = _box_sum(o, r) / cnt, _box_sum(f, r) / cnt
    m = mask.astype(bool)
    num = ((pf - po) ** 2)[:, m].sum(1)
    den = (pf ** 2 + po ** 2)[:, m].sum(1)
    tday = pd.Series(np.asarray(day)).groupby(codes).first().to_numpy()
    return pd.DataFrame({"day": tday, "num": num, "den": den}).groupby("day")[["num", "den"]].sum()


def fss_ci(terms: pd.DataFrame, n_boot: int, seed: int, level: float) -> tuple[float, float, float]:
    from src.models import metrics as M
    tot = terms[["num", "den"]].to_numpy()
    point = 1 - tot[:, 0].sum() / tot[:, 1].sum() if tot[:, 1].sum() > 0 else float("nan")
    w = M._resample_weights(len(terms), n_boot, seed)
    s = w @ tot
    with np.errstate(invalid="ignore", divide="ignore"):
        v = 1 - s[:, 0] / s[:, 1]
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    return float(point), float(np.nanquantile(v, lo)), float(np.nanquantile(v, hi))


def fss(F: dict, models: list[str], thrs: list[float], leads: list[int], out: Path, title: str, sizes: list[int],
        n_boot: int, seed: int, level: float) -> tuple[list[Path], pd.DataFrame]:
    plt = _plt()
    names = models + [PERS]
    rows = []
    for t in thrs:
        for L in leads:
            for m in names:
                b = _binary(F, models, m, t, L)
                if b is None:
                    continue
                j, y, f = b
                for sz in sizes:
                    tr = fss_terms(j["pixel_id"].to_numpy(), j["t0"], j["day"].to_numpy(), y, f, sz)
                    v, lo, hi = fss_ci(tr, n_boot, seed, level)
                    rows.append({"threshold": t, "lead_min": L, "model": m, "size": sz, "fss": v, "lo": lo, "hi": hi,
                                 "days": int(len(tr))})
    r = pd.DataFrame(rows)
    fig, ax = plt.subplots(len(thrs), len(leads), figsize=(4.2 * len(leads), 3.6 * len(thrs)),
                           constrained_layout=True, squeeze=False)
    for i, t in enumerate(thrs):
        for jx, L in enumerate(leads):
            a = ax[i, jx]
            for m in names:
                d = r[(r["threshold"] == t) & (r["lead_min"] == L) & (r["model"] == m)].sort_values("size")
                if not len(d):
                    continue
                c = _color(models, m)
                a.plot(d["size"], d["fss"], marker="o", ms=4, lw=1.2, color=c, label=m)
                a.fill_between(d["size"], d["lo"], d["hi"], color=c, alpha=0.15, lw=0)
            a.set_xticks(sizes, [f"{s}x{s}" for s in sizes])
            a.set_ylim(0, 1); a.grid(alpha=0.3)
            a.set_title(f"FSS >= {t:g} mm/h, F{L}", fontsize=9)
            a.set_xlabel("neighbourhood (pixels, truncated to the city mask)", fontsize=8)
            if i == 0 and jx == 0:
                a.legend(fontsize=7)
    fig.suptitle(f"{title} — Fractions Skill Score, 47-pixel city mask (5x5 windows often truncated)\n{NOT_FLOOD}",
                 fontsize=9)
    p = out / "fss.png"
    fig.savefig(p, dpi=100); plt.close(fig)
    return [p], r


# ============================================================================ CSI per half-month
def halfmonth_csi(F: dict, models: list[str], thrs: list[float], leads: list[int], out: Path,
                  title: str) -> tuple[list[Path], pd.DataFrame]:
    from src.models import metrics as M
    plt = _plt()
    names = models + [PERS]
    rows = []
    for t in thrs:
        for L in leads:
            for m in names:
                b = _binary(F, models, m, t, L)
                if b is None:
                    continue
                j, y, f = b
                for h, idx in pd.Series(j["half"].to_numpy()).groupby(j["half"].to_numpy()).indices.items():
                    sc = M.scores(**M.contingency(y[idx], f[idx]))
                    rows.append({"threshold": t, "lead_min": L, "model": m, "half_month": h, "n": int(len(idx)),
                                 "positives": int(y[idx].sum()), "csi": float(sc["csi"])})
    r = pd.DataFrame(rows)
    fig, ax = plt.subplots(len(thrs), len(leads), figsize=(4.6 * len(leads), 3.4 * len(thrs)),
                           constrained_layout=True, squeeze=False)
    halves = sorted(r["half_month"].unique()) if len(r) else []
    for i, t in enumerate(thrs):
        for jx, L in enumerate(leads):
            a = ax[i, jx]
            for m in names:
                d = r[(r["threshold"] == t) & (r["lead_min"] == L) & (r["model"] == m)].set_index("half_month")
                if not len(d):
                    continue
                a.plot(range(len(halves)), d.reindex(halves)["csi"], marker="o", ms=3, lw=1.1,
                       color=_color(models, m), label=m)
            a.set_xticks(range(len(halves)), halves, rotation=60, fontsize=6)
            a.set_ylim(0, 1); a.grid(alpha=0.3)
            a.set_title(f"CSI >= {t:g} mm/h, F{L} (seasons pooled)", fontsize=9)
            if i == 0 and jx == 0:
                a.legend(fontsize=7)
    fig.suptitle(f"{title} — CSI per half-month (MM-1 = days 1-15, MM-2 = 16-end, UTC date of t0)\n{NOT_FLOOD}",
                 fontsize=9)
    p = out / "halfmonth_csi.png"
    fig.savefig(p, dpi=100); plt.close(fig)
    return [p], r


# ============================================================================ replay maps
def replay(F: dict, pair: list[str], thr: float, leads: list[int], days: list[dict], out: Path,
           title: str) -> tuple[list[Path], pd.DataFrame]:
    """days: [{"name", "day" (local YYYY-MM-DD)}]. One PNG per day and frame: rows = IMERG label (mm/h), P(>=thr) of the
    two models (recalibrated); 8 consecutive t0 centred on the wettest (city-mean label). Late labels are marked."""
    from src.gold import features as Fe
    plt = _plt()
    paths, rows = [], []
    for ev in days:
        for L in leads:
            fr = {m: _get(F, m, thr, L) for m in pair}
            if any(v is None for v in fr.values()):
                rows.append({"name": ev["name"], "day": ev["day"], "lead_min": L, "status": "model missing"})
                continue
            d = {m: v[v["day"] == ev["day"]] for m, v in fr.items()}
            base = d[pair[0]]
            if not len(base):
                rows.append({"name": ev["name"], "day": ev["day"], "lead_min": L, "status": "day not in evaluation"})
                continue
            lab = base.groupby("t0")["label_rate_mm_h"].mean()
            ts = list(lab.index)
            c = ts.index(lab.idxmax()) if lab.notna().any() else 0
            sel = ts[max(0, min(c - 3, len(ts) - 8)):][:8]
            src = sorted(set(base["label_source"].astype(str)))
            late = any("late" in s.lower() for s in src)
            spec = [("label_rate_mm_h", pair[0], dict(cmap="Blues", vmin=0, vmax=20),
                     "IMERG label mm/h" + (" (LATE, provisional)" if late else ""))]
            spec += [("p_cal", m, dict(cmap="viridis", vmin=0, vmax=1), f"P(>={thr:g}) {m}") for m in pair]
            fig, ax = plt.subplots(len(spec), len(sel), figsize=(2.1 * len(sel), 2.0 * len(spec) + 0.6),
                                   constrained_layout=True, squeeze=False)
            for jx, t in enumerate(sel):
                loc = (pd.Timestamp(t) + pd.Timedelta(minutes=int(L))).tz_convert(TZ)
                for i, (col, m, kw, ttl) in enumerate(spec):
                    s = d[m][d[m]["t0"] == t]
                    lat, lon = Fe.Grid().latlon(s["pixel_id"].to_numpy())
                    a = ax[i, jx]
                    sc = a.scatter(lon, lat, c=s[col], marker="s", s=60, **kw)
                    if jx == 0:
                        a.set_ylabel(ttl, fontsize=7)
                    if i == 0:
                        a.set_title(f"frame {loc:%m-%d %H:%M}", fontsize=8)
                    a.set_xticks([]); a.set_yticks([]); a.set_aspect("equal")
                    if jx == len(sel) - 1:
                        fig.colorbar(sc, ax=a, shrink=0.8)
            fig.suptitle(f"{title} replay {ev['name']} {ev['day']} — F{L} (frame start, local time); labels = "
                         f"{', '.join(src)}{' — LATE labels, reported apart' if late else ''}\n{NOT_FLOOD}", fontsize=8)
            p = out / f"replay_{ev['name']}_{ev['day']}_F{L}.png"
            fig.savefig(p, dpi=90); plt.close(fig)
            paths.append(p)
            rows.append({"name": ev["name"], "day": ev["day"], "lead_min": L, "status": "ok", "late": late,
                         "times": len(sel), "file": p.name})
    return paths, pd.DataFrame(rows)


# ============================================================================ P4E.2 diagnosis
def diagnosis(F: dict, pair: list[str], thrs: list[float], leads: list[int], out: Path, title: str,
              label_min: float = 7.6) -> tuple[list[Path], pd.DataFrame]:
    """Distribution of P(>=t) (recalibrated and raw) for t in thrs on city pixels whose label >= label_min."""
    plt = _plt()
    qs = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    rows = []
    fig, ax = plt.subplots(len(thrs), len(leads), figsize=(4.2 * len(leads), 3.4 * len(thrs)),
                           constrained_layout=True, squeeze=False)
    edges = np.linspace(0, 1, 21)
    for i, t in enumerate(thrs):
        for jx, L in enumerate(leads):
            a = ax[i, jx]
            for m in pair:
                j = _get(F, m, t, L)
                if j is None or not len(j):
                    continue
                sel = j["label_rate_mm_h"].to_numpy("float64") >= label_min
                for kind in ("cal", "raw"):
                    v = j.loc[sel, f"p_{kind}"].to_numpy("float64")
                    rows.append({"threshold": t, "lead_min": L, "model": m, "prob": kind, "n": int(len(v)),
                                 "mean": float(v.mean()) if len(v) else np.nan,
                                 **{f"q{int(q * 100):02d}": float(np.quantile(v, q)) if len(v) else np.nan
                                    for q in qs}})
                v = j.loc[sel, "p_cal"].to_numpy("float64")
                if len(v):
                    a.hist(v, bins=edges, histtype="step", lw=1.4, color=_color(pair, m),
                           label=f"{m} (n {len(v)}, median {np.median(v):.2f})")
            a.set_xlim(0, 1)
            a.set_title(f"P(>={t:g}) recalibrated where label >= {label_min:g}, F{L}", fontsize=9)
            a.set_xlabel("probability", fontsize=8); a.set_ylabel("rows (city pixel x t0)", fontsize=8)
            a.legend(fontsize=7)
    fig.suptitle(f"{title} — P4E.2 diagnosis, city pixels with IMERG label >= {label_min:g} mm/h (descriptive only)"
                 f"\n{NOT_FLOOD}", fontsize=9)
    p = out / "diagnosis_p_on_heavy.png"
    fig.savefig(p, dpi=100); plt.close(fig)
    return [p], pd.DataFrame(rows)


def event_days(cfg: dict, F: dict, model: str, thr: float, lead: int) -> list[dict]:
    """Named events of evaluation.events (each local day) + the wettest day (city-mean label at `lead`, eval role) of
    every held-out season present in the frame. Descriptive only, never a selection criterion (hub 6)."""
    out = []
    for name, (a, b) in cfg["evaluation"]["events"].items():
        for d in pd.date_range(a, b, freq="D"):
            out.append({"name": name, "day": d.strftime("%Y-%m-%d")})
    j = _get(F, model, thr, lead)
    if j is not None and len(j):
        e = j[j["role"] == "eval"]
        for s, g in e.groupby("season_fold"):
            lab = g.groupby("day")["label_rate_mm_h"].mean()
            if len(lab):
                out.append({"name": f"wettest{int(s)}", "day": str(lab.idxmax())})
    return out
