"""U1.4 — reported rain amounts (press-quoted gauges) vs satellite accumulations.

For every catalog row with `rain_validation` in usable_for: accumulate IMERG Final, IMERG Late
(substitute after 2025-09-30, labelled), GSMaP NOW `rain_rate_mm_h` and `rain_rate_gc_mm_h` over
the stated period for the rain pixels of the unit; min / mean / max over pixels.

Period rules (flag `period_basis`):
  clock      — clock times given (both ends; several windows summed, e.g. 0h-6h + 8h-9h)
  end_only_24h — only the end time given → (end − 24 h, end]
  local_day  — only a date / part of day → local calendar day(s) [00:00, 24:00) Asia/Ho_Chi_Minh
Accumulation: IMERG half-hour rate × overlap hours; GSMaP NOW hourly windows issued every 30 min
overlap, so acc = Σ rate × overlap hours / 2 (every instant is covered by two windows).
Memory: Silver read one UTC-date partition at a time, only needed pixels and columns; records
processed month by month (user rule 2026-11-20: peak RSS < 0.4 GB).

Usage: .venv/Scripts/python.exe -m src.events.rain_validation
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from src.events.build_event_catalog import TZ, clock_windows
from src.events.common import CATALOG_DIR, REPORTS, ROOT, RamGuard, log_line, rel, write_json_atomic

MET = ROOT / "data/silver/met"
UTC = timezone.utc
PRODUCTS = {  # name: (silver source, value column, start col, end col, overlap divisor)
    "imerg_final": ("imerg_final", "rain_rate_mm_h", "interval_start_utc", "interval_end_utc", 1.0),
    "imerg_late": ("imerg_late", "rain_rate_mm_h", "interval_start_utc", "interval_end_utc", 1.0),
    "gsmap": ("gsmap_now", "rain_rate_mm_h", "window_start_utc", "window_end_utc", 2.0),
    "gsmap_gc": ("gsmap_now", "rain_rate_gc_mm_h", "window_start_utc", "window_end_utc", 2.0),
}
OUT_CSV = REPORTS / "rain_validation.csv"
OVERRIDES = ROOT / "configs/event_catalog_overrides.yaml"


# period bases of the primary ratio set (clock times stated; end time + 24 h rule; explicit noon-to-noon window)
PRIMARY_BASES = ("clock", "end_only_24h", "noon_to_noon_24h")


def load_overrides() -> tuple[dict, dict, dict]:
    """(duplicate_of, period_basis override, explicit window) by record id, from the user-approved overrides."""
    import yaml
    ov = yaml.safe_load(OVERRIDES.read_text(encoding="utf-8")) if OVERRIDES.exists() else {}
    dup = {k: v["duplicate_of"] for k, v in (ov.get("rain_validation_duplicates") or {}).items()}
    pb = {rid: basis for basis, blk in (ov.get("period_basis") or {}).items() for rid in blk.get("records", [])}
    pw = {}
    for basis, blk in (ov.get("period_windows") or {}).items():
        a = datetime.strptime(blk["start_local"], "%Y-%m-%d %H:%M").replace(tzinfo=TZ).astimezone(UTC)
        b = datetime.strptime(blk["end_local"], "%Y-%m-%d %H:%M").replace(tzinfo=TZ).astimezone(UTC)
        for rid in blk.get("records", []):
            pw[rid] = (basis, [(a, b)])
    return dup, pb, pw


def periods(row) -> tuple[list[tuple[datetime, datetime]], str]:
    """Accumulation windows (UTC) and the basis flag for one catalog row."""
    d0 = row["event_date_start"]
    wins = clock_windows(row["temporal_text"] or "", d0) if d0 is not None else []
    ts, te = row["time_start_utc"], row["time_end_utc"]
    if wins and len(wins) > 1:
        return [(a.astimezone(UTC), b.astimezone(UTC)) for a, b in wins], "clock"
    if ts is not None and te is not None and not _isnat(ts) and not _isnat(te):
        return [(ts.to_pydatetime(), te.to_pydatetime())], "clock"
    if te is not None and not _isnat(te):
        e = te.to_pydatetime()
        return [(e - timedelta(hours=24), e)], "end_only_24h"
    d1 = row["event_date_end"] or d0
    a = datetime(d0.year, d0.month, d0.day, tzinfo=TZ)
    b = datetime(d1.year, d1.month, d1.day, tzinfo=TZ) + timedelta(days=1)
    return [(a.astimezone(UTC), b.astimezone(UTC))], "local_day"


def _isnat(x) -> bool:
    try:
        import pandas as pd
        return pd.isna(x)
    except Exception:
        return x is None


def read_partition(source: str, day, pixels: list[int], cols: list[str]):
    import pyarrow.parquet as pq
    p = MET / source / f"date={day.isoformat()}" / "part.parquet"
    if not p.exists():
        return None
    return pq.read_table(p, columns=["pixel_id"] + cols, filters=[("pixel_id", "in", pixels)]).to_pandas()


def accumulate(product: str, wins: list[tuple[datetime, datetime]], pixels: list[int]) -> dict:
    source, val, c0, c1, div = PRODUCTS[product]
    days = sorted({(a + timedelta(days=k)).date() for a, b in wins
                   for k in range(-1, (b - a).days + 2)})
    acc = defaultdict(float)
    covered = defaultdict(float)
    for day in days:
        df = read_partition(source, day, pixels, [c0, c1, val])
        if df is None or df.empty:
            continue
        s = df[c0].dt.to_pydatetime()
        e = df[c1].dt.to_pydatetime()
        v = df[val].to_numpy()
        pid = df["pixel_id"].to_numpy()
        for a, b in wins:
            for i in range(len(df)):
                ov = (min(e[i], b) - max(s[i], a)).total_seconds() / 3600
                if ov <= 0 or np.isnan(v[i]):
                    continue
                acc[pid[i]] += v[i] * ov / div
                covered[pid[i]] += ov / div
    total_h = sum((b - a).total_seconds() / 3600 for a, b in wins)
    vals = [acc[p] for p in pixels if covered[p] > 0]
    cov = min((covered[p] for p in pixels), default=0.0) / total_h if total_h else 0.0
    if not vals:
        return {"min": None, "mean": None, "max": None, "coverage": 0.0}
    return {"min": float(np.min(vals)), "mean": float(np.mean(vals)), "max": float(np.max(vals)),
            "coverage": round(float(cov), 3)}


def run() -> dict:
    import pandas as pd
    cat = pd.read_parquet(CATALOG_DIR / "events.parquet")
    sel = cat[cat["usable_for"].map(lambda u: "rain_validation" in list(u))].copy()
    sel["month"] = sel["event_date_start"].map(lambda d: d.strftime("%Y-%m"))
    guard = RamGuard()
    dup_of, basis_override, windows = load_overrides()
    rows, skipped_months = [], []
    for month, grp in sel.groupby("month", sort=True):
        if not guard.wait_ok():
            skipped_months.append(month)
            log_line("u1_4", f"RAM low for 5 min, month {month} skipped")
            continue
        for _, r in grp.iterrows():
            wins, basis = periods(r)
            # override (user 2026-11-20): only the time of the statement is known → window kept for sensitivity
            basis = basis_override.get(r["silver_record_id"], basis)
            if r["silver_record_id"] in windows:  # explicit window from a part-of-day period (overrides yaml)
                basis, wins = windows[r["silver_record_id"]]
            pix = [int(p) for p in r["rain_pixel_ids"]]
            out = {"silver_record_id": r["silver_record_id"], "source_name": r["source_name"],
                   "event_date": r["event_date_start"].isoformat(), "location_text": r["location_text"],
                   "location_level": r["location_level"], "feature_name": r["feature_name"],
                   "n_pixels": len(pix), "rain_period_text": r["rain_period_text"],
                   "period_basis": basis, "time_precision": r["time_precision"],
                   "period_local": "; ".join(f"{a.astimezone(TZ):%Y-%m-%d %H:%M}–{b.astimezone(TZ):%Y-%m-%d %H:%M}"
                                             for a, b in wins),
                   "period_hours": round(sum((b - a).total_seconds() for a, b in wins) / 3600, 2),
                   "reported_mm": r["rainfall_mm"], "qualifier": r["rainfall_qualifier"],
                   "confidence": r["confidence"], "duplicate_of": dup_of.get(r["silver_record_id"])}
            for prod in PRODUCTS:
                a = accumulate(prod, wins, pix)
                for k, v in a.items():
                    out[f"{prod}_{k}"] = None if v is None else round(v, 2)
            # label product for the ratio: IMERG Final, else Late after its end (labelled)
            out["imerg_ref"] = "final" if out["imerg_final_mean"] is not None and out["imerg_final_coverage"] >= 0.9 else (
                "late" if out["imerg_late_mean"] is not None else None)
            for prod in ("imerg_ref", "gsmap", "gsmap_gc"):
                if prod == "imerg_ref":
                    m = out.get(f"imerg_{out['imerg_ref']}_mean") if out["imerg_ref"] else None
                else:
                    m = out.get(f"{prod}_mean")
                out[f"ratio_{prod}"] = round(m / out["reported_mm"], 3) if m is not None and out["reported_mm"] else None
                out[f"absdiff_{prod}"] = round(m - out["reported_mm"], 1) if m is not None else None
            rows.append(out)
            guard.check()
        log_line("u1_4", f"month {month}: {len(grp)} records")
    df = pd.DataFrame(rows).sort_values(["event_date", "source_name", "location_text"])
    tmp = OUT_CSV.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False, encoding="utf-8")
    tmp.replace(OUT_CSV)
    return {"records": len(df), "skipped_months": skipped_months, "peak_rss_gb": round(guard.peak_rss, 3),
            "min_free_gb": round(guard.min_free, 3), "df": df}


def plot(df) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted = "#0b0b0b", "#52514e"
    series = [("imerg_ref", "IMERG (Final; Late sau 30/9/2025)", "#2a78d6", "o"),
              ("gsmap", "GSMaP NOW", "#eb6834", "s"), ("gsmap_gc", "GSMaP NOW GC", "#1baf7a", "^")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=130, sharex=True, sharey=True)
    fig.patch.set_facecolor("#fcfcfb")
    lim = max(600.0, float(df["reported_mm"].max()) * 1.05)
    prim = df["period_basis"].isin(PRIMARY_BASES)
    panels = [(df[prim], "Kỳ có giờ (clock, end_only_24h, noon_to_noon_24h)"),
              (df[~prim], "Chỉ có ngày (cả ngày ĐP) hoặc thiếu đầu kỳ (end_only_unknown_start) — gắn cờ")]
    for ax, (sub, title) in zip(axes, panels):
        ax.set_facecolor("#fcfcfb")
        ax.plot([0, lim], [0, lim], color=muted, lw=1, ls="--", zorder=1)
        ax.text(lim * 0.97, lim * 0.93, "1:1", color=muted, ha="right", fontsize=9)
        for key, label, col, mk in series:
            if key == "imerg_ref":
                f = lambda r, q: r.get(f"imerg_{r['imerg_ref']}_{q}") if isinstance(r["imerg_ref"], str) else np.nan
                mean = sub.apply(lambda r: f(r, "mean"), axis=1)
                lo = sub.apply(lambda r: f(r, "min"), axis=1)
                hi = sub.apply(lambda r: f(r, "max"), axis=1)
            else:
                mean, lo, hi = sub[f"{key}_mean"], sub[f"{key}_min"], sub[f"{key}_max"]
            x = sub["reported_mm"].astype(float)
            ok = mean.notna()
            ax.errorbar(x[ok], mean[ok].astype(float),
                        yerr=[(mean - lo)[ok].astype(float), (hi - mean)[ok].astype(float)],
                        fmt=mk, ms=6, color=col, ecolor=col, elinewidth=1, alpha=0.9, label=label,
                        markeredgecolor="#fcfcfb", markeredgewidth=1, zorder=3)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_title(f"{title} — n = {len(sub)}", color=ink, fontsize=10, loc="left")
        ax.set_xlabel("Lượng mưa báo chí trích dẫn (mm, trạm đo)", color=ink)
        ax.grid(color="#e6e5e1", lw=0.6)
        for sp in ax.spines.values():
            sp.set_color("#c9c8c3")
        ax.tick_params(colors=muted)
    axes[0].set_ylabel("Tích lũy vệ tinh, trung bình pixel 0,1° (mm); thanh = min–max", color=ink)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("U1.4 — Mưa báo cáo so với vệ tinh (IMERG Late thay Final sau 30/9/2025)",
                 color=ink, fontsize=11, x=0.01, ha="left")
    out = REPORTS / "rain_validation_scatter.png"
    fig.tight_layout()
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def summarise(df) -> dict:
    ev = {}
    for d, g in df.groupby("event_date"):
        e = {"n": int(len(g))}
        for p in ("imerg_ref", "gsmap", "gsmap_gc"):
            r = g[f"ratio_{p}"].dropna()
            e[f"median_ratio_{p}"] = round(float(r.median()), 3) if len(r) else None
        e["imerg_products"] = sorted(set(g["imerg_ref"].dropna()))
        ev[d] = e
    return ev


def main() -> int:
    res = run()
    df = res.pop("df")
    png = plot(df)
    ev = summarise(df)
    df.groupby("event_date")  # noqa
    import pandas as pd
    pd.DataFrame([{"event_date": k, **v} for k, v in ev.items()]).to_csv(
        REPORTS / "rain_validation_by_event.csv", index=False, encoding="utf-8")
    write_json_atomic(REPORTS / "rain_validation_summary.json", {**res, "by_event": ev, "plot": rel(png)})
    print(json.dumps({**res, "plot": rel(png)}, ensure_ascii=False))
    for k, v in ev.items():
        print(k, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
