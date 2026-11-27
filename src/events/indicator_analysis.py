"""U1.7 analysis — gauge vs satellite factor, event check, frequency (alarm load), plots.

Stages (each resumable; `--stage all` runs them in order):
  series   per product × month: half-hour rain (mm) for the 47 city pixels + A6h/A24h, cached in
           data/_scratch/u1/acc/<product>/<YYYY-MM>.parquet (skip when present). One Silver UTC-date
           partition at a time, only city pixels; 24 h warm-up from the previous day.
  ratio    satellite / reported from reports/u1/rain_validation.csv → median, IQR per product;
           proposed satellite thresholds = round5(a × median), round5(b × median), round5(150 × median)
           → configs/flood_indicator.yaml `satellite_proposal` (status NEEDS_REVIEW).
  events   indicator level per city pixel at the first reported flood time (else local-day max, flagged)
           for event days with flood evidence, and the wettest days without evidence
           ("chưa thu thập bằng chứng", never "no flood").
  freq     share of pixel-hours and of local days at each level, both threshold sets.
  plots    A6h at reported flood time vs level bands; satellite vs reported with 1:1 and fitted ratio.

GSMaP NOW hourly windows are issued every 30 min: half-hour slot [t, t+30) gets
0.5 h × (r(window t−30) + r(window t)) / 2. IMERG: rate × 0.5 h. Missing input → NaN (strict).

Usage: .venv/Scripts/python.exe -m src.events.indicator_analysis --stage all
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import date, datetime, timedelta, timezone

import numpy as np

from src.events.common import (CATALOG_DIR, REPORTS, ROOT, SCRATCH, RamGuard, log_line, rel,
                               write_json_atomic)
from src.events.flood_indicator import CONFIG, indicator_levels, load_config

UTC = timezone.utc
MET = ROOT / "data/silver/met"
ACC = SCRATCH / "acc"
PRODUCTS = {"imerg_final": ("imerg_final", "rain_rate_mm_h"), "imerg_late": ("imerg_late", "rain_rate_mm_h"),
            "gsmap": ("gsmap_now", "rain_rate_mm_h"), "gsmap_gc": ("gsmap_now", "rain_rate_gc_mm_h")}
RATIO_PRODUCTS = {"imerg_final": "IMERG Final V07", "gsmap": "GSMaP NOW", "gsmap_gc": "GSMaP NOW GC"}
MONTHS = [f"{y}-{m:02d}" for y in (2023, 2024, 2025) for m in range(5, 11)]
LEVEL_NAMES = {0: "none", 1: "possible", 2: "high", -1: "no_data"}


def round5(x: float) -> float:
    return float(5 * round(x / 5))


# ------------------------------------------------------------------------------------------ series

def city_pixels() -> list[int]:
    import pandas as pd
    px = pd.read_parquet(ROOT / "data/reference/rain_pixels_hanoi_city.parquet")
    return sorted(int(p) for p in px.loc[px["is_city_pixel"], "pixel_id"])


def _read_day(product: str, day: date, pixels: list[int]):
    import pyarrow.parquet as pq
    src, col = PRODUCTS[product]
    p = MET / src / f"date={day.isoformat()}" / "part.parquet"
    if not p.exists():
        return None
    start = "interval_start_utc" if src.startswith("imerg") else "window_start_utc"
    t = pq.read_table(p, columns=["pixel_id", start, col], filters=[("pixel_id", "in", pixels)]).to_pandas()
    return t.rename(columns={start: "start", col: "rate"})


def build_month(product: str, month: str, pixels: list[int]) -> str:
    import pandas as pd
    out = ACC / product / f"{month}.parquet"
    if out.exists():
        return "skip"
    y, m = map(int, month.split("-"))
    first = date(y, m, 1)
    nxt = date(y + (m == 12), m % 12 + 1, 1)
    days = [first - timedelta(days=1) + timedelta(days=k) for k in range((nxt - first).days + 1)]
    frames = [f for f in (_read_day(product, d, pixels) for d in days) if f is not None and len(f)]
    if not frames or all(f["start"].min().date() < first for f in frames):
        return "no_data"
    df = pd.concat(frames, ignore_index=True)
    t0 = pd.Timestamp(datetime(first.year, first.month, first.day, tzinfo=UTC)) - pd.Timedelta(days=1)
    t1 = pd.Timestamp(datetime(nxt.year, nxt.month, nxt.day, tzinfo=UTC))
    grid = pd.date_range(t0, t1 - pd.Timedelta(minutes=30), freq="30min")  # slot starts
    piv = df.pivot_table(index="start", columns="pixel_id", values="rate", aggfunc="mean").reindex(grid)
    piv = piv.reindex(columns=pixels)
    if PRODUCTS[product][0].startswith("imerg"):
        mm = piv * 0.5
    else:  # slot [t, t+30) ← windows starting t−30 and t
        mm = 0.25 * (piv.shift(1) + piv)
    mm.index = mm.index + pd.Timedelta(minutes=30)  # label by slot end
    a6 = mm.rolling(12, min_periods=12).sum()
    a24 = mm.rolling(48, min_periods=48).sum()
    keep = mm.index > pd.Timestamp(datetime(first.year, first.month, first.day, tzinfo=UTC))
    long = pd.DataFrame({
        "slot_end_utc": np.repeat(mm.index[keep], len(pixels)),
        "pixel_id": np.tile(np.array(pixels, dtype=np.int64), int(keep.sum())),
        "mm": mm[keep].to_numpy(dtype=np.float32).ravel(),
        "a6h": a6[keep].to_numpy(dtype=np.float32).ravel(),
        "a24h": a24[keep].to_numpy(dtype=np.float32).ravel()})
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.tmp")
    long.to_parquet(tmp, index=False)
    tmp.replace(out)
    return "built"


def stage_series() -> dict:
    pixels = city_pixels()
    guard = RamGuard()
    res = {}
    for product in PRODUCTS:
        for month in MONTHS:
            if not guard.wait_ok():
                res[f"{product}/{month}"] = "NEEDS_REVIEW_low_ram"
                continue
            res[f"{product}/{month}"] = build_month(product, month, pixels)
            guard.check()
    log_line("u1_7", f"series {json.dumps(res)} peak_rss={guard.peak_rss:.3f}")
    return {"status": res, "peak_rss_gb": round(guard.peak_rss, 3), "min_free_gb": round(guard.min_free, 3)}


def load_acc(product: str, months: list[str] | None = None, cols=("slot_end_utc", "pixel_id", "a6h", "a24h")):
    import pandas as pd
    parts = [ACC / product / f"{m}.parquet" for m in (months or MONTHS)]
    parts = [p for p in parts if p.exists()]
    if not parts:
        return None
    return pd.concat([pd.read_parquet(p, columns=list(cols)) for p in parts], ignore_index=True)


# ------------------------------------------------------------------------------------------- ratio

def stage_ratio() -> dict:
    import pandas as pd
    from src.events.rain_validation import PRIMARY_BASES
    rv = pd.read_csv(REPORTS / "rain_validation.csv")
    if "duplicate_of" in rv.columns:  # user-approved duplicates (configs/event_catalog_overrides.yaml)
        rv = rv[rv["duplicate_of"].isna()]
    rv["dedup_key"] = rv["feature_name"].astype(str) + "|" + rv["period_local"] + "|" + rv["reported_mm"].astype(str)
    sets = {"primary": rv[rv["period_basis"].isin(PRIMARY_BASES)].drop_duplicates("dedup_key"),
            "primary_eq_only": rv[rv["period_basis"].isin(PRIMARY_BASES)
                                  & rv["qualifier"].isin(["eq", "approx"])].drop_duplicates("dedup_key"),
            "all_incl_local_day": rv.drop_duplicates("dedup_key")}
    rows = []
    for set_name, d in sets.items():
        for prod in list(RATIO_PRODUCTS) + ["imerg_late"]:
            for stat in ("min", "mean", "max"):
                r = (d[f"{prod}_{stat}"] / d["reported_mm"]).dropna()
                if prod == "imerg_final":
                    r = r[d.loc[r.index, "imerg_final_coverage"] >= 0.9]
                if not len(r):
                    continue
                rows.append({"record_set": set_name, "product": prod, "pixel_stat": stat, "n": int(len(r)),
                             "median": round(float(r.median()), 3), "q25": round(float(r.quantile(0.25)), 3),
                             "q75": round(float(r.quantile(0.75)), 3)})
    tab = pd.DataFrame(rows)
    tab.to_csv(REPORTS / "indicator_ratio_table.csv", index=False, encoding="utf-8")
    prim = tab[(tab["record_set"] == "primary") & (tab["pixel_stat"] == "mean")].set_index("product")
    cfg = load_config()
    a, b, lm = float(cfg["a_mm"]), float(cfg["b_mm"]), float(cfg["long_rule"]["mm"])
    proposal = {}
    for prod, label in RATIO_PRODUCTS.items():
        med = float(prim.loc[prod, "median"])
        proposal[prod] = {"label": label, "n_records": int(prim.loc[prod, "n"]), "median_ratio": med,
                          "iqr_ratio": [float(prim.loc[prod, "q25"]), float(prim.loc[prod, "q75"])],
                          "proposed_satellite_a_mm": round5(a * med), "proposed_satellite_b_mm": round5(b * med),
                          "proposed_satellite_long_mm": round5(lm * med)}
    if (cfg.get("satellite_proposal") or {}).get("status", "NEEDS_REVIEW") != "NEEDS_REVIEW":
        # user decided (U1-D1, 2026-11-20): never overwrite the yaml; tables only
        return {"proposal": proposal, "yaml_written": False, "table": rel(REPORTS / "indicator_ratio_table.csv")}
    write_proposal(proposal)
    return {"proposal": proposal, "table": rel(REPORTS / "indicator_ratio_table.csv")}


def write_proposal(proposal: dict) -> None:
    """Append / replace the `satellite_proposal` block; user-decided keys are left untouched."""
    text = CONFIG.read_text(encoding="utf-8")
    text = re.split(r"\n# --- satellite_proposal \(agent, U1\.7\)", text)[0].rstrip("\n") + "\n"
    lines = ["", "# --- satellite_proposal (agent, U1.7) — NEEDS_REVIEW; the user decides. Ratio = satellite pixel-mean",
             "# accumulation / reported gauge amount over the stated period (records with clock period or end time,",
             "# duplicates removed; reports/u1/indicator_ratio_table.csv). Proposed = round5(threshold × median ratio).",
             "satellite_proposal:", "  status: NEEDS_REVIEW", f"  computed: {date.today().isoformat()}",
             "  record_set: primary (period_basis clock or end_only_24h, dedup)", "  products:"]
    for prod, p in proposal.items():
        lines.append(f"    {prod}: {{label: \"{p['label']}\", n_records: {p['n_records']}, median_ratio: {p['median_ratio']}, "
                     f"iqr_ratio: [{p['iqr_ratio'][0]}, {p['iqr_ratio'][1]}], proposed_satellite_a_mm: {p['proposed_satellite_a_mm']:g}, "
                     f"proposed_satellite_b_mm: {p['proposed_satellite_b_mm']:g}, proposed_satellite_long_mm: {p['proposed_satellite_long_mm']:g}}}")
    tmp = CONFIG.with_suffix(".yaml.tmp")
    tmp.write_text(text + "\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(CONFIG)


def threshold_sets(cfg: dict, product: str) -> dict[str, tuple[float, float, float]]:
    sets = {"gauge": (float(cfg["a_mm"]), float(cfg["b_mm"]), float(cfg["long_rule"]["mm"]))}
    ref = "imerg_final" if product.startswith("imerg") else product
    p = cfg.get("satellite_proposal", {}).get("products", {}).get(ref)
    if p:
        sets["satellite_proposed"] = (p["proposed_satellite_a_mm"], p["proposed_satellite_b_mm"],
                                      p["proposed_satellite_long_mm"])
    return sets


# ------------------------------------------------------------------------------------------ events

def first_flood_time(cat, day: date):
    """Earliest stated flood time (UTC) among flood records of the local day, else None."""
    import pandas as pd
    f = cat[(cat["event_type"].isin(["flood_observation", "flood_impact"])) & (cat["flood_cause"] == "rain")
            & (cat["event_date_start"] <= day) & (cat["event_date_end"].fillna(cat["event_date_start"]) >= day)
            & (cat["time_precision"] == "timestamp")]
    t = pd.concat([f["time_start_utc"].dropna(), f["time_end_utc"].dropna()])
    t = t[t.map(lambda x: x.tz_convert("Asia/Ho_Chi_Minh").date() == day)]
    return None if t.empty else t.min()


def stage_events() -> dict:
    import pandas as pd
    cfg = load_config()
    cat = pd.read_parquet(CATALOG_DIR / "events.parquet")
    ed = pd.read_parquet(CATALOG_DIR / "event_days.parquet")
    ed["has_flood_evidence"] = (ed["n_flood_observation"] + ed["n_flood_impact"]) > 0
    rows = []
    for _, d in ed.iterrows():
        day = d["date_local"]
        if not d["has_flood_evidence"] and d["n_evidence_records"] > 0:
            continue  # days with only rain/context evidence: not part of either table
        label = "có bằng chứng ngập" if d["has_flood_evidence"] else "chưa thu thập bằng chứng"
        t_first = first_flood_time(cat, day) if d["has_flood_evidence"] else None
        lo = pd.Timestamp(datetime(day.year, day.month, day.day), tz="Asia/Ho_Chi_Minh").tz_convert("UTC")
        hi = lo + pd.Timedelta(days=1)
        month = lo.tz_convert("Asia/Ho_Chi_Minh").strftime("%Y-%m")
        months = sorted({month, (lo - pd.Timedelta(hours=1)).strftime("%Y-%m"), (hi - pd.Timedelta(minutes=1)).strftime("%Y-%m")})
        for product in ("imerg_final", "imerg_late", "gsmap", "gsmap_gc"):
            acc = load_acc(product, months)
            if acc is None:
                continue
            day_acc = acc[(acc["slot_end_utc"] > lo) & (acc["slot_end_utc"] <= hi)]
            if day_acc.empty or day_acc["a6h"].notna().sum() == 0:
                continue
            if t_first is not None:
                slot = t_first.floor("30min")
                snap = day_acc[day_acc["slot_end_utc"] == slot]
                basis = f"first reported flood {t_first.tz_convert('Asia/Ho_Chi_Minh'):%H:%M} (slot end {slot.tz_convert('Asia/Ho_Chi_Minh'):%H:%M})"
                a6, a24 = snap["a6h"].to_numpy(), snap["a24h"].to_numpy()
            else:
                g = day_acc.groupby("pixel_id")[["a6h", "a24h"]].max()
                basis = "local-day maximum (no stated flood time, flagged)"
                a6, a24 = g["a6h"].to_numpy(), g["a24h"].to_numpy()
            for set_name, (a, b, lm) in threshold_sets(cfg, product).items():
                codes = indicator_levels(a6, a24, a, b, lm)
                rows.append({"date_local": day, "evidence": label, "product": product, "threshold_set": set_name,
                             "a_mm": a, "b_mm": b, "long_mm": lm, "time_basis": basis, "n_pixels": int(len(codes)),
                             "n_none": int((codes == 0).sum()), "n_possible": int((codes == 1).sum()),
                             "n_high": int((codes == 2).sum()), "n_no_data": int((codes == -1).sum()),
                             "max_a6h_mm": round(float(np.nanmax(a6)), 1) if np.isfinite(a6).any() else None,
                             "median_a6h_mm": round(float(np.nanmedian(a6)), 1) if np.isfinite(a6).any() else None,
                             "max_a24h_mm": round(float(np.nanmax(a24)), 1) if np.isfinite(a24).any() else None})
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS / "indicator_event_check.csv", index=False, encoding="utf-8")
    return {"rows": int(len(df)), "days_with_flood_evidence": int(ed["has_flood_evidence"].sum()),
            "days_without_evidence": int((ed["n_evidence_records"] == 0).sum())}


# -------------------------------------------------------------------------------------------- freq

def stage_freq() -> dict:
    import pandas as pd
    cfg = load_config()
    guard = RamGuard()
    rows = []
    for product in PRODUCTS:
        sets = threshold_sets(cfg, product)
        tot = {s: np.zeros(4, dtype=np.int64) for s in sets}
        days = {s: {} for s in sets}
        n_hours = 0
        for month in MONTHS:
            if not guard.wait_ok():
                log_line("u1_7", f"freq {product} {month} skipped (low RAM)")
                continue
            acc = load_acc(product, [month])
            if acc is None:
                continue
            acc = acc[acc["slot_end_utc"].dt.minute == 0]  # pixel-hours
            acc = acc[acc["a6h"].notna() | acc["a24h"].notna()]
            local_day = acc["slot_end_utc"].dt.tz_convert("Asia/Ho_Chi_Minh").dt.date
            n_hours += len(acc)
            for s, (a, b, lm) in sets.items():
                codes = indicator_levels(acc["a6h"].to_numpy(), acc["a24h"].to_numpy(), a, b, lm)
                for k, c in enumerate((0, 1, 2, -1)):
                    tot[s][k] += int((codes == c).sum())
                dm = pd.Series(codes).groupby(local_day.to_numpy()).max()
                for dd, v in dm.items():
                    days[s][dd] = max(days[s].get(dd, -1), int(v))
            guard.check()
        for s in sets:
            nd = len(days[s])
            dv = np.array(list(days[s].values())) if nd else np.array([])
            ph = tot[s][:3].sum()
            a, b, lm = sets[s]
            rows.append({"product": product, "threshold_set": s, "a_mm": a, "b_mm": b, "long_mm": lm,
                         "pixel_hours": int(ph), "share_ph_possible": round(tot[s][1] / ph, 5) if ph else None,
                         "share_ph_high": round(tot[s][2] / ph, 5) if ph else None,
                         "share_ph_any_icon": round((tot[s][1] + tot[s][2]) / ph, 5) if ph else None,
                         "days": nd, "days_with_possible_or_high": int((dv >= 1).sum()),
                         "days_with_high": int((dv == 2).sum()),
                         "share_days_any_icon": round(float((dv >= 1).mean()), 4) if nd else None,
                         "share_days_high": round(float((dv == 2).mean()), 4) if nd else None})
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS / "indicator_frequency.csv", index=False, encoding="utf-8")
    return {"rows": int(len(df)), "peak_rss_gb": round(guard.peak_rss, 3)}


# ------------------------------------------------------------------------------------------- plots

def stage_plots() -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    cfg = load_config()
    ink, muted, surf = "#0b0b0b", "#52514e", "#fcfcfb"
    cols = {"imerg_final": "#2a78d6", "gsmap": "#eb6834", "gsmap_gc": "#1baf7a"}
    marks = {"imerg_final": "o", "gsmap": "s", "gsmap_gc": "^"}
    # (1) A6h at reported flood time vs level bands
    ev = pd.read_csv(REPORTS / "indicator_event_check.csv")
    ev = ev[(ev["threshold_set"] == "gauge") & ev["product"].isin(cols)]
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=130)
    fig.patch.set_facecolor(surf)
    ax.set_facecolor(surf)
    days = sorted(ev["date_local"].unique())
    ax.axhspan(cfg["a_mm"], cfg["b_mm"], color="#eda100", alpha=0.12, lw=0)
    ax.axhspan(cfg["b_mm"], 1000, color="#e34948", alpha=0.10, lw=0)
    ax.text(len(days) - 0.4, cfg["a_mm"] + 2, "a = 50 mm (thang trạm đo)", color=muted, fontsize=8, ha="right")
    ax.text(len(days) - 0.4, cfg["b_mm"] + 2, "b = 100 mm", color=muted, fontsize=8, ha="right")
    for j, (prod, c) in enumerate(cols.items()):
        sub = ev[ev["product"] == prod].set_index("date_local")
        x = [i + (j - 1) * 0.22 for i, d in enumerate(days) if d in sub.index]
        y = [sub.loc[d, "max_a6h_mm"] for d in days if d in sub.index]
        ym = [sub.loc[d, "median_a6h_mm"] for d in days if d in sub.index]
        ax.scatter(x, y, color=c, marker=marks[prod], s=42, label=f"{RATIO_PRODUCTS[prod]} — max pixel",
                   edgecolor=surf, linewidth=1, zorder=3)
        ax.scatter(x, ym, facecolor="none", edgecolor=c, marker=marks[prod], s=34, linewidth=1.2,
                   label=f"{RATIO_PRODUCTS[prod]} — trung vị pixel", zorder=3)
        p = cfg.get("satellite_proposal", {}).get("products", {}).get(prod)
        if p:
            ax.axhline(p["proposed_satellite_a_mm"], color=c, lw=1, ls=":", zorder=2)
    lab = []
    for d in days:
        e = ev[ev["date_local"] == d]["evidence"].iloc[0]
        lab.append(f"{d}\n{'bằng chứng' if e.startswith('có') else 'chưa thu thập'}")
    ax.set_xticks(range(len(days)))
    ax.set_xticklabels(lab, rotation=90, fontsize=7, color=muted)
    ax.set_ylim(0, max(260, float(ev["max_a6h_mm"].max()) * 1.08))
    ax.set_xlim(-0.6, len(days) - 0.4)
    ax.set_ylabel("A6h vệ tinh (mm) tại giờ ngập đầu tiên / max trong ngày", color=ink)
    ax.set_title("U1.7 — A6h so với dải mức chỉ báo (vàng: có thể ngập, đỏ: nguy cơ cao); "
                 "chấm: ngưỡng a đề xuất theo sản phẩm", fontsize=10, loc="left", color=ink)
    ax.grid(axis="y", color="#e6e5e1", lw=0.6)
    for s in ax.spines.values():
        s.set_color("#c9c8c3")
    ax.tick_params(colors=muted)
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper left")
    fig.tight_layout()
    p1 = REPORTS / "indicator_a6h_event_days.png"
    fig.savefig(p1, facecolor=surf)
    plt.close(fig)
    # (2) satellite vs reported, 1:1 and fitted ratio, one panel per product
    rv = pd.read_csv(REPORTS / "rain_validation.csv")
    from src.events.rain_validation import PRIMARY_BASES
    rv = rv[rv["period_basis"].isin(PRIMARY_BASES)]
    if "duplicate_of" in rv.columns:
        rv = rv[rv["duplicate_of"].isna()]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), dpi=130, sharey=True)
    fig.patch.set_facecolor(surf)
    lim = max(550.0, float(rv["reported_mm"].max()) * 1.05)
    for ax, (prod, c) in zip(axes, cols.items()):
        ax.set_facecolor(surf)
        sub = rv[rv[f"{prod}_mean"].notna()]
        if prod == "imerg_final":
            sub = sub[sub["imerg_final_coverage"] >= 0.9]
        ax.plot([0, lim], [0, lim], color=muted, ls="--", lw=1)
        med = cfg["satellite_proposal"]["products"][prod]["median_ratio"]
        ax.plot([0, lim], [0, lim * med], color=c, lw=2)
        ax.text(lim * 0.97, lim * med + 8, f"tỷ số trung vị {med:.2f}", color=muted, fontsize=8, ha="right")
        ax.scatter(sub["reported_mm"], sub[f"{prod}_mean"], color=c, marker=marks[prod], s=34,
                   edgecolor=surf, linewidth=1, zorder=3)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_title(f"{RATIO_PRODUCTS[prod]} (n = {len(sub)})", fontsize=10, loc="left", color=ink)
        ax.set_xlabel("Báo cáo (mm, trạm đo)", color=ink)
        ax.grid(color="#e6e5e1", lw=0.6)
        for s in ax.spines.values():
            s.set_color("#c9c8c3")
        ax.tick_params(colors=muted)
    axes[0].set_ylabel("Vệ tinh, TB pixel (mm)", color=ink)
    fig.suptitle("U1.7 — Vệ tinh so với báo cáo, đường 1:1 (đứt) và tỷ số trung vị (liền); kỳ có giờ",
                 fontsize=10, x=0.01, ha="left", color=ink)
    fig.tight_layout()
    p2 = REPORTS / "indicator_sat_vs_reported.png"
    fig.savefig(p2, facecolor=surf)
    plt.close(fig)
    return {"plots": [rel(p1), rel(p2)]}


STAGES = {"series": stage_series, "ratio": stage_ratio, "events": stage_events, "freq": stage_freq,
          "plots": stage_plots}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=list(STAGES) + ["all"], default="all")
    a = ap.parse_args()
    out = {}
    for name in (STAGES if a.stage == "all" else [a.stage]):
        out[name] = STAGES[name]()
        print(name, json.dumps(out[name], ensure_ascii=False, default=str)[:900])
    write_json_atomic(REPORTS / f"indicator_{a.stage}_result.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
