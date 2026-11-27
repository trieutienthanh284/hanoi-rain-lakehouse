"""U1 follow-up 3 (user decisions 2026-11-20, REPORT_U1 §16).

Stages:
  reference  reference set = rain-driven evidence days where the IMERG reference at gauge scale 50/100/150
             shows >= 1 city pixel >= yellow at the first-flood slot (day max, flagged, if no time).
             Reference product: IMERG Final when the slot lies inside Final (<= 2025-09-30 UTC), else Late.
  sweep      GSMaP 2-D sweep (gsmap, gsmap_gc): a in 50..150 step 10, b = 2a, A24h threshold L in
             {150, 180, ..., 330} independent of a. Loads on 2024–2025 (calibration) and 2023–2025.
             Reference days caught, missed list, 2024-09-07 12:00 level, river days separately.
  rank       settings with icon days <= 15 % and red days <= 5 % (2024–2025); rank by reference days caught,
             then lower icon-day load; top 3 per product; Pareto table (load vs days caught); one plot.
  yaml       add the top 3 per product to `gsmap_now_thresholds` (status NEEDS_REVIEW) and fix the two header
             comment lines. Everything else in the file must stay byte-identical (asserted).

Reads only the A6h/A24h month caches of U1.7 (data/_scratch/u1/acc) and data/gold/event_catalog/.
Writes only reports/u1/followup3/ and configs/flood_indicator.yaml (approved, CLAUDE.md §8 follow-up 3).
Usage: .venv/Scripts/python.exe -m src.events.u1_followup3 --stage {reference|sweep|rank|yaml|all}
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import date, datetime

import numpy as np

from src.events.common import REPORTS, ROOT, RamGuard, rel, write_json_atomic

OUT = REPORTS / "followup3"
CONFIG = ROOT / "configs/flood_indicator.yaml"
FINAL_END_UTC = "2025-10-01T00:00:00+00:00"   # IMERG Final V07 ends 2025-09-30 (D92)
GAUGE = (50.0, 100.0, 150.0)
A_VALUES = list(range(50, 160, 10))
L_VALUES = [150, 180, 210, 240, 270, 300, 330]
PRODUCTS = ("gsmap", "gsmap_gc")
MAX_ICON, MAX_RED = 0.15, 0.05
HEADER_OLD = ("# The agent never changes a, b or the long rule. Satellite-scale proposals live in `satellite_proposal`\n"
              "# (status NEEDS_REVIEW, written by src/events/indicator_analysis.py; the user decides).\n")
HEADER_NEW = ("# The agent never changes a, b or the long rule. `satellite_proposal` (ratio-based, U1.7) is REJECTED\n"
              "# (user decision U1-D1, 2026-11-20); GSMaP proposals live in `gsmap_now_thresholds` (NEEDS_REVIEW).\n")


def _w(path, rows):
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def slot_values(product: str, day: date, slot):
    """A6h, A24h of the 47 city pixels at `slot` (UTC slot end) or the local-day max if slot is None."""
    import pandas as pd
    from src.events.indicator_analysis import city_pixels, load_acc
    lo = pd.Timestamp(datetime(day.year, day.month, day.day), tz="Asia/Ho_Chi_Minh").tz_convert("UTC")
    hi = lo + pd.Timedelta(days=1)
    months = sorted({lo.strftime("%Y-%m"), (hi - pd.Timedelta(minutes=1)).strftime("%Y-%m"),
                     lo.tz_convert("Asia/Ho_Chi_Minh").strftime("%Y-%m")})
    acc = load_acc(product, months)
    if acc is None:
        return None
    d = acc[(acc["slot_end_utc"] > lo) & (acc["slot_end_utc"] <= hi)]
    if slot is not None:
        s = d[d["slot_end_utc"] == slot].set_index("pixel_id")[["a6h", "a24h"]]
    else:
        s = d.groupby("pixel_id")[["a6h", "a24h"]].max()
    s = s.reindex(city_pixels())
    return None if s["a6h"].notna().sum() == 0 else s


# ============================================================================ reference

def stage_reference() -> dict:
    import pandas as pd
    from src.events.flood_indicator import indicator_levels
    from src.events.u1_followup import evidence_slots
    rows = []
    for day, group, slot, basis, _strict, merge_into in evidence_slots():
        when = slot if slot is not None else pd.Timestamp(datetime(day.year, day.month, day.day, 12), tz="Asia/Ho_Chi_Minh")
        prod = "imerg_final" if when.tz_convert("UTC") < pd.Timestamp(FINAL_END_UTC) else "imerg_late"
        s = slot_values(prod, day, slot)
        if s is None and prod == "imerg_final":
            prod, s = "imerg_late", slot_values("imerg_late", day, slot)
        c = indicator_levels(s["a6h"].to_numpy(), s["a24h"].to_numpy(), *GAUGE)
        lvl = int(c.max())
        if group == "rain_driven_evidence_day":
            role = "reference" if lvl >= 1 else "missed by the reference product"
        elif group == "river_flood_day":
            role = "river flood day (separate)"
        else:
            role = "2024-09-07 12:00: no inner-city flooding reported in the collected evidence"
        rows.append({"date_local": str(day), "day_group": group, "role": role, "reference_product": prod,
                     "time_basis": basis, "merge_into": str(merge_into or ""),
                     "n_none": int((c == 0).sum()), "n_yellow": int((c == 1).sum()), "n_red": int((c == 2).sum()),
                     "max_level": lvl, "a6h_max_mm": round(float(np.nanmax(s["a6h"])), 1),
                     "a24h_max_mm": round(float(np.nanmax(s["a24h"])), 1) if s["a24h"].notna().any() else None})
    _w(OUT / "reference_set.csv", rows)
    ref = [r["date_local"] for r in rows if r["role"] == "reference"]
    missed = [{k: r[k] for k in ("date_local", "reference_product", "time_basis", "a6h_max_mm", "a24h_max_mm")}
              for r in rows if r["role"].startswith("missed")]
    res = {"n_reference": len(ref), "reference_days": ref, "missed_by_reference": missed}
    write_json_atomic(OUT / "reference_set.json", res)
    return res


# ============================================================================ sweep

def settings():
    return [(f"a{a}_L{L}", float(a), 2.0 * a, float(L)) for a in A_VALUES for L in L_VALUES]


def stage_sweep() -> dict:
    import pandas as pd
    from src.events.flood_indicator import indicator_levels
    from src.events.indicator_analysis import MONTHS, load_acc
    from src.events.u1_followup import evidence_slots
    ref = pd.read_csv(OUT / "reference_set.csv")
    ref_days = set(ref.loc[ref["role"] == "reference", "date_local"])
    slots = evidence_slots()
    S = settings()
    guard = RamGuard()
    rows = []
    for product in PRODUCTS:
        # loads
        tot = {p: {s[0]: np.zeros(3, dtype=np.int64) for s in S} for p in ("2023-2025", "2024-2025")}
        days = {s[0]: {} for s in S}
        for month in MONTHS:
            if not guard.wait_ok():
                raise RuntimeError("free RAM < 0.5 GB for 5 min")
            acc = load_acc(product, [month])
            if acc is None:
                continue
            acc = acc[acc["slot_end_utc"].dt.minute == 0]
            acc = acc[acc["a6h"].notna() | acc["a24h"].notna()]
            ld = acc["slot_end_utc"].dt.tz_convert("Asia/Ho_Chi_Minh").dt.date.to_numpy()
            y24 = not month.startswith("2023")
            a6, a24 = acc["a6h"].to_numpy(), acc["a24h"].to_numpy()
            for name, a, b, L in S:
                c = indicator_levels(a6, a24, a, b, L)
                cnt = np.array([(c == k).sum() for k in range(3)], dtype=np.int64)
                tot["2023-2025"][name] += cnt
                if y24:
                    tot["2024-2025"][name] += cnt
                for dd, v in pd.Series(c).groupby(ld).max().items():
                    days[name][dd] = max(days[name].get(dd, -1), int(v))
            del acc
            guard.check()
        # evidence days: snapshot once per day, evaluate all settings
        snaps = {}
        for day, group, slot, basis, _s, merge_into in slots:
            s = slot_values(product, day, slot)
            if s is not None:
                snaps[str(day)] = (group, s)
        for name, a, b, L in S:
            dv = pd.Series(days[name])
            dv24 = dv[[d.year >= 2024 for d in dv.index]]
            lv = {d: int(indicator_levels(s["a6h"].to_numpy(), s["a24h"].to_numpy(), a, b, L).max())
                  for d, (g, s) in snaps.items()}
            caught = sorted(d for d in ref_days if lv.get(d, -1) >= 1)
            missed = sorted(d for d in ref_days if lv.get(d, -1) < 1)
            other_rain = sorted(d for d, (g, _) in snaps.items() if g == "rain_driven_evidence_day" and d not in ref_days)
            row = {"product": product, "setting": name, "a_mm": a, "b_mm": b, "L24h_mm": L}
            for per, sub in (("2024_2025", dv24), ("2023_2025", dv)):
                t = tot["2024-2025" if per == "2024_2025" else "2023-2025"][name]
                row[f"share_days_icon_{per}"] = round(float((sub >= 1).mean()), 4)
                row[f"share_days_red_{per}"] = round(float((sub == 2).mean()), 4)
                row[f"share_ph_icon_{per}"] = round(float((t[1] + t[2]) / t.sum()), 5)
                row[f"share_ph_red_{per}"] = round(float(t[2] / t.sum()), 5)
            row.update({"ref_days_caught": len(caught), "ref_days_total": len(ref_days),
                        "ref_days_missed": ";".join(missed),
                        "other_rain_days_with_icon": ";".join(d for d in other_rain if lv.get(d, -1) >= 1),
                        "level_2024_09_07_1200": lv.get("2024-09-07"),
                        "level_river_2024_09_10": lv.get("2024-09-10"), "level_river_2024_09_11": lv.get("2024-09-11")})
            rows.append(row)
    _w(OUT / "gsmap_2d_sweep.csv", rows)
    res = {"rows": len(rows), "settings_per_product": len(S), "reference_days": sorted(ref_days),
           "peak_rss_gb": round(guard.peak_rss, 3), "min_free_gb": round(guard.min_free, 3)}
    write_json_atomic(OUT / "gsmap_2d_sweep_summary.json", res)
    return res


# ============================================================================ rank + pareto + plot

def stage_rank() -> dict:
    import pandas as pd
    d = pd.read_csv(OUT / "gsmap_2d_sweep.csv")
    d["passes_load_2024_2025"] = (d["share_days_icon_2024_2025"] <= MAX_ICON) & (d["share_days_red_2024_2025"] <= MAX_RED)
    d["passes_load_2023_2025"] = (d["share_days_icon_2023_2025"] <= MAX_ICON) & (d["share_days_red_2023_2025"] <= MAX_RED)
    top, pareto = [], []
    for product, g in d.groupby("product"):
        ok = g[g["passes_load_2024_2025"]].sort_values(["ref_days_caught", "share_days_icon_2024_2025", "share_days_red_2024_2025"],
                                                      ascending=[False, True, True])
        for rank, (_, r) in enumerate(ok.head(3).iterrows(), 1):
            top.append({"product": product, "rank": rank, **r.to_dict()})
        # Pareto frontier: for each number of reference days caught, the lowest icon-day load (2024–2025)
        best = None
        for k in sorted(g["ref_days_caught"].unique(), reverse=True):
            gk = g[g["ref_days_caught"] >= k].sort_values(["share_days_icon_2024_2025", "share_days_red_2024_2025"]).iloc[0]
            if best is None or gk["share_days_icon_2024_2025"] < best:
                best = gk["share_days_icon_2024_2025"]
                pareto.append({"product": product, "ref_days_caught_at_least": int(k), "setting": gk["setting"],
                               "a_mm": gk["a_mm"], "b_mm": gk["b_mm"], "L24h_mm": gk["L24h_mm"],
                               "ref_days_caught": int(gk["ref_days_caught"]),
                               "share_days_icon_2024_2025": gk["share_days_icon_2024_2025"],
                               "share_days_red_2024_2025": gk["share_days_red_2024_2025"],
                               "share_days_icon_2023_2025": gk["share_days_icon_2023_2025"],
                               "passes_load_2024_2025": bool(gk["passes_load_2024_2025"]),
                               "ref_days_missed": gk["ref_days_missed"]})
    d.to_csv(OUT / "gsmap_2d_sweep.csv", index=False, encoding="utf-8")
    _w(OUT / "gsmap_top3.csv", top)
    _w(OUT / "pareto.csv", pareto)
    png = plot(d, pd.DataFrame(pareto))
    res = {"n_pass_load_2024_2025": {p: int(g["passes_load_2024_2025"].sum()) for p, g in d.groupby("product")},
           "top3": [{k: t[k] for k in ("product", "rank", "setting", "ref_days_caught", "share_days_icon_2024_2025",
                                       "share_days_red_2024_2025", "ref_days_missed")} for t in top],
           "plot": rel(png)}
    write_json_atomic(OUT / "rank_summary.json", res)
    return res


def plot(d, pareto):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, surf = "#0b0b0b", "#52514e", "#fcfcfb"
    cols = {"gsmap": "#eb6834", "gsmap_gc": "#1baf7a"}
    labels = {"gsmap": "GSMaP NOW", "gsmap_gc": "GSMaP NOW GC"}
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=130)
    fig.patch.set_facecolor(surf)
    ax.set_facecolor(surf)
    ax.axvspan(0, MAX_ICON * 100, color="#2a78d6", alpha=0.06, lw=0)
    ax.axvline(MAX_ICON * 100, color=muted, ls="--", lw=1)
    ax.text(MAX_ICON * 100 + 0.3, 0.3, "15 % ngày có icon", color=muted, fontsize=8)
    off = {"gsmap": -0.12, "gsmap_gc": 0.12}
    for p, c in cols.items():
        g = d[d["product"] == p]
        ax.scatter(g["share_days_icon_2024_2025"] * 100, g["ref_days_caught"] + off[p], s=14, color=c, alpha=0.35,
                   edgecolor="none", label=f"{labels[p]} — mọi (a, L)")
        f = pareto[pareto["product"] == p].sort_values("share_days_icon_2024_2025")
        ax.step(f["share_days_icon_2024_2025"] * 100, f["ref_days_caught"] + off[p], where="post", color=c, lw=2,
                label=f"{labels[p]} — biên Pareto")
        ok = g[g["passes_load_2024_2025"]]
        ax.scatter(ok["share_days_icon_2024_2025"] * 100, ok["ref_days_caught"] + off[p], s=26, facecolor="none",
                   edgecolor=c, linewidth=1.1, label=f"{labels[p]} — đạt tải (icon ≤ 15 %, đỏ ≤ 5 %)")
    n_ref = int(d["ref_days_total"].iloc[0])
    ax.set_ylim(-0.5, n_ref + 0.5)
    ax.set_yticks(range(0, n_ref + 1))
    ax.set_xlabel("Tỷ lệ ngày mùa mưa có icon, 2024–2025 (%)", color=ink)
    ax.set_ylabel(f"Số ngày tham chiếu bắt được (≥ vàng), / {n_ref}", color=ink)
    ax.set_title("U1 follow-up 3 — GSMaP: tải cảnh báo so với ngày tham chiếu (a 50–150, b = 2a, L 150–330)",
                 fontsize=10, loc="left", color=ink)
    ax.grid(color="#e6e5e1", lw=0.6)
    for s in ax.spines.values():
        s.set_color("#c9c8c3")
    ax.tick_params(colors=muted)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout()
    out = OUT / "pareto_gsmap.png"
    fig.savefig(out, facecolor=surf)
    plt.close(fig)
    return out


# ============================================================================ yaml

def stage_yaml() -> dict:
    import pandas as pd
    import yaml
    text = CONFIG.read_text(encoding="utf-8")
    before = yaml.safe_load(text)
    top = pd.read_csv(OUT / "gsmap_top3.csv")
    new = text
    if HEADER_OLD in new:
        new = new.replace(HEADER_OLD, HEADER_NEW)
    assert HEADER_NEW in new, "header lines not found"
    # drop an earlier follow-up 3 insertion (idempotent), then insert under each product
    new = re.sub(r"(?m)^      followup3_top3:.*\n(?:^        .*\n)*", "", new)
    for prod_key, p in (("gsmap_now", "gsmap"), ("gsmap_now_gc", "gsmap_gc")):
        t = top[top["product"] == p].sort_values("rank")
        lines = ["      followup3_top3:  # REPORT_U1 §16; b = 2a, L independent; loads on 2024–2025; NEEDS_REVIEW"]
        for _, r in t.iterrows():
            miss = r["ref_days_missed"] if isinstance(r["ref_days_missed"], str) else ""
            lines.append(f"        rank{int(r['rank'])}: {{a_mm: {r['a_mm']:g}, b_mm: {r['b_mm']:g}, long_mm: {r['L24h_mm']:g}, "
                         f"share_days_any_icon_2024_2025: {r['share_days_icon_2024_2025']}, "
                         f"share_days_red_2024_2025: {r['share_days_red_2024_2025']}, "
                         f"share_days_any_icon_2023_2025: {r['share_days_icon_2023_2025']}, "
                         f"ref_days_caught: \"{int(r['ref_days_caught'])}/{int(r['ref_days_total'])}\", "
                         f"ref_days_missed: \"{miss}\", status: NEEDS_REVIEW}}")
        # end of this product's block = next line with indent <= 4 inside gsmap_now_thresholds
        m = re.search(rf"(?m)^    {prod_key}:\n(?:^      .*\n|^        .*\n)*", new)
        assert m, prod_key
        new = new[:m.end()] + "\n".join(lines) + "\n" + new[m.end():]
    after = yaml.safe_load(new)
    for k in ("a_mm", "b_mm", "long_rule", "scale", "satellite_factor", "status", "variable", "product_thresholds",
              "satellite_proposal"):
        assert after[k] == before[k], k
    for k in ("gsmap_now", "gsmap_now_gc"):
        b0, a0 = before["gsmap_now_thresholds"]["products"][k], after["gsmap_now_thresholds"]["products"][k]
        assert {x: a0[x] for x in b0} == b0, f"existing {k} entries changed"
    # outside the gsmap_now_thresholds block and the two header lines: byte-identical
    def outside(s):
        s = s.replace(HEADER_OLD, "").replace(HEADER_NEW, "")
        return re.sub(r"(?ms)^gsmap_now_thresholds:.*?(?=^# --- satellite_proposal)", "", s)
    assert outside(new) == outside(text), "change outside the approved block"
    tmp = CONFIG.with_suffix(".yaml.tmp")
    tmp.write_text(new, encoding="utf-8")
    tmp.replace(CONFIG)
    return {"header_fixed": True, "top3": {k: after["gsmap_now_thresholds"]["products"][k]["followup3_top3"]
                                          for k in ("gsmap_now", "gsmap_now_gc")}}


STAGES = {"reference": stage_reference, "sweep": stage_sweep, "rank": stage_rank, "yaml": stage_yaml}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=list(STAGES) + ["all"], default="all")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in (STAGES if a.stage == "all" else [a.stage]):
        print(name, json.dumps(STAGES[name](), ensure_ascii=False, default=str)[:3000])
    import psutil
    print(f"PEAK_RSS_GB={psutil.Process().memory_info().peak_wset / 2**30:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
