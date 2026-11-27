"""U1 follow-up 2 (user decisions 2026-11-20, REPORT_U1 §14).

Stages:
  fixrecord  confirm the malformed-JSON diagnosis of fer_b101085e8c50fd10 and write ONE corrected record
             through the existing Silver writer (same values, extracted_text restored verbatim; checked as a
             substring of the Bronze artifact). `--dry` only prints the new record id.
  eqfreq     GSMaP / GSMaP GC A6h and A24h thresholds with the same pixel-hour exceedance share as IMERG Final
             at 50 / 100 mm (A6h) and 150 mm (A24h), over the 47 city pixels and the pixel-hours where both
             products exist; 2023–2025 and 2024–2025.
  sweep2     GSMaP / GSMaP GC candidates: a = 70..150 step 10 (b = 2a, A24h = 3a) + the equal-frequency sets;
             icon-day / red-day shares; level at every rain-driven evidence day (25/26 Aug 2025 also merged),
             river days separately, 2024-09-07 12:00.
  yaml       write the user decision U1-D1 and the GSMaP proposals (NEEDS_REVIEW) to configs/flood_indicator.yaml.
             a_mm, b_mm and long_rule are not touched.

Reads Silver/Bronze/catalog and the A6h/A24h month caches of U1.7 (data/_scratch/u1/acc).
Usage: .venv/Scripts/python.exe -m src.events.u1_followup2 --stage <name>
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import date, datetime
from pathlib import Path

import numpy as np
import yaml

from src.events.common import CATALOG_DIR, REPORTS, ROOT, SILVER_FE, RamGuard, rel, write_json_atomic

OUT = REPORTS / "followup2"
OLD_ID = "fer_b101085e8c50fd10"
MARK = '",\n  "extracted_text": "'
OVERRIDES = ROOT / "configs/event_catalog_overrides.yaml"
CONFIG = ROOT / "configs/flood_indicator.yaml"


def _w(path: Path, rows: list[dict]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


# ============================================================================ fixrecord

def corrected_record() -> tuple[dict, dict]:
    from src.events.text_extract import artifact_text, is_verbatim
    old_path = next(SILVER_FE.rglob(f"{OLD_ID}.json"))
    old = json.loads(old_path.read_text(encoding="utf-8"))
    loc = old["evidence_locator"]
    diag = {"old_path": rel(old_path), "has_extracted_text_key": "extracted_text" in old,
            "locator_contains_json_fragment": MARK in loc}
    if not diag["locator_contains_json_fragment"] or diag["has_extracted_text_key"]:
        raise SystemExit(f"diagnosis not confirmed: {diag}")
    head, swallowed = loc.split(MARK, 1)
    man = json.loads(next((ROOT / "data/bronze/flood_evidence/manifest").rglob(f"{old['evidence_id']}.json"))
                     .read_text(encoding="utf-8"))
    text = artifact_text(ROOT / man["raw_artifact_path"])
    diag.update(locator_head=head, swallowed_text=swallowed, swallowed_verbatim=is_verbatim(swallowed, text))
    restored = swallowed
    if not diag["swallowed_verbatim"]:
        # the artifact continues "… Hà Đông 124 mm; Hoàng Mai 123 mm" — drop the trailing full stop only
        restored = swallowed.rstrip(".").rstrip()
    if not is_verbatim(restored, text):
        raise SystemExit("restored passage is not verbatim — stop (TASK_U1 §9)")
    diag["restored_extracted_text"] = restored
    diag["restored_verbatim"] = True
    keys = ["evidence_id", "event_type", "observed_time_start", "observed_time_end", "temporal_text",
            "location_text", "latitude", "longitude", "flood_depth_m", "flood_depth_text",
            "inundation_duration_hours", "rainfall_mm", "rainfall_duration_hours", "extraction_method",
            "confidence", "review_status"]
    rec = {k: old.get(k) for k in keys}
    rec["extracted_text"] = restored
    rec["evidence_locator"] = head
    return rec, diag


def stage_fixrecord(dry: bool) -> dict:
    from flood_evidence_silver_builder import build_record_id, write_silver_record
    from flood_evidence_silver_validator import load_contract, validate_silver_record
    contract = load_contract()
    rec, diag = corrected_record()
    rid, _ = build_record_id(record=rec, contract=contract)
    validate_silver_record(record={**rec, "record_id": rid}, contract=contract)
    diag["new_record_id"] = rid
    if not dry:
        res = write_silver_record(record=rec, contract=contract)
        diag["writer_action"] = res["action"]
        diag["new_path"] = rel(res["record_path"])
        ov = yaml.safe_load(OVERRIDES.read_text(encoding="utf-8"))
        diag["overrides_superseded_by"] = ov["superseded"][OLD_ID]["superseded_by"]
        diag["overrides_match"] = diag["overrides_superseded_by"] == rid
    write_json_atomic(OUT / "fixrecord.json", diag)
    return diag


# ============================================================================ eqfreq

def _pairs(product: str, months: list[str]):
    """IMERG Final and `product` A6h/A24h on the same (slot, pixel) hours where both exist."""
    from src.events.indicator_analysis import load_acc
    import pandas as pd
    out = []
    for m in months:
        a = load_acc("imerg_final", [m])
        b = load_acc(product, [m])
        if a is None or b is None:
            continue
        a = a[a["slot_end_utc"].dt.minute == 0]
        b = b[b["slot_end_utc"].dt.minute == 0]
        j = a.merge(b, on=["slot_end_utc", "pixel_id"], suffixes=("_i", "_g"))
        out.append(j[["a6h_i", "a24h_i", "a6h_g", "a24h_g"]].astype("float32"))
        del a, b, j
    return pd.concat(out, ignore_index=True)


def eq_threshold(ref: np.ndarray, other: np.ndarray, t: float) -> tuple[float, float]:
    """Threshold on `other` with the same exceedance share as ref >= t (both arrays without NaN)."""
    share = float((ref >= t).mean())
    if share <= 0:
        return float("nan"), 0.0
    return float(np.quantile(other, 1 - share)), share


def stage_eqfreq() -> dict:
    from src.events.indicator_analysis import MONTHS
    guard = RamGuard()
    res, rows = {}, []
    for period, months in (("2023-2025", MONTHS), ("2024-2025", [m for m in MONTHS if not m.startswith("2023")])):
        for product in ("gsmap", "gsmap_gc"):
            if not guard.wait_ok():
                raise RuntimeError("low RAM for 5 min")
            d = _pairs(product, months)
            r = {"period": period, "product": product}
            m6 = d["a6h_i"].notna() & d["a6h_g"].notna()
            m24 = d["a24h_i"].notna() & d["a24h_g"].notna()
            r["pixel_hours_a6h"], r["pixel_hours_a24h"] = int(m6.sum()), int(m24.sum())
            for t in (50.0, 100.0):
                th, sh = eq_threshold(d.loc[m6, "a6h_i"].to_numpy(), d.loc[m6, "a6h_g"].to_numpy(), t)
                r[f"imerg_share_a6h_ge_{t:g}"] = round(sh, 6)
                r[f"eq_a6h_for_{t:g}"] = round(th, 1)
            th, sh = eq_threshold(d.loc[m24, "a24h_i"].to_numpy(), d.loc[m24, "a24h_g"].to_numpy(), 150.0)
            r["imerg_share_a24h_ge_150"] = round(sh, 6)
            r["eq_a24h_for_150"] = round(th, 1)
            rows.append(r)
            res[f"{product}|{period}"] = r
            del d
            guard.check()
    _w(OUT / "eqfreq_thresholds.csv", rows)
    write_json_atomic(OUT / "eqfreq_thresholds.json", {"rows": rows, "peak_rss_gb": round(guard.peak_rss, 3)})
    return {"rows": rows, "peak_rss_gb": round(guard.peak_rss, 3)}


# ============================================================================ sweep2

def round5(x: float) -> float:
    return float(5 * round(x / 5))


def candidates(product: str) -> list[tuple[str, float, float, float]]:
    eq = json.loads((OUT / "eqfreq_thresholds.json").read_text(encoding="utf-8"))["rows"]
    s = [(f"a{a}", float(a), 2.0 * a, 3.0 * a) for a in range(70, 160, 10)]
    for r in eq:
        if r["product"] == product:
            s.append((f"eqfreq_{r['period']}", round5(r["eq_a6h_for_50"]), round5(r["eq_a6h_for_100"]),
                      round5(r["eq_a24h_for_150"])))
    s.append(("gauge_50_100_150", 50.0, 100.0, 150.0))
    return s


def stage_sweep2() -> dict:
    import pandas as pd
    from src.events.flood_indicator import indicator_levels
    from src.events.indicator_analysis import MONTHS, city_pixels, load_acc
    from src.events.u1_followup import evidence_slots
    guard = RamGuard()
    freq, evrows = [], []
    slots = evidence_slots()
    city = city_pixels()
    for product in ("gsmap", "gsmap_gc"):
        S = candidates(product)
        tot = {s[0]: np.zeros(3, dtype=np.int64) for s in S}
        days = {s[0]: {} for s in S}
        for month in MONTHS:
            if not guard.wait_ok():
                raise RuntimeError("low RAM for 5 min")
            acc = load_acc(product, [month])
            if acc is None:
                continue
            acc = acc[acc["slot_end_utc"].dt.minute == 0]
            acc = acc[acc["a6h"].notna() | acc["a24h"].notna()]
            ld = acc["slot_end_utc"].dt.tz_convert("Asia/Ho_Chi_Minh").dt.date.to_numpy()
            yr = acc["slot_end_utc"].dt.year.to_numpy()
            a6, a24 = acc["a6h"].to_numpy(), acc["a24h"].to_numpy()
            for name, a, b, lm in S:
                c = indicator_levels(a6, a24, a, b, lm)
                for k in range(3):
                    tot[name][k] += int((c == k).sum())
                for dd, v in pd.Series(c).groupby(ld).max().items():
                    days[name][dd] = max(days[name].get(dd, -1), int(v))
            del acc
            guard.check()
        for name, a, b, lm in S:
            dv = pd.Series(days[name])
            dv24 = dv[[d.year >= 2024 for d in dv.index]]
            ph = int(tot[name].sum())
            freq.append({"product": product, "setting": name, "a_mm": a, "b_mm": b, "long24h_mm": lm,
                         "share_days_any_icon": round(float((dv >= 1).mean()), 4),
                         "share_days_red": round(float((dv == 2).mean()), 4),
                         "share_days_any_icon_2024_2025": round(float((dv24 >= 1).mean()), 4),
                         "share_days_red_2024_2025": round(float((dv24 == 2).mean()), 4),
                         "share_ph_any_icon": round((tot[name][1] + tot[name][2]) / ph, 5),
                         "share_ph_red": round(tot[name][2] / ph, 5), "days": int(len(dv))})
        for day, group, slot, basis, strict, merge_into in slots:
            lo = pd.Timestamp(datetime(day.year, day.month, day.day), tz="Asia/Ho_Chi_Minh").tz_convert("UTC")
            hi = lo + pd.Timedelta(days=1)
            months = sorted({lo.strftime("%Y-%m"), (hi - pd.Timedelta(minutes=1)).strftime("%Y-%m"),
                             lo.tz_convert("Asia/Ho_Chi_Minh").strftime("%Y-%m")})
            acc = load_acc(product, months)
            day_acc = acc[(acc["slot_end_utc"] > lo) & (acc["slot_end_utc"] <= hi)]
            if slot is not None:
                snap = day_acc[day_acc["slot_end_utc"] == slot].set_index("pixel_id")
                a6s, a24s = snap["a6h"], snap["a24h"]
            else:
                g = day_acc.groupby("pixel_id")[["a6h", "a24h"]].max()
                a6s, a24s = g["a6h"], g["a24h"]
            a6s, a24s = a6s.reindex(city), a24s.reindex(city)
            for name, a, b, lm in S:
                c = indicator_levels(a6s.to_numpy(), a24s.to_numpy(), a, b, lm)
                evrows.append({"date_local": str(day), "day_group": group, "merge_into": str(merge_into or ""),
                               "product": product, "setting": name, "a_mm": a, "b_mm": b, "long24h_mm": lm,
                               "time_basis": basis, "n_none": int((c == 0).sum()), "n_yellow": int((c == 1).sum()),
                               "n_red": int((c == 2).sum()), "max_level_city": int(c.max())})
    _w(OUT / "gsmap_sweep_frequency.csv", freq)
    _w(OUT / "gsmap_sweep_event_days.csv", evrows)
    fr, ev = pd.DataFrame(freq), pd.DataFrame(evrows)
    marks = []
    for (product, name), f in fr.groupby(["product", "setting"]):
        f = f.iloc[0]
        e = ev[(ev["product"] == product) & (ev["setting"] == name)]
        rd = e[e["day_group"] == "rain_driven_evidence_day"]
        ok = set(rd.loc[rd["max_level_city"] >= 1, "date_local"])
        missed = sorted(set(rd["date_local"]) - ok)
        mi = rd.set_index("date_local")["merge_into"]
        missed_m = [d for d in missed if not (mi.get(d) and mi.get(d) in ok)]
        lvl = lambda g: ";".join(f"{d}:{l}" for d, l in zip(e.loc[e["day_group"] == g, "date_local"],
                                                             e.loc[e["day_group"] == g, "max_level_city"]))
        yag = e[e["day_group"].str.startswith("yagi")]
        marks.append({"product": product, "setting": name, "a_mm": f["a_mm"], "b_mm": f["b_mm"],
                      "long24h_mm": f["long24h_mm"], "share_days_any_icon": f["share_days_any_icon"],
                      "share_days_red": f["share_days_red"],
                      "share_days_any_icon_2024_2025": f["share_days_any_icon_2024_2025"],
                      "share_days_red_2024_2025": f["share_days_red_2024_2025"],
                      "icon_days_le_15pct": bool(f["share_days_any_icon"] <= 0.15),
                      "red_days_le_5pct": bool(f["share_days_red"] <= 0.05),
                      "rain_days_missed": ";".join(missed), "rain_days_missed_2508_merged": ";".join(missed_m),
                      "levels_rain_days": lvl("rain_driven_evidence_day"), "levels_river_days": lvl("river_flood_day"),
                      "level_2024_09_07_1200": int(yag["max_level_city"].iloc[0]) if len(yag) else None,
                      "MARK_all_three": bool(f["share_days_any_icon"] <= 0.15 and f["share_days_red"] <= 0.05 and not missed),
                      "MARK_all_three_2508_merged": bool(f["share_days_any_icon"] <= 0.15 and f["share_days_red"] <= 0.05
                                                         and not missed_m)})
    order = {n: i for i, n in enumerate([c[0] for c in candidates("gsmap")])}
    marks.sort(key=lambda m: (m["product"], order.get(m["setting"], 99)))
    _w(OUT / "gsmap_sweep_marks.csv", marks)
    res = {"freq_rows": len(freq), "event_rows": len(evrows), "peak_rss_gb": round(guard.peak_rss, 3),
           "passing": {p: [m["setting"] for m in marks if m["product"] == p and m["MARK_all_three"]] for p in ("gsmap", "gsmap_gc")},
           "passing_2508_merged": {p: [m["setting"] for m in marks if m["product"] == p and m["MARK_all_three_2508_merged"]]
                                   for p in ("gsmap", "gsmap_gc")}}
    write_json_atomic(OUT / "gsmap_sweep_summary.json", res)
    return res


# ============================================================================ yaml

def stage_yaml() -> dict:
    import pandas as pd
    text = CONFIG.read_text(encoding="utf-8")
    before = yaml.safe_load(text)
    head = re.split(r"\n# --- (?:product_thresholds|satellite_proposal) \(", text)[0].rstrip("\n") + "\n"
    sp = before["satellite_proposal"]
    eq = json.loads((OUT / "eqfreq_thresholds.json").read_text(encoding="utf-8"))["rows"]
    mk = pd.read_csv(OUT / "gsmap_sweep_marks.csv")
    L = ["", "# --- product_thresholds (user decision U1-D1, 2026-11-20) -------------------------------------------",
         "# IMERG Final / Late: the gauge-scale thresholds are used directly (factor 1.0). Basis: REPORT_U1 §13.5 —",
         "# 12.2 % (Final) / 14.8 % (Late) of rainy-season days with an icon, 2.3 % / 3.1 % with red, and a signal on",
         "# 28/9/2023, 24/7/2024 and 30/9/2025. Chỉ báo dựa trên lượng mưa tích lũy; not a flood probability.",
         "product_thresholds:"]
    for p, lab in (("imerg_final", "IMERG Final V07"), ("imerg_late", "IMERG Late V07")):
        L.append(f"  {p}: {{label: \"{lab}\", a_mm: 50, b_mm: 100, long_mm: 150, satellite_factor: 1.0, "
                 f"status: provisional, decided: 2026-11-20, decision: U1-D1, basis: \"REPORT_U1 §13.5\"}}")
    L += ["", "# GSMaP NOW (raw rate and gauge-corrected GC): NOT decided. Candidate sets below are agent proposals",
          "# (REPORT_U1 §14.2); equal-frequency = same pixel-hour exceedance share as IMERG Final at 50 / 100 mm (A6h)",
          "# and 150 mm (A24h) over the 47 city pixels and the hours where both products exist; rounded to 5 mm.",
          "gsmap_now_thresholds:", "  status: NEEDS_REVIEW", "  computed: " + date.today().isoformat(),
          "  source_tables: [reports/u1/followup2/eqfreq_thresholds.csv, reports/u1/followup2/gsmap_sweep_marks.csv]",
          "  products:"]
    for p, key, lab in (("gsmap", "gsmap_now", "GSMaP NOW rain_rate_mm_h"), ("gsmap_gc", "gsmap_now_gc", "GSMaP NOW rain_rate_gc_mm_h")):
        L.append(f"    {key}:")
        L.append(f"      label: \"{lab}\"")
        L.append("      status: NEEDS_REVIEW")
        L.append("      proposals:")
        for r in eq:
            if r["product"] != p:
                continue
            name = f"eqfreq_{r['period']}"
            m = mk[(mk["product"] == p) & (mk["setting"] == name)].iloc[0]
            L.append(f"        {name.replace('-', '_')}: {{a_mm: {round5(r['eq_a6h_for_50']):g}, b_mm: {round5(r['eq_a6h_for_100']):g}, "
                     f"long_mm: {round5(r['eq_a24h_for_150']):g}, unrounded: [{r['eq_a6h_for_50']}, {r['eq_a6h_for_100']}, {r['eq_a24h_for_150']}], "
                     f"share_days_any_icon: {m['share_days_any_icon']}, share_days_red: {m['share_days_red']}, "
                     f"rain_days_missed: \"{m['rain_days_missed'] if isinstance(m['rain_days_missed'], str) else ''}\", status: NEEDS_REVIEW}}")
        L.append("      sweep_candidates:  # b = 2a, long = 3a; shares over 2023–2025 rainy-season days; NEEDS_REVIEW")
        for _, m in mk[(mk["product"] == p) & mk["setting"].str.match(r"a\d+$")].iterrows():
            miss = m["rain_days_missed"] if isinstance(m["rain_days_missed"], str) else ""
            L.append(f"        {m['setting']}: {{a_mm: {m['a_mm']:g}, b_mm: {m['b_mm']:g}, long_mm: {m['long24h_mm']:g}, "
                     f"share_days_any_icon: {m['share_days_any_icon']}, share_days_red: {m['share_days_red']}, "
                     f"n_rain_days_missed: {len([x for x in miss.split(';') if x])}, status: NEEDS_REVIEW}}")
    L += ["", "# --- satellite_proposal (agent, U1.7) — REJECTED by the user 2026-11-20 (decision U1-D1). Kept for the record.",
          "# Ratio = satellite pixel-mean accumulation / reported gauge amount (reports/u1/indicator_ratio_table.csv).",
          "satellite_proposal:", "  status: REJECTED", "  rejected: 2026-11-20",
          "  reason: \"47 % of rainy-season days with an icon (IMERG Final 20/40/60); ratio estimated from 2 extreme episodes only\"",
          f"  computed: {sp.get('computed')}", f"  record_set: {sp.get('record_set')}", "  products:"]
    for prod, p in sp["products"].items():
        L.append(f"    {prod}: {{label: \"{p['label']}\", n_records: {p['n_records']}, median_ratio: {p['median_ratio']}, "
                 f"iqr_ratio: [{p['iqr_ratio'][0]}, {p['iqr_ratio'][1]}], proposed_satellite_a_mm: {p['proposed_satellite_a_mm']:g}, "
                 f"proposed_satellite_b_mm: {p['proposed_satellite_b_mm']:g}, proposed_satellite_long_mm: {p['proposed_satellite_long_mm']:g}}}")
    new = head + "\n".join(L) + "\n"
    after = yaml.safe_load(new)
    for k in ("a_mm", "b_mm", "long_rule", "scale", "satellite_factor", "status", "variable"):
        assert after[k] == before[k], k
    for key in ("product_thresholds:", "gsmap_now_thresholds:", "satellite_proposal:"):
        assert len(re.findall(rf"^{key}", new, re.M)) == 1, f"duplicate block {key}"
    tmp = CONFIG.with_suffix(".yaml.tmp")
    tmp.write_text(new, encoding="utf-8")
    tmp.replace(CONFIG)
    return {"unchanged_keys_ok": True, "product_thresholds": after["product_thresholds"],
            "gsmap_now_thresholds": after["gsmap_now_thresholds"], "satellite_proposal_status": after["satellite_proposal"]["status"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["fixrecord", "eqfreq", "sweep2", "yaml"], required=True)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    res = stage_fixrecord(a.dry) if a.stage == "fixrecord" else {"eqfreq": stage_eqfreq, "sweep2": stage_sweep2,
                                                                  "yaml": stage_yaml}[a.stage]()
    print(a.stage, json.dumps(res, ensure_ascii=False, default=str)[:3500])
    import psutil
    print(f"PEAK_RSS_GB={psutil.Process().memory_info().peak_wset / 2**30:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
