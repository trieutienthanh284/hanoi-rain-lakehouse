"""U1 follow-up (user request 2026-11-20): provenance / tone-mark check, field verbatim check,
deduplicated rain validation, threshold sweep, Silver review sheet.

Reads only: Silver flood-evidence records, Bronze evidence artifacts, the event catalog, the
U1.4 table and the A6h/A24h month caches of U1.7 (data/_scratch/u1/acc). Writes only under
reports/u1/followup/ and reports/u1/U1_SILVER_REVIEW.md. Never edits configs, Silver or Bronze.
a_mm, b_mm and the 150 mm rule are not touched; the sweep is a report only.

Stages: prov, verbatim, rain, sweep, review (or all).
Usage: .venv/Scripts/python.exe -m src.events.u1_followup --stage all
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from src.events.common import CATALOG_DIR, REPORTS, ROOT, SILVER_FE, RamGuard, rel, sha256_file, write_json_atomic

OUT = REPORTS / "followup"
UTC = timezone.utc
RIVER_DAYS = {date(2024, 9, 10), date(2024, 9, 11)}
YAGI_MORNING = date(2024, 9, 7)
# "tính đến sáng 7/9" — the passage gives only "sáng" (morning); the article was published
# 2024-09-07 12:16 +07, so the latest end of the stated morning period is 12:00 local (D113).
YAGI_MORNING_END_LOCAL = "12:00"
YAGI_LABEL = "no inner-city flooding reported in the collected evidence"


def _w(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["empty"])
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


# ============================================================================ 1. provenance

UNIT_NF = r"(?:thành phố|quận|huyện|thị xã|phường|xã|district|ward|commune|city|khu vực|tại)"


def nofold(s: str) -> str:
    """NFC + lower case + dash/space normalisation, tone marks and đ KEPT."""
    s = unicodedata.normalize("NFC", s or "").lower()
    s = re.sub(r"[‐-―–—]", "-", s)
    return re.sub(r"\s+", " ", s).strip()


def nofold_core(s: str) -> str:
    s = re.sub(r"(,\s*(hà nội|hanoi|ha noi))+\s*$", "", nofold(s)).strip(" ,")
    s = re.sub(r"\(.*?\)", "", s).split(",")[0].strip()
    s = re.sub(rf"^(?:{UNIT_NF}\s+)+", "", s)
    s = re.sub(rf"\s+(?:{UNIT_NF})$", "", s)
    return s.strip(" -")


def nofold_street(s: str) -> str:
    s = re.sub(r"(,\s*(hà nội|hanoi|ha noi))+\s*$", "", nofold(s)).strip(" ,")
    m = re.search(r"(?:^|\s|\()(tuyến đường|đường|phố|đại lộ|ngõ)\s+(.+)", s)
    name = (("ngõ " if m.group(1) == "ngõ" else "") + m.group(2)) if m else re.sub(r"\s+(street|road|boulevard)\b.*$", "", s)
    return re.split(r"[,(]| đoạn | qua | gần | in ", name)[0].strip()


def stage_prov() -> dict:
    import geopandas as gpd
    import pandas as pd
    import pyogrio
    from src.events import build_event_catalog as bec

    prov = json.loads((ROOT / "data/reference/hanoi_city_admin_osm_20261012.provenance.json").read_text(encoding="utf-8"))
    pbf = ROOT / prov["source"]
    gpkg = ROOT / prov["output"]
    checks = {"pbf": rel(pbf), "pbf_exists": pbf.exists(),
              "pbf_sha256_now": sha256_file(pbf) if pbf.exists() else None,
              "pbf_sha256_provenance": prov["source_sha256"],
              "gpkg_sha256_now": sha256_file(gpkg), "gpkg_sha256_provenance": prov["output_sha256"]}
    checks["pbf_sha_match"] = checks["pbf_sha256_now"] == checks["pbf_sha256_provenance"]
    checks["gpkg_sha_match"] = checks["gpkg_sha256_now"] == checks["gpkg_sha256_provenance"]
    pbf_manifest = sorted(p.as_posix() for p in pbf.parent.glob("*") if p.is_file())
    checks["pbf_dir_files"] = [rel(Path(p)) for p in pbf_manifest]

    # admin units of Hanoi as used by the resolver
    adm_all = gpd.read_file(gpkg)
    hn = adm_all[(adm_all["level"] == "4") & (adm_all["name"] == "Thành phố Hà Nội") & (adm_all["era"] == "current")]
    hanoi = hn.geometry.union_all()
    adm = adm_all[adm_all["level"] == "6"].copy()
    adm = adm[adm.representative_point().within(hanoi)].copy()
    adm["key"] = adm["name"].map(bec.core_name)
    adm["key_nf"] = adm["name"].map(nofold_core)
    coll = []
    for (k, era), g in adm.groupby(["key", "era"]):
        if g["key_nf"].nunique() > 1:
            coll.append({"folded_key": k, "era": era, "names": " | ".join(sorted(g["name"]))})
    checks["admin_units_hanoi_level6"] = int(len(adm))
    checks["admin_counts_by_era"] = {str(k): int(v) for k, v in adm["era"].value_counts().items()}
    checks["admin_fold_collisions"] = coll

    cat = pd.read_parquet(CATALOG_DIR / "events.parquet")
    res = cat[cat["geo_status"] == "resolved"]
    ln = pyogrio.read_dataframe(bec.LINES, columns=["name"], read_geometry=False)
    ln = ln[ln["name"].notna()].copy()
    ln["key"] = ln["name"].map(lambda s: re.sub(r"^(duong|pho|dai lo)\s+", "", bec.fold(s)))
    wy = pyogrio.read_dataframe(bec.WAYS, columns=["name"], read_geometry=False)
    wy = wy[wy["name"].notna()].copy()
    wy["key"] = wy["name"].map(bec.fold)
    rows = []
    for _, r in res.iterrows():
        loc, lv, fid = r["location_text"], r["location_level"], r["feature_id"] or ""
        if lv == "city":
            dep = bool(re.search(r"\bha ?noi\b", nofold(loc)))
            text_nf, feat_nf = nofold(bec.strip_city(nofold(loc))) or nofold(loc), "hà nội"
            other = ""
        elif fid.startswith("osm:relation/"):
            text_nf, feat_nf = nofold_core(loc), nofold_core(r["feature_name"])
            dep, other = text_nf != feat_nf, ""
        elif fid.startswith("osm:line_name/"):
            key = fid.split("/", 1)[1]
            names = sorted(set(ln.loc[ln["key"] == key, "name"]))
            names_nf = sorted({re.sub(r"^(đường|phố|đại lộ)\s+", "", nofold(n)) for n in names})
            text_nf = nofold_street(loc)
            feat_nf = " | ".join(names_nf)
            dep = text_nf not in names_nf
            other = " | ".join(names_nf) if len(names_nf) > 1 else ""  # folding merged distinct toned names
        else:
            key = fid.split("/", 1)[1]
            names_nf = sorted({nofold(n) for n in wy.loc[wy["key"] == key, "name"]})
            text_nf = nofold(bec.strip_city(nofold(loc))).split(",")[0].strip()
            feat_nf = " | ".join(names_nf)
            dep = text_nf not in names_nf
            other = " | ".join(names_nf) if len(names_nf) > 1 else ""
        rows.append({"silver_record_id": r["silver_record_id"], "event_date": str(r["event_date_start"]),
                     "location_level": lv, "location_text": loc, "feature_id": fid, "feature_name": r["feature_name"],
                     "text_key_tone_kept": text_nf, "feature_key_tone_kept": feat_nf,
                     "match_needs_tone_folding": dep, "other_tone_variants_pulled_in": other})
    _w(OUT / "tone_fold_matches.csv", rows)
    dep_rows = [x for x in rows if x["match_needs_tone_folding"] or x["other_tone_variants_pulled_in"]]
    summary = {"checks": checks, "resolved_rows": len(rows),
               "needs_folding": sum(x["match_needs_tone_folding"] for x in rows),
               "with_other_tone_variants": sum(bool(x["other_tone_variants_pulled_in"]) for x in rows),
               "rows_needing_folding": dep_rows}
    # network: static scan of the U1 code for network modules
    net = {}
    for p in sorted((ROOT / "src/events").glob("*.py")):
        t = p.read_text(encoding="utf-8")
        hits = sorted(set(re.findall(r"^\s*(?:import|from)\s+(requests|urllib|http|httpx|socket|ftplib|boto3|aiohttp)\b", t, re.M)))
        net[rel(p)] = hits
    summary["network_imports_src_events"] = net
    write_json_atomic(OUT / "prov_summary.json", summary)
    return {k: v for k, v in summary.items() if k != "rows_needing_folding"} | {"rows_needing_folding_n": len(dep_rows)}


# ============================================================================ 3. verbatim

def silver_records() -> dict[str, dict]:
    out = {}
    for p in SILVER_FE.glob("evidence_id=*/*.json"):
        j = json.loads(p.read_text(encoding="utf-8"))
        j["_path"] = rel(p)
        out[j["record_id"]] = j
    return out


def stage_verbatim() -> dict:
    import pandas as pd
    from src.events.text_extract import artifact_text, is_verbatim, normalise

    u1 = list(csv.DictReader(open(REPORTS / "silver_records_u1.csv", encoding="utf-8")))
    ing = {r["evidence_id"]: r for r in csv.DictReader(open(REPORTS / "ingest_inbox_run1.csv", encoding="utf-8"))}
    sil = silver_records()
    texts: dict[str, str] = {}
    rows, misses = [], []
    for r in u1:
        rec = sil[r["record_id"]]
        eid = rec["evidence_id"]
        if eid not in texts:
            texts[eid] = artifact_text(ROOT / ing[eid]["bronze_artifact"])
        t = texts[eid]
        for field in ("temporal_text", "location_text", "flood_depth_text"):
            v = rec.get(field)
            if v in (None, ""):
                continue
            ok = is_verbatim(v, t)
            diag = ""
            if not ok:
                stripped = re.sub(r"(,\s*(Hà Nội|Hanoi))+\s*$", "", v).strip()
                bare = re.sub(r"^(phường|xã|huyện|quận|đường|phố)\s+", "", stripped)
                if stripped != v and is_verbatim(stripped, t):
                    diag = "verbatim after removing the agent-appended ', Hà Nội'"
                elif bare != stripped and is_verbatim(bare, t):
                    diag = (f"bare name verbatim; agent added unit word '{stripped[:len(stripped) - len(bare)].strip()}'"
                            " (taken from the list header) and ', Hà Nội'")
                elif normalise(v).lower() in normalise(t.replace("\n", " ")).lower():
                    diag = "verbatim except letter case"
                elif is_verbatim(v, rec.get("extracted_text") or ""):
                    diag = "inside extracted_text only (?)"
                else:
                    parts = [p.strip() for p in re.split(r"[,;]", stripped) if p.strip()]
                    found = [p for p in parts if is_verbatim(p, t)]
                    diag = (f"not verbatim; parts found: {found}" if found else "not verbatim (agent wording)")
            row = {"record_id": r["record_id"], "file_name": r["file_name"], "event_type": rec["event_type"],
                   "field": field, "value": v, "verbatim": ok, "diagnosis": diag}
            rows.append(row)
            if not ok:
                misses.append(row)
    _w(OUT / "verbatim_fields.csv", rows)
    # end_only_24h passages
    rv = pd.read_csv(REPORTS / "rain_validation.csv")
    e24 = rv[rv["period_basis"].isin(["end_only_24h", "end_only_unknown_start"])]
    lines = ["# U1 follow-up — đoạn nguyên văn biện minh cho kỳ `end_only_24h`", "",
             "Quy tắc D107: chỉ có giờ kết thúc → kỳ (kết thúc − 24 h, kết thúc]. `end_only_unknown_start` (người dùng "
             "2026-11-20): chỉ biết thời điểm đưa tin, không biết đầu kỳ → ngoài tập chính, chỉ dùng cho độ nhạy. Ngữ cảnh = ±220 ký tự quanh "
             "`temporal_text` trong văn bản artifact Bronze (đã chuẩn hoá NFC + khoảng trắng).", ""]
    e24_out = []
    for _, r in e24.iterrows():
        rec = sil[r["silver_record_id"]]
        eid = rec["evidence_id"]
        if eid not in texts:
            texts[eid] = artifact_text(ROOT / ing[eid]["bronze_artifact"])
        t = normalise(texts[eid].replace("\n", " "))
        tt = normalise(rec.get("temporal_text") or "")
        i = t.find(tt) if tt else -1
        ctx = t[max(0, i - 220): i + len(tt) + 220] if i >= 0 else None
        e24_out.append({"record_id": r["silver_record_id"], "source": r["source_name"], "location": r["location_text"],
                        "reported_mm": r["reported_mm"], "temporal_text": rec.get("temporal_text"),
                        "period_local": r["period_local"], "temporal_text_found": i >= 0})
        lines += [f"## {r['source_name']} — {r['location_text']} — {r['reported_mm']} mm — `{r['period_basis']}`", "",
                  f"- record `{r['silver_record_id']}`; `temporal_text` = \"{rec.get('temporal_text')}\" → kỳ {r['period_local']} (giờ ĐP)",
                  f"- `extracted_text`: \"{rec.get('extracted_text')}\"",
                  f"- ngữ cảnh trong artifact: " + (f"\"…{ctx}…\"" if ctx else "**temporal_text không tìm thấy nguyên văn**"), ""]
    (OUT / "end_only_24h_passages.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    by_field = {}
    for row in rows:
        k = row["field"]
        by_field.setdefault(k, [0, 0])
        by_field[k][0] += 1
        by_field[k][1] += int(not row["verbatim"])
    diag_counts = {}
    for m in misses:
        key = m["diagnosis"].split(";")[0]
        diag_counts[key] = diag_counts.get(key, 0) + 1
    res = {"records": len(u1), "fields_checked": len(rows),
           "by_field_checked_misses": by_field, "misses": len(misses), "miss_diagnosis": diag_counts,
           "end_only_24h": e24_out}
    write_json_atomic(OUT / "verbatim_summary.json", res)
    return res


# ============================================================================ 4. rain validation

PRODS = {"imerg_final": "IMERG Final", "imerg_late": "IMERG Late", "gsmap": "GSMaP NOW", "gsmap_gc": "GSMaP NOW GC"}


def gauge_key(loc: str) -> str:
    from src.events.build_event_catalog import fold, strip_city, UNIT_WORDS
    s = strip_city(fold(loc))
    s = re.sub(rf"^(?:{UNIT_WORDS}\s+)+", "", s)
    s = re.sub(rf"\s+(?:{UNIT_WORDS})(?=\s*(\(|$))", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _ratios(d, prod):
    r = d[f"{prod}_mean"] / d["reported_mm"]
    if prod == "imerg_final":
        r = r[d["imerg_final_coverage"] >= 0.9]
    return r.dropna()


def _stats(d, prod) -> dict:
    r = _ratios(d, prod)
    if not len(r):
        return {"n": 0}
    ev = d.loc[r.index].assign(ratio=r).groupby("episode")["ratio"].median()
    return {"n": int(len(r)), "median": round(float(r.median()), 3), "q25": round(float(r.quantile(.25)), 3),
            "q75": round(float(r.quantile(.75)), 3), "n_events": int(len(ev)),
            "median_of_event_medians": round(float(ev.median()), 3),
            "event_medians": {k: round(float(v), 3) for k, v in ev.items()}}


def rain_dup_key(rv):
    """U1.4 dedup key (U1-R2b, D201): same gauge, same period basis and same accumulation window. Two records whose
    stated periods differ (e.g. 'ngày 7/10' = local_day vs 'từ 0h10 đêm 7/10' = start_only_unknown_end) are never
    merged, even when the computed window coincides."""
    return rv["gauge_key"].astype(str) + "|" + rv["period_basis"].astype(str) + "|" + rv["period_local"].astype(str)


def stage_rain() -> dict:
    import pandas as pd
    rv = pd.read_csv(REPORTS / "rain_validation.csv")
    cat = pd.read_parquet(CATALOG_DIR / "events.parquet").set_index("silver_record_id")
    rv["gauge_key"] = rv["location_text"].map(gauge_key)
    rv["dup_key"] = rain_dup_key(rv)
    rv["lower_bound_gt"] = rv["qualifier"].eq("gt")
    rv["pixels"] = rv["silver_record_id"].map(lambda s: ",".join(str(int(p)) for p in cat.loc[s, "rain_pixel_ids"]))
    # keep one record per gauge+period: prefer exact ('eq'), then the earliest-sorted source
    rv["_pref"] = rv["qualifier"].map({"eq": 0, "approx": 1, "max": 2, "gt": 3}).fillna(4)
    rv = rv.sort_values(["dup_key", "_pref", "source_name"])
    rv["kept"] = ~rv.duplicated("dup_key")
    if "duplicate_of" not in rv.columns:
        rv["duplicate_of"] = None
    ov_dup = rv["duplicate_of"].notna()
    rv.loc[ov_dup, "kept"] = False  # user-approved duplicates (configs/event_catalog_overrides.yaml)
    dup_groups = []
    for k, g in rv.groupby("dup_key"):
        if len(g) > 1:
            dup_groups.append({"gauge_period": k, "records": [f"{s} {m} mm ({q})" for s, m, q in
                                                              zip(g["source_name"], g["reported_mm"], g["qualifier"])],
                               "kept": g.loc[g["kept"], "source_name"].iloc[0]})
    # rain episodes: merge record periods (local) that overlap or touch → independent events
    def span(pl: str):
        parts = [p.split("–") for p in pl.split("; ")]
        return (min(datetime.strptime(a, "%Y-%m-%d %H:%M") for a, _ in parts),
                max(datetime.strptime(b, "%Y-%m-%d %H:%M") for _, b in parts))
    sp = rv["period_local"].map(span)
    rv["p0"], rv["p1"] = sp.map(lambda x: x[0]), sp.map(lambda x: x[1])
    order = rv.sort_values("p0")
    ep, cur_end, n = {}, None, 0
    for i, r in order.iterrows():
        if cur_end is None or r["p0"] > cur_end:
            n += 1
            cur_start = r["p0"]
            cur_end = r["p1"]
            label = cur_start.strftime("%Y-%m-%d")
        cur_end = max(cur_end, r["p1"])
        ep[i] = label
    rv["episode"] = pd.Series(ep)
    rv["period_class"] = np.where(rv["period_hours"] <= 6, "<=6h", ">6h")
    rv["year"] = rv["event_date"].str[:4]
    out_cols = ["silver_record_id", "source_name", "event_date", "episode", "location_text", "gauge_key", "feature_name",
                "pixels", "period_basis", "period_local", "period_hours", "period_class", "reported_mm", "qualifier",
                "lower_bound_gt", "kept", "duplicate_of"] + [f"{p}_{s}" for p in PRODS for s in ("mean", "coverage")]
    rv[out_cols].sort_values(["episode", "source_name", "location_text"]).to_csv(
        OUT / "rain_validation_dedup.csv", index=False, encoding="utf-8")
    kept = rv[rv["kept"]]
    from src.events.rain_validation import PRIMARY_BASES
    sets = {"primary": kept[kept["period_basis"].isin(PRIMARY_BASES)],
            "all_incl_local_day": kept}
    table, summary = [], {"input_records": int(len(rv)), "duplicates_removed": int((~rv["kept"]).sum()),
                          "duplicates_by_override": [f"{a} -> {b}" for a, b in zip(rv.loc[ov_dup, "silver_record_id"],
                                                                                    rv.loc[ov_dup, "duplicate_of"])],
                          "duplicate_groups": dup_groups, "kept_records": int(len(kept)),
                          "gt_records_kept": int(kept["lower_bound_gt"].sum()),
                          "max_qualifier_kept": int(kept["qualifier"].eq("max").sum()),
                          "episodes_all": sorted(kept["episode"].unique().tolist()), "sets": {}}
    for sname, d in sets.items():
        uniq_pix_period = d.drop_duplicates(["pixels", "period_local"])
        S = {"n_records": int(len(d)), "n_episodes": int(d["episode"].nunique()),
             "episodes": sorted(d["episode"].unique().tolist()),
             "n_unique_pixelset_period": int(len(uniq_pix_period)), "products": {}}
        for prod in PRODS:
            P = {}
            for variant, dd in (("with_gt", d), ("without_gt", d[~d["lower_bound_gt"]])):
                P[variant] = _stats(dd, prod)
                for pc in ("<=6h", ">6h"):
                    P[f"{variant}|{pc}"] = _stats(dd[dd["period_class"] == pc], prod)
                if prod.startswith("gsmap"):
                    P[f"{variant}|2023"] = _stats(dd[dd["year"] == "2023"], prod)
                    P[f"{variant}|2024-2025"] = _stats(dd[dd["year"] != "2023"], prod)
                    P[f"{variant}|no2023"] = P[f"{variant}|2024-2025"]
            for k, v in P.items():
                if k.endswith("|no2023"):
                    continue
                table.append({"record_set": sname, "product": prod, "subset": k, **{x: v.get(x) for x in
                              ("n", "n_events", "median", "q25", "q75", "median_of_event_medians")},
                              "event_medians": json.dumps(v.get("event_medians", {}))})
            S["products"][prod] = P
        summary["sets"][sname] = S
    # sensitivity for end_only_24h: window (local midnight of the end day, end] instead of 24 h
    from src.events.rain_validation import accumulate as acc_period
    from src.events.build_event_catalog import TZ
    sens = []
    for _, r in rv[rv["period_basis"].isin(["end_only_24h", "end_only_unknown_start"]) & rv["kept"]].iterrows():
        end = datetime.strptime(r["period_local"].split("–")[1], "%Y-%m-%d %H:%M").replace(tzinfo=TZ)
        start = end.replace(hour=0, minute=0)
        pix = [int(p) for p in r["pixels"].split(",")]
        row = {"silver_record_id": r["silver_record_id"], "period_basis": r["period_basis"],
               "source_name": r["source_name"], "location_text": r["location_text"],
               "reported_mm": r["reported_mm"], "window_24h": r["period_local"],
               "window_midnight": f"{start:%Y-%m-%d %H:%M}–{end:%Y-%m-%d %H:%M}"}
        for prod in PRODS:
            a = acc_period(prod, [(start.astimezone(UTC), end.astimezone(UTC))], pix)
            ok = a["mean"] is not None and (prod != "imerg_final" or a["coverage"] >= 0.9)
            row[f"{prod}_ratio_24h"] = round(r[f"{prod}_mean"] / r["reported_mm"], 3) if pd.notna(r[f"{prod}_mean"]) and (
                prod != "imerg_final" or r["imerg_final_coverage"] >= 0.9) else None
            row[f"{prod}_ratio_midnight"] = round(a["mean"] / r["reported_mm"], 3) if ok else None
        sens.append(row)
    _w(OUT / "end_only_24h_sensitivity.csv", sens)
    summary["end_only_24h_sensitivity"] = sens
    sd = pd.DataFrame(sens)
    sd = sd[sd["period_basis"] == "end_only_unknown_start"] if len(sd) else sd
    summary["sensitivity_end_only_unknown_start"] = {
        "n": int(len(sd)), "note": "not in the primary set; window 24 h or local midnight -> end, both unproven",
        **{f"{p}_median_ratio_{w}": (round(float(sd[f"{p}_ratio_{w}"].dropna().median()), 3)
                                     if len(sd) and sd[f"{p}_ratio_{w}"].notna().any() else None)
           for p in PRODS for w in ("24h", "midnight")}}
    _w(OUT / "rain_validation_followup_stats.csv", table)
    write_json_atomic(OUT / "rain_validation_followup_summary.json", summary)
    return {k: v for k, v in summary.items() if k != "sets"} | {
        s: {"n": v["n_records"], "episodes": v["episodes"], "uniq_pix_period": v["n_unique_pixelset_period"]}
        for s, v in summary["sets"].items()}


# ============================================================================ 5. threshold sweep

SWEEP_PRODUCTS = ("imerg_final", "imerg_late", "gsmap", "gsmap_gc")


def settings() -> list[tuple[str, float, float, float]]:
    s = [(f"a{a}", float(a), 2.0 * a, 3.0 * a) for a in range(20, 75, 5)]
    return s + [("gauge_50_100_150", 50.0, 100.0, 150.0)]


def evidence_slots():
    """(day, label, slot_end_utc or None (→ day max), basis text, strict pixel list)."""
    import pandas as pd
    from src.events.indicator_analysis import first_flood_time
    cat = pd.read_parquet(CATALOG_DIR / "events.parquet")
    ed = pd.read_parquet(CATALOG_DIR / "event_days.parquet")
    ed = ed[(ed["n_flood_observation"] + ed["n_flood_impact"]) > 0]
    out = []
    for _, d in ed.iterrows():
        day = d["date_local"]
        group = "river_flood_day" if day in RIVER_DAYS else "rain_driven_evidence_day"
        t = first_flood_time(cat, day)
        f = cat[cat["event_type"].isin(["flood_observation", "flood_impact"])
                & (cat["event_date_start"] <= day) & (cat["event_date_end"].fillna(cat["event_date_start"]) >= day)
                & cat["location_level"].isin(["district", "ward", "street", "feature"]) & (cat["geo_status"] == "resolved")]
        strict = sorted({int(p) for ps in f["rain_pixel_ids"] for p in ps})
        fd = cat[cat["event_type"].isin(["flood_observation", "flood_impact"])
                 & (cat["event_date_start"] <= day) & (cat["event_date_end"].fillna(cat["event_date_start"]) >= day)]
        nxt = day + timedelta(days=1)
        merge_into = nxt if len(fd) and ((fd["time_precision"] == "date_range") & (fd["event_date_end"] == nxt)).all() else None
        if t is not None:
            slot = t.floor("30min")
            basis = f"first reported flood {t.tz_convert('Asia/Ho_Chi_Minh'):%H:%M} (slot end {slot.tz_convert('Asia/Ho_Chi_Minh'):%H:%M})"
        else:
            slot, basis = None, "local-day maximum (no stated flood time, flagged)"
        out.append((day, group, slot, basis, strict, merge_into))
    hh, mm = map(int, YAGI_MORNING_END_LOCAL.split(":"))
    ys = pd.Timestamp(datetime(YAGI_MORNING.year, YAGI_MORNING.month, YAGI_MORNING.day, hh, mm),
                      tz="Asia/Ho_Chi_Minh").tz_convert("UTC")
    out.append((YAGI_MORNING, "yagi_morning_" + YAGI_LABEL.replace(" ", "_"), ys,
                f"end of the morning period 'tính đến sáng 7/9' ({YAGI_MORNING_END_LOCAL} local, D113)", [], None))
    return out


def stage_sweep() -> dict:
    import pandas as pd
    from src.events.flood_indicator import indicator_levels
    from src.events.indicator_analysis import MONTHS, city_pixels, load_acc
    S = settings()
    guard = RamGuard()
    freq_rows = []
    for product in SWEEP_PRODUCTS:
        tot = {s[0]: np.zeros(3, dtype=np.int64) for s in S}
        days = {s[0]: {} for s in S}
        for month in MONTHS:
            if not guard.wait_ok():
                raise RuntimeError(f"low RAM for 5 min at {product} {month}")
            acc = load_acc(product, [month])
            if acc is None:
                continue
            acc = acc[acc["slot_end_utc"].dt.minute == 0]
            acc = acc[acc["a6h"].notna() | acc["a24h"].notna()]
            ld = acc["slot_end_utc"].dt.tz_convert("Asia/Ho_Chi_Minh").dt.date.to_numpy()
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
            dv = np.array(list(days[name].values()))
            ph = int(tot[name].sum())
            freq_rows.append({"product": product, "setting": name, "a_mm": a, "b_mm": b, "long24h_mm": lm,
                              "pixel_hours": ph, "share_ph_any_icon": round((tot[name][1] + tot[name][2]) / ph, 5),
                              "share_ph_red": round(tot[name][2] / ph, 5), "days": int(len(dv)),
                              "days_any_icon": int((dv >= 1).sum()), "days_red": int((dv == 2).sum()),
                              "share_days_any_icon": round(float((dv >= 1).mean()), 4),
                              "share_days_red": round(float((dv == 2).mean()), 4)})
    _w(OUT / "threshold_sweep_frequency.csv", freq_rows)

    # event days
    slots = evidence_slots()
    city = city_pixels()
    ev_rows = []
    for day, group, slot, basis, strict, merge_into in slots:
        lo = pd.Timestamp(datetime(day.year, day.month, day.day), tz="Asia/Ho_Chi_Minh").tz_convert("UTC")
        hi = lo + pd.Timedelta(days=1)
        months = sorted({lo.strftime("%Y-%m"), (hi - pd.Timedelta(minutes=1)).strftime("%Y-%m"),
                         lo.tz_convert("Asia/Ho_Chi_Minh").strftime("%Y-%m")})
        for product in SWEEP_PRODUCTS:
            acc = load_acc(product, months)
            if acc is None:
                continue
            day_acc = acc[(acc["slot_end_utc"] > lo) & (acc["slot_end_utc"] <= hi)]
            if day_acc.empty or day_acc["a6h"].notna().sum() == 0:
                continue
            if slot is not None:
                snap = day_acc[day_acc["slot_end_utc"] == slot].set_index("pixel_id")
                a6s, a24s = snap["a6h"], snap["a24h"]
            else:
                g = day_acc.groupby("pixel_id")[["a6h", "a24h"]].max()
                a6s, a24s = g["a6h"], g["a24h"]
            a6s, a24s = a6s.reindex(city), a24s.reindex(city)
            sp = [p for p in strict if p in set(city)]
            for name, a, b, lm in S:
                c = indicator_levels(a6s.to_numpy(), a24s.to_numpy(), a, b, lm)
                cs = pd.Series(c, index=city)
                strict_max = int(cs.loc[sp].max()) if sp else None
                ev_rows.append({"date_local": str(day), "day_group": group, "merge_into": str(merge_into or ""),
                                "product": product, "setting": name,
                                "a_mm": a, "b_mm": b, "long24h_mm": lm, "time_basis": basis,
                                "n_none": int((c == 0).sum()), "n_yellow": int((c == 1).sum()), "n_red": int((c == 2).sum()),
                                "n_no_data": int((c == -1).sum()), "max_level_city": int(c.max()),
                                "max_level_flood_location_pixels": strict_max, "n_flood_location_pixels": len(sp),
                                "a6h_max": round(float(np.nanmax(a6s)), 1) if a6s.notna().any() else None,
                                "a24h_max": round(float(np.nanmax(a24s)), 1) if a24s.notna().any() else None})
    _w(OUT / "threshold_sweep_event_days.csv", ev_rows)

    # marks
    fr = pd.DataFrame(freq_rows)
    ev = pd.DataFrame(ev_rows)
    marks = []
    for product in SWEEP_PRODUCTS:
        rd = ev[(ev["product"] == product) & (ev["day_group"] == "rain_driven_evidence_day")]
        n_days_eval = rd["date_local"].nunique()
        for name, a, b, lm in S:
            f = fr[(fr["product"] == product) & (fr["setting"] == name)].iloc[0]
            e = rd[rd["setting"] == name]
            all_yellow = bool(len(e) and (e["max_level_city"] >= 1).all())
            strict_e = e[e["max_level_flood_location_pixels"].notna()]
            strict_yellow = bool(len(strict_e) and (strict_e["max_level_flood_location_pixels"] >= 1).all())
            missed = sorted(e.loc[e["max_level_city"] < 1, "date_local"])
            ok_days = set(e.loc[e["max_level_city"] >= 1, "date_local"])
            mi = e.set_index("date_local")["merge_into"]
            missed_merged = [d for d in missed if not (mi.get(d) and mi.get(d) in ok_days)]
            y = ev[(ev["product"] == product) & (ev["setting"] == name) & ev["day_group"].str.startswith("yagi")]
            marks.append({"product": product, "setting": name, "a_mm": a, "b_mm": b, "long24h_mm": lm,
                          "share_days_any_icon": f["share_days_any_icon"], "share_days_red": f["share_days_red"],
                          "share_ph_any_icon": f["share_ph_any_icon"],
                          "icon_days_le_15pct": bool(f["share_days_any_icon"] <= 0.15),
                          "red_days_le_5pct": bool(f["share_days_red"] <= 0.05),
                          "rain_days_evaluated": int(n_days_eval),
                          "yellow_on_all_rain_days_city": all_yellow, "rain_days_missed_city": ";".join(missed),
                          "yellow_on_all_rain_days_night_range_merged": bool(len(e)) and not missed_merged,
                          "rain_days_missed_night_range_merged": ";".join(missed_merged),
                          "yellow_on_all_rain_days_flood_pixels": strict_yellow,
                          "rain_days_with_flood_pixels": int(strict_e["date_local"].nunique()),
                          "yagi_0907_morning_max_level": int(y["max_level_city"].iloc[0]) if len(y) else None,
                          "MARK_all_three": bool(f["share_days_any_icon"] <= 0.15 and f["share_days_red"] <= 0.05 and all_yellow),
                          "MARK_sensitivity_night_merged": bool(f["share_days_any_icon"] <= 0.15 and f["share_days_red"] <= 0.05
                                                                and bool(len(e)) and not missed_merged)})
    _w(OUT / "threshold_sweep_marks.csv", marks)
    passing = {p: [m["setting"] for m in marks if m["product"] == p and m["MARK_all_three"]] for p in SWEEP_PRODUCTS}
    passing_merged = {p: [m["setting"] for m in marks if m["product"] == p and m["MARK_sensitivity_night_merged"]]
                      for p in SWEEP_PRODUCTS}
    res = {"settings": [s[0] for s in S], "freq_rows": len(freq_rows), "event_rows": len(ev_rows),
           "evidence_slots": [(str(d), g, None if s is None else str(s), b, str(m or "")) for d, g, s, b, _, m in slots],
           "passing_settings": passing, "passing_settings_sensitivity_night_merged": passing_merged, "peak_rss_gb": round(guard.peak_rss, 3), "min_free_gb": round(guard.min_free, 3)}
    write_json_atomic(OUT / "threshold_sweep_summary.json", res)
    return res


# ============================================================================ 6. review sheet

FIELD_ORDER = ["event_type", "observed_time_start", "observed_time_end", "temporal_text", "location_text",
               "rainfall_mm", "rainfall_duration_hours", "flood_depth_m", "flood_depth_text",
               "inundation_duration_hours", "latitude", "longitude", "confidence", "extraction_method",
               "review_status", "evidence_locator"]


def _md(s) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def stage_review() -> dict:
    import pandas as pd
    sil = silver_records()
    cat = pd.read_parquet(CATALOG_DIR / "events.parquet").set_index("silver_record_id")
    rv = pd.read_csv(REPORTS / "rain_validation.csv")
    dd = pd.read_csv(OUT / "rain_validation_dedup.csv").set_index("silver_record_id") if (OUT / "rain_validation_dedup.csv").exists() else None
    vf = pd.read_csv(OUT / "verbatim_fields.csv") if (OUT / "verbatim_fields.csv").exists() else None
    from src.events.rain_validation import PRIMARY_BASES
    used_all = list(dict.fromkeys(rv.sort_values(["event_date", "source_name", "location_text"])["silver_record_id"]))
    primary = set()
    if dd is not None:
        primary = set(dd.index[dd["kept"] & dd["period_basis"].isin(PRIMARY_BASES)])
    a0 = [r for r in used_all if r in primary]
    used = [r for r in used_all if r not in primary]
    agent_ids = set()
    agent_csv = REPORTS / "u1b" / "silver_records_agent.csv"
    if agent_csv.exists():
        agent_ids = set(pd.read_csv(agent_csv)["record_id"])
    # keep existing checkbox states and notes (by record id) from the current sheet
    state = {}
    path = REPORTS / "U1_SILVER_REVIEW.md"
    if path.exists():
        cur = None
        for line in path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^### .*`(fer_[0-9a-f]+)`\s*$", line)
            if m:
                cur = m.group(1)
                continue
            m = re.match(r"^- \[([ xX])\] Duyệt \(đúng nguyên văn, đúng trường\) — Ghi chú: ?(.*)$", line)
            if m and cur:
                state[cur] = (m.group(1).lower() == "x", m.group(2))
                cur = None
    flood = sorted([k for k, v in sil.items() if v["event_type"] == "flood_observation" and k not in set(used)],
                   key=lambda k: (str(cat.loc[k, "event_date_start"]) if k in cat.index else "", sil[k]["evidence_id"], k))
    flood = [k for k in flood if k not in set(a0)]
    L = ["# U1 — Phiếu duyệt bản ghi Silver (người dùng)", "",
         f"Tạo {datetime.now(UTC):%Y-%m-%d %H:%M} UTC bởi `python -m src.events.u1_followup --stage review`. "
         "Mỗi bản ghi một ô đánh dấu; đánh `[x]` khi đã duyệt, ghi chú sửa ở dòng \"Ghi chú\". "
         "Agent **không** thay đổi `review_status` trong Silver; phiếu này chỉ là danh sách kiểm tra.", "",
         f"- Phần A0: {len(a0)} bản ghi **tập chính** của U1.4 (REPORT_U1 §15.3; kỳ có giờ hoặc trưa–trưa, đã bỏ trùng) — tiêu đề gắn [TẬP CHÍNH].",
         f"- Phần A: {len(used)} bản ghi U1.4 còn lại (ngoài tập chính, trùng, hoặc kỳ chưa rõ).",
         f"- Phần B: {len(flood)} bản ghi `flood_observation` còn lại (bản ghi cũ và mới).", "",
         "Đoạn nguyên văn = `extracted_text` của Silver (đã kiểm tra là chuỗi con của artifact cho 99 bản ghi U1.2 và 94 bản ghi U1.1b). "
         "Các trường đã điền = mọi trường không rỗng của bản ghi Silver; dòng \"Catalog\" là giá trị suy ra ở U1.3/U1.4, không nằm trong Silver.", ""]

    def block(i: int, rid: str, part: str) -> list[str]:
        rec = sil[rid]
        c = cat.loc[rid] if rid in cat.index else None
        src = c["source_name"] if c is not None else "?"
        tag = "[TẬP CHÍNH] " if rid in primary else ""
        head = f"### {part}{i}. {tag}{src} — {_md(rec.get('location_text') or '')} — `{rid}`"
        checked, note = state.get(rid, (False, ""))
        out = [head, "", f"- [{'x' if checked else ' '}] Duyệt (đúng nguyên văn, đúng trường) — Ghi chú: {note}".rstrip() + ("" if note else " "), ""]
        et = rec.get("extracted_text")
        out.append(f"> {_md(et)}" if et else "> **(không có `extracted_text` — bản ghi cũ trước U1, không qua validator; D103)**")
        out.append("")
        filled = [(k, rec.get(k)) for k in FIELD_ORDER if rec.get(k) not in (None, "")]
        out.append("| Trường | Giá trị |")
        out.append("|---|---|")
        out += [f"| `{k}` | {_md(v)} |" for k, v in filled]
        out.append(f"| `evidence_id` | {rec['evidence_id']} |")
        if c is not None:
            extra = f"precision {c['time_precision']}; cấp {c['location_level']}; geo {c['geo_status']}"
            if c["feature_name"]:
                extra += f" → {_md(c['feature_name'])}"
            if c["rainfall_qualifier"]:
                extra += f"; qualifier {c['rainfall_qualifier']}"
            out.append(f"| Catalog | {extra} |")
        if vf is not None:
            v = vf[vf["record_id"] == rid]
            if len(v):
                bad = v[~v["verbatim"]]
                out.append("| Nguyên văn trường (follow-up §3) | " + ("tất cả đúng nguyên văn" if bad.empty else "; ".join(
                    f"`{f}` không nguyên văn: {_md(dg)}" for f, dg in zip(bad["field"], bad["diagnosis"]))) + " |")
        if rid in agent_ids:
            out.append("| Nguyên văn trường (U1.1b) | location_text, temporal_text, flood_depth_text đều nguyên văn (201/201 trường của 94 bản ghi) |")
        r = rv[rv["silver_record_id"] == rid]
        if len(r):
            r = r.iloc[0]
            dup = "" if dd is None or rid not in dd.index else ("; giữ lại" if dd.loc[rid, "kept"] else "; **trùng, bị bỏ ở follow-up**")
            warn = (" — **kỳ 24 h là quy ước của agent (D107); đoạn văn chỉ nêu giờ kết thúc, xem "
                    "`reports/u1/followup/end_only_24h_passages.md`**") if r["period_basis"] == "end_only_24h" else ""
            out.append(f"| U1.4 | kỳ {r['period_local']} ({r['period_basis']}, {r['period_hours']} h){dup}{warn} |")
        out.append("")
        return out

    L += ["## A0. Tập chính U1.4 [TẬP CHÍNH]", ""]
    for i, rid in enumerate(a0, 1):
        L += block(i, rid, "A0.")
    L += ["## A. Bản ghi U1.4 còn lại", ""]
    for i, rid in enumerate(used, 1):
        L += block(i, rid, "A")
    L += ["## B. Bản ghi flood_observation", ""]
    for i, rid in enumerate(flood, 1):
        L += block(i, rid, "B")
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text("\n".join(L) + "\n", encoding="utf-8")
    tmp.replace(path)
    kept_states = sum(1 for r in a0 + used + flood if r in state and (state[r][0] or state[r][1]))
    return {"part_a0_primary": len(a0), "part_a": len(used), "part_b": len(flood), "checkboxes": len(a0) + len(used) + len(flood),
            "states_carried_over": kept_states, "path": rel(path)}


STAGES = {"prov": stage_prov, "verbatim": stage_verbatim, "rain": stage_rain, "sweep": stage_sweep, "review": stage_review}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=list(STAGES) + ["all"], default="all")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in (STAGES if a.stage == "all" else [a.stage]):
        res = STAGES[name]()
        print(name, json.dumps(res, ensure_ascii=False, default=str)[:3000])
    import psutil
    peak = getattr(psutil.Process().memory_info(), "peak_wset", None)  # Windows peak working set
    print(f"PEAK_RSS_GB={peak / 2**30:.3f}" if peak else "PEAK_RSS_GB=n/a")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
