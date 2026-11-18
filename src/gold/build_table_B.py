"""P4A.6 — Gold table B = table A + NWP features (Open-Meteo previous_day1, GFS and ECMWF IFS 0.25),
seasons 2024-2025 only. data/gold/met_training/pilot_v0/table_B/season=YYYY/month=MM/part.parquet + schema.json.
Resumable per month; frame-mode G2 checks before writing (exit 4 on violation)."""
from __future__ import annotations

import argparse, json, sys, time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.gold import features as F  # noqa: E402
from src.gold import p4a_common as P  # noqa: E402
from src.gold.build_labels import month_days, months_in_scope  # noqa: E402
from src.quality.temporal_checks import FeatureGroup, FrameConfig, check_frame_table  # noqa: E402

SOURCE = "openmeteo"
OUT = P.GOLD / "table_B"
VERSION = "tableB-1"
CODE = [ROOT / "src/gold/features.py", ROOT / "src/gold/build_table_B.py"]
OM_COLS = ["pixel_id", "model", "interval_start_utc", "precip_mm", "forecast_issue_time_utc", "available_time_utc",
           "lat", "lon"]


def silver_file(d: date) -> Path:
    P.guard_source(SOURCE)
    return P.SILVER / SOURCE / f"date={d.isoformat()}" / "part.parquet"


def load_om(days: list[date]) -> pd.DataFrame:
    parts = [pd.read_parquet(silver_file(d), columns=OM_COLS) for d in days if silver_file(d).is_file()]
    df = pd.concat(parts, ignore_index=True)
    for c in ("interval_start_utc", "forecast_issue_time_utc", "available_time_utc"):
        df[c] = df[c].astype("datetime64[ns, UTC]")
    return df


def groups_for(cols):
    by = {}
    for c in cols:
        by.setdefault(F.group_of(c), []).append(c)
    return [FeatureGroup(g, F.GROUP_KIND[g], tuple(cs)) for g, cs in by.items()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--months", nargs="*")
    a = ap.parse_args(argv)
    P.require_guard()
    lg = P.get_logger("p4a6_table_B")
    cfg = P.load_cfg()
    ch = P.cfg_hash({"v": VERSION, "code": P.code_hash(*CODE), "frames": cfg["frames"],
                     "om_rule": (ROOT / cfg["availability"]["openmeteo_rule_file"]).read_text(encoding="utf-8")})
    months = [mm for mm in months_in_scope(cfg) if mm[0] >= 2024]
    if a.months:
        months = [mm for mm in months if f"{mm[0]}-{mm[1]:02d}" in a.months]
    fc = FrameConfig.from_yaml(P.CFG_PATH)
    t0, built, skipped, bad, fcols = time.time(), 0, 0, 0, None
    pmap = None
    for y, m in months:
        part = OUT / f"season={y}" / f"month={m:02d}" / "part.parquet"
        meta_p = part.with_name("part.meta.json")
        days = month_days(y, m, cfg)
        dd = [days[0] - timedelta(days=1)] + days + [days[-1] + timedelta(days=1)]
        a_part = P.GOLD / "table_A" / f"season={y}" / f"month={m:02d}" / "part.parquet"
        fp = P.sha256_bytes((P.sha256_file(a_part) + "".join(P.sha256_file(silver_file(d)) for d in dd
                                                              if silver_file(d).is_file())).encode())
        if part.is_file() and meta_p.is_file():
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            if meta.get("config_hash") == ch and meta.get("input_fingerprint") == fp:
                skipped += 1; fcols = meta["feature_columns"]; continue
        A = pd.read_parquet(a_part)
        om = load_om(dd)
        if pmap is None:
            pmap = F.nwp_point_map(om, np.sort(A["pixel_id"].unique()))
        rows = A[["pixel_id", "prediction_time", "frame_start"]].copy()
        rows["prediction_time"] = rows["prediction_time"].astype("datetime64[ns, UTC]")
        rows["frame_start"] = rows["frame_start"].astype("datetime64[ns, UTC]")
        nw = F.compute_nwp_features(rows.reset_index(drop=True), om, pmap)
        for c in [c for c in nw.columns if c.endswith("_time")]:
            nw[c] = nw[c].astype("datetime64[us, UTC]")
        B = pd.concat([A.reset_index(drop=True), nw], axis=1)
        a_feats = json.loads((P.GOLD / "table_A" / "schema.json").read_text(encoding="utf-8"))["features"]
        fcols = [f["name"] for f in a_feats] + [c for c in nw.columns if "__" not in c]
        viol = check_frame_table(B, groups_for(fcols), fcols, fc)
        if viol:
            bad += 1; lg.error("G2 FAIL %s-%02d %s", y, m, viol); print(f"G2 FAIL {y}-{m:02d}: {viol[:3]}"); continue
        P.write_parquet(B, part)
        st = {"rows": len(B), "nan_share_nwp": {c: round(float(B[c].isna().mean()), 5) for c in nw.columns if "__" not in c}}
        P.write_json(meta_p, {"config_hash": ch, "input_fingerprint": fp, "version": VERSION, "feature_columns": fcols,
                              "stats": st})
        built += 1
        lg.info("built %s-%02d rows=%d", y, m, len(B))
        P.check_rss(cfg["resources"]["max_rss_gb"])
    if fcols:
        P.write_json(OUT / "schema.json", {
            "version": VERSION, "config_hash": ch, "code_hash": P.code_hash(*CODE),
            "features": [{"name": c, "dtype": "float32", "group": F.group_of(c), "kind": F.GROUP_KIND[F.group_of(c)]}
                         for c in fcols],
            "availability_rule": {"gsmap": "as table_A", "nwp": "Open-Meteo previous_day1; forecast_issue_time <= "
                                  "prediction_time and available_time (= issue + delay, configs/openmeteo_availability.yaml) "
                                  "<= prediction_time", "nwp_points": "nearest of 81 points; mean of points within 0.3 deg"},
            "nwp_point_map": {str(k): v for k, v in (pmap or {}).items()},
            "keys": ["pixel_id", "t0", "lead_min"], "labels": ["label_rate_mm_h", "y_0p5", "y_2p5", "y_7p6"]})
    print(f"table_B: built={built} skipped={skipped} g2_fail={bad} features={len(fcols or [])} in {time.time() - t0:.0f}s "
          f"rss={P.rss_gb():.2f}GB")
    return 4 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
