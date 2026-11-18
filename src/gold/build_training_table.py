"""P4A.3 — Gold table A (pilot v0): label rows + satellite-only features (src/gold/features.py).

data/gold/met_training/pilot_v0/table_A/season=YYYY/month=MM/part.parquet (+ part.meta.json), schema.json.
Resumable per month (config hash + code hash + input fingerprint). Frame-mode temporal checks (G2) are
run on every partition before it is written; a violating partition is not written (exit 4).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
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

SOURCE = "gsmap_now"
OUT = P.GOLD / "table_A"
VERSION = "tableA-1"
CODE = [ROOT / "src/gold/features.py", ROOT / "src/gold/build_training_table.py"]


def silver_file(d: date) -> Path:
    P.guard_source(SOURCE)
    return P.SILVER / SOURCE / f"date={d.isoformat()}" / "part.parquet"


def load_windows(days: list[date], grid: F.Grid) -> dict:
    parts = []
    for d in days:
        f = silver_file(d)
        if f.is_file():
            parts.append(pd.read_parquet(f, columns=["pixel_id", "window_end_utc", "rain_rate_mm_h", "rain_rate_gc_mm_h"]))
    if not parts:
        return {}
    df = pd.concat(parts, ignore_index=True)
    df["window_end_utc"] = df["window_end_utc"].astype("datetime64[ns, UTC]")
    df = df.drop_duplicates(["pixel_id", "window_end_utc"])
    i, j = grid.ij(df["pixel_id"].to_numpy())
    ok = (i >= 0) & (i < grid.n_lat) & (j >= 0) & (j < grid.n_lon)
    df, i, j = df[ok], i[ok], j[ok]
    out = {}
    for we, idx in df.groupby("window_end_utc").indices.items():
        gc = np.full((grid.n_lat, grid.n_lon), np.nan, "float32"); raw = gc.copy()
        gc[i[idx], j[idx]] = df["rain_rate_gc_mm_h"].to_numpy()[idx]
        raw[i[idx], j[idx]] = df["rain_rate_mm_h"].to_numpy()[idx]
        out[pd.Timestamp(we)] = (gc, raw)
    return out


def groups_for(cols: list[str]) -> list[FeatureGroup]:
    by = {}
    for c in cols:
        by.setdefault(F.group_of(c), []).append(c)
    return [FeatureGroup(g, F.GROUP_KIND[g], tuple(cs)) for g, cs in by.items()]


def build_month(y: int, m: int, cfg: dict, grid: F.Grid, p: dict) -> tuple[pd.DataFrame, dict, list]:
    lab = pd.read_parquet(P.GOLD / "labels" / f"season={y}" / f"month={m:02d}" / "part.parquet")
    days = month_days(y, m, cfg)
    wins = load_windows([days[0] - timedelta(days=1)] + days, grid)
    t0s = pd.DatetimeIndex(np.sort(lab["t0"].unique())).tz_convert("UTC").as_unit("ns")
    pix = np.sort(lab["pixel_id"].unique())
    feats = []
    for s in range(0, len(t0s), 240):                      # 5-day blocks keep memory small
        tt = t0s[s:s + 240]
        gc, raw = F.stack_windows(wins, tt, p, grid.n_lat, grid.n_lon)
        feats.append(F.compute_features(gc, raw, tt, pix, p, grid))
    fe = pd.concat(feats, ignore_index=True)
    fcols = F.feature_columns(fe)
    lab["t0"] = lab["t0"].astype("datetime64[ns, UTC]")
    out = lab.merge(fe, on=["pixel_id", "t0"], how="left", validate="many_to_one")
    for c in ("t0", "prediction_time", "frame_start", "target_time", "gsmap__event_time", "gsmap__available_time"):
        out[c] = out[c].astype("datetime64[us, UTC]")
    fc = FrameConfig.from_yaml(P.CFG_PATH)
    viol = check_frame_table(out, groups_for(fcols), fcols, fc)
    st = {"season": y, "month": m, "rows": len(out), "label_rows": len(lab), "windows_loaded": len(wins),
          "g2_violations": [str(v) for v in viol]}
    return out, st, fcols


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="*", help="YYYY-MM (default: whole scope)")
    ap.add_argument("--max-minutes", type=float, default=18.0)
    a = ap.parse_args(argv)
    P.require_guard()
    lg = P.get_logger("p4a3_table_A")
    cfg = P.load_cfg()
    p = F.feature_params(cfg)
    grid = F.Grid()
    ch = P.cfg_hash({"features": cfg["features"], "frames": cfg["frames"], "availability": cfg["availability"],
                     "v": VERSION, "code": P.code_hash(*CODE)})
    months = months_in_scope(cfg)
    if a.months:
        months = [mm for mm in months if f"{mm[0]}-{mm[1]:02d}" in a.months]
    t_start, built, skipped, bad, fcols = time.time(), 0, 0, 0, None
    for y, m in months:
        part = OUT / f"season={y}" / f"month={m:02d}" / "part.parquet"
        meta_p = part.with_name("part.meta.json")
        days = month_days(y, m, cfg)
        lab_p = P.GOLD / "labels" / f"season={y}" / f"month={m:02d}" / "part.parquet"
        fp = P.sha256_bytes((P.sha256_file(lab_p) + "".join(
            P.sha256_file(silver_file(d)) for d in [days[0] - timedelta(days=1)] + days if silver_file(d).is_file())).encode())
        if part.is_file() and meta_p.is_file():
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            if meta.get("config_hash") == ch and meta.get("input_fingerprint") == fp:
                skipped += 1; fcols = meta["feature_columns"]; continue
        if (time.time() - t_start) / 60 > a.max_minutes:
            print(f"time budget reached; rerun to continue (built={built})"); break
        df, st, fcols = build_month(y, m, cfg, grid, p)
        if st["g2_violations"]:
            bad += 1
            lg.error("G2 FAIL %s-%02d %s", y, m, st["g2_violations"])
            print(f"G2 FAIL {y}-{m:02d}: {st['g2_violations'][:3]}")
            continue
        P.write_parquet(df, part)
        st["nan_share"] = {c: round(float(df[c].isna().mean()), 5) for c in fcols}
        P.write_json(meta_p, {"config_hash": ch, "input_fingerprint": fp, "version": VERSION,
                              "feature_columns": fcols, "stats": st})
        built += 1
        lg.info("built %s-%02d rows=%d rss=%.2fGB", y, m, len(df), P.rss_gb())
        P.check_rss(cfg["resources"]["max_rss_gb"])
    if fcols:
        schema = {"version": VERSION, "config_hash": ch, "code_hash": P.code_hash(*CODE),
                  "features": [{"name": c, "dtype": "float32", "group": F.group_of(c),
                                "kind": F.GROUP_KIND[F.group_of(c)]} for c in fcols],
                  "availability_rule": {"gsmap": "window_end <= t0 and window_end + "
                                        f"{p['avail_after']} min <= prediction_time = t0 + {p['issue']} min "
                                        "(availability_model = realtime_measured)",
                                        "calendar/static": "deterministic from t0 / pixel"},
                  "keys": ["pixel_id", "t0", "lead_min"],
                  "labels": ["label_rate_mm_h", "y_0p5", "y_2p5", "y_7p6"]}
        P.write_json(OUT / "schema.json", schema)
    print(f"table_A: built={built} skipped={skipped} g2_fail_partitions={bad} features={len(fcols or [])} "
          f"in {time.time() - t_start:.0f}s rss={P.rss_gb():.2f}GB")
    return 4 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
