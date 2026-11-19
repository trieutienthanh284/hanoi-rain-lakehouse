"""P4A.2 — label table (pilot v0): one row per (city pixel, t0, lead_min), IMERG Final only.

data/gold/met_training/pilot_v0/labels/season=YYYY/month=MM/part.parquet (+ part.meta.json).
Resumable per month: a partition is skipped when its meta has the same config hash and input
fingerprint. IMERG Late is never read (labels_provisional for Oct 2025 is deferred).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.gold import p4a_common as P  # noqa: E402

SOURCE = "imerg_final"
OUT = P.GOLD / "labels"
VERSION = "labels-2"


def thr_col(t: float) -> str:
    return "y_" + f"{t:g}".replace(".", "p")


def months_in_scope(cfg: dict) -> list[tuple[int, int]]:
    s, e = date.fromisoformat(cfg["scope"]["start"]), date.fromisoformat(cfg["scope"]["end"])
    out, d = [], s.replace(day=1)
    while d <= e:
        if d.month in cfg["scope"]["season_months"]:
            out.append((d.year, d.month))
        d = (d.replace(day=28) + timedelta(days=4)).replace(day=1)
    return out


def month_days(y: int, m: int, cfg: dict) -> list[date]:
    s, e = date.fromisoformat(cfg["scope"]["start"]), date.fromisoformat(cfg["scope"]["end"])
    d, out = date(y, m, 1), []
    while d.month == m:
        if s <= d <= e:
            out.append(d)
        d += timedelta(days=1)
    return out


def silver_file(d: date, source: str = SOURCE) -> Path:
    P.guard_source(source)
    return P.SILVER / source / f"date={d.isoformat()}" / "part.parquet"


def load_final(days: list[date], source: str = SOURCE) -> pd.DataFrame:
    """IMERG Silver rows of the city pixels (source imerg_final; imerg_late only for P4B provisional labels)."""
    parts = []
    for d in days:
        f = silver_file(d, source)
        if f.is_file():
            t = pd.read_parquet(f, columns=["pixel_id", "interval_start_utc", "rain_rate_mm_h", "is_city_pixel"])
            parts.append(t[t["is_city_pixel"]].drop(columns="is_city_pixel"))
    if not parts:
        return pd.DataFrame(columns=["pixel_id", "interval_start_utc", "rain_rate_mm_h"])
    df = pd.concat(parts, ignore_index=True)
    df["interval_start_utc"] = df["interval_start_utc"].astype("datetime64[ns, UTC]")
    return df


def build_month(y: int, m: int, cfg: dict, city_pixels: np.ndarray, source: str = SOURCE,
                days: list[date] | None = None) -> tuple[pd.DataFrame, dict]:
    days = days or month_days(y, m, cfg)
    fr = cfg["frames"]
    nxt = days[-1] + timedelta(days=1)
    fin = load_final(days + [nxt], source)
    t0 = pd.DatetimeIndex(np.concatenate([pd.date_range(pd.Timestamp(d, tz="UTC"), periods=24 * 60 // fr['analysis_time_step_min'],
                                                        freq=f"{fr['analysis_time_step_min']}min").values for d in days]),
                          tz="UTC")
    grid = pd.MultiIndex.from_product([city_pixels, t0, fr["lead_min"]], names=["pixel_id", "t0", "lead_min"]).to_frame(index=False)
    grid["t0"] = grid["t0"].astype("datetime64[ns, UTC]")
    grid["prediction_time"] = grid["t0"] + pd.Timedelta(minutes=fr["issue_offset_min"])
    grid["frame_start"] = grid["t0"] + pd.to_timedelta(grid["lead_min"], unit="min")
    grid["target_time"] = grid["frame_start"] + pd.Timedelta(minutes=fr["frame_length_min"])
    lab = fin.rename(columns={"interval_start_utc": "frame_start", "rain_rate_mm_h": "label_rate_mm_h"})
    lab = lab.drop_duplicates(["pixel_id", "frame_start"])
    out = grid.merge(lab, on=["pixel_id", "frame_start"], how="left")
    n_all = len(out)
    miss = out["label_rate_mm_h"].isna()
    miss_by_lead = out[miss].groupby("lead_min").size().to_dict()
    out = out[~miss].reset_index(drop=True)
    for t in cfg["targets"]["thresholds_mm_h"]:
        out[thr_col(t)] = (out["label_rate_mm_h"] >= t).astype("int8")
    for c in ("t0", "prediction_time", "frame_start", "target_time"):
        out[c] = out[c].astype("datetime64[us, UTC]")
    out["label_source"] = f"{source}_v07"
    out["season"] = np.int16(y)
    out["lead_min"] = out["lead_min"].astype("int16")
    stats = {"season": y, "month": m, "days": len(days), "t0": len(t0), "pixels": int(len(city_pixels)),
             "expected_rows": n_all, "missing_label_rows": int(miss.sum()),
             "missing_by_lead": {str(k): int(v) for k, v in miss_by_lead.items()}, "rows": len(out)}
    return out, stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="*", help="YYYY-MM (default: whole scope)")
    a = ap.parse_args(argv)
    P.require_guard()
    lg = P.get_logger("p4a2_labels")
    cfg = P.load_cfg()
    ch = P.cfg_hash({"frames": cfg["frames"], "targets": cfg["targets"], "scope": cfg["scope"], "v": VERSION})
    dom = pd.read_parquet(silver_file(date(2024, 9, 6)), columns=["pixel_id", "is_city_pixel"]).drop_duplicates()
    city = np.sort(dom.loc[dom["is_city_pixel"], "pixel_id"].to_numpy())
    months = months_in_scope(cfg)
    if a.months:
        months = [mm for mm in months if f"{mm[0]}-{mm[1]:02d}" in a.months]
    allstats, done, skipped = [], 0, 0
    for y, m in months:
        part = OUT / f"season={y}" / f"month={m:02d}" / "part.parquet"
        meta_p = part.with_name("part.meta.json")
        days = month_days(y, m, cfg)
        fp = P.sha256_bytes("".join(P.sha256_file(silver_file(d)) for d in days + [days[-1] + timedelta(days=1)]
                                    if silver_file(d).is_file()).encode())
        if part.is_file() and meta_p.is_file():
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            if meta.get("config_hash") == ch and meta.get("input_fingerprint") == fp:
                allstats.append(meta["stats"]); skipped += 1; continue
        df, st = build_month(y, m, cfg, city)
        P.write_parquet(df, part)
        P.write_json(meta_p, {"config_hash": ch, "input_fingerprint": fp, "stats": st, "version": VERSION})
        allstats.append(st); done += 1
        lg.info("built %s-%02d %s", y, m, json.dumps(st))
        P.check_rss(cfg["resources"]["max_rss_gb"])
    s = pd.DataFrame(allstats)
    print(f"months built={done} skipped={skipped}; rows={int(s['rows'].sum()):,} expected={int(s['expected_rows'].sum()):,} "
          f"missing={int(s['missing_label_rows'].sum()):,}; city pixels={len(city)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
