"""Load Gold tables by time split (pilot v0). Splits are by t0 date (UTC) only, never by pixel.
Rows whose target_time falls after the split end are purged (no label overlap across splits)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.gold import p4a_common as P  # noqa: E402

LABELS = ["label_rate_mm_h", "y_0p5", "y_2p5", "y_7p6"]
KEYS = ["pixel_id", "t0", "lead_min", "frame_start", "season"]


def feature_list(table: str = "table_A") -> list[str]:
    s = json.loads((P.GOLD / table / "schema.json").read_text(encoding="utf-8"))
    return [f["name"] for f in s["features"]]


def split_range(cfg: dict, split: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    a, b = cfg["splits"][split]
    return pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC") + pd.Timedelta("1D")


def load_split(cfg: dict, split: str | tuple, columns: list[str] | None = None, table: str = "table_A",
               lead: int | None = None) -> pd.DataFrame:
    lo, hi = split_range(cfg, split) if isinstance(split, str) else \
        (pd.Timestamp(split[0], tz="UTC"), pd.Timestamp(split[1], tz="UTC") + pd.Timedelta("1D"))
    cols = None if columns is None else list(dict.fromkeys(KEYS + ["target_time"] + LABELS + columns))
    parts = []
    for part in sorted((P.GOLD / table).glob("season=*/month=*/part.parquet")):
        y = int(part.parent.parent.name.split("=")[1]); m = int(part.parent.name.split("=")[1])
        ms = pd.Timestamp(year=y, month=m, day=1, tz="UTC")
        if ms >= hi or ms + pd.offsets.MonthBegin(1) <= lo:
            continue
        df = pd.read_parquet(part, columns=cols)
        if lead is not None:
            df = df[df["lead_min"] == lead]
        df = df[(df["t0"] >= lo) & (df["t0"] < hi) & (df["target_time"] <= hi)]
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    out["day"] = out["t0"].dt.tz_convert("Asia/Ho_Chi_Minh").dt.date.astype(str)
    return out


def thr_col(t: float) -> str:
    return "y_" + f"{t:g}".replace(".", "p")


def asof_latest(df: pd.DataFrame, ks=range(7)) -> np.ndarray:
    arr = np.vstack([df[f"gc_k{k}"].to_numpy("float64") for k in ks])
    first = np.argmax(~np.isnan(arr), axis=0)
    return np.where((~np.isnan(arr)).any(0), arr[first, np.arange(arr.shape[1])], np.nan)


def max_avail_0_60(df: pd.DataFrame) -> np.ndarray:
    arr = np.vstack([df[f"gc_k{k}"].to_numpy("float64") for k in (0, 1, 2)])
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmax(arr, axis=0)
