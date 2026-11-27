"""Flood indicator (map icon): rolling rain accumulation → level none | possible | high.

Pure functions, reusable by serving (P5). Wording: "chỉ báo dựa trên lượng mưa tích lũy" — never a
flood probability or a FLOOD / NO FLOOD prediction.

History mode (implemented): A6h(t) / A24h(t) = observed rain in (t − 6 h, t] / (t − 24 h, t].
Serving mode (P5, documented only): A6h(t) = observed (t − 5 h, t] + nowcast (t, t + 1 h]; the forecast
term is not implemented here.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "flood_indicator.yaml"
LEVELS = ("none", "possible", "high")


def load_config(path: Path = CONFIG) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def thresholds(cfg: dict[str, Any], scale: str = "gauge", product: str | None = None) -> tuple[float, float, float]:
    """(a, b, long_mm) for the gauge scale or a proposed satellite scale of `product`."""
    if scale == "gauge":
        return float(cfg["a_mm"]), float(cfg["b_mm"]), float(cfg["long_rule"]["mm"])
    p = cfg["satellite_proposal"]["products"][product]
    return float(p["proposed_satellite_a_mm"]), float(p["proposed_satellite_b_mm"]), float(p["proposed_satellite_long_mm"])


class ProductNotUsedForIcon(ValueError):
    """The product is explicitly excluded from the icon by a user decision (e.g. GSMaP NOW raw, U1-D2)."""


def product_thresholds(cfg: dict[str, Any], product: str) -> tuple[float, float, float]:
    """(a, b, long_mm) decided for `product` in `product_thresholds` of the config.

    imerg_final / imerg_late: gauge scale 50/100/150 (U1-D1); gsmap_now_gc: 110/220/270 (U1-D2).
    Raises ProductNotUsedForIcon for products marked `not_used_for_icon` (gsmap_now) and KeyError for
    products without a decided entry. Never falls back to proposals (NEEDS_REVIEW / REJECTED blocks).
    """
    entry = (cfg.get("product_thresholds") or {}).get(product)
    if entry is None:
        raise KeyError(f"no decided thresholds for product {product!r} in product_thresholds")
    if entry.get("status") == "not_used_for_icon":
        raise ProductNotUsedForIcon(f"{product} is not used for the icon: {entry.get('reason', '')}")
    return float(entry["a_mm"]), float(entry["b_mm"]), float(entry["long_mm"])


def accumulate(rain_series, window_h: float):
    """Trailing accumulation (mm) over `window_h` hours ending at each index time.

    `rain_series`: pandas Series of rain amounts (mm per interval) on a regular, tz-aware index of
    interval END times. The value at t sums the intervals ending in (t − window_h, t]. Incomplete
    windows (missing intervals or the first window_h of the series) give NaN.
    """
    import pandas as pd
    s = rain_series.sort_index()
    if len(s) < 2:
        return pd.Series(float("nan"), index=s.index)
    step = (s.index[1:] - s.index[:-1]).min()
    full = pd.date_range(s.index[0], s.index[-1], freq=step)
    s = s.reindex(full)
    n = int(round(pd.Timedelta(hours=window_h) / step))
    return s.rolling(n, min_periods=n).sum()


def indicator_level(a6h: float | None, a24h: float | None, cfg: dict[str, Any],
                    scale: str = "gauge", product: str | None = None) -> str | None:
    """Level for one cell/time; None when A6h and A24h are both missing (no data ≠ no icon)."""
    a, b, long_mm = thresholds(cfg, scale, product)
    v6 = None if a6h is None or (isinstance(a6h, float) and math.isnan(a6h)) else float(a6h)
    v24 = None if a24h is None or (isinstance(a24h, float) and math.isnan(a24h)) else float(a24h)
    if v6 is None and v24 is None:
        return None
    level = "none"
    if v6 is not None:
        level = "high" if v6 >= b else "possible" if v6 >= a else "none"
    if v24 is not None and v24 >= long_mm:
        level = "high"
    return level


def indicator_levels(a6h, a24h, a: float, b: float, long_mm: float):
    """Vectorised levels (numpy arrays) → int codes 0 none, 1 possible, 2 high, -1 no data."""
    import numpy as np
    a6h = np.asarray(a6h, dtype=float)
    a24h = np.asarray(a24h, dtype=float)
    out = np.where(a6h >= b, 2, np.where(a6h >= a, 1, 0))
    out = np.where(a24h >= long_mm, 2, out)
    out = np.where(np.isnan(a6h) & ~(a24h >= long_mm), np.where(np.isnan(a24h), -1, out), out)
    return out
