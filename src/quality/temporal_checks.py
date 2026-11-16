"""Point-in-time (leakage) checks per temporal_contract.md v1.0 section 12.

A training/feature table is a DataFrame with one row per (location, prediction_time,
horizon) and these columns:

  prediction_time, target_time, horizon_min           (required)
  <feature columns>                                   (must be in FEATURE_ALLOWLIST)
  <group>__available_time, <group>__event_time        (per feature group)
  <group>__issue_time, <group>__ingestion_time        (forecast groups)

Feature groups are declared with ``FeatureGroup``. All timestamps must be
tz-aware UTC.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

VALID_HORIZONS = frozenset({30, 60})

# Explicit allowlist of feature columns that may enter a model. Anything else,
# in particular anything derived from IMERG Final (a delayed, reprocessed
# product that never exists at prediction time), is rejected.
FEATURE_ALLOWLIST = frozenset({
    # GSMaP NOW as-of features (realtime candidate)
    "gsmap_rate_mm_h", "gsmap_rate_gc_mm_h", "gsmap_age_min",
    "gsmap_rate_lag1_mm_h", "gsmap_rate_lag2_mm_h",
    "gsmap_nbr_max_mm_h", "gsmap_nbr_mean_mm_h",
    # IMERG Early/Late (available only when their own available_time <= t)
    "imerg_early_rate_mm_h", "imerg_late_rate_mm_h",
    # forecasts (Open-Meteo, P2)
    "fc_precip_mm_h", "fc_lead_min",
    # calendar / static
    "hour_utc", "hour_local", "day_of_year", "lat", "lon",
    "is_city_pixel", "is_core9_pixel",
})

FORBIDDEN_FEATURE_PATTERNS = (
    re.compile(r"imerg_final", re.I),
    re.compile(r"(^|_)final(_|$)", re.I),
    re.compile(r"^target_", re.I),
)

KEY_COLUMNS = ("prediction_time", "target_time", "horizon_min")


@dataclass(frozen=True)
class FeatureGroup:
    name: str
    kind: str = "observation"            # observation | forecast | static
    columns: tuple[str, ...] = field(default_factory=tuple)

    @property
    def available_col(self) -> str:
        return f"{self.name}__available_time"

    @property
    def event_col(self) -> str:
        return f"{self.name}__event_time"

    @property
    def issue_col(self) -> str:
        return f"{self.name}__issue_time"

    @property
    def ingestion_col(self) -> str:
        return f"{self.name}__ingestion_time"


@dataclass
class Violation:
    rule: str
    detail: str
    n_rows: int = 0

    def __str__(self) -> str:
        return f"[{self.rule}] {self.detail} (rows={self.n_rows})"


class TemporalContractError(AssertionError):
    pass


def is_tz_aware_utc(s: pd.Series) -> bool:
    """True only for datetime columns whose zone is UTC itself (not merely a +00:00 zone)."""
    tz = getattr(s.dtype, "tz", None)
    return tz is not None and str(tz) in {"UTC", "Etc/UTC"}


def _time_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.endswith("_time") or c.endswith("_time_utc")]


def check_feature_table(df: pd.DataFrame, groups: list[FeatureGroup],
                        allowlist: frozenset[str] = FEATURE_ALLOWLIST) -> list[Violation]:
    v: list[Violation] = []
    for c in KEY_COLUMNS:
        if c not in df.columns:
            v.append(Violation("schema", f"missing required column {c}"))
    if v:
        return v

    # tz-aware UTC
    for c in _time_columns(df):
        if not pd.api.types.is_datetime64_any_dtype(df[c]) or not is_tz_aware_utc(df[c]):
            v.append(Violation("tz_utc", f"column {c} is not tz-aware UTC (dtype={df[c].dtype})", len(df)))
    if any(x.rule == "tz_utc" for x in v):
        return v

    pt = df["prediction_time"]

    # horizon and target time
    bad_h = ~df["horizon_min"].isin(VALID_HORIZONS)
    if bad_h.any():
        v.append(Violation("horizon", f"horizon_min not in {sorted(VALID_HORIZONS)}", int(bad_h.sum())))
    bad_t = ~(df["target_time"] > pt)
    if bad_t.any():
        v.append(Violation("target_after_prediction", "target_time <= prediction_time", int(bad_t.sum())))
    exp_t = pt + pd.to_timedelta(df["horizon_min"], unit="min")
    bad_e = (df["target_time"] != exp_t) & ~bad_h
    if bad_e.any():
        v.append(Violation("target_equals_horizon", "target_time != prediction_time + horizon", int(bad_e.sum())))

    # feature columns: allowlist and forbidden patterns
    declared = {c for g in groups for c in g.columns}
    meta_cols = set(KEY_COLUMNS) | {c for c in df.columns if "__" in c} | \
        {"pixel_id", "h3_id", "location_id"} | {c for c in df.columns if c.startswith("target_")}
    feature_cols = [c for c in df.columns if c not in meta_cols]
    for c in feature_cols:
        if any(p.search(c) for p in FORBIDDEN_FEATURE_PATTERNS):
            v.append(Violation("forbidden_feature", f"feature column {c} is forbidden (IMERG Final / target)"))
        elif c not in allowlist:
            v.append(Violation("not_allowlisted", f"feature column {c} is not in FEATURE_ALLOWLIST"))
        if c not in declared:
            v.append(Violation("undeclared_feature", f"feature column {c} belongs to no feature group"))

    # per-group point-in-time rules
    for g in groups:
        if g.kind == "static":
            continue
        for col in g.columns:
            if col not in df.columns:
                v.append(Violation("schema", f"group {g.name}: column {col} missing"))
        if g.available_col not in df.columns:
            v.append(Violation("schema", f"group {g.name}: {g.available_col} missing"))
            continue
        present = df[list(c for c in g.columns if c in df.columns)].notna().any(axis=1) if g.columns else \
            pd.Series(True, index=df.index)
        av = df[g.available_col]
        bad = present & ~(av <= pt)
        if bad.any():
            v.append(Violation("available_time", f"group {g.name}: available_time > prediction_time "
                               "(or missing while a value is present)", int(bad.sum())))
        if g.kind == "observation":
            if g.event_col not in df.columns:
                v.append(Violation("schema", f"group {g.name}: {g.event_col} missing"))
            else:
                bad = present & ~(df[g.event_col] <= pt)
                if bad.any():
                    v.append(Violation("event_time", f"group {g.name}: event_time > prediction_time "
                                       "(future observation)", int(bad.sum())))
                bad = present & ~(df[g.event_col] <= av)
                if bad.any():
                    v.append(Violation("event_before_available",
                                       f"group {g.name}: event_time > available_time", int(bad.sum())))
        if g.kind == "forecast":
            if g.issue_col not in df.columns:
                v.append(Violation("schema", f"group {g.name}: {g.issue_col} missing"))
            else:
                bad = present & ~(df[g.issue_col] <= pt)
                if bad.any():
                    v.append(Violation("forecast_issue_time", f"group {g.name}: issue_time > prediction_time",
                                       int(bad.sum())))
        if g.ingestion_col in df.columns:
            bad = present & ~(df[g.ingestion_col] <= pt)
            if bad.any():
                v.append(Violation("ingestion_time", f"group {g.name}: ingestion_time > prediction_time",
                                   int(bad.sum())))
    return v


def assert_feature_table(df: pd.DataFrame, groups: list[FeatureGroup], **kw) -> None:
    v = check_feature_table(df, groups, **kw)
    if v:
        raise TemporalContractError("; ".join(map(str, v)))


def as_of_latest(records: pd.DataFrame, times: pd.DataFrame, by: str = "pixel_id",
                 time_col: str = "prediction_time", available_col: str = "available_time_utc") -> pd.DataFrame:
    """As-of join (contract section 10): for each (by, t) the latest record with available_time <= t.

    Ties on available_time are broken by the latest event (window_end) so the
    result is deterministic.
    """
    for c in (available_col,):
        if not is_tz_aware_utc(records[c]):
            raise TemporalContractError(f"{c} must be tz-aware UTC")
    if not is_tz_aware_utc(times[time_col]):
        raise TemporalContractError(f"{time_col} must be tz-aware UTC")
    # Parquet columns may carry different resolutions (ms vs us); merge_asof
    # requires identical key dtypes, so normalise both join keys to ns UTC.
    records = records.assign(**{available_col: records[available_col].astype("datetime64[ns, UTC]")})
    times = times.assign(**{time_col: times[time_col].astype("datetime64[ns, UTC]")})
    r = records.sort_values([available_col] + (["window_end_utc"] if "window_end_utc" in records else []))
    r = r.drop_duplicates([by, available_col], keep="last")
    left = times.sort_values(time_col)
    out = pd.merge_asof(left, r.sort_values(available_col), left_on=time_col, right_on=available_col,
                        by=by, direction="backward", allow_exact_matches=True)
    leaked = out[available_col].notna() & (out[available_col] > out[time_col])
    if leaked.any():  # defensive: merge_asof backward must never do this
        raise TemporalContractError(f"as-of join produced {int(leaked.sum())} future rows")
    return out
