"""Prediction module for the rainfall nowcast: v1 (P4B, current) and the PILOT v0 (P4A.8, kept loadable).
Rain nowcasting against IMERG (satellite estimate, not gauges); no flood prediction.

v1 (P4B.7, `models/v1/{A,B}/lead{L}_thr{k}/`, see `models/v1/MODEL_CARD.md`):

  from src.models.predict import load_v1, predict_v1
  mv = load_v1()
  out = predict_v1(feature_rows, mv, nwp=nwp_rows)

  * set B (set A + Open-Meteo `nwp_*`) is used for a row when every `nwp_*` input of the set-B model is present
    and, if the group times `nwp_<model>__issue_time` / `__available_time` are given, the run is the previous-day
    run named in `meta.json` `serving_rule` (issue <= prediction_time - 22 h, available_time <= prediction_time);
    otherwise (NWP missing or stale) the set-A model is used. Column `model_set` says which one ("A" / "B").
  * probabilities are the raw LightGBM outputs (calibration "raw" in every model); `yhat_<k>` = p >= cut of
    `meta.json`; the 7.6 mm/h class uses the nested rule when `heavy_rule == "nested"` (also above the 2.5 mm/h
    cut of the same set); rain classes as below.

v0 (unchanged API):

Input: the feature rows produced by `src/gold/features.py` (`compute_features`, one row per (pixel, t0)), i.e. the
single feature code path used for training. Output: one row per (pixel, t0, lead_min) with, per threshold,
the raw LightGBM probability `p_raw_<k>`, the isotonic-calibrated probability `p_<k>`, the binary decision
`yhat_<k>` (calibrated p >= the cut chosen on validation) and the rain class:
  0 = no rain, 1 = light (>= 0.5 mm/h), 2 = moderate (>= 2.5 mm/h), 3 = heavy (>= 7.6 mm/h); highest class wins.
Probabilities refer to IMERG Final (satellite estimate) exceeding the threshold in the frame window
[t0 + lead, t0 + lead + 30 min).

Model version: `models/pilot_v0/predict_manifest.json` maps each threshold to a variant directory (D74).

  from src.models.predict import load_version, predict
  mv = load_version()                      # default manifest
  out = predict(feature_rows, mv)
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models" / "pilot_v0"
MANIFEST = MODELS / "predict_manifest.json"
CLASS_NAMES = {0: "no_rain", 1: "light", 2: "moderate", 3: "heavy"}


def kstr(t: float) -> str:
    return f"{t:g}".replace(".", "p")


@dataclass
class OneModel:
    variant: str
    lead: int
    thr: float
    booster: object
    calibrator: object
    cut: float
    features: list[str]
    meta: dict = field(repr=False)


@dataclass
class ModelVersion:
    name: str
    leads: list[int]
    thresholds: list[float]
    models: dict                                  # (lead, thr) -> OneModel
    manifest: dict = field(repr=False)


def load_version(manifest: Path | dict | None = None, model_file: str = "model.txt") -> ModelVersion:
    """Load every (lead, threshold) model of a version. `model_file="model_prerefit.txt"` loads the pre-refit
    models (trained on the tune period only) for diagnostics."""
    import lightgbm as lgb
    man = manifest if isinstance(manifest, dict) else json.loads(Path(manifest or MANIFEST).read_text(encoding="utf-8"))
    models = {}
    for thr_s, variant in man["threshold_variant"].items():
        thr = float(thr_s)
        for lead in man["leads"]:
            d = MODELS / variant / f"lead{lead}_thr{kstr(thr)}"
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
            models[(lead, thr)] = OneModel(variant, lead, thr, lgb.Booster(model_file=str(d / model_file)),
                                           pickle.loads((d / "calibrator.pkl").read_bytes()),
                                           float(meta["cut"]["cut"]), list(meta["features"]), meta)
    thrs = sorted(float(t) for t in man["threshold_variant"])
    return ModelVersion(man["name"], list(man["leads"]), thrs, models, man)


def predict(features: pd.DataFrame, mv: ModelVersion, leads: list[int] | None = None) -> pd.DataFrame:
    """features: rows from features.compute_features (columns pixel_id, t0 + feature columns)."""
    out = []
    for lead in (leads or mv.leads):
        r = pd.DataFrame({"pixel_id": features["pixel_id"].to_numpy(), "t0": features["t0"].to_numpy(),
                          "lead_min": np.int16(lead)})
        r["t0"] = pd.DatetimeIndex(r["t0"]).tz_localize("UTC") if pd.DatetimeIndex(r["t0"]).tz is None else r["t0"]
        r["frame_start"] = r["t0"] + pd.Timedelta(minutes=lead)
        r["frame_end"] = r["frame_start"] + pd.Timedelta(minutes=30)
        cls = np.zeros(len(r), dtype="int8")
        for ci, thr in enumerate(mv.thresholds, start=1):
            m = mv.models[(lead, thr)]
            missing = [c for c in m.features if c not in features.columns]
            if missing:
                raise KeyError(f"features missing for {m.variant} lead{lead} thr{thr:g}: {missing[:5]}")
            pr = m.booster.predict(features[m.features].to_numpy("float32"))
            pc = m.calibrator.predict(pr)
            k = kstr(thr)
            r[f"p_raw_{k}"] = pr.astype("float32")
            r[f"p_{k}"] = pc.astype("float32")
            r[f"yhat_{k}"] = (pc >= m.cut).astype("int8")
            cls = np.where(r[f"yhat_{k}"].to_numpy() == 1, np.int8(ci), cls)   # highest class wins
        r["rain_class"] = cls.astype("int8")
        r["rain_class_name"] = pd.Series(cls).map(CLASS_NAMES).to_numpy()
        out.append(r)
    res = pd.concat(out, ignore_index=True)
    res.attrs["model_version"] = mv.name
    return res


# ================================================================================================ v1 (P4B)
V1_DIR = ROOT / "models" / "v1"
V1_LEADS = (0, 30, 60)
V1_THRESHOLDS = (0.5, 2.5, 7.6)
V1_PRIMARY = 2.5
ISSUE_OFFSET_MIN = 10                    # ADD-005: prediction_time = t0 + 10 min
NWP_MIN_AGE_RE = r"issue <= prediction_time - (\d+(?:\.\d+)?) h"


@dataclass
class V1Model:
    variant: str
    lead: int
    thr: float
    booster: object
    calibration: dict
    cut: float
    heavy_rule: str | None
    features: list[str]
    meta: dict = field(repr=False)


@dataclass
class V1Version:
    name: str
    run_id: str
    leads: list[int]
    thresholds: list[float]
    ship_B: bool
    serving_rule: str
    nwp_min_age_h: float | None              # parsed from serving_rule; None = only "all nwp_* present"
    models: dict = field(repr=False)         # (variant, lead, thr) -> V1Model


def _lgb_booster(path: Path):
    import lightgbm as lgb
    return lgb.Booster(model_file=str(path))


def apply_calibration(cal: dict, p) -> np.ndarray:
    """Same mapping as src.models.p4b.apply_calibrator (raw / platt / isotonic table)."""
    p = np.asarray(p, "float64")
    if cal["method"] == "raw":
        return p
    if cal["method"] == "platt":
        q = np.clip(p, 1e-6, 1 - 1e-6)
        return 1 / (1 + np.exp(-(cal["a"] * np.log(q / (1 - q)) + cal["b"])))
    return np.interp(p, np.asarray(cal["x"]), np.asarray(cal["y"]))


def load_v1(root: Path | None = None, booster_loader=None) -> V1Version:
    """Load every v1 model (set A always; set B when meta says ship_B). `booster_loader(path)` is for tests."""
    import re
    root = Path(root or V1_DIR)
    load_b = booster_loader or _lgb_booster
    models, run_ids, ship = {}, set(), set()
    for variant in ("A", "B"):
        for lead in V1_LEADS:
            for thr in V1_THRESHOLDS:
                d = root / variant / f"lead{lead}_thr{kstr(thr)}"
                if not (d / "meta.json").is_file():
                    if variant == "A":
                        raise FileNotFoundError(f"v1 set-A model missing: {d}")
                    continue
                meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
                if int(meta["lead_min"]) != lead or float(meta["threshold_mm_h"]) != thr or meta["variant"] != variant:
                    raise ValueError(f"meta.json does not match its directory: {d}")
                cal = json.loads((d / "calibration.json").read_text(encoding="utf-8"))
                models[(variant, lead, thr)] = V1Model(variant, lead, thr, load_b(d / "model.txt"), cal,
                                                       float(meta["cut"]["cut"]), meta.get("heavy_rule"),
                                                       list(meta["features"]), meta)
                run_ids.add(meta["run_id"])
                ship.add(bool(meta["ship_B"]))
    if len(run_ids) != 1 or len(ship) != 1:
        raise ValueError(f"v1 models come from mixed runs or disagree on ship_B: {run_ids}, {ship}")
    ship_B = ship.pop()
    have_B = all(("B", L, t) in models for L in V1_LEADS for t in V1_THRESHOLDS)
    if ship_B and not have_B:
        raise FileNotFoundError("meta.json says ship_B but some set-B models are missing")
    rule = models[("B", V1_LEADS[0], V1_PRIMARY)].meta["serving_rule"] if ship_B else "set A only"
    m = re.search(NWP_MIN_AGE_RE, rule)
    return V1Version("v1", run_ids.pop(), list(V1_LEADS), list(V1_THRESHOLDS), ship_B, rule,
                     float(m.group(1)) if m else None, models)


def _utc(s) -> pd.DatetimeIndex:
    d = pd.DatetimeIndex(s)
    return d.tz_localize("UTC") if d.tz is None else d.tz_convert("UTC")


def nwp_ok_mask(f: pd.DataFrame, nwp_cols: list[str], min_age_h: float | None) -> np.ndarray:
    """True where the set-B model may be used: every nwp_* input present and, when the group times are given,
    issue <= prediction_time - min_age_h and available_time <= prediction_time (meta.json serving_rule)."""
    if not nwp_cols or any(c not in f.columns for c in nwp_cols):
        return np.zeros(len(f), dtype=bool)
    ok = f[nwp_cols].notna().all(axis=1).to_numpy().copy()
    pt = _utc(f["prediction_time"]) if "prediction_time" in f.columns else \
        _utc(f["t0"]) + pd.Timedelta(minutes=ISSUE_OFFSET_MIN)
    for short in sorted({c.split("_")[1] for c in nwp_cols}):
        ic, ac = f"nwp_{short}__issue_time", f"nwp_{short}__available_time"
        if ic in f.columns and min_age_h is not None:
            iss = _utc(f[ic])
            ok &= np.asarray(iss.notna() & (iss <= pt - pd.Timedelta(hours=min_age_h)))
        if ac in f.columns:
            av = _utc(f[ac])
            ok &= np.asarray(av.notna() & (av <= pt))
    return ok


def _score(m: V1Model, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    missing = [c for c in m.features if c not in X.columns]
    if missing:
        raise KeyError(f"features missing for v1 {m.variant} lead{m.lead} thr{m.thr:g}: {missing[:5]}")
    pr = np.asarray(m.booster.predict(X[m.features].to_numpy("float32")), "float64")
    return pr, apply_calibration(m.calibration, pr)


def predict_v1(features: pd.DataFrame, mv: V1Version, nwp: pd.DataFrame | None = None,
               leads: list[int] | None = None) -> pd.DataFrame:
    """features: one row per (pixel_id, t0) from features.compute_features (set-A columns), or Gold-style rows
    with a `lead_min` column (then nwp_* may already be inline). nwp (optional): rows (pixel_id, t0, lead_min)
    with the nwp_* columns of features.compute_nwp_features (and optionally its group times); NWP features
    depend on the frame, so they are joined per lead. Returns one row per (pixel_id, t0, lead_min)."""
    out = []
    for lead in (leads or mv.leads):
        f = features[features["lead_min"] == lead] if "lead_min" in features.columns else features
        f = f.reset_index(drop=True)
        if nwp is not None:
            n = nwp[nwp["lead_min"] == lead]
            n = n[["pixel_id", "t0"] + [c for c in n.columns if c.startswith("nwp_") and c not in f.columns]]
            f = f.merge(n, on=["pixel_id", "t0"], how="left", validate="one_to_one")
        r = pd.DataFrame({"pixel_id": f["pixel_id"].to_numpy(), "t0": _utc(f["t0"]), "lead_min": np.int16(lead)})
        r["frame_start"] = r["t0"] + pd.Timedelta(minutes=lead)
        r["frame_end"] = r["frame_start"] + pd.Timedelta(minutes=30)
        use_B = np.zeros(len(f), dtype=bool)
        if mv.ship_B:
            nwp_cols = [c for c in mv.models[("B", lead, V1_PRIMARY)].features if c.startswith("nwp_")]
            use_B = nwp_ok_mask(f, nwp_cols, mv.nwp_min_age_h)
        sets = {"A": ~use_B, "B": use_B}
        p_raw = {t: np.full(len(f), np.nan) for t in mv.thresholds}
        p_cal = {t: np.full(len(f), np.nan) for t in mv.thresholds}
        for v, rows in sets.items():
            if rows.any():
                X = f.loc[rows]
                for t in mv.thresholds:
                    p_raw[t][rows], p_cal[t][rows] = _score(mv.models[(v, lead, t)], X)
        cls = np.zeros(len(f), dtype="int8")
        for ci, t in enumerate(mv.thresholds, start=1):
            yh = np.zeros(len(f), dtype=bool)
            for v, rows in sets.items():
                if not rows.any():
                    continue
                m = mv.models[(v, lead, t)]
                y = p_cal[t][rows] >= m.cut
                if m.heavy_rule == "nested":                       # 7.6 mm/h: also above the 2.5 cut of the same set
                    y &= p_cal[V1_PRIMARY][rows] >= mv.models[(v, lead, V1_PRIMARY)].cut
                yh[rows] = y
            k = kstr(t)
            r[f"p_raw_{k}"] = p_raw[t].astype("float32")
            r[f"p_{k}"] = p_cal[t].astype("float32")
            r[f"yhat_{k}"] = yh.astype("int8")
            cls = np.where(yh, np.int8(ci), cls)                     # highest class wins
        r["model_set"] = np.where(use_B, "B", "A")
        r["rain_class"] = cls.astype("int8")
        r["rain_class_name"] = pd.Series(cls).map(CLASS_NAMES).to_numpy()
        out.append(r)
    res = pd.concat(out, ignore_index=True)
    res.attrs["model_version"] = f"{mv.name} ({mv.run_id})"
    res.attrs["note"] = "Rain nowcasting against IMERG (satellite estimate, not gauges); not flood prediction."
    return res


def load(version: str = "v1", **kw):
    """Version switch: "v1" (default, P4B) or "v0" (PILOT, models/pilot_v0; same as load_version)."""
    if version == "v1":
        return load_v1(**kw)
    if version == "v0":
        return load_version(**kw)
    raise ValueError(f"unknown model version {version!r} (v1 or v0)")
