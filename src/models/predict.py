"""P4A.8 — prediction module for the PILOT rainfall nowcast v0 (not a product model; no flood prediction).

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
