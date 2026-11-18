"""P4A.8 — replay packages (PILOT v0). One event per invocation (resumable: an existing package is skipped).

  python src/models/replay.py --list
  python src/models/replay.py --event test2025_wet1

Each package reports/p4a/replay/<event>/ holds replay.parquet (per pixel, t0, frame: observed IMERG Final rate,
raw + calibrated probabilities, classes; for the Yagi event also raw probabilities of the pre-refit models),
summary.json (CSI per frame and threshold, base rates) and panel PNGs (observed vs predicted, one per frame).
Predictions come from src/models/predict.py applied to the Gold feature rows (single feature code path).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.gold import p4a_common as P  # noqa: E402
from src.models import data as D  # noqa: E402
from src.models.predict import CLASS_NAMES, kstr, load_version, predict  # noqa: E402

OUT = ROOT / "reports" / "p4a" / "replay"
TZ = "Asia/Ho_Chi_Minh"
RSS_LIMIT_GB = 0.7
NOTE_YAGI = ("Yagi 2024-09-06..08: the A_no2023 models were tuned on 2024-05-01..2024-08-31 (these days were NOT in "
             "the tune period) but 2024-09/10 was their VALIDATION period (early stopping, isotonic calibrator, "
             "probability cut) and is inside the REFIT data (tune + validation). The refit-model replay is therefore "
             "in-sample, not an out-of-sample test. The leave-2024-out fold was not trained (deferred to cloud "
             "training). Column p_raw_prerefit_* = models trained on the tune period only (Yagi used only for early "
             "stopping). The 0.5 mm/h model is set A (tune 2023-05..2024-08), whose refit also includes these days.")


def events() -> dict:
    wd = pd.read_csv(ROOT / "reports" / "p4a" / "robustness" / "wettest_test_days.csv")
    ev = {f"test2025_wet{i + 1}_{d}": {"days": [d], "split": "test", "note": f"test 2025, wettest day #{i + 1} "
                                       f"(city mean IMERG Final {wd['city_mean_mm_day'].iloc[i]:.1f} mm/day); out-of-sample"}
          for i, d in enumerate(wd["day"].head(3))}
    ev["yagi_2024-09-06_08"] = {"days": ["2024-09-06", "2024-09-07", "2024-09-08"], "split": "validation",
                                "note": NOTE_YAGI}
    return ev


def load_rows(days: list[str]) -> pd.DataFrame:
    lo = pd.Timestamp(days[0]).tz_localize(TZ).tz_convert("UTC")
    hi = (pd.Timestamp(days[-1]) + pd.Timedelta("1D")).tz_localize(TZ).tz_convert("UTC")
    parts = []
    for m in pd.period_range(lo.tz_convert(None).to_period("M"), hi.tz_convert(None).to_period("M"), freq="M"):
        f = P.GOLD / "table_A" / f"season={m.year}" / f"month={m.month:02d}" / "part.parquet"
        if f.is_file():
            parts.append(pd.read_parquet(f, filters=[("t0", ">=", lo), ("t0", "<", hi)]))
    return pd.concat(parts, ignore_index=True)


def obs_class(rate: np.ndarray) -> np.ndarray:
    c = np.zeros(len(rate), dtype="int8")
    for ci, t in enumerate((0.5, 2.5, 7.6), start=1):
        c = np.where(rate >= t, ci, c)
    return np.where(np.isnan(rate), -1, c).astype("int8")


def panels(df: pd.DataFrame, name: str, out: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    paths = []
    f0 = df[df["lead_min"] == 0].groupby("t0")["obs_rate_mm_h"].mean()
    # 8 consecutive analysis times centred on the wettest one (by city-mean observed F0 rate)
    t_all = list(f0.index)
    c = t_all.index(f0.idxmax())
    sel = t_all[max(0, min(c - 3, len(t_all) - 8)):][:8]
    ccmap = ListedColormap(["#f0f0f0", "#a6d96a", "#fdae61", "#d7191c"])
    cnorm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4)
    for lead in sorted(df["lead_min"].unique()):
        d = df[df["lead_min"] == lead]
        fig, ax = plt.subplots(4, len(sel), figsize=(2.1 * len(sel), 8.4), constrained_layout=True)
        for j, t in enumerate(sel):
            s = d[d["t0"] == t]
            loc = (t + pd.Timedelta(minutes=int(lead))).tz_convert(TZ)
            for i, (col, kw, ttl) in enumerate([
                    ("obs_rate_mm_h", dict(cmap="Blues", vmin=0, vmax=20), "IMERG obs mm/h"),
                    ("p_2p5", dict(cmap="viridis", vmin=0, vmax=1), "P(>=2.5) cal"),
                    ("p_raw_2p5", dict(cmap="viridis", vmin=0, vmax=1), "P(>=2.5) raw"),
                    ("rain_class", dict(cmap=ccmap, norm=cnorm), "class pred")]):
                a = ax[i, j]
                sc = a.scatter(s["lon"], s["lat"], c=s[col], marker="s", s=60, **kw)
                if j == 0:
                    a.set_ylabel(ttl, fontsize=8)
                if i == 0:
                    a.set_title(f"frame {loc:%m-%d %H:%M}", fontsize=8)
                a.set_xticks([]); a.set_yticks([]); a.set_aspect("equal")
                if j == len(sel) - 1:
                    fig.colorbar(sc, ax=a, shrink=0.8)
        fig.suptitle(f"PILOT v0 replay {name} — F{lead} (frame start, local time); observed = IMERG Final (satellite)",
                     fontsize=9)
        p = out / f"panels_F{lead}.png"
        fig.savefig(p, dpi=90); plt.close(fig)
        paths.append(p)
    return paths


def csi(y, f) -> float:
    y = np.asarray(y).astype(bool); f = np.asarray(f).astype(bool)
    den = (y | f).sum()
    return float((y & f).sum() / den) if den else float("nan")


def run(name: str, ev: dict) -> int:
    out = OUT / name
    if (out / "summary.json").is_file():
        print(f"skip {name} (exists)"); return 0
    out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(ev["days"])
    feats = rows[rows["lead_min"] == 0]
    mv = load_version()
    pr = predict(feats, mv)
    if name.startswith("yagi"):
        mp = load_version(model_file="model_prerefit.txt")
        pp = predict(feats, mp)
        for t in mp.thresholds:
            pr[f"p_raw_prerefit_{kstr(t)}"] = pp[f"p_raw_{kstr(t)}"].to_numpy()
        del pp
    lab = rows[["pixel_id", "t0", "lead_min", "label_rate_mm_h", "lat", "lon"]].rename(
        columns={"label_rate_mm_h": "obs_rate_mm_h"})
    df = pr.merge(lab, on=["pixel_id", "t0", "lead_min"], how="left", validate="one_to_one")
    df["obs_class"] = obs_class(df["obs_rate_mm_h"].to_numpy("float64"))
    df["frame_start_local"] = df["frame_start"].dt.tz_convert(TZ)
    df["model_version"] = mv.name
    P.write_parquet(df, out / "replay.parquet")
    summ = {"event": name, "days_local": ev["days"], "status": "PILOT v0 — not a product model; observed = IMERG "
            "Final (satellite estimate), not gauges; no flood prediction", "note": ev["note"],
            "model_version": mv.name, "threshold_variant": mv.manifest["threshold_variant"], "rows": len(df),
            "metrics": []}
    ok = df["obs_rate_mm_h"].notna()
    for lead in mv.leads:
        for t in mv.thresholds:
            k = kstr(t); s = ok & (df["lead_min"] == lead)
            y = df.loc[s, "obs_rate_mm_h"] >= t
            r = {"lead_min": lead, "threshold": t, "base_rate": float(y.mean()), "csi": csi(y, df.loc[s, f"yhat_{k}"]),
                 "mean_p_cal": float(df.loc[s, f"p_{k}"].mean()), "mean_p_raw": float(df.loc[s, f"p_raw_{k}"].mean())}
            if f"p_raw_prerefit_{k}" in df:
                r["mean_p_raw_prerefit"] = float(df.loc[s, f"p_raw_prerefit_{k}"].mean())
            summ["metrics"].append(r)
    figs = panels(df, name, out)
    summ["files"] = {P.rel(p): P.sha256_file(p) for p in [out / "replay.parquet", *figs]}
    P.write_json(out / "summary.json", summ)
    P.check_rss(RSS_LIMIT_GB)
    m25 = [m for m in summ["metrics"] if m["threshold"] == 2.5]
    print(f"{name}: rows={len(df)} CSI2.5 F0/F30/F60=" + "/".join(f"{m['csi']:.3f}" for m in m25)
          + f" base2.5={m25[0]['base_rate']:.3f} rss={P.rss_gb():.2f}GB")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    P.require_guard()
    ev = events()
    if a.list or not a.event:
        for k, v in ev.items():
            print(k, v["days"])
        return 0
    return run(a.event, ev[a.event])


if __name__ == "__main__":
    raise SystemExit(main())
