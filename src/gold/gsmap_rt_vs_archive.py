"""P4A.1 — realtime-vs-archive identity for GSMaP NOW (read-only on R2, sequential FTP).

Picks up to N windows of the given dates that exist under lake/v1/realtime/gsmap_now/ on R2,
downloads the same windows from the JAXA archive (/now/netcdf/YYYY/MM/DD), crops them with
the shared crop code and compares arrays. Scratch files live in data/_scratch/p4a_rt/ and are
removed. Writes reports/p4a/availability/rt_vs_archive.json.
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.gold import p4a_common as P  # noqa: E402
from src.ingestion import gsmap_now_archive_collector as ac  # noqa: E402
from src.ingestion import gsmap_now_collector as gc  # noqa: E402
from src.lake import regional_crop as rc  # noqa: E402
from src.realtime import common as C  # noqa: E402

SCRATCH = ROOT / "data" / "_scratch" / "p4a_rt"
VARS = ("hourlyPrecipRate", "hourlyPrecipRateGC")


def h5_arrays(b: bytes) -> dict:
    out = {}
    with h5py.File(io.BytesIO(b), "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and name.split("/")[-1] in VARS:
                out[name.split("/")[-1]] = obj[...]
        f.visititems(visit)
    return out


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    a = np.asarray(a, dtype="float64").squeeze(); b = np.asarray(b, dtype="float64").squeeze()
    if a.shape != b.shape:
        return {"shape_rt": list(a.shape), "shape_archive": list(b.shape), "identical": False}
    both = np.isfinite(a) & np.isfinite(b)
    d = np.abs(a - b)[both]
    nan_mismatch = int((np.isfinite(a) != np.isfinite(b)).sum())
    return {"identical": bool(np.array_equal(a, b, equal_nan=True)),
            "max_abs_diff": float(d.max()) if d.size else 0.0,
            "share_pixels_changed": float(((d > 0).sum() + nan_mismatch) / a.size),
            "nan_mismatch": nan_mismatch,
            "mask_0p5_flips": int(((a >= 0.5) != (b >= 0.5)).sum()),
            "mask_0p5_rt": int((a >= 0.5).sum()), "mask_0p5_archive": int((b >= 0.5).sum())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="+", default=["2026-11-15", "2026-11-16"])
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args(argv)
    lg = P.get_logger("p4a1_rt_vs_archive")
    rcfg = C.load_config()
    store = C.open_store("r2", rcfg)
    keys = []
    for d in a.dates:
        ks = sorted(k for k in store.list(f"lake/v1/realtime/gsmap_now/date={d}/") if k.endswith(".h5"))
        keys += ks
    lg.info("realtime keys found: %d", len(keys))
    if not keys:
        print("no realtime windows found"); return 1
    idx = np.unique(np.linspace(0, len(keys) - 1, min(a.n, len(keys))).round().astype(int))
    chosen = [keys[i] for i in idx]
    config = gc.load_config()
    dom = rc.load_domain()
    results, listings = [], {}
    SCRATCH.mkdir(parents=True, exist_ok=True)
    try:
        for key in chosen:
            stamp = key.rsplit("_", 1)[-1].removesuffix(".h5")        # YYYYMMDDTHHMM
            d = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}"; hhmm = stamp[9:13]
            rt = h5_arrays(store.get(key))
            if d not in listings:
                listings[d] = ac.list_date(config, d)
            ent = [e for e in listings[d]["files"] if f".{stamp[:8]}.{hhmm}." in e["name"]]
            if not ent:
                results.append({"realtime_key": key, "archive": "not_listed"}); continue
            e = ent[0]
            dest = SCRATCH / e["name"]
            ac.retrieve(config, d, e["name"], e["size"], dest)
            w = rc.gsmap_window(dest)
            crop = rc.read_crop(w, dom)
            r = {"realtime_key": key, "archive_name": e["name"], "archive_mdtm": str(e.get("mdtm")),
                 "window_start": str(w.start)}
            for v in VARS:
                r[v] = compare(rt[v], crop["vars"][v][0]) if v in rt else {"missing_in_realtime": True}
            results.append(r)
            dest.unlink(missing_ok=True)
            lg.info("%s %s", key, json.dumps({v: r[v].get("identical") for v in VARS}))
    finally:
        shutil.rmtree(SCRATCH, ignore_errors=True)
    ok = [r for r in results if "archive_name" in r]
    ident = all(r[v].get("identical") for r in ok for v in VARS) and len(ok) > 0
    out = {"dates": a.dates, "realtime_keys_total": len(keys), "compared": len(ok),
           "all_identical": ident, "windows": results, "r2_ops": store.ops}
    p = ROOT / "reports" / "p4a" / "availability" / "rt_vs_archive.json"
    P.write_json(p, out)
    print(f"compared={len(ok)} all_identical={ident} -> {P.rel(p)}")
    for r in ok:
        print(r["window_start"], {v: (r[v].get("identical"), r[v].get("max_abs_diff")) for v in VARS})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
