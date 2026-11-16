"""Regional crop of global 0.1 degree rain products to the whole-Hanoi domain.

Bronze-regional layout (ADD-002):
  data/bronze_regional/{source}/date=YYYY-MM-DD/{source}_{YYYYMMDD}.h5
  data/bronze_regional/{source}/manifest.jsonl   (one line per original file)

Native variable names, units, dtypes, axis order and raw sentinel values are
kept. Arrays of one day are stacked on the time axis (axis 0).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.lake import grid  # noqa: E402

MET_CONFIG = ROOT / "configs" / "met_dataset.yaml"
REGIONAL_ROOT = ROOT / "data" / "bronze_regional"
IMERG_ROOT = ROOT / "data" / "bronze" / "imerg" / "raw"
GSMAP_ROOT = ROOT / "data" / "bronze" / "rainfall" / "gsmap_now" / "raw"

IMERG_RE = re.compile(
    r"^3B-HHR(?:-[LE])?\.MS\.MRG\.3IMERG\.(?P<date>\d{8})-S(?P<start>\d{6})-E(?P<end>\d{6})"
    r"\.(?P<minutes>\d{4})\.(?P<version>V\d{2}[A-Z])\.(?:HDF5|RT-H5)$")
GSMAP_RE = re.compile(r"^gsmap_now_rain\.(?P<date>\d{8})\.(?P<time>\d{4})\.nc$")

SOURCES = ("imerg_final", "gsmap_now")
ALL_SOURCES = ("imerg_final", "imerg_late", "imerg_early", "gsmap_now")
ATTR_KEYS = ("units", "Units", "_FillValue", "CodeMissingValue", "missing_value",
             "long_name", "DimensionNames")


# ---------------------------------------------------------------- helpers
def load_domain() -> dict:
    cfg = yaml.safe_load(MET_CONFIG.read_text(encoding="utf-8"))
    g = cfg["domain"]["grid"]
    return {"i0": g["lat_idx_min"], "i1": g["lat_idx_max"],
            "j0": g["lon_idx_min"], "j1": g["lon_idx_max"],
            "provisional": cfg["domain"]["provisional"],
            "config": cfg}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def rel(p: Path) -> str:
    return Path(p).resolve().relative_to(ROOT).as_posix()


def _attr(v):
    if isinstance(v, (bytes, np.bytes_)):
        return v.decode("utf-8", "replace")
    if isinstance(v, np.ndarray) and v.size == 1:
        return v.reshape(-1)[0].item()
    if isinstance(v, np.generic):
        return v.item()
    return v


# ---------------------------------------------------------------- windows
@dataclass
class Window:
    source: str
    path: Path
    start: datetime
    end: datetime
    version: str


def imerg_window(path: Path) -> Window:
    m = IMERG_RE.fullmatch(path.name)
    if not m:
        raise ValueError(f"not an IMERG HHR file: {path.name}")
    d = datetime.strptime(m["date"], "%Y%m%d").replace(tzinfo=timezone.utc)
    start = d + timedelta(hours=int(m["start"][:2]), minutes=int(m["start"][2:4]),
                          seconds=int(m["start"][4:]))
    end = start + timedelta(minutes=30)
    return Window("imerg_final", path, start, end, m["version"])


def gsmap_window(path: Path) -> Window:
    m = GSMAP_RE.fullmatch(path.name)
    if not m:
        raise ValueError(f"not a GSMaP NOW rain file: {path.name}")
    start = datetime.strptime(m["date"] + m["time"], "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    return Window("gsmap_now", path, start, start + timedelta(hours=1), "GSMaP_NOW_V8")


def windows_for_date(source: str, date: str, files: list[Path] | None = None) -> list[Window]:
    d = datetime.strptime(date, "%Y-%m-%d")
    if files is not None:
        mk = imerg_window if source.startswith("imerg") else gsmap_window
        ws = [mk(Path(p)) for p in files]
    elif source == "imerg_final":
        folder = IMERG_ROOT / f"year={d:%Y}" / f"month={d:%m}" / f"day={d:%d}"
        files = sorted(folder.glob("3B-HHR*.HDF5")) if folder.is_dir() else []
        ws = [imerg_window(p) for p in files]
    elif source == "gsmap_now":
        folder = GSMAP_ROOT / f"event_date={date}"
        files = sorted(folder.glob("gsmap_now_rain.*.nc")) if folder.is_dir() else []
        ws = [gsmap_window(p) for p in files]
    else:
        raise ValueError(source)
    ws = [w for w in ws if w.start.strftime("%Y-%m-%d") == date]
    return sorted(ws, key=lambda w: w.start)


def dates_on_disk(source: str) -> list[str]:
    out = set()
    if source == "imerg_final":
        for p in IMERG_ROOT.glob("year=*/month=*/day=*"):
            if any(p.glob("3B-HHR*.HDF5")):
                y, m, d = (x.split("=")[1] for x in p.parts[-3:])
                out.add(f"{y}-{m}-{d}")
    else:
        for p in GSMAP_ROOT.glob("event_date=*"):
            if any(p.glob("gsmap_now_rain.*.nc")):
                out.add(p.name.split("=")[1])
    return sorted(out)


# ---------------------------------------------------------------- readers
def _check_axis(lat: np.ndarray, lon: np.ndarray, dom: dict) -> None:
    """Confirm the file's own coordinates match the shared grid convention."""
    if lat.shape != (grid.N_LAT,) or lon.shape != (grid.N_LON,):
        raise ValueError(f"unexpected grid shape lat={lat.shape} lon={lon.shape}")
    i = np.arange(dom["i0"], dom["i1"] + 1)
    j = np.arange(dom["j0"], dom["j1"] + 1)
    if not np.allclose(lat[i], grid.lat_centre(i), atol=1e-4):
        raise ValueError("latitude axis does not match grid convention")
    if not np.allclose(lon[j], grid.lon_centre(j), atol=1e-4):
        raise ValueError("longitude axis does not match grid convention")


def imerg_variables(f: h5py.File) -> list[str]:
    names = []

    def visit(n, o):
        if isinstance(o, h5py.Dataset) and o.ndim == 3 and o.shape[1:] == (grid.N_LON, grid.N_LAT):
            names.append(n)
    f.visititems(visit)
    return sorted(names)


def read_crop(w: Window, dom: dict) -> dict:
    """Return {'vars': {name: (array[1,...], attrs)}, 'meta': {...}} for one original file."""
    i0, i1, j0, j1 = dom["i0"], dom["i1"], dom["j0"], dom["j1"]
    out = {}
    with h5py.File(w.path, "r") as f:
        if w.source == "imerg_final":
            g = f["Grid"]
            _check_axis(g["lat"][:], g["lon"][:], dom)
            for name in imerg_variables(f):
                ds = f[name]
                dn = _attr(ds.attrs.get("DimensionNames", b""))
                if dn != "time,lon,lat":
                    raise ValueError(f"{name}: unexpected DimensionNames {dn!r}")
                out[name] = (ds[:, j0:j1 + 1, i0:i1 + 1], {k: ds.attrs[k] for k in ATTR_KEYS if k in ds.attrs})
            tb = g["time_bnds"][:]
            meta = {"time_bnds": tb, "header": _attr(f.attrs.get("FileHeader", b"")),
                    "lat": g["lat"][i0:i1 + 1], "lon": g["lon"][j0:j1 + 1]}
            epoch = datetime(1980, 1, 6, tzinfo=timezone.utc)
            if epoch + timedelta(seconds=int(tb[0, 0])) != w.start:
                raise ValueError(f"time_bnds start mismatch for {w.path.name}")
        else:
            _check_axis(f["Latitude"][:], f["Longitude"][:], dom)
            for name in ("hourlyPrecipRate", "hourlyPrecipRateGC"):
                ds = f[name]
                if ds.shape != (1, grid.N_LAT, grid.N_LON):
                    raise ValueError(f"{name}: unexpected shape {ds.shape}")
                out[name] = (ds[:, i0:i1 + 1, j0:j1 + 1], {k: ds.attrs[k] for k in ATTR_KEYS if k in ds.attrs})
            units = _attr(f["Time"].attrs.get("units", b""))
            m = re.match(r"1hour since (\d{4}-\d{2}-\d{2}) (\d{1,2}):(\d{1,2})", units)
            if not m:
                raise ValueError(f"unexpected Time units {units!r}")
            t0 = datetime.strptime(f"{m[1]} {int(m[2]):02d}:{int(m[3]):02d}", "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
            t0 += timedelta(hours=int(f["Time"][0]))
            if t0 != w.start:
                raise ValueError(f"Time units {units!r} disagree with filename {w.path.name}")
            meta = {"time_units": units, "title": _attr(f.attrs.get("title", b"")),
                    "lat": f["Latitude"][i0:i1 + 1], "lon": f["Longitude"][j0:j1 + 1]}
    return {"vars": out, "meta": meta}


# ---------------------------------------------------------------- writer
def crop_path(source: str, date: str) -> Path:
    return REGIONAL_ROOT / source / f"date={date}" / f"{source}_{date.replace('-', '')}.h5"


def manifest_path(source: str) -> Path:
    return REGIONAL_ROOT / source / "manifest.jsonl"


def remote_listing(source: str, date: str) -> dict:
    """Remote size/MDTM recorded by an archive collector, keyed by filename."""
    p = REGIONAL_ROOT / source / "remote_listing" / f"date={date}.json"
    if not p.is_file():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return {e["name"]: e for e in data.get("files", [])}


def build_day(source: str, date: str, dom: dict | None = None,
              original_hashes: dict | None = None, files: list[Path] | None = None,
              origin_paths: dict | None = None) -> dict:
    """Build one day's crop file and manifest lines. Idempotent (rewrites the day).

    ``original_hashes`` maps filename -> (size, sha256) when the caller already
    hashed the original (e.g. a crop-only download that is deleted afterwards).
    ``files`` overrides discovery on disk; ``origin_paths`` maps filename -> the
    string recorded as ``original_path`` (e.g. the remote path for crop-only).
    """
    dom = dom or load_domain()
    ws = windows_for_date(source, date, files)
    if not ws:
        raise FileNotFoundError(f"no {source} files on disk for {date}")
    crops = [read_crop(w, dom) for w in ws]
    out, crop_sha, var_names = write_day_file(source, date, dom, ws, crops)

    listing = remote_listing(source, date)
    now = iso(datetime.now(timezone.utc))
    lines = []
    for t, w in enumerate(ws):
        if original_hashes and w.path.name in original_hashes:
            size, sha = original_hashes[w.path.name]
        else:
            size, sha = w.path.stat().st_size, sha256_file(w.path)
        r = listing.get(w.path.name, {})
        lines.append({
            "source": source, "date": date, "time_index": t,
            "original_name": w.path.name,
            "original_path": (origin_paths or {}).get(w.path.name) or rel(w.path),
            "original_size": size, "original_sha256": sha,
            "window_start_utc": iso(w.start), "window_end_utc": iso(w.end),
            "product_version": w.version,
            "remote_size": r.get("size"), "remote_mtime_utc": r.get("mdtm_utc"),
            "crop_file": rel(out), "crop_sha256": crop_sha, "built_utc": now,
        })
    write_manifest_day(source, date, lines)
    return {"source": source, "date": date, "windows": len(ws), "variables": var_names,
            "crop_file": rel(out), "crop_bytes": out.stat().st_size, "crop_sha256": crop_sha}


def write_day_file(source: str, date: str, dom: dict, ws: list[Window], crops: list[dict]) -> tuple[Path, str, list[str]]:
    """Write the regional day file from per-window crops (format of ``read_crop``),
    read every array back, compare exactly, then promote atomically.
    Returns (path, sha256, variable names)."""
    var_names = list(crops[0]["vars"].keys())
    for c, w in zip(crops, ws):
        if list(c["vars"].keys()) != var_names:
            raise ValueError(f"variable set differs in {w.path.name}")

    out = crop_path(source, date)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".h5.part")
    if tmp.exists():
        tmp.unlink()
    n = len(ws)
    starts = np.array([int(w.start.timestamp()) for w in ws], dtype=np.int64)
    ends = np.array([int(w.end.timestamp()) for w in ws], dtype=np.int64)
    with h5py.File(tmp, "w", track_order=True) as h:
        h.attrs["source"] = source
        h.attrs["date_utc"] = date
        h.attrs["product_version"] = ",".join(sorted({w.version for w in ws}))
        h.attrs["domain_lat_idx_min"] = dom["i0"]
        h.attrs["domain_lat_idx_max"] = dom["i1"]
        h.attrs["domain_lon_idx_min"] = dom["j0"]
        h.attrs["domain_lon_idx_max"] = dom["j1"]
        h.attrs["domain_provisional"] = bool(dom["provisional"])
        h.attrs["grid_convention"] = "src/lake/grid.py: lat_idx from -89.95, lon_idx from -179.95, step 0.1"
        h.attrs["axis_order"] = "time,lon,lat" if source.startswith("imerg") else "time,lat,lon"
        h.attrs["n_windows"] = n
        h.attrs["note"] = "Regional crop (ADD-002). Native variables, units and raw sentinel values unchanged."
        kw = dict(compression="gzip", compression_opts=4, shuffle=True, track_times=False)
        for name in var_names:
            arr = np.concatenate([c["vars"][name][0] for c in crops], axis=0)
            ds = h.create_dataset(name, data=arr, chunks=(1,) + arr.shape[1:], **kw)
            for k, v in crops[0]["vars"][name][1].items():
                ds.attrs[k] = v
        h.create_dataset("lat", data=crops[0]["meta"]["lat"], track_times=False)
        h.create_dataset("lon", data=crops[0]["meta"]["lon"], track_times=False)
        h.create_dataset("lat_idx", data=np.arange(dom["i0"], dom["i1"] + 1, dtype=np.int32), track_times=False)
        h.create_dataset("lon_idx", data=np.arange(dom["j0"], dom["j1"] + 1, dtype=np.int32), track_times=False)
        h.create_dataset("window_start_unix", data=starts, track_times=False)
        h.create_dataset("window_end_unix", data=ends, track_times=False)
        h.create_dataset("source_file", data=np.array([w.path.name for w in ws], dtype="S"), track_times=False)
        if source.startswith("imerg"):
            h.create_dataset("time_bnds_gps_like", data=np.concatenate([c["meta"]["time_bnds"] for c in crops]),
                             track_times=False)
    # read back and compare every array exactly before promoting
    with h5py.File(tmp, "r") as h:
        for name in var_names:
            arr = h[name][:]
            for t, c in enumerate(crops):
                if not np.array_equal(arr[t], c["vars"][name][0][0], equal_nan=True):
                    raise RuntimeError(f"read-back mismatch {name} t={t}")
    os.replace(tmp, out)
    return out, sha256_file(out), var_names


def read_manifest(source: str) -> list[dict]:
    p = manifest_path(source)
    if not p.is_file():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_manifest_day(source: str, date: str, lines: list[dict]) -> None:
    keep = [r for r in read_manifest(source) if r["date"] != date]
    allr = sorted(keep + lines, key=lambda r: (r["date"], r["time_index"]))
    p = manifest_path(source)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".jsonl.part")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in allr), encoding="utf-8")
    os.replace(tmp, p)


# ---------------------------------------------------------------- verification
def verify_samples(source: str, n: int = 200, seed: int = 20261115) -> dict:
    """Independent check: random (file, pixel, variable) values equal originals bit-for-bit.

    Locates the pixel in the original via the original's own lat/lon arrays
    (not via the domain index range used by the cropper).
    """
    rng = np.random.default_rng(seed)
    man = [r for r in read_manifest(source)
           if ":" not in r["original_path"] and (ROOT / r["original_path"]).is_file()]
    if not man:
        return {"source": source, "checked": 0, "mismatches": 0, "note": "no originals on disk"}
    mism, checked, details = 0, 0, []
    for _ in range(n):
        r = man[int(rng.integers(len(man)))]
        with h5py.File(ROOT / r["crop_file"], "r") as h:
            names = []
            h.visititems(lambda nm, o: names.append(nm)
                         if isinstance(o, h5py.Dataset) and o.ndim == 3 else None)
            names.sort()
            name = names[int(rng.integers(len(names)))]
            clat, clon = h["lat"][:], h["lon"][:]
            a, b = int(rng.integers(len(clat))), int(rng.integers(len(clon)))
            t = r["time_index"]
            v_crop = h[name][t, b, a] if source == "imerg_final" else h[name][t, a, b]
        with h5py.File(ROOT / r["original_path"], "r") as f:
            if source == "imerg_final":
                olat, olon = f["Grid/lat"][:], f["Grid/lon"][:]
                ia = int(np.where(olat == clat[a])[0][0]); jb = int(np.where(olon == clon[b])[0][0])
                v_orig = f[name][0, jb, ia]
            else:
                olat, olon = f["Latitude"][:], f["Longitude"][:]
                ia = int(np.where(olat == clat[a])[0][0]); jb = int(np.where(olon == clon[b])[0][0])
                v_orig = f[name][0, ia, jb]
        checked += 1
        same = np.array_equal(np.asarray(v_crop), np.asarray(v_orig), equal_nan=True) and \
            np.asarray(v_crop).dtype == np.asarray(v_orig).dtype
        if not same:
            mism += 1
            details.append({"file": r["original_name"], "var": name, "lat": float(clat[a]),
                            "lon": float(clon[b]), "crop": float(v_crop), "orig": float(v_orig)})
    return {"source": source, "checked": checked, "mismatches": mism, "details": details[:10]}
