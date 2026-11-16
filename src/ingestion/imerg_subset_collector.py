"""IMERG V07 regional subsetting via OPeNDAP (Final / Late / Early), with fallback.

  .venv/Scripts/python.exe -m src.ingestion.imerg_subset_collector --product late \
      --start 2025-10-06T00:00:00Z --end 2025-10-06T00:59:59Z [--download] [--fallback] [--compare-local]

* Own CMR query by short name (GPM_3IMERGHH / HHL / HHE, version 07).
* OPeNDAP URL taken from the granule metadata links; request the domain index
  range (configs/met_dataset.yaml) as netCDF-4 (DAP4 constraint expression).
* Token from ``imerg_collector.get_earthdata_token`` (never printed).
* --fallback: full HDF5 download to data/_scratch -> crop -> remove scratch.
* --compare-local: compare the subset with the crop of the global file on disk.
* --download: keep subsets under data/bronze_regional/imerg_{product}/subsets/date=.../
Prints one JSON line per granule with seconds and bytes.

Backfill, one UTC day (ADD-004):
  ... --product final --crop-day 2025-08-01 [--path opendap|fallback] [--verify-full]
* --path opendap (default): one DAP4 subset per granule (all 10 Grid variables +
  lat/lon/time_bnds) -> the same regional day file as the full path. Manifest:
  CMR granule UR / concept id / revision / production time / size, checksum if
  CMR has one, OPeNDAP URL (token only in the header), subset bytes + sha256.
* Verification: on the 1st of each month (or --verify-full) every granule is
  also fetched by full download; all arrays must match exactly, and the original
  size + sha256 are recorded. Result appended to
  data/bronze_regional/imerg_{product}/verification.jsonl; mismatch -> day fails.
* --path fallback: full download -> crop -> remove scratch (P1 path).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion.imerg_collector import get_earthdata_token  # noqa: E402
from src.lake import regional_crop as rc  # noqa: E402

CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"
SHORT = {"final": "GPM_3IMERGHH", "late": "GPM_3IMERGHHL", "early": "GPM_3IMERGHHE"}
VARS = ("precipitation", "precipitationQualityIndex", "probabilityLiquidPrecipitation", "randomError")
SCRATCH = ROOT / "data" / "_scratch" / "imerg_subset"
RETRY_WAIT_S = 60
HDF5_MAGIC = bytes([0x89, 0x48, 0x44, 0x46, 0x0D, 0x0A, 0x1A, 0x0A])  # HDF5 signature


def cmr_granules(product: str, start: str, end: str) -> list[dict]:
    r = requests.get(CMR, params={"short_name": SHORT[product], "version": "07",
                                  "temporal": f"{start},{end}", "page_size": 200}, timeout=60)
    r.raise_for_status()
    out = []
    for e in r.json().get("feed", {}).get("entry", []):
        links = e.get("links", [])
        opendap = next((l["href"] for l in links if "opendap" in l.get("href", "").lower()), None)
        data = next((l["href"] for l in links
                     if l.get("rel", "").endswith("/data#") and l.get("href", "").lower().endswith((".hdf5", ".rt-h5"))), None)
        out.append({"title": e["title"], "time_start": e["time_start"], "time_end": e["time_end"],
                    "opendap": opendap, "data_url": data, "granule_size_mb": e.get("granule_size")})
    return sorted(out, key=lambda g: g["time_start"])


def dap4_url(opendap: str, dom: dict) -> str:
    base = opendap[:-5] if opendap.endswith(".html") else opendap
    j = f"[{dom['j0']}:1:{dom['j1']}]"
    i = f"[{dom['i0']}:1:{dom['i1']}]"
    ce = ";".join([f"/Grid/{v}[0:1:0]{j}{i}" for v in VARS] + [f"/Grid/lat{i}", f"/Grid/lon{j}", "/Grid/time"])
    return f"{base}.dap.nc4?dap4.ce={ce}"


def _get(url: str, token: str, timeout: int = 300) -> bytes:
    with requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=timeout,
                      allow_redirects=True, stream=True) as r:
        r.raise_for_status()
        return r.content


def retry_once(fn, what):
    try:
        return fn(), None
    except Exception as e1:
        time.sleep(RETRY_WAIT_S)
        try:
            return fn(), f"retried after: {e1!r}"[:300]
        except Exception as e2:
            raise RuntimeError(f"{what} failed twice: {e2!r}"[:600]) from e2


def read_subset(blob: bytes) -> dict:
    with h5py.File(io.BytesIO(blob), "r") as f:
        found = {}

        def visit(n, o):
            base = n.split("/")[-1]
            if isinstance(o, h5py.Dataset) and base in VARS + ("lat", "lon"):
                found[base] = o[...]
        f.visititems(visit)
    return found


def local_crop(title: str, dom: dict) -> dict | None:
    """Crop of the matching global Final file on disk, axis order (time, lon, lat)."""
    name = title.split(":")[-1]  # CMR titles look like "GPM_3IMERGHH.07:<file name>"
    try:
        w = rc.imerg_window(Path(name))
    except ValueError:
        return None
    d = w.start
    p = rc.IMERG_ROOT / f"year={d:%Y}" / f"month={d:%m}" / f"day={d:%d}" / name
    if not p.is_file():
        return None
    w.path = p
    c = rc.read_crop(w, dom)
    return {k.split("/")[-1]: v[0] for k, v in c["vars"].items()} | {"lat": c["meta"]["lat"], "lon": c["meta"]["lon"]}


def compare(sub: dict, ref: dict) -> dict:
    res = {}
    for v in VARS + ("lat", "lon"):
        if v not in sub or v not in ref:
            res[v] = "missing"
            continue
        a, b = np.asarray(sub[v]), np.asarray(ref[v])
        res[v] = bool(a.shape == b.shape and np.array_equal(a, b, equal_nan=True))
    res["all_equal"] = all(x is True for x in res.values())
    return res


def fallback(g: dict, token: str, dom: dict) -> dict:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    name = Path(g["data_url"]).name
    dest = SCRATCH / name
    t0 = time.time()
    try:
        blob, note = retry_once(lambda: _get(g["data_url"], token), f"full download {name}")
        dest.write_bytes(blob)
        w = rc.imerg_window(dest) if g["data_url"] else None
        c = rc.read_crop(w, dom)
        arrs = {k.split("/")[-1]: v[0] for k, v in c["vars"].items()} | {"lat": c["meta"]["lat"], "lon": c["meta"]["lon"]}
        return {"path": "fallback_full_download", "seconds": round(time.time() - t0, 2), "bytes": len(blob),
                "sha256_original": hashlib.sha256(blob).hexdigest(), "retry_note": note, "arrays": arrs}
    finally:
        dest.unlink(missing_ok=True)


def crop_day_full(product: str, date: str, dom: dict) -> dict:
    """Backfill path: download each granule of a UTC day to scratch (sequential,
    one retry), validate the HDF5 signature, build the regional day file with
    rc.build_day, then remove the scratch files."""
    token = get_earthdata_token()
    gs = [g for g in cmr_granules(product, f"{date}T00:00:00Z", f"{date}T23:59:59Z")
          if g["time_start"].startswith(date) and g["data_url"]]
    folder = SCRATCH / product / f"date={date}"
    folder.mkdir(parents=True, exist_ok=True)
    files, hashes, origins, t0, nbytes = [], {}, {}, time.time(), 0
    try:
        for g in gs:
            name = Path(g["data_url"]).name
            blob, _ = retry_once(lambda: _get(g["data_url"], token), f"download {name}")
            if not blob.startswith(HDF5_MAGIC):
                raise RuntimeError(f"{name}: not an HDF5 file")
            dest = folder / name
            dest.write_bytes(blob)
            files.append(dest)
            hashes[name] = (len(blob), hashlib.sha256(blob).hexdigest())
            origins[name] = g["data_url"]
            nbytes += len(blob)
        res = rc.build_day(f"imerg_{product}", date, dom, original_hashes=hashes,
                           files=files, origin_paths=origins) if files else None
    finally:
        for p in folder.glob("3B-HHR*"):
            p.unlink(missing_ok=True)
        try:
            folder.rmdir()
        except OSError:
            pass
    return {"product": product, "date": date, "granules_listed": len(gs), "windows": (res or {}).get("windows", 0),
            "download_bytes": nbytes, "seconds": round(time.time() - t0, 1), "crop": res}


CMR_UMM = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
DAY_VARS = ("precipitation", "precipitationQualityIndex", "probabilityLiquidPrecipitation", "randomError",
            "Intermediate/IRinfluence", "Intermediate/IRprecipitation", "Intermediate/MWobservationTime",
            "Intermediate/MWprecipSource", "Intermediate/MWprecipitation", "Intermediate/precipitationUncal")
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)


def cmr_granules_umm(product: str, date: str) -> list[dict]:
    """Granules starting on ``date`` (UTC) with the provenance fields of ADD-004."""
    r = requests.get(CMR_UMM, params={"short_name": SHORT[product], "version": "07",
                                      "temporal": f"{date}T00:00:00Z,{date}T23:59:59Z", "page_size": 200},
                     timeout=60)
    r.raise_for_status()
    out = []
    for it in r.json().get("items", []):
        u, meta = it["umm"], it["meta"]
        start = u.get("TemporalExtent", {}).get("RangeDateTime", {}).get("BeginningDateTime", "")
        if not start.startswith(date):
            continue
        urls = {x.get("Type"): x.get("URL") for x in u.get("RelatedUrls", [])}
        adi = (u.get("DataGranule", {}).get("ArchiveAndDistributionInformation") or [{}])[0]
        ck = adi.get("Checksum") or {}
        out.append({"granule_ur": u["GranuleUR"], "name": u["GranuleUR"].split(":")[-1],
                    "concept_id": meta.get("concept-id"), "revision_id": meta.get("revision-id"),
                    "production_time": u.get("DataGranule", {}).get("ProductionDateTime"),
                    "cmr_size": adi.get("Size"), "cmr_size_unit": adi.get("SizeUnit"),
                    "cmr_checksum": ck.get("Value"), "cmr_checksum_algorithm": ck.get("Algorithm"),
                    "time_start": start, "opendap": urls.get("USE SERVICE API"), "data_url": urls.get("GET DATA")})
    return sorted(out, key=lambda g: g["time_start"])


def dap4_day_url(opendap: str, dom: dict) -> str:
    base = opendap[:-5] if opendap.endswith(".html") else opendap
    j = f"[{dom['j0']}:1:{dom['j1']}]"
    i = f"[{dom['i0']}:1:{dom['i1']}]"
    ce = ";".join([f"/Grid/{v}[0:1:0]{j}{i}" for v in DAY_VARS] +
                  [f"/Grid/lat{i}", f"/Grid/lon{j}", "/Grid/time_bnds"])
    return f"{base}.dap.nc4?dap4.ce={ce}"


def subset_to_crop(blob: bytes, w: rc.Window, dom: dict) -> dict:
    """Parse one DAP4 netCDF-4 subset into the ``rc.read_crop`` structure and
    check grid and time against the shared conventions."""
    from src.lake import grid
    out = {}
    with h5py.File(io.BytesIO(blob), "r") as f:
        lat, lon = f["Grid/lat"][:], f["Grid/lon"][:]
        i = np.arange(dom["i0"], dom["i1"] + 1)
        j = np.arange(dom["j0"], dom["j1"] + 1)
        if lat.shape != i.shape or not np.allclose(lat, grid.lat_centre(i), atol=1e-4):
            raise ValueError(f"{w.path.name}: subset latitude does not match grid convention")
        if lon.shape != j.shape or not np.allclose(lon, grid.lon_centre(j), atol=1e-4):
            raise ValueError(f"{w.path.name}: subset longitude does not match grid convention")
        for v in DAY_VARS:
            ds = f[f"Grid/{v}"]
            dn = rc._attr(ds.attrs.get("DimensionNames", b""))
            if dn != "time,lon,lat" or ds.shape != (1, len(j), len(i)):
                raise ValueError(f"{w.path.name}: Grid/{v} unexpected dims {dn!r} {ds.shape}")
            attrs = {}
            for k in rc.ATTR_KEYS:
                if k in ds.attrs:
                    a = ds.attrs[k]
                    attrs[k] = a.reshape(-1)[0] if isinstance(a, np.ndarray) and a.size == 1 else a
            out[f"Grid/{v}"] = (ds[...], attrs)
        tb = f["Grid/time_bnds"][...]
    if GPS_EPOCH + timedelta(seconds=int(tb[0, 0])) != w.start:
        raise ValueError(f"time_bnds start mismatch for {w.path.name}")
    return {"vars": out, "meta": {"time_bnds": tb, "lat": lat, "lon": lon}}


def _append_jsonl(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


def crop_day_opendap(product: str, date: str, dom: dict, verify_full: bool | None = None) -> dict:
    """Backfill path (ADD-004): OPeNDAP subset per granule -> regional day file.
    ``verify_full`` None = verify on the first day of the month."""
    source = f"imerg_{product}"
    verify = date.endswith("-01") if verify_full is None else verify_full
    token = get_earthdata_token()
    gs = [g for g in cmr_granules_umm(product, date) if g["opendap"]]
    t0, nbytes, ws, crops, subs = time.time(), 0, [], [], []
    for g in gs:
        w = rc.imerg_window(Path(g["name"]))
        w.source = source
        url = dap4_day_url(g["opendap"], dom)
        blob, note = retry_once(lambda: _get(url, token), f"opendap {g['name']}")
        if not blob.startswith(HDF5_MAGIC):
            raise RuntimeError(f"{g['name']}: OPeNDAP response is not netCDF-4/HDF5")
        crops.append(subset_to_crop(blob, w, dom))
        ws.append(w)
        subs.append({"url": url, "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest(), "retry": note})
        nbytes += len(blob)
    if not ws:
        return {"product": product, "date": date, "path": "opendap", "granules_listed": len(gs), "windows": 0,
                "download_bytes": 0, "seconds": round(time.time() - t0, 1)}
    t_dap = time.time() - t0

    ver = None
    originals = {}
    if verify:
        t1, mism, vbytes = time.time(), [], 0
        SCRATCH.mkdir(parents=True, exist_ok=True)
        for g, w, c in zip(gs, ws, crops):
            name = Path(g["data_url"]).name
            dest = SCRATCH / name
            try:
                blob, _ = retry_once(lambda: _get(g["data_url"], token), f"verify download {name}")
                if not blob.startswith(HDF5_MAGIC):
                    raise RuntimeError(f"{name}: not an HDF5 file")
                dest.write_bytes(blob)
                vbytes += len(blob)
                originals[g["name"]] = (len(blob), hashlib.sha256(blob).hexdigest())
                ref = rc.read_crop(rc.imerg_window(dest), dom)
            finally:
                dest.unlink(missing_ok=True)
            if set(ref["vars"]) != set(c["vars"]):
                mism.append({"granule": g["name"], "issue": "variable sets differ",
                             "full": sorted(ref["vars"]), "opendap": sorted(c["vars"])})
                continue
            for k in ref["vars"]:
                a, b = c["vars"][k][0], ref["vars"][k][0]
                if a.dtype != b.dtype or not np.array_equal(a, b, equal_nan=True):
                    mism.append({"granule": g["name"], "variable": k})
            if not np.array_equal(c["meta"]["time_bnds"], ref["meta"]["time_bnds"]):
                mism.append({"granule": g["name"], "variable": "time_bnds"})
        ver = {"source": source, "date": date, "granules": len(ws), "exact_match": not mism,
               "mismatches": mism[:20], "full_download_bytes": vbytes,
               "seconds": round(time.time() - t1, 1), "checked_utc": rc.iso(datetime.now(timezone.utc))}
        _append_jsonl(rc.REGIONAL_ROOT / source / "verification.jsonl", ver)
        if mism:
            raise RuntimeError(f"verification FAILED for {date}: {len(mism)} mismatching arrays (see verification.jsonl)")

    out, crop_sha, var_names = rc.write_day_file(source, date, dom, ws, crops)
    now = rc.iso(datetime.now(timezone.utc))
    lines = []
    for t, (g, w, sb) in enumerate(zip(gs, ws, subs)):
        size, sha = originals.get(g["name"], (None, None))
        lines.append({
            "source": source, "date": date, "time_index": t,
            "original_name": g["name"], "original_path": g["data_url"],
            "original_size": size, "original_sha256": sha,
            "window_start_utc": rc.iso(w.start), "window_end_utc": rc.iso(w.end),
            "product_version": w.version,
            "retrieval_path": "opendap_dap4",
            "cmr_granule_ur": g["granule_ur"], "cmr_concept_id": g["concept_id"],
            "cmr_revision_id": g["revision_id"], "cmr_production_time": g["production_time"],
            "cmr_size": g["cmr_size"], "cmr_size_unit": g["cmr_size_unit"],
            "cmr_checksum": g["cmr_checksum"], "cmr_checksum_algorithm": g["cmr_checksum_algorithm"],
            "opendap_url": sb["url"], "subset_bytes": sb["bytes"], "subset_sha256": sb["sha256"],
            "verified_full_download": bool(ver and ver["exact_match"]),
            "crop_file": rc.rel(out), "crop_sha256": crop_sha, "built_utc": now,
        })
    rc.write_manifest_day(source, date, lines)
    return {"product": product, "date": date, "path": "opendap", "granules_listed": len(gs), "windows": len(ws),
            "download_bytes": nbytes, "opendap_seconds": round(t_dap, 1),
            "verification": None if ver is None else {k: ver[k] for k in ("exact_match", "granules", "seconds",
                                                                            "full_download_bytes")},
            "seconds": round(time.time() - t0, 1), "crop_bytes": out.stat().st_size}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", required=True, choices=SHORT)
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--crop-day", metavar="YYYY-MM-DD", help="backfill one UTC day (see --path)")
    ap.add_argument("--path", choices=("opendap", "fallback"), default="opendap",
                    help="--crop-day retrieval path (default opendap, ADD-004)")
    ap.add_argument("--verify-full", action="store_true",
                    help="--path opendap: verify every granule by full download (default: only 1st of month)")
    ap.add_argument("--limit", type=int, default=2)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--fallback", action="store_true", help="also measure full-download path")
    ap.add_argument("--compare-local", action="store_true")
    a = ap.parse_args(argv)
    dom = rc.load_domain()
    if a.crop_day:
        try:
            if a.path == "opendap":
                r = crop_day_opendap(a.product, a.crop_day, dom, True if a.verify_full else None)
            else:
                r = crop_day_full(a.product, a.crop_day, dom)
        except Exception as exc:
            print(json.dumps({"product": a.product, "date": a.crop_day, "status": "FAILED", "error": str(exc)[:600]}))
            return 2
        r.pop("crop", None)
        print(json.dumps(r))
        return 0 if r["windows"] >= 46 else 1
    if not (a.start and a.end):
        ap.error("--start and --end required unless --crop-day")
    token = get_earthdata_token()
    gs = cmr_granules(a.product, a.start, a.end)[: a.limit]
    if not gs:
        print(json.dumps({"product": a.product, "error": "no granules in CMR for range"}))
        return 1
    rc_code = 0
    for g in gs:
        row = {"product": a.product, "granule": g["title"], "time_start": g["time_start"],
               "global_size_mb": g["granule_size_mb"]}
        sub = None
        if g["opendap"]:
            url = dap4_url(g["opendap"], dom)
            t0 = time.time()
            try:
                blob, note = retry_once(lambda: _get(url, token), "opendap subset")
                row |= {"opendap_seconds": round(time.time() - t0, 2), "opendap_bytes": len(blob),
                        "opendap_retry_note": note}
                sub = read_subset(blob)
                if a.download:
                    d = datetime.fromisoformat(g["time_start"].replace("Z", "+00:00")).astimezone(timezone.utc)
                    out = rc.REGIONAL_ROOT / f"imerg_{a.product}" / "subsets" / f"date={d:%Y-%m-%d}" / \
                        (Path(g["title"].split(":")[-1]).stem + ".subset.nc4")  # ":" is illegal in Windows names
                    out.parent.mkdir(parents=True, exist_ok=True)
                    tmp = out.with_suffix(".part")
                    tmp.write_bytes(blob)
                    os.replace(tmp, out)
                    row["subset_file"] = rc.rel(out)
                    row["subset_sha256"] = hashlib.sha256(blob).hexdigest()
            except Exception as exc:
                row["opendap_error"] = str(exc)[:600]
                rc_code = 3
        else:
            row["opendap_error"] = "no OPeNDAP link in granule metadata"
            rc_code = 3
        if a.compare_local and sub is not None:
            ref = local_crop(g["title"], dom)
            row["compare_local"] = compare(sub, ref) if ref else "no local global file"
        if a.fallback or sub is None:
            try:
                fb = fallback(g, token, dom)
                arrs = fb.pop("arrays")
                row["fallback"] = fb
                if sub is not None:
                    row["opendap_vs_fallback"] = compare(sub, arrs)
            except Exception as exc:
                row["fallback_error"] = str(exc)[:600]
                rc_code = 4
        print(json.dumps(row, default=str), flush=True)
    return rc_code


if __name__ == "__main__":
    raise SystemExit(main())
