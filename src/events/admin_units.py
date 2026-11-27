"""Administrative units of Hanoi city from the existing OSM snapshot (read-only on Bronze).

Extracts `boundary=administrative` multipolygons intersecting the Hanoi city bbox from
`data/bronze/urban_vector/osm/raw/snapshot_date=2026-10-12/vietnam-20261012.osm.pbf` with the
GDAL OSM driver and writes `data/reference/hanoi_city_admin_osm_20261012.gpkg` (+ provenance).
Licence: OpenStreetMap contributors, ODbL 1.0. Limits: OSM snapshot of 2026-10-12, i.e. the
units after the 1 July 2025 reform where mappers updated them; old districts may coexist.

Usage (subprocess with an RSS watchdog):
  .venv/Scripts/python.exe -m src.events.admin_units [--max-rss-gb 0.4]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

from src.events.common import ROOT, rel, sha256_file, write_json_atomic

PBF = ROOT / "data/bronze/urban_vector/osm/raw/snapshot_date=2026-10-12/vietnam-20261012.osm.pbf"
OUT = ROOT / "data/reference/hanoi_city_admin_osm_20261012.gpkg"
PROV = ROOT / "data/reference/hanoi_city_admin_osm_20261012.provenance.json"
DOMAIN = ROOT / "data/reference/hanoi_city_domain.json"
TAG_KEYS = ("admin_level", "was:admin_level", "was:boundary", "old_name", "name:en", "name:vi", "official_name", "start_date",
            "end_date", "place", "ref", "wikidata", "border_type")


def city_bbox() -> tuple[float, float, float, float]:
    import geopandas as gpd
    b = gpd.read_file(ROOT / "data/reference/hanoi_city_boundary.geojson").total_bounds
    return float(b[0]), float(b[1]), float(b[2]), float(b[3])


def parse_hstore(s: str | None) -> dict[str, str]:
    import re
    if not isinstance(s, str):
        return {}
    return dict(re.findall(r'"([^"]+)"=>"((?:[^"\\]|\\.)*)"', s))


def extract() -> int:
    os.environ.setdefault("OSM_MAX_TMPFILE_SIZE", "8")  # MB kept in RAM, rest spills to disk
    os.environ.setdefault("CPL_TMPDIR", str(ROOT / "data/_scratch/u1/gdal_tmp"))
    (ROOT / "data/_scratch/u1/gdal_tmp").mkdir(parents=True, exist_ok=True)
    import pyogrio

    bbox = city_bbox()
    t0 = time.time()
    df = pyogrio.read_dataframe(PBF, layer="multipolygons", bbox=bbox,
                                where="boundary = 'administrative' OR other_tags LIKE "
                                      "'%\"was:boundary\"=>\"administrative\"%'",
                                columns=["osm_id", "name", "admin_level", "boundary", "other_tags"])
    tags = df["other_tags"].map(parse_hstore)
    for k in TAG_KEYS:
        if k == "admin_level":
            continue
        df[k.replace(":", "_")] = tags.map(lambda d, k=k: d.get(k))
    df = df.drop(columns=["other_tags"])
    # era: current units (boundary=administrative) vs units retired by the 2025-07-01 reform
    df["era"] = df["boundary"].map(lambda b: "current" if b == "administrative" else "pre_2025_07")
    df["level"] = df["admin_level"].fillna(df["was_admin_level"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_name(OUT.stem + ".tmp.gpkg")
    df.to_file(tmp, driver="GPKG", layer="admin_units")
    os.replace(tmp, OUT)
    prov = {"source": rel(PBF), "source_sha256": sha256_file(PBF), "driver": "GDAL OSM",
            "filter": "layer=multipolygons, boundary=administrative, bbox=Hanoi city boundary bounds",
            "bbox": bbox, "feature_count": int(len(df)),
            "level_era_counts": {f"{k[0]}|{k[1]}": int(v) for k, v in
                                 df.groupby(["level", "era"]).size().items()},
            "licence": "OpenStreetMap contributors, ODbL 1.0",
            "limits": "Snapshot 2026-10-12; post-2025-07-01 units where mapped; may also hold "
                      "pre-reform districts; not an official cadastral boundary.",
            "created_utc": datetime.now(timezone.utc).isoformat(), "seconds": round(time.time() - t0, 1),
            "output": rel(OUT), "output_sha256": sha256_file(OUT)}
    write_json_atomic(PROV, prov)
    print(f"ADMIN_UNITS={len(df)} LEVELS={prov['level_era_counts']} SECONDS={prov['seconds']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--force", action="store_true", help="rebuild own output")
    ap.add_argument("--max-rss-gb", type=float, default=0.4)
    ap.add_argument("--timeout-s", type=float, default=590)
    a = ap.parse_args()
    if a.child:
        return extract()
    if OUT.exists() and "--force" not in sys.argv:
        print(f"EXISTS {rel(OUT)} (skip)")
        return 0
    import psutil
    p = subprocess.Popen([sys.executable, "-m", "src.events.admin_units", "--child"], cwd=ROOT)
    proc, peak, t0 = psutil.Process(p.pid), 0.0, time.time()
    while p.poll() is None:
        try:
            procs = [proc] + proc.children(recursive=True)
            peak = max(peak, sum(q.memory_info().rss for q in procs) / 2**30)
        except psutil.NoSuchProcess:
            break
        if peak > a.max_rss_gb or time.time() - t0 > a.timeout_s:
            p.kill()
            print(f"KILLED peak_rss_gb={peak:.3f} elapsed_s={time.time() - t0:.0f}")
            return 3
        time.sleep(0.5)
    print(f"EXIT={p.returncode} PEAK_RSS_GB={peak:.3f} ELAPSED_S={time.time() - t0:.0f}")
    return p.returncode


if __name__ == "__main__":
    raise SystemExit(main())
