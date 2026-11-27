"""U1.3 — event catalog (not labels): one row per Silver flood-evidence record.

Output: data/gold/event_catalog/events.parquet + schema.json (+ reports/u1/catalog_stats.json).
Reads Silver records, Bronze manifests and reference files read-only.

Temporal rules (never finer than the text):
  * observed_time_start/end present → `timestamp`.
  * clock windows in the text ("0h-6h và 8h-9h ngày 28/09/2023") → `timestamp`.
  * part-of-day words (sáng, chiều, tối, rạng sáng, morning, a.m. …) with a date → `part_of_day`.
  * night words (đêm, night, overnight, previous night) without a parsable date → `date_range`
    [article event date − 1 day, article event date].
  * explicit range ("29-30/09/1993", "31/10-02/11/2008", "cuối tháng 10/2008") → `date_range`.
  * a single date → `date`; no date at all → `none` (or the inbox event date for observations,
    flagged `date_source = inbox_event_date`, precision `date` at most).
Spatial rules: exact equality after normalisation (NFC, lower case, diacritics folded, đ→d,
unit words and fillers removed), unique match only, era-aware administrative units (event before
2025-07-01 → pre-reform districts, after → current wards/communes). No fuzzy matching, no geocoding,
no nearest-feature guess.

Usage: .venv/Scripts/python.exe -m src.events.build_event_catalog
"""
from __future__ import annotations

import json
import re
import unicodedata
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from src.events.common import (BRONZE_FE, CATALOG_DIR, REPORTS, ROOT, SILVER_FE, rel, sha256_file,
                               write_json_atomic)

TZ = ZoneInfo("Asia/Ho_Chi_Minh")
REFORM = date(2025, 7, 1)
ADMIN = ROOT / "data/reference/hanoi_city_admin_osm_20261012.gpkg"
CITY = ROOT / "data/reference/hanoi_city_boundary.geojson"
PIXELS = ROOT / "data/reference/rain_pixels_hanoi_city.parquet"
PIX_H3 = ROOT / "data/reference/pixel_h3_map_hanoi_city.parquet"
OSM_DIR = ROOT / "data/silver/spatial_reference/osm/snapshot_date=2026-10-12"
LINES = OSM_DIR / "hanoi_core_osm_lines.gpkg"
WAYS = OSM_DIR / "hanoi_core_osm_way_multipolygons.gpkg"
MET_SEASONS = [(date(y, 5, 1), date(y, 10, 31)) for y in (2023, 2024, 2025)]
ALIASES = ROOT / "configs/admin_aliases.yaml"
OVERRIDES = ROOT / "configs/event_catalog_overrides.yaml"


def load_yaml(path: Path) -> dict:
    import yaml
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}


TONE_UNITS = r"(?:thành phố|quận|huyện|thị xã|phường|xã|district|ward|commune|city|khu vực|tại)"


def tone_core(s: str) -> str:
    """Core name with tone marks KEPT (NFC, lower case): used for exact-name aliases only."""
    s = re.sub(r"\s+", " ", unicodedata.normalize("NFC", s or "").lower()).strip()
    s = re.sub(r"(,\s*(hà nội|hanoi|ha noi))+\s*$", "", s).strip(" ,")
    s = re.sub(r"\(.*?\)", "", s).split(",")[0].strip()
    s = re.sub(rf"^(?:{TONE_UNITS}\s+)+", "", s)
    return s.strip(" -")


def location_verbatim(loc: str | None, text: str) -> tuple[str | None, bool | None]:
    """Verbatim span of `loc` in the artifact text, or verbatim parts joined by ' | ' (None if none).

    Returns (span, context_added); context_added = the location_text holds words not found verbatim
    (agent-added ', Hà Nội', unit word, case change or composition). (None, None) when the artifact has
    no text layer (image) or there is no location text.
    """
    from src.events.text_extract import normalise
    if not loc or not text:
        return None, None
    t = normalise(text.replace("\n", " "))
    tl = t.lower()

    def find(c: str) -> str | None:
        c = normalise(c)
        if not c:
            return None
        if c in t:
            return c
        i = tl.find(c.lower())
        if i >= 0 and len(tl) == len(t):
            return t[i:i + len(c)]  # original casing of the artifact
        return None

    def bare(c: str) -> str:
        return re.sub(r"^(?:phường|xã|huyện|quận|đường|phố)\s+", "", c.strip(), flags=re.I)

    full = normalise(loc)
    if full in t:
        return full, False
    s = re.sub(r"(,\s*(Hà Nội|Hanoi))+\s*$", "", full).strip(" ,")
    for c in (s, bare(s)):
        hit = find(c)
        if hit:
            return hit, True
    parts = []
    for part in (x.strip() for x in s.split(",")):
        hit = find(part) or find(bare(part))
        if hit:
            parts.append(hit)
    return (" | ".join(parts) if parts else None), True

# ----------------------------------------------------------------------------------- text utils

def fold(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "").lower().replace("đ", "d")
    s = "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")
    s = re.sub(r"[‐-―–—]", "-", s)
    return re.sub(r"\s+", " ", s).strip()


UNIT_WORDS = r"(?:thanh pho|quan|huyen|thi xa|phuong|phuong|xa|district|ward|commune|city|khu vuc|tai)"
STREET_WORDS = ("duong", "pho", "street", "road", "boulevard", "tuyen duong", "ngo", "dai lo")
FEATURE_WORDS = ("khu do thi", "nut giao", "nga ba", "nga tu", "ky tuc xa", "ham", "san ", "vien ",
                 "university", "tunnel", "chung cu", "geleximco", "dai hoc")
CITY_TEXTS = {"ha noi", "hanoi", "noi thanh ha noi", "thanh pho ha noi", "mot so khu vuc, ha noi",
              "several areas, hanoi", "certain districts, hanoi", "khu vuc noi thanh ha noi",
              "noi thanh ha noi, luu vuc song nhue"}
CITY_SUFFIX = re.compile(r"(,\s*(ha noi|hanoi))+\s*$")


def strip_city(s: str) -> str:
    return CITY_SUFFIX.sub("", s).strip(" ,")


def core_name(s: str) -> str:
    """Folded name without unit words, fillers, parentheses and trailing city."""
    s = strip_city(fold(s))
    s = re.sub(r"\(.*?\)", "", s)
    s = s.split(",")[0]
    s = re.sub(rf"^(?:{UNIT_WORDS}\s+)+", "", s.strip())
    s = re.sub(rf"\s+(?:{UNIT_WORDS})$", "", s)
    return s.strip(" -")


# ------------------------------------------------------------------------------- temporal parse

PART_OF_DAY = re.compile(r"\b(sang|chieu|toi|trua|rang sang|morning|afternoon|evening|a\.m\.|p\.m\.|"
                         r"sang som|wee hours)\b")
NIGHT = re.compile(r"\b(dem|night|overnight|previous night)\b")
MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july",
                                       "august", "september", "october", "november", "december"], 1)}


def _d(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def parse_dates(text: str, year_hint: int | None) -> tuple[date | None, date | None, str]:
    """Return (start, end, kind) from a temporal text; kind ∈ {range, single, month, none}."""
    t = fold(text or "")
    m = re.search(r"(\d{1,2})/(\d{1,2})\s*-\s*(\d{1,2})/(\d{1,2})/(\d{4})", t)
    if m:
        y = int(m.group(5))
        a, b = _d(y, int(m.group(2)), int(m.group(1))), _d(y, int(m.group(4)), int(m.group(3)))
        if a and b and a > b:
            a = _d(y - 1, int(m.group(2)), int(m.group(1)))
        return a, b, "range"
    m = re.search(r"(\d{1,2})\s*-\s*(\d{1,2})/(\d{1,2})/(\d{4})", t)
    if m:
        y, mo = int(m.group(4)), int(m.group(3))
        return _d(y, mo, int(m.group(1))), _d(y, mo, int(m.group(2))), "range"
    m = re.search(r"tu \d{1,2}h(?:\d{2})? ngay (\d{1,2})/(\d{1,2}) den \d{1,2}h(?:\d{2})? ngay (\d{1,2})/(\d{1,2})", t)
    if m and year_hint:
        return (_d(year_hint, int(m.group(2)), int(m.group(1))),
                _d(year_hint, int(m.group(4)), int(m.group(3))), "range")
    # "từ trưa 25/8 đến trưa 26/8", "từ 25/8 đến 26/8" (optional part-of-day word before each date)
    m = re.search(r"tu (?:[^/]{0,15}? )?(\d{1,2})/(\d{1,2}) den (?:[^/]{0,15}? )?(\d{1,2})/(\d{1,2})(?![\d/])", t)
    if m and year_hint:
        return (_d(year_hint, int(m.group(2)), int(m.group(1))),
                _d(year_hint, int(m.group(4)), int(m.group(3))), "range")
    m = re.search(r"ngay (\d{1,2}) thang (\d{1,2}) nam (\d{4})", t)
    if m:
        d = _d(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        return d, d, "single"
    m = re.search(r"(?:cuoi |dau |giua )?thang (\d{1,2})/(\d{4})", t)
    if m:
        y, mo = int(m.group(2)), int(m.group(1))
        nxt = date(y + (mo == 12), mo % 12 + 1, 1)
        return date(y, mo, 1), nxt - timedelta(days=1), "month"
    m = re.search(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", t)
    if m:
        d = _d(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        return d, d, "single"
    m = re.search(r"(" + "|".join(MONTHS) + r") (\d{1,2}), (\d{4})", t)
    if m:
        d = _d(int(m.group(3)), MONTHS[m.group(1)], int(m.group(2)))
        return d, d, "single"
    m = re.search(r"(?<![\d/])(\d{1,2})[/-](\d{1,2})(?![\d/])", t)
    if m and year_hint:
        d = _d(year_hint, int(m.group(2)), int(m.group(1)))
        return d, d, "single"
    return None, None, "none"


CLOCK_WIN = re.compile(r"(\d{1,2})h(\d{2})?\s*-\s*(\d{1,2})h(\d{2})?")


def clock_windows(text: str, day: date | None) -> list[tuple[datetime, datetime]]:
    """Clock windows like '0h-6h và 8h-9h' on `day` (local) → list of tz-aware local datetimes."""
    if day is None:
        return []
    out = []
    for m in CLOCK_WIN.finditer(fold(text or "")):
        h0, m0, h1, m1 = int(m.group(1)), int(m.group(2) or 0), int(m.group(3)), int(m.group(4) or 0)
        if h0 > 24 or h1 > 24:
            continue
        a = datetime(day.year, day.month, day.day, tzinfo=TZ) + timedelta(hours=h0, minutes=m0)
        b = datetime(day.year, day.month, day.day, tzinfo=TZ) + timedelta(hours=h1, minutes=m1)
        if b > a:
            out.append((a, b))
    return out


def temporal(rec: dict, ctx: dict) -> dict:
    """Normalised time fields of one Silver record. ctx: event_date/published_date from Bronze."""
    tt = rec.get("temporal_text") or ""
    t0, t1 = rec.get("observed_time_start"), rec.get("observed_time_end")
    hint_date = ctx.get("event_date") or ctx.get("published_date")
    year_hint = hint_date.year if hint_date else None
    res = {"event_date_start": None, "event_date_end": None, "time_start_utc": None, "time_end_utc": None,
           "time_precision": "none", "date_source": None}
    if t0 or t1:
        a = datetime.fromisoformat(t0) if t0 else None
        b = datetime.fromisoformat(t1) if t1 else None
        la, lb = (a or b).astimezone(TZ), (b or a).astimezone(TZ)
        res.update(event_date_start=la.date(), event_date_end=lb.date(), time_precision="timestamp",
                   time_start_utc=a.astimezone(ZoneInfo("UTC")) if a else None,
                   time_end_utc=b.astimezone(ZoneInfo("UTC")) if b else None, date_source="observed_time")
        return res
    d0, d1, kind = parse_dates(tt, year_hint)
    if kind in ("range", "month"):
        res.update(event_date_start=d0, event_date_end=d1, time_precision="date_range", date_source="temporal_text")
        return res
    if kind == "single":
        wins = clock_windows(tt, d0)
        if wins:
            res.update(event_date_start=d0, event_date_end=d0, time_precision="timestamp",
                       time_start_utc=wins[0][0].astimezone(ZoneInfo("UTC")),
                       time_end_utc=wins[-1][1].astimezone(ZoneInfo("UTC")), date_source="temporal_text")
        else:
            prec = "part_of_day" if PART_OF_DAY.search(fold(tt)) else "date"
            res.update(event_date_start=d0, event_date_end=d0, time_precision=prec, date_source="temporal_text")
        return res
    # no date in the text: observations fall back to the article's event date (never finer than date
    # except a stated part of day); night words span the previous day.
    if rec.get("event_type") in ("flood_observation", "rainfall_observation", "flood_impact") and ctx.get("event_date"):
        ed = ctx["event_date"]
        if NIGHT.search(fold(tt)):
            res.update(event_date_start=ed - timedelta(days=1), event_date_end=ed, time_precision="date_range")
        elif PART_OF_DAY.search(fold(tt)) or re.search(r"\d{1,2}h|\d:\d{2}", fold(tt)):
            res.update(event_date_start=ed, event_date_end=ed, time_precision="part_of_day")
        else:
            res.update(event_date_start=ed, event_date_end=ed, time_precision="date")
        res["date_source"] = "inbox_event_date"
    return res


# ------------------------------------------------------------------------------- spatial parse

def location_level(loc: str) -> tuple[str | None, str]:
    """(level, kind) where kind ∈ {city, street, feature, ward, district, bare}."""
    f = fold(loc or "")
    if not f:
        return None, "none"
    if strip_city(f) == "" or f in CITY_TEXTS:
        return "city", "city"
    markers = []
    for w in STREET_WORDS:
        m = re.search(rf"(?:^|[\s(,]){re.escape(w)}(?=\s|$)", f)
        if m:
            markers.append((m.start(), "street"))
    for w in FEATURE_WORDS:
        i = f.find(w)
        if i >= 0:
            markers.append((i, "feature"))
    for w, lv in (("phuong", "ward"), ("xa", "ward"), ("ward", "ward"), ("commune", "ward"),
                  ("quan", "district"), ("huyen", "district"), ("district", "district")):
        m = re.search(rf"(?:^|[\s(,']){w}(?=\s|$|,)", f)
        if m:
            markers.append((m.start(), lv))
    if markers:
        markers.sort()
        lv = markers[0][1]
        return lv, lv
    return None, "bare"


class Resolver:
    def __init__(self) -> None:
        import geopandas as gpd
        import pandas as pd
        import pyogrio

        self.gpd, self.pd = gpd, pd
        adm_all = gpd.read_file(ADMIN)
        hn = adm_all[(adm_all["level"] == "4") & (adm_all["name"] == "Thành phố Hà Nội")
                     & (adm_all["era"] == "current")]
        self.hanoi_osm = hn.geometry.union_all()
        adm = adm_all[adm_all["level"] == "6"].copy()
        # units of Hanoi only (the extraction bbox also holds neighbouring provinces)
        adm = adm[adm.representative_point().within(self.hanoi_osm)].copy()
        adm["key"] = adm["name"].map(core_name)
        self.admin = adm
        self.aliases = load_yaml(ALIASES).get("aliases", [])
        self.city = gpd.read_file(CITY).to_crs(4326).geometry.union_all()
        px = pd.read_parquet(PIXELS)
        from shapely.geometry import box
        px["geometry"] = [box(lo - 0.05, la - 0.05, lo + 0.05, la + 0.05) for la, lo in zip(px.lat, px.lon)]
        self.pixels = gpd.GeoDataFrame(px, geometry="geometry", crs=4326)
        self.pix_h3 = pd.read_parquet(PIX_H3, columns=["h3_id", "pixel_id"]).set_index("h3_id")["pixel_id"]
        ln = pyogrio.read_dataframe(LINES, columns=["osm_id", "name"], read_geometry=False, fid_as_index=True)
        ln = ln[ln["name"].notna()]
        ln["key"] = ln["name"].map(lambda s: re.sub(r"^(duong|pho|dai lo)\s+", "", fold(s)))
        self.lines = ln
        wy = pyogrio.read_dataframe(WAYS, columns=["osm_id", "osm_way_id", "name"], read_geometry=False,
                                    fid_as_index=True)
        wy = wy[wy["name"].notna()]
        wy["key"] = wy["name"].map(fold)
        self.ways = wy

    # -- helpers
    def pixels_for(self, geom) -> list[int]:
        hit = self.pixels[self.pixels.intersects(geom)]
        return sorted(int(p) for p in hit.pixel_id)

    def pixels_for_h3(self, cells: list[str]) -> list[int]:
        return sorted({int(self.pix_h3[c]) for c in cells if c in self.pix_h3.index})

    def admin_match(self, name_key: str, era: str | None):
        cand = self.admin[self.admin["key"] == name_key]
        if era:
            cand = cand[cand["era"] == era]
        return cand

    def resolve(self, loc: str, ev_date: date | None) -> dict:
        import h3
        from shapely.geometry import mapping

        lv, kind = location_level(loc)
        out = {"location_level": lv, "geo_status": "unresolved", "geo_reason": None, "feature_id": None,
               "feature_name": None, "n_features": 0, "h3_cells": [], "rain_pixel_ids": []}
        era = None if ev_date is None else ("current" if ev_date >= REFORM else "pre_2025_07")
        if kind == "none":
            out["geo_reason"] = "no location text"
            return out
        if kind == "city":
            out.update(geo_status="resolved", feature_id="geoboundaries:VNM-ADM1-63759600",
                       feature_name="Hà Nội", n_features=1, rain_pixel_ids=self.pixels_for(self.city))
            return out
        if kind in ("ward", "district", "bare"):
            key = core_name(loc)
            if not key or "," in key or " gom" in key or re.search(r"\d", key):
                out["geo_reason"] = "multiple or composite unit names"
                return out
            alias_note = None
            for al in self.aliases:  # exact-name alias (tone marks kept), user-approved only
                if tone_core(loc) == tone_core(al["text_name"]) and (al.get("era") in (None, era)):
                    key = core_name(al["osm_name"])
                    alias_note = f"alias (configs/admin_aliases.yaml): {al['text_name']} -> {al['osm_name']}"
            cand = self.admin_match(key, era)
            if len(cand) == 1:
                r = cand.iloc[0]
                unit_level = "ward" if r["era"] == "current" else "district"
                if kind == "ward" and unit_level != "ward" or kind == "district" and unit_level != "district":
                    out["geo_reason"] = f"text level {kind} but only a {unit_level} named '{key}' in era {era}"
                    return out
                out.update(location_level=unit_level, geo_status="resolved", feature_id=f"osm:relation/{r['osm_id']}",
                           geo_reason=alias_note,
                           feature_name=r["name"], n_features=1, rain_pixel_ids=self.pixels_for(r.geometry))
                return out
            if len(cand) > 1:
                out["geo_reason"] = f"{len(cand)} units named '{key}' (era {era})"
                return out
            if kind != "bare":
                out["geo_reason"] = f"no {kind} named '{key}' in OSM admin units (era {era})"
                return out
            kind = "street"  # bare name: try a street of that exact name
            out["location_level"] = "street"
        if kind == "street":
            f = strip_city(fold(loc))
            m = re.search(r"(?:^|\s|\()(tuyen duong|duong|pho|dai lo|ngo)\s+(.+)", f)
            if m:
                name = (("ngo " if m.group(1) == "ngo" else "") + m.group(2))
            else:
                name = re.sub(r"\s+(street|road|boulevard)\b.*$", "", f)
            name = re.split(r"[,(]| doan | qua | gan | in ", name)[0].strip()
            if " - " in name:
                out.update(location_level="street", geo_reason="compound street name")
                return out
            hits = self.lines[self.lines["key"] == name]
            if hits.empty:
                out.update(location_level="street",
                           geo_reason=f"no OSM line named '{name}' in the core-area cache")
                return out
            import pyogrio
            g = pyogrio.read_dataframe(LINES, fids=list(hits.index), columns=["osm_id", "name"])
            cells = set()
            for geom in g.geometry:
                for part in getattr(geom, "geoms", [geom]):
                    n = max(2, int(part.length / 0.0005))
                    for i in range(n + 1):
                        p = part.interpolate(i / n, normalized=True)
                        cells.add(h3.latlng_to_cell(p.y, p.x, 9))
            out.update(location_level="street", geo_status="resolved", feature_id=f"osm:line_name/{name}",
                       feature_name=str(g["name"].iloc[0]), n_features=int(len(g)), h3_cells=sorted(cells),
                       rain_pixel_ids=self.pixels_for_h3(sorted(cells)))
            return out
        # feature
        key = strip_city(fold(loc)).split(",")[0].strip()
        hits = self.ways[self.ways["key"] == key]
        if len(hits) == 0:
            out["geo_reason"] = f"no OSM area named '{key}' in the core-area cache"
            return out
        import pyogrio
        g = pyogrio.read_dataframe(WAYS, fids=list(hits.index), columns=["osm_id", "osm_way_id", "name"])
        cells = set()
        for geom in g.geometry:
            cs = h3.geo_to_cells(mapping(geom), 9)
            cells.update(cs or [h3.latlng_to_cell(geom.centroid.y, geom.centroid.x, 9)])
        out.update(geo_status="resolved", feature_id=f"osm:area_name/{key}", feature_name=str(g["name"].iloc[0]),
                   n_features=int(len(g)), h3_cells=sorted(cells), rain_pixel_ids=self.pixels_for_h3(sorted(cells)))
        return out


# ------------------------------------------------------------------------------- catalog rows

QUAL = [(r"(tren|hon|more than|exceeded|over)\s*$", "gt"), (r"(gan|khoang|around|about)\s*$", "approx"),
        (r"(len toi|up to|toi)\s*$", "max")]


def rainfall_qualifier(text: str, mm: float | None) -> str | None:
    if mm is None:
        return None
    f = fold(text)
    num = f"{mm:g}".replace(".", "[.,]")
    m = re.search(rf"([a-z\s]{{0,25}})\b{num}\s*mm", f)
    if not m:
        return "eq"
    for pat, q in QUAL:
        if re.search(pat, m.group(1)):
            return q
    return "eq"


def manifest_context(evidence_id: str) -> dict:
    p = next((BRONZE_FE / "manifest").rglob(f"{evidence_id}.json"))
    m = json.loads(p.read_text(encoding="utf-8"))
    notes = m.get("notes") or ""
    ctx = {"source_name": m.get("source_name"), "source_uri": m.get("source_uri"), "notes": notes,
           "evidence_type": m.get("evidence_type")}
    me = re.search(r"event_date=(\d{4}-\d{2}-\d{2})", notes)
    mp = re.search(r"published_date=(\d{4}-\d{2}-\d{2})", notes)
    ctx["event_date"] = date.fromisoformat(me.group(1)) if me else None
    if mp:
        ctx["published_date"] = date.fromisoformat(mp.group(1))
    elif m.get("published_time"):
        ctx["published_date"] = datetime.fromisoformat(m["published_time"]).date()
    else:
        ctx["published_date"] = None
    return ctx


def in_met_history(d0: date | None, d1: date | None) -> bool:
    if d0 is None:
        return False
    d1 = d1 or d0
    return any(a <= d0 <= b or a <= d1 <= b for a, b in MET_SEASONS)


def flood_cause(rec: dict, ctx: dict) -> str | None:
    if rec["event_type"] not in ("flood_observation", "flood_impact"):
        return None
    t = fold(rec.get("extracted_text", "")) + " " + fold(ctx.get("notes", ""))
    if re.search(r"nuoc song dang|lu song|song hong|red river|ven song", t):
        return "river"
    return "rain"


def usable_for(row: dict) -> list[str]:
    u = []
    fine = row["location_level"] in ("feature", "street", "ward", "district") and row["geo_status"] == "resolved"
    if not row["in_met_history"]:
        return ["history_only"] if row["event_date_start"] is not None else []
    if row["event_type"] == "rainfall_observation" and row["rainfall_mm"] is not None and fine:
        u.append("rain_validation")
    if row["event_type"] == "flood_observation" and row["flood_cause"] == "rain" and row["geo_status"] == "resolved":
        u.append("risk_validation")
    if row["event_type"] in ("flood_observation", "rainfall_observation", "flood_impact"):
        u.append("replay")
    return u


def load_silver() -> list[tuple[Path, dict]]:
    return [(p, json.loads(p.read_text(encoding="utf-8"))) for p in sorted(SILVER_FE.rglob("fer_*.json"))]


SCHEMA = {
    "silver_record_id": ("string", "Silver record id (fer_…)"),
    "evidence_id": ("string", "Bronze evidence id"),
    "evidence_type": ("string", "article | pdf | image"),
    "source_name": ("string", "From the Bronze manifest"),
    "source_uri": ("string", "From the Bronze manifest"),
    "event_type": ("string", "flood_observation | rainfall_observation | flood_impact | context"),
    "event_date_start": ("date", "Local (Asia/Ho_Chi_Minh) date"),
    "event_date_end": ("date", "Local date"),
    "time_start_utc": ("timestamp[UTC]", "Only when the text gives clock times"),
    "time_end_utc": ("timestamp[UTC]", "Only when the text gives clock times"),
    "time_precision": ("string", "timestamp | part_of_day | date | date_range | none"),
    "date_source": ("string", "observed_time | temporal_text | inbox_event_date"),
    "temporal_text": ("string", "Original wording"),
    "location_text": ("string", "Original wording"),
    "location_text_verbatim": ("string", "Verbatim span of location_text found in the artifact, or verbatim parts joined by ' | '; null if none / no text layer"),
    "location_context_added": ("bool", "location_text holds words not found verbatim (agent-added ', Hà Nội', unit word, case, composition); null if no text layer"),
    "location_level": ("string", "feature | street | ward | district | city (null if not determinable)"),
    "geo_status": ("string", "resolved | unresolved"),
    "geo_reason": ("string", "Why unresolved"),
    "feature_id": ("string", "osm:relation/<id>, osm:line_name/<name>, osm:area_name/<name> or geoBoundaries id"),
    "feature_name": ("string", "Matched reference name"),
    "n_features": ("int", "Number of matched OSM features"),
    "h3_cells": ("list<string>", "H3 res 9 cells (feature, street)"),
    "rain_pixel_ids": ("list<int>", "0.1° rain pixels intersecting the unit / covering the cells"),
    "rainfall_mm": ("double", "Stated amount"),
    "rainfall_qualifier": ("string", "eq | gt (lower bound) | approx | max (up to)"),
    "rainfall_duration_hours": ("double", "Stated duration"),
    "rain_period_text": ("string", "temporal_text of rainfall records"),
    "flood_depth_m": ("double", "Stated single depth"),
    "flood_depth_text": ("string", "Original wording"),
    "inundation_duration_hours": ("double", "Stated duration"),
    "flood_cause": ("string", "rain | river (keyword rule) for flood records"),
    "confidence": ("string", "high | medium | low"),
    "review_status": ("string", "pending | reviewed | rejected"),
    "extraction_method": ("string", "manual | rule_based | automated"),
    "silver_valid": ("bool", "Passes the current Silver validator"),
    "in_met_history": ("bool", "Date inside 1 May–31 Oct 2023/2024/2025 (met history)"),
    "usable_for": ("list<string>", "rain_validation | risk_validation | replay | history_only"),
    "extracted_text": ("string", "Verbatim passage"),
    "superseded_by": ("string", "Corrected Silver record replacing this one (configs/event_catalog_overrides.yaml); row kept, usable_for = []"),
    "review_effective": ("string", "U1-R review overlay (data/gold/event_catalog/review_overlay.parquet): auto_checked | pending | duplicate | human_reviewed | human_rejected | ai_reviewed | ai_rejected (U1-AI; human values win); null for superseded rows or before U1-R"),
}


def build() -> dict:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from flood_evidence_silver_validator import load_contract, validate_silver_record

    contract = load_contract()
    from src.events.text_extract import artifact_text
    res = Resolver()
    superseded = load_yaml(OVERRIDES).get("superseded", {}) or {}
    rows, ctx_cache, text_cache = [], {}, {}
    for path, rec in load_silver():
        eid = rec["evidence_id"]
        ctx = ctx_cache.setdefault(eid, manifest_context(eid))
        if eid not in text_cache:
            man = json.loads(next((BRONZE_FE / "manifest").rglob(f"{eid}.json")).read_text(encoding="utf-8"))
            text_cache[eid] = artifact_text(ROOT / man["raw_artifact_path"])
        loc_vb, loc_added = location_verbatim(rec.get("location_text"), text_cache[eid])
        try:
            validate_silver_record(rec, contract)
            valid = True
        except Exception:
            valid = False
        tm = temporal(rec, ctx)
        geo = res.resolve(rec.get("location_text") or "", tm["event_date_start"])
        row = {"silver_record_id": rec.get("record_id") or path.stem, "evidence_id": eid,
               "evidence_type": ctx["evidence_type"], "source_name": ctx["source_name"],
               "source_uri": ctx["source_uri"], "event_type": rec["event_type"], **tm,
               "temporal_text": rec.get("temporal_text"), "location_text": rec.get("location_text"), **geo,
               "rainfall_mm": rec.get("rainfall_mm"),
               "rainfall_qualifier": rainfall_qualifier(rec.get("extracted_text") or "", rec.get("rainfall_mm")),
               "rainfall_duration_hours": rec.get("rainfall_duration_hours"),
               "rain_period_text": rec.get("temporal_text") if rec["event_type"] == "rainfall_observation" else None,
               "flood_depth_m": rec.get("flood_depth_m"), "flood_depth_text": rec.get("flood_depth_text"),
               "inundation_duration_hours": rec.get("inundation_duration_hours"),
               "flood_cause": flood_cause(rec, ctx), "confidence": rec.get("confidence"),
               "review_status": rec.get("review_status"), "extraction_method": rec.get("extraction_method"),
               "silver_valid": valid, "extracted_text": rec.get("extracted_text"),
               "location_text_verbatim": loc_vb, "location_context_added": loc_added}
        rid = row["silver_record_id"]
        row["superseded_by"] = (superseded.get(rid) or {}).get("superseded_by")
        row["in_met_history"] = in_met_history(row["event_date_start"], row["event_date_end"])
        row["usable_for"] = [] if row["superseded_by"] else usable_for(row)
        rows.append(row)
    overlay = CATALOG_DIR / "review_overlay.parquet"
    if overlay.exists():  # U1-R: review state lives in the overlay, never in Silver
        rv = pd.read_parquet(overlay, columns=["record_id", "review_effective"]).set_index("record_id")["review_effective"]
        for row in rows:
            row["review_effective"] = None if row["superseded_by"] else rv.get(row["silver_record_id"])
    df = pd.DataFrame(rows, columns=list(SCHEMA))
    for c in ("time_start_utc", "time_end_utc"):
        df[c] = pd.to_datetime(df[c], utc=True)
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    out = CATALOG_DIR / "events.parquet"
    tmp = out.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp)
    tmp.replace(out)
    write_json_atomic(CATALOG_DIR / "schema.json", {
        "table": "event_catalog.events", "version": "1.2", "grain": "one row per Silver flood-evidence record",
        "not_a_label_table": True,
        "wording": "Evidence catalog; absence of evidence is not absence of flooding; no flood probability.",
        "fields": {k: {"type": v[0], "description": v[1]} for k, v in SCHEMA.items()},
        "sources": {"silver": rel(SILVER_FE), "admin_units": rel(ADMIN), "osm_lines": rel(LINES),
                    "osm_areas": rel(WAYS), "city": rel(CITY), "rain_pixels": rel(PIXELS)}})
    st = {"rows": int(len(df)), "silver_records": len(rows),
          "by_level": {str(k): int(v) for k, v in df["location_level"].fillna("null").value_counts().items()},
          "resolved_share_by_level": {str(k): round(float(g["geo_status"].eq("resolved").mean()), 3)
                                      for k, g in df.groupby(df["location_level"].fillna("null"))},
          "resolved_total": int(df["geo_status"].eq("resolved").sum()),
          "by_precision": {k: int(v) for k, v in df["time_precision"].value_counts().items()},
          "by_event_type": {k: int(v) for k, v in df["event_type"].value_counts().items()},
          "in_met_history": int(df["in_met_history"].sum()),
          "usable_for": {u: int(df["usable_for"].map(lambda x, u=u: u in x).sum())
                         for u in ("rain_validation", "risk_validation", "replay", "history_only")},
          "silver_invalid": int((~df["silver_valid"]).sum()),
          "superseded": int(df["superseded_by"].notna().sum()),
          "review_effective": {str(k): int(v) for k, v in df["review_effective"].fillna("null").value_counts().items()},
          "location_verbatim": {"full": int((df["location_context_added"] == False).sum()),  # noqa: E712
                                "context_added": int((df["location_context_added"] == True).sum()),  # noqa: E712
                                "span_null": int(df["location_text_verbatim"].isna().sum())},
          "events_sha256": sha256_file(out)}
    write_json_atomic(REPORTS / "catalog_stats.json", st)
    return st


def main() -> int:
    st = build()
    print(json.dumps({k: v for k, v in st.items() if k != "events_sha256"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
