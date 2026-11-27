"""U1-R — automated pre-review of Silver flood-evidence records (user task 2026-11-20).

Deterministic checks C1–C8 per record, an independent blind re-extraction (C9, run outside this
module by a separate agent that sees only the passage, the article date and the preceding sentence),
classification auto_pass / needs_human, a reproducible audit sample, the review overlay and the
regenerated review sheet. Never edits Silver; `review_status` in Silver stays as it is.

Stages:
  prepare  → reports/u1/review/checks_c1_c8.csv, blind_input.jsonl (passages only, no record fields)
  classify → reports/u1/review/auto_review.csv, audit_sample.csv, summary.json,
             data/gold/event_catalog/review_overlay.parquet, reports/u1/U1_SILVER_REVIEW.md

U1-R2 (user 2026-11-20, decisions D143–D150): C2 place after the value in the same clause, shared-value
lists and table rows (unit from the header, city from the caption); C3 'Hà Nội' ↔ 'Hanoi' for city-level
records; C4 paraphrased temporal_text supported by verbatim time phrases → warning; C5 more flood cues and the
forecast test limited to the sentence carrying the value; C8 user-confirmed duplicates → class 'duplicate';
C9 disagreement caused only by extra information found by the blind pass → warning. A check may now return
'warn' (counts as passed, listed as a warning). The audit sample also covers records that moved from
needs_human to auto_pass relative to the U1-R run (reports/u1/review/auto_review_u1r.csv).

Usage: .venv/Scripts/python.exe -m src.events.auto_review --stage prepare|classify
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import unicodedata
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from src.events.common import BRONZE_FE, CATALOG_DIR, REPORTS, ROOT, rel, write_json_atomic

REVIEW_DIR = REPORTS / "review"
OVERLAY = CATALOG_DIR / "review_overlay.parquet"
SHEET = REPORTS / "U1_SILVER_REVIEW.md"
BLIND_IN = REVIEW_DIR / "blind_input.jsonl"
BLIND_OUT = REVIEW_DIR / "blind_extraction.jsonl"
CHECKS_CSV = REVIEW_DIR / "checks_c1_c8.csv"
AUTO_CSV = REVIEW_DIR / "auto_review.csv"
PREV_CSV = REVIEW_DIR / "auto_review_u1r.csv"   # U1-R classification kept for the before/after comparison
AUDIT_SEED = 20261120
REVIEWER_AUTO = "auto_review/v2"
CHECKS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
CHECK_NAMES = {"C1": "extracted_text là chuỗi con của artifact",
               "C2": "giá trị số + địa danh cùng mệnh đề (trước/sau giá trị, danh sách chung giá trị, dòng bảng) = location_text_verbatim",
               "C3": "location_text_verbatim có trong đoạn văn/artifact (cấp thành phố: 'Hà Nội' ↔ 'Hanoi')",
               "C4": "temporal_text có trong artifact (hoặc diễn đạt lại nhưng được cụm thời gian nguyên văn chứng minh); thời gian không mịn hơn văn bản",
               "C5": "event_type khớp từ báo hiệu (dự báo trong câu chứa giá trị/ngập/lượng mưa)",
               "C6": "qualifier (gt/max/approx/eq) khớp từ đứng trước giá trị",
               "C7": "ngày sự kiện khớp ngày bài hoặc ngày ghi trong văn bản (±1 ngày)",
               "C8": "trạng thái trùng (overrides + bỏ trùng U1.4)",
               "C9": "trích xuất mù độc lập đồng ý (thông tin thêm của lượt mù = cảnh báo)"}
PASSING = ("pass", "na", "warn")
CITY_CORES = {"hà nội", "hanoi", "ha noi"}


# ------------------------------------------------------------------------------------ text utils

def nfc_lower(s: str | None) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s or "").replace(" ", " ").lower()).strip()


def fold(s: str | None) -> str:
    s = nfc_lower(s).replace("đ", "d")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


LEAD_UNITS = (r"(?:thành phố|tp\.?|quận|huyện|thị xã|thị trấn|phường|xã|trạm đo|trạm|tuyến đường|đường|phố|"
              r"ngõ|khu vực|khu đô thị|tại|ở|district|ward|commune|station|street|road|town|city|province)")
TAIL_UNITS = r"(?:district|ward|commune|station|street|road|town|city|province)"


def place_core(s: str | None) -> str:
    """Core of a place string: NFC lower case, unit words, parentheses and trailing ', Hà Nội' removed."""
    s = nfc_lower(s)
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"(,\s*(hà nội|hanoi|ha noi))+\s*$", "", s).split(",")[0].strip()
    prev = None
    while prev != s:
        prev = s
        s = re.sub(rf"^{LEAD_UNITS}\s+", "", s).strip()
        s = re.sub(rf"\s+{TAIL_UNITS}$", "", s).strip()
    return s.strip(" -'\"")


def place_parts(verbatim: str | None) -> list[str]:
    """Verbatim location parts: ' | ' pieces plus parenthetical inner names."""
    out = []
    for part in (verbatim or "").split(" | "):
        part = part.strip()
        if not part:
            continue
        out.append(part)
        out += [m.strip() for m in re.findall(r"\(([^)]*)\)", part) if m.strip()]
    return out


class Gazetteer:
    """Known place names (NFC lower and diacritic-free variants), matched with word boundaries."""

    def __init__(self, names: list[str]):
        keys = set()
        for n in names:
            c = place_core(n)
            if len(c) >= 4:
                keys.add(c)
                keys.add(fold(c))
        keys |= {"hà nội", "hanoi", "ha noi"}
        self.keys = sorted(keys, key=lambda k: (-len(k), k))
        alt = "|".join(re.escape(k) for k in self.keys)
        self.rx = re.compile(rf"(?<!\w)(?:{alt})(?!\w)")

    def find(self, text_lower: str) -> list[tuple[int, int, str]]:
        return [(m.start(), m.end(), m.group(0)) for m in self.rx.finditer(text_lower)]


# ----------------------------------------------------------------------------- C2 / C6 numbers

RAIN_UNIT = r"\s*(?:mm\b|milimét\b|milimet\b|mi-li-mét\b|millimet(?:er|re)s?\b|ly\b)"
M_UNIT = r"\s*(?:m\b|mét\b|met\b|metres?\b|meters?\b)"
CM_UNIT = r"\s*(?:cm\b|centimet(?:er|re)s?\b|xen-ti-mét\b|phân\b)"


def number_forms(v: float) -> list[str]:
    """Written forms of a value: '104.2' / '104,2'; integers also '105.0' / '105,0'."""
    s = f"{v:.6f}".rstrip("0").rstrip(".")
    forms = {s, s.replace(".", ",")}
    if "." not in s:
        forms |= {s + ".0", s + ",0"}
    return sorted(forms, key=len, reverse=True)


def value_occurrences(text: str, value: float, kind: str) -> list[tuple[int, int, str]]:
    """(start, end, matched) of `value` with a unit in `text` (kind 'rain' → mm; 'depth' → m, or cm ×100)."""
    t = unicodedata.normalize("NFC", text or "")
    variants = [(value, RAIN_UNIT)] if kind == "rain" else [(value, M_UNIT), (round(value * 100, 4), CM_UNIT)]
    out = []
    for v, unit in variants:
        alt = "|".join(re.escape(f) for f in number_forms(v))
        for m in re.finditer(rf"(?<![\d.,])(?:{alt})(?!\d|[.,]\d)(?={unit})", t, flags=re.I):
            out.append((m.start(), m.end(), m.group(0)))
    return sorted(set(out))


_BREAK = re.compile(r"[;!?\n]|:(?!\d)|[.…](?=\s+(\S)|\s*$)|,(?!\d)")
_LIST_SEP = re.compile(r"^\s*(?:,\s*(?:and|và|&)?|and|và|&)\s*$")


def clause_breaks(text: str) -> list[tuple[int, int]]:
    """Clause boundaries: ; ! ? : (not in 6:00) , (not a decimal comma) and . … before an upper-case word.

    A full stop followed by a lower-case word (6 a.m. in …) is not a boundary (decision D143)."""
    out = []
    for m in _BREAK.finditer(text):
        if m.group(0) in ".…" and m.group(1) and not (m.group(1)[0].isupper() or m.group(1)[0] in "\"“'(‘"):
            continue
        out.append((m.start(), m.end()))
    return out


def clause_bounds(text: str, start: int, end: int | None = None) -> tuple[int, int]:
    end = start if end is None else end
    a, b = 0, len(text)
    for s, e in clause_breaks(text):
        if e <= start:
            a = e
        elif s >= end:
            b = s
            break
    return a, b


TABLE_ROW = re.compile(r"^\s*(?P<date>[\d\-–/.]*/[\d\-–/.]+)\s+(?P<val>\d+(?:[.,]\d+)?)\s+(?P<dur>\d+(?:[.,]\d+)?)\s*"
                       r"(?P<dunit>days?|hours?|ngày|giờ)\s*$", re.I)
TABLE_HEADER_MM = re.compile(r"(?:rain(?:fall)?|lượng mưa)\s*[（(]\s*mm\s*[)）]", re.I)
TABLE_CAPTION = re.compile(r"(?:table|bảng)\s+[\w.\-]+\s*:", re.I)


def table_row_context(extracted: str | None, art: str | None, max_gap: int = 1500) -> dict | None:
    """A JICA-style table row 'date value duration-unit' whose rain unit comes from a header 'Rain (mm)' that
    precedes the row in the artifact (≤ max_gap characters, only rows/header in between); the caption before the
    header names the place (decision D144). None when the passage is not such a row."""
    m = TABLE_ROW.match(extracted or "")
    if not m or not art:
        return None
    pos = art.find((extracted or "").strip())
    if pos < 0:
        return None
    win = art[max(0, pos - 2500):pos]
    hs = list(TABLE_HEADER_MM.finditer(win))
    if not hs or len(win) - hs[-1].end() > max_gap:
        return None
    h = hs[-1]
    caps = list(TABLE_CAPTION.finditer(win[:h.start()]))
    caption = win[caps[-1].start():h.start()] if caps and h.start() - caps[-1].start() <= 300 else ""
    return {"value": float(m.group("val").replace(",", ".")), "val_start": m.start("val"), "unit": "mm",
            "header": h.group(0), "caption": re.sub(r"\s+", " ", caption).strip(),
            "duration": f"{m.group('dur')} {m.group('dunit')}"}


def check_c2(extracted: str, value: float | None, kind: str, verbatim: str | None,
             gaz: Gazetteer, table: dict | None = None,
             location_text: str | None = None) -> tuple[str, str, int | None]:
    """C2: value with unit occurs and the place it belongs to is the verbatim location.

    The place is accepted when it is (U1-R) the nearest gazetteer place before the value, or (U1-R2) a member of
    a list of places sharing that value ("Thanh Xuan, Cau Giay and Nam Tu Liem all received 60 mm"), or the
    nearest place after the value in the same clause when no place precedes the value in that clause
    ("105 mm from midnight to 6 a.m. in Hoang Mai District"). A table row takes its mm unit from the table
    header and the city named in the caption as its place.
    Returns (status, reason, start offset of the matched value occurrence or None).
    """
    if value is None:
        return "na", "no rainfall_mm / flood_depth_m", None
    occ = value_occurrences(extracted, value, kind)
    if not occ and kind == "rain" and table and abs(table["value"] - value) < 0.051:
        city = place_core(location_text) in CITY_CORES or place_core(verbatim) in CITY_CORES
        if city and re.search(r"(?<![a-z])(?:hanoi|ha noi)(?![a-z])", fold(table["caption"])):
            return ("pass", f"table row: {value:g} with unit mm from header '{table['header']}', place = city from "
                    f"caption '{table['caption'][:60]}'", table["val_start"])
        return "fail", f"table row {value:g} (unit from header) but place is not the city named in the caption", table["val_start"]
    if not occ:
        return "fail", f"value {value:g} with a {'mm' if kind == 'rain' else 'm/cm'} unit not found in extracted_text", None
    if not verbatim:
        return "fail", "location_text_verbatim is null", occ[0][0]
    low = nfc_lower(extracted)
    if len(low) != len(unicodedata.normalize("NFC", extracted or "")):
        low = unicodedata.normalize("NFC", extracted or "").lower()
    vb_spans = []
    for part in place_parts(verbatim):
        p = nfc_lower(part)
        for m in re.finditer(re.escape(p), low):
            vb_spans.append((m.start(), m.end()))
    # a parenthetical right after a verbatim location qualifies that place ("Vĩnh Thanh (huyện Đông Anh cũ)
    # 148mm"); place names inside it are not "another place in between" (decision D135)
    paren = []
    for a, b in vb_spans:
        m = re.match(r"\s*\(([^)]*)\)", low[b:])
        if m:
            paren.append((b, b + m.end()))
    places = sorted(p for p in gaz.find(low) if not any(a <= p[0] and p[1] <= b for a, b in paren))
    cores = {place_core(p) for p in place_parts(verbatim)} | {fold(place_core(p)) for p in place_parts(verbatim)}
    orig = unicodedata.normalize("NFC", extracted or "")
    ctext = orig if len(orig) == len(low) else low      # clause boundaries need the original case

    def is_record_place(p) -> bool:
        return any(a <= p[0] and p[1] <= b for a, b in vb_spans) or p[2] in cores or fold(p[2]) in cores

    seen = []
    for s, e, _ in occ:
        ca, cb = clause_bounds(ctext, s, e)
        before = [p for p in places if p[1] <= s]
        if before:
            near = max(before, key=lambda p: (p[1], p[1] - p[0]))
            if is_record_place(near):
                return "pass", f"'{near[2]}' → {value:g}", s
            if near[0] >= ca:       # shared-value list ending in the value's clause (D143)
                i = places.index(near)
                chain = [near]
                while i > 0 and _LIST_SEP.match(low[places[i - 1][1]:places[i][0]]):
                    i -= 1
                    chain.insert(0, places[i])
                hit = next((p for p in chain[:-1] if is_record_place(p)), None)
                if hit:
                    return "pass", f"'{hit[2]}' in a list sharing {value:g} ({', '.join(p[2] for p in chain)})", s
            seen.append(f"nearest place before the value is '{near[2]}', not '{verbatim}'")
        if not [p for p in before if p[0] >= ca]:  # place after the value, same clause (D143)
            after = [p for p in places if p[0] >= e and p[1] <= cb]
            if after:
                nxt = min(after)
                if is_record_place(nxt):
                    return "pass", f"{value:g} → '{nxt[2]}' (place after the value in the same clause)", s
                seen.append(f"nearest place after the value in its clause is '{nxt[2]}', not '{verbatim}'")
            elif not before:
                seen.append("no place name before the value or after it in the same clause")
    return "fail", "; ".join(dict.fromkeys(seen)), occ[0][0]


QUAL_RULES = [(r"(?:len toi|up to)\s*$", "max"), (r"(?:tren|hon|more than)\s*$", "gt"),
              (r"(?:gan|khoang|about)\s*$", "approx")]


def expected_qualifier(extracted: str, start: int) -> str:
    """C6 rule: trên/hơn/more than → gt; lên tới/up to → max; gần/khoảng/about → approx; else eq."""
    window = fold((extracted or "")[max(0, start - 30):start])
    window = re.sub(r"(?:muc|luong mua)\s*$", "", window.rstrip()).rstrip()
    for pat, q in QUAL_RULES:
        if re.search(rf"(?<![a-z]){pat}", window):
            return q
    return "eq"


def check_c6(extracted: str, value: float | None, start: int | None, recorded: str | None) -> tuple[str, str]:
    if value is None:
        return "na", "no rainfall_mm"
    if start is None:
        occ = value_occurrences(extracted, value, "rain")
        if not occ:
            return "fail", "value not found, qualifier not checkable"
        start = occ[0][0]
    exp = expected_qualifier(extracted, start)
    if exp == (recorded or None):
        return "pass", exp
    return "fail", f"text implies '{exp}', catalog has '{recorded}'"


# ------------------------------------------------------------------------------------------- C5

FORECAST_CUES = ["dự báo", "có khả năng", "cảnh báo", "forecast", "expected"]
# U1-R2 (D146): + đọng nước, nước … dâng (≤ 3 words between, same sentence), chìm, ngập úng, inundation, water level
FLOOD_CUES = [r"ngập", r"(?<!\w)úng(?!\w)", r"flooded", r"flooding", r"submerged", r"inundat",
              r"đọng nước", r"(?<!\w)nước(?:\s+[^\s.!?;]+){0,3}?\s+dâng(?!\w)", r"(?<!\w)chìm(?!\w)", r"ngập úng",
              r"inundation", r"water level"]
# agency names are not forecast statements ("Trung tâm Dự báo KTTV quốc gia ghi nhận …"; D146)
AGENCY = re.compile(r"trung tâm dự báo|đài dự báo|dự báo khí tượng thủy văn|dự báo khí tượng thuỷ văn")


def sentence_at(text: str, pos: int) -> tuple[int, int]:
    """Sentence around `pos`: bounded by . ! ? … before an upper-case word, or a line break."""
    a, b = 0, len(text)
    for m in re.finditer(r"[.!?…](?=\s+(\S))|\n", text):
        if m.group(0) != "\n" and not (m.group(1)[0].isupper() or m.group(1)[0] in "\"“'(‘"):
            continue
        if m.end() <= pos:
            a = m.end()
        elif m.start() >= pos:
            b = m.end()
            break
    return a, b


def check_c5(extracted: str, event_type: str, rainfall_mm: float | None, flood_depth_m: float | None = None,
             cue_context: str = "") -> tuple[str, str]:
    """C5. A forecast cue fails a non-context record only when the sentence carrying the record's value is itself
    a forecast; records without a value keep the passage-level test. `cue_context` (table caption + header of a
    table row) is searched for flood cues as well (D146)."""
    low = nfc_lower(extracted)
    v, kind = ((rainfall_mm, "rain") if rainfall_mm is not None else
               (flood_depth_m, "depth") if flood_depth_m is not None else (None, None))
    scope, scope_name = low, "passage"
    if v is not None:
        t = unicodedata.normalize("NFC", extracted or "")
        occ = value_occurrences(t, float(v), kind)
        if occ:
            a, b = sentence_at(t, occ[0][0])
            scope, scope_name = nfc_lower(t[a:b]), "value sentence"
    fc_all = [c for c in FORECAST_CUES if c in AGENCY.sub(" ", low)]
    fc = [c for c in FORECAST_CUES if c in AGENCY.sub(" ", scope)]
    probs = []
    if fc and event_type != "context":
        probs.append(f"forecast cue {fc} in the {scope_name} but event_type {event_type}")
    if event_type == "rainfall_observation" and rainfall_mm is None:
        probs.append("rainfall_observation without rainfall_mm")
    cue_text = low + " " + nfc_lower(cue_context)
    if event_type == "flood_observation" and not any(re.search(p, cue_text) for p in FLOOD_CUES):
        probs.append("flood_observation without a flood cue (ngập/úng/đọng nước/nước dâng/chìm/flooded/submerged/"
                     "inundat/water level)")
    if probs:
        return "fail", "; ".join(probs)
    if fc:
        return "pass", "forecast cue → context"
    if fc_all or "dự báo" in low:
        return "pass", "forecast word only outside the value sentence or in an agency name"
    return "pass", "ok"


# ------------------------------------------------------------------------------------------- C4

PREC_RANK = {"timestamp": 3, "part_of_day": 2, "date": 1, "date_range": 1, "none": 0, None: 0}
DURATION = [r"(?:trong|in the past|in the last|over the past|over the last|past|last)\s+\d+\s*(?:h|giờ|tiếng|hours?)(?!\w)",
            r"\d+\s*(?:h|giờ|tiếng|hours?)\s+(?:qua|vừa qua|liên tục|liền)(?!\w)",
            r"\d+\s*(?:giờ|tiếng)(?=\s+từ)", r"\d+\s*-?\s*hours?(?!\w)"]
CLOCK = [r"(?<![\d/])\d{1,2}\s*(?:h|giờ|g)(?:\s*\d{1,2}(?:\s*phút)?)?(?![\w/])",
         r"(?<![\d/])\d{1,2}[:.]\d{2}(?!\d)", r"(?<![\d/])\d{1,2}(?:[:.]\d{2})?\s*(?:a\.m\.|p\.m\.|am|pm)(?!\w)"]
PART = r"(?<!\w)(?:sáng|chiều|tối|trưa|đêm|khuya|rạng sáng|morning|afternoon|evening|night|overnight|wee hours|dawn|midnight|noon|nửa đêm)(?!\w)"
DAYW = (r"(?<!\w)(?:hôm nay|hôm qua|ngày|nay|qua|today|yesterday|january|february|march|april|may|june|july|august|"
        r"september|october|november|december|sept?\.?|oct\.?)(?!\w)|\d{1,2}\s*[/-]\s*\d{1,2}|\d{4}")


def text_granularity(tt: str | None) -> tuple[int, list[int]]:
    """(rank, clock hours) of a temporal text: 3 clock, 2 part of day, 1 date, 0 none. Durations ignored."""
    t = nfc_lower(tt)
    for p in DURATION:
        t = re.sub(p, " ", t)
    hours = []
    for p in CLOCK:
        for m in re.finditer(p, t):
            h = int(re.match(r"\d+", m.group(0).strip()).group(0))
            if h <= 24:
                if re.search(r"p\.m\.|pm", m.group(0)) and h < 12:
                    h += 12
                hours.append(h)
    if hours:
        return 3, hours
    if re.search(PART, t):
        return 2, []
    if re.search(DAYW, t):
        return 1, []
    return 0, []


def occurs(fragment: str | None, text_norm_lower: str) -> bool:
    from src.events.text_extract import normalise
    f = normalise(fragment or "").lower()
    return bool(f) and f in text_norm_lower


def check_c3(verbatim: str | None, location_text: str | None, location_level: str | None,
             ex_lower: str, art_lower: str) -> tuple[str, str]:
    """C3: every verbatim location part occurs in the passage or the artifact. U1-R2 (D147): a city-level record
    whose place is 'Hà Nội' passes when the artifact names the city as 'Hanoi' / 'Ha Noi' / 'Hà Nội'."""
    parts = [p for p in (verbatim or "").split(" | ") if p.strip()]
    miss = [p for p in parts if not (occurs(p, ex_lower) or occurs(p, art_lower))]
    if parts and not miss:
        return "pass", "found"
    city = location_level == "city" and place_core(location_text) in CITY_CORES
    if city and all(place_core(p) in CITY_CORES for p in miss) and \
            re.search(r"(?<![a-z])(?:hanoi|ha noi)(?![a-z])", fold(art_lower + " " + ex_lower)):
        return "pass", "city-level record: 'Hà Nội' ↔ artifact 'Hanoi'/'Ha Noi'"
    if not parts:
        return "fail", "location_text_verbatim is null (location not found verbatim)"
    return "fail", f"not found: {miss}"


def check_c4(rec: dict, cat: dict, art_lower: str) -> tuple[str, str]:
    tt = rec.get("temporal_text")
    if not tt:
        return "fail", "temporal_text missing (time derived without a stated time)"
    parts = [p.strip() for p in tt.split(" | ") if p.strip()]
    missing = [p for p in parts if not occurs(p, art_lower)]
    prec = cat.get("time_precision") or "none"
    probs = []
    if missing:
        # U1-R2 (D145): a paraphrased temporal_text passes with a warning when time_precision and the observed
        # times are each supported by a verbatim time phrase in extracted_text (no inbox-date allowance here)
        rank, hours = text_granularity(rec.get("extracted_text"))
        if rank == 0:
            return "fail", f"temporal_text part not in artifact: {missing}; extracted_text has no time phrase"
        allowed = rank
    else:
        rank, hours = text_granularity(tt)
        # a date taken from the inbox event date (U1.3 rule, date_source = inbox_event_date) is checked by C7;
        # here it may reach 'date' but never finer (decision D135)
        allowed = max(rank, 1 if cat.get("date_source") == "inbox_event_date" else 0)
    if PREC_RANK.get(prec, 0) > allowed:
        probs.append(f"catalog precision '{prec}' finer than the text (rank {rank})")
    if rec.get("observed_time_start") or rec.get("observed_time_end"):
        if rank < 3:
            probs.append("observed_time set but the text gives no clock time")
        else:
            from zoneinfo import ZoneInfo
            tz = ZoneInfo("Asia/Ho_Chi_Minh")
            obs = [datetime.fromisoformat(x).astimezone(tz).hour for x in
                   (rec.get("observed_time_start"), rec.get("observed_time_end")) if x]
            if not any(h in hours or (h == 0 and 24 in hours) for h in obs):
                probs.append(f"observed hours {obs} not among the text's clock hours {sorted(set(hours))}")
    elif prec == "timestamp" and rank == 3:
        import pandas as pd
        ts = [pd.Timestamp(cat[k]).tz_convert("Asia/Ho_Chi_Minh").hour for k in ("time_start_utc", "time_end_utc")
              if cat.get(k) is not None and not pd.isna(cat.get(k))]
        if ts and not all(h in hours or (h == 0 and 24 in hours) for h in ts):
            probs.append(f"catalog window hours {ts} not among the text's clock hours {sorted(set(hours))}")
    if missing:
        if probs:
            return "fail", f"temporal_text part not in artifact: {missing}; extracted_text does not support it: " + "; ".join(probs)
        return "warn", (f"temporal_text paraphrased ({missing}); precision '{prec}' and observed times supported by "
                        f"extracted_text (rank {rank}, clock hours {sorted(set(hours))})")
    return ("fail", "; ".join(probs)) if probs else ("pass", f"text rank {rank} ≥ precision '{prec}'")


# ------------------------------------------------------------------------------------------- C7

MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                      "september", "october", "november", "december"], 1)}
MON_ABBR = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9,
            "oct": 10, "nov": 11, "dec": 12}


def _d(y, m, d):
    try:
        return date(int(y), int(m), int(d))
    except (ValueError, TypeError):
        return None


def explicit_dates(text: str | None, ref: date | None) -> list[date]:
    """All explicit calendar dates in a text; dates without a year take the article year (or year − 1
    when that would put them more than 1 day after the article)."""
    t = fold(text)
    out = []

    def no_year(d, m):
        if ref is None:
            return None
        x = _d(ref.year, m, d)
        if x and x > ref + timedelta(days=1):
            x = _d(ref.year - 1, m, d)
        return x

    for m in re.finditer(r"(?<!\d)(\d{1,2})\s*[-–]\s*(\d{1,2})[/.](\d{1,2})(?:[/.](\d{4}))?(?!\d)", t):
        for dd in (m.group(1), m.group(2)):
            out.append(_d(m.group(4), m.group(3), dd) if m.group(4) else no_year(dd, m.group(3)))
    for m in re.finditer(r"(?<![\d/])(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})(?!\d)", t):
        out.append(_d(m.group(3), m.group(2), m.group(1)))
    for m in re.finditer(r"(?<![\d/.-])(\d{1,2})/(\d{1,2})(?![\d/])", t):
        out.append(no_year(m.group(1), m.group(2)))
    for m in re.finditer(r"ngay (\d{1,2}) thang (\d{1,2})(?: nam (\d{4}))?", t):
        out.append(_d(m.group(3), m.group(2), m.group(1)) if m.group(3) else no_year(m.group(1), m.group(2)))
    names = "|".join(list(MONTHS) + list(MON_ABBR))
    for m in re.finditer(rf"(?<![a-z])({names})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(\d{{4}}))?(?!\d)", t):
        mo = MONTHS.get(m.group(1)) or MON_ABBR.get(m.group(1))
        out.append(_d(m.group(3), mo, m.group(2)) if m.group(3) else no_year(m.group(2), mo))
    for m in re.finditer(rf"(?<!\d)(\d{{1,2}})(?:st|nd|rd|th)?\s+({names})\.?(?:,?\s+(\d{{4}}))?(?![a-z])", t):
        mo = MONTHS.get(m.group(2)) or MON_ABBR.get(m.group(2))
        out.append(_d(m.group(3), mo, m.group(1)) if m.group(3) else no_year(m.group(1), mo))
    return sorted({d for d in out if d})


def check_c7(cat: dict, rec: dict, published: date | None) -> tuple[str, str]:
    d0, d1 = cat.get("event_date_start"), cat.get("event_date_end")
    if d0 is None or (not isinstance(d0, date)):
        return "fail", "no event date in the catalog"
    d1 = d1 if isinstance(d1, date) else d0
    refs = ([("article", published)] if published else []) + \
        [("text", d) for d in explicit_dates(rec.get("extracted_text"), published)] + \
        [("temporal_text", d) for d in explicit_dates(rec.get("temporal_text"), published)]
    for kind, r in refs:
        if d0 - timedelta(days=1) <= r <= d1 + timedelta(days=1):
            return "pass", f"event {d0}..{d1} ~ {kind} date {r}"
    return "fail", f"event {d0}..{d1} not within ±1 day of article {published} or text dates {[str(r) for _, r in refs if _ != 'article']}"


# ------------------------------------------------------------------------------------------- C9

def blind_value_agree(rec: dict, item: dict) -> bool:
    v, unit = item.get("value"), (item.get("unit") or "").lower()
    try:
        v = None if v is None else float(str(v).replace(",", "."))
    except ValueError:
        return False
    if rec.get("rainfall_mm") is not None:
        return v is not None and unit == "mm" and abs(v - float(rec["rainfall_mm"])) < 0.051
    if rec.get("flood_depth_m") is not None:
        if v is None:
            return False
        m = v if unit == "m" else v / 100 if unit == "cm" else None
        return m is not None and abs(m - float(rec["flood_depth_m"])) < 0.0051
    return v is None or unit in ("", "none")


def blind_place_agree(record_places: list[str], blind_place: str | None) -> bool:
    if not blind_place:
        return False
    b = place_core(blind_place)
    for p in record_places:
        c = place_core(p)
        if c and (c == b or fold(c) == fold(b) or (len(b) >= 4 and (fold(b) in fold(c) or fold(c) in fold(b)))):
            return True
    return False


def period_agree(rec_tt: str | None, blind_period: str | None) -> bool:
    a = set(re.findall(r"\d+", fold(rec_tt)))
    b = set(re.findall(r"\d+", fold(blind_period)))
    if not blind_period:
        # the blind pass sees only the passage and the sentence before it; a period stated elsewhere in the
        # article is not visible to it → not determinable, not a disagreement (C4 checks the period; D140)
        return True
    if not rec_tt:
        return False
    if not a and not b:
        return fold(place_core(rec_tt)) in fold(blind_period) or fold(blind_period) in fold(rec_tt)
    return a <= b or b <= a


C9_TABLE_ROW = "table row, no header visible to blind pass"


def check_c9(rec: dict, verbatim: str | None, items: list[dict] | None, table_row: bool = False) -> tuple[str, str]:
    """`table_row` (U1-R2b, D202): the passage is a table row (numbers, no words); the blind pass sees no header or
    caption, so agreement cannot be judged → 'na' with a warning, never pass or fail."""
    if table_row:
        return "na", C9_TABLE_ROW
    if items is None:
        return "not_run", "no blind extraction output for this passage"
    places = place_parts(verbatim) + [rec.get("location_text") or ""]
    best, best_score = None, -1
    for it in items:
        score = (2 * blind_place_agree(places, it.get("place")) + 2 * blind_value_agree(rec, it)
                 + (it.get("event_type") == rec["event_type"]) + period_agree(rec.get("temporal_text"), it.get("period")))
        if score > best_score:
            best, best_score = it, score
    if best is None:
        return "fail", "blind pass returned no items"
    # each difference is (text, extra): extra = the blind pass found a field the record does not store (D148)
    diffs = []
    rec_place = any(p for p in places if p)
    if not blind_place_agree(places, best.get("place")):
        diffs.append((f"place '{best.get('place')}'", not rec_place and bool(best.get("place"))))
    if not blind_value_agree(rec, best):
        no_rec_value = rec.get("rainfall_mm") is None and rec.get("flood_depth_m") is None
        diffs.append((f"value {best.get('value')} {best.get('unit')}", no_rec_value and best.get("value") is not None))
    if best.get("event_type") != rec["event_type"]:
        diffs.append((f"event_type {best.get('event_type')}", False))
    if not period_agree(rec.get("temporal_text"), best.get("period")):
        diffs.append((f"period '{best.get('period')}'", not rec.get("temporal_text") and bool(best.get("period"))))
    if diffs and all(x for _, x in diffs):
        return "warn", "blind pass found extra information the record does not store: " + "; ".join(d for d, _ in diffs)
    if diffs:
        return "fail", "blind differs: " + "; ".join(d for d, _ in diffs)
    return "pass", "agree" + (" (period not visible to the blind pass)" if rec.get("temporal_text") and not best.get("period") else "")


# ----------------------------------------------------------------------------------- data access

def load_inputs():
    import pandas as pd
    from src.events.build_event_catalog import manifest_context
    from src.events.text_extract import artifact_text, normalise
    from src.events.u1_followup import silver_records

    sil = silver_records()
    cat = pd.read_parquet(CATALOG_DIR / "events.parquet").set_index("silver_record_id")
    texts, ctxs = {}, {}
    for rid, rec in sil.items():
        eid = rec["evidence_id"]
        if eid not in texts:
            man = json.loads(next((BRONZE_FE / "manifest").rglob(f"{eid}.json")).read_text(encoding="utf-8"))
            texts[eid] = artifact_text(ROOT / man["raw_artifact_path"])
            ctxs[eid] = manifest_context(eid)
    norm = {eid: normalise(t.replace("\n", " ")) for eid, t in texts.items()}
    return sil, cat, norm, ctxs


def gazetteer(sil: dict, cat) -> Gazetteer:
    import pyogrio
    adm = pyogrio.read_dataframe(ROOT / "data/reference/hanoi_city_admin_osm_20261012.gpkg",
                                 columns=["name", "name_en", "name_vi", "old_name", "level"], read_geometry=False)
    names = []
    for c in ("name", "name_en", "name_vi", "old_name"):
        names += [str(x) for x in adm[c].dropna()]
    for v in cat["location_text_verbatim"].dropna():
        names += place_parts(v)
    names += [r.get("location_text") or "" for r in sil.values()]
    return Gazetteer(names)


def overrides():
    import yaml
    ov = yaml.safe_load((ROOT / "configs/event_catalog_overrides.yaml").read_text(encoding="utf-8")) or {}
    return ov


def primary_set() -> set[str]:
    import pandas as pd
    from src.events.rain_validation import PRIMARY_BASES
    dd = pd.read_csv(REPORTS / "followup" / "rain_validation_dedup.csv")
    return set(dd.loc[dd["kept"] & dd["period_basis"].isin(PRIMARY_BASES), "silver_record_id"])


def sentence_before(art_norm: str, passage: str) -> str:
    from src.events.text_extract import normalise
    p = normalise(passage)
    i = art_norm.find(p)
    if i <= 0:
        return ""
    head = art_norm[max(0, i - 600):i].rstrip()
    sents = [s for s in re.split(r"(?<=[.!?…])\s+", head) if s.strip()]
    return sents[-1][-400:] if sents else ""


# --------------------------------------------------------------------------------------- stages

def _v(x):
    import pandas as pd
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except (TypeError, ValueError):
        pass
    return x


def stage_prepare() -> dict:
    import pandas as pd
    from src.events.text_extract import is_verbatim
    sil, cat, norm, ctxs = load_inputs()
    gaz = gazetteer(sil, cat)
    ov = overrides()
    superseded = ov.get("superseded") or {}
    ov_dup = {k: v["duplicate_of"] for k, v in (ov.get("rain_validation_duplicates") or {}).items()}
    dd = pd.read_csv(REPORTS / "followup" / "rain_validation_dedup.csv").set_index("silver_record_id")
    rows, passages = [], {}
    for rid in sorted(sil):
        if rid in superseded:
            continue
        rec, c = sil[rid], {k: _v(v) for k, v in cat.loc[rid].to_dict().items()}
        eid = rec["evidence_id"]
        art, ex = norm[eid], rec.get("extracted_text") or ""
        art_lower = art.lower()
        vb = c.get("location_text_verbatim")
        r = {"record_id": rid, "evidence_id": eid, "source_name": c.get("source_name"),
             "event_type": rec["event_type"], "location_text": rec.get("location_text"),
             "location_text_verbatim": vb, "temporal_text": rec.get("temporal_text"),
             "rainfall_mm": rec.get("rainfall_mm"), "flood_depth_m": rec.get("flood_depth_m"),
             "rainfall_qualifier": c.get("rainfall_qualifier"), "time_precision": c.get("time_precision"),
             "event_date_start": c.get("event_date_start"), "event_date_end": c.get("event_date_end"),
             "published_date": ctxs[eid].get("published_date"), "confidence": rec.get("confidence"),
             "geo_status": c.get("geo_status"), "location_context_added": c.get("location_context_added"),
             "location_level": c.get("location_level")}
        tab = table_row_context(ex, art)
        r["table_row"] = tab is not None  # C9 'na' in classify (D202)
        # C1
        if not ex:
            r["C1"], r["C1_reason"] = "fail", "extracted_text missing"
        elif not art:
            r["C1"], r["C1_reason"] = "fail", "artifact has no text layer"
        else:
            ok = is_verbatim(ex, art)
            r["C1"], r["C1_reason"] = ("pass", "substring") if ok else ("fail", "extracted_text not a substring of the artifact")
        # C2 (+ offset for C6)
        lt = rec.get("location_text")
        if rec.get("rainfall_mm") is not None:
            st, why, off = check_c2(ex, float(rec["rainfall_mm"]), "rain", vb, gaz, tab, lt)
        elif rec.get("flood_depth_m") is not None:
            st, why, off = check_c2(ex, float(rec["flood_depth_m"]), "depth", vb, gaz, tab, lt)
        else:
            st, why, off = check_c2(ex, None, "rain", vb, gaz)
        r["C2"], r["C2_reason"] = st, why
        # C3
        r["C3"], r["C3_reason"] = check_c3(vb, lt, c.get("location_level"), nfc_lower(ex), art_lower)
        # C4–C7
        r["C4"], r["C4_reason"] = check_c4(rec, c, art_lower)
        r["C5"], r["C5_reason"] = check_c5(ex, rec["event_type"], rec.get("rainfall_mm"), rec.get("flood_depth_m"),
                                           f"{tab['caption']} {tab['header']}" if tab else "")
        r["C6"], r["C6_reason"] = check_c6(ex, rec.get("rainfall_mm"), off if rec.get("rainfall_mm") is not None else None,
                                           c.get("rainfall_qualifier"))
        r["C7"], r["C7_reason"] = check_c7(c, rec, ctxs[eid].get("published_date"))
        # C8
        if rid in ov_dup:
            r["C8"], r["C8_reason"], r["duplicate_status"] = "pass", f"duplicate_of {ov_dup[rid]} (user override)", "duplicate_override"
        elif rid in dd.index and not bool(dd.loc[rid, "kept"]):
            r["C8"], r["C8_reason"], r["duplicate_status"] = ("fail", "dropped by U1.4 gauge+period-basis+window dedup, "
                                                                      "not confirmed in overrides",
                                                              "duplicate_auto")
        else:
            r["C8"], r["C8_reason"], r["duplicate_status"] = "pass", "not a duplicate", "unique"
        # blind passage (text only)
        key = (ex, str(ctxs[eid].get("published_date")))
        if ex and key not in passages:
            passages[key] = {"passage_id": f"p{len(passages) + 1:03d}", "article_date": str(ctxs[eid].get("published_date")),
                             "sentence_before": sentence_before(art, ex), "extracted_text": ex}
        r["passage_id"] = passages[key]["passage_id"] if ex else None
        rows.append(r)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(CHECKS_CSV, index=False, encoding="utf-8")
    with open(BLIND_IN, "w", encoding="utf-8") as f:
        for p in passages.values():
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    df = pd.DataFrame(rows)
    return {"records": len(rows), "passages": len(passages), "gazetteer_keys": len(gaz.keys),
            "fail_counts": {k: int((df[k] == "fail").sum()) for k in CHECKS[:8]},
            "warn_counts": {k: int((df[k] == "warn").sum()) for k in CHECKS[:8]}}


def warnings_for(r: dict, ov: dict) -> list[str]:
    w = []
    if r.get("location_context_added") is True:
        w.append("location_text có từ thêm ngoài nguyên văn")
    if r.get("geo_status") != "resolved":
        w.append("địa danh chưa phân giải")
    if (r.get("confidence") or "") == "low":
        w.append("confidence low")
    if r.get("rainfall_qualifier") in ("gt", "max", "approx"):
        w.append(f"qualifier {r['rainfall_qualifier']}")
    pb = {rid: b for b, blk in (ov.get("period_basis") or {}).items() for rid in blk.get("records", [])}
    if r["record_id"] in pb:
        w.append(f"period_basis {pb[r['record_id']]}")
    if r.get("duplicate_status") == "duplicate_override":
        w.append("trùng (overrides)")
    w += [f"{k}: {r.get(k + '_reason')}" for k in CHECKS if r.get(k) in ("warn", "na")]
    return w


def moved_quota(n_moved: int) -> int:
    """Audit quota for records that moved needs_human → auto_pass: 10 %, at least 5 (all when fewer than 5)."""
    return min(n_moved, max(5, math.ceil(0.10 * n_moved)))


def draw_audit(auto: list[dict], primary: set[str], seed: int = AUDIT_SEED, moved: list[str] | None = None) -> list[str]:
    """10 % of auto_pass (min 10), ≥ 1 per source, plus every primary-set auto_pass record with a warning, plus
    (U1-R2, D149) records that moved needs_human → auto_pass until moved_quota() of them are in the sample."""
    rng = random.Random(seed)
    ids = sorted(r["record_id"] for r in auto)
    n = min(len(ids), max(10, math.ceil(0.10 * len(ids))))
    by_src: dict[str, list[str]] = {}
    for r in sorted(auto, key=lambda r: r["record_id"]):
        by_src.setdefault(str(r["source_name"]), []).append(r["record_id"])
    pick = [rng.choice(by_src[s]) for s in sorted(by_src)]
    rest = [i for i in ids if i not in set(pick)]
    if len(pick) < n:
        pick += rng.sample(rest, n - len(pick))
    pick += [r["record_id"] for r in sorted(auto, key=lambda r: r["record_id"])
             if r["record_id"] in primary and r["warnings"] and r["record_id"] not in set(pick)]
    mv = sorted(set(moved or []) & set(ids))
    need = moved_quota(len(mv)) - len(set(mv) & set(pick))
    if need > 0:
        pick += rng.sample([i for i in mv if i not in set(pick)], need)
    return pick


def before_counts() -> dict:
    """Class and fail counts of the U1-R run (PREV_CSV) for the before/after table."""
    import pandas as pd
    if not PREV_CSV.exists():
        return {}
    p = pd.read_csv(PREV_CSV)
    return {"class": p["class"].value_counts().to_dict(),
            "fail_by_check": {k: int((p[k] == "fail").sum()) for k in CHECKS}}


def load_prev_class() -> dict[str, str]:
    import pandas as pd
    if not PREV_CSV.exists():
        return {}
    p = pd.read_csv(PREV_CSV, usecols=["record_id", "class"])
    return dict(zip(p["record_id"], p["class"]))


def load_blind() -> dict[str, list[dict]]:
    out = {}
    if not BLIND_OUT.exists():
        return out
    for line in BLIND_OUT.read_text(encoding="utf-8").splitlines():
        if line.strip():
            j = json.loads(line)
            out[j["passage_id"]] = j.get("items") or []
    return out


def stage_classify() -> dict:
    import pandas as pd
    df = pd.read_csv(CHECKS_CSV, dtype={"passage_id": str}, keep_default_na=False, na_values=[""])
    blind = load_blind()
    ov = overrides()
    primary = primary_set()
    prev = load_prev_class()
    run_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    rows = []
    for r in df.to_dict("records"):
        r = {k: _v(v) for k, v in r.items()}
        rec = {"event_type": r["event_type"], "rainfall_mm": r.get("rainfall_mm"), "flood_depth_m": r.get("flood_depth_m"),
               "temporal_text": r.get("temporal_text"), "location_text": r.get("location_text")}
        items = blind.get(r["passage_id"]) if r.get("passage_id") else None
        r["C9"], r["C9_reason"] = check_c9(rec, r.get("location_text_verbatim"), items,
                                           table_row=str(r.get("table_row")).lower() == "true")
        fails = [k for k in CHECKS if r[k] in ("fail", "not_run")]
        r["failing_checks"] = ",".join(fails)
        r["warn_checks"] = ",".join(k for k in CHECKS if r[k] == "warn")
        # user-confirmed duplicates (overrides duplicate_of) are not sent to human review (U1-R2, D150)
        r["class"] = ("duplicate" if r.get("duplicate_status") == "duplicate_override" else
                      "auto_pass" if not fails else "needs_human")
        r["prev_class"] = prev.get(r["record_id"])
        r["primary_set"] = r["record_id"] in primary
        r["warnings"] = "; ".join(warnings_for(r, ov))
        r["reviewer"] = REVIEWER_AUTO
        r["run_utc"] = run_utc
        rows.append(r)
    auto = [r for r in rows if r["class"] == "auto_pass"]
    moved = [r["record_id"] for r in auto if r["prev_class"] == "needs_human"]
    audit = set(draw_audit(auto, primary, moved=moved))
    for r in rows:
        r["in_audit_sample"] = r["record_id"] in audit
        r["moved_to_auto_pass"] = r["record_id"] in set(moved)
    cols = (["record_id", "source_name", "event_type", "primary_set", "class", "prev_class", "moved_to_auto_pass",
             "failing_checks", "warn_checks", "in_audit_sample", "warnings"]
            + [x for k in CHECKS for x in (k, f"{k}_reason")]
            + ["duplicate_status", "passage_id", "evidence_id", "location_text", "location_text_verbatim", "temporal_text",
               "rainfall_mm", "flood_depth_m", "rainfall_qualifier", "time_precision", "event_date_start", "event_date_end",
               "published_date", "reviewer", "run_utc"])
    out = pd.DataFrame(rows)[cols]
    out.to_csv(AUTO_CSV, index=False, encoding="utf-8")
    out[out["in_audit_sample"]][["record_id", "source_name", "event_type", "primary_set", "moved_to_auto_pass",
                                 "warnings"]].to_csv(REVIEW_DIR / "audit_sample.csv", index=False, encoding="utf-8")
    from src.events.apply_review import build_overlay, parse_sheet, write_overlay
    states = parse_sheet(SHEET.read_text(encoding="utf-8")) if SHEET.exists() else {}
    sheet = write_sheet(out, states)
    ovl = build_overlay(out, parse_sheet(SHEET.read_text(encoding="utf-8")))
    write_overlay(ovl)
    summary = {"run_utc": run_utc, "records": len(out), "superseded_excluded": len(ov.get("superseded") or {}),
               "passages": int(out["passage_id"].nunique()), "blind_passages_returned": len(blind),
               "class": out["class"].value_counts().to_dict(),
               "class_primary_set": out[out["primary_set"]]["class"].value_counts().to_dict(),
               "fail_by_check": {k: int((out[k] == "fail").sum()) for k in CHECKS},
               "warn_by_check": {k: int((out[k] == "warn").sum()) for k in CHECKS},
               "not_run_by_check": {k: int((out[k] == "not_run").sum()) for k in CHECKS},
               "na_by_check": {k: int((out[k] == "na").sum()) for k in CHECKS},
               "only_failing_check": {k: int((out["failing_checks"] == k).sum()) for k in CHECKS},
               "before_u1r": before_counts(),
               "class_transitions": {f"{a}->{b}": int(n) for (a, b), n in
                                     out.groupby(["prev_class", "class"], dropna=False).size().items() if a != b},
               "moved_to_auto_pass": len(moved), "moved_quota": moved_quota(len(moved)),
               "moved_in_audit": int((out["in_audit_sample"] & out["moved_to_auto_pass"]).sum()),
               "reviewer": REVIEWER_AUTO,
               "audit_sample": len(audit), "audit_sources": int(out[out["in_audit_sample"]]["source_name"].nunique()),
               "auto_pass_sources": int(out[out["class"] == "auto_pass"]["source_name"].nunique()),
               "audit_primary_with_warning": int((out["in_audit_sample"] & out["primary_set"] & (out["warnings"] != "")).sum()),
               "sheet": sheet, "seed": AUDIT_SEED}
    write_json_atomic(REVIEW_DIR / "summary.json", summary)
    return summary


# --------------------------------------------------------------------------------- review sheet

CHECKBOX = "Duyệt (đúng nguyên văn, đúng trường) — Ghi chú:"


def _sort_key(r: dict) -> tuple:
    return (not r["primary_set"], str(r.get("event_date_start") or ""), str(r.get("source_name")), r["record_id"])


def write_sheet(out, states: dict[str, tuple[bool, str]]) -> dict:
    """Regenerate reports/u1/U1_SILVER_REVIEW.md; checkbox states and notes are carried over by record id."""
    from src.events.u1_followup import FIELD_ORDER, _md, silver_records
    sil = silver_records()
    rows = out.to_dict("records")
    need = sorted([r for r in rows if r["class"] == "needs_human"], key=_sort_key)
    audit = sorted([r for r in rows if r["class"] == "auto_pass" and r["in_audit_sample"]], key=_sort_key)
    other = sorted([r for r in rows if r["class"] == "auto_pass" and not r["in_audit_sample"]], key=_sort_key)
    dups = sorted([r for r in rows if r["class"] == "duplicate"], key=_sort_key)

    def box(rid: str) -> str:
        checked, note = states.get(rid, (False, ""))
        return f"[{'x' if checked else ' '}]", note

    def block(i: str, r: dict, show_fail: bool) -> list[str]:
        rid, rec = r["record_id"], sil[r["record_id"]]
        tag = ("[TẬP CHÍNH] " if r["primary_set"] else "") + ("[CHUYỂN LỚP] " if r.get("moved_to_auto_pass") else "")
        b, note = box(rid)
        L = [f"### {i}. {tag}{_md(r['source_name'])} — {_md(rec.get('location_text') or '')} — `{rid}`", "",
             f"- {b} {CHECKBOX} {note}".rstrip() + ("" if note else " "), ""]
        if show_fail:
            fl = [k for k in CHECKS if r[k] in ("fail", "not_run")]
            L.append("**Kiểm tra không đạt:** " + "; ".join(f"**{k}** ({CHECK_NAMES[k]}): {_md(r[k + '_reason'])}" for k in fl))
            L.append("")
        if r.get("warnings"):
            L += [f"**Cảnh báo:** {_md(r['warnings'])}", ""]
        et = rec.get("extracted_text")
        L += [f"> {_md(et)}" if et else "> **(không có `extracted_text`)**", ""]
        L += ["| Trường | Giá trị |", "|---|---|"]
        L += [f"| `{k}` | {_md(rec.get(k))} |" for k in FIELD_ORDER if rec.get(k) not in (None, "")]
        L.append(f"| `evidence_id` | {rec['evidence_id']} |")
        L.append(f"| Catalog | precision {r.get('time_precision')}; ngày {r.get('event_date_start')}–{r.get('event_date_end')}; "
                 f"qualifier {r.get('rainfall_qualifier')}; trùng: {r.get('duplicate_status')} |")
        L.append(f"| Trích xuất mù (C9) | {_md(r['C9_reason'])} |")
        L.append("")
        return L

    def compact(r: dict) -> str:
        rec = sil[r["record_id"]]
        b, note = box(r["record_id"])
        tag = " [TẬP CHÍNH]" if r["primary_set"] else ""
        if r["class"] == "duplicate":
            tag += f" [TRÙNG: {_md(r['C8_reason'])}]"
        val = (f"{rec['rainfall_mm']:g} mm" if rec.get("rainfall_mm") is not None else
               f"{rec['flood_depth_m']:g} m" if rec.get("flood_depth_m") is not None else "—")
        return (f"- {b} `{r['record_id']}`{tag} {_md(r['source_name'])} — {_md(rec.get('location_text') or '')} — "
                f"{rec['event_type']} — {val} — {_md(rec.get('temporal_text') or '')} — Ghi chú: {note}").rstrip() + ("" if note else " ")

    n_prim = lambda xs: sum(1 for r in xs if r["primary_set"])  # noqa: E731
    n_moved = sum(1 for r in rows if r.get("moved_to_auto_pass"))
    L = ["# U1 — Phiếu duyệt bản ghi Silver (U1-R2: tiền duyệt tự động, quy tắc tinh chỉnh)", "",
         f"Tạo {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC bởi `python -m src.events.auto_review --stage classify`. "
         f"{len(rows)} bản ghi Silver (không gồm bản ghi bị thay thế). Agent **không** thay đổi `review_status` trong Silver; "
         "kết quả duyệt nằm trong `data/gold/event_catalog/review_overlay.parquet`.", "",
         "Cách duyệt: đánh `[x]` → `human_reviewed`; ghi chú bắt đầu bằng `SAI` (vd. `SAI: giá trị là 124 mm`) → "
         "`human_rejected` kèm ghi chú (có tích hay không). Sau khi sửa phiếu, chạy:", "",
         "```", ".venv/Scripts/python.exe -m src.events.apply_review", "```", "",
         f"- Phần 1 — **Cần người duyệt**: {len(need)} bản ghi (tập chính {n_prim(need)}); mỗi mục ghi các kiểm tra không đạt.",
         f"- Phần 2 — **Mẫu kiểm tra**: {len(audit)} bản ghi auto_pass (seed {AUDIT_SEED}; 10 %, tối thiểu 10, ≥ 1 mỗi nguồn, "
         "cộng mọi bản ghi tập chính có cảnh báo, cộng ≥ 10 % (tối thiểu 5) trong "
         f"{n_moved} bản ghi chuyển từ needs_human sang auto_pass nhờ U1-R2, đánh dấu `[CHUYỂN LỚP]`). "
         "Mỗi bản ghi `SAI` ở đây được tính vào tỷ lệ lỗi kiểm tra.",
         f"- Phần 3 — {len(other)} bản ghi auto_pass khác và {len(dups)} bản ghi trùng đã xác nhận trong overrides "
         "(`[TRÙNG]`, review_effective = duplicate, không cần duyệt) — danh sách gọn.", "",
         "Kiểm tra: " + "; ".join(f"{k} = {v}" for k, v in CHECK_NAMES.items()) + ".", ""]
    L += [f"## 1. Cần người duyệt ({len(need)})", ""]
    for i, r in enumerate(need, 1):
        L += block(f"1.{i}", r, True)
    L += [f"## 2. Mẫu kiểm tra ({len(audit)})", ""]
    for i, r in enumerate(audit, 1):
        L += block(f"2.{i}", r, False)
    L += [f"## 3. Các bản ghi auto_pass khác ({len(other)}) và bản ghi trùng ({len(dups)})", ""]
    L += [compact(r) for r in other] + [""]
    if dups:
        L += ["Bản ghi trùng (configs/event_catalog_overrides.yaml, `duplicate_of`):", ""]
        L += [compact(r) for r in dups] + [""]
    tmp = SHEET.with_suffix(".md.tmp")
    tmp.write_text("\n".join(L), encoding="utf-8")
    tmp.replace(SHEET)
    carried = sum(1 for r in rows if r["record_id"] in states and (states[r["record_id"]][0] or states[r["record_id"]][1]))
    return {"path": rel(SHEET), "needs_human": len(need), "audit": len(audit), "other_auto_pass": len(other),
            "duplicate": len(dups), "checkboxes": len(rows), "states_carried_over": carried}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["prepare", "classify"], required=True)
    a = ap.parse_args()
    res = stage_prepare() if a.stage == "prepare" else stage_classify()
    print(json.dumps(res, ensure_ascii=False, default=str))
    import psutil
    peak = getattr(psutil.Process().memory_info(), "peak_wset", None)
    print(f"PEAK_RSS_GB={peak / 2**30:.3f}" if peak else "PEAK_RSS_GB=n/a")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
