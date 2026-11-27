"""U1.1b — fetch the user-listed evidence documents (TASK_U1 §U1.1b; user instruction 2026-11-20).

Rules:
  * only the URLs in reports/u1/u1b/urls.csv, in their order; reserve rows only while fewer than
    MAX_DOCS documents have been fetched; at most MAX_DOCS fetched documents per run;
  * skip a URL whose source_uri (or same article id on the same host) is already in inbox.csv /
    inbox_agent.csv → "already in inbox";
  * before the first fetch on a host: GET https://<host>/robots.txt once. Fetch only if robots.txt
    returned HTTP 200, parsed, and the path is allowed for User-agent * AND for our UA token.
    Any other robots result (4xx, 5xx, network error, redirect elsewhere) → "unreachable/unclear",
    not fetched (stricter than U0: when in doubt, not allowed);
  * one GET per document, no redirects followed except same-host (each robots-checked), no link
    following, >= 10 s between requests to the same host, timeout 30 s, no retry on HTTP status,
    one retry after 60 s on network error;
  * 401/403/429, cookie wall, paywall, login, captcha → not saved, listed as candidate;
  * files saved with mode 'xb' (never overwrite) to data/_inbox/evidence/agent/<file_name>.
The inbox_agent.csv rows are appended in a separate step (`--append`) after the content check
recorded in reports/u1/u1b/content_check.yaml.

Usage: .venv/Scripts/python.exe -m src.events.agent_fetch [--fetch] [--append]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import robotparser
from urllib.parse import urljoin, urlsplit

from src.events.common import INBOX_AGENT_CSV, INBOX_CSV, INBOX_DIR, REPORTS, SCRATCH, rel

UA = "hanoi-rain-lakehouse-research/0.1 (student thesis; manual curation)"
UA_TOKEN = "hanoi-rain-lakehouse-research"
MIN_GAP_S, TIMEOUT_S, RETRY_WAIT_S, MAX_REDIRECTS = 10.0, 30, 60, 5
MAX_DOCS = 20
U1B = REPORTS / "u1b"
URLS = U1B / "urls.csv"
LOG = U1B / "fetch_log.csv"
STATE = SCRATCH / "u1b_state.json"
AGENT_DIR = INBOX_DIR / "agent"
LOG_COLS = ["order", "file_name", "url", "host", "robots_check", "robots_detail", "http_status", "final_url",
            "bytes", "sha256", "status", "reason", "retrieved_utc"]
FINAL = {"FETCHED", "NOT_FETCHED", "ALREADY_IN_INBOX", "FAILED"}
BLOCK = ("captcha", "cf-challenge", "just a moment", "attention required", "please enable cookies",
         "đăng nhập để", "paywall", "subscribe to continue", "sign in to continue", "log in to continue")
KEYWORDS = ("ngập", "mưa", "flood", "rain")
INBOX_COLS = ["file_name", "evidence_type", "source_name", "source_uri", "published_date", "event_date",
              "location_hint", "notes", "collected_by", "retrieved_utc", "robots_check"]


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_state() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {"robots": {}, "last": {}}


def save_state(s: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")


def read_log() -> dict[str, dict]:
    if not LOG.exists():
        return {}
    with open(LOG, encoding="utf-8", newline="") as f:
        return {r["url"]: r for r in csv.DictReader(f)}


def append_log(row: dict) -> None:
    new = not LOG.exists()
    with open(LOG, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LOG_COLS})


def article_key(url: str) -> tuple[str, str]:
    p = urlsplit(url)
    host = (p.hostname or "").removeprefix("www.")
    m = re.search(r"(\d{6,})(?:\.html?|\.htm)?$", p.path)
    return host, (m.group(1) if m else p.path.rstrip("/"))


def known_uris() -> dict[tuple[str, str], str]:
    out = {}
    for path in (INBOX_CSV, INBOX_AGENT_CSV):
        if path.exists():
            with open(path, encoding="utf-8-sig", newline="") as f:
                for r in csv.DictReader(f):
                    if r.get("source_uri"):
                        out[article_key(r["source_uri"].strip())] = f"{path.name}: {r['source_uri'].strip()}"
    return out


class Fetcher:
    def __init__(self, state: dict):
        import requests
        self.req = requests
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA})
        self.state = state

    def get(self, url: str):
        host = urlsplit(url).hostname or ""
        last = self.state["last"].get(host)
        if last and (wait := MIN_GAP_S - (time.time() - last)) > 0:
            time.sleep(wait)
        for attempt in (1, 2):
            try:
                r = self.s.get(url, timeout=TIMEOUT_S, allow_redirects=False)
                self.state["last"][host] = time.time()
                save_state(self.state)
                return r
            except self.req.RequestException:
                self.state["last"][host] = time.time()
                save_state(self.state)
                if attempt == 2:
                    raise
                time.sleep(RETRY_WAIT_S)

    def robots(self, url: str) -> tuple[str, str]:
        """('allowed' | 'disallowed' | 'unreachable', detail)."""
        p = urlsplit(url)
        key = f"{p.scheme}://{p.netloc}"
        if key not in self.state["robots"]:
            try:
                r = self.get(key + "/robots.txt")
                st, txt = r.status_code, (r.text if r.status_code == 200 else "")
                loc = r.headers.get("Location", "") if r.is_redirect else ""
            except self.req.RequestException as e:
                st, txt, loc = -1, "", f"network error {type(e).__name__}"
            self.state["robots"][key] = {"status": st, "text": txt, "location": loc, "retrieved_utc": now()}
            save_state(self.state)
            (SCRATCH / "u1b_robots").mkdir(parents=True, exist_ok=True)
            (SCRATCH / "u1b_robots" / f"{p.netloc}.txt").write_text(txt, encoding="utf-8")
        e = self.state["robots"][key]
        if e["status"] != 200 or not e["text"].strip():
            return "unreachable", f"robots.txt HTTP {e['status']} {e.get('location', '')}".strip()
        rp = robotparser.RobotFileParser()
        rp.parse(e["text"].splitlines())
        star, ua = rp.can_fetch("*", url), rp.can_fetch(UA_TOKEN, url)
        return ("allowed" if star and ua else "disallowed"), f"robots.txt HTTP 200; *={star}; ua={ua}"


def visible_text(body: bytes) -> str:
    from src.events.text_extract import decode_bytes, normalise
    t = decode_bytes(body)
    t = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", t)
    return normalise(re.sub(r"<[^>]+>", " ", t))


def fetch() -> dict:
    with open(URLS, encoding="utf-8", newline="") as f:
        urls = list(csv.DictReader(f))
    log = read_log()
    known = known_uris()
    state = load_state()
    F = Fetcher(state)
    fetched = sum(1 for r in log.values() if r["status"] == "FETCHED")
    for u in urls:
        if u["url"] in log and log[u["url"]]["status"] in FINAL:
            continue
        if u["reserve"] == "1" and fetched >= MAX_DOCS:
            append_log({"order": u["order"], "file_name": u["file_name"], "url": u["url"],
                        "host": urlsplit(u["url"]).hostname, "retrieved_utc": now(), "status": "NOT_FETCHED",
                        "reason": f"reserve not needed: {MAX_DOCS} documents already fetched in this run"})
            continue
        row = {"order": u["order"], "file_name": u["file_name"], "url": u["url"],
               "host": urlsplit(u["url"]).hostname, "retrieved_utc": now()}
        if fetched >= MAX_DOCS:
            append_log({**row, "status": "NOT_FETCHED", "reason": f"run limit {MAX_DOCS} documents reached"})
            continue
        k = article_key(u["url"])
        if k in known:
            append_log({**row, "status": "ALREADY_IN_INBOX", "reason": f"already in inbox ({known[k]})"})
            continue
        if u["reserve"] == "1" and not any(r["status"] in ("ALREADY_IN_INBOX", "NOT_FETCHED", "FAILED")
                                           for r in read_log().values() if r["order"] and int(r["order"]) <= 20):
            append_log({**row, "status": "NOT_FETCHED", "reason": "reserve not needed"})
            continue
        rc, detail = F.robots(u["url"])
        row.update(robots_check=rc, robots_detail=detail)
        if rc != "allowed":
            append_log({**row, "status": "NOT_FETCHED", "reason": f"robots {rc} ({detail})"})
            print(u["order"], u["file_name"], "NOT_FETCHED", rc, detail, flush=True)
            continue
        try:
            r, cur, hops = F.get(u["url"]), u["url"], 0
            while r.is_redirect and hops < MAX_REDIRECTS:
                nxt = urljoin(cur, r.headers.get("Location", ""))
                if (urlsplit(nxt).hostname or "").removeprefix("www.") != (urlsplit(cur).hostname or "").removeprefix("www."):
                    raise RuntimeError(f"cross-host redirect to {nxt} not followed")
                rc2, d2 = F.robots(nxt)
                if rc2 != "allowed":
                    raise RuntimeError(f"redirect target robots {rc2}: {nxt}")
                cur, hops = nxt, hops + 1
                r = F.get(nxt)
        except Exception as e:  # network error after retry, refused redirect
            append_log({**row, "status": "FAILED", "reason": f"{type(e).__name__}: {e}"[:300]})
            print(u["order"], u["file_name"], "FAILED", e, flush=True)
            continue
        body = r.content
        row.update(http_status=r.status_code, final_url=cur, bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
        if r.status_code in (401, 403, 429):
            append_log({**row, "status": "NOT_FETCHED", "reason": f"HTTP {r.status_code} (access control)"})
            continue
        if r.status_code != 200:
            append_log({**row, "status": "FAILED", "reason": f"HTTP {r.status_code}"})
            continue
        text = visible_text(body)
        low = text.lower()
        blocked = [m for m in BLOCK if m in low]
        has_kw = any(k in low for k in KEYWORDS)
        if blocked and (not has_kw or len(text) < 3000):
            append_log({**row, "status": "NOT_FETCHED", "reason": f"access-control page ({', '.join(blocked)}); not saved"})
            continue
        if not has_kw:
            append_log({**row, "status": "FAILED", "reason": f"no rain/flood keyword in {len(text)} chars; not saved"})
            continue
        AGENT_DIR.mkdir(parents=True, exist_ok=True)
        with open(AGENT_DIR / u["file_name"], "xb") as fh:  # never overwrite
            fh.write(body)
        fetched += 1
        append_log({**row, "status": "FETCHED", "reason": f"HTML {len(text)} visible chars" + ("; same-host redirect" if cur != u["url"] else "")})
        print(u["order"], u["file_name"], "FETCHED", r.status_code, len(body), flush=True)
    return {"fetched": fetched, "log": rel(LOG)}


def append_inbox() -> dict:
    """Append inbox_agent.csv rows for FETCHED documents, with the content-check notes."""
    import yaml
    chk = yaml.safe_load((U1B / "content_check.yaml").read_text(encoding="utf-8"))
    with open(URLS, encoding="utf-8", newline="") as f:
        urls = {r["url"]: r for r in csv.DictReader(f)}
    existing = set()
    if INBOX_AGENT_CSV.exists():
        with open(INBOX_AGENT_CSV, encoding="utf-8", newline="") as f:
            existing = {r["file_name"] for r in csv.DictReader(f)}
    new = not INBOX_AGENT_CSV.exists()
    n = 0
    with open(INBOX_AGENT_CSV, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INBOX_COLS)
        if new:
            w.writeheader()
        for url, lg in read_log().items():
            if lg["status"] != "FETCHED" or lg["file_name"] in existing:
                continue
            u, c = urls[url], chk[lg["file_name"]]
            w.writerow({"file_name": lg["file_name"], "evidence_type": "article", "source_name": u["source_name"],
                        "source_uri": url, "published_date": c["published_date"], "event_date": u["event_date"],
                        "location_hint": "Hà Nội", "notes": c["notes"], "collected_by": "agent",
                        "retrieved_utc": lg["retrieved_utc"],
                        "robots_check": f"{lg['robots_check']} ({lg['robots_detail']})"})
            n += 1
    return {"appended": n, "inbox_agent": rel(INBOX_AGENT_CSV)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--append", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        print(json.dumps(fetch(), ensure_ascii=False))
    if a.append:
        print(json.dumps(append_inbox(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
