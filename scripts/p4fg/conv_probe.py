"""P4G.1 probe: are convective variables available from the Open-Meteo Previous Runs API (*_previous_day1)?

One HTTP request per invocation (bounded, sequential). Hard cap: 20 requests in total (CLAUDE.md §8 approval
P4F/P4G), >= 2 s between requests, one retry after 60 s on a non-429 error, stop at HTTP 429. Never the Historical
Forecast API (run-availability leakage, TASK decision 7).

  python scripts/p4fg/conv_probe.py --date 2024-06-15 [--vars cape,lifted_index] [--models gfs_global]
  python scripts/p4fg/conv_probe.py --summarize        # writes reports/p4fg/conv_probe.json from the log

Log: data/_scratch/p4fg/conv_probe_log.jsonl (one line per request: url, status, reason, per model x variable the
number of non-null hourly values). The URL has no key (the free API needs none).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.rss_gate import RssGate  # noqa: E402

API = "https://previous-runs-api.open-meteo.com/v1/forecast"
POINT = (21.0285, 105.8542)  # Hoan Kiem, Hanoi
VARS = ["cape", "lifted_index", "convective_inhibition", "total_column_integrated_water_vapour"]
MODELS = ["gfs_global", "ecmwf_ifs025"]
SUFFIX = "_previous_day1"
CAP = 20
MIN_GAP_S = 2.0
LOG = ROOT / "data" / "_scratch" / "p4fg" / "conv_probe_log.jsonl"
OUT = ROOT / "reports" / "p4fg" / "conv_probe.json"
USER_AGENT = "hanoi-flood-lakehouse-research/0.1 (non-commercial thesis prototype)"
TERMS = ("docs/agent_runs/openmeteo_feasibility.md line 13 (recorded in P2A from the open-meteo.com terms and "
         "pricing pages): free for non-commercial use within daily (10,000), hourly (5,000), per-minute (600) and "
         "monthly (300,000) call limits; data licence CC BY 4.0 with attribution to Open-Meteo.com. Not re-fetched in "
         "P4G.1: the P4F/P4G approval covers only the Previous Runs API host.")


def _log_lines() -> list[dict]:
    if not LOG.exists():
        return []
    return [json.loads(x) for x in LOG.read_text(encoding="utf-8").splitlines() if x.strip()]


def request(day: str, variables: list[str], models: list[str]) -> dict:
    import requests
    lines = _log_lines()
    if len(lines) >= CAP:
        raise SystemExit(f"STOP: request cap {CAP} reached")
    if lines:
        gap = time.time() - lines[-1]["epoch"]
        if gap < MIN_GAP_S:
            time.sleep(MIN_GAP_S - gap)
    params = {"latitude": POINT[0], "longitude": POINT[1], "start_date": day, "end_date": day,
              "hourly": ",".join(f"{v}{SUFFIX}" for v in variables), "models": ",".join(models),
              "timezone": "GMT"}
    r = None
    for attempt in (1, 2):
        try:
            r = requests.get(API, params=params, timeout=60, headers={"User-Agent": USER_AGENT})
        except requests.RequestException as e:  # network error
            r = e
        status = getattr(r, "status_code", None)
        if status == 429:
            break
        if status is not None and status < 500:
            break
        if attempt == 1:
            time.sleep(60)
    status = getattr(r, "status_code", None)
    body = {}
    try:
        body = r.json() if status is not None else {}
    except ValueError:
        body = {"_text": (r.text or "")[:300]}
    rec = {"n": len(lines) + 1, "epoch": time.time(), "utc": datetime.now(timezone.utc).isoformat(),
           "url": getattr(r, "url", API), "date": day, "variables": variables, "models": models, "status": status,
           "reason": body.get("reason") if isinstance(body, dict) else None, "counts": {}}
    if status == 200:
        for m in models:
            for v in variables:
                nn, n = _count_models(body, v, m, models)
                rec["counts"][f"{v}|{m}"] = [nn, n]
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    if status == 429:
        raise SystemExit("STOP: HTTP 429")
    return rec


def _count_models(resp: dict, var: str, model: str, models: list[str]) -> tuple[int, int]:
    h = resp.get("hourly") or {}
    for k in (f"{var}{SUFFIX}_{model}",) + ((f"{var}{SUFFIX}",) if len(models) == 1 else ()):
        if k in h:
            vals = h[k]
            return sum(v is not None for v in vals), len(vals)
    return -1, 0


def summarize(first_available: dict | None = None, notes: list[str] | None = None) -> dict:
    lines = _log_lines()
    avail = {}
    for rec in lines:
        for key, (nn, n) in rec.get("counts", {}).items():
            avail.setdefault(key, {})[rec["date"]] = nn
    in_scope = {k: any(nn > 0 for d, nn in v.items() if "2024-05-01" <= d <= "2025-10-31") for k, v in avail.items()}
    out = {"probe": "P4G.1 convective variables, Open-Meteo Previous Runs API (*_previous_day1)",
           "written_utc": datetime.now(timezone.utc).isoformat(), "api": API, "point": POINT,
           "n_requests": len(lines), "cap": CAP, "min_gap_s": MIN_GAP_S,
           "requests": [{k: r[k] for k in ("n", "utc", "url", "date", "status", "reason", "counts")} for r in lines],
           "non_null_hours_by_date": avail, "available_in_2024_05__2025_10": in_scope,
           "first_available_date": first_available or {}, "terms": TERMS,
           "verdict": "positive" if any(in_scope.values()) else "negative", "notes": notes or []}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date")
    ap.add_argument("--vars", default=",".join(VARS))
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--first-available", default=None, help="JSON {var|model: date} found by the bisection")
    ap.add_argument("--note", action="append", default=[])
    a = ap.parse_args(argv)
    gate = RssGate(0.4)
    gate.check("start")
    if a.summarize:
        out = summarize(json.loads(a.first_available) if a.first_available else None, a.note)
        print(json.dumps({"verdict": out["verdict"], "n_requests": out["n_requests"],
                          "available_in_2024_05__2025_10": out["available_in_2024_05__2025_10"]}))
    else:
        rec = request(a.date, a.vars.split(","), a.models.split(","))
        print(json.dumps({k: rec[k] for k in ("n", "date", "status", "reason", "counts")}))
    gate.check("end")
    print(f"PEAK_RSS_GB={gate.peak:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
