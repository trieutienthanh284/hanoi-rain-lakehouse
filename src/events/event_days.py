"""U1.5 — event days for replay and the dashboard.

Union of: the 30 wettest UTC days of IMERG Final and of IMERG Late (P2B.5b lists) and every local
date with evidence in the met history. Per local day (Asia/Ho_Chi_Minh): satellite rain summary
over the city pixels (IMERG Final, Late where Final is absent, GSMaP NOW, GSMaP NOW GC; mean and
max of the daily pixel accumulations), evidence counts by kind, replay packages of P4A.
Days without evidence are listed as "chưa thu thập bằng chứng" (never "no flood").

Usage: .venv/Scripts/python.exe -m src.events.event_days
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

from src.events.build_event_catalog import TZ
from src.events.common import CATALOG_DIR, REPORTS, ROOT, RamGuard, rel, write_json_atomic
from src.events.rain_validation import accumulate

UTC = timezone.utc
P2B = ROOT / "reports/p2b"
REPLAY = ROOT / "reports/p4a/replay"


def city_pixels() -> list[int]:
    import pandas as pd
    px = pd.read_parquet(ROOT / "data/reference/rain_pixels_hanoi_city.parquet")
    return [int(p) for p in px.loc[px["is_city_pixel"], "pixel_id"]]


def replay_days() -> dict[date, list[str]]:
    import pyarrow.parquet as pq
    out: dict[date, list[str]] = {}
    for d in sorted(REPLAY.iterdir()):
        f = d / "replay.parquet"
        if not f.exists():
            continue
        t0 = pq.read_table(f, columns=["t0"]).column("t0").to_pandas()
        for day in sorted({t.tz_convert("Asia/Ho_Chi_Minh").date() for t in t0.unique()}):
            out.setdefault(day, []).append(d.name)
    return out


def day_summary(day: date, pixels: list[int]) -> dict:
    a = datetime(day.year, day.month, day.day, tzinfo=TZ).astimezone(UTC)
    wins = [(a, a + timedelta(days=1))]
    res = {}
    for prod in ("imerg_final", "imerg_late", "gsmap", "gsmap_gc"):
        r = accumulate(prod, wins, pixels)
        res[f"{prod}_city_mean_mm"] = None if r["mean"] is None else round(r["mean"], 1)
        res[f"{prod}_city_max_mm"] = None if r["max"] is None else round(r["max"], 1)
        res[f"{prod}_coverage"] = r["coverage"]
    res["imerg_product"] = "final" if (res["imerg_final_coverage"] or 0) >= 0.9 else (
        "late" if (res["imerg_late_coverage"] or 0) >= 0.9 else None)
    return res


def main() -> int:
    import pandas as pd

    cat = pd.read_parquet(CATALOG_DIR / "events.parquet")
    hist = cat[cat["in_met_history"]]
    if "superseded_by" in cat.columns:  # corrected records replace superseded ones (event_catalog_overrides.yaml)
        hist = hist[hist["superseded_by"].isna()]
    wf = pd.read_csv(P2B / "events/wettest_days_imerg_final_city_mean.csv")
    wl = pd.read_csv(P2B / "events_late/wettest_days_imerg_late_city_mean.csv")
    rank_f = {date.fromisoformat(d): int(r) for d, r in zip(wf["date"], wf["rank"])}
    rank_l = {date.fromisoformat(d): int(r) for d, r in zip(wl["date"], wl["rank"])}
    ev_days: dict[date, list[int]] = {}
    for i, r in hist.iterrows():
        d0, d1 = r["event_date_start"], r["event_date_end"] or r["event_date_start"]
        d = d0
        while d <= d1:
            ev_days.setdefault(d, []).append(i)
            d += timedelta(days=1)
    days = sorted(set(rank_f) | set(rank_l) | set(ev_days))
    rp = replay_days()
    pix = city_pixels()
    guard = RamGuard()
    rows = []
    for day in days:
        if not guard.wait_ok():
            rows.append({"date_local": day, "note": "NEEDS_REVIEW: RAM < 0.5 GB for 5 min"})
            continue
        idx = ev_days.get(day, [])
        sub = cat.loc[idx]
        kinds = sub["event_type"].value_counts().to_dict()
        flood = sub[sub["event_type"].isin(["flood_observation", "flood_impact"])]
        row = {"date_local": day, "wettest_rank_imerg_final_utc": rank_f.get(day),
               "wettest_rank_imerg_late_utc": rank_l.get(day), **day_summary(day, pix),
               "n_evidence_records": int(len(sub)),
               "n_flood_observation": int(kinds.get("flood_observation", 0)),
               "n_rainfall_observation": int(kinds.get("rainfall_observation", 0)),
               "n_flood_impact": int(kinds.get("flood_impact", 0)), "n_context": int(kinds.get("context", 0)),
               "n_rain_validation": int(sub["usable_for"].map(lambda u: "rain_validation" in list(u)).sum()),
               "flood_causes": ",".join(sorted(set(flood["flood_cause"].dropna()))),
               "sources": "; ".join(sorted(set(sub["source_name"].dropna()))),
               "replay_packages": ",".join(rp.get(day, [])),
               "evidence_status": "có bằng chứng" if len(sub) else "chưa thu thập bằng chứng"}
        rows.append(row)
        guard.check()
    df = pd.DataFrame(rows)
    for c in ("wettest_rank_imerg_final_utc", "wettest_rank_imerg_late_utc"):
        df[c] = df[c].astype("Int64")
    out = CATALOG_DIR / "event_days.parquet"
    tmp = out.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(out)
    df.to_csv(REPORTS / "event_days.csv", index=False, encoding="utf-8")
    write_events_md(df)
    st = {"days": int(len(df)), "with_evidence": int((df["n_evidence_records"] > 0).sum()),
          "wettest_without_evidence": int(((df["n_evidence_records"] == 0)).sum()),
          "with_replay": int((df["replay_packages"] != "").sum()), "peak_rss_gb": round(guard.peak_rss, 3)}
    write_json_atomic(REPORTS / "event_days_summary.json", st)
    print(json.dumps(st))
    return 0


def _rk(x):
    import pandas as pd
    return "–" if x is None or pd.isna(x) else str(int(x))


def _f(x):
    return "–" if x is None or x != x else f"{x:.1f}".replace(".", ",")


def write_events_md(df) -> None:
    def imerg(r, k):
        p = r["imerg_product"]
        return _f(r.get(f"imerg_{p}_{k}")) if isinstance(p, str) else "–"

    lines = ["# U1.5 — Ngày sự kiện cho replay và dashboard", "",
             "Ngày địa phương (Asia/Ho_Chi_Minh). Mưa vệ tinh = tích lũy ngày địa phương trên 47 pixel 0,1° của "
             "thành phố (trung bình / pixel lớn nhất). IMERG: Final đến 30/9/2025, Late sau đó (cột *sp*). "
             "Hạng = hạng trong 30 ngày UTC mưa nhất của P2B.5b (F = Final, L = Late). "
             "\"Chưa thu thập bằng chứng\" **không** có nghĩa là không ngập. Đây là chỉ báo dựa trên lượng mưa và bằng chứng "
             "văn bản, không phải xác suất ngập.", "",
             "## A. Ngày có bằng chứng", "",
             "| Ngày | Hạng F/L | IMERG TB/max (sp) | GSMaP TB/max | GSMaP GC TB/max | Bản ghi (ngập/mưa/tác động/bối cảnh) | Nguyên nhân ngập | Replay | Nguồn |",
             "|---|---|---|---|---|---|---|---|---|"]
    ev = df[df["n_evidence_records"] > 0].sort_values("date_local")
    for _, r in ev.iterrows():
        lines.append(
            f"| {r['date_local']} | {_rk(r['wettest_rank_imerg_final_utc'])}/{_rk(r['wettest_rank_imerg_late_utc'])} | "
            f"{imerg(r, 'city_mean_mm')}/{imerg(r, 'city_max_mm')} ({r['imerg_product'] or '–'}) | "
            f"{_f(r['gsmap_city_mean_mm'])}/{_f(r['gsmap_city_max_mm'])} | {_f(r['gsmap_gc_city_mean_mm'])}/{_f(r['gsmap_gc_city_max_mm'])} | "
            f"{r['n_evidence_records']} ({r['n_flood_observation']}/{r['n_rainfall_observation']}/{r['n_flood_impact']}/{r['n_context']}) | "
            f"{r['flood_causes'] or '–'} | {r['replay_packages'] or '–'} | {r['sources']} |")
    lines += ["", "## B. Ngày mưa nhất chưa thu thập bằng chứng (ưu tiên thu thập)", "",
              "| Ngày | Hạng F/L | IMERG TB/max (sp) | GSMaP GC TB/max | Replay |", "|---|---|---|---|---|"]
    no = df[df["n_evidence_records"] == 0].copy()
    no["key"] = no.apply(lambda r: -(r[f"imerg_{r['imerg_product']}_city_mean_mm"] or 0)
                         if isinstance(r["imerg_product"], str) else 0, axis=1)
    for _, r in no.sort_values("key").iterrows():
        lines.append(f"| {r['date_local']} | {_rk(r['wettest_rank_imerg_final_utc'])}/{_rk(r['wettest_rank_imerg_late_utc'])} | "
                     f"{imerg(r, 'city_mean_mm')}/{imerg(r, 'city_max_mm')} ({r['imerg_product'] or '–'}) | "
                     f"{_f(r['gsmap_gc_city_mean_mm'])}/{_f(r['gsmap_gc_city_max_mm'])} | {r['replay_packages'] or '–'} |")
    lines += ["", f"Bảng đầy đủ: `{rel(CATALOG_DIR / 'event_days.parquet')}` và `reports/u1/event_days.csv`."]
    (REPORTS / "events.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
