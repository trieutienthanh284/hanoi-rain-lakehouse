"""P4A.1 — documentation only: distribution of archive available_time - window_end per season (Silver GSMaP)."""
from __future__ import annotations
import sys
from pathlib import Path
import duckdb
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.gold import p4a_common as P  # noqa: E402


def main() -> int:
    P.require_guard()
    con = duckdb.connect(); con.execute("SET memory_limit='1GB'; SET threads=2")
    glob = (P.SILVER / "gsmap_now" / "date=*" / "part.parquet").as_posix()
    q = f"""
    WITH w AS (SELECT DISTINCT window_end_utc, available_time_utc FROM read_parquet('{glob}')
               WHERE is_city_pixel AND month(window_end_utc) BETWEEN 5 AND 10)
    SELECT year(window_end_utc) AS season, count(*) AS windows,
      round(quantile_cont(epoch(available_time_utc - window_end_utc)/60, 0.10), 2) AS p10_min,
      round(median(epoch(available_time_utc - window_end_utc)/60), 2) AS p50_min,
      round(quantile_cont(epoch(available_time_utc - window_end_utc)/60, 0.90), 2) AS p90_min,
      round(max(epoch(available_time_utc - window_end_utc)/60), 2) AS max_min,
      round(avg(CASE WHEN available_time_utc <= window_end_utc THEN 1 ELSE 0 END), 4) AS share_at_or_before_end,
      round(avg(CASE WHEN available_time_utc <= window_end_utc + INTERVAL 5 MINUTE THEN 1 ELSE 0 END), 4) AS share_le_5min,
      round(avg(CASE WHEN available_time_utc <= window_end_utc + INTERVAL 10 MINUTE THEN 1 ELSE 0 END), 4) AS share_le_10min
    FROM w GROUP BY 1 ORDER BY 1"""
    df = con.execute(q).df()
    out = ROOT / "reports" / "p4a" / "availability" / "archive_latency_by_season.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
