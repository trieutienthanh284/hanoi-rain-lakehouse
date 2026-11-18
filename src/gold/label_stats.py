"""P4A.2 verification: row-count identity and base rates per threshold, season, month."""
import sys
from pathlib import Path
import duckdb
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.gold import p4a_common as P  # noqa: E402

con = duckdb.connect(); con.execute("SET memory_limit='1GB'; SET threads=2; SET TimeZone='UTC'")
g = (P.GOLD / "labels" / "season=*" / "month=*" / "part.parquet").as_posix()
dup = con.execute(f"SELECT count(*) - count(DISTINCT (pixel_id, t0, lead_min)) FROM read_parquet('{g}')").fetchone()[0]
br = con.execute(f"""SELECT season, month(t0) AS month, count(*) AS rows,
  round(100*avg(y_0p5),3) AS pct_0p5, round(100*avg(y_2p5),3) AS pct_2p5, round(100*avg(y_7p6),3) AS pct_7p6
  FROM read_parquet('{g}', hive_partitioning=false) GROUP BY ROLLUP(season, month(t0)) ORDER BY 1 NULLS LAST, 2 NULLS LAST""").df()
out = ROOT / "reports" / "p4a" / "labels" / "base_rates.csv"
out.parent.mkdir(parents=True, exist_ok=True)
br.to_csv(out, index=False)
print("duplicates:", dup)
print(br[br["month"].isna()].to_string(index=False))
