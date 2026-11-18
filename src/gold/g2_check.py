"""Gate G2 on a whole Gold table: re-read every partition, run frame-mode checks, compare with labels.

Usage: python src/gold/g2_check.py --table table_A
Writes reports/p4a/g2/<table>_partitions.csv and <table>_nan_share.csv; exit 4 on any violation.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.gold import p4a_common as P  # noqa: E402
from src.quality.temporal_checks import FeatureGroup, FrameConfig, check_frame_table  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--table", default="table_A")
    a = ap.parse_args(argv)
    P.require_guard()
    schema = json.loads((P.GOLD / a.table / "schema.json").read_text(encoding="utf-8"))
    fcols = [f["name"] for f in schema["features"]]
    by = {}
    for f in schema["features"]:
        by.setdefault((f["group"], f["kind"]), []).append(f["name"])
    groups = [FeatureGroup(g, k, tuple(c)) for (g, k), c in by.items()]
    fc = FrameConfig.from_yaml(P.CFG_PATH)
    rows, nan = [], []
    for part in sorted((P.GOLD / a.table).glob("season=*/month=*/part.parquet")):
        df = pd.read_parquet(part)
        lab = pd.read_parquet(P.GOLD / "labels" / part.parent.parent.name / part.parent.name / "part.parquet",
                              columns=["pixel_id", "t0", "lead_min"])
        v = check_frame_table(df, groups, fcols, fc)
        dup = int(df.duplicated(["pixel_id", "t0", "lead_min"]).sum())
        keys_equal = len(df) == len(lab) and df[["pixel_id", "t0", "lead_min"]].sort_values(
            ["pixel_id", "t0", "lead_min"]).reset_index(drop=True).equals(
            lab.sort_values(["pixel_id", "t0", "lead_min"]).reset_index(drop=True))
        rows.append({"partition": P.rel(part), "rows": len(df), "label_rows": len(lab), "keys_equal_labels": keys_equal,
                     "duplicates": dup, "violations": len(v), "detail": "; ".join(map(str, v))})
        nan.append({"partition": f"{part.parent.parent.name}/{part.parent.name}"} |
                   {c: float(df[c].isna().mean()) for c in fcols})
    r = pd.DataFrame(rows)
    out = ROOT / "reports" / "p4a" / "g2"
    out.mkdir(parents=True, exist_ok=True)
    r.to_csv(out / f"{a.table}_partitions.csv", index=False)
    pd.DataFrame(nan).to_csv(out / f"{a.table}_nan_share.csv", index=False)
    ok = (r["violations"] == 0).all() and r["keys_equal_labels"].all() and (r["duplicates"] == 0).all()
    print(f"G2 {a.table}: partitions={len(r)} rows={int(r['rows'].sum()):,} violations={int(r['violations'].sum())} "
          f"keys_equal_labels={bool(r['keys_equal_labels'].all())} duplicates={int(r['duplicates'].sum())} -> "
          f"{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
