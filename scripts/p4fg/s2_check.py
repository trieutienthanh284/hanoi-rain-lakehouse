"""P4FG S2 review (read-only on R2, RSS gate 0.4 GB): nwpconv_backfill Bronze/Silver and the smoke conv incident.

  python scripts/p4fg/s2_check.py [--out reports/p4fg/s2_check.json]

Reads only (list / get): the Bronze request records (HTTP status, retry note, raw sha256 of every response),
every Silver dq.json, two Silver days (issue / available mapping), the STOP_429 marker, the table_v3_conv objects
written by the smoke (LastModified, sha256, meta) and the Gold table_v2 meta rows of the same months. For each conv
unit it checks that every Silver day of its read window existed (complete, dq_pass) before its output was written.
No row-level processing beyond the two Silver days (1 944 rows each); nothing is written to R2.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import rss_gate as RG  # noqa: E402

BACKFILL_RUN = ("nwpconv_backfill", "9900e8d4f8af")
SMOKE_RUN = ("p4fg_smoke", "db0c4d8a01db")
CONV_MONTHS = ((2024, 9), (2024, 10), (2025, 7), (2025, 10))
SAMPLE_DAYS = ("2024-09-01", "2025-07-15")


def listing(st, prefix: str) -> dict[str, dict]:
    """key -> {size, last_modified (ISO UTC)} (paginated list_objects_v2; read only)."""
    out, token = {}, None
    while True:
        kw = {"Bucket": st.bucket, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        r = st.s3.list_objects_v2(**kw)
        for o in r.get("Contents", []):
            out[o["Key"]] = {"size": int(o["Size"]), "last_modified": o["LastModified"].astimezone().isoformat()}
        if not r.get("IsTruncated"):
            return out
        token = r["NextContinuationToken"]


def utc(s: str):
    import pandas as pd
    return pd.Timestamp(s).tz_convert("UTC")


def window_days(y: int, m: int) -> list[date]:
    """Same read window as p4fg.run_conv: first-of-month - 2 days, 36 days, cut at the day after the month end."""
    first = date(y, m, 1)
    days = [first - timedelta(days=2) + timedelta(days=k) for k in range(36)]
    return [d for d in days if d < date(y + (m == 12), m % 12 + 1, 1) + timedelta(days=1)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/p4fg/s2_check.json")
    a = ap.parse_args(argv)
    gate = RG.RssGate()
    import pandas as pd
    from src.models import cloud_train as CT
    from src.models import p4fg as FG
    cfg = FG.load_cfg(smoke=False)
    cs = cfg["conv_store"]
    st = CT.open_run_store("r2", "p4fg")          # reads anywhere; this script never puts
    res: dict = {"note": "read-only S2 check; rainfall nowcast, not a flood prediction"}

    # ---------------------------------------------------------------- Bronze
    br = listing(st, cs["bronze_prefix"])
    blocks = sorted({k.split("block_start=")[1].split("/")[0] for k in br})
    status, retry, raw_ok, raw_bad, bytes_gz = {}, [], 0, [], 0
    for b in blocks:
        bk = f"{cs['bronze_prefix']}block_start={b}/"
        req = json.loads(st.get(bk + "request.json"))
        status[str(req.get("status"))] = status.get(str(req.get("status")), 0) + 1
        if req.get("retry_note"):
            retry.append({"block": b, "retry_note": req.get("retry_note")})
        blob = gzip.decompress(st.get(bk + "response.json.gz"))
        bytes_gz += br[bk + "response.json.gz"]["size"]
        if hashlib.sha256(blob).hexdigest() == req.get("raw_sha256") and len(blob) == req.get("raw_bytes"):
            raw_ok += 1
        else:
            raw_bad.append(b)
        del blob
        gate.check(f"bronze {b}")
    res["bronze"] = {"objects": len(br), "blocks": len(blocks), "first": blocks[:1], "last": blocks[-1:],
                     "http_status": status, "retries": retry, "raw_sha256_ok": raw_ok, "raw_sha256_bad": raw_bad,
                     "response_gz_bytes": bytes_gz,
                     "other_objects": sorted(k for k in br if not k.endswith(("request.json", "response.json.gz")))}

    # ---------------------------------------------------------------- Silver
    sv = listing(st, cs["silver_prefix"])
    days = sorted({k.split("date=")[1].split("/")[0] for k in sv})
    bad, rows, dq_fail, last_mod = [], 0, [], {}
    for d in days:
        sp = f"{cs['silver_prefix']}date={d}/"
        if sp + "part.parquet" not in sv or sp + "dq.json" not in sv:
            bad.append({"day": d, "why": "part or dq missing"})
            continue
        dq = json.loads(st.get(sp + "dq.json"))
        rows += int(dq["rows"])
        last_mod[d] = max(sv[sp + "part.parquet"]["last_modified"], sv[sp + "dq.json"]["last_modified"])
        if not (dq["dq_pass"] and dq["rows"] == 1944 and dq["points"] == 81 and dq["hours"] == 24
                and dq["duplicates"] == 0 and dq["available_before_time"]
                and max(dq["nan_share"].values()) == 0.0):
            dq_fail.append({"day": d, "dq": dq})
        gate.check(f"silver {d}")
    exp = pd.date_range(*cfg["pre_registered"]["convective"]["period"], freq="D").strftime("%Y-%m-%d").tolist()
    res["silver"] = {"objects": len(sv), "days": len(days), "expected_days": len(exp),
                     "missing_days": sorted(set(exp) - set(days)), "extra_days": sorted(set(days) - set(exp)),
                     "rows": rows, "incomplete": bad, "dq_fail": dq_fail,
                     "last_modified_min": min(last_mod.values()) if last_mod else None,
                     "last_modified_max": max(last_mod.values()) if last_mod else None}

    # mapping on two sample days (issue = floor_6h(T) - 24 h, available = issue + 420 min, available <= T)
    mp = []
    for d in SAMPLE_DAYS:
        df = pd.read_parquet(io.BytesIO(st.get(f"{cs['silver_prefix']}date={d}/part.parquet")),
                             columns=["pixel_id", "time_utc", "forecast_issue_time_utc", "available_time_utc"])
        t = df["time_utc"].dt.tz_convert("UTC")
        iss = df["forecast_issue_time_utc"].dt.tz_convert("UTC")
        av = df["available_time_utc"].dt.tz_convert("UTC")
        mp.append({"day": d, "rows": int(len(df)),
                   "issue_rule_ok": bool((iss == t.dt.floor("6h") - pd.Timedelta(hours=24)).all()),
                   "available_rule_ok": bool((av == iss + pd.Timedelta(minutes=420)).all()),
                   "available_le_time": bool((av <= t).all()),
                   "min_lead_h": float(((t - iss).dt.total_seconds() / 3600).min()),
                   "max_lead_h": float(((t - iss).dt.total_seconds() / 3600).max())})
        del df
        gate.check(f"sample {d}")
    res["mapping_samples"] = mp

    # ---------------------------------------------------------------- run prefixes
    bpre = CT.run_prefix(*BACKFILL_RUN)
    bl = listing(st, bpre)
    res["backfill_run"] = {"objects": len(bl), "stop_429": (bpre + "STOP_429.json") in bl,
                           "first_object": min(v["last_modified"] for v in bl.values()),
                           "last_object": max(v["last_modified"] for v in bl.values())}
    res["run_ids_on_r2"] = {j: sorted({k[len(f"lake/v1/runs/{j}/"):].split("/")[0]
                                       for k in listing(st, f"lake/v1/runs/{j}/")})
                            for j in ("p4fg_smoke", "p4fg", "p4fg_final", "nwpconv_backfill")}

    # ---------------------------------------------------------------- conv outputs of the smoke vs final Silver
    ctx_pre = f"{cs['gold_prefix']}gid={FG.PE.gold_id(cfg)}/cid={FG.conv_id(cfg)}/"
    cl = listing(st, cs["gold_prefix"])
    spre = CT.run_prefix(*SMOKE_RUN)
    conv = []
    for y, m in CONV_MONTHS:
        kp = f"{ctx_pre}season={y}/month={m:02d}/part.parquet"
        km = kp.replace("part.parquet", "part.meta.json")
        mk = json.loads(st.get(f"{spre}markers/conv-{y}-{m:02d}.done.json"))
        claims = sorted(k for k in listing(st, f"{spre}claims/conv-{y}-{m:02d}/"))
        meta = json.loads(st.get(km)) if km in cl else {}
        b = st.get(kp) if kp in cl else b""
        sha = hashlib.sha256(b).hexdigest() if b else None
        del b
        gm = json.loads(st.get(f"{FG.PE.GOLD_REMOTE}gid={FG.PE.gold_id(cfg)}/season={y}/month={m:02d}/part.meta.json"))
        wd = [d.isoformat() for d in window_days(y, m)]
        written = cl.get(kp, {}).get("last_modified")
        win_missing = [d for d in wd if d not in last_mod]
        win_after = [d for d in wd if d in last_mod and written and utc(last_mod[d]) >= utc(written)]
        conv.append({"unit": f"conv-{y}-{m:02d}", "parquet_sha256": sha, "parquet_bytes": cl.get(kp, {}).get("size"),
                     "parquet_last_modified": written, "meta_last_modified": cl.get(km, {}).get("last_modified"),
                     "marker_finished_utc": mk["finished_utc"], "marker_rows": mk["metrics"].get("rows"),
                     "marker_reused": bool(mk["metrics"].get("reused")), "meta_rows": meta.get("rows"),
                     "gold_rows": gm.get("rows"), "meta_g2_violations": meta.get("g2_violations"),
                     "meta_points": meta.get("points"), "nan_share_max": max(meta.get("nan_share", {0: None}).values()),
                     "claims": [k.rsplit("/", 1)[1] for k in claims],
                     "window_days": len(wd), "window_missing": win_missing, "window_written_after_output": win_after,
                     "window_dq_fail": [x["day"] for x in res["silver"]["dq_fail"] if x["day"] in wd],
                     "silver_last_in_window": max(last_mod[d] for d in wd if d in last_mod)})
        gate.check(f"conv {y}-{m}")
    res["conv_units"] = conv
    res["table_v3_conv_objects"] = sorted(k[len(cs["gold_prefix"]):] for k in cl)
    res["peak_rss_gb"] = round(gate.peak, 3)
    p = ROOT / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, default=str)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
