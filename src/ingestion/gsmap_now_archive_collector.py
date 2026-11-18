"""GSMaP NOW archive collector (/now/netcdf/YYYY/MM/DD), one UTC date per command.

Reuses config, credential loading, validation and promotion from
``src/ingestion/gsmap_now_collector.py`` (not modified). Credentials are read
from the environment / .env at runtime and never printed.

  --date D --list-only            list rain files, record SIZE + MDTM, describe flag files
  --date D --download --promote   keep originals in Bronze (atomic promotion), then crop
  --date D --crop-only            download to data/_scratch, validate, crop, hash, remove scratch

Idempotent: valid Bronze files are skipped; listing and crop are rewritten.
Downloads are strictly sequential; a failed transfer is retried once after 60 s,
a second failure stops the run (exit code 2).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from datetime import datetime, timedelta, timezone
from ftplib import FTP, error_perm
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion import gsmap_now_collector as gc  # noqa: E402
from src.lake import regional_crop as rc  # noqa: E402

SOURCE = "gsmap_now"
LISTING_DIR = rc.REGIONAL_ROOT / SOURCE / "remote_listing"
SCRATCH = ROOT / "data" / "_scratch" / SOURCE
FLAG_RE = re.compile(r"^gsmap_now_flag\.(?P<date>\d{8})\.(?P<time>\d{4})\.nc$")
RETRY_WAIT_S = 60
LATENCY_MEDIAN_LIMIT_H = 3.0
FALLBACK_LATENCY = timedelta(minutes=30)


class SourceStopped(RuntimeError):
    pass


def archive_dir(config: dict, date: str) -> str:
    y, m, d = date.split("-")
    return config["access"]["archive_directory_template"].format(YYYY=y, MM=m, DD=d)


def connect(config: dict) -> FTP:
    host, user, password = gc.load_ftp_credentials(config)
    ftp = FTP()
    ftp.connect(host=host, port=21, timeout=60)
    ftp.login(user=user, passwd=password)
    return ftp


def close(ftp: FTP) -> None:
    try:
        ftp.quit()
    except Exception:
        ftp.close()


def parse_mdtm(resp: str) -> datetime:
    m = re.search(r"(\d{14})", resp)
    if not m:
        raise ValueError(f"unexpected MDTM response: {resp!r}")
    return datetime.strptime(m[1], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


def with_retry(fn, what: str):
    """Run fn(); on failure wait 60 s and retry once; then raise SourceStopped."""
    try:
        return fn()
    except Exception as exc1:
        print(json.dumps({"event": "retry", "what": what, "error": repr(exc1)[:300]}), flush=True)
        time.sleep(RETRY_WAIT_S)
        try:
            return fn()
        except Exception as exc2:
            raise SourceStopped(f"{what}: {exc2!r}") from exc2


def list_date(config: dict, date: str) -> dict:
    """List rain files with SIZE/MDTM; flag files described by name only."""
    def _do():
        ftp = connect(config)
        try:
            ftp.cwd(archive_dir(config, date))
            names = sorted(ftp.nlst())
            rain, flag = [], []
            for n in names:
                if gc.filename_regex(config).fullmatch(n):
                    try:
                        start = gc.parse_event_start_time(n, config)
                    except ValueError:
                        continue
                    if start.strftime("%Y-%m-%d") != date:
                        continue
                    ftp.voidcmd("TYPE I")
                    size = ftp.size(n)
                    mdtm = parse_mdtm(ftp.sendcmd(f"MDTM {n}"))
                    end = start + timedelta(hours=1)
                    rain.append({"name": n, "size": size, "mdtm_utc": rc.iso(mdtm),
                                 "window_start_utc": rc.iso(start), "window_end_utc": rc.iso(end),
                                 "latency_min": round((mdtm - end).total_seconds() / 60, 2)})
                elif FLAG_RE.fullmatch(n):
                    flag.append(n)
            return rain, flag
        finally:
            close(ftp)

    rain, flag = with_retry(_do, f"list {date}")
    lat = [r["latency_min"] for r in rain]
    med = statistics.median(lat) if lat else None
    suspect = med is None or med > LATENCY_MEDIAN_LIMIT_H * 60
    summary = {
        "n_rain": len(rain), "n_flag": len(flag), "expected_windows": 48,
        "latency_min_median": med, "latency_min_p10": _pct(lat, 10), "latency_min_p90": _pct(lat, 90),
        "latency_min_min": min(lat) if lat else None, "latency_min_max": max(lat) if lat else None,
        "available_time_rule": ("window_end + 30 min (fallback)" if suspect
                                else "window_end + observed latency (MDTM - window_end)"),
        "archive_reprocessed_suspect": bool(suspect),
    }
    flag_desc = {"count": len(flag), "pattern": "gsmap_now_flag.YYYYMMDD.HHNN.nc",
                 "first": flag[0] if flag else None, "last": flag[-1] if flag else None,
                 "note": "described from names only; not downloaded"}
    out = {"source": SOURCE, "date": date, "remote_dir": archive_dir(config, date),
           "listed_utc": rc.iso(datetime.now(timezone.utc)), "files": rain,
           "flag_files": flag_desc, "latency_summary": summary}
    LISTING_DIR.mkdir(parents=True, exist_ok=True)
    p = LISTING_DIR / f"date={date}.json"
    tmp = p.with_suffix(".json.part")
    tmp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    os.replace(tmp, p)
    return out


def _pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 2)


def retrieve(config: dict, date: str, name: str, size: int, dest: Path) -> None:
    """Download one file sequentially; TYPE I right before SIZE and RETR."""
    def _do():
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.unlink(missing_ok=True)
        ftp = connect(config)
        try:
            ftp.cwd(archive_dir(config, date))
            ftp.voidcmd("TYPE I")
            rsize = ftp.size(name)
            if rsize != size:
                raise RuntimeError(f"remote size changed {size} -> {rsize}")
            ftp.voidcmd("TYPE I")
            with dest.open("wb") as fh:
                ftp.retrbinary(f"RETR {name}", fh.write, blocksize=1 << 20)
        except Exception:
            dest.unlink(missing_ok=True)
            raise
        finally:
            close(ftp)
        if dest.stat().st_size != size:
            got = dest.stat().st_size
            dest.unlink(missing_ok=True)
            raise RuntimeError(f"size mismatch remote={size} local={got}")
    with_retry(_do, f"download {name}")


def download_promote(config: dict, date: str, listing: dict) -> dict:
    done, skipped, failed = 0, 0, []
    for e in listing["files"]:
        name = e["name"]
        try:
            if gc.get_valid_existing_bronze(name, config) is not None:
                skipped += 1
                continue
        except Exception as exc:  # existing but invalid: never overwrite Bronze, report it
            failed.append({"name": name, "error": f"existing Bronze invalid: {exc!r}"})
            continue
        # Download into our own scratch area (deletable), then move into the
        # collector's staging path and use its atomic promotion. Nothing under
        # data/bronze is ever deleted by this script.
        scratch = SCRATCH / f"date={date}" / name
        retrieve(config, date, name, e["size"], scratch)
        staging = gc.staging_path_for_filename(name, config)
        if staging.exists():
            failed.append({"name": name, "error": "staging file already exists; left untouched"})
            scratch.unlink(missing_ok=True)
            continue
        staging.parent.mkdir(parents=True, exist_ok=True)
        os.replace(scratch, staging)
        try:
            gc.promote_staging_to_bronze(name, config)
            done += 1
        except Exception as exc:
            failed.append({"name": name, "error": repr(exc)})
            if staging.exists():  # move our own failed download back to scratch and drop it
                os.replace(staging, scratch)
                scratch.unlink(missing_ok=True)
    return {"downloaded": done, "skipped_existing": skipped, "failed": failed}


def crop_only(config: dict, date: str, listing: dict) -> dict:
    """Download to scratch, validate, crop, hash; remove own scratch files."""
    folder = SCRATCH / f"date={date}"
    files, hashes, origins, failed = [], {}, {}, []
    try:
        for e in listing["files"]:
            dest = folder / e["name"]
            retrieve(config, date, e["name"], e["size"], dest)
            try:
                gc.validate_gsmap_now_netcdf(dest, e["name"], config)
            except Exception as exc:
                failed.append({"name": e["name"], "error": repr(exc)})
                dest.unlink(missing_ok=True)
                continue
            files.append(dest)
            hashes[e["name"]] = (dest.stat().st_size, rc.sha256_file(dest))
            origins[e["name"]] = f"jaxa-ftp:{listing['remote_dir']}/{e['name']}"
        res = rc.build_day(SOURCE, date, files=files, original_hashes=hashes, origin_paths=origins) if files else None
    finally:
        for p in folder.glob("gsmap_now_*.nc"):
            p.unlink(missing_ok=True)
        try:
            folder.rmdir()
        except OSError:
            pass
    return {"cropped": len(files), "failed": failed, "crop": res}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--promote", action="store_true")
    ap.add_argument("--crop-only", action="store_true")
    a = ap.parse_args(argv)
    if a.download != a.promote:
        ap.error("--download and --promote go together")
    if sum([a.list_only, a.download, a.crop_only]) != 1:
        ap.error("choose exactly one of --list-only, --download --promote, --crop-only")
    datetime.strptime(a.date, "%Y-%m-%d")

    config = gc.load_config()
    t0 = time.time()
    try:
        listing = list_date(config, a.date)
        report = {"date": a.date, "listed_rain": len(listing["files"]),
                  "latency": listing["latency_summary"]}
        if a.download:
            report["download"] = download_promote(config, a.date, listing)
            report["crop"] = rc.build_day(SOURCE, a.date)
        elif a.crop_only:
            report["crop_only"] = crop_only(config, a.date, listing)
    except SourceStopped as exc:
        print(json.dumps({"date": a.date, "status": "SOURCE_STOPPED", "error": str(exc)[:500]}))
        return 2
    except error_perm as exc:
        print(json.dumps({"date": a.date, "status": "FTP_PERMISSION_ERROR", "error": str(exc)[:300]}))
        return 2
    report["seconds"] = round(time.time() - t0, 1)
    n_cropped = (report.get("crop") or {}).get("windows") or \
        ((report.get("crop_only") or {}).get("crop") or {}).get("windows") or 0
    if not a.list_only:
        report["ingested_share"] = round(n_cropped / max(len(listing["files"]), 1), 4)
    print(json.dumps(report, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
