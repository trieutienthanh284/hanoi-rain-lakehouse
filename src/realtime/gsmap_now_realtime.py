"""GSMaP NOW realtime ingestion: /now/latest -> validate -> Hanoi crop -> R2.

  python -m src.realtime.gsmap_now_realtime [--store r2|local:DIR] [--limit N]

* Lists ``/now/latest`` (rolling ~24 h, 48 hourly-mean rain windows every 30 min)
  and processes every rain file whose window id is not yet in R2 (catch-up: a
  missed or delayed schedule loses nothing as long as it is < ~24 h).
* Newest windows first; one FTP connection per run; downloads strictly sequential;
  a failed transfer is retried once after 60 s on a fresh connection, a second
  failure stops the source (exit 2). Files already published stay published.
* Validation and crop reuse ``gsmap_now_collector.validate_gsmap_now_netcdf`` and
  ``regional_crop.read_crop`` / ``write_day_file`` (same HDF5 layout as the
  backfill day files, one window per object).
* Time semantics (ADD-003): filename time = window start; window = 1 h;
  available_time = max(MDTM, window_end). ``ingestion_time_utc`` = moment the
  file was fully retrieved on this runner (the time from which the record is
  legally usable by a realtime system).

Objects: lake/v1/realtime/gsmap_now/date=D/gsmap_now_YYYYMMDDTHHMM.h5 + manifest JSON.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from ftplib import FTP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion import gsmap_now_collector as gc  # noqa: E402
from src.lake import regional_crop as rc  # noqa: E402
from src.realtime import common as C  # noqa: E402

SOURCE = "gsmap_now"
RETRY_WAIT_S = 60


class SourceStopped(RuntimeError):
    pass


def window_id(start: datetime) -> str:
    return f"gsmap_now_{start:%Y%m%dT%H%M}"


def parse_mdtm(resp: str) -> datetime:
    m = re.search(r"(\d{14})", resp)
    if not m:
        raise ValueError(f"unexpected MDTM response: {resp!r}")
    return datetime.strptime(m[1], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


def connect(gcfg: dict) -> FTP:
    host, user, password = gc.load_ftp_credentials(gcfg)
    ftp = FTP()
    ftp.connect(host=host, port=21, timeout=60)
    ftp.login(user=user, passwd=password)
    return ftp


def close(ftp) -> None:
    try:
        ftp.quit()
    except Exception:
        try:
            ftp.close()
        except Exception:
            pass


def rain_windows(names: list[str], gcfg: dict) -> list[dict]:
    """Rain files of a listing -> [{name, start, end, id}] (flag/.dat files ignored)."""
    out = []
    rx = gc.filename_regex(gcfg)
    for n in names:
        if not rx.fullmatch(n):
            continue
        try:
            start = gc.parse_event_start_time(n, gcfg)
        except ValueError:
            continue
        out.append({"name": n, "start": start, "end": start + timedelta(hours=1), "id": window_id(start)})
    return out


def select_pending(windows: list[dict], existing: set[str], limit: int) -> list[dict]:
    """Catch-up selection: every listed window not yet in R2, newest first, at most ``limit``."""
    todo = [w for w in windows if w["id"] not in existing]
    todo.sort(key=lambda w: w["start"], reverse=True)
    return todo[:limit] if limit else todo


class Session:
    """One FTP connection reused for the run; reconnects on retry."""

    def __init__(self, gcfg: dict, directory: str, factory=None):
        self.gcfg, self.dir = gcfg, directory
        self.factory = factory or (lambda: connect(gcfg))
        self.ftp = None

    def open(self):
        if self.ftp is None:
            self.ftp = self.factory()
            self.ftp.cwd(self.dir)
        return self.ftp

    def reset(self):
        if self.ftp is not None:
            close(self.ftp)
        self.ftp = None

    def call(self, fn, what: str):
        try:
            return fn(self.open())
        except Exception as e1:
            C.log("retry", what=what, error=repr(e1)[:300], wait_s=RETRY_WAIT_S)
            self.reset()
            time.sleep(RETRY_WAIT_S)
            try:
                return fn(self.open())
            except Exception as e2:
                self.reset()
                raise SourceStopped(f"{what}: {e2!r}"[:500]) from e2

    def close(self):
        self.reset()


def retrieve(sess: Session, name: str, dest: Path) -> tuple[int, datetime, datetime, datetime]:
    """SIZE + MDTM + RETR; returns (size, mdtm, retrieval_started, ingestion_time)."""
    def _do(ftp):
        ftp.voidcmd("TYPE I")
        size = ftp.size(name)
        mdtm = parse_mdtm(ftp.sendcmd(f"MDTM {name}"))
        started = C.utcnow()
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        with tmp.open("wb") as fh:
            ftp.retrbinary(f"RETR {name}", fh.write, blocksize=1 << 20)
        got = tmp.stat().st_size
        if got != size:
            raise RuntimeError(f"size mismatch remote={size} local={got}")
        os.replace(tmp, dest)
        return size, mdtm, started, C.utcnow()
    return sess.call(_do, f"retrieve {name}")


def crop_window(path: Path, dom: dict, work: Path) -> Path:
    """Crop one validated original into a one-window file with the backfill layout."""
    w = rc.gsmap_window(path)
    crop = rc.read_crop(w, dom)
    date = w.start.strftime("%Y-%m-%d")
    with C.redirected(rc, REGIONAL_ROOT=work / "regional_tmp"):
        out, _, _ = rc.write_day_file(SOURCE, date, dom, [w], [crop])
    final = work / "crops" / f"date={date}" / f"{window_id(w.start)}.h5"
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(out, final)
    return final


def build_manifest(w: dict, remote_dir: str, size: int, mdtm: datetime, started: datetime,
                   ingested: datetime, original_sha: str, validation: dict, crop_sha: str) -> dict:
    end = w["end"]
    available = max(mdtm, end)
    return {
        "source": SOURCE, "id": w["id"], "product_version": "GSMaP_NOW_V8",
        "original_name": w["name"], "remote_dir": remote_dir,
        "original_size": size, "original_sha256": original_sha,
        "remote_mdtm_utc": C.iso(mdtm),
        "window_start_utc": C.iso(w["start"]), "window_end_utc": C.iso(end),
        "filename_time_semantics": "start_time (ADD-003)",
        "retrieval_started_utc": C.iso(started), "ingestion_time_utc": C.iso(ingested),
        "available_time_utc": C.iso(available),
        "available_time_rule": "max(MDTM, window_end) (ADD-003); realtime use only after ingestion_time_utc",
        "mdtm_minus_window_end_min": round((mdtm - end).total_seconds() / 60, 2),
        "ingestion_minus_window_end_min": round((ingested - end).total_seconds() / 60, 2),
        "validation": {"ok": True, **{k: (C.iso(v) if isinstance(v, datetime) else v)
                                       for k, v in validation.items() if k != "root_datasets"}},
        "crop_sha256": crop_sha,
        "data_time_utc": C.iso(end),
        "run": C.run_context(),
    }


def run(store: C.Store, cfg: dict, limit: int | None = None, ftp_factory=None,
        validator=None, dom: dict | None = None, newest: int = 0) -> dict:
    gcfg = gc.load_config()
    directory = cfg["gsmap_now"]["latest_directory"]
    limit = cfg["gsmap_now"]["max_files_per_run"] if limit is None else limit
    validator = validator or (lambda p, n: gc.validate_gsmap_now_netcdf(p, n, gcfg))
    dom = dom or rc.load_domain()
    work = C.workdir(cfg, SOURCE)
    t0 = time.time()
    sess = Session(gcfg, directory, ftp_factory)
    res = {"source": SOURCE, "listed": 0, "pending": 0, "published": 0, "skipped_existing": 0,
           "failed": [], "latency": []}
    try:
        names = sess.call(lambda ftp: sorted(ftp.nlst()), f"list {directory}")
        wins = rain_windows(names, gcfg)
        res["listed"] = len(wins)
        if newest:                       # test aid: restrict candidates to the newest N listed windows
            wins = sorted(wins, key=lambda w: w["start"])[-newest:]
            res["candidates_restricted_to_newest"] = newest
        dates = {w["start"].strftime("%Y-%m-%d") for w in wins}
        have = C.existing_ids(store, SOURCE, dates)
        todo = select_pending(wins, have, limit)
        res["pending"] = len(todo)
        res["skipped_existing"] = len(wins) - len([w for w in wins if w["id"] not in have])
        for w in todo:
            date = w["start"].strftime("%Y-%m-%d")
            raw = work / "raw" / f"date={date}" / w["name"]
            size, mdtm, started, ingested = retrieve(sess, w["name"], raw)
            try:
                v = validator(raw, w["name"])
            except Exception as exc:
                res["failed"].append({"id": w["id"], "error": f"validation: {exc!r}"[:300]})
                C.log("invalid", id=w["id"], error=repr(exc)[:300])
                continue
            crop = crop_window(raw, dom, work)
            data = crop.read_bytes()
            man = build_manifest(w, directory, size, mdtm, started, ingested, rc.sha256_file(raw), v,
                                 C.sha256_bytes(data))
            pub = C.publish(store, SOURCE, date, w["id"], "h5", data, man)
            res["published"] += 1
            res["latency"].append({"id": w["id"], "mdtm_minus_end_min": man["mdtm_minus_window_end_min"],
                                   "ingest_minus_end_min": man["ingestion_minus_window_end_min"]})
            C.log("published", **pub)
    finally:
        sess.close()
    res["seconds"] = round(time.time() - t0, 1)
    res["store"] = store.stats()
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="r2")
    ap.add_argument("--limit", type=int, default=None, help="max windows this run (default from config)")
    ap.add_argument("--newest", type=int, default=0,
                    help="test aid: consider only the newest N listed windows (0 = all; CI uses 0)")
    a = ap.parse_args(argv)
    cfg = C.load_config()
    store = C.open_store(a.store, cfg)
    try:
        res = run(store, cfg, a.limit, newest=a.newest)
    except SourceStopped as exc:
        C.log("source_stopped", source=SOURCE, error=str(exc), store=store.stats())
        return 2
    C.log("summary", **res)
    if res["failed"] and not res["published"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(C.cli(main))
