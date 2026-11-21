"""P4B post-final job p4b_post on synthetic data (no network, no LightGBM, no real Gold data): it reads only the
fixed source run (sha-checked), writes only under lake/v1/runs/p4b_post/, never touches the source run or
OOF_DONE.json, joins label_rate_mm_h, draws the reliability diagrams, builds the replay packages (test days, Yagi
from OOF fold 2024-09 with the nested 7.6 rule, Matmo with IMERG Late labels) and labels everything."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from src.models import cloud_train as CT
from src.models import p4b as PB
from src.models import p4b_post as PP
from src.realtime import common as C

CFG = PB.load_cfg()
LEADS, THRS = [0, 30, 60], [0.5, 2.5, 7.6]
PIX = np.array([1095 * 3600 + 2843 + 3600 * 12 + 14, 1095 * 3600 + 2843 + 3600 * 12 + 15,
                1095 * 3600 + 2843 + 3600 * 13 + 14], "int64")
SRC = f"lake/v1/runs/{PP.SOURCE_JOB}/{PP.SOURCE_RUN}/"
CUT = {"A": {0.5: 0.30, 2.5: 0.20, 7.6: 0.06}, "B": {0.5: 0.32, 2.5: 0.22, 7.6: 0.07}}


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("HFL_P2B_EXCLUDE_SOURCES", "none")
    monkeypatch.setattr(CT, "ROOT", tmp_path / "ws")
    monkeypatch.setattr(PB, "ROOT", tmp_path / "ws")


def _t0s(day_local: str) -> pd.DatetimeIndex:
    lo = pd.Timestamp(day_local).tz_localize(PP.TZ).tz_convert("UTC")
    return pd.date_range(lo, periods=48, freq="30min")


def _rate(t0: pd.Series, pix: np.ndarray, lead: int) -> np.ndarray:
    h = t0.dt.hour.to_numpy() + lead / 60
    return np.round(np.clip(12 * np.sin(h / 24 * np.pi) + (pix % 7) - 2, 0, None), 2)


def _rows(days: list[str]) -> pd.DataFrame:
    t0 = pd.DatetimeIndex(sorted(set().union(*[set(_t0s(d)) for d in days])))
    g = pd.MultiIndex.from_product([PIX, t0, LEADS], names=["pixel_id", "t0", "lead_min"]).to_frame(index=False)
    g["lead_min"] = g["lead_min"].astype("int16")
    g["label_rate_mm_h"] = _rate(g["t0"], g["pixel_id"].to_numpy(), 0).astype("float32")
    return g


class Src:
    """Builds the fake source run (files + markers + done.json) and the Gold partitions on a local store."""

    def __init__(self, tmp_path):
        self.spec = f"local:{tmp_path / 'store'}"
        self.root = tmp_path / "store"
        self.outputs = {}

    def put(self, rel: str, data: bytes, marker_listed=True):
        k = SRC + "files/" + rel
        p = self.root / k
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        if marker_listed:
            self.outputs[k] = hashlib.sha256(data).hexdigest()

    def parquet(self, rel, df):
        import io
        b = io.BytesIO(); df.to_parquet(b, index=False)
        self.put(rel, b.getvalue())

    def json(self, rel, obj):
        self.put(rel, json.dumps(obj).encode())

    def finish(self):
        m = self.root / (SRC + "markers/final-test.done.json")
        m.parent.mkdir(parents=True, exist_ok=True)
        m.write_text(json.dumps({"unit": "final-test", "outputs": self.outputs}))
        (self.root / (SRC + "done.json")).write_text(json.dumps({"run_id": PP.SOURCE_RUN}))

    def gold(self, df: pd.DataFrame):
        df = df.copy()
        df["frame_start"] = df["t0"] + pd.to_timedelta(df["lead_min"].astype(int), unit="min")
        df["target_time"] = df["frame_start"] + pd.Timedelta("30min")
        df["season"] = df["t0"].dt.year
        for t in THRS:
            df[PB.thr_col(t)] = (df["label_rate_mm_h"] >= t).astype("int8")
        for k in range(7):
            df[f"gc_k{k}"] = np.where(k == 0, df["label_rate_mm_h"] * 0.8, np.nan).astype("float32")
        df["lat"], df["lon"] = np.float32(21.0), np.float32(105.8)
        ym = df["t0"].dt.strftime("%Y-%m")
        for m, part in df.groupby(ym):
            k = f"lake/v1/gold/met_training/table_A1_v1/gid={PP.SOURCE_GOLD}/season={m[:4]}/month={m[5:]}/part.parquet"
            p = self.root / k
            p.parent.mkdir(parents=True, exist_ok=True)
            part.to_parquet(p, index=False)


def _final_preds(src: Src, period: str, days: list[str], g: pd.DataFrame):
    for L in LEADS:
        for t in THRS:
            d = g[(g["lead_min"] == L) & g["t0"].dt.tz_convert(PP.TZ).dt.date.astype(str).isin(days)].copy()
            y = (d["label_rate_mm_h"] >= t).astype("int8")
            p = np.clip(d["label_rate_mm_h"].to_numpy() / (3 * t) + 0.05, 0, 1).astype("float32")
            pr = d[["pixel_id", "t0", "lead_min"]].assign(y=y.to_numpy(), p_v1_A=p, p_v1_B=p * 0.9, p_v1=p * 0.9,
                                                         yhat_v1_A=(p >= 0.3).astype("int8"),
                                                         yhat_v1_B=(p * 0.9 >= 0.3).astype("int8"),
                                                         yhat_v1=(p * 0.9 >= 0.3).astype("int8"))
            src.parquet(f"reports/p4b/final/preds/{period}_lead{L}_thr{PB.kstr(t)}.parquet", pr)


def _oof(src: Src, days: list[str]):
    g = _rows(days)
    for v in ("A", "B"):
        for L in LEADS:
            for t in THRS:
                k = PB.kstr(t)
                uid = {("A", 2.5): f"cv-g1-lead{L}-thr{k}"}.get((v, t), f"cv-e-lead{L}-thr{k}" if v == "A"
                                                                  else f"cv-B24-lead{L}-thr{k}")
                d = g[g["lead_min"] == L].copy()
                rate = d["label_rate_mm_h"].to_numpy()
                d["p_raw"] = np.clip(rate / (3 * t), 0, 1).astype("float32")
                d["y"] = (rate >= t).astype("int8")
                d["fold"] = np.int16(4 if v == "B" else 10)
                d["day"] = d["t0"].dt.tz_convert(PP.TZ).dt.date.astype(str)
                d["p_best"] = d["p_raw"]; d["p_clim"] = np.float32(0.05); d["pers"] = (rate * 0.8).astype("float32")
                src.parquet(f"reports/p4b/oof/{uid}.parquet", d)
                folds = [{"fold": i, "start": f"{m}T00:00:00+00:00"} for i, m in
                         (enumerate(["2024-05-01", "2024-06-01", "2024-07-01", "2024-08-01", "2024-09-01",
                                     "2024-10-01"]) if v == "B" else
                          enumerate([f"2023-{x:02d}-01" for x in range(5, 11)] + [f"2024-{x:02d}-01" for x in range(5, 11)]))]
                src.json(f"reports/p4b/cv/{uid}.json", {"unit": uid, "folds": folds})
                src.json(f"models/v1/{v}/lead{L}_thr{k}/meta.json",
                         {"cv_unit": uid, "calibration": "raw", "cut": {"cut": CUT[v][t]},
                          "heavy_rule": "nested" if t == 7.6 else None})


def _reliability(src: Src):
    rows = []
    for per in ("test", "secondary"):
        for L in LEADS:
            for t in THRS:
                for m in PP.MODELS_FINAL:
                    for b in range(10):
                        n = 0 if b == 9 else 1000 // (b + 1)
                        rows.append({"bin": b, "p_lo": b / 10, "p_hi": (b + 1) / 10, "n": n,
                                     "mean_p": np.nan if n == 0 else b / 10 + 0.05,
                                     "obs_freq": np.nan if n == 0 else b / 10 + 0.03,
                                     "period": per, "lead_min": L, "threshold": t, "model": m})
    src.put("reports/p4b/final/reliability.csv", pd.DataFrame(rows).to_csv(index=False).encode())


@pytest.fixture
def world(tmp_path):
    src = Src(tmp_path)
    test_days = ["2025-08-26", "2025-09-29", "2025-09-30"]
    gt = _rows(test_days)
    gt = gt[gt["t0"] < pd.Timestamp("2025-10-01", tz="UTC")]
    gs = _rows(["2025-10-06", "2025-10-07"])
    other = _rows(["2025-05-15", "2025-06-15", "2025-07-15"])       # every test month has a Gold partition
    src.gold(pd.concat([gt, gs, other], ignore_index=True))
    _final_preds(src, "test", test_days, gt)
    _final_preds(src, "secondary", ["2025-10-06", "2025-10-07"], gs)
    _oof(src, ["2024-09-06", "2024-09-07", "2024-09-08"])
    _reliability(src)
    src.finish()
    return src


def _run(src, run_id="p1"):
    return C.cli(CT.main, ["--job", "p4b_post", "--store", src.spec, "--run-id", run_id])


# ---------------------------------------------------------------- the whole job
def test_post_job_end_to_end(world, tmp_path, capsys):
    before = {p.relative_to(world.root).as_posix(): p.read_bytes() for p in world.root.rglob("*") if p.is_file()}
    assert _run(world) == 0
    after = {p.relative_to(world.root).as_posix(): p.read_bytes() for p in world.root.rglob("*") if p.is_file()}
    new = sorted(set(after) - set(before))
    assert new and all(k.startswith("lake/v1/runs/p4b_post/") for k in new)      # own prefix only
    assert all(after[k] == v for k, v in before.items())                           # source run untouched
    assert not (world.root / PB.OOF_DONE_KEY).exists()
    st = CT.open_run_store(world.spec, "p4b_post")
    assert CT.done_units(st, "p4b_post", "p1") == {u.id for u in PP.units()}
    assert st.list("lake/v1/runs/p4b_post/p1/done.json") and st.list("lake/v1/runs/p4b_post/p1/summary.json")
    ws = CT.ROOT
    out = ws / PP.OUT_REL
    # labels joined, label source per row, y consistent
    lt = pd.read_parquet(out / "preds_labelled" / "test_lead30_thr2p5.parquet")
    assert set(lt["label_source"]) == {"imerg_final"} and lt["label_rate_mm_h"].notna().all()
    assert ((lt["label_rate_mm_h"] >= 2.5).astype(int) == lt["y"]).all()
    ls = pd.read_parquet(out / "preds_labelled" / "secondary_lead0_thr7p6.parquet")
    assert set(ls["label_source"]) == {"imerg_late"}
    # reliability diagrams + ECE table, labelled
    assert (out / "reliability" / "reliability_test.png").stat().st_size > 10_000
    assert (out / "reliability" / "reliability_secondary.png").is_file()
    e = pd.read_csv(out / "reliability" / "ece.csv")
    assert len(e) == 2 * 3 * 3 * 3 and set(e["label_source"]) == {"imerg_final", "imerg_late"}
    assert e["note"].str.contains("not flood prediction").all()
    # replay packages
    for name in PP.EVENTS:
        d = out / "replay" / name
        s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        assert "not flood prediction" in s["status"] and s["label_name"]
        assert {"panels_F0.png", "panels_F30.png", "panels_F60.png", "replay.parquet",
                "per_day_metrics.csv"} <= {p.name for p in d.iterdir()}
        r = pd.read_parquet(d / "replay.parquet")
        assert set(r["day"]) == set(PP.EVENTS[name]["days"])
    m = json.loads((out / "replay" / "matmo_2025-10-06_07" / "summary.json").read_text(encoding="utf-8"))
    assert m["label_source"] == ["imerg_late"] and "provisional" in m["label_name"]
    pdm = pd.read_csv(out / "replay" / "matmo_2025-10-06_07" / "per_day_metrics.csv")
    assert sorted(pdm["day_local"].unique()) == ["2025-10-06", "2025-10-07"]
    assert {"v1", "pers"} <= set(pdm["model"]) and set(pdm["label_source"]) == {"imerg_late"}
    assert len(pdm[pdm["model"].isin(["v1", "pers"])]) == 2 * 3 * 3 * 2
    idx = json.loads((out / "index.json").read_text(encoding="utf-8"))
    assert idx["source_run"] == f"p4b/{PP.SOURCE_RUN}" and idx["outputs"]
    log = capsys.readouterr().out
    assert '"source_run_id": "c0df12477947"' in log
    # resume: nothing runs again
    assert _run(world) == 0
    assert '"unit_start"' not in capsys.readouterr().out


def test_yagi_from_oof_rebuilds_yhat_with_nested_rule(world):
    assert _run(world) == 0
    r = pd.read_parquet(CT.ROOT / PP.OUT_REL / "replay" / "yagi_2024-09-06_08" / "replay.parquet")
    assert set(r["label_source"]) == {"imerg_final"}
    for v in ("A", "B"):
        for t in (0.5, 2.5):
            k = PB.kstr(t)
            assert (r[f"yhat_v1_{v}_{k}"] == (r[f"p_v1_{v}_{k}"] >= np.float32(CUT[v][t])).astype("int8")).all()
        nested = (r[f"p_v1_{v}_7p6"] >= np.float32(CUT[v][7.6])) & (r[f"p_v1_{v}_2p5"] >= np.float32(CUT[v][2.5]))
        assert (r[f"yhat_v1_{v}_7p6"] == nested.astype("int8")).all()
    s = json.loads((CT.ROOT / PP.OUT_REL / "replay" / "yagi_2024-09-06_08" / "summary.json").read_text("utf-8"))
    assert s["oof_units"]["A-lead0-thr2p5"]["cv_unit"] == "cv-g1-lead0-thr2p5"
    assert s["oof_units"]["B-lead60-thr7p6"]["cv_unit"] == "cv-B24-lead60-thr7p6"
    assert "OUT-OF-FOLD" in s["note"]


# ---------------------------------------------------------------- guards
def test_store_cannot_write_source_run_or_oof_done(tmp_path):
    st = CT.open_run_store(f"local:{tmp_path}", "p4b_post")
    for key in (SRC + "files/x", SRC + "markers/x.done.json", PB.OOF_DONE_KEY,
                "lake/v1/gold/met_training/table_A1_v1/x", "lake/v1/runs/p4b_smoke/x"):
        with pytest.raises(C.KeyRefused):
            st.put(key, b"{}")
    st.put("lake/v1/runs/p4b_post/r/files/ok.txt", b"ok")


def test_source_file_with_wrong_sha_fails(world):
    k = world.root / (SRC + "files/reports/p4b/final/reliability.csv")
    k.write_bytes(k.read_bytes() + b"\n")                 # bytes no longer match the marker
    assert _run(world) == 1
    st = CT.open_run_store(world.spec, "p4b_post")
    assert "reliability" not in CT.done_units(st, "p4b_post", "p1")


def test_unfinished_source_run_is_refused(world):
    (world.root / (SRC + "done.json")).unlink()
    assert _run(world) == 1


def test_file_not_listed_in_a_marker_is_refused(world):
    world.put("reports/p4b/final/extra.csv", b"x", marker_listed=False)
    world.finish()
    with pytest.raises(KeyError):
        PP.SourceRun(CT.open_run_store(world.spec, "p4b_post")).fetch("reports/p4b/final/extra.csv")


def test_post_refuses_final_and_oof_flags(tmp_path):
    for extra in (["--final"], ["--write-oof-done"], ["--write-oof-done", "--approve-final"]):
        assert C.cli(CT.main, ["--job", "p4b_post", "--store", f"local:{tmp_path}"] + extra) == 2


def test_post_never_loads_a_model_or_trains():
    code = (PB.CODE_ROOT / "src" / "models" / "p4b_post.py").read_text(encoding="utf-8")
    for bad in ("lightgbm", "lgb.", "_calibrated(", "run_final(", "run_cv(", "run_refit(", ".predict("):
        assert bad not in code, bad


def test_units_and_run_id(monkeypatch):
    monkeypatch.setattr(CT, "ROOT", PB.CODE_ROOT)            # run ids hash the code files of the repository
    us = PP.units()
    assert [u.id for u in us][-1] == "post-report" and len(us) == 9
    assert set(us[-1].deps) == {u.id for u in us[:-1]}
    rid = CT.run_id_for("p4b_post")
    assert rid == CT.run_id_for("p4b_post") and rid not in (CT.run_id_for("p4b"), CT.run_id_for("p4b_smoke"))
    assert PP.SOURCE_RUN == "c0df12477947" and PP.SOURCE_GOLD == "0803d9d8ed72"
