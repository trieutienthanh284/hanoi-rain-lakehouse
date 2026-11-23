"""P4E.6 figures (D196) on SYNTHETIC pooled frames (no real data, no LightGBM): FSS inside a truncated city mask,
bootstrap CIs, and every figure function writes its PNG + table."""
import numpy as np
import pandas as pd
import pytest

from src.gold import features as F
from src.models import p4e_figs as FG

G = F.Grid()
# a small irregular "city": 3x3 block plus one isolated pixel (truncated windows at every edge)
CITY = np.array([(G.lat_idx_min + 10 + i) * 3600 + G.lon_idx_min + 10 + j for i in range(3) for j in range(3)]
                + [(G.lat_idx_min + 14) * 3600 + G.lon_idx_min + 14])


def _pooled(seed, days=("2024-07-01", "2024-07-02", "2024-07-20"), skill=0.8, late_day=None):
    rng = np.random.default_rng(seed)
    t0 = pd.DatetimeIndex(np.concatenate([pd.date_range(pd.Timestamp(d, tz="UTC"), periods=12, freq="30min").values
                                          for d in days]), tz="UTC")
    n = len(CITY)
    df = pd.DataFrame({"pixel_id": np.tile(CITY, len(t0)), "t0": np.repeat(t0, n)})
    df["t0"] = df["t0"].astype("datetime64[us, UTC]")
    lab = rng.gamma(0.6, 4.0, len(df)).astype("float32")
    df["label_rate_mm_h"] = lab
    df["day"] = df["t0"].dt.tz_convert("Asia/Ho_Chi_Minh").dt.date.astype(str)
    utc = df["t0"]
    df["half"] = [f"{a:02d}-{'1' if b <= 15 else '2'}" for a, b in zip(utc.dt.month, utc.dt.day)]
    df["role"] = "eval"
    df["label_source"] = "imerg_final_v07"
    if late_day:
        df.loc[df["day"] == late_day, ["role", "label_source"]] = ["secondary", "imerg_late_v07"]
    df["pers"] = (lab * rng.lognormal(0, 0.6, len(df))).astype("float32")
    df["season_fold"] = 2024
    return df


def _frames(models=("A", "B"), thrs=(2.5, 7.6), leads=(0, 30), late_day=None):
    out = {}
    for k, m in enumerate(models):
        for t in thrs:
            for L in leads:
                j = _pooled(1, late_day=late_day)
                rng = np.random.default_rng(10 + k)
                y = j["label_rate_mm_h"].to_numpy() >= t
                p = np.clip(0.7 * y + 0.15 + rng.normal(0, 0.2, len(j)), 0.001, 0.999)
                j["y"], j["p_raw"], j["p_cal"] = y, p, np.clip(p * 0.9, 0, 1)
                j["yhat"] = j["p_cal"] >= 0.5
                out[(m, float(t), int(L))] = j
    return out


def test_fss_perfect_and_null_forecasts():
    j = _pooled(3)
    y = j["label_rate_mm_h"].to_numpy() >= 2.5
    for size in (1, 3, 5):
        tr = FG.fss_terms(j["pixel_id"].to_numpy(), j["t0"], j["day"].to_numpy(), y, y, size)
        assert (tr["num"] == 0).all() and tr["den"].sum() > 0
        assert FG.fss_ci(tr, 50, 1, 0.9)[0] == pytest.approx(1.0)
    tr0 = FG.fss_terms(j["pixel_id"].to_numpy(), j["t0"], j["day"].to_numpy(), y, np.zeros(len(y), bool), 3)
    assert FG.fss_ci(tr0, 50, 1, 0.9)[0] == pytest.approx(0.0)   # no forecast: FSS 0


def test_fss_window_truncated_to_mask():
    # one t0, observation only at the isolated pixel, forecast only at the block centre: at 1x1 they never overlap;
    # with a 3x3 window the isolated pixel's window holds only itself (truncated, count 1), so its fraction stays 1
    t = pd.Series(pd.to_datetime(["2024-07-01T00:00Z"] * len(CITY)))
    obs = np.zeros(len(CITY), bool); obs[-1] = True
    fc = np.zeros(len(CITY), bool); fc[4] = True
    day = np.array(["2024-07-01"] * len(CITY))
    t1 = FG.fss_terms(CITY, t, day, obs, fc, 1)
    assert t1["num"].sum() == pytest.approx(2.0) and t1["den"].sum() == pytest.approx(2.0)
    t3 = FG.fss_terms(CITY, t, day, obs, fc, 3)
    # Po = 1 at the isolated pixel only; Pf = 1/count over the 3x3 block (centre window 9, edges 6, corners 4)
    pf = np.array([1 / 4, 1 / 6, 1 / 4, 1 / 6, 1 / 9, 1 / 6, 1 / 4, 1 / 6, 1 / 4])
    assert t3["num"].sum() == pytest.approx((pf ** 2).sum() + 1.0)
    assert t3["den"].sum() == pytest.approx((pf ** 2).sum() + 1.0)


def test_all_figures_write_png_and_tables(tmp_path):
    Fr = _frames(late_day="2024-07-20")
    models, thrs, leads = ["A", "B"], [2.5, 7.6], [0, 30]
    outs = [FG.reliability(Fr, models, thrs, leads, tmp_path, "T", 10),
            FG.scores_ci(Fr, models, thrs, leads, tmp_path, "T", 50, 1, 0.9),
            FG.fss(Fr, models, thrs, leads, tmp_path, "T", [1, 3, 5], 50, 1, 0.9),
            FG.halfmonth_csi(Fr, models, thrs, leads, tmp_path, "T"),
            FG.diagnosis(Fr, models, thrs, leads, tmp_path, "T")]
    cfg = {"evaluation": {"events": {"x": ["2024-07-20", "2024-07-20"], "y": ["2023-01-01", "2023-01-01"]}}}
    days = FG.event_days(cfg, Fr, "A", 2.5, 0)
    assert {d["name"] for d in days} == {"x", "y", "wettest2024"}
    outs.append(FG.replay(Fr, ["A", "B"], 2.5, leads, days, tmp_path, "T"))
    for paths, tab in outs:
        assert paths and all(p.is_file() and p.stat().st_size > 1000 for p in paths) and len(tab)
    sc = outs[1][1]
    assert set(sc["model"]) == {"A", "B", "persistence"} and (sc["lo"] <= sc["hi"]).all()
    assert set(outs[2][1]["size"]) == {1, 3, 5}
    rp = outs[-1][1]
    assert set(rp.loc[rp["name"] == "y", "status"]) == {"day not in evaluation"}
    late = rp[(rp["name"] == "x") & (rp["status"] == "ok")]
    assert len(late) == 2 and late["late"].all()                   # Late labels are marked
    dg = outs[4][1]
    assert set(dg["prob"]) == {"cal", "raw"} and (dg["n"] > 0).all()


def test_fetch_copies_figs_without_overwriting(tmp_path):
    from src.models import fetch_run as FR
    src = tmp_path / "reports/p4e_smoke/cloud/r1/files/reports/p4e/figs"
    src.mkdir(parents=True)
    (src / "fss.png").write_bytes(b"png1")
    (src / "fss.csv").write_bytes(b"a,b\n")
    got = FR.copy_figs(tmp_path, "p4e_smoke", "r1")
    assert got == ["reports/p4e/figs/p4e_smoke_r1/fss.csv", "reports/p4e/figs/p4e_smoke_r1/fss.png"]
    assert FR.copy_figs(tmp_path, "p4e_smoke", "r1") == got          # identical bytes: no-op
    (src / "fss.png").write_bytes(b"other")
    with pytest.raises(RuntimeError):
        FR.copy_figs(tmp_path, "p4e_smoke", "r1")
    assert (tmp_path / got[1]).read_bytes() == b"png1"
    assert FR.copy_figs(tmp_path, "p4e_smoke", "missing") == []
