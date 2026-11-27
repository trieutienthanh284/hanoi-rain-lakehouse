"""U1.6 — historical rain total vs inundation duration (JICA Table 6-12, Silver records).

No model and no threshold decision: a table and a plot as input for the hazard discussion
(docs/product/PRODUCT_SPEC_WEB_V0.md §4).

Usage: .venv/Scripts/python.exe -m src.events.jica_history
"""
from __future__ import annotations

import json

from src.events.common import CATALOG_DIR, REPORTS, write_json_atomic

JICA_EVIDENCE = "fe_pdf_2b85b288a97795b1"


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    cat = pd.read_parquet(CATALOG_DIR / "events.parquet")
    j = cat[(cat["evidence_id"] == JICA_EVIDENCE)].copy()
    j = j[["temporal_text", "event_date_start", "event_date_end", "time_precision", "rainfall_mm",
           "inundation_duration_hours", "extracted_text", "silver_record_id"]].sort_values("event_date_start")
    j["page_reference"] = "JICA report, PDF page 46, Table 6-12 'Record of inundation time in Hanoi City'"
    j.to_csv(REPORTS / "jica_rain_inundation.csv", index=False, encoding="utf-8")
    ok = j.dropna(subset=["rainfall_mm", "inundation_duration_hours"])
    rho = float(ok[["rainfall_mm", "inundation_duration_hours"]].corr(method="spearman").iloc[0, 1]) if len(ok) > 2 else None

    fig, ax = plt.subplots(figsize=(7, 5), dpi=130)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.scatter(ok["rainfall_mm"], ok["inundation_duration_hours"], s=48, color="#2a78d6",
               edgecolor="#fcfcfb", linewidth=1, zorder=3)
    for _, r in ok.iterrows():
        if r["rainfall_mm"] >= 290 or r["inundation_duration_hours"] >= np.nanpercentile(ok["inundation_duration_hours"], 85):
            ax.annotate(r["temporal_text"], (r["rainfall_mm"], r["inundation_duration_hours"]), fontsize=7,
                        color="#52514e", xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("Tổng lượng mưa đợt (mm, JICA)", color="#0b0b0b")
    ax.set_ylabel("Thời gian ngập (giờ)", color="#0b0b0b")
    ax.set_title(f"U1.6 — Mưa và thời gian ngập lịch sử, Hà Nội (JICA bảng 6-12), n = {len(ok)}",
                 fontsize=10, loc="left", color="#0b0b0b")
    ax.grid(color="#e6e5e1", lw=0.6)
    for s in ax.spines.values():
        s.set_color("#c9c8c3")
    ax.tick_params(colors="#52514e")
    fig.tight_layout()
    fig.savefig(REPORTS / "jica_rain_inundation.png", facecolor=fig.get_facecolor())
    plt.close(fig)
    st = {"rows": int(len(j)), "with_both": int(len(ok)), "spearman_rho": None if rho is None else round(rho, 3),
          "rain_mm_range": [float(ok["rainfall_mm"].min()), float(ok["rainfall_mm"].max())],
          "hours_range": [float(ok["inundation_duration_hours"].min()), float(ok["inundation_duration_hours"].max())],
          "years": [int(j["event_date_start"].min().year), int(j["event_date_start"].max().year)]}
    write_json_atomic(REPORTS / "jica_summary.json", st)
    print(json.dumps(st))
    print(ok[["temporal_text", "rainfall_mm", "inundation_duration_hours"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
