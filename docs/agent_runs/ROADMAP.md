# Roadmap (approved 2026-11-15, target date about 2026-12-10)

Each phase is one autonomous agent run that ends with a report the user reviews. Long downloads are launched by the user from the runbook and run outside the agent.

| Phase | Outcome | Main technology |
|---|---|---|
| P1 Data foundation | Whole-Hanoi domain, regional-crop Bronze, per-pixel Silver, temporal tests, readiness probes, first R2 sync, backfill runbook | Python, h5py, DuckDB/Parquet, pytest, boto3 |
| P2 History and new sources | Backfill IMERG Final (rainy seasons 2025, 2024, then 2023), GSMaP NOW archive, IMERG Late, Open-Meteo forecast runs; full-history quality report; threshold analysis | same + Open-Meteo API |
| P3 Continuous ingestion | GitHub Actions schedules: GSMaP every 30 min, forecasts hourly, IMERG Late daily, all written to R2; freshness monitor | GitHub Actions, R2 |
| P4 Gold and model | Training table, rolling-origin backtests, LightGBM vs baselines, model registry, promotion rule | LightGBM, MLflow |
| P5 Serving and web | Batch inference every 30 min, city map (30/60 min), dashboard with verification and data freshness | FastAPI or static files on R2, MapLibre/deck.gl, Cloudflare Pages |
| P6 Extensions | Flood susceptibility + risk levels for the whole city, Iceberg tables (Basin Catalog), Airflow DAGs for backfill/retrain, Spark scale test, Kafka event bus, Himawari features, gauges if permitted | Iceberg, Airflow, Spark, Kafka |

## Gates before any product model (P4)

- **G1 Coverage:** at least two full rainy seasons of IMERG Final + GSMaP NOW for the whole-Hanoi domain (three is the target), including early-summer convective rain and typhoon rain; completeness ≥ 98 % of expected windows.
- **G2 Temporal correctness:** leakage tests pass on the full training table.
- **G3 Source consistency:** GSMaP vs IMERG bias, correlation and timing measured on the full history and documented; calibration decided if needed.
- **G4 Threshold:** rain threshold(s) locked by the user in `configs/targets.yaml`.

## Quality loop once data flows continuously (P3–P5)

- Labels are versioned: IMERG Late = provisional label (about 14 h delay), IMERG Final V08 = definitive when released.
- Daily verification of yesterday's forecasts against arriving labels.
- A retrained model replaces the running one only if it beats both the running model and persistence on a fixed hold-out and on the most recent weeks.
- Stale or missing sources set a stale flag and fall back to the simple baseline.
