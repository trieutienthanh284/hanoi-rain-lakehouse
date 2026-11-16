# Hanoi rain lakehouse — continuous ingestion

Scheduled GitHub Actions keep Hanoi rainfall data fresh in Cloudflare R2:
GSMaP NOW (JAXA, every 30 min), IMERG Late V07 (NASA GES DISC, daily, provisional),
Open-Meteo NWP runs (GFS Global, ECMWF IFS 0.25°, hourly check), daily compaction and an
hourly freshness monitor (a failed run = alert e-mail). Code: src/realtime/. Design: docs/.

Rainfall truth in this project is IMERG, a satellite estimate (not rain gauges).
No flood probability or FLOOD / NO FLOOD prediction is claimed.

Data sources: GSMaP NOW © JAXA EORC; IMERG © NASA GES DISC;
Weather data by Open-Meteo.com (CC BY 4.0).

Status: continuous ingestion started on 2026-11-16.
