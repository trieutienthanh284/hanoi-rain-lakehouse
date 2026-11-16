# BÁO CÁO PHASE 3A — Thu thập liên tục trên GitHub Actions (dựng và kiểm thử cục bộ)

Ngày: 2026-11-16. Nhánh: `feat/met-dataset` (thay đổi chưa commit). Tệp task: `TASK_P3A_CONTINUOUS_INGESTION.md`.
Trong suốt phase, hai cửa sổ backfill vẫn chạy; không có ghi nào vào `data/bronze_regional/`, `data/silver/met/`, `logs/backfill/`, `data/_scratch/` (bằng chứng §9). Mọi đầu ra thử nghiệm nằm trong `data/_ci_test/`.

Lưu ý phương pháp: mưa "thật" của dự án là IMERG — ước lượng vệ tinh, không phải trạm đo. Phase này không tạo mô hình, không đưa ra xác suất ngập hay dự báo FLOOD / NO FLOOD.

## 1. Tóm tắt

| Task | Kết quả | Bằng chứng chính |
|---|---|---|
| P3A.1 Entry points `src/realtime/` | **PASS** | 5 entry point + `common.py`; `pytest -q tests/test_realtime.py` → **19 passed** (không mạng) |
| P3A.2 Workflows | **PASS** | 6 tệp `.github/workflows/`, `requirements-ci.txt`; tập test CI `pytest -q tests/test_realtime.py tests/test_temporal_contract.py` → **42 passed** (cũng 42 trong snapshot sạch) |
| P3A.3 E2E cục bộ (nguồn thật + R2) | **PASS** | GSMaP 3 tệp/12 s, lần 2 tải lên 0; IMERG Late 2026-11-14 48/48 granule/109 s; Open-Meteo 1 request/2 run; compaction; freshness OK (exit 0) |
| P3A.4 Sẵn sàng công khai (phân tích) | **PASS** | detect-secrets + quét giá trị chính xác trên 19 commit/61 blob và working tree: **0 bí mật**; `.gitignore` thiếu `data/`, `_archive/`, `.env.*` (đề xuất trong runbook); khuyến nghị lộ trình (b); `RUNBOOK_P3_PUBLISH.md` |
| Toàn bộ test dự án | PASS | `pytest -q tests` → **56 passed** (37 cũ + 19 mới) |

Không gặp điều kiện dừng nào: không có giá trị bí mật trong bất kỳ đầu ra nào (quét 91 tệp do tôi tạo + 216 tệp working tree: 0 kết quả), không cần ghi vào thư mục backfill, không push/publish/đổi cài đặt GitHub.

## 2. Tệp đã tạo / sửa

Tạo mới:
- `src/realtime/__init__.py`, `common.py` (store R2/cục bộ put-if-absent, chặn khóa ngoài `lake/v1/realtime/` và `lake/v1/status/`, che bí mật trên stdout/stderr, chuyển hướng writer P1/P2A), `gsmap_now_realtime.py`, `imerg_late_daily.py`, `openmeteo_latest.py`, `freshness_monitor.py`, `compact_daily.py`.
- `configs/realtime.yaml` (ngưỡng freshness, lookback, 81 điểm Open-Meteo, model, endpoint metadata).
- `tests/test_realtime.py` (19 test).
- `.github/workflows/gsmap_now.yml`, `openmeteo.yml`, `imerg_late.yml`, `compact.yml`, `freshness.yml`, `tests.yml`; `requirements-ci.txt`.
- `docs/agent_runs/RUNBOOK_P3_PUBLISH.md`, `docs/agent_runs/publish_paths_p3.txt`, báo cáo này.

Sửa: `CLAUDE.md` (dòng "Current task file"), `docs/agent_runs/STATUS.md`, `DECISIONS.md` (D24–D37), `env.txt` (thêm `detect-secrets==1.5.0`).
Không sửa: collector hiện có, `*_dq.py`, `regional_crop.py`, `imerg_subset_collector.py`, `r2_sync.py`, `run_backfill.py`, các hợp đồng và config nguồn chân lý.

## 3. Thiết kế (tóm tắt)

| Thành phần | Lịch (UTC) | Hành vi | Đối tượng R2 |
|---|---|---|---|
| `gsmap_now_realtime` | phút 7, 37 | liệt kê `/now/latest` (48 cửa sổ ~24 h), xử lý mọi cửa sổ chưa có trên R2, mới nhất trước; validate (`gsmap_now_collector`), crop (`regional_crop`) | `realtime/gsmap_now/date=D/gsmap_now_YYYYMMDDTHHMM.h5` + manifest |
| `openmeteo_latest` | phút 20 | đọc metadata model; chỉ gọi Forecast API khi có run mới; 81 điểm, `precipitation`, 3 ngày | `realtime/openmeteo/date=D/{model}_runYYYYMMDDTHHMM.parquet` + manifest |
| `imerg_late_daily` | 17:40 | ngày mới nhất đủ 48 granule chưa có trên R2 (nhìn lại 7 ngày, ≤ 2 ngày/lần), đường OPeNDAP P2A; nhãn `provisional` | `realtime/imerg_late/date=D/imerg_late_YYYYMMDD.h5` + manifest |
| `compact_daily` | 01:15 | gộp cửa sổ GSMaP của ngày đã kết thúc (nhìn lại 3 ngày) bằng `write_day_file` | `realtime/compacted/gsmap_now/date=D/gsmap_now_YYYYMMDD.h5` + manifest |
| `freshness_monitor` | phút 50 | tuổi bản ghi mới nhất mỗi nguồn/luồng; exit 1 khi cũ ⇒ GitHub gửi e-mail | `status/freshness.json` (đối tượng duy nhất được ghi đè) |

Manifest mỗi đối tượng ghi: nguồn, tên gốc, MDTM hoặc issue time, kích thước, sha256 gốc, **`ingestion_time_utc` = thời điểm tải xong thật trên runner**, sha256 crop, kết quả validate, khóa + sha256 đối tượng, `data_time_utc`, run context (`GITHUB_RUN_ID`, `GITHUB_SHA`). GSMaP giữ `available_time = max(MDTM, window_end)` (ADD-003) và ghi rõ chỉ dùng được ở realtime sau `ingestion_time_utc`; Open-Meteo `available_time = ingestion_time`.

Idempotency: manifest ghi sau dữ liệu (commit marker); "đã có manifest ⇒ bỏ qua"; mọi lần ghi là put-if-absent (`If-None-Match: *` + so sha256; nội dung khác ⇒ lỗi, không ghi đè).

Phiên bản: Python **3.14** trên runner (khớp `.venv` 3.14.0; có 3.14.10 trong `actions/python-versions`; PyPI có wheel cp314 manylinux cho mọi gói biên dịch đã pin), `actions/checkout@v7`, `actions/setup-python@v7` (major mới nhất, node24), `ubuntu-24.04`. Mỗi workflow: `workflow_dispatch`, `permissions: contents: read`, `concurrency` với `cancel-in-progress: false`, `timeout-minutes`, cache pip, secret chỉ tham chiếu theo tên trong `env:` của một bước; chú thích về giới hạn lịch và quy tắc 60 ngày nằm ở đầu mỗi tệp.

## 4. Bằng chứng kiểm thử

### 4.1 Test không mạng (`tests/test_realtime.py`, 19 passed)

| Nhóm | Test |
|---|---|
| Catch-up | `test_catch_up_selects_all_missing_newest_first` (ngừng 9 h + thiếu 4 h mới nhất → đúng 18 cửa sổ, mới nhất trước, `--limit`), `test_rain_windows_ignores_flag_and_dat_files`, `test_gsmap_limit_and_catch_up`, `test_imerg_late_selects_newest_complete_missing_days` (không truy vấn CMR cho ngày đã có), `test_due_days_waits_for_day_end` |
| Idempotency | `test_gsmap_idempotent_manifest_and_crop` (lần 2: 0 tải về, 0 tải lên), `test_imerg_late_run_idempotent_and_provisional`, `test_openmeteo_run_rows_and_idempotency` (lần 2 không gọi Forecast API), compaction lần 2 = `exists` |
| Trường manifest | 20 trường bắt buộc GSMaP; crop bằng từng bit crop P1; compaction trùng từng mảng và thuộc tính với writer backfill; Open-Meteo `interval_start ≥ issue`, `lead ≥ 60 min`, `available = ingestion` |
| Staleness | `test_staleness_logic` (OK, STALE, thiếu nguồn, biên đúng ngưỡng không cũ), `test_freshness_monitor_end_to_end_exit_codes` (exit 1 khi cũ) |
| An toàn | `test_store_refuses_keys_outside_realtime_and_status`, `test_imerg_regional_env_works_outside_repository`, `test_openmeteo_skips_run_that_changes_during_request` |
| Bí mật | 8 bí mật giả qua env: lỗi FTP chứa user/password/host, lỗi HTTP/R2 chứa token/account/key/bucket, traceback không bắt, secret nạp sau khi khởi động, manifest; `test_workflows_never_echo_secrets` — giá trị không bao giờ xuất hiện, thay bằng `***` |

### 4.2 E2E cục bộ với nguồn thật và R2 (log: `data/_ci_test/logs/`)

| Bước | Lệnh | Kết quả | Thời gian |
|---|---|---|---|
| GSMaP lần 1 | `python -m src.realtime.gsmap_now_realtime --newest 3` | listed 48, published 3, failed 0; 6 đối tượng, 66.070 B; R2 A=7, B=6 | 12,1 s (13 s tường) |
| GSMaP lần 2 | cùng lệnh | pending 0, published 0, **uploaded_objects 0**; A=1, B=0 | 1,5 s |
| Kiểm tra độc lập | đọc lại từ R2 | 3/3: sha256 đối tượng đúng, sha256 tệp gốc đúng, mảng **bằng từng bit** crop P1 | — |
| Open-Meteo | `python -m src.realtime.openmeteo_latest` | 1 request; `gfs_global_run20261116T0000` 5.751 dòng, lead 60–4.260 min; `ecmwf_ifs025_run20261115T1800` 5.832 dòng, lead 360–4.620 min; 81 điểm, NaN 0 | 8,5 s |
| IMERG Late | `python -m src.realtime.imerg_late_daily --max-days 1` | 2026-11-15 "incomplete (30/48)"; **2026-11-14 48/48** published (80 KB + manifest 106 KB); tải OPeNDAP 4,2 MB | 108,9 s |
| IMERG Late lần 2 | `--date 2026-11-14` | `exists`, 0 tải lên | 1 s |
| IMERG Late mô phỏng runner | `RUNNER_TEMP` ngoài repo, `--date 2026-11-13 --store local:data/_ci_test/r2_mirror` | 48/48, `crop_file` = khóa R2 (xác nhận sửa lỗi D34) | 104,8 s |
| Định dạng | so tệp ngày realtime với tệp backfill `imerg_late_20250715.h5` | cùng 18 dataset, cùng shape/dtype mỗi cửa sổ, cùng khóa thuộc tính gốc | — |
| Compaction (R2→R2, mặc định) | `python -m src.realtime.compact_daily` | 2026-11-13..15: "no realtime windows" (đúng: chưa có dữ liệu realtime các ngày đó) | 3 s |
| Compaction (dữ liệu thật trên R2 → store cục bộ) | `--date 2026-11-16 --dest local:data/_ci_test/r2_mirror` | 3 cửa sổ, `complete: false`; không ghi ngày thiếu vào R2 (D31) | 3 s |
| Freshness lần 1 | `python -m src.realtime.freshness_monitor` | **STALE** (exit 1): ECMWF 728 min > 720 theo issue time → phát hiện báo động giả có hệ thống, đổi mốc (D27) | 4 s |
| Freshness lần 2 | cùng lệnh | **OK**, exit 0: GSMaP 8,6 min; GFS 35,5 min; ECMWF 291,4 min; IMERG Late 1.808,6 min; A=8, B=4 | 4 s |

Tồn kho R2 sau E2E: `lake/v1/realtime/` **12 đối tượng, 293.603 B** (GSMaP 3+3, IMERG Late 1+1, Open-Meteo 2+2) + `lake/v1/status/freshness.json` 1.638 B. Không có ghi nào ngoài hai tiền tố cho phép.

## 5. Độ trễ đo được

GSMaP NOW (3 cửa sổ, 2026-11-16):

| Cửa sổ (bắt đầu) | MDTM − cuối cửa sổ | ingestion − cuối cửa sổ |
|---|---|---|
| 05:00 | +1,18 min | **+4,23 min** (cửa sổ mới nhất) |
| 04:30 | +1,18 min | +34,28 min (catch-up) |
| 04:00 | +1,18 min | +64,34 min (catch-up) |

Đọc: JAXA đăng tệp ≈ 1,2 phút sau khi cửa sổ 1 giờ kết thúc (khớp mẫu /now/latest lúc thăm dò: 04:31:11, 05:01:11, 05:31:11). Với cron phút 7/37, độ trễ ingestion danh nghĩa ≈ 6 phút + độ trễ lịch của GitHub (thường 5–15 phút) + ≈ 10–40 s cài đặt và chạy.

IMERG Late (ngày 2026-11-14, 48 granule): CMR production time − cuối cửa sổ = **14,18–14,67 h** (trung vị 14,43 h). Lúc 06:05 UTC ngày 16, ngày 15 mới có 30/48 granule. → cron 17:40 (D29).

Open-Meteo (metadata): GFS 00Z có mặt 05:33 (≈ 5,6 h sau init), ECMWF 18Z có mặt 01:17 (≈ 7,3 h).

## 6. Ước lượng chi phí

GitHub Actions (mỗi job làm tròn lên phút; ước ≈ 40–60 s cài đặt + chạy):

| Workflow | Lần/tháng | Phút/lần | Phút/tháng |
|---|---|---|---|
| gsmap_now | 1.440 | 1 | 1.440 |
| openmeteo | 720 | 1 | 720 |
| freshness | 720 | 1 | 720 |
| imerg_late | 30 | 2–3 (ngày 1: +5) | ≈ 90 |
| compact | 30 | 1 | 30 |
| tests | theo push | 1–2 | nhỏ |
| **Tổng** | | | **≈ 3.000** |

Repo **public**: runner chuẩn miễn phí, không giới hạn phút ⇒ 0 USD. Repo private gói Free (2.000 phút) sẽ vượt ⇒ lý do thêm cho repo public (hoặc giảm GSMaP xuống mỗi giờ).

Cloudflare R2 (đo trên E2E, nhân theo lịch):

| Thành phần | Class A/tháng | Class B/tháng |
|---|---|---|
| gsmap_now (≈ 1 cửa sổ mới/lần: 1–2 LIST + 2 PUT; 2 HEAD) | ≈ 5.000 | ≈ 2.900 |
| openmeteo (2 LIST/lần; run mới 8/ngày: 2 PUT + 2 HEAD) | ≈ 1.900 | ≈ 500 |
| freshness (đo: A=8, B=4/lần) | ≈ 5.800 | ≈ 2.900 |
| imerg_late (đo: A=9, B=2/lần) | ≈ 300 | ≈ 60 |
| compact (4 LIST + 2 PUT; 96 GET + 2 HEAD/ngày) | ≈ 200 | ≈ 2.900 |
| **Tổng** | **≈ 13.200** (miễn phí 1.000.000) | **≈ 9.300** (miễn phí 10.000.000) |

Dung lượng tăng thêm ≈ 1,7 MB/ngày (GSMaP 48 × 22 KB ≈ 1,06 MB — chủ yếu là metadata HDF5 cố định ≈ 20 KB/đối tượng; compacted ≈ 0,2 MB; Open-Meteo ≈ 0,17 MB; IMERG Late ≈ 0,25 MB) ≈ 51 MB/tháng ≈ 0,62 GB/năm. Cùng backfill (≈ 0,5 GB) vẫn xa ngưỡng 10 GB miễn phí ⇒ **0 USD/tháng** (trần ngân sách 5 USD). Egress R2 miễn phí.

## 7. Kết quả quét bảo mật (P3A.4)

| Kiểm tra | Phạm vi | Kết quả |
|---|---|---|
| Giá trị chính xác của 8 bí mật (script tự nạp `.env`, chỉ in tên biến) | 61 blob của mọi commit, mọi nhánh (19 commit; `main`, `feat/met-dataset`, `origin/main`) + 216 tệp working tree (trừ `.git`, `.venv`, `data`, `.env`) + 91 tệp đầu ra của tôi | **0** |
| detect-secrets 1.5.0 — tệp có thể công bố | 120 tệp (`git ls-files -co --exclude-standard`, trừ `.env*`, `data/`, `_archive/`, nhị phân) | 22 phát hiện / 9 tệp, **tất cả dương tính giả**: `password_env: "GSMAP_FTP_PASSWORD"` (tên biến), sha256/md5 khóa cấu hình OSM, hash khóa tệp trong `tools/gate_runner.py`, bí mật giả trong `tests/test_realtime.py` |
| detect-secrets — lịch sử | 61 blob xuất ra scratchpad | 9 phát hiện / 3 tệp, dương tính giả (tên biến, md5, `shapeID` hex trong GeoJSON) |
| Mẫu token (AKIA…, JWT `eyJ…`, khóa riêng, `NAME=<giá trị ≥ 12 ký tự>`) | lịch sử | 0 |
| `git log --all -- .env .env.*` | lịch sử | chỉ `.env.example` (commit `9e67c65`), nội dung `VRAIN_API_KEY=` rỗng; `.env` chưa từng được commit |
| Tệp lớn trong lịch sử | 61 blob | lớn nhất `data/reference/hanoi_core_h3_res9.geojson` 892 KB, tiếp theo 123 KB; không có tệp nhị phân lớn |

`.gitignore` hiện tại (kiểm tra `git check-ignore`):

| Mục | Được bỏ qua? |
|---|---|
| `.env` | có |
| `.env.local`, `.env.production` (`.env.*`) | **không** |
| `.env.example` (phải được giữ) | không bị bỏ qua — đúng |
| `data/` (chỉ có `data/bronze|silver|gold`) | **không**: `data/bronze_regional` (886 tệp), `data/_scratch`, `data/_ci_test`, `data/reference` lộ ra |
| `logs/`, `.venv/`, `.idea/`, `audit_runs/` | có |
| `_archive/`, `reports/`, `.claude/`, `.vscode/` | **không** (`reports/` hiện 189 KB, không tệp > 1 MB) |

Đề xuất `.gitignore` đầy đủ nằm ở `RUNBOOK_P3_PUBLISH.md` §3.2 (không tự sửa tệp vì P3A.4 là phân tích).

Sai sót tự ghi nhận (D36): lần chạy detect-secrets đầu dùng nhiều cờ `--exclude-files` không có hiệu lực, nên công cụ có thể đã mở `.env`. Kết quả không chứa mục nào cho `.env`, và detect-secrets chỉ lưu hash SHA-1, không lưu giá trị; không có giá trị nào được in. Các lần quét sau dùng danh sách tệp tường minh.

## 8. Lộ trình công bố khuyến nghị và việc người dùng cần làm

Khuyến nghị **(b) repository public mới từ snapshot sạch** gồm đúng 38 đường dẫn (`docs/agent_runs/publish_paths_p3.txt`) + `.gitignore`, `.env.example`, `README.md` mới. Lý do: lịch sử hiện có không chứa bí mật nhưng chứa nhiều phần chưa rà soát cho công khai (flood-evidence, OSM, web, dữ liệu GeoJSON); lộ trình (b) không đụng 11 tệp staged và không đổi visibility của repo `origin` hiện có (`origin/main` trùng `main`). Snapshot dựng thử từ danh sách: 42 test đạt, 5 entry point import được.

Các bước thủ công (chi tiết, lệnh PowerShell chính xác: `RUNBOOK_P3_PUBLISH.md`):
1. Tạo repo public mới trên GitHub, không tự sinh README.
2. Dựng snapshot vào `D:\hanoi-rain-lakehouse-public` theo danh sách; thêm `.gitignore`, `.env.example`, `README.md` (ghi nguồn Open-Meteo CC BY 4.0); chạy pytest (kỳ vọng 42 passed).
3. `git init -b main`, `git add --` từng đường dẫn, kiểm tra `git status`, commit, push.
4. Thêm 8 secret (khuyến nghị token R2 riêng cho CI, giới hạn đúng bucket).
5. Chạy tay lần lượt: gsmap → openmeteo → imerg_late → compact → freshness; đọc dòng `summary`/`freshness`.
6. Bật thông báo e-mail cho workflow thất bại; ghi nhớ quy tắc 60 ngày.
7. Xoay vòng token Earthdata trước 2027-01-12 (nhắc ngày 2026-12-29).

## 9. Quyết định tự đưa ra

D24–D37 trong `DECISIONS.md`. Quan trọng nhất:
- D24: không sửa module P1/P2A đang được backfill dùng; tái sử dụng bằng chuyển hướng biến module trong tiến trình realtime.
- D27: freshness Open-Meteo tính theo thời điểm có mặt của run (metadata), ngưỡng giữ 12 h — sau khi E2E chỉ ra báo động giả theo issue time (ECMWF có mặt ≈ 7,3 h sau init).
- D28: Open-Meteo realtime dùng metadata endpoint cho thời điểm init; `gfs_global` = init mới nhất của hai thành phần (cận trên).
- D29: IMERG Late cron 17:40 UTC theo độ trễ đo được 14,2–14,7 h.
- D31: cờ thử nghiệm `--newest`, `--dest` để chứng minh idempotency và không ghi ngày thiếu vào R2.
- D34: sửa lỗi chỉ có trên runner (`rc.rel` với `$RUNNER_TEMP` ngoài repo); manifest R2 `imerg_late_20261114` (ghi trước khi sửa) giữ `crop_file` là đường dẫn workdir cục bộ — không ghi đè theo quy tắc put-if-absent.
- D36: sai sót cờ loại trừ của detect-secrets (§7).

Giới hạn còn lại:
1. Workflow chưa chạy thật trên GitHub (đúng phạm vi: không publish). Rủi ro chính: JAXA FTP có chấp nhận IP runner hay không; nếu bị chặn, `gsmap-now-realtime` sẽ thoát mã 2 và freshness báo STALE sau 2 h.
2. Open-Meteo: dữ liệu ECMWF 0,25° gốc 3 giờ được Open-Meteo nội suy theo giờ; issue time của `gfs_global` là cận trên vì đây là mô hình ghép.
3. Ước lượng phút Actions dựa trên thời gian cài đặt giả định 40–60 s; số thật sẽ thấy sau tuần đầu.
4. Cửa sổ GSMaP nào rơi khỏi `/now/latest` (> 24 h không chạy được) thì realtime không lấy lại; backfill lưu trữ (`/now/netcdf`) vẫn lấp được.

## 10. Trạng thái git và kiểm tra danh sách staged

```
$ git branch --show-current
feat/met-dataset
$ git diff --cached --name-only | diff - docs/agent_runs/staged_before.txt && echo STAGED_SAME
STAGED_SAME
```

11 tệp staged giữ nguyên (đúng như `staged_before.txt`). Mọi tệp mới của phase này ở trạng thái untracked (`??`), không có thao tác `git add/commit/push/reset/restore/checkout/stash/rebase/clean/rm`. Không xóa tệp nào; không ghi vào `data/bronze`.

Bằng chứng không ghi vào thư mục backfill: không tồn tại `data/bronze_regional/imerg_late/date=2026-11-13|14`, `data/bronze_regional/gsmap_now/date=2026-11-16`, `data/silver/met/gsmap_now/date=2026-11-16`, `data/_scratch/gsmap_now/date=2026-11-16`; manifest backfill không có dòng nào cho 2026-11-13..16. Toàn bộ workdir realtime nằm trong `data/_ci_test/work` (18 MB tổng `data/_ci_test`).
