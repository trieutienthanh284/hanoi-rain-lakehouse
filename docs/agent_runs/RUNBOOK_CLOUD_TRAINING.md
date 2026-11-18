# RUNBOOK — Huấn luyện mô hình pilot trên GitHub Actions (P4C)

Tài liệu cho người vận hành repo công khai `hanoi-rain-lakehouse`. Agent chỉ chuẩn bị mã; **bạn** là người push và bấm chạy job. Không cần thêm secret mới: workflow dùng lại 8 secret sẵn có (`R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET`, `EARTHDATA_TOKEN`, `GSMAP_FTP_HOST`, `GSMAP_FTP_USER`, `GSMAP_FTP_PASSWORD`); thực tế chỉ 4 secret R2 được dùng.

## 1. Điều kiện trước khi chạy

1. Bảng Gold và mô hình pilot đã có trên R2 (P4C.1 đã làm 2026-11-18): `lake/v1/data/gold/met_training/` và `lake/v1/models/` (215 đối tượng, ≈ 340 MiB). Nếu sau này dựng lại bảng Gold, đồng bộ lại:
   `.venv/Scripts/python.exe -m src.lake.r2_sync --only data/gold/met_training --only models` rồi `--verify`.
2. Đã chép mã vào thư mục công khai và push (các lệnh git ở `REPORT_P4A.md`, mục P4C).

## 2. Chạy một job

GitHub → repo `hanoi-rain-lakehouse` → tab **Actions** → workflow **train** → **Run workflow**, điền:

| Ô nhập | Giá trị | Ý nghĩa |
|---|---|---|
| `job` | `p4a6` | P4A.6: 9 mô hình bộ B (có NWP), 3 mô hình A_no2023 ở 0,5 mm/h, rồi so sánh công bằng A-2024 vs B-2024 + baseline "NWP thô" và phán quyết bộ B |
| | `p4a7` | P4A.7: leave-one-season-out bộ A (27 mô hình), rồi chỉ số ngoài mẫu theo mùa và độ phân tán |
| | `p4b` | dành chỗ cho Phase 4B — hiện dừng ngay với mã 2 (chưa có mã, xem D77) |
| `part` | `1/1` | chạy mọi đơn vị trong một job. Muốn song song: chạy `1/3`, `2/3`, `3/3` cùng lúc (mỗi phần nhận một khối đơn vị; phần nào thấy đủ marker sẽ tự chạy bước tổng hợp `final`) |
| `final` | để trống | chỉ dùng cho `p4b` sau này |

Khuyến nghị: chạy `p4a6` với `1/1` trước (ngắn), sau đó `p4a7` với `1/1` hoặc `1/3`…`3/3`.

## 3. Thời gian ước lượng (runner Linux 4 vCPU, 16 GB RAM; LightGBM 2 luồng như cục bộ)

| Job | Đơn vị | Ước lượng | Ghi chú |
|---|---|---|---|
| cài đặt | — | 2–4 phút | pip cache từ lần thứ hai |
| `p4a6` | 12 mô hình + final | 15–30 phút | cục bộ: 9 mô hình bộ A mất 11 phút; bước final có bootstrap 1.000 lần |
| `p4a7` | 27 mô hình + final | 40–90 phút (một phần); 20–35 phút/phần nếu chia 3 | tập tune 0,7–1,0 triệu dòng mỗi khung |

Tổng ≈ 1–2 giờ phút-runner. Repo **công khai** nên phút runner chuẩn miễn phí; giới hạn một job 6 giờ (workflow đặt `timeout-minutes: 350`, mã tự dừng nhận đơn vị mới sau 320 phút).

## 4. Đọc kết quả trên GitHub

- Mở lần chạy → job **train** → phần **Summary**: bảng ngắn (số đơn vị xong trong lần chạy, số còn thiếu; khi bước final xong: hiệu B − A theo khung/ngưỡng và phán quyết, hoặc CSI theo mùa giữ lại).
- Log từng dòng là JSON: `plan` (đơn vị sẽ chạy), `unit_done`, `final_start`, `end` (`final: done` hoặc `waiting`).
- Artifact `train-<job>-<run_id>-<attempt>` (giữ 7 ngày) chứa `cloud_out/<job>/<run_id>/` — bản sao mọi tệp đã ghi lên R2.
- Trên R2: `lake/v1/runs/<job>/<run_id>/` (`files/`, `markers/`, `summary.json`, `done.json`) và con trỏ `lake/v1/runs/<job>/latest.json`.

## 5. Khi job dừng vì giới hạn thời gian hoặc lỗi mạng

Chạy lại **đúng job và part đó** (Run workflow với cùng giá trị). Các đơn vị đã có marker trên R2 được bỏ qua, job tiếp tục từ đơn vị đầu tiên chưa xong. Không cần xóa gì. Nếu mã thay đổi giữa hai lần chạy, `run_id` đổi và job bắt đầu một run mới (run cũ vẫn giữ nguyên trên R2).

Lỗi thường gặp:
- `missing environment variables` → secret R2 chưa đặt trong repo (Settings → Secrets and variables → Actions).
- Cài `lightgbm` lỗi biên dịch → không có wheel cho Python 3.14 trên Linux; báo agent (phương án: đổi phiên bản `lightgbm` trong `requirements-train.txt`).
- `set HFL_P2B_EXCLUDE_SOURCES=imerg_late` → workflow đã đặt sẵn; chỉ gặp khi chạy tay.

## 6. Báo agent lấy kết quả

Khi job báo `final: done`, viết cho agent: **"p4a6 finished"** (hoặc "p4a7 finished"). Agent sẽ chạy
`.venv/Scripts/python.exe -m src.models.fetch_run --job p4a6` — chỉ đọc R2, sao toàn bộ run vào `reports/p4a6/cloud/<run_id>/`, cài mô hình/dự báo vào `models/` và `data/gold/met_training/pilot_v0/preds/` (tệp cục bộ khác nội dung được chuyển sang `_archive/fetch_run/…`, không xóa), rồi cập nhật `REPORT_P4A.md` (P4A.6, P4A.7), dựng replay Yagi từ fold bỏ-2024 và nhập bản ghi run vào MLflow cục bộ.

## 7. Chạy thử cục bộ (không cần R2)

```
set HFL_P2B_EXCLUDE_SOURCES=imerg_late
python -m src.models.cloud_train --job p4a7 --plan --store local:<thư mục>     # chỉ liệt kê đơn vị
python -m pytest -q tests/test_cloud_train.py
```
Không huấn luyện thật trên laptop (RAM trống ≈ 1 GB).
