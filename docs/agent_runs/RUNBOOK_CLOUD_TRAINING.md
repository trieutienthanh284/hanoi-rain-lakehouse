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

---

# P4B — Mô hình v1 trên GitHub Actions (job `p4b_smoke`, `p4b`)

Định nghĩa đánh giá (D93): CV 2023-05..2024-10 với IMERG Final; test 2025-05..09 với Final, chạy **một lần** khi có `final = true` và người dùng đã duyệt; tháng 10/2025 (IMERG Late) là kiểm tra phụ, báo cáo riêng. Laptop không chạy dữ liệu thật hay huấn luyện (D94).

## P4B.1 Các lần chạy theo thứ tự

| # | Form (job / part / final) | Chạy khi | Song song? | Ước lượng phút runner |
|---|---|---|---|---|
| 1 | `p4b_smoke` / `1/1` / `false` | ngay sau khi push | — | 8–15 |
| 2a | `p4b` / `1/3` / `false` | sau khi agent xác nhận smoke PASS | chạy cùng lúc 2b, 2c | 110–170 mỗi phần |
| 2b | `p4b` / `2/3` / `false` | như 2a | có | 110–170 |
| 2c | `p4b` / `3/3` / `false` | như 2a | có | 110–170 |
| 3 | `p4b` / `1/1` / `true` | chỉ sau khi người dùng đọc báo cáo OOF và agent đã ghi `final_approved = true` | — | 15–30 |

Phương án tuần tự thay cho 2a–2c: `p4b` / `1/1` / `false`, chạy lại cùng input khi hết 320 phút (≈ 2 lần, tổng 330–480 phút).

Các phần song song tự chờ nhau ở mỗi giai đoạn (đọc marker trên R2 mỗi phút, tối đa 120 phút). Nếu log có `waiting_for_other_parts` hoặc `time_budget_reached`, chạy lại **đúng phần đó** với cùng input: job tiếp tục, không làm lại đơn vị đã có marker.

## P4B.2 Kiểm tra log smoke (tách biệt với job thật, D96)

Trong bước **Train** của lần chạy `p4b_smoke`:
1. Dòng `{"event": "plan", "job": "p4b_smoke", "run_id": "…", "gold_id": "…"}`: `run_id` phải khác run của `p4b`, `gold_id` khác Gold của `p4b` (agent in sẵn cả hai giá trị).
2. Không dòng nào chứa `OOF_DONE` (tìm bằng ô "Search logs").
3. Dòng cuối `{"event": "end", … "oof_complete": true, "final": "done"}`: smoke chạy hết, kể cả đơn vị final trên ngày giả-test 2024.
4. Artifact `train-p4b_smoke-…` chỉ có thư mục `p4b_smoke/<run_id>/`.

Trên R2, smoke chỉ ghi được `lake/v1/runs/p4b_smoke/` và `lake/v1/gold/met_training/table_*_v1/gid=<gold id của smoke>/`; mọi khóa khác bị từ chối (`KeyRefused`).

## P4B.3 Sau khi chạy

Báo agent "p4b_smoke finished" / "p4b OOF finished" / "p4b final finished". Agent lấy kết quả bằng `python -m src.models.fetch_run --job <job>`, viết `REPORT_P4B.md`. Sau phần OOF, agent ghi `lake/v1/reports/p4b/OOF_DONE.json` với `final_approved = false`; chỉ khi người dùng nói chạy test cuối, agent mới ghi lại với `--approve-final`. `--final` bị từ chối (exit 2, `final_refused`) nếu tệp không có, chưa duyệt, hoặc thuộc run khác (mã đã đổi).

# P4B sau lượt final — thay đổi cloud một lần (bước 4 của kế hoạch, 2026-11-21)

Nội dung thay đổi: (i) sửa race REPORT_P4B §3.5 (`commit_unit`: chỉ part ghi được marker mới ghi `summary.json` / `done.json`; marker đã có thì không tải output lên); (ii) step summary ghi ra `cloud_out/step_summary_<job>.md`, workflow chỉ chép vào trang GitHub khi bước Train thành công, ngược lại ghi một dòng "Train step failed, no summary"; (iii) `periods.json` ghi nguồn nhãn theo kỳ; (iv) job mới `p4b_post` chỉ đọc run `p4b/c0df12477947` (run id cố định, sha256 theo marker), không huấn luyện, không chạy lại hay chấm lại test, chỉ ghi `lake/v1/runs/p4b_post/<run_id>/`.

Run id sau thay đổi: `p4b` → `02be2f85f454` (**không chạy**: sẽ là một lượt CV mới ≈ 165 phút; `--final` bị từ chối vì `OOF_DONE.json` gắn với `c0df12477947`), `p4b_smoke` → `10161c20403f`, `p4b_post` → `fe6059a77ff1`. Gold id không đổi (`0803d9d8ed72`; smoke `790a59df4110`).

## Thứ tự chạy (form: job / part / final / race_drill)

| # | Form | Khi nào | Song song? | Ước lượng phút runner |
|---|---|---|---|---|
| 1a | `p4b_smoke` / `1/2` / `false` / `false` | ngay sau khi push | dispatch 1a và 1b cách nhau < 1 phút | 8–15 |
| 1b | `p4b_smoke` / `2/2` / `false` / `false` | cùng lúc 1a | có | 8–15 |
| 2 | `p4b_smoke` / `1/1` / `false` / `true` | sau khi cả 1a, 1b xanh | — | 3–8 |
| 3 | `p4b_post` / `1/1` / `false` / `false` | sau bước 2 (độc lập; có thể chạy song song với 1–2) | — | 5–15 |

Kiểm tra race (rẻ, chỉ trên tiền tố smoke):
- 1a/1b: hai part cùng thấy các gate sẵn sàng; part nào ghi marker trước thì thắng. Cả hai run phải **xanh** (mã cũ: part thua dừng với `ObjectConflict` trên `summary.json`). Trong log Train, các dòng `marker_exists_skip_upload` hoặc `marker_lost` (nếu có) cho biết race thực sự xảy ra. `marker_lost` kèm `mismatch` > 0 → exit 3, báo agent.
- 2 (`race_drill = true`): chạy lại mọi gate và đơn vị final của run smoke đã xong. Mỗi đơn vị phải ra `skipped`, `summary.json` / `done.json` và mọi output giữ nguyên byte. Dòng log `{"event": "race_drill", "ok": true, …}`; `ok: false` → exit 3.
- Agent: `python -m src.models.fetch_run --job p4b_smoke --run-id 10161c20403f --verify` (mọi output khớp sha256 trong marker, không tệp thừa, có `summary.json` và `done.json`).

Sau bước 3: báo agent "p4b_post finished". Agent chạy `python -m src.models.fetch_run --job p4b_post --run-id fe6059a77ff1 --verify` (chỉ chép vào `reports/p4b_post/cloud/…`, không cài gì vào `models/`).

Ghi chú: đơn vị `feat-*` của run smoke mới ghi lại các phân vùng Gold smoke (cùng gid `790a59df4110`, cùng nội dung) như mọi run smoke trước (hành vi có sẵn, chỉ trong tiền tố Gold của smoke).
