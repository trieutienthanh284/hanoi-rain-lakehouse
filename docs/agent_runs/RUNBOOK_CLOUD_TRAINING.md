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

# P4D — Model v2: "cắt" (Model 1) và "trọng số" (Model 2) trên GitHub Actions (job `p4d_smoke`, `p4d`)

Một job làm trọn P4D.3 → P4D.7: Gold `table_v2` (47 pixel thành phố + 177 pixel vành đệm, lề ≥ 7 px, 2023-05 … 2025-10; D181), LOSO 3 fold, M0 → M1a → M1b, lưới Model 2 (8 thiết lập), so sánh, rồi **dừng** ở `COMPARE_DONE.json`. Không có `final`, không có `race_drill`, không đọc 2022. Thiết kế: `configs/model_v2.yaml`, REPORT_P4D §1–§2, D169–D183.

Định danh (tính từ mã đã publish; đổi nếu mã đổi): `p4d` run `650b5ed0d051`, gold `a4a6329bcf13`; `p4d_smoke` run `72224a82daba`, gold `0f172efca37b` (D185; lượt smoke `793004743201` hỏng, không tiếp tục).

## P4D.1 Thứ tự chạy (form: job / part / final / race_drill)

| # | Form | Khi nào | Ước lượng |
|---|---|---|---|
| 1 | `p4d_smoke` / `1/1` / `false` / `false` | ngay sau khi push | ≈ 30 phút runner |
| 2 | agent kiểm tra smoke (`fetch_run --job p4d_smoke --run-id 72224a82daba --verify`) | sau khi 1 xanh | cục bộ, chỉ đọc |
| 3 | `p4d` / `1/13` … `13/13` / `false` / `false` | sau khi agent báo smoke PASS; dispatch 13 phần cách nhau < 5 phút | ≈ 250 phút phần chậm nhất (≈ 2 300 phút runner tổng) |

- **Giữ đúng 13 phần cho cả run.** Lần chạy đầu ghi `parts.json`; mọi lần chạy cùng run với số phần khác bị từ chối (exit 2, `refused … split into 13 parts`). Khi một phần dừng (`time_budget_reached`, `waiting_for_other_parts`, lỗi mạng), chạy lại **đúng phần đó** (`i/13`): đơn vị có marker được bỏ qua.
- Các phần chờ nhau ở mỗi giai đoạn (đọc marker R2 mỗi phút, tối đa 150 phút): 0 preflight → 1 feat (18 tháng) → 2 g2 → 3 base + M0 + M1a → sel-m1a → 4 M1b → sel-m1b → 5 Model 1 (0,5/7,6) + lưới Model 2 (144 đơn vị, phần lớn thời gian) → sel-m2 → 6 Model 2 0,5 → 7 cmp-main, cmp-detail → 8 compare-done.
- `--max-minutes 320`, `timeout-minutes 350` như P4B. Gói GitHub miễn phí cho 20 job song song; nếu ít hơn 13, các phần xếp hàng: phần đến sau có thể hết `--wait-minutes` khi chờ ở gate → chạy lại đúng phần đó. Không đổi số phần.

## P4D.2 Dòng log cần kiểm tra (bước **Train**)

1. `{"event": "plan", "job": "p4d_smoke", "run_id": "cd317e638365", …, "gold_id": "daf4deb2b58e"}` (smoke) / `"job": "p4d", "run_id": "fc90d9b3362e", …, "gold_id": "24df8ebdbbde"` (thật).
2. `{"event": "unit_done", "unit": "preflight", …, "r2_total_gb": …, "projected_total_gb": …, "ring_pixels": 177, "ring_margin_px": 7}` (smoke: `ring_pixels` 4): `projected_total_gb` ≤ 6,0 (ước lượng sát ngưỡng, ≈ 5,9). Nếu vượt, unit lỗi (`preflight FAIL`), không có gì được dựng; báo agent.
3. `{"event": "unit_done", "unit": "g2", …, "violations": 0, "a1_equal": true, "ring_pixels": 177}` (smoke: 4). Tóm tắt `ring_coverage` (`ring_coverage_min` ≈ 0,84, `ring_coverage_mean` ≈ 0,96, `ring_pixels_truncated` 144; smoke theo 4 pixel được chọn) nằm trong marker / `reports/p4d/g2/summary.json` và `truncation_city_vs_ring.csv`.
4. `{"event": "unit_done", "unit": "fit-…", …, "rss_gb": …}`: mọi `rss_gb` ≤ 10 (ước ≈ 5,7 cho đơn vị có vành; unit tự lỗi nếu vượt).
5. `sel-m1a` (`kept`), `sel-m1b` (`kept`, `chain_end`, `ring`, `cut`), `sel-m2` (`chosen`).
6. Dòng cuối mỗi phần `{"event": "end", …, "lost": 0, "output_mismatch": 0, …}`; ở phần thắng `compare-done`: `"compare_complete": true`. Exit 3 hoặc `output_mismatch` > 0 → báo agent, không chạy lại.
7. Không dòng nào chứa `OOF_DONE` hay `final_refused`; các `unit_start` `feat-…` chỉ có tháng 2023-05 … 2025-10 (smoke: 2023-05, 2024-10, 2025-05, 2025-10). Dòng `FINAL: false` trong phần env của bước là bình thường.

Trên R2, job chỉ ghi được `lake/v1/runs/<job>/` và `lake/v1/gold/met_training/table_v2/gid=<gold id>/` (khóa khác: `KeyRefused`). Báo agent "p4d_smoke finished" rồi "p4d finished"; agent chạy `python -m src.models.fetch_run --job p4d --run-id 84679b7ec0cf --verify` (chỉ chép vào `reports/p4d/cloud/…`) và viết phần kết quả của REPORT_P4D.

## P4E.1 Smoke `p4e_smoke` (form: job / part / final / race_drill)

| # | Form | Khi nào | Ước lượng |
|---|---|---|---|
| 1 | `p4e_smoke` / `1/2` / `false` / `true` | ngay sau khi push | ≈ 38 phút (phần chậm nhất) |
| 2 | `p4e_smoke` / `2/2` / `false` / `true` | dispatch ngay sau 1 (< 1 phút), để hai phần chạy **song song** | ≈ 16 phút; tổng ≈ 54 phút runner |
| 3 | (agent) `python -m src.models.fetch_run --job p4e_smoke --run-id d62d80e2f956 --verify` | sau khi cả hai xanh | cục bộ, chỉ đọc |

- Phải dùng **2 phần**: nhóm `concurrency` của workflow là `job + part`, nên hai lần `1/1` sẽ xếp hàng chứ không chạy song song. Với `1/2` và `2/2`, hai phần cùng thấy gate `cmp` / `compare-done` sẵn sàng → race thật trên R2; cộng thêm drill trong tiến trình (`race_drill = true`, hai luồng cùng ghi một gate với byte khác nhau).
- Mỗi phần chờ tối đa 150 phút cho gate mở đầu (preflight, feat, g2), rồi chỉ 5 phút cho gate kết (cmp, compare-done): phần hết việc thì thoát. Phần không chạy được gate kết vì phần kia đang giữ claim sẽ ghi `claim_yielded` và kết thúc với **exit 0** (không còn exit 3 như D162 / D186).
- Giữ đúng 2 phần cho cả run (`parts.json`).

## P4E.2 Dòng log cần kiểm tra (bước **Train**)

1. `{"event": "plan", "job": "p4e_smoke", "run_id": "d62d80e2f956", …, "gold_id": "f889057f41ac", "prereg_sha": "58eb5a7e…"}`; `{"event": "estimate", "parts": 2, "part_minutes": […], …}`.
2. `feat-…`: `"reused": false` ở lần đầu (Gold smoke mới, 4 tháng), `unit_done … "violations": 0` ở `g2`.
3. `fit-…`: `median_rounds`, `rss_gb`, `work_grows`; mọi `rss_gb` ≤ 10.
4. `claim_won` (một lần mỗi đơn vị) và, ở phần còn lại, có thể `claim_yielded` / `marker_exists_skip_upload`; **không** có `marker_lost`, `replaced_unmarked_outputs` hay `single-writer breach`.
5. `{"event": "race_drill_single", "ok": true, "states": {"A": "won", "B": "yielded"} (hoặc ngược lại), "final_equals_marker": true, "loser_exit_code": 0, …}`.
6. Dòng `end`: `"lost": 0, "output_mismatch": 0`; ở phần thắng `compare-done`: `"compare_complete": true`. Nếu `cmp` lỗi với `rounds check FAIL` (luật pre-registered D189): **dừng**, báo agent, không chạy lại.

Trên R2, job chỉ ghi được `lake/v1/runs/p4e_smoke/` và `lake/v1/gold/met_training/table_v2*/` (khóa khác: `KeyRefused`).

### P4E.3 Smoke lần 2 sau D196 (run `048a2def9cf8`, gold smoke `f889057f41ac` dùng lại)

Cùng ngày, cùng form như P4E.1 (2 phần, `race_drill = true`), run id mới vì mã đổi:

| # | Form | Ước lượng (mô hình D191, hằng số cỡ đầy đủ) |
|---|---|---|
| 1 | `p4e_smoke` / `1/2` / `false` / `true` | ≈ 45 phút (thực tế smoke trước ≈ 3–4 phút) |
| 2 | `p4e_smoke` / `2/2` / `false` / `true`, dispatch ngay sau 1 | ≈ 18 phút; tổng ≈ 62 phút runner |
| 3 | (agent) `python -m src.models.fetch_run --job p4e_smoke --run-id 048a2def9cf8 --verify` | cục bộ, chỉ đọc; hình được chép vào `reports/p4e/figs/p4e_smoke_048a2def9cf8/` |

Dòng log thêm so với P4E.2:
- `dry-full-2024-07` (đơn vị D196): đọc **chỉ đọc** một tháng Gold đầy đủ `gid=a4a6329bcf13/season=2024/month=07/part.parquet`, `unit_done … "dry_scene_share": [x, x, x]` (F0/F30/F60).
- `fit-m1ds10-…`: `rows_saved` > 0 (định nghĩa cảnh khô D196).
- `figs` (gate kết, sau `cmp`): `"png": n` (≥ 6), `replay_ok`, `replay_skipped` (Yagi / Matmo không nằm trong fold smoke → bỏ qua, có ghi lý do).
- Gate kết nay là `cmp` → `figs` → `compare-done`.


### P4E.4 Smoke lần 3 sau D198 (run `a98f99cc8388`, gold smoke mới `620730fc8d8d`, vành đủ 177 pixel)

Cùng ngày smoke, 2 part, `race_drill = true`. Smoke này chạy mọi đường mã của job `p4e`: Gold smoke dựng lại với 224 pixel (4 đơn vị `feat`), NWP bộ B cho 224 pixel (`nwp-2024-10`, `nwp-2025-07`, `nwp-2025-10`), fit M0 / Model 1 / R-near / W-out / hai tầng (fold 2024) và v1 (B) / Model 1-B (fold 2025, chỉ 2,5 mm/h, khối CV trong theo ngày UTC, D199), `dry-full-2024-07`, gate `cmp-diag` → `cmp-2stage` → `cmp` → `figs` → `compare-done`.

| # | Form `train` (job / part / final / race_drill) | Ước lượng D191 (hằng số cỡ đầy đủ, smoke thực tế nhỏ hơn nhiều) |
|---|---|---|
| 1 | `p4e_smoke` / `1/2` / `false` / `true` | ≈ 34 phút |
| 2 | `p4e_smoke` / `2/2` / `false` / `true`, dispatch ngay sau 1 | ≈ 103 phút; tổng ≈ 137 phút runner |
| 3 | (agent) `python -m src.models.fetch_run --job p4e_smoke --run-id a98f99cc8388 --verify` | cục bộ, chỉ đọc |

Dòng log cần kiểm thêm: `nwp-…` (`rows`, `pixels` 224, `points` 81, `nan_share_max`); `g2` (`set_b_inner_cv_ok true`, `nwp_partitions 3`); `fit-rnear-…` (`train_rows` < Model 1); `cmp-diag` (`proposed`), `cmp-2stage` (`use_two_stage`), `cmp` (`shadow_gate_complete false` là đúng trong smoke: không có fit bộ B ở 7,6).

### P4E.5 Job đầy đủ `p4e` (D198; CHỈ dispatch sau khi người dùng duyệt smoke `a98f99cc8388`)

Run `ae8fd71879ba`, Gold `a4a6329bcf13` (tái dùng), NWP `table_v2_nwp/gid=a4a6329bcf13/nid=e72f7fd94ad0/` (dựng mới, 12 tháng). 193 đơn vị: 129 fit, 12 `nwp`, 18 `dry-full-*`, 9 `base`, 18 `feat` (tái dùng), 7 gate. Đơn vị được gán cho part bằng LPT theo ước lượng D191 (mọi part phải dùng cùng n = 7).

| Part | Ước lượng (phút) |
|---|---|
| 1/7 … 7/7 | 196,1 (part chạy gate kết: ≈ 120 + 76), 126,8, 126,9, 126,0, 125,8, 126,0, 125,8 |
| Tổng | ≈ 953 phút runner; part chậm nhất ≤ 200 (= 0,8 × 250); `--max-minutes 320` giữ làm chặn |

Dispatch 7 form `p4e` / `i/7` / `false` / `false` liền nhau. Part hết việc thì thoát; part chạy cuối làm gate kết. Nếu part nào hết giờ (`time_budget_reached`), dispatch lại đúng part đó (tiếp tục được). Sau run: agent `python -m src.models.fetch_run --job p4e --run-id ae8fd71879ba --verify`, rồi báo `compare/cost.csv` dự báo/thực tế (đơn vị lệch > 1,5× được đánh dấu).

## P4FG — vòng mô hình cuối (TASK_P4F_P4G_FINAL_MODEL.md)

Mọi job ghi R2 chỉ dưới tiền tố đã duyệt (CLAUDE.md §8, P4F/P4G): `runs/<job>/`, `gold/met_training/table_v3*/` (p4fg, p4fg_smoke), `bronze/openmeteo_conv/` + `silver/nwp_conv/` (nwpconv_backfill), `serving/replay/pkg=2026-11/` (p4fg_final). Khóa khác → `KeyRefused`. Gold `table_v2` (gid `a4a6329bcf13`) và NWP bộ B (`nid=e72f7fd94ad0`) chỉ được **dùng lại**; thiếu phân vùng → đơn vị lỗi, không dựng.

### P4FG.1 Thứ tự dispatch sau S1 (form: job / part / final / race_drill)

| # | Form | Khi nào | Ước lượng |
|---|---|---|---|
| 1 | `nwpconv_backfill` / `1/1` / `false` / `false` | ngay sau khi push | 43 khối × ≥ 180 s ≈ 130–150 phút, 1 runner (tải tuần tự; run `9900e8d4f8af`) |
| 2 | `p4fg_smoke` / `1/2` / `false` / `true` | **sau khi #1 xanh** (smoke cần Silver `nwp_conv` của 2024-09, 2024-10, 2025-07, 2025-10) | ước ≈ 97 phút (hằng số bảo thủ; run `db0c4d8a01db`) |
| 3 | `p4fg_smoke` / `2/2` / `false` / `true` | dispatch ngay sau #2 (< 1 phút) | ≈ 51 phút; tổng ước ≈ 148 phút runner |
| 4 | (agent) `python -m src.models.fetch_run --job nwpconv_backfill --run-id 9900e8d4f8af --verify` và `python -m src.models.fetch_run --job p4fg_smoke --run-id db0c4d8a01db --verify` | sau khi #1–#3 xanh | cục bộ, chỉ đọc |

Dòng log cần kiểm (bước **Train**):
- `nwpconv_backfill`: `plan` (`blocks 43`, `first ["2024-05-01","2024-05-13"]`, `last [... "2025-10-31"]`); mỗi `unit_done block-…` có `fetched true` (lần đầu), `dq_fail 0`. Nếu thấy `stop_429` (exit 4): **dừng**, không chạy lại, báo agent. Chạy lại sau lỗi khác: tiếp tục từ khối chưa có Bronze.
- `p4fg_smoke`: `plan` (`gold_id a4a6329bcf13`, `conv_id …`, `prereg_sha fbc7ec89…`); `feat-*`, `nwp-*` `reused true`; `conv-*` (`rows`, `nan_share_max`), `orc-*` (`oracle true`); `g2` (`conv_g2_violations 0`, `set_b_inner_cv_ok true`); `dtcheck` (`verdict`, `pooled_ratio`); `hp-select` (`chosen`); `cmp` (`model2`, `steps`, `union_adopted`, `dt_verdict`); `figs` (`p4e_relabel_ok`); `finalfit-smoke`, `replay-smoke` (`verified 3`); `race_drill_single` `ok true`; dòng `end` `lost 0`, `output_mismatch 0`.
- Nếu `dtcheck` cho `verdict` khác `ok`: smoke vẫn chạy xong (ghi lại), nhưng agent **dừng ở S2** và báo (P4F.0).

### P4FG.1b Smoke lại sau sửa S2 (D218, 2026-11-28)

Smoke `db0c4d8a01db` (S2) tìm thấy 2 lỗi mã, đã sửa: vẽ lại hình P4E đọc sai thư mục (`p4e_relabel_ok false`, D218-1) và manifest REPLAY chưa ghi độ dịch cửa sổ 30 phút của D217-5 (D218-2). Mã đổi nên run id đổi, và TASK yêu cầu smoke trước mọi run đầy đủ. Vì vậy chạy lại smoke trước `p4fg`. Smoke mới dùng lại Gold, NWP và `table_v3_conv` (`cid=4e5f3ddb0727` không đổi), không ghi Bronze/Silver.

| # | Form | Khi nào | Ước lượng |
|---|---|---|---|
| 0 | lệnh git trong REPORT_P4FG §S2.10 | trước tiên | — |
| 1 | `p4fg_smoke` / `1/2` / `false` / `true` | sau push | ước (hằng số bảo thủ) 96,6 phút; thực tế lần trước ≈ 21 phút/part |
| 2 | `p4fg_smoke` / `2/2` / `false` / `true` | ngay sau #1 | ước 51,3 phút |
| 3 | (agent) `python scripts/p4fg/fetch_gated.py --job p4fg_smoke --run-id 8a88b62a0cd7 --verify` | sau #1–#2 xanh | cục bộ, chỉ đọc, cổng RSS 0,4 GB |

Dòng log cần kiểm thêm: `figs` có `p4e_relabel_ok true`; `replay_pkg/2025-07-03/_manifest.json` có `icon_windows.operational.window_shift_min = 30`.

### P4FG.2 Job đầy đủ `p4fg` (CHỈ dispatch sau khi smoke `8a88b62a0cd7` xanh và agent đã kiểm; S2)

Run `82b2df2b97bb` (trước sửa D218 là `63baa3224b58`; đổi nếu mã đổi tiếp). 297 đơn vị: 228 fit (B0 18, lưới 8 × 12, đối lưu trên B0 36, đối lưu trên mô hình tinh chỉnh 36 [alias khi fold chọn g1], oracle B 12, mục tiêu vùng 12, v1 (B) 18), 12 `conv`, 12 `orc`, 6 `base`, `dtcheck`, `fssv1`, 18 `feat` + 12 `nwp` (dùng lại), 7 gate. Gán part bằng LPT theo ước lượng (mọi part cùng n).

| Phương án | Part | Part chậm nhất (ước) | Runner (ước) |
|---|---|---|---|
| đề xuất (0,8 × 250) | 10 | 199,7 phút | ≈ 1 148 phút |
| tối thiểu (≤ 250) | 7 | 242,8 phút | ≈ 1 120 phút |

Hằng số ước lượng = thời gian thực P4E × 1,5 (Model 1-B 109 s/fit, v1 (B) 23 s/fit) × hệ số lá; P4E thực tế bằng 0,61 × ước. Dispatch 10 form `p4fg` / `i/10` / `false` / `false` liền nhau. Part xong việc thì thoát; mọi part chờ `hp-select` (≤ 150 phút) trước các fit giai đoạn 5. Winner của `compare-done` ghi `SELECTION.json` + `COMPARE_DONE.json`. Sau run: agent `python scripts/p4fg/fetch_gated.py --job p4fg --run-id 82b2df2b97bb --verify`.

### P4FG.3 Job `p4fg_final` (sau chọn lựa; S3)

Từ chối chạy khi chưa có `lake/v1/runs/p4fg/<run>/SELECTION.json` (run nguồn = `latest.json` của p4fg, hoặc `final_job.source_run_id` trong `configs/model_p4fg.yaml`), khi marker đến từ smoke, và không xuất REPLAY nếu phán quyết P4F.0 ≠ `ok`. Đơn vị: fit 0,5 LOSO của Model 2 (nếu khác Model 1-B), 9–18 final fit, 4 sự kiện REPLAY, `replay-publish` (put-if-absent dưới `serving/replay/pkg=2026-11/`, tải lại kiểm sha256, xoá tệp scratch), `final-done`. Form: `p4fg_final` / `1/1` / `false` / `false`. Mô hình chỉ cài cục bộ bằng `fetch_run` vào `models/v2_candidate/` (không bao giờ `models/v1/`).
