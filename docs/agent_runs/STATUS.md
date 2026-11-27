# STATUS — U1 publish prep + R2 upload (2026-11-27)

**Trạng thái (2026-11-27, D209–D212): XONG — PASS với 2 điểm NEEDS_REVIEW. DỪNG.** REPORT_U1 §22. Phê duyệt CLAUDE.md §8 dòng 96 có. Danh sách `publish_paths_u1.txt` 29 tệp (+29 vào `publish_paths_p3.txt`). Quét bản quyền: 372 đoạn ≥ 60 ký tự; `REPORT_U1.md` 7 lần trùng (dòng 310, 314, 1314) → **không publish (NEEDS_REVIEW)**; các tệp khác 0. Sync 29 thêm → 29 unchanged; test công khai từng tệp: event_catalog 24 passed / 2 skipped (cần `data/`), flood_indicator 36, auto_review 29, ai_review 8 passed, cả khi chặn gói ngoài `requirements-ci.txt`. NEEDS_REVIEW: module pipeline cần 7 gói ngoài `requirements-ci.txt` và 4 module `flood_evidence_*` chưa publish; `tests.yml` chưa chạy test U1. R2: 5 đối tượng dưới `lake/v1/gold/event_catalog/snapshot=2026-11-27/` (events 215 dòng, event_days 45, review_overlay 214, schema.json, `_manifest.json`), tải lại khớp sha256 5/5. STAGED_SAME (11). Điểm tiếp tục: người dùng chạy 3 lệnh `git -C "D:\hanoi-rain-lakehouse-public"` (REPORT_U1 §22.6) và quyết định về REPORT_U1 (bản công khai rút gọn hay không publish) và `tests.yml`.

---

# STATUS — Phase 4E v2 challenger (`TASK_P4E_MODEL_V2_CHALLENGER.md`)

Nhánh: `feat/met-dataset`. Tiến độ: `P4E_PROGRESS.md`. Báo cáo: `REPORT_P4E.md`.

**Trạng thái (2026-11-24, D200): run `p4e` `ae8fd71879ba` XONG — DỪNG tại P4E.6, hub quyết định shadow mode cho Model 1-B.** REPORT_P4E §6. Verify PASS (193 marker, 355 output, 0 lệch/thiếu/thừa, 193 claim thế hệ 1, COMPARE_DONE có); part 5/7 (#43) chạy gate kết; race thật g2/cmp-2stage/cmp xử lý đúng. Model 1-B − v1 (B): log-loss 2,5 −0,0021…−0,0026*, CSI 7,6 +0,019…+0,025*; use_two_stage false; R-near/W-out không đề xuất; Model 1 không thiếu tự tin hơn M0. **Shadow gate PASS 15/15**, Matmo không gắn cờ. Runner ≈ 578 phút vs ước 953 (101 đơn vị ước quá cao > 1,5×). NEEDS_REVIEW: RSS cục bộ 0,544 GB > 0,4 (script KTC sự kiện); thiếu FSS của v1 trên dữ liệu P4B; persistence ngang/hơn ở CSI 7,6 F0. STAGED_SAME (11). Điểm tiếp tục: hub đọc REPORT_P4E §6.13 và quyết định; không có run cloud nào đang chờ.

**Cập nhật (2026-11-23): smoke `a98f99cc8388` verify ok (32 marker, 107 output, 0 lệch). Drill PASS; race thật đầu tiên (`cmp-diag`, `figs`) được xử lý đúng, byte trùng. Bộ B: NWP 224 px, NaN 0, CV trong theo ngày dùng 4 khối, v1 (B)/Model 1-B có OOF. R-near 90 px, W-out trọng số đúng. Hai tầng vi phạm 0,000. Tỉ lệ khô 2024-07 là 0. Shadow gate `complete = false` chỉ vì thiếu bộ B 7,6. NEEDS_REVIEW: nhãn hình là `v1B`/`Model1-B`/`Model1-A`. Run `p4e` vẫn `ae8fd71879ba`, 7 part. Điểm tiếp tục: người dùng duyệt REPORT_P4E §5 (gồm quyết định có sửa nhãn hình hay không; nếu sửa thì run id đổi) rồi mới dispatch `p4e` 1/7…7/7.**

**Cập nhật (2026-11-23, D198–D199): người dùng duyệt smoke `048a2def9cf8` PASS. P4E.1 không chạy, downsampling tắt; NWP vành phương án A (bảng phụ `table_v2_nwp`); R-near 90 pixel trong config; P4E.5 bỏ. Đã cài: smoke mở rộng (vành 177 px, gid `620730fc8d8d`, NWP 224 px, v1 (B)/Model 1-B fold 2025 ở 2,5, R-near, W-out, hai tầng, dry 1 tháng, gate `cmp-diag`/`cmp-2stage`/`cmp`) và job đầy đủ `p4e` (193 đơn vị, 129 fit, 7 part ≈ 196 phút max, ≈ 953 phút runner) trong `train.yml`. Test dev + công khai passed; sync 7 đổi → 43 unchanged; STAGED_SAME. Điểm tiếp tục: người dùng chạy lệnh git (REPORT_P4E §4.7), dispatch `p4e_smoke` 1/2 và 2/2 (race_drill = true), run `a98f99cc8388` → agent `fetch_run --job p4e_smoke --run-id a98f99cc8388 --verify` → báo cáo → người dùng duyệt. KHÔNG dispatch `p4e` (run `ae8fd71879ba`, 1/7…7/7) trước khi duyệt.**

**Cập nhật (2026-11-23): smoke `048a2def9cf8` verify ok (19 marker, 52 output, 0 lệch); drill PASS; downsampling đúng định nghĩa D196 (smoke 23,5 % t0 khô, bỏ 9 996 dòng); hai tầng vi phạm 0; 6 PNG. NEEDS_REVIEW: tháng 2024-07 đầy đủ 0 % cảnh khô (224 pixel/cảnh) → downsampling gần như không tiết kiệm. Job `p4e` chưa có trong `train.yml`. Điểm tiếp tục: người dùng duyệt REPORT_P4E §3; không bắt đầu P4E.1.**

**Cập nhật (2026-11-23, D196–D197): người dùng duyệt smoke `d62d80e2f956` có điều kiện. Đã sửa định nghĩa cảnh khô (NaN `adv_*`/`upwind_*` chỉ tương thích khô khi mọi cửa sổ mưa thô không NaN và = 0; Gold smoke 23,5 % t0 khô), thêm đơn vị `dry-full-2024-07` (tỉ lệ khô một tháng Gold đầy đủ, đo trên cloud), gate `figs` (hình P4E.6 trên cloud, chép về `reports/p4e/figs/`), cờ chi phí > 1,5×. Test dev + 18 tệp công khai passed; sync 43 unchanged; STAGED_SAME. Điểm tiếp tục: người dùng chạy lệnh git (REPORT_P4E §2.6), dispatch `p4e_smoke` 1/2 và 2/2 (race_drill = true), run `048a2def9cf8` → agent `fetch_run --job p4e_smoke --run-id 048a2def9cf8 --verify` → báo cáo → người dùng duyệt. Không bắt đầu P4E.1 trước đó.**

**Cập nhật (2026-11-23, D194–D195): smoke `d62d80e2f956` verify ok; một-writer + drill PASS; downsampling vô hiệu (0 cảnh khô vì `upwind_*` NaN) → NEEDS_REVIEW, chờ người dùng duyệt sửa định nghĩa (REPORT_P4E §1.5). Không sửa code/config. R-near = 90 pixel (D194). NWP vành: phương án A (81 điểm, không tải thêm).**

**Trạng thái (2026-11-23, D189–D193): P4E.0 XONG (PASS mã + test tổng hợp) — DỪNG trước smoke cloud.** Pre-registration D189 (hash mục `58eb5a7e…`, mã chặn nếu đổi). Sửa D188: claim nguyên tử, một writer, bên thua exit 0; race drill mở rộng (D192). lr 0,06 / N_max 1 500 / ES 75 (D190). Mô hình thời gian s = 167,6 + 31,94·W (D191). Job `p4e_smoke` run `d62d80e2f956` / gold `f889057f41ac`. Test dev + 16 tệp công khai passed; sync 41 unchanged; STAGED_SAME. Mở: R-near "lề 3" mâu thuẫn (NEEDS_REVIEW); NWP vành P4E.4 chưa kiểm. Điểm tiếp tục: người dùng chạy lệnh git (REPORT_P4E §0.7), dispatch `p4e_smoke` 1/2 và 2/2 (race_drill = true) cùng lúc, báo agent → `fetch_run --job p4e_smoke --run-id d62d80e2f956 --verify` → dự báo/thực tế phút, rồi định nghĩa job `p4e` (P4E.1).

---

# STATUS — Phase 4D Model v2 cut vs weight (`TASK_P4D_MODEL_V2_CUT_VS_WEIGHT.md`)

Nhánh: `feat/met-dataset`. Tiến độ: `P4D_PROGRESS.md`. Báo cáo: `REPORT_P4D.md`.

**Trạng thái (2026-11-23, D186–D188): P4D.3 → P4D.7 XONG — DỪNG tại điểm dừng P4D.7, người dùng chọn Model 1 hay Model 2.** REPORT_P4D §3. Run `p4d` `650b5ed0d051` (gold `a4a6329bcf13`), 2 lượt × 13 part. `fetch_run --verify`: 251 marker, 468 output, missing 0, sha_mismatch 6 (`cmp-main`, race part 12/#31) → chấp nhận R2 (lệch ≤ 9,9e-17, D186); `COMPARE_DONE.json` có. M1a (cắt) không giữ; M1b (+ vành) giữ → Model 1 = M0 + vành, không cắt; Model 2 chọn g5 (k 3, α 0, β 0). Cả hai tốt hơn M0 ở 2,5 mm/h (log-loss −0,0012…−0,0016*); Model 2 − Model 1 không phân biệt về log-loss. Khuyến nghị: Model 1. Race còn sót D188 (NEEDS_REVIEW). STAGED_SAME (11). Điểm tiếp tục: người dùng đọc REPORT_P4D §3.11 và chọn; sau đó P4D.9 / bước tiếp theo theo hub.

**Cập nhật (2026-11-23, D184–D185): smoke `793004743201` hỏng ("no usable inner fold": khối 2023-05-13…18 không có sự kiện ≥ 2,5). Đã sửa: khối smoke 2023-05-10…20 / 2024-10-14…19 / 2025-07-01…06 (+ 2025-10 Late), kiểm tra fail-fast lúc lập kế hoạch (≥ 2 khối, ≥ 2 tháng dương lịch) và ở g2 (nhãn). Test dev 9 tệp + công khai 13 tệp passed, sync 36 unchanged, STAGED_SAME. ID mới: `p4d_smoke` `72224a82daba` / gold `0f172efca37b`; `p4d` `650b5ed0d051` / gold `a4a6329bcf13`; 13 part. Điểm tiếp tục: người dùng commit + push thư mục công khai, dispatch `p4d_smoke` 1/1, agent `fetch_run --verify` (REPORT_P4D §2.5).**

**Cập nhật (2026-11-23, D181–D183): vành đệm đổi thành 177 pixel (lề 7 px, lề lớn nhất cho ≥ 150), thêm đặc trưng `ring_coverage`, đặc trưng cụt tính trên phần trong miền; ước lượng lại 5 935 104 dòng/khung, Gold ≈ 1,42 GB, RAM ≈ 5,7 GB, ≈ 2 300 phút runner → 13 part (≈ 248 phút/part). R2 dự phóng ≈ 5,85 GB (sát 6, gate `preflight` kiểm). Test dev + công khai passed, sync 36 unchanged, STAGED_SAME. ID mới: `p4d` `84679b7ec0cf` / gold `a4a6329bcf13`; `p4d_smoke` `793004743201` / gold `68c63e533e62`. Điểm tiếp tục: REPORT_P4D §2.4 (lệnh git, smoke 1/1, rồi `p4d` 1/13…13/13). Các ID và số part trong đoạn dưới đã cũ.**

**Trạng thái (2026-11-22, D169–D180): P4D.3 → P4D.8 ĐÃ CÀI ĐẶT (PASS mã + test tổng hợp) — DỪNG trước mọi lượt chạy cloud.** REPORT_P4D §1. Hub sau P4D.0: bỏ 2022, không nguồn thay thế, không M1-alt / M1c; luật thay thế (a)+(b) đăng ký trước (D169). Gold `table_v2` = 47 pixel thành phố + 33 pixel vành (lề ≥ 10 px để mọi đặc trưng nằm trong miền; NEEDS_REVIEW so với 150–300 của TASK cũ, D171). Ước lượng: 2 119 680 dòng/khung, ≈ 0,51 GB Gold, RAM đỉnh ≈ 2,9 GB, không lấy mẫu con; R2 kiểm bằng gate `preflight` (≤ 6 GB). Job `p4d` (251 đơn vị, 216 fit) và `p4d_smoke`; test: `test_p4d_job` 22, `test_p4d_gold_synthetic` 4, `test_p4d_fit_synthetic` 1 passed; test cũ bị ảnh hưởng passed. Publish: sync 6 thêm / 6 đổi → 36 unchanged; test công khai passed. STAGED_SAME.

Điểm tiếp tục: người dùng chạy 3 lệnh `git -C` (REPORT_P4D §1.9), dispatch `p4d_smoke` 1/1 (run `cd317e638365`, gold `daf4deb2b58e`), báo agent → agent `fetch_run --job p4d_smoke --run-id cd317e638365 --verify` → nếu PASS, người dùng dispatch `p4d` 1/5…5/5 (run `fc90d9b3362e`, gold `24df8ebdbbde`) → agent fetch + viết phần kết quả REPORT_P4D, dừng ở P4D.7 (người dùng chọn Model 1 hay Model 2).

(Trạng thái trước, P4D.0:)

**Trạng thái (2026-11-21, D163–D168): P4D.0 XONG — DỪNG tại điểm dừng P4D.0 (NEEDS_REVIEW), chờ hub.** REPORT_P4D §0. Thống kê 2023–2024: phần cắt Model 1 = 16,8 % số ngày, 7,2 % cửa sổ pixel ≥7,6, 8,2 % sự kiện ≥7,6. CMR: IMERG Final 2022 và Early 2022/2023/2024 đủ 48/ngày. PDIR-Now đủ 2022 + mùa 2023/24 (0,04°, theo giờ, trễ khoảng 6–16 phút, kho bị ghi lại; điều khoản chưa xác minh). CMORPH-2 chỉ từ 2023, 0,25°. JAXA EORC: 404 (chưa trở lại). 68 yêu cầu mạng, ≥10 s/host. STAGED_SAME (11).

Điểm tiếp tục: hub chọn nguồn thay thế (khuyến nghị: PDIR-Now chính, IMERG Early ứng viên thứ hai, CMORPH-2 chỉ kiểm tra phụ) và trả lời REPORT_P4D §0.7 (cửa sổ 2022 đến 10-15 hay 10-31; điều khoản PDIR; IMERG Early song song; cắt theo ngày UTC hay giờ Hà Nội). Sau đó mới làm P4D.1. U1 tạm dừng; U1-R2b đã chạy 2026-11-26 (xem mục U1 bên dưới).

---

# STATUS — Phase U1 Event catalog (`TASK_U1_EVENT_CATALOG.md`)

Nhánh: `feat/met-dataset`. Tiến độ: `U1_PROGRESS.md`. Báo cáo: `REPORT_U1.md`. Quyết định D102–D109, D112–D115 (follow-up), D116–D121 (follow-up 2), D122–D127 (U1.1b), D128–D130 (follow-up 3), D131–D134 (U1-D2), D135–D140 (U1-R).

**U1 DONE (2026-11-20) — REPORT_U1 §17 "Kết luận U1".** Quyết định U1-D1 (IMERG Final/Late 50/100/150) và U1-D2 (GSMaP NOW GC 110/220/270, provisional; GSMaP NOW thô không dùng cho icon) đã ghi vào `configs/flood_indicator.yaml`; `flood_indicator.product_thresholds()` + test; phiếu duyệt Silver A0 (46 bản ghi tập chính) trước. `pytest -q tests` → 242 passed. STAGED_SAME. Quyết định D131–D134. Việc mở (không chặn): duyệt 214 bản ghi Silver pending, bằng chứng cho 23 ngày, chế độ phục vụ P5.

**Trạng thái trước (2026-11-20): HOÀN TẤT — U1.0–U1.7 PASS; U1.1b NEEDS_REVIEW (không tải: chưa phê duyệt ghi `_inbox/evidence/agent/` + mạng cho U1); ngưỡng vệ tinh đề xuất của U1.7 NEEDS_REVIEW.** Bronze +22 artifact, Silver +99 bản ghi (`pending`), catalog 120 dòng (phân giải 111), đối chiếu mưa 58 bản ghi, 41 ngày sự kiện (28 chưa có bằng chứng). Test: `test_event_catalog.py` 22 passed, `test_flood_indicator.py` 12 passed. STAGED_SAME. Peak RSS mọi bước ≤ 0,364 GB.

**Follow-up U1 (2026-11-20): DONE — REPORT_U1 §13.** Mục 1, 2, 4, 5, 6, 7 PASS; mục 3 NEEDS_REVIEW (`location_text` nguyên văn 20/99; kỳ 24 h của 6 bản ghi VnExpress 07/10/2025 không được đoạn văn chứng minh). gpkg ranh giới `data/reference/hanoi_city_admin_osm_20261012.gpkg` ngoài danh sách ghi §7 → giữ nguyên, chờ phê duyệt (D115). Quét ngưỡng: không thiết lập nào đạt cả 3 tiêu chí. `pytest -q tests` → 214 passed. STAGED_SAME. Quyết định D112–D115.

**Follow-up 2 U1 (2026-11-20): DONE — REPORT_U1 §14.** Mục 1–7 PASS; đề xuất ngưỡng GSMaP NEEDS_REVIEW (không ứng viên nào đạt cả 3 tiêu chí). yaml: IMERG Final/Late dùng 50/100/150 (U1-D1), satellite_proposal REJECTED. Silver +1 bản ghi sửa `fer_297b4a659e6a8aee` (thay `fer_b101085e8c50fd10`, overrides). Catalog 121 dòng, cột vị trí nguyên văn, alias Văn Miếu. U1.4: 59 bản ghi, tập chính 27 / 2 đợt. `pytest -q tests` → 217 passed. STAGED_SAME. Lưu ý: dòng phê duyệt follow-up 2 không có trong CLAUDE.md §8 (D116). Quyết định D116–D121.

**U1.1b (2026-11-20): DONE — REPORT_U1 §15.** 20 tài liệu tải (robots cho phép ở cả 11 host), 2 already in inbox, 2 mismatch (không trích); Bronze +20, Silver +94 (215), catalog 215, U1.4 120 (tập chính 46 / 5 đợt), 22 ngày có bằng chứng. Với tập bằng chứng mở rộng, không ngưỡng A6h nào đạt tiêu chí "vàng ở mọi ngày ngập do mưa" (mưa đối lưu ngắn bị pixel 0,1° bỏ sót); mọi đề xuất NEEDS_REVIEW. `pytest -q tests` → 218 passed. STAGED_SAME. Dòng phê duyệt U1.1b trong CLAUDE.md §8 vẫn không có (D122). Quyết định D122–D127.

**Follow-up 3 U1 (2026-11-20): DONE — REPORT_U1 §16.** Mục 1–5 PASS; đề xuất GSMaP NEEDS_REVIEW. Tập tham chiếu 10 ngày; quét 2-D GSMaP (a 50–150, L 150–330); top 1: GSMaP NOW GC 110/220/270 bắt 9/10 ngày tham chiếu với 11,6 % ngày có icon, 4,9 % ngày đỏ (2024–2025); GSMaP thô tốt nhất 6/10. yaml: chỉ `followup3_top3` + 2 dòng chú thích (REJECTED U1-D1). `pytest -q tests` → 218 passed. STAGED_SAME. Quyết định D128–D130.

**U1-R (2026-11-20): PASS — REPORT_U1 §18.** Tiền duyệt tự động 214 bản ghi Silver: auto_pass 137 (`auto_checked`), needs_human 77 (`pending`); tập chính 40 / 6. Lượt trích xuất mù bằng subagent riêng (114 đoạn). Mẫu kiểm tra 40 (seed 20261120). Overlay `data/gold/event_catalog/review_overlay.parquet`, catalog có cột `review_effective`. Silver không sửa. `pytest -q tests` → 258 passed. Quyết định D135–D140. Sai lệch quy trình: 2 lệnh ngắn chạy khi RAM trống < 0,5 GB (§18.6).

**U1-R2 (2026-11-20): NEEDS_REVIEW → hoàn tất bởi U1-R2b (dưới) — D143–D151.** Quy tắc 1–6 đã viết trong `src/events/auto_review.py` (+ `apply_review.py` cho giá trị `duplicate`, D150) với 12 test mới; 5 duplicate_of đã thêm vào overrides. Chưa chạy test hay phân loại lại: RAM trống < 0,5 GB liên tục 5 phút (D151). `reports/u1/review/*`, overlay, catalog và phiếu duyệt vẫn là bản U1-R (137 / 77).

**U1-R2b (2026-11-26): PASS — REPORT_U1 §19, D201–D203.** Quyết định người dùng 2026-11-23: `duplicate` chỉ khi cùng trạm, cùng kỳ nêu trong văn bản, cùng giá trị (≤ 1 %) → giữ Tây Mỗ 169, bỏ 4 duplicate_of; khoá bỏ trùng U1.4 = trạm + period_basis + cửa sổ; C9 dòng bảng = `na` + cảnh báo. Đã chạy (mọi lệnh sau cổng RAM ≥ 0,5 GB): pytest từng tệp × 28 → 403 passed; rain_validation 120; follow-up rain (loại trùng 11→9, tập chính 46 / 5 không đổi); prepare (sha blind_input khớp); classify → auto_pass 173, needs_human 34, duplicate 7 (tập chính 44 / 2), mẫu 44; catalog 215; apply_review chạy thử → overlay_written false. Silver/Bronze không đổi, không mạng, không commit.

**U1-AI (2026-11-26): PASS — REPORT_U1 §20, D204–D207.** Duyệt AI độc lập (một subagent ngữ cảnh mới) cho 34 needs_human + 44 mẫu: accept 63, reject 4, uncertain 11; tỉ lệ lỗi AI trên mẫu 0/39 = 0,0. review_effective: auto_checked 134, ai_reviewed 63, duplicate 7, pending 6, ai_rejected 4 (+1 null bị thay thế). 4 reject đều ngoài tập chính (3 bản ghi mưa rỗng giá trị, 1 qualifier) → tập chính U1.4 không đổi, không chạy lại. `apply_review --ai-review` idempotent (sha overlay `0085007e…`, events `e1f9b19f…`); pytest từng tệp × 29 → 411 passed. Silver/Bronze/phiếu duyệt không đổi, không mạng, không commit.

**U1-AI2 (2026-11-26): PASS — REPORT_U1 §21, D208.** Hai bản ghi tập chính có giờ bắt đầu trong văn bản (SGGP 'Từ khoảng 17 giờ 30'; Nhân Dân 'từ 0 giờ 10 phút') → giữ `clock`, cảnh báo 'approximate start stated in text' trong `configs/event_catalog_overrides.yaml` (`period_basis_warnings`, chỉ tài liệu). Chạy lại rain_validation, follow-up rain, catalog, apply_review --ai-review: mọi đầu ra sha256 giống trước; tập chính 46 / 5 đợt và tỉ số không đổi; overlay counts không đổi. pytest 4 tệp sự kiện → 99 passed.

Điểm tiếp tục U1: không bắt buộc duyệt thủ công. Tùy chọn: người dùng xem 6 `pending` + 5 mẫu uncertain (REPORT_U1 §20.2) trong phiếu, rồi `python -m src.events.apply_review` (giá trị người thắng AI, dòng AI được giữ). Task hiện tại vẫn là P4E (dòng CLAUDE.md không đổi). P4B vẫn ở điểm dừng OOF (chờ người dùng).

---

# STATUS — Phase 4B Mô hình v1 (`TASK_P4B_MODEL_V1.md`)

Nhánh: `feat/met-dataset`. Tiến độ: `P4B_PROGRESS.md`. Báo cáo: `REPORT_P4B.md`.

**Trạng thái (2026-11-21, D160–D162): BƯỚC 6 (P4B.7) XONG — DỪNG, chờ người dùng chạy git và xem hình.** Fetch + verify: `p4b_post` `fe6059a77ff1` PASS (54/54). `p4b_smoke` `10161c20403f` **FAIL 1/101** (`A/lead60_thr2p5/meta.json` chỉ khác `seconds`; lượt 1/1 race_drill bị hủy ghi đè; chỉ smoke; NEEDS_REVIEW D162). REPORT_P4B §8.1 (bằng chứng race), §9 (reliability, ECE, replay; §7 không đổi), §10 (đóng gói). `models/v1/MODEL_CARD.md`. `predict.py` nạp v1 (B + dự phòng A theo `serving_rule`, v0 giữ nguyên), `test_predict_v1.py` 14 passed (dev), 12 passed + 2 skipped (công khai). Sync `publish_paths_p4b7.txt` (1 thêm, 1 đổi), chưa commit. STAGED_SAME (11). Điểm tiếp tục: người dùng chạy lệnh git in ở báo cáo bàn giao, xem các PNG, quyết các việc mở trong P4B_PROGRESS.

**Trạng thái trước (2026-11-21, D155–D159): BƯỚC 4 XONG — DỪNG trước publish, chờ người dùng xem.** Sửa race, step summary, `periods.json`, job `p4b_post` (REPORT_P4B §8); test tổng hợp từng tệp PASS (công khai 102 passed); thư mục công khai đã sync (3 thêm, 5 đổi), chưa commit. Điểm tiếp tục: người dùng chạy lệnh git và các form theo RUNBOOK "P4B sau lượt final" (smoke 1/2 + 2/2 → smoke race_drill → p4b_post), báo agent để `fetch_run --verify`; sau đó bước 6 (MODEL_CARD, predict.py).

**Trạng thái trước (2026-11-20, D152–D154): P4B.6 XONG — DỪNG trước bước 4 (thay đổi cloud sau-final).** Lượt final run #10 xong; fetch + kiểm tra PASS (25/25 sha output final, 263/263 toàn run, `done.json`, periods đúng, nhãn Final/Late theo `g2/partitions.csv`); REPORT_P4B §7 viết từ CSV. 2,5 mm/h BEATS_ALL ở F0/F30/F60; 7,6 TIES (test), LOSES (10/2025 Late); luật thăng hạng v1 > v0 đạt. Điểm tiếp tục: người dùng duyệt bước 4 của kế hoạch trong `P4B_PROGRESS.md`.

**Trạng thái trước (2026-11-20, D142): FINAL ĐÃ DUYỆT — chờ người dùng chạy `train` `p4b` / `1/1` / `final = true`.** `OOF_DONE.json` `final_approved = true`, `check_final_guard` → approved cho `c0df12477947`; run_id không đổi; public = dev. Kế hoạch P4B.6–P4B.7 trong `P4B_PROGRESS.md`. Điểm tiếp tục: người dùng báo lượt final xong → `fetch_run --job p4b --run-id c0df12477947`.

**Trạng thái trước (2026-11-20, sau hub-chat D141): VẪN ĐIỂM DỪNG OOF — chưa duyệt final.** Duyệt của người dùng có điều kiện (a) dự đoán cấp dòng cho mọi mô hình; `run_final` chỉ lưu cho v1/v1_A/v1_B (baseline và pilot v0 không) → FAIL, dừng, `final_approved = false` không đổi (REPORT_P4B §5.1). Điểm tiếp tục: người dùng chọn cách xử lý (a).

**Trạng thái trước (2026-11-20): ĐIỂM DỪNG OOF (sau P4B.5).** Run `p4b` `c0df12477947` 116/116 đơn vị OOF; part 3/3 lỗi `ObjectConflict` summary.json chỉ ở cuối (race gate `oof-report`, D110; kết quả không ảnh hưởng, lượt final 1/1 không bị ảnh hưởng). Kiểm tra R2 PASS (238/238 sha, gate nhất quán, 0 metric 2025 — D111). REPORT_P4B §3–§6 viết xong. `OOF_DONE.json` `final_approved = false`, guard từ chối `--final`. STAGED_SAME (11). Điểm tiếp tục: người dùng đọc REPORT_P4B §4–§5 và quyết ba mục §5; khi được phép → `cloud_train --job p4b --write-oof-done --approve-final`, người dùng chạy `train` `p4b` / `1/1` / `true`.

**Trạng thái trước (2026-11-20): CHUẨN BỊ CLOUD HOÀN TẤT — chờ người dùng push và chạy `p4b_smoke`.** Định nghĩa đánh giá D93; quy tắc RAM D94–D95; smoke tách biệt D96; thiết kế D97–D99. Test tổng hợp (từng tệp): 117 passed ở repo dev, 137 passed trong thư mục công khai. STAGED_SAME. Điểm tiếp tục: người dùng chạy lệnh git + `p4b_smoke` (REPORT_P4B §2) và báo "p4b_smoke finished" → agent `fetch_run --job p4b_smoke`, kiểm tra, rồi người dùng chạy `p4b` 1/3..3/3.

---

# STATUS — Phase 2B Phần 3 (imerg_late, sau "imerg_late finished" 2026-11-19)

Nhánh: `feat/met-dataset`. Mọi lệnh: `HFL_P2B_PART2=1`, `HFL_P2B_EXCLUDE_SOURCES=` (rỗng — gỡ chốt chặn D49). Bước theo REPORT_P2B §11.

**Trạng thái (2026-11-19): HOÀN TẤT.** Báo cáo: `REPORT_P2B.md` §13. Quyết định D89–D90. Chốt chặn imerg_late đã gỡ: từ nay KHÔNG cần đặt `HFL_P2B_EXCLUDE_SOURCES=imerg_late`.

| Bước | Trạng thái | Bằng chứng |
|---|---|---|
| Integrity (4 nguồn) | PASS | Late 552/552 ngày OK; 26.496 dòng manifest valid; 19 s |
| Silver Late | PASS | 552 built, dq_pass 100 %, 0 trùng/NaN |
| G1 | PASS | 3/3 mùa; Late 100 % mọi mùa (gồm 10/2025) |
| G3 | ACCEPTED_WITH_NOTES | Late vs Final ACCEPTABLE 3 mùa × 2 vùng (CSI0,5 h60 0,66–0,83; |rel bias| ≤ 0,13). Đóng theo quyết định người dùng 2026-11-17: không quantile mapping; giữ 2023, đánh giá theo mùa, ablation P4A.5b. Bất thường 2023 = giới hạn (REPORT_P2B §12, §13.8) |
| IMERG Final 10/2025 | không có | CMR V07: 0 granule, collection `time_end` 2025-09-30; mọi phiên bản: 0 → không backfill (§13.9, D92) |
| rain_events | PASS | Final giống hệt từng byte; Late test 2025-05..10: 191/169/123 sự kiện; 2025-10-06/07 là 2 ngày mưa nhất |
| r2_sync | PASS | +5.115 tệp / 360,9 MB, remaining 0; verify khớp mọi tệp dữ liệu; R2 ≈ 2,12 GB |
| Test / ràng buộc | PASS | 138 passed; STAGED_SAME; không commit/push |

Điểm tiếp tục: G3 đóng (ACCEPTED_WITH_NOTES), G4 đã khóa 2026-11-17 (2,5 / 7,6 mm/h). Nhãn 10/2025 = IMERG Late. Người dùng duyệt U0 → U1; chạy lại quy tắc Open-Meteo từ ~2026-11-30.

---

# STATUS — Phase U0 Evidence fetch (`TASK_U0_EVIDENCE_FETCH.md`)

Nhánh: `feat/met-dataset`. Tiến độ: `U0_PROGRESS.md` (một dòng/URL). **Trạng thái (2026-11-18): HOÀN TẤT — NEEDS_REVIEW (chờ người dùng duyệt).** 25/25 URL có trạng thái cuối: 22 SAVED (sha256 khớp nhật ký), 2 FAILED (Lao Động 177 byte; ReliefWeb HTTP 202), 1 SKIPPED (Hanoi Times robots.txt 401). 2 lượt chạy, 30 + 13 yêu cầu (D81). 22 dòng nối vào `inbox.csv` (322 byte gốc không đổi). Phiếu duyệt: `U0_REVIEW.md`. Quyết định D80–D86. Mã: `scripts/evidence/u0_fetch.py`, `u0_extract.py`, `u0_build_review.py`, `u0_append_inbox.py`. Điểm tiếp tục: người dùng duyệt `inbox.csv` theo `U0_REVIEW.md`, rồi `TASK_U1_EVENT_CATALOG.md`.

---

# STATUS — Phase 4C Cloud training (`TASK_P4C_CLOUD_TRAINING.md`)

Nhánh: `feat/met-dataset`. Tiến độ: `P4C_PROGRESS.md`. Mọi lệnh đặt `HFL_P2B_EXCLUDE_SOURCES=imerg_late`.

**Trạng thái (2026-11-18): P4C HOÀN TẤT — kết quả p4a6 (run `366c444a44ff`) và p4a7 (run `1132ff716152`) đã lấy về; REPORT_P4A P4A.6, P4A.7 PASS; replay Yagi ngoài mẫu (fold `LOSO_2024`) xong; MLflow +39 run; `pytest -q tests` → 137 passed.** Quyết định D87–D88. Báo cáo: `REPORT_P4A.md` (P4A.6/P4A.7/P4A.8 cập nhật P4C, mục P4C.4). Điểm tiếp tục: người dùng duyệt báo cáo, rồi `TASK_P4B_MODEL_V1.md` (job `p4b` hiện chỉ dành chỗ, D77). (2026-11-19: imerg_late đã xong, chốt chặn gỡ — xem Phase 2B Phần 3.)

---

# STATUS — Phase 4A Pilot model v0 (`TASK_P4A_PILOT_MODEL.md`)

Nhánh: `feat/met-dataset`. Tiến độ chi tiết: `P4A_PROGRESS.md`. imerg_late vẫn đang backfill: mọi lệnh đặt `HFL_P2B_EXCLUDE_SOURCES=imerg_late`; nhãn = IMERG Final.

**Trạng thái (2026-11-18): PHASE 4A KẾT THÚC phần chạy cục bộ — P4A.0–P4A.5b, P4A.8, P4A.9 PASS; P4A.6 và P4A.7 NEEDS_REVIEW (huấn luyện còn lại hoãn sang `TASK_P4C_CLOUD_TRAINING.md`).** Báo cáo: `REPORT_P4A.md`; model card: `models/pilot_v0/MODEL_CARD.md`. Quyết định D71–D75. Điểm tiếp tục: người dùng duyệt báo cáo, rồi P4C (7 mô hình B, A_no2023 0,5, compare_A_B, leave-one-season-out, replay Yagi từ fold bỏ-2024).

---

# STATUS — P3B hotfix Open-Meteo (`TASK_P3B_HOTFIX_OPENMETEO.md`)

Nhánh: `feat/met-dataset`. **Trạng thái (2026-11-17): HOÀN TẤT — PASS.** Báo cáo: `REPORT_P3A.md` §P3B hotfix. Quyết định D55–D57.

| Hạng mục | Trạng thái | Bằng chứng |
|---|---|---|
| Timeout/retry, cô lập model, rule_fallback, mã thoát | PASS | `pytest -q tests` → 83 passed (+7 test) |
| Chạy thật (store cục bộ) | PASS | exit 0, 2 run `metadata`, 5,6 s |
| Snapshot công khai | PASS | sync 5 tệp (D56), pytest trong snapshot → 49 passed; người dùng tự `git add/commit/push` |
| Rủi ro | NEEDS_REVIEW | ECMWF 00Z 2026-11-17 có mặt sau 543 phút > quy tắc 530 phút; chạy lại quy tắc ~2026-11-30 |

Điểm tiếp tục: người dùng chạy 3 lệnh git trong REPORT_P3A §P3B hotfix, rồi chạy tay `openmeteo-latest`. Ràng buộc imerg_late của Phần 2 vẫn giữ (không đọc/ghi thư mục imerg_late).

---

# STATUS — Phase 2B Phần 2 (chạy trên toàn lịch sử — 3 nguồn đã xong)

Nhánh: `feat/met-dataset`. Người dùng (2026-11-17): backfill xong cho gsmap_now, imerg_final, openmeteo. **imerg_late VẪN ĐANG CHẠY** (1 cửa sổ PowerShell): KHÔNG đọc/ghi `data/bronze_regional/imerg_late`, `data/silver/met/imerg_late`; loại khỏi `r2_sync`; mọi mục phụ thuộc Late = PENDING đến khi người dùng viết "imerg_late finished". Mọi lệnh của tôi đặt `HFL_P2B_PART2=1` và `HFL_P2B_EXCLUDE_SOURCES=imerg_late`.

**Trạng thái (2026-11-17): PHẦN 2 HOÀN TẤT cho 3 nguồn — mục phụ thuộc imerg_late PENDING.** Báo cáo: `REPORT_P2B.md`. Quyết định D49–D54.

| Task | Trạng thái | Bằng chứng / ghi chú |
|---|---|---|
| Chốt chặn imerg_late | PASS | D49; 2 test; dry-run R2 0 khóa imerg_late |
| P2B.1 Integrity | PASS | 0 dòng lỗi; gsmap 2023-10-24 thiếu 9 cửa sổ (nguồn JAXA, D53) |
| P2B.2 Silver | PASS (3 nguồn) / Late PENDING | 1.441 ngày-nguồn, dq_pass 100 % |
| P2B.3 G1 | PASS 3/3 mùa / Late PENDING | `reports/p2b/g1/` |
| P2B.4 G3 | NEEDS_REVIEW | Late PENDING; đề xuất quantile mapping; bất thường GSMaP 2023 |
| P2B.5 Ngưỡng | PASS (đề xuất 5,0 / 7,6) | `configs/targets.proposed.yaml` |
| P2B.5b Sự kiện | PASS | không ngưỡng nào LOW_CONFIDENCE |
| P2B.6 Open-Meteo | NEEDS_REVIEW | GFS 420 / ECMWF 530 phút; 1,0 ngày run < 14 |
| P2B.8 R2 | PASS (3 nguồn) | 9.604 tệp / 406,8 MB lên `lake/v1/`; verify khớp mọi tệp dữ liệu; `bronze/` 59 đối tượng / 898 MB, 0 trùng, khuyến nghị giữ |
| Test | PASS | 76 passed |

Điểm tiếp tục: người dùng viết "imerg_late finished" → REPORT_P2B §11. Chạy lại quy tắc Open-Meteo từ ~2026-11-30.

---

# (Lưu trữ) STATUS — Phase 2B Phần 1 (History quality: viết mã + thử trên ngày đã hoàn tất)

Nhánh: `feat/met-dataset`. Tệp task: `TASK_P2B_HISTORY_QUALITY.md`. Backfill đang chạy ở 3 cửa sổ PowerShell: chỉ ĐỌC đầu ra backfill; KHÔNG ghi `data/bronze_regional/`, `data/silver/met/`, `logs/backfill/`, `data/_scratch/`; không chạy `run_backfill.py`, `r2_sync.py`. Đầu ra thử → `data/_ci_test/p2b/`, `reports/p2b_dev/`.

**Trạng thái (2026-11-16): PHẦN 1 HOÀN TẤT — dừng theo task, chờ người dùng viết "backfill finished".** Báo cáo: `REPORT_P2B_PART1.md`. Quyết định D38–D48. Staged 11 tệp: `STAGED_SAME`.

| Task | Trạng thái | Bằng chứng / ghi chú |
|---|---|---|
| P2B.1 Integrity | PASS | `bronze_integrity.py`; 1 ngày ORPHAN_FILE: imerg_final 2023-05-16 (WinError 5, rất có thể do lần quét detect-secrets của tôi ở P3A, D39) |
| P2B.2 Silver | PASS | 15 ngày-nguồn → `data/_ci_test/p2b/silver/`, dq_pass 15/15, chạy lại → unchanged |
| P2B.3 G1 | PASS (mã); gate chờ Phần 2 | tạm thời 1/3 mùa đạt (backfill chưa xong) |
| P2B.4 G3 | PASS (mã); gate chờ Phần 2 | `reports/p2b_dev/g3/` |
| P2B.5 Ngưỡng | PASS (mã); đề xuất chờ Phần 2 | `reports/p2b_dev/thresholds/` |
| P2B.6 Open-Meteo availability | NEEDS_REVIEW | GFS 420 / ECMWF 520 phút từ 3 manifest; kiểm tra lại rò rỉ: 0 dòng đổi |
| P2B.7 sync_public | PASS | dry-run / sao chép / từ chối exit 2 |
| Test | PASS | `pytest -q tests` → 72 passed |

Điểm tiếp tục:
1. Người dùng: chạy lại imerg_final 2023-05-16; chỉ để một tiến trình ghi imerg_late (REPORT §0).
2. Sau "backfill finished": chạy Phần 2 theo REPORT_P2B_PART1 §4 với `HFL_P2B_PART2=1`.

---|---|---|
| P2B.1 Integrity | đang thử | `bronze_integrity.py` chạy chỉ-đọc 8,7 s; phát hiện imerg_final 2023-05-16 ORPHAN_FILE (WinError 5 khi thay manifest) |
| P2B.2 Silver | đang thử | 15 ngày-nguồn → `data/_ci_test/p2b/silver/`, dq_pass hết; chốt chặn ghi hoạt động |
| P2B.3 G1 | đang thử | `coverage_g1.py` chạy trên trạng thái tạm (FAIL tạm thời do backfill chưa xong) |
| P2B.4 G3 | chưa | |
| P2B.5 Ngưỡng | chưa | |
| P2B.6 Open-Meteo availability | đang thử | 3 manifest R2; quy tắc GFS 420 phút, ECMWF 520 phút (mẫu nhỏ) |
| P2B.7 sync_public | chưa | |

---

# (Lưu trữ) STATUS — Phase 3A (Continuous ingestion, GitHub Actions — chỉ dựng và thử cục bộ)

Nhánh: `feat/met-dataset`. Tệp task: `TASK_P3A_CONTINUOUS_INGESTION.md`. Backfill đang chạy ở 2 cửa sổ PowerShell: KHÔNG ghi `data/bronze_regional/`, `data/silver/met/`, `logs/backfill/`, `data/_scratch/`; đầu ra thử nghiệm → `data/_ci_test/`.

**Trạng thái (2026-11-16): P3A HOÀN TẤT — 4 PASS.** Báo cáo: `REPORT_P3A.md`. Runbook công bố: `RUNBOOK_P3_PUBLISH.md`. Quyết định D24–D37. Staged 11 tệp: giống hệt `staged_before.txt` (kiểm tra 2026-11-16).

| Task | Trạng thái | Bằng chứng / ghi chú |
|---|---|---|
| P3A.1 Entry points | PASS | `src/realtime/` (common + 5 entry point), `configs/realtime.yaml`; `pytest -q tests/test_realtime.py` → 19 passed |
| P3A.2 Workflows | PASS | 6 tệp `.github/workflows/`, `requirements-ci.txt`; Python 3.14, checkout@v7, setup-python@v7, ubuntu-24.04; tập test CI → 42 passed |
| P3A.3 E2E cục bộ | PASS | log `data/_ci_test/logs/`; GSMaP 3 tệp 12 s, lần 2 upload 0; IMERG Late 2026-11-14 48/48 109 s; Open-Meteo 1 request 2 run; compaction; freshness OK (exit 0) sau D27; R2 `lake/v1/realtime/` 12 đối tượng 293.603 B + status |
| P3A.4 Public readiness | PASS | 0 bí mật (giá trị chính xác + detect-secrets, 19 commit/61 blob + working tree); `.gitignore` thiếu `data/`, `_archive/`, `.env.*`; khuyến nghị lộ trình (b), `publish_paths_p3.txt` (38 đường dẫn, snapshot 42 passed) |
| Toàn bộ test | PASS | `pytest -q tests` → 56 passed |

Điểm tiếp tục (chờ người dùng):
1. Backfill (2 cửa sổ) tiếp tục theo `RUNBOOK_BACKFILL.md`; sau đó Silver + `r2_sync`.
2. Công bố theo `RUNBOOK_P3_PUBLISH.md` (lộ trình b), thêm 8 secret, chạy tay 5 workflow, đọc lần chạy đầu.
3. Nhắc xoay vòng token Earthdata ngày 2026-12-29 (hết hạn 2027-01-12).
4. Sau backfill: có thể thêm tham số `out_path` vào `regional_crop.write_day_file` để bỏ cơ chế chuyển hướng (D24).

---

# (Lưu trữ) STATUS — Phase 2A (Backfill preparation)

Nhánh: `feat/met-dataset`. Tệp task: `TASK_P2A_BACKFILL_PREP.md`. Danh sách staged chuẩn: `docs/agent_runs/staged_before.txt` (11 tệp) — kiểm tra lần cuối 2026-11-16: giống hệt.

**Trạng thái chung (2026-11-16): P2A HOÀN TẤT — 3 PASS.** Báo cáo: `REPORT_P2A.md`. CLAUDE.md đã cập nhật (task file + quyết định 7–9). Quyết định D14–D23.

| Task | Trạng thái | Bằng chứng / ghi chú |
|---|---|---|
| P2A.1 Quyết định 1–2 | PASS | ADD-003 (JAXA NOW doc §5.3: HHNN = start time), ADD-004; proposal → `_archive/configs/`; driver `--path opendap` (mặc định): imerg_final 2025-08-01 OK 394,7 s (xác minh 48/48 khớp chính xác), 2025-08-02 OK 122,0 s; imerg_late 2025-10-01 OK 393,5 s (48/48 khớp); manifest đủ trường ADD-004; readiness đặc trưng trễ: `reports/p1_readiness/lag_feature_*` |
| P2A.2 Open-Meteo prototype | PASS | `openmeteo_feasibility.md`; Previous Runs `previous_day1`, GFS issue = floor6h(V)−24h (24/24 giờ khớp), ECMWF cận trên −18h; 13 ngày test + 2 ngày driver, dq_pass; min(t − available) 630 phút; `reports/p2a_openmeteo/`; thêm vào backfill (chỉ 2024–2025) |
| P2A.3 Runbook | PASS | `RUNBOOK_BACKFILL.md` + `scripts/backfill/window1_gsmap_openmeteo.ps1`, `window2_imerg.ps1` (parse 0 lỗi); ước lượng song song ≈ 38 h, lưu ≈ 0,5 GB |
| Kiểm thử | PASS | `pytest -q tests` → 37 passed |

Điểm tiếp tục (chờ người dùng):
1. Chạy hai cửa sổ backfill theo `RUNBOOK_BACKFILL.md` §3 (≈ 38 h), rồi §7 (Silver) và §8 (R2 sync).
2. Sau backfill: chạy lại `readiness_probes`, `readiness_lag_features`, `readiness_openmeteo` trên toàn bộ mùa (gate G3).
3. Khóa ngưỡng trong `configs/targets.yaml` (G4).

---

# (Lưu trữ) STATUS Phase 1 — Data foundation

Nhánh: `feat/met-dataset`. Danh sách staged chuẩn: `docs/agent_runs/staged_before.txt` (11 tệp) — đã kiểm tra lần cuối: giống hệt.

**Trạng thái chung (2026-11-16): P1 HOÀN TẤT — 9 PASS, 1 NEEDS_REVIEW (P1.7: ngữ nghĩa nhãn thời gian GSMaP).** Báo cáo: `REPORT_P1.md`.

| Task | Trạng thái | Bằng chứng / ghi chú |
|---|---|---|
| P1.0 Housekeeping | PASS | nhánh feat/met-dataset; staged 11 tệp không đổi; archive 3 mục; pip OK (pyarrow 26.0.1, pytest 9.2.0, boto3 1.43.119); inventory_p1_before.md; D trống 187 GB |
| P1.1 Domain | PASS | geoBoundaries VNM-ADM1-63759600; diện tích 3214.7 km²; core9 trong 100%; H3 27,918 ô (D06); 840 pixel / 47 city / 5 core9; met_dataset.yaml; ADD-001, ADD-002 |
| P1.2 Bronze regional | PASS | `build_bronze_regional --all`: 13 ngày IMERG × 48 = 624 dòng manifest, GSMaP 2 tệp; `--verify 200` → 0 mismatch cho cả hai nguồn; ~3.6 s/ngày |
| P1.3 GSMaP archive | PASS | 15/15 ngày `--download --promote`: 48/48 cửa sổ rain mỗi ngày (ingested_share 1.0), 0 lỗi; ~4 phút/ngày; manifest 722 dòng; `--verify 200` 0 mismatch; Bronze GSMaP 3.1 GB. Độ trễ trung vị MDTM−window_end: 2024 ≈ −4.1…−4.3 phút, 2025 ≈ −0.03…−0.18 phút → quy tắc observed latency, available_time = max(MDTM, window_end) (D08) |
| P1.4 IMERG subset | PASS | OPeNDAP DAP4 netCDF-4: Final 2 granule 2025-09-30 khớp chính xác crop tệp gốc (all_equal); Late 2 granule 2025-09-28 + 2 granule 2025-10-06 khớp chính xác đường fallback. OPeNDAP ≈ 2.0–2.4 s, 52–54 KB/granule; fallback ≈ 5.4–5.7 s, 7.8–8.5 MB (scratch tự dọn). Khuyến nghị: OPeNDAP cho backfill. Sửa lỗi tên tệp `:` (D12) |
| P1.5 Silver | PASS | IMERG 13 ngày + GSMaP 16 ngày, mỗi ngày 840 pixel × 48 = 40,320 dòng, 0 trùng, NaN 0, dq_pass; GSMaP available_time ≥ window_end 100%; lệch tâm lưới gốc ≤ 1.5e-6°; `--regression` 2025-09-30: 71,088/71,088 khớp (max diff 4.7e-7). ERA5-Land hoãn |
| P1.6 Temporal tests | PASS | `pytest -q tests/test_temporal_contract.py` → 23 passed |
| P1.7 Readiness | NEEDS_REVIEW | bảng CSV + PNG + README trong `reports/p1_readiness/`; 13 ngày; corr lag0 city GC 0.53, buffer GC 0.51; **cực đại tương quan ở lag −60/−90 phút** → nghi nhãn tên tệp GSMaP là cuối cửa sổ (D11, `configs/gsmap_now.proposed.yaml`); G3 chưa đạt (cần toàn lịch sử) |
| P1.8 R2 sync | PASS | dry-run 117 đối tượng (tất cả dưới `lake/v1/`); chạy thật 117 tệp, 15,233,170 B, 51 s; `--verify`: remote 117 / 15,233,170 B, missing 0, size_mismatch 0; dry-run lại: skipped_identical 117 |
| P1.9 Backfill runbook | PASS | thử một ngày 2025-07-15: imerg_final OK 278.5 s, imerg_late OK 275.7 s, gsmap_now OK 205.1 s; mỗi nguồn 48 cửa sổ trong crop + 48 dòng manifest có sha256 gốc; scratch 0 tệp; chạy lại → SKIP; RUNBOOK §4 cập nhật số đo (D13) |

Điểm tiếp tục (chờ người dùng):
1. Quyết định `configs/gsmap_now.proposed.yaml` (nhãn GSMaP = cuối cửa sổ?). Nếu phê duyệt: dựng lại Silver GSMaP (`met_silver_builder --source gsmap_now --all`) và chạy lại `readiness_probes`.
2. Chọn đường IMERG cho backfill (fallback có sha256 gốc / OPeNDAP); chạy backfill theo `RUNBOOK_BACKFILL.md`; `r2_sync`.
3. Khóa ngưỡng trong `configs/targets.yaml` (G4).
