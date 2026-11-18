# ADD-005 — Khung dự báo (frames), thời điểm phát hành và ngưỡng G4

**Trạng thái:** Approved (người dùng quyết định 1A ngày 2026-11-16 và quyết định G4 ngày 2026-11-17; addendum do tác tử soạn trong P4A.0)
**Áp dụng cho:** mô hình nowcast mưa (pilot v0 trở đi), bảng nhãn, bảng đặc trưng Gold, kiểm tra rò rỉ thời gian (`src/quality/temporal_checks.py`, chế độ frame), bản đồ.
**Cấu hình máy đọc được:** `configs/model_pilot.yaml` (§`frames`, §`targets`).

## 1. Định nghĩa

- **Thời điểm phân tích `t0`**: mọi mốc nửa giờ UTC (:00, :30).
- **Thời điểm phát hành** (`prediction_time`) = `t0 + 10 phút`. Giả định vận hành: cửa sổ GSMaP NOW kết thúc tại `t0` được JAXA công bố khoảng 1,2 phút sau `t0` (45 mẫu realtime ngày 2026-11-16) và được lịch :07/:37 thu nhận.
- **Khung (frame)** — mỗi khung là một cửa sổ nửa giờ IMERG:
  - `F0 = [t0, t0+30)` ("đang diễn ra": đã trôi qua một phần khi phát hành nhưng chưa nguồn nào quan sát);
  - `F30 = [t0+30, t0+60)`;
  - `F60 = [t0+60, t0+90)`.
  - Cột `lead_min ∈ {0, 30, 60}` = đầu khung − `t0`; `frame_start = t0 + lead_min`; `target_time` = cuối khung = `frame_start + 30 phút`.
- **Nhãn**: `y = 1` nếu `rain_rate_mm_h` IMERG Final của khung tại pixel ≥ ngưỡng. Dòng thiếu nhãn bị loại và được đếm. IMERG là ước lượng vệ tinh, không phải số đo trạm.
- **Đặc trưng**: chỉ dùng bản ghi có `available_time ≤ prediction_time` và `event_time ≤ t0`; giá trị NWP cần `forecast_issue_time ≤ prediction_time` và `available_time ≤ prediction_time` (quy tắc P2A, sửa bởi P2B.6, `configs/openmeteo_availability.yaml`). Mọi mốc thời gian tz-aware UTC; hiển thị Asia/Ho_Chi_Minh.

## 2. Ngưỡng (gate G4, người dùng khóa 2026-11-17)

- Ngưỡng chính **2,5 mm/h**; ngưỡng phụ **7,6 mm/h**; 0,5 mm/h giữ làm lớp phụ trợ "có mưa" cho bản đồ.
- Chỉ số tiêu đề và kết luận M1 báo cáo cho 2,5 mm/h trước.
- Đây là ngưỡng cường độ mưa, **không** phải ngưỡng ngập.

## 3. Các câu trong hợp đồng bị thay thế

| Tài liệu | Vị trí | Câu cũ | Thay bằng |
|---|---|---|---|
| `problem_contract.md` v2.2 | §7 "Prediction Horizon" | "Hai horizon chính thức: 30 phút; 60 phút." | Ba khung F0/F30/F60 (`lead_min ∈ {0, 30, 60}`) quanh `t0`, phát hành tại `t0 + 10 phút`. |
| `problem_contract.md` v2.2 | §8 và §10 (danh sách cột) | `prediction_horizon`, `horizon_min` | `lead_min`, thêm `t0`, `frame_start`; `target_time` = cuối khung. |
| `temporal_contract.md` v1.0 | đầu tài liệu | "**Prediction horizons:** 30 phút và 60 phút" | Khung F0/F30/F60. |
| `temporal_contract.md` v1.0 | §2 `target_time` | "horizon 30 phút: `target_time = prediction_time + 30 phút`; horizon 60 phút: `target_time = prediction_time + 60 phút`" | `target_time = t0 + lead_min + 30 phút` (cuối khung); vẫn luôn `> prediction_time`. |
| `temporal_contract.md` v1.0 | §2 `prediction_horizon` | "Giá trị hợp lệ: 30, 60" | `lead_min ∈ {0, 30, 60}`. |
| `temporal_contract.md` v1.0 | §12 | "`prediction_horizon in {30, 60}`" | `lead_min ∈ {0, 30, 60}`, `prediction_time = t0 + 10 phút`, `event_time ≤ t0`. |
| `configs/targets.yaml` (P2A) | `target_fields` / `raw_target_fields` | cửa sổ tích lũy 30/60 phút từ `prediction_time` (`target_max_rate_30m`, `target_max_rate_60m`) | cường độ IMERG Final của **một** khung 30 phút. |

Quy tắc giữ nguyên: `available_time ≤ prediction_time`; không dùng tái phân tích/bản sửa chưa tồn tại tại `t`; IMERG Final không bao giờ là đặc trưng; độ phân giải hazard 0,1°, H3 res 9 chỉ để hiển thị.

## 4. Lý do

Bản đồ được làm mới mỗi 30 phút với độ trễ dữ liệu khoảng 10 phút. Nếu giữ horizon tích lũy tính từ `prediction_time`, khung đầu tiên người xem thấy sẽ bao trùm khoảng thời gian đã trôi qua một phần và trộn lẫn hai cửa sổ IMERG. Khung nửa giờ thẳng hàng với lưới IMERG cho phép (a) nhãn sạch một cửa sổ, (b) bản đồ luôn hiển thị các khung vẫn còn ở tương lai (F0 còn ≥ 20 phút, F30, F60), (c) phép kiểm rò rỉ đơn giản và tất định.

## 5. Thay đổi người dùng cần làm trong `configs/targets.yaml`

Tác tử không được sửa tệp này (quy tắc deny). Các dòng cần đổi:

```yaml
  horizons_min:            # dòng 6-8  → thay bằng
  lead_min: [0, 30, 60]
  issue_offset_min: 10
  frame_length_min: 30
  threshold_mm_h: 2.5      # dòng 10   → giữ, thêm:
  secondary_threshold_mm_h: 7.6
  auxiliary_threshold_mm_h: 0.5
  target_fields:           # dòng 14-16 → {0: y_2p5 (F0), 30: y_2p5 (F30), 60: y_2p5 (F60)}
  raw_target_fields:       # dòng 18-20 → label_rate_mm_h (một khung IMERG Final)
  notes[0]:                # dòng 23   → "threshold đã khóa 2026-11-17 (G4): 2.5 chính, 7.6 phụ, 0.5 phụ trợ"
```
