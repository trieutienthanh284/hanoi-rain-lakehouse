# ADD-003 — Ngữ nghĩa thời gian GSMaP NOW và độ trễ hiệu dụng

**Trạng thái:** Approved (người dùng quyết định 2026-11-16 sau khi xem xét P1; addendum do tác tử soạn trong P2A.1)
**Áp dụng cho:** `configs/gsmap_now.yaml` (`temporal.filename_time_semantics`), `temporal_contract.md` v1.0 (quy tắc available_time), Silver `data/silver/met/gsmap_now/`, mọi đặc trưng dùng GSMaP NOW.
**Thay thế:** đề xuất `configs/gsmap_now.proposed.yaml` (P1.7, D11) — **bị bác bỏ**, đã chuyển sang `_archive/configs/gsmap_now.proposed.yaml`.

## 1. Quyết định

1. Giữ `filename_time_semantics: start_time`. Tệp `gsmap_now_rain.YYYYMMDD.HHNN.nc` mô tả lượng mưa trung bình của cửa sổ một giờ `[HHNN, HHNN + 60 min)`.
2. Không bao giờ gán lại nhãn thời gian (relabel) dữ liệu Bronze hay Silver vì lý do tương quan.
3. Đỉnh tương quan ở độ trễ −60 phút (buffer) / −90 phút (city) trong P1.7 được hiểu là **độ trễ hiệu dụng của sản phẩm**: GSMaP NOW ngoại suy bản đồ mưa nửa giờ về phía trước bằng vector chuyển động mây từ vệ tinh địa tĩnh, dựa trên quan trắc vi sóng cũ hơn. Không phải lỗi gán nhãn.
4. Xử lý độ trễ này bằng **đặc trưng trễ** (lag features: các cửa sổ GSMaP kết thúc tại t, t−30, …, t−180 phút, mỗi cửa sổ vẫn tuân thủ `available_time ≤ t`) và bằng hiệu chỉnh (calibration) trong mô hình về sau.
5. `available_time = max(MDTM, window_end)` cho dữ liệu lưu trữ (giữ D08). Bản ghi không bao giờ được dùng trước khi cửa sổ quan trắc kết thúc.

## 2. Căn cứ tài liệu JAXA

Tài liệu *Data Format Description for Global Rainfall Map Realtime version (GSMaP_NOW) and Gauge-calibrated Rainfall Product (GSMaP_Gauge_NOW)*, JAXA/EORC, "updated on April 16, 2026" (`https://sharaku.eorc.jaxa.jp/GSMaP/document/new/DataFormatDescription_NOW.pdf`, liên kết từ FAQ Q18):

- §5.3 (NetCDF, product (4)): tên tệp `gsmap_now_rain.YYYYMMDD.HHNN.nc`, trong đó "HH: 2-digit hour of start time; and NN: 2-digit minute of start time."
- §2.3 (nhị phân, tệp 24 h gần nhất): `gsmap_now.YYYYMMDD.HHNN_hhnn.dat`, "HH … start time … hh: 2-digit hour of end time" — xác nhận HHNN là đầu cửa sổ.
- §5.1: "Temporal resolution: 1 hour (hourly data), but updated half-hourly"; "Data latency: 0-hour after observation for 'Hourly Rain Rate & Gauge-calibrated Rain Rate' (product (4))".
- §5.10: với `timeinfo_RT`, "0 means start time of the file (HHNN in file name)".
- Trang 1 và §6.1: "using passive microwave observations that are available within a half-hour after observation … and applying a half-hour extrapolation of rainfall map toward future direction by using cloud moving vector from the geostationary satellite"; "GSMaP_NOW at 'current hour' is estimated by extrapolating half-hour by cloud moving vector, so data accuracy may become worse."
- FAQ Q5 (`https://sharaku.eorc.jaxa.jp/GSMaP/faq/GSMaP_faq05.html`): "'01:00Z' means 'from 01:00Z to 01:59Z'."

Tài liệu do tệp task gợi ý (`DataFormatDescription_RNC.pdf`) là của sản phẩm GSMaP RIKEN Nowcast, không nói về ngữ nghĩa thời gian của GSMaP_NOW; vì vậy dùng tài liệu NOW ở trên. Không có câu nào mâu thuẫn với quyết định 1.

## 3. Bằng chứng thực nghiệm (P1.7, 13 ngày tháng 9, tham chiếu IMERG Final — ước lượng vệ tinh, không phải trạm đo)

| Vùng | Biến | r tại lag 0 | Lag tốt nhất | r tốt nhất |
|---|---|---|---|---|
| city | GC | 0,529 | −90 phút | 0,623 |
| city | raw | 0,428 | −90 phút | 0,511 |
| buffer | GC | 0,505 | −60 phút | 0,533 |
| buffer | raw | 0,408 | −60 phút | 0,434 |

Lag = so GSMaP cửa sổ bắt đầu s với giờ IMERG bắt đầu s+lag. MDTM − window_end có trung vị −4,1…−4,3 phút (2024) và −0,03…−0,18 phút (2025): tệp xuất hiện khoảng lúc cửa sổ kết thúc, phù hợp "0-hour latency" và phần "giờ hiện tại" được ngoại suy.

## 4. Hệ quả

- `configs/gsmap_now.yaml` và Silver GSMaP **không đổi**; không cần dựng lại.
- Đánh giá readiness và mô hình về sau phải thử tập đặc trưng trễ (P2A.1 đo hiệu quả bằng CSI persistence).
- Khi so sánh GSMaP với IMERG trong báo cáo, luôn ghi rõ lag được dùng.
