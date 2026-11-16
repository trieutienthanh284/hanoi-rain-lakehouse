# ADD-004 — Nguồn gốc (provenance) của IMERG lấy qua OPeNDAP

**Trạng thái:** Approved (người dùng quyết định 2026-11-16; addendum do tác tử soạn trong P2A.1)
**Áp dụng cho:** IMERG Final / Late / Early V07 (`imerg_final`, `imerg_late`, `imerg_early`) trong `data/bronze_regional/`.
**Sửa đổi:** ADD-002 §1.2 (manifest phải có sha256 tệp gốc) **chỉ cho IMERG**. GSMaP NOW vẫn theo ADD-002 nguyên vẹn.

## 1. Quyết định

1. Đường backfill IMERG mặc định là **OPeNDAP DAP4** (Hyrax tại `opendap.earthdata.nasa.gov`): mỗi granule nửa giờ được yêu cầu đúng miền Hà Nội (chỉ số lưới trong `configs/met_dataset.yaml`), cả 10 biến 3 chiều của nhóm `Grid` và `Grid/Intermediate`, cùng `lat`, `lon`, `time_bnds`. Kết quả được ghi vào cùng tệp ngày như đường tải đầy đủ: `data/bronze_regional/imerg_{run}/date=YYYY-MM-DD/imerg_{run}_YYYYMMDD.h5`. Biến, kiểu dữ liệu, thuộc tính và giá trị sentinel giữ nguyên.
2. **Manifest** (`data/bronze_regional/imerg_{run}/manifest.jsonl`), một dòng mỗi granule:
   - `cmr_granule_ur`, `cmr_concept_id`, `cmr_revision_id`, `cmr_production_time`, `product_version`;
   - `cmr_size` + `cmr_size_unit` (CMR chỉ cho kích thước gần đúng tính bằng MB), `cmr_checksum` + `cmr_checksum_algorithm` (để `null` vì CMR/GES DISC hiện không công bố checksum);
   - `original_name`, `original_path` (URL HTTPS của tệp gốc);
   - `opendap_url`: URL yêu cầu DAP4 đầy đủ (token Earthdata chỉ gửi trong header `Authorization`, không bao giờ nằm trong URL hay log);
   - `subset_bytes`, `subset_sha256`: kích thước và sha256 của phản hồi netCDF-4 nhận được;
   - `retrieval_path: "opendap_dap4"`, `verified_full_download`, `crop_file`, `crop_sha256`, cửa sổ thời gian UTC;
   - `original_size`, `original_sha256`: chỉ điền vào ngày có xác minh (mục 3), `null` ở ngày khác.
3. **Xác minh định kỳ:** với **ngày 1 của mỗi tháng**, mọi granule của ngày đó còn được tải đầy đủ (HDF5 gốc) vào `data/_scratch/`, cắt bằng `rc.read_crop` và so với mảng OPeNDAP; yêu cầu **khớp chính xác** (cùng dtype, `array_equal`, kể cả `time_bnds`). Kết quả ghi một dòng vào `data/bronze_regional/imerg_{run}/verification.jsonl`. Nếu lệch, ngày đó thất bại và tệp ngày không được ghi. Tệp tạm bị xóa ngay sau khi so.
4. Đường tải đầy đủ (`--path fallback`) vẫn giữ làm tùy chọn và cho ngày cần hash gốc.

## 2. Lý do

- OPeNDAP nhanh hơn khoảng 2–3 lần và tải ít hơn khoảng 80 lần (≈ 95 KB thay vì ≈ 7,8 MB mỗi granule).
- P1.4 và P2A.1 cho thấy mảng OPeNDAP khớp chính xác với crop tệp gốc (Final và Late); xác minh hàng tháng giữ bằng chứng này trong suốt backfill.
- CMR không cung cấp checksum cho IMERG HHR, nên không thể ghi hash gốc nếu không tải tệp gốc.

## 3. Giới hạn ghi nhận

- `cmr_production_time` của IMERG Late cho thấy bản lưu trữ hiện tại có thể được sản xuất lại sau thời điểm thời gian thực (ví dụ granule 2024-06-01 có ProductionDateTime 2024-06-06). Vì vậy IMERG Late lịch sử chỉ được dùng làm **nhãn tạm thời** và để so Final–Late, **không** dùng làm đặc trưng tại thời điểm dự báo (temporal_contract: không dùng bản sửa đổi chưa tồn tại tại t).
- `subset_sha256` chứng minh tính toàn vẹn của dữ liệu đã nhận; nó không thay thế hash tệp gốc. Mối liên hệ với tệp gốc được bảo đảm bằng GranuleUR + revision và xác minh hàng tháng.
