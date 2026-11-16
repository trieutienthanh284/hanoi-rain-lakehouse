# ADD-002 — Cắt vùng (regional crop) khi ingest và Cloudflare R2 là hệ thống lưu trữ gốc

**Trạng thái:** Approved (người dùng phê duyệt 2026-11-15; addendum được tác tử soạn trong P1.1)
**Áp dụng cho:** `source_matrix.md` v1.0, `configs/gsmap_now.yaml` (ghi chú Bronze), chính sách Bronze của các nguồn mưa lưới toàn cầu (IMERG, GSMaP NOW)
**Quy tắc ưu tiên:** khi mâu thuẫn, addendum này thắng contract/config.

## 1. Quyết định

1. **Regional crop khi ingest.** Với sản phẩm mưa lưới toàn cầu, Bronze lưu trữ lâu dài chỉ miền Hà Nội (bbox thành phố + 1,0°; chỉ số lưới trong `configs/met_dataset.yaml`) trong tầng `data/bronze_regional/{source}/date=YYYY-MM-DD/`. Biến và đơn vị gốc giữ nguyên, giá trị sentinel/fill thô không bị sửa, không nội suy, không đổi lưới.
2. **Manifest bảo toàn nguồn gốc.** Mỗi tệp gốc có một dòng trong `data/bronze_regional/{source}/manifest.jsonl`: tên tệp gốc, kích thước, sha256, cửa sổ thời gian UTC, MDTM từ xa (nếu biết), tệp crop và sha256 của tệp crop. Như vậy có thể chứng minh tệp crop được dẫn xuất từ tệp gốc nào.
3. **Tệp gốc toàn cầu đã có trên đĩa được giữ nguyên**, không bị xóa hay ghi đè. Các đợt backfill sau có thể dùng chế độ `--crop-only` (tải vào `data/_scratch/`, kiểm tra, cắt, ghi hash, xóa tệp tạm).
4. **Cloudflare R2 là hệ thống lưu trữ gốc (system of record).** Một tài khoản, bucket lấy từ biến môi trường `R2_BUCKET`, mọi object nằm dưới tiền tố `lake/v1/` với cùng bố cục tương đối như trên đĩa. Ổ cục bộ là bản làm việc. Ngân sách trần ~5 USD/tháng; mục tiêu nằm trong 10 GB miễn phí. Không xóa/ghi đè object ngoài `lake/v1/`.

## 2. Câu bị thay thế

| Tệp | Câu gốc | Thay bằng |
|---|---|---|
| configs/gsmap_now.yaml `notes[0]` | "Bronze stores the original GSMaP NOW NetCDF artifact without clipping, reprojection, unit conversion, or H3 aggregation." | Tệp gốc chỉ bắt buộc với dữ liệu đã có trên đĩa và 15 ngày P1.3; lưu trữ lâu dài là bản crop miền Hà Nội + manifest hash của tệp gốc. Vẫn không đổi lưới, đơn vị, không gộp H3 trong Bronze. |
| configs/gsmap_now.yaml `bronze.raw_artifact_policy: keep_original_netcdf` | keep_original_netcdf | keep_original_netcdf cho dữ liệu hiện có; `regional_crop_with_source_hash` cho backfill. (Không sửa tệp config; quy tắc nằm ở addendum này.) |
| Chính sách Bronze IMERG (ngầm định lưu HDF5 toàn cầu) | Lưu HDF5 toàn cầu | Lưu crop miền Hà Nội + manifest; HDF5 toàn cầu hiện có được giữ. |

## 3. Không thay đổi

Chính sách của SRTM/WorldCover/OSM/flood-evidence giữ nguyên. Point-in-time: `available_time` vẫn bắt buộc cho từng cửa sổ.
