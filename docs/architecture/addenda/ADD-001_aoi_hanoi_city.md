# ADD-001 — Product AOI = toàn thành phố Hà Nội

**Trạng thái:** Approved (người dùng phê duyệt 2026-11-15; addendum được tác tử soạn trong P1.1)
**Áp dụng cho:** `problem_contract.md` v2.2, `source_matrix.md` v1.0
**Quy tắc ưu tiên:** khi mâu thuẫn, addendum này thắng contract.

## 1. Quyết định

1. AOI sản phẩm là **toàn bộ địa giới hành chính thành phố Hà Nội** (ranh giới ADM1).
2. 9 quận đô thị cũ (Ba Đình, Cầu Giấy, Đống Đa, Hà Đông, Hai Bà Trưng, Hoàn Kiếm, Hoàng Mai, Tây Hồ, Thanh Xuân) được giữ lại **chỉ như tập con so sánh**, tên mã `core9` (cờ `in_core9`, `is_core9_pixel`).
3. Độ phân giải hazard **không thay đổi**: hazard mưa giữ độ phân giải nguồn (0,1°); H3 res 9 chỉ là định danh hiển thị/không gian; `hazard resolution != H3 risk resolution` vẫn hiệu lực.
4. Miền dữ liệu khí tượng (domain) = bbox của ranh giới thành phố mở rộng 1,0° mỗi phía, để có vùng đệm cho đặc trưng không gian và kiểm tra lệch vị trí.

## 2. Ranh giới sử dụng

- Nguồn: geoBoundaries gbOpen VNM ADM1, `boundaryID = VNM-ADM1-63759600`, năm đại diện 2008, giấy phép ghi trong metadata: Public Domain (chi tiết: `data/reference/hanoi_city_boundary.provenance.json`).
- Diện tích trắc địa (WGS84): 3.214,7 km² (số chính thức ~3.359 km²; chênh lệch do đường biên đã đơn giản hóa).
- H3 res 9 (centroid trong đa giác): 27.918 ô; toàn bộ 1.481 ô core9 nằm trong.
- Lưới mưa 0,1°: 840 pixel trong domain, 47 pixel giao thành phố, 5 pixel chứa ô core9.

## 3. Câu trong contract bị thay thế

| Tệp | Dòng | Câu gốc | Thay bằng |
|---|---|---|---|
| problem_contract.md §9 | 170–172 | "Trong AOI hiện tại gồm 9 quận đô thị cũ của Hà Nội, H3 resolution 9 tạo khoảng: `1,481 cells`" | Trong AOI toàn thành phố, H3 res 9 tạo 27.918 ô; 1.481 ô thuộc tập con core9. |
| problem_contract.md §9 | 184–186 | "Trong cùng AOI, H3 resolution 8 tạo khoảng: `210 cells`" | Con số 210 chỉ áp dụng cho core9. |
| problem_contract.md §9 | 213–223 | "Project AOI được giới hạn trong footprint của 9 quận đô thị cũ: …" | Project AOI = toàn thành phố Hà Nội; 9 quận cũ là tập con so sánh `core9`. |
| source_matrix.md | 6 | "**AOI:** 9 quận đô thị cũ của Hà Nội" | AOI: toàn thành phố Hà Nội (core9 là tập con so sánh). |
| source_matrix.md §10.1 | 252–275 | Provenance AOI 9 quận (ADM2) | Vẫn đúng cho core9; AOI thành phố dùng ADM1 như mục 2 ở trên. |

Mọi quy tắc khác của contract (horizon 30/60, point-in-time, UTC, không tuyên bố xác suất ngập đã kiểm định) giữ nguyên.
