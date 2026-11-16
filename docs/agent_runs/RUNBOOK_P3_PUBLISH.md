# RUNBOOK P3 — Công bố repository và bật thu thập liên tục trên GitHub Actions

Người thực hiện: **người dùng** (agent không chạy bất kỳ bước nào trong tài liệu này).
Ngày soạn: 2026-11-16. Căn cứ: `TASK_P3A_CONTINUOUS_INGESTION.md`, `REPORT_P3A.md`.

Thời gian ước tính: 30–45 phút. Shell: PowerShell (các lệnh dưới đây viết cho PowerShell; đường dẫn repo có dấu cách nên luôn đặt trong nháy kép).

---

## 0. Trước khi bắt đầu — ba điều không được làm

1. **Không chạy `git add .` hay `git add -A`** trong repo hiện tại: `.gitignore` hiện chưa loại `data/` (≈ 900 tệp `data/bronze_regional`, `data/_scratch`, `data/_ci_test`), `_archive/`, `reports/`, `.env.*` (kết quả kiểm tra ở `REPORT_P3A.md` §7).
2. **Không commit 11 tệp đang staged trên `main`** (công việc cũ: flood-evidence/OSM, danh sách ở `docs/agent_runs/staged_before.txt`). Lộ trình khuyến nghị dưới đây không đụng tới repo hiện tại nên 11 tệp này giữ nguyên trạng thái.
3. **Không dán giá trị bí mật vào bất kỳ tệp nào** trong repo; giá trị chỉ nằm trong `.env` cục bộ và trong GitHub Secrets.

Đợi hai cửa sổ backfill chạy xong **không** bắt buộc: lộ trình (b) làm việc trong một thư mục mới, không ghi vào repo hiện tại; các workflow chỉ ghi dưới `lake/v1/realtime/` và `lake/v1/status/` trên R2, không chạm `lake/v1/data/...` mà `r2_sync` sẽ dùng.

---

## 1. Chọn lộ trình: khuyến nghị **(b) repository công khai mới từ snapshot sạch**

| Tiêu chí | (a) Đẩy lịch sử hiện có | (b) Snapshot sạch (khuyến nghị) |
|---|---|---|
| Bí mật trong lịch sử | Quét 61 blob của 19 commit, mọi nhánh: **0** giá trị bí mật (khớp chính xác 8 biến) và 0 phát hiện thật của detect-secrets → về mặt bí mật là an toàn | Không có lịch sử cũ |
| 11 tệp staged của công việc cũ | Phải commit cẩn thận bằng `git commit --only` để không cuốn theo; nhánh `feat/met-dataset` còn thiếu nhiều tệp chưa track | Không liên quan (repo khác) |
| Nội dung công khai | Toàn bộ dự án cũ (flood-evidence, OSM, web, tools, dữ liệu `data/reference/*.geojson` 0,9 MB trong lịch sử) — nhiều phần ngoài phạm vi, chưa rà soát cho công khai | Chỉ 38 đường dẫn đã rà soát (`docs/agent_runs/publish_paths_p3.txt`): pipeline realtime + hợp đồng + tài liệu |
| Remote hiện có (`origin` = repo GitHub hiện tại, `origin/main` = `main`) | Phải đổi visibility repo cũ thành public — công bố cả công việc cũ | Repo cũ giữ nguyên (riêng tư hay không là tùy bạn) |
| Hoạt động để tránh bị tắt lịch sau 60 ngày | Như nhau | Như nhau |
| Đã kiểm chứng | — | Snapshot dựng từ đúng danh sách: `pytest` 42 passed, 5 entry point import được |

Lý do chọn (b): ít rủi ro nhất (không công bố phần chưa rà soát, không đụng 11 tệp staged, không đổi visibility repo cũ), và đã được kiểm chứng là chạy được. Nhược điểm: repo công khai không có lịch sử commit cũ; repo hiện tại vẫn là nơi phát triển chính, mỗi khi sửa mã realtime cần sao chép lại theo danh sách (mục 9).

Nếu vẫn muốn (a), xem mục 10.

---

## 2. Tạo repository trên GitHub (giao diện web)

1. github.com → **New repository**.
2. Owner: tài khoản của bạn; tên gợi ý: `hanoi-rain-lakehouse`; **Public** (Actions miễn phí không giới hạn phút cho runner chuẩn ở repo public; repo private sẽ cần ≈ 3.000 phút/tháng > 2.000 phút miễn phí — xem `REPORT_P3A.md` §5).
3. **Không** chọn "Add a README", ".gitignore" hay "license" (tránh xung đột lần push đầu).
4. Create repository. Ghi lại URL dạng `https://github.com/<tài-khoản>/hanoi-rain-lakehouse.git`.

---

## 3. Dựng snapshot sạch (PowerShell)

```powershell
$SRC = "D:\flood lakehouse\hanoi-flood-lakehouse"
$DST = "D:\hanoi-rain-lakehouse-public"          # thư mục mới, ngoài repo hiện tại
New-Item -ItemType Directory -Path $DST | Out-Null

# 3.1 Sao chép đúng các đường dẫn đã liệt kê (dòng bắt đầu bằng # bị bỏ qua)
Get-Content "$SRC\docs\agent_runs\publish_paths_p3.txt" |
  Where-Object { $_ -and -not $_.StartsWith("#") } |
  ForEach-Object {
    $from = Join-Path $SRC $_; $to = Join-Path $DST $_
    New-Item -ItemType Directory -Force -Path (Split-Path $to) | Out-Null
    Copy-Item -LiteralPath $from -Destination $to
  }
(Get-ChildItem -Recurse -File $DST).Count     # kỳ vọng: 38
```

3.2 Tạo `.gitignore` trong `$DST` (nội dung đề xuất; cũng nên áp dụng cho repo hiện tại khi tiện):

```powershell
@"
# secrets
.env
.env.*
!.env.example
# environments / tools
.venv/
venv/
__pycache__/
*.py[cod]
.pytest_cache/
.idea/
.vscode/
.claude/
# data and generated outputs (system of record = Cloudflare R2)
data/
logs/
*.log
_archive/
audit_runs/
reports/**/*.parquet
reports/**/*.h5
reports/**/*.png
"@ | Set-Content -Encoding utf8 "$DST\.gitignore"
```

3.3 Tạo `.env.example` (chỉ tên biến, không giá trị):

```powershell
@"
# Copy to .env for local runs (never commit .env). In GitHub Actions these come from repository secrets.
EARTHDATA_TOKEN=
GSMAP_FTP_HOST=
GSMAP_FTP_USER=
GSMAP_FTP_PASSWORD=
R2_ACCOUNT_ID=
R2_ACCESS_KEY_ID=
R2_SECRET_ACCESS_KEY=
R2_BUCKET=
"@ | Set-Content -Encoding utf8 "$DST\.env.example"
```

3.4 Tạo `README.md` ngắn (bắt buộc ghi nguồn Open-Meteo theo giấy phép CC BY 4.0):

```powershell
@"
# Hanoi rain lakehouse — continuous ingestion

Scheduled GitHub Actions keep Hanoi rainfall data fresh in Cloudflare R2:
GSMaP NOW (JAXA, every 30 min), IMERG Late V07 (NASA GES DISC, daily, provisional),
Open-Meteo NWP runs (GFS Global, ECMWF IFS 0.25°, hourly check), daily compaction and an
hourly freshness monitor (a failed run = alert e-mail). Code: src/realtime/. Design: docs/.

Rainfall truth in this project is IMERG, a satellite estimate (not rain gauges).
No flood probability or FLOOD / NO FLOOD prediction is claimed.

Data sources: GSMaP NOW © JAXA EORC; IMERG © NASA GES DISC;
Weather data by Open-Meteo.com (CC BY 4.0).
"@ | Set-Content -Encoding utf8 "$DST\README.md"
```

3.5 Kiểm tra cục bộ trước khi commit (dùng Python của repo hiện tại):

```powershell
Set-Location $DST
& "$SRC\.venv\Scripts\python.exe" -m pytest -q -p no:cacheprovider tests/test_realtime.py tests/test_temporal_contract.py
# kỳ vọng: 42 passed
```

---

## 4. Commit và push (chỉ các đường dẫn đã liệt kê)

```powershell
Set-Location $DST
git init -b main
git add -- .gitignore .env.example README.md
Get-Content "$SRC\docs\agent_runs\publish_paths_p3.txt" |
  Where-Object { $_ -and -not $_.StartsWith("#") } |
  ForEach-Object { git add -- $_ }
git status --short          # đọc kỹ: đúng 41 dòng "A", không có .env, không có data/
git commit -m "Continuous ingestion of Hanoi rainfall data (GSMaP NOW, IMERG Late, Open-Meteo) to Cloudflare R2"
git remote add origin https://github.com/<tài-khoản>/hanoi-rain-lakehouse.git
git push -u origin main
```

Lần push này kích hoạt workflow `tests` (push). Mở tab **Actions** → `tests` phải xanh (≈ 1–2 phút).

---

## 5. Thêm 8 secret

Repository → **Settings → Secrets and variables → Actions → New repository secret**, lần lượt:

| Tên | Giá trị lấy từ |
|---|---|
| `EARTHDATA_TOKEN` | `.env` cục bộ (token NASA Earthdata, hết hạn 2027-01-12) |
| `GSMAP_FTP_HOST`, `GSMAP_FTP_USER`, `GSMAP_FTP_PASSWORD` | `.env` cục bộ (tài khoản JAXA G-Portal/FTP) |
| `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET` | Khuyến nghị tạo **API token R2 riêng cho CI**: Cloudflare dashboard → R2 → Manage R2 API Tokens → Create → quyền *Object Read & Write*, *Apply to specific buckets only* = bucket dự án. Token riêng có thể thu hồi độc lập với token cục bộ. |

Cách khác bằng GitHub CLI (giá trị được nhập vào prompt ẩn, không xuất hiện trong lịch sử shell): `gh secret set EARTHDATA_TOKEN` …

Không bao giờ in secret trong workflow (`echo`, `printenv`, `set -x`); test `test_workflows_never_echo_secrets` kiểm tra điều này.

---

## 6. Chạy mỗi workflow một lần bằng "Run workflow" và đọc kết quả

Tab **Actions** → chọn workflow → **Run workflow** → nhánh `main`. Chạy theo thứ tự (để freshness thấy dữ liệu của cả ba nguồn):

| # | Workflow | Thời gian kỳ vọng | Dòng log cần thấy ở bước "Run" |
|---|---|---|---|
| 1 | `gsmap-now-realtime` | 2–5 phút lần đầu (catch-up ≈ 45 cửa sổ, ~4 MB/tệp), sau đó < 1 phút | `{"event": "summary", ... "published": N, "failed": []}`; lần sau `"pending": 0..2` |
| 2 | `openmeteo-latest` | < 1 phút | `"published": [...]` hoặc `"skipped_existing"` (đã có run hiện hành) |
| 3 | `imerg-late-daily` | 2–5 phút (ngày 1 hằng tháng: +5 phút kiểm chứng tải đầy đủ ≈ 400 MB) | `"candidates": {...: "selected"}`, `"published"` ≤ 2 ngày |
| 4 | `compact-daily` | < 1 phút | `"days": {...}`; ngày không có cửa sổ realtime ghi `"no realtime windows"` (bình thường trong ngày đầu) |
| 5 | `freshness-monitor` | < 1 phút | `{"event": "freshness", "overall": "OK", "stale_sources": []}` — exit 0 (xanh) |

Cách đọc lần chạy đầu:
* Workflow đỏ ở bước *Install dependencies* → lỗi pip/wheel; mở log, đối chiếu `requirements-ci.txt` (đã kiểm tra có wheel cp314 manylinux cho mọi gói biên dịch).
* `gsmap-now-realtime` exit 2 (`source_stopped`) → FTP không kết nối được hai lần (kiểm tra 3 secret GSMaP; JAXA có thể chặn IP runner — nếu lặp lại, ghi nhận và báo agent).
* `openmeteo-latest` exit 4 → HTTP 429 (giới hạn tần suất), không thử lại; exit 3 → lỗi mạng sau 1 lần thử lại.
* `freshness-monitor` đỏ ngay sau khi bật là bình thường nếu một nguồn chưa có bản ghi (`"reason": "no record found"`); chạy lại sau khi bước 1–3 xong.
* Mọi log đã được che bí mật (`***`) bởi GitHub và bởi mã (`RedactingStream`).

Kiểm tra trên R2 (Cloudflare dashboard → R2 → bucket → `lake/v1/realtime/`): có các thư mục `gsmap_now/`, `openmeteo/`, `imerg_late/`, sau ngày đầu có `compacted/gsmap_now/`; `lake/v1/status/freshness.json` được cập nhật mỗi giờ.

---

## 7. Cảnh báo qua e-mail và giới hạn của GitHub

* **Cảnh báo = workflow thất bại.** GitHub gửi e-mail khi một lần chạy theo lịch thất bại, cho người dùng **đã sửa cron gần nhất** trong tệp workflow (người push commit đầu). Kiểm tra: Settings (tài khoản) → Notifications → *Actions* → bật "Send notifications for failed workflows only".
* **Lịch có thể trễ hoặc bị bỏ** khi GitHub tải cao (thường trễ 5–15 phút, đôi khi bỏ lượt). Không mất dữ liệu: GSMaP catch-up toàn bộ cửa sổ ~24 h chưa có trên R2; IMERG Late nhìn lại 7 ngày (tối đa 2 ngày/lần); compaction nhìn lại 3 ngày.
* **Workflow theo lịch bị tắt sau 60 ngày repo public không có hoạt động.** GitHub gửi e-mail báo trước. Cách giữ: mỗi < 60 ngày push một commit (ví dụ cập nhật `docs/`), hoặc bật lại ở tab Actions → workflow → *Enable workflow*.
* Cron dùng giờ UTC: GSMaP phút 7 và 37; Open-Meteo phút 20; freshness phút 50; compaction 01:15; IMERG Late 17:40 (ngày D đủ 48 granule ≈ 14,7 h sau khi D kết thúc).

---

## 8. Xoay vòng token Earthdata (hết hạn 2027-01-12)

Đặt lịch nhắc **2026-12-29**:
1. https://urs.earthdata.nasa.gov → đăng nhập → *Generate Token* (token mới hạn 60 ngày).
2. Cập nhật secret `EARTHDATA_TOKEN` trên GitHub (Settings → Secrets → Actions → `EARTHDATA_TOKEN` → Update).
3. Cập nhật `EARTHDATA_TOKEN` trong `.env` cục bộ (tự mở bằng trình soạn thảo; không dán vào terminal).
4. Actions → `imerg-late-daily` → Run workflow; kiểm tra `"published"` hoặc `"exists"`, không có `401`.
5. Thu hồi token cũ trên trang Earthdata.

Nếu quên: `imerg-late-daily` đỏ từ 2027-01-12, freshness báo `imerg_late` STALE sau ≤ 48 h; sau khi thay token, lần chạy kế tiếp tự lấp 2 ngày/lần trong cửa sổ 7 ngày (ngày cũ hơn: chạy tay với `python -m src.realtime.imerg_late_daily --date YYYY-MM-DD`).

---

## 9. Cập nhật mã về sau

Phát triển tiếp trong repo hiện tại (`D:\flood lakehouse\hanoi-flood-lakehouse`, nhánh `feat/met-dataset`). Khi cần đưa thay đổi realtime lên repo công khai: lặp lại mục 3.1 (ghi đè các tệp trong `$DST`), chạy mục 3.5, rồi `git -C $DST add -- <đường dẫn đã đổi>`; `git -C $DST commit`; `git -C $DST push`.

---

## 10. (Không khuyến nghị) Lộ trình (a): đẩy lịch sử hiện có

Chỉ khi bạn chấp nhận công bố toàn bộ dự án cũ:
1. Áp dụng `.gitignore` ở mục 3.2 vào repo hiện tại, kiểm tra `git status --short` không còn `data/`, `_archive/`, `.env.*`.
2. Stage **chỉ** các đường dẫn trong `publish_paths_p3.txt` (+ `.gitignore`): `git add -- <đường dẫn>`.
3. Commit **chỉ** các đường dẫn đó, để 11 tệp staged cũ không bị cuốn theo: `git commit --only -m "..." -- <cùng danh sách đường dẫn>`; sau đó `git diff --cached --name-only` phải vẫn đúng 11 tệp của `staged_before.txt`.
4. `git push -u origin feat/met-dataset`; trên GitHub đặt nhánh mặc định (hoặc merge) để workflow theo lịch chạy — **workflow theo lịch chỉ chạy trên nhánh mặc định**.
5. Đổi visibility repo thành Public (Settings → General → Danger Zone) — thao tác này công bố toàn bộ lịch sử 19 commit (đã quét: không có bí mật).
6. Tiếp tục từ mục 5.
