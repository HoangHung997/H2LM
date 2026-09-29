# M1-B2 — Bộ văn bản Công báo thật: seed nghiên cứu tokenizer

## Phạm vi

Lần này triển khai đường chạy tải **8 văn bản thật** từ các trang đã chọn của Công báo,
trích lớp chữ PDF, lưu bản nguồn/hash, kiểm tra trùng giữa các tập, train các tokenizer và
đánh giá một ứng viên trên holdout sau khi ghi quyết định từ validation.
Kết quả thực tế và lỗi truy cập phải đọc ở Actions/PR trên đúng SHA; có code không đồng nghĩa
đã tải thành công. Đây không phải dự án tạo thêm dữ liệu tổng hợp thay văn bản thật.

**Không phải corpus sản phẩm đã nghiệm thu. Không phải OCR, vision training hay neural checkpoint.**
Chưa đủ các cổng 30 family train / 10 family validation của cấu hình pilot. Cấu hình
`h2lm_comparison_pilot.yaml` được giữ nguyên. Cấu hình seed riêng có tên và mode rõ ràng,
không đổi nhãn dữ liệu để vượt gate. M1-B2 vẫn ACTIVE cho tới khi có corpus đại diện được duyệt.

## Nguồn và phiên bản

`data/public_legal_seed/catalog.json` ghi danh mục, trang gốc và phân tập **trước khi chạy**:

| Tập | Văn bản | Nhóm dùng để chống nhiễm |
|---|---|---|
| Train | Luật 22/2023/QH15 | procurement |
| Train | Luật 20/2023/QH15 | electronic-transactions |
| Train | Thông tư 10/2021/TT-BXD và 09/2021/TT-BXD | construction-housing (gộp bảo thủ) |
| Validation | Nghị định 30/2020/NĐ-CP | records-administration |
| Validation | Nghị định 43/2022/NĐ-CP | water-infrastructure |
| Holdout | Nghị định 13/2023/NĐ-CP | personal-data |
| Holdout | Thông tư 01/2020/TT-BTP | certification |

Các văn bản là **bản theo lần công bố**, không hợp nhất sửa đổi. Không dùng metadata website
để khẳng định hôm nay văn bản còn hiệu lực. Chỉ có 7 nhóm bảo thủ; không phải đồ thị quan hệ
pháp luật đầy đủ và chưa chứng minh không có tất cả dẫn chiếu giữa các nhóm.

Nguồn: Công báo nước CHXHCN Việt Nam — `congbao.chinhphu.vn`; liên kết PDF của chính trang
Công báo trên `g7.cdnchinhphu.vn`. Chỉ chọn phần văn bản quy phạm được công bố, không cào bài
bình luận/ảnh báo chí. Catalog ghi attribution và căn cứ xem xét phạm vi quyền sử dụng.
Công báo yêu cầu ghi rõ nguồn khi tái phát hành; bản snapshot phải đi kèm nguồn/catalog.
Đối chiếu phạm vi văn bản không thuộc bảo hộ tại khoản 2 Điều 15 Luật SHTT bản WIPO Lex VN063:
https://www.wipo.int/wipolex/en/legislation/details/12011
Không suy từ đó rằng mọi website, dataset, bình luận hoặc output teacher đều được tự do dùng.

## Không giả mạo người kiểm chứng

Origin mới: `official_pdf_text`. Metadata bắt buộc `human_verified: false`,
`scope: tokenizer-seed-only`, phương pháp/parser, URL, hash PDF/text/page dump.
`approved: true` trong origin này **chỉ cho phép thử nghiệm tokenizer seed**, không phải duyệt
độ đúng pháp lý/chữ trích. Không gắn nó thành `human_verified` hoặc `teacher_verified`.
Mode pilot từ chối origin này kể cả các ngưỡng đếm đạt. Muốn đưa vào corpus sản phẩm phải có
review nội dung/nguồn/độ đại diện thực sự, không chỉ sửa một boolean.

pypdf chỉ trích text layer, không nhìn trang, không OCR scan, không kiểm được text layer có khớp
ảnh không. Không sửa dấu/chính tả/bảng hay suy luận luật ở bước này. Lưu nguyên từng trang được
parser trả về và TXT nối bằng hai LF; offset là ký tự trong TXT, **không phải bbox PDF**.
Các kiểm tra cơ học: đúng mã văn bản ở hai trang đầu, số trang, giới hạn tài nguyên, lượng chữ,
anchor Điều, ký tự lỗi/private-use. Trang ít chữ hoặc không có text bị quarantine: không bỏ
trang âm thầm, không tự gọi OCR. Dữ liệu có thể vượt gate cơ học nhưng vẫn sai thứ tự bảng/chữ;
không dùng làm ground truth vision hoặc tư vấn pháp lý.

## Chạy trên máy cá nhân

Windows mở `RUN_PUBLIC_LEGAL_SEED.cmd`, hoặc:

```powershell
python -m venv .venv-seed
.\.venv-seed\Scripts\python.exe -m pip install -e ".[public_seed]"
.\.venv-seed\Scripts\python.exe scripts/public_legal_seed.py --allow-network
```

Chạy bằng CPU. Mỗi lần tạo thư mục khác trong `artifacts/public-legal-seed/`.
Không có `--allow-network` thì không tải. Chỉ allowlist hai host HTTPS; redirect ngoài danh sách,
robots cấm, PDF nhiều phần/không khớp số hiệu, file vượt giới hạn đều làm job lỗi. Không đoán URL.
Kiểm robots trước mỗi tài nguyên, chạy tuần tự; không crawl các trang liên quan.
Tối đa 16 MiB/PDF, tổng 80 MiB, 200 trang/tài liệu, subprocess parser 180 giây,
Linux giới hạn address space 2 GiB; không coi đây là sandbox an toàn cho PDF bất kỳ.
Không tự cài driver CUDA, runner máy cá nhân, thuê GPU hay gọi API có phí.

## Chống trùng và khóa đánh giá

Đã có exact/family checks của M1-A/B1. Thêm so sánh 5-word shingle trên toàn văn và từng trang
ở hai split khác nhau. Báo nghi trùng khi có >=150 shingle chung và Jaccard >=0.85 hoặc
containment >=0.95. Không tự xóa đoạn/đổi split sau khi thấy điểm tokenizer. Đây là thuật toán
nhỏ, không phải tìm paraphrase; có thể bỏ sót câu ngắn, scan/OCR khác nhiều và các quan hệ ngữ nghĩa.

`corpus-freeze.json` khóa hash manifest và mọi shard **trước so sánh**. Chỉ validation được dùng
chọn trong 4 ứng viên BPE/Unigram 8192/16384. Ghi `selection.json` với model/hash/freeze/comparison
trước khi mở holdout để tính điểm. Lần chạy lại cùng thư mục đánh giá bị từ chối. Check hash có thể
đọc bytes holdout, nhưng không dùng nội dung/metrics để chọn. Kết quả vẫn `production_ready: false`.

Đây là **quy trình tái lập**, không phải kho test bất khả xâm phạm: chủ file vẫn có thể đổi code,
copy thư mục hoặc reset kết quả. Test seed công khai không thay được holdout sản phẩm độc lập,
quyền truy cập riêng và người nghiệm thu khác. Không được chọn lại theo điểm test đã xem.

## GitHub / evidence

PR nhánh `h2lm-m1b2-public-seed`, base `h2lm-m1b-corpus`. Workflow public-seed chỉ chạy nhánh
này từ cùng repo trên runner CPU GitHub, quyền contents:read, không secrets, không self-hosted.
Job giới hạn 25 phút, lưu snapshot công khai và reports làm artifact 14 ngày (không phải kho lâu dài).
Không có write-back tự động lên Git và không upload hồ sơ người dùng.

Nếu nguồn ngoài mạng lỗi, job phải báo lỗi; không lấy kết quả synthetic thay thế. Full local tests
và public-data job có phạm vi khác nhau. CI thành công về tokenizer không chứng minh GTX 1070
chạy model được, không chứng minh H2LM hiểu luật hay vượt Gemma.

## Các việc còn lại

Đọc kết quả nguồn thật; kiểm định trực quan/PDF-vs-text, bổ sung corpus theo thể loại, thời kỳ,
địa phương, bảng, scan; giữ family và near-dup tách biệt; review quyền dùng của teacher từng nguồn;
chạy pilot đủ đại diện rồi mới khóa vocab sản phẩm và đi tiếp M2. Không tự hạ gate để sang train lớn.

Tham khảo API trích text và giới hạn: https://pypdf.readthedocs.io/en/5.9.0/user/extract-text.html
