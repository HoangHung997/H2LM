# H2LM V1 — Trạng thái triển khai và SESSION HANDOFF

## Nguồn yêu cầu

Đọc README → docs/00_PRODUCT_SPEC.md → 01_ARCHITECTURE.md → 02_DATA_AND_TEACHERS.md
→ 03_ROADMAP.md → 05_TOKENIZER.md trước khi tiếp tục.
Giữ phương án B: neural weights từ khởi tạo ngẫu nhiên, teacher chỉ tạo/kiểm tra dữ liệu có nguồn.
Đích tham chiếu GTX 1070 8 GB; chưa có nghiệm thu GPU thật hay checkpoint model sản phẩm.

## Trạng thái

| Task | Trạng thái | Bằng chứng / còn thiếu |
|---|---|---|
| M0 bootstrap | CPU CI đạt; PR #1 còn mở tại lúc khởi tạo M1-A | Head e135de2a8017d3801a7abb265508bd7b2967a8fe; run 36549909487, 3 tests đạt; CI cũ checkout merge a6d1d3dc5168ef768a3f74da2d795225389e68ee |
| M1-A công cụ tokenizer và fixture | Engineering gate đạt, PR #2 còn mở | Head 5ddce928b934d91601039cea5f7e2562df983df6; run 36553295223 thành công; 40 tokenizer tests Windows, 44 full tests CPU |
| M1-B1 importer + so sánh tokenizer | Engineering gate đạt; PR #3 mở, head 389cd0d; run 36557607218 thành công | Offline reviewed TXT registry, dev/holdout separation, bounded subprocess trials, resume, tests, docs |
| M1-B2 corpus thật + benchmark đại diện | ACTIVE: thêm public Gazette seed; kiểm evidence thực chạy trên PR mới | 8 nguồn định trước, PDF text-layer/mechanical scope, near-dup + freeze + holdout; chưa corpus sản phẩm đã duyệt |
| M1-B tổng thể | ACTIVE | M1-B1 công cụ không thay thế thu thập/kiểm định corpus thật |
| M1 toàn milestone | ACTIVE | M1-A đạt không đồng nghĩa M1 nghiên cứu đã hoàn tất |
| M2–M7 | NOT STARTED | Giữ dependency như roadmap; không pretrain model lớn trước corpus/tokenizer gates |

## Evidence M1-A trước push (lịch sử; CI mới đã đối chiếu ở bảng trên)

Môi trường thực thi local: Linux, Python 3.13, SentencePiece 0.2.1, PyTorch 2.10.0+cpu.
Full suite: **44 passed**; compileall đạt. Có warning SWIG/PyTorch từ dependency, không phải lỗi test.
Không có GTX 1070 hoặc Windows local trong lần thử này; chờ Windows CI riêng.
Model M0 giữ nguyên, chỉ đổi package lazy import và tách dependency model để tokenizer không tải torch.

Corpus fixture: 24 train / 12 validation / 12 test, tất cả là ví dụ tổng hợp, không phải luật thật.
Tokenizer BPE: 1024 tokens. Validation 12/12 exact, 0 unknown, 391 tokens.
Test 12/12 exact, 0 unknown, 386 tokens. Chỉ là số đo khôi phục chữ, không phải reasoning/OCR score.
Có kiểm thử train lại cùng cấu hình ra cùng model bytes trong môi trường kiểm thử.

Local được dựng từ nội dung GitHub ở base SHA trên vì git clone trực tiếp từ container không kết nối được.
Không giả định Git HEAD local là SHA remote. Reports ghi source hash và `code_revision: null` khi không có Git.
CI mới checkout trực tiếp `pull_request.head.sha` và ghi bằng chứng trên code SHA thật.
Kết quả CI sau push được ghi vào PR conversation/Actions để không tạo vòng lặp sửa evidence rồi lại đổi SHA.

## SESSION HANDOFF — M1-B1

Nhánh triển khai `h2lm-m1b-corpus`, kế thừa head M1-A `5ddce928b934d91601039cea5f7e2562df983df6`.
PR base `h2lm-m1-tokenizer`; giữ main và hai PR trước nguyên trạng. Đọc branch/PR/Actions thật
trước khi tiếp tục, không suy merge từ bảng này. Đọc thêm docs/07_CORPUS_AND_BENCHMARK.md.

M1-B1 chỉ tạo công cụ chuẩn bị TXT đã duyệt và so sánh tokenizer trên validation; demo/CI dùng
fixture tổng hợp, không có dataset pháp lý thật, không gọi teacher/API có phí, không có GPU runner.
Nếu CI đỏ sửa M1-B1 trước; nếu xanh tiếp M1-B2 thu thập/kiểm định corpus thật, chống nhiễm theo
family và gần đúng, khóa holdout/evaluation độc lập, chọn vocabulary. Chưa nhảy M2–M7.
Các ngưỡng trong config pilot là cổng kỹ thuật dự kiến, không phải bằng chứng tính đại diện.

Base source ZIP trong hội thoại đã được kiểm hash toàn bộ Git tree: d5ea17d9b01d455ad96b95ee699b7b662d20ab76.
Container không kết nối DNS để git clone; làm việc trên bản sao đã kiểm hash, không bịa Git HEAD local.
CI mới phải checkout đúng PR head và lưu evidence; kết quả cuối ghi ở PR conversation để tránh vòng lặp đổi SHA.

Giới hạn còn lại: importer TXT bounded/in-memory; chưa PDF ingestion; chưa tự dò family/near-duplicate;
chưa formal holdout freeze; resume giữa ứng viên tokenizer, chưa neural checkpoint/optimizer resume;
chưa kiểm thử GTX 1070 thật, không có H2LM neural checkpoint đã hiểu tài liệu.

## SESSION HANDOFF — M1-B2 public seed

Base 389cd0d383bfcc2662764278de6ca5727992cbae, source tree bb0c77f26f0c91b69dad0f8bb77c3024591fbfd4
đã đối chiếu với ZIP bằng toàn bộ Git tree hash trước sửa. Nhánh h2lm-m1b2-public-seed;
PR base h2lm-m1b-corpus. Giữ main và PR trước nguyên trạng. Đọc docs/08_PUBLIC_LEGAL_SEED.md.

Không suy job tải corpus đã thành công từ code. Đọc workflow public-legal-seed và CI trên đúng HEAD.
Origin official_pdf_text là seed-only, không fake human review. Corpus pilot gate giữ nguyên.
Chưa có neural checkpoint/GTX 1070 test, không teacher API, không chi tiền.
Nếu acquisition thất bại do nguồn/robots/parser, báo đúng và sửa trong M1-B2; không fallback fixture.
Nếu seed chạy xong, dùng snapshot và báo cáo để kiểm định/mở rộng corpus M1-B2 trước M2.
Kết quả cuối ghi vào PR conversation gắn SHA, tránh sửa evidence rồi đổi code SHA vô hạn.

## SESSION HANDOFF — M1-SCAN-A (ưu tiên mới: tài liệu scan lại)

Nguồn tiếp tục thực tế: PR #4, head c0049c77a71a30ab7fad20136de5079a95194bb4,
tree 809f06a90b0b957b9d4b66db893bd1814d977a2d. Không lấy ZIP M1-B2 cũ ghi đè nhánh.
CI mã run 36562685102 thành công; acquisition run 36562685281 thất bại xác minh SSL certificate.
Giữ lỗi này OPEN, không hạ bảo mật TLS. Corpus sản phẩm/tokenizer selection M1 vẫn ACTIVE.

Theo yêu cầu mới, làm một task dữ liệu độc lập M1-SCAN-A: nhận PDF local kể cả scan không chữ,
render/tile, giữ nguồn, tọa độ, giới hạn xử lý, preview và regression stress cases. Đây là bước
chuẩn bị dữ liệu trong M1; không chuyển M2 pretraining. Spec scan và điểm sửa: docs/09_SCAN_FIRST.md.
Nhánh h2lm-scan-input-v1 kế thừa PR #4; chưa merge các PR hay đổi cấu hình neural model.

Bằng chứng trước push: 54 tests scan local đạt trên Linux/Python 3.13.5/PDFium wrapper 5.8.0/
Pillow 12.3.0; 8 trang mô phỏng scan, 192 vùng ảnh tại cấu hình mặc định. Lỗi context manager
PdfPage và cấu tạo PDF fixture lớp chữ ẩn được phát hiện rồi sửa, test chạy lại đạt.
Đây là test riêng các module scan; không nhận bản dựng local M1-B1 là full remote PR #4.
CI cập nhật cài scan extra, chạy cả test cũ lẫn test scan trên đúng HEAD và xuất git archive.
Kết quả CI cuối ghi vào PR discussion, không suy GPU test từ CI CPU.

Còn phải làm: lấy PDF scan thật được người dùng cho phép; đối chiếu từng vùng, gán unreadable/
missing/uncertain trung thực; tách toàn bộ các bản scan/ảnh cắt/phiên bản cùng family về một split;
đo CER và lỗi chữ số/dấu riêng, huấn luyện vision sau khi corpus gate đạt. Synthetic rescan
không thay nghiệm thu scanner thật. Output scan chưa có text label, không train-ready.
Chưa auto-rotate/deskew, chưa OCR, chưa teacher call, chưa neural train hay GPU nghiệm thu.
