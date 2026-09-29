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
| M1-A công cụ tokenizer và fixture | Đã triển khai; chờ CI trên commit PR mới để nghiệm thu remote | Train/evaluate/inspect, manifest checks, byte-preserving codec, tests, Windows demo |
| M1-B corpus thật + benchmark đại diện | NEXT / chưa làm | Chưa có corpus sản phẩm đã duyệt; chưa khóa BPE/Unigram/vocab size |
| M1 toàn milestone | ACTIVE | M1-A đạt không đồng nghĩa M1 nghiên cứu đã hoàn tất |
| M2–M7 | NOT STARTED | Giữ dependency như roadmap; không pretrain model lớn trước corpus/tokenizer gates |

## Evidence M1-A trước push

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

## SESSION HANDOFF

Nhánh triển khai dự kiến: `h2lm-m1-tokenizer`, kế thừa `h2lm-v1-bootstrap`; giữ main nguyên trạng.
PR M1-A dùng base `h2lm-v1-bootstrap` để diff chỉ có một task, không trộn lại bootstrap.
Khi tiếp tục: đọc branch/PR/Actions thực tế trước, không suy trạng thái merge từ bảng này.
Nếu CI M1-A đỏ, sửa chính M1-A trước. Nếu đã xanh, tiếp M1-B (corpus và benchmark), chưa nhảy M2.
Không tuyên bố model đã thông minh vì train tokenizer thành công; không gọi teacher API có phí
hoặc upload hồ sơ người dùng khi chưa có phê duyệt nguồn/phạm vi.
