# H2LM V1 — Trạng thái và SESSION HANDOFF

## Nguồn yêu cầu

Đọc README, docs/00_PRODUCT_SPEC.md, 01_ARCHITECTURE.md, 02_DATA_AND_TEACHERS.md,
03_ROADMAP.md, 05_TOKENIZER.md, 07_CORPUS_AND_BENCHMARK.md, 09_SCAN_FIRST.md và
10_REAL_SCAN_SEED.md trước khi tiếp tục. Giữ model vision-first, tiếng Việt/pháp lý,
neural weights khởi tạo ngẫu nhiên; teacher chỉ hỗ trợ dữ liệu có nguồn/kiểm chứng.
Đích GTX 1070 8 GB chưa nghiệm thu. Không có checkpoint model sản phẩm đọc hiểu PDF.

## Trạng thái

| Task | Trạng thái và bằng chứng |
|---|---|
| M0 bootstrap | CPU CI đạt; PR #1; head e135de2, run 36549909487; 3 tests. Workflow cũ checkout merge a6d1d3d, không phải head trực tiếp. |
| M1-A tokenizer | Engineering gate đạt; PR #2, head 5ddce92, run 36553295223; 40 Windows tokenizer / 44 full CPU tests. |
| M1-B1 corpus tooling | Engineering gate đạt; PR #3, head 389cd0d, run 36557607218; 80 Windows / 84 local full tests. |
| M1-B2 public corpus | ACTIVE; PR #4 head c0049c7, code CI 36562685102 đạt; acquisition 36562685281 lỗi SSL issuer. Chưa corpus đại diện và chưa sửa TLS trong các task scan. |
| M1-SCAN-A | Engineering gate đạt; PR #5 head 902d16c, CI 36570835195 đạt; 54 scan tests local; 8 trang synthetic / 192 tile. Không phải nghiệm thu scan thật. |
| M1-SCAN-B | Có ba PDF thật / 10 trang, 28 vùng nháp / 12 QA nháp; đã sửa AcroForm appearance. Kiểm CI trên PR head mới và evidence trong PR trước nghiệm thu remote. |
| M1 tổng thể | ACTIVE; công cụ chạy không đồng nghĩa corpus/tokenizer đã khóa hoặc model đã thông minh. |
| M2–M7 | Chưa pretrain model sản phẩm. Giữ dependency/corpus gate trong roadmap. |

Chi tiết lịch sử trước khi cô đọng bảng này còn trong Git tại head 902d16cf6139d16b36aa1fba044ef7f863222da4
và các PR conversations. Không suy trạng thái merge từ bảng; đọc branch/PR/CI hiện tại.

## CURRENT SESSION HANDOFF — M1-SCAN-B

Base: PR #5, head 902d16cf6139d16b36aa1fba044ef7f863222da4;
base tree 6c18e081a7605f5f888d0eb511d417a92effcc6c.
Nhánh mới: h2lm-real-scan-seed; PR base h2lm-scan-input-v1. Không merge, không đổi main.

Người dùng cung cấp ba PDF và cho phép công khai; scope/hash/filename ở
`data/scan_seed_20260929/draft_labels.json`. PDF và ảnh lớn giữ trong bundle hội thoại;
Git chỉ chứa code, metadata, quyền được người dùng tuyên bố và nhãn nháp. Không suy quyền
công khai sang các tài liệu khác. Không gọi API teacher hay dịch vụ trả phí.

Đã chạy cả 10 trang: DA700 2 trang/48 tile; ĐT391 4 trang/80 tile; Thạch Bích 4 trang/96 tile.
File Thạch Bích bị bản cũ từ chối vì AcroForm có trường chữ ký số. Sửa bằng khởi tạo form
sớm, giữ appearance/annotations; không rewrite PDF, không xác minh mật mã chữ ký, không
thực thi document actions. XFA/unknown vẫn chặn. Không tắt TLS/không flatten để làm xanh.

DA700 có 6 dòng nhân sự: 4 ở trang đầu và 2 ở trang sau. Không dạy model chỉ đếm 4.
Thạch Bích có dòng ngày 16/06/2026 khác phần appearance 18-06-2026: lưu hai field riêng,
không kết luận hiệu lực. Nhãn/QA là assistant_visual_draft; training_eligible=false;
chưa independent human review. Chỉ 28 vùng lựa chọn, không full transcript của 10 trang.
Cả ba là development_seed, không gọi bộ mẫu nhỏ đã xem là benchmark độc lập.

Local scan harness lấy từ patch M1-SCAN-A và đối chiếu blob với GitHub trước sửa.
Không có full git clone do DNS; không giả Git HEAD local. Đã chạy tách nhóm: 54 regression
scan + 8 form + 27 review tests; compile đạt. Lượt gộp đầu bị tool timeout, không tính đạt.
CI mới chạy full repository và test Windows/Linux trên đúng head. Kết quả cuối gắn SHA
ở PR conversation để tránh vòng lặp ghi evidence rồi thay SHA mã nguồn.

## Điểm tiếp tục

Nếu CI task mới đỏ: sửa chính task mới trước. Sau đó rà nhãn độc lập, bổ sung transcript
đúng phạm vi, nhãn unreadable/uncertain/missing và bộ scan đa dạng hơn. Không tự promotion
nhãn AI thành human_verified. Các bản rescan/crop cùng document/family và mẫu quyết định
trùng phải nằm cùng nhóm khi chia train/validation/test. Hai quyết định giao nhiệm vụ
ĐT391/Thạch Bích đang cùng leakage group vì mẫu tương tự.

M1-B2 vẫn cần sửa lỗi acquisition có xác minh TLS, corpus toàn văn đủ đại diện, đánh giá
extract/vision đúng nguồn, freeze và holdout độc lập, rồi chốt tokenizer/model vocabulary.
Không dùng exact token round-trip làm OCR/reasoning accuracy. Chưa tự rotate/deskew theo
nội dung, chưa neural optimizer checkpoint/resume, chưa GPU runner hoặc GTX 1070 test.
