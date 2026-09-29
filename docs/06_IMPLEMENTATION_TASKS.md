# H2LM V1 — Trạng thái và SESSION HANDOFF

Đọc README, docs/00_PRODUCT_SPEC.md, 01_ARCHITECTURE.md, 02_DATA_AND_TEACHERS.md,
03_ROADMAP.md và tài liệu task hiện tại trước sửa. Vision-first, PDF in rồi scan lại tiếng
Việt/pháp lý là ưu tiên. Random weights, teacher chỉ hỗ trợ dữ liệu; source/config mở được.
Chủ dự án yêu cầu tự tìm cách train, không chờ họ gán nhãn/cài runner/bật PC. Không chi tiền
API/GPU hoặc phát hành thêm hồ sơ ngoài phạm vi đã cho phép.

## Trạng thái

| Task | Bằng chứng / giới hạn |
|---|---|
| M0 | CPU bootstrap, PR #1 head e135de2, run 36549909487. |
| M1-A | Tokenizer tooling, PR #2 head 5ddce92, run 36553295223 đạt. |
| M1-B1 | Corpus tooling, PR #3 head 389cd0d, run 36557607218 đạt. |
| M1-B2 | ACTIVE; PR #4 c0049c7; acquisition 36562685281 lỗi SSL issuer. TLS không tắt. |
| M1-SCAN-A | Engineering đạt; PR #5 902d16c, run 36570835195. |
| M1-SCAN-B | 3 PDF thật/10 trang, nhãn AI nháp; PR #6 6151fa0, run 36582948020, 199 CPU tests. |
| M1-EXP-01 | Neural 710616 tham số/1000 bước; PR #7 5bb0ea2, CI 36588561504 và train 36588561433 đạt; 223 CPU tests. Real DA700 probe 0/2, chưa tổng quát hóa. |
| M1-EXP-02 | Thử đọc trường số/ngày bằng CNN+decoder; local train/probe đã chạy. Phải đối chiếu CI và training trên head PR mới trước nghiệm thu remote. Docs/12_CROP_GENERALIZATION.md. |
| M1 sản phẩm | ACTIVE, corpus/tokenizer chưa khóa. |
| M2–M7 sản phẩm | Chưa pretrain model sản phẩm/GTX 1070 nghiệm thu; các micro experiments không thay gate. |

## CURRENT SESSION HANDOFF — M1-EXP-02

Base PR #7 head 5bb0ea21db47086373da6c7f7f20812804858f16,
tree 984191064ed12b5e80fc12e007678ff18a8fa9ea. Nhánh mới h2lm-crop-generalization-v2,
PR base h2lm-neural-micro-training. Giữ main và các PR trước nguyên trạng, không suy merge
từ bảng: đọc branch/HEAD/open PR/Actions thật. Lịch sử đầy đủ nằm trong Git/PR trước.

Một task nghiên cứu có giới hạn: mở rộng khỏi hai số 1–12 sang trường số hiệu/ngày nhiều
ký tự, train từ đầu bằng 1200 ảnh lập trình + 96 validation đáp án không trùng. Giữ tỷ lệ
crop chữ nhật; CNN + cross-attention decoder 524852 tham số. CTC chỉ là loss phụ, không OCR
text trung gian, không input đáp án/tên file vào predict. Kiến trúc one-line thử nghiệm
không thay kiến trúc model sản phẩm hoặc xóa năng lực/code EXP-01.

Local synthetic-only: 1800 bước, 88/96 validation khớp, blank 0/96, wrong-image giữ đáp án
cũ 0/96 và đọc khớp ảnh thay thế 88/96. Sáu crop thật chưa train: 0/6 theo nhãn nháp.
Thử adaptation riêng 4 crop DT391/Thạch Bích dạng silver, DA700 không vào loss. Lượt thực
chạy local fit 4/4, probe 0/2, synthetic validation 89/96; báo cáo chi tiết được giữ lại.
Không gọi 4/4 fit là model chính xác 100%, không so 88/96 trực tiếp với EXP-01 16/32 vì khác tập.

Local là harness dựa trên gói source chạy checkpoint; các dependency đã kiểm Git blob trên
base, không full clone/full repository regression. 27 tests mới local đạt. Có test resume
bitwise, causal mask, bỏ CTC khi inference, dữ liệu/tọa độ/nhãn/hash/silver-opt-in. Lint và
full regression cần đối chiếu CI đúng SHA; không dùng CI cũ cho bản mới. Workflow mới train
CPU GitHub bằng ảnh tổng hợp, không cần PC người dùng và không tải PDF riêng vào Actions.
Kết quả cuối ghi trong PR conversation theo SHA để tránh vòng lặp cập nhật tài liệu rồi đổi SHA.

## Điểm tiếp tục

Nếu CI đỏ, sửa chính EXP-02 trước. Khi xanh, đọc các lỗi số/ngày trên ảnh mới và scan thật:
bố cục dòng địa danh/ngày nhiều từ, chữ viết tay và phân bố ảnh khác render. Tự mở rộng dữ
liệu có nguồn/nhãn chương trình; có thể dùng silver trong experiment riêng nhưng không tự
nâng thành human_verified. Probe đã xem không gọi là test độc lập. Giữ toàn bộ bản scan,
ảnh crop và mẫu văn bản liên quan cùng family/leakage group, không dùng vào cả train/test.

Các bộ nguồn thật: chỉ ba PDF user đã cho phép theo data/scan_seed_20260929. Registry nhãn
gốc vẫn là assistant_visual_draft, không được đổi âm thầm. Chưa toàn văn đủ đại diện, chưa
chốt tokenizer, chưa gán nhãn bảng/pháp lý đầy đủ để train, chưa speech/video/agent stack.
Lỗi SSL Công báo vẫn OPEN, không bypass TLS. Không hứa tương thích Ollama/GGUF hay hiệu năng
GTX 1070. Mọi job mô tả trong bàn giao là đợt giới hạn, không dịch vụ train nền vô thời hạn.
