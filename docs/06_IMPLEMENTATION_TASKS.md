# H2LM V1 — Trạng thái và SESSION HANDOFF

## Nguồn yêu cầu

Đọc README, docs/00_PRODUCT_SPEC.md, 01_ARCHITECTURE.md, 02_DATA_AND_TEACHERS.md,
03_ROADMAP.md và tài liệu task hiện tại. Giữ vision-first/document-first, đặc biệt PDF
in rồi scan lại bằng máy; tiếng Việt/pháp lý; random weights theo phương án B. Teacher
chỉ hỗ trợ dữ liệu, không nhập weights hay mặc định nhãn AI đúng. Đích GTX 1070 8 GB chưa
nghiệm thu. Source/config phải mở và sửa được; không có lõi đóng kín.

## Trạng thái và evidence lịch sử

| Task | Trạng thái / bằng chứng |
|---|---|
| M0 bootstrap | CPU CI đạt; head e135de2, run 36549909487; 3 tests, workflow cũ checkout merge a6d1d3d. |
| M1-A tokenizer | Engineering gate đạt; PR #2, head 5ddce92, run 36553295223; 40 Windows / 44 full CPU tests. |
| M1-B1 corpus tooling | Engineering gate đạt; PR #3, head 389cd0d, run 36557607218; 80 Windows / 84 local full tests. |
| M1-B2 public corpus | ACTIVE; PR #4 head c0049c7, code CI 36562685102 đạt; acquisition 36562685281 lỗi SSL issuer. Không tắt TLS. |
| M1-SCAN-A | Engineering gate đạt; PR #5 head 902d16c, run 36570835195; 54 scan tests local, 8 synthetic pages / 192 tiles. |
| M1-SCAN-B | Engineering gate đạt; PR #6 head 6151fa0, run 36582948020; 199 full CPU tests. Ba PDF thật / 10 trang, 28 vùng nháp / 12 QA nháp, đã giữ AcroForm appearance. |
| M1-EXP-01 neural micro training | Đã chạy local random-weight model 710616 tham số / 1000 bước; cần đối chiếu CI và training job trên PR head cuối. Chi tiết docs/11_NEURAL_MICRO_EXPERIMENT.md. |
| M1 tổng thể | ACTIVE: corpus/tokenizer sản phẩm chưa khóa. |
| M2–M7 sản phẩm | Chưa pretrain model lớn/production; thử nghiệm vi mô không thay gate chất lượng. |

Lịch sử bàn giao chi tiết còn ở commit 6151fa0ccc88b59eb63f2663131f4b74be7e6442,
các PR conversations và docs/05 đến docs/10. Không suy trạng thái merge từ bảng này;
đọc branch/HEAD/open PR/CI thật trước khi tiếp tục. Không ghi đè main hay nhánh cũ.

## CURRENT SESSION HANDOFF — M1-EXP-01

User yêu cầu: “Tiếp tục đi bạn tự tìm cách để huấn luyện đi k phụ thuộc vào tôi nữa”.
Tự xử lý kỹ thuật bằng tài nguyên hiện có, không yêu cầu họ gán nhãn/cài runner/bật PC để
bắt đầu. Không tự thuê GPU, gọi API trả phí, thay quyền repo hay upload hồ sơ ngoài phạm vi.
Base PR #6: 6151fa0ccc88b59eb63f2663131f4b74be7e6442,
tree eae5528e743139db8454b1a5d2a44584c878a393.
Nhánh mới: h2lm-neural-micro-training, PR base h2lm-real-scan-seed.

Đây là exception có giới hạn để thử neural thật trong lúc corpus production chưa hoàn tất.
Không đổi nhãn gốc assistant_visual_draft / training_eligible=false. Cho phép opted-in
silver AI trong experiment_only, không tự promotion thành human_verified/gold hay đưa
vào benchmark nghiệm thu sản phẩm. Không cần người dùng bấm duyệt từng trường để chạy
thử, nhưng phải báo chính xác mức tin cậy của nhãn và kết quả.

Đã triển khai generator ảnh/câu hỏi có nhãn chương trình; byte codec riêng cho micro;
answer-only loss cân bằng từng mẫu; resampler hai dải học được; auxiliary visual reading
head cho dữ liệu tổng hợp (không dùng khi generation); gradient clipping; checkpoint model,
optimizer, RNG; config/data/source hash; single-writer lock; test chạy liền và resume khớp.
Causal core H2LM sản phẩm giữ nguyên; thay encoder/resampler chỉ trong make_model của experiment.

Đã chạy bốn cấu hình/phương pháp thử local. Các lượt đầu có loss giảm nhưng che ảnh vẫn
trả lời gần như cũ. Lượt cuối tăng phụ thuộc ảnh: synthetic train 98/128 so với che ảnh
10/128; validation 15/32 so với 4/32. Bốn crop thật đã học 4/4 theo silver; hai crop DA700
không train 0/2. Đây là bộ development đã quan sát/tinh chỉnh, không independent test.
Không báo fit 4/4 thành OCR accuracy hay hiểu luật. Chi tiết và giới hạn trong docs/11.

Local là harness với các module dùng trong huấn luyện khớp blob trên base; git clone từ
container vẫn lỗi DNS, không giả Git HEAD local/full repository regression. 24 tests mới
local đạt, compile đạt. CI hiện có chạy full repository; workflow micro-neural mới tự train
1.000 bước bằng CPU trên PR cùng repo và xuất checkpoint, không tải tài liệu người dùng.
CI/checkpoint cuối phải đọc Actions theo đúng SHA và ghi evidence ở PR conversation.
Không coi việc tạo YAML là đã có kết quả remote. Các cảnh báo dependency không giấu đi.

## Điểm tiếp tục không cần chờ chủ dự án

Nếu CI đỏ: sửa chính M1-EXP-01 trước. Khi xanh: dùng checkpoint/report để cải thiện khả
năng tổng quát hóa ảnh thực; tăng đa dạng font/bố cục/chất lượng dữ liệu có nhãn tự tạo;
tự kiểm tra scan đã được cho phép, dùng silver có nguồn và đo riêng. Chưa tự tuyên bố đã
có verifier độc lập. Không dùng lại các mẫu đã học làm test độc lập, không chia rescans/crops
cùng document/family/leakage group sang các split khác. Lỗi acquisition SSL vẫn OPEN.

Ba PDF người dùng được phép công khai chỉ đúng scope/hash của data/scan_seed_20260929;
không suy sang tài liệu khác. Chỉ sáu vùng số/ngày tham gia micro run, không full transcript
10 trang hoặc bộ QA/bảng đầy đủ. DA700 là real development probe, không mẫu test niêm phong.

Mục tiêu kế tiếp là đường học vision hiệu quả và dữ liệu rộng hơn, không vội tăng model
2–3B, không hứa tương thích Ollama/GGUF hoặc chạy tốt GTX 1070 khi chưa đo. Không có training
service chạy vô hạn sau chat; job đã được khởi chạy có giới hạn và tự lưu artifact theo workflow.
