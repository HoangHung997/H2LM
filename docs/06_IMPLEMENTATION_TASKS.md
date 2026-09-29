# H2LM V1 — Trạng thái và SESSION HANDOFF

Đọc README, Product/Architecture/Data/Roadmap và tài liệu task hiện tại trước sửa.
Vision-first, PDF scan lại tiếng Việt/pháp lý. Random weights, teacher hỗ trợ dữ liệu có
nguồn, source/config sửa được. User muốn tự tiếp tục không chờ họ gán nhãn/cài runner/bật PC.
Không tự chi tiền GPU/API, đổi quyền repo hoặc dùng hồ sơ ngoài scope cho phép.

## Trạng thái

| Task | Bằng chứng / giới hạn |
|---|---|
| M0 | Bootstrap PR #1 e135de2, run 36549909487. |
| M1-A | Tokenizer PR #2 5ddce92, run 36553295223 đạt. |
| M1-B1 | Corpus tooling PR #3 389cd0d, run 36557607218 đạt. |
| M1-B2 | ACTIVE; PR #4 c0049c7, acquisition 36562685281 lỗi SSL issuer, không tắt TLS. |
| M1-SCAN-A | PR #5 902d16c, run 36570835195 đạt. |
| M1-SCAN-B | PR #6 6151fa0, run 36582948020, 199 tests CPU; 3 PDF/10 trang, nhãn AI draft. |
| M1-EXP-01 | PR #7 5bb0ea2, CI 36588561504/train 36588561433 đạt; 223 CPU tests. |
| M1-EXP-02 | PR #8 2b964b3, CI 36595618918/train 36595618872 đạt; 250 CPU tests. Synthetic validation local 88/96, remote 82/96; real DA700 0/2 vẫn chưa đạt. |
| SC-1B | Theo yêu cầu mới: 1.001.571.584 tham số đã khởi tạo/train pilot local; kiểm CI, 1B FP32 và head cuối ở PR mới trước nghiệm thu remote. |
| Model/corpus sản phẩm | Chưa pretrain đủ corpus hoặc đạt chất lượng đọc/suy luận; quy mô 1B là gate riêng, không tự đóng M1–M7. |

## CURRENT HANDOFF — SC-1B

User: “Tiếp tục làm cho đến đủ 1 tỉ tham số đi”. Base PR #8 head
2b964b3807fb71c7339639babde1b35981825397, tree 0751cfa09771c8c21070d913cca1fbc6c648c680.
Nhánh mới h2lm-scale-1b, PR base h2lm-crop-generalization-v2. Không merge/đổi main hoặc
các PR trước. Đọc trạng thái thật, không suy merge/CI từ bảng. Docs/13_SCALE_1B.md là
phạm vi chi tiết, giới hạn optimizer/bộ nhớ, checkpoint và các lệnh tự mở/sửa.

Mốc mới có 1.001.571.584 unique trainable parameters, 361 tensors, zero aliases:
vision 85.543.680; fusion 8.485.632; language 907.542.272. Chạy được cả image+text và text-only,
image prefix không thấy text tương lai, giữ geometry của tile/trang. Không thêm dummy
parameters hay cộng số từ checkpoints. Byte vocab 266 là quyết định prototype có khai báo,
không phải tokenizer sản phẩm đã khóa. Chưa KV cache, quantization, Ollama/GGUF hoặc GPU test.

Local CPU 4-GiB cgroup dùng BF16 file-backed parameters + streaming SGD. Đã chạy đủ 8 bước,
mỗi bước 361 tensor có gradient khác không; nhiều giá trị không đổi vì làm tròn BF16,
audit ghi số element thực sự cập nhật. Không tuyên bố tất cả một tỷ scalar đều đã học.
Checkpoint 36 shard ~2 GB lưu và reload độc lập; prediction trên fixture còn rỗng/EOS,
chưa đọc đúng. Đây là pilot kiểm đường huấn luyện, không full pretraining. Không che lỗi
này bằng count hay loss giảm. 43 tests mới local đạt. Lỗi version-counter khi dùng shared
arena views được sửa trước full run, kiểm chứng disjoint storage và SGD equivalence.

Local source harness dựng từ gói trước; git clone DNS thất bại nên không giả full local
checkout/HEAD. Các module cũ không bị thay. GitHub full regression và scale workflow phải
kiểm trên head mới, không lấy số CI cũ. Lint/source sửa thì retrain checkpoint theo source
hash cuối, không đổi nhãn weights cũ. Kết quả cuối ghi vào PR conversation theo đúng SHA.

Scale workflow: standard public Ubuntu CPU, tám bước full 1B FP32 rồi fresh-process reload.
Không reduced model trong bước quy mô; tiny unit tests riêng. Job timeout 20 phút, no
self-hosted runner, no paid API/GPU, no user PDFs. Artifact chỉ nhỏ gồm reports/index/source,
không tự upload 4 GB weights lên Actions storage; checkpoint BF16 giao ở hội thoại.
Không claim remote checkpoint tải được khi runner đã xóa còn artifact chỉ có index.

## Điểm tiếp tục

Nếu CI/lượt full 1B đỏ, sửa đúng SC-1B trước. Khi đạt mốc quy mô, GIỮ 1B và chuyển trọng tâm
sang corpus + trainer dài hạn + chất lượng vision/pháp lý; không tăng tham số thêm để né
lỗi đọc scan. Pilot hiện giới hạn 64 bước tổng, SGD không momentum/accumulation/global
clipping. Trainer sản phẩm cần thêm optimizer/schedule/validation/curriculum đáng tin,
được triển khai và đo riêng, không khai rằng pilot đã có đầy đủ.

Không tiếp tục chờ chủ dự án thao tác: tự dùng dữ liệu tạo có nhãn và nguồn được phép,
kiểm/sửa nhãn AI có provenance nhưng không tự đổi thành human_verified. Real development
probe đã xem không gọi benchmark độc lập. Giữ rescans/crops/phiên bản và template liên quan
cùng split/family. Lỗi TLS Công báo vẫn OPEN, không bypass. GTX 1070 8 GB chưa nghiệm thu.

## CURRENT HANDOFF — CQ-01 (supersedes previous next-step scope)

User yêu cầu đào tạo 1B để nhìn scan và suy luận tốt nhất có thể, không chờ họ vận hành.
Base thực tế: PR #9, head 138ed6ea82299eeb77fbab67621fe822a38271f9 / tree
810703d380c54e36ebdf38aba8c4b2e1630bf33e. Source ZIP từ artifact 11050340123 đã kiểm SHA256
và Git tree khớp. Đây là full source được dựng local có commit tái dựng riêng, không bịa
local HEAD là remote HEAD. Đọc docs/14_TRAINING_CURRICULUM.md trước khi tiếp tục.

Nhánh CQ-01: h2lm-curriculum-quality-v1, base h2lm-scale-1b. Không merge/đổi main/các PR cũ.
Giữ 1.001.571.584 tham số hiện trên GitHub; không sử dụng ZIP cũ có kiến trúc 1.000.032.768.
Artifact base chỉ có index/report, không weights; lượt CQ-01 khởi tạo từ đầu có khai báo,
không gọi là đã resume model cũ. Không nhập teacher weights, không gọi API/GPU trả phí.

Có factory 96 family /1152 bài: 768 train,192 validation,192 holdout; giữ cặp phản chứng và
các biến thể cùng split. Sáu task: read,locate,compare,table_sum,role,missing. Nhãn solver
cho thẻ giả lập, không tư vấn luật hay gold scan thật. Giữ input 256x256 không âm thầm resize.
Training curriculum 48 bước, schedule warmup/cosine, loss answer/evidence/syntax phân biệt.
Chấm sáu cặp validation định trước với ảnh thật/blank/counterfactual, không chấm holdout.
Checkpoint/resume có contract model/source/data/plan; pilot SGD không momentum/global clip
và không phải trainer AdamW production. CI xanh không đồng nghĩa chất lượng AI xanh.

32 tests CQ-01 local ban đầu đạt; kết quả full repository/training cuối và số step/token
thực được ghi riêng trong PR conversation/report theo đúng head cuối. Không lấy CI cũ
thay CI mới. Không gọi việc tạo1152 mẫu là đã học hết1152 mẫu. Không báo ảnh mô phỏng là
scan vật lý. Chưa có pretraining dài hạn, benchmark thực độc lập hoặc GTX1070 nghiệm thu.

Điểm tiếp tục: sửa CQ-01 nếu có lỗi; đánh giá read+evidence và thất bại sinh tự do trước khi
mở rộng. Triển khai nền ngôn ngữ/vision cùng corpus đủ đa dạng, mục tiêu thật theo các pha
trong docs/14. Dùng nhãn AI có provenance và mức tin cậy rõ, không tự đổi thành human_verified.
Ba PDF người dùng chỉ là development đã được xem; không đưa cùng bản scan/template vào
train và test. Lỗi acquisition TLS vẫn OPEN, không bypass. Không dịch vụ chạy nền vô hạn.
