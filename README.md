# H2LM V1

Model vision-first/document-first cho tài liệu pháp lý tiếng Việt, ưu tiên PDF in rồi
scan lại. Phương án B: tự huấn luyện từ random weights, teacher chỉ hỗ trợ dữ liệu có
nguồn/kiểm chứng; không nhập weights của model khác. Source Python/YAML/JSON mở và sửa được.
Không tự thuê GPU, gọi API có phí hay đưa hồ sơ chưa được cho phép ra ngoài.

## Mới nhất — mốc 1 tỷ tham số

Theo yêu cầu chủ dự án, nhánh **h2lm-scale-1b** bổ sung một kiến trúc thực sự có
**1.001.571.584 tham số duy nhất**, gồm vision 85.543.680, fusion 8.485.632 và language
907.542.272. Không cộng các checkpoint, không thêm tensor thừa hoặc nhân bản alias để đủ số.

**Đạt quy mô 1B không có nghĩa đã pretrain xong hoặc đọc hiểu PDF tốt.** Lượt hiện tại là
pilot tám bước cập nhật toàn bộ kiến trúc và kiểm checkpoint. Corpus, OCR/vision chất lượng,
suy luận pháp lý, tokenizer sản phẩm, Ollama/GGUF và GTX 1070 vẫn chưa nghiệm thu.
Không thay kết quả các thí nghiệm nhỏ trước bằng con số 1B để tuyên bố thông minh hơn.

```powershell
git clone --branch h2lm-scale-1b https://github.com/HoangHung997/H2LM.git
cd H2LM
python -m venv .venv-scale
.\.venv-scale\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv-scale\Scripts\python.exe -m pip install -e ".[scan,dev]"
.\.venv-scale\Scripts\python.exe -m h2lm.scale.runner count
```

Count dùng meta, không cấp phát hàng GB. Train/verify lớn cần cờ `--allow-large` và kiểm
bộ nhớ/đĩa; đọc [docs/13_SCALE_1B.md](docs/13_SCALE_1B.md) trước khi chạy. Các lệnh chỉ là
để chủ dự án tự mở/sửa/chạy lại, không yêu cầu bật PC của họ để tôi triển khai tiếp.

| Muốn sửa | Vị trí |
|---|---|
| Kích thước 1B, vision/decoder, context | `configs/model/h2lm_v1_1b.yaml` |
| Kiến trúc 2D vision/resampler/GQA decoder | `src/h2lm/scale/model.py` |
| Huấn luyện pilot và đếm tham số | `src/h2lm/scale/runner.py` |
| SGD chuẩn/optimizer-in-backward ít RAM | `src/h2lm/scale/optim.py` |
| Checkpoint nhiều shard, hash và load | `src/h2lm/scale/storage.py` |
| Bằng chứng/gate quy mô/chất lượng | `docs/13_SCALE_1B.md`, PR conversation/Actions đúng SHA |

Workflow `scale-1b.yml` kiểm toàn bộ model 1B FP32 trên standard public Ubuntu CPU. Không
thay bằng tiny ở bước chứng minh quy mô. Tiny tests chạy riêng. Artifact GitHub chỉ giữ
report/index/source nhỏ, **không giữ checkpoint nhiều GB**; trọng số local giao riêng trong
hội thoại. Không claim file index là trọng số tải được. Chưa nghiệm thu BF16/GPU GTX 1070.

## Các phần trước vẫn được giữ

| Công việc | Điểm chạy / tài liệu |
|---|---|
| Tokenizer riêng | `RUN_TOKENIZER_DEMO.cmd`, docs/05_TOKENIZER.md |
| Corpus TXT / so sánh tokenizer | `RUN_CORPUS_BENCHMARK.cmd`, docs/07_CORPUS_AND_BENCHMARK.md |
| Public legal seed | `RUN_PUBLIC_LEGAL_SEED.cmd`, docs/08_PUBLIC_LEGAL_SEED.md; TLS acquisition còn lỗi |
| Chuẩn bị ảnh PDF scan local | `RUN_SCAN_PREPARE.cmd`, docs/09_SCAN_FIRST.md |
| Ba bản scan thật, nhãn AI draft | `RUN_REAL_SCAN_REVIEW.cmd`, docs/10_REAL_SCAN_SEED.md |
| EXP-01 neural 0,71M | `src/h2lm/experiments/`, docs/11_NEURAL_MICRO_EXPERIMENT.md |
| EXP-02 neural crop 0,525M | `crop2_train.py`, docs/12_CROP_GENERALIZATION.md |

EXP-02: validation synthetic local 88/96, GitHub 82/96; ảnh thật chưa fit 0/6. Adaptation
riêng fit bốn crop 4/4 nhưng DA700 chưa fit vẫn 0/2. Đó là lỗi còn mở, không bị xóa khi tạo
1B. Các bộ đo khác nhau không được gộp hoặc so trực tiếp. Nhãn gốc vẫn assistant_visual_draft,
không tự nâng thành human_verified; ba PDF được phép công khai chỉ theo scope/hash đã ghi.

## Nguồn thiết kế

README → docs/00_PRODUCT_SPEC.md → docs/01_ARCHITECTURE.md → docs/02_DATA_AND_TEACHERS.md
→ docs/03_ROADMAP.md → docs/06_IMPLEMENTATION_TASKS.md → tài liệu task hiện tại.
Đích inference GTX 1070 8 GB vẫn là mục tiêu cần đo; số bytes weights không đại diện toàn
bộ RAM/VRAM chạy model. Chưa có checkpoint sản phẩm đủ tin cậy để đưa ra kết luận pháp lý.

## CQ-01 — Huấn luyện bám bằng chứng, giữ quy mô 1B

Nhánh mới `h2lm-curriculum-quality-v1` kế thừa **h2lm-scale-1b tại 138ed6e**; model hiện tại
có 1.001.571.584 tham số. Không trộn với gói local cũ 1.000.032.768 tham số khác kiến trúc.

Đã triển khai factory sáu task có nhãn từ chương trình, cặp cùng câu hỏi/khác ảnh/khác đáp
án, lịch đọc -> định vị/vai trò -> kết hợp hai trang, loss có trọng số căn cứ và kiểm che ảnh.
Cấu hình pilot 48 bước không phải đã hoàn tất pretrain. Runner chấm sáu cặp validation định
trước; bộ holdout synthetic không dùng để train/chọn model. Scan thật chưa được nghiệm thu.

Xem [chiến lược và phần đã/chưa làm](docs/14_TRAINING_CURRICULUM.md), config
`configs/training/curriculum_q1.yaml`, source `src/h2lm/curriculum/` và SESSION HANDOFF.
Workflow mới chạy full 1B FP32 và giữ báo cáo + source, không tự thuê GPU/API và không upload
trọng số nhiều GB vào Actions storage. Kết quả cuối phải đọc từ đúng SHA trong PR mới.
