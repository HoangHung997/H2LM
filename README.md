# H2LM V1

Model **vision-first / document-first**, ưu tiên PDF tiếng Việt được in rồi scan lại,
tài liệu pháp lý nhiều trang, bảng, điều/khoản/điểm và đáp án có căn cứ. Đích chạy sản phẩm:
GTX 1070 8 GB, **chưa nghiệm thu**. Không thiết kế thành OCR ngoài rồi nhờ agent khác hiểu hộ.

## Nguyên tắc — phương án B

Neural weights khởi tạo ngẫu nhiên; teacher có thể hỗ trợ tạo/kiểm tra dữ liệu, không nhập
weights của teacher. Không mặc định output AI là ground truth. Không tự chi tiền thuê GPU,
gọi API trả phí hoặc đưa hồ sơ chưa được cho phép ra ngoài. Mã Python, YAML/JSON và tài liệu
Markdown đều mở được bằng VS Code/PyCharm; không có lõi binary bí mật để khóa người dùng.

## Mới nhất: EXP-02 — đọc trường số/ngày chưa dùng để học

Nhánh **h2lm-crop-generalization-v2** thêm một thí nghiệm CNN + decoder cross-attention,
524.852 tham số; 1.200 ảnh train / 96 validation có đáp án đầy đủ không trùng. Train thật
1.800 bước, lưu checkpoint và kiểm tra che/đổi ảnh. Đây là crop một dòng, không phải OCR
cả trang hoặc suy luận pháp luật. Scan viết tay thật vẫn còn lỗi; giữ báo cáo cả phần sai.
Không thay model/core/tokenizer sản phẩm hoặc xóa thử nghiệm trước.

```powershell
git clone --branch h2lm-crop-generalization-v2 https://github.com/HoangHung997/H2LM.git
cd H2LM
python -m venv .venv-crop
.\.venv-crop\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv-crop\Scripts\python.exe -m pip install -e ".[scan,dev]"
.\.venv-crop\Scripts\python.exe -m h2lm.experiments.crop2_train --output artifacts/my-crop-run
```

Workflow `crop-neural.yml` tự chạy một đợt có giới hạn trên CPU GitHub cho PR cùng repo;
không cần chủ dự án bật PC. Lệnh trên chỉ dành cho tự mở/sửa/chạy lại. Cấu hình nằm trong
`configs/training/crop_v2.yaml`; chi tiết và giới hạn: [EXP-02](docs/12_CROP_GENERALIZATION.md).
Nhãn silver chỉ dùng thử riêng khi bật cờ rõ ràng; không đổi nhãn AI thành nhãn người duyệt.

## EXP-01: đã chạy thử huấn luyện neural

Nhánh **h2lm-neural-micro-training** có một thí nghiệm nhỏ, không phải model sản phẩm:
710.616 tham số, vision + resampler + language decoder, checkpoint/optimizer/resume và kiểm
tra che ảnh. Đã train local từ random weights trên dữ liệu tổng hợp cùng một số nhãn scan
thật dạng silver AI. **Chưa đạt chất lượng đọc PDF mới hoặc suy luận pháp lý**; xem kết quả
thành công và thất bại tại [docs/11_NEURAL_MICRO_EXPERIMENT.md](docs/11_NEURAL_MICRO_EXPERIMENT.md).

Checkpoint EXP-01/EXP-02 không tương thích trực tiếp Ollama/GGUF. Codec byte thí nghiệm không
thay quyết định tokenizer sản phẩm. H2LM 2–3B chưa được pretrain; M1 corpus/tokenizer vẫn ACTIVE.

## Các công cụ đã có

| Công việc | Điểm chạy / tài liệu |
|---|---|
| Train và kiểm tokenizer mẫu | `RUN_TOKENIZER_DEMO.cmd`, docs/05_TOKENIZER.md |
| Chuẩn bị corpus TXT / so sánh tokenizer | `RUN_CORPUS_BENCHMARK.cmd`, docs/07_CORPUS_AND_BENCHMARK.md |
| Public legal seed | `RUN_PUBLIC_LEGAL_SEED.cmd`, docs/08_PUBLIC_LEGAL_SEED.md; lỗi acquisition TLS còn mở |
| Chuẩn bị ảnh PDF scan local | `RUN_SCAN_PREPARE.cmd`, docs/09_SCAN_FIRST.md |
| Đối chiếu ba bản scan thật | `RUN_REAL_SCAN_REVIEW.cmd`, docs/10_REAL_SCAN_SEED.md |
| Neural EXP-01 + resume | `src/h2lm/experiments/`, `configs/training/micro_vlm.yaml` |
| Neural EXP-02 + resume | `crop2_train.py`, `configs/training/crop_v2.yaml` |
| Chạy checkpoint EXP-02 trên crop | `scripts/predict_crop_v2.py` |

Không commit hồ sơ riêng tư/dataset lớn/checkpoints trực tiếp vào Git. Ba PDF được người
dùng cho phép công khai có metadata/nhãn nháp riêng; không suy quyền đó sang tài liệu khác.
Nhãn gốc `assistant_visual_draft` không bị đổi thành `human_verified` trong thử nghiệm.

## Nguồn thiết kế và trạng thái

README → [Product](docs/00_PRODUCT_SPEC.md) → [Architecture](docs/01_ARCHITECTURE.md) →
[Data/teachers](docs/02_DATA_AND_TEACHERS.md) → [Roadmap](docs/03_ROADMAP.md) →
[Editing guide](docs/04_EDITING_GUIDE.md) → [SESSION HANDOFF](docs/06_IMPLEMENTATION_TASKS.md).
Đọc thêm tài liệu task hiện tại trước sửa; giữ từng bước có bằng chứng. Không dùng loss giảm,
round-trip tokenizer hay fit vài ảnh để tuyên bố model thông minh/ngang Gemma.
