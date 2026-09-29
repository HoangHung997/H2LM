# H2LM

**H2LM V1** xây dựng mô hình multimodal **Vision-First / Document-First**, ưu tiên tài liệu
pháp lý tiếng Việt: văn bản nhiều trang, bảng, điều/khoản/điểm, dẫn chiếu và trả lời có căn cứ.
Đích inference tham chiếu: GTX 1070 8 GB VRAM. Đây là mục tiêu cần đo, chưa phải cấu hình đã nghiệm thu.

## Nguyên tắc model — phương án B

Neural weights của H2LM được khởi tạo ngẫu nhiên. Teacher có thể tạo/kiểm tra dữ liệu,
nhưng không nhập weights teacher và không mặc định output teacher là ground truth.
Mỗi nguồn cần provenance, quyền sử dụng và kiểm chứng. Không tự gọi API có phí hay đưa hồ sơ
riêng tư ra ngoài. Không hứa phán đoán pháp lý đúng khi chưa đủ bằng chứng.

## Trạng thái hiện tại

M0 có reference model PyTorch và CPU smoke tests. M1-A có **công cụ tokenizer train được trên CPU**,
fixture tiếng Việt và kiểm thử. **Chưa có checkpoint H2LM biết đọc PDF hoặc suy luận pháp luật.**
M1-B1 có importer corpus đã duyệt và bộ so sánh tokenizer có resume.
M1-B2 corpus thật/benchmark đại diện còn phải làm; vocabulary sản phẩm 2–3B chưa khóa.
Trạng thái/điểm tiếp tục: [docs/06_IMPLEMENTATION_TASKS.md](docs/06_IMPLEMENTATION_TASKS.md).

## Tải và chạy phần mới

Trong khi PR chưa merge, dùng nhánh **h2lm-m1b2-public-seed**, không tải main rồi tìm code M1.

```powershell
git clone --branch h2lm-m1b2-public-seed https://github.com/HoangHung997/H2LM.git
cd H2LM
```

Trên Windows mở **RUN_TOKENIZER_DEMO.cmd**. Script tạo môi trường riêng, cài dependency nhỏ
của tokenizer rồi train/evaluate fixture; không tải neural weights hoặc CUDA.
Python 3.11 được dùng trong CI. Hoặc chạy thủ công:

```powershell
python -m venv .venv-tokenizer
.\.venv-tokenizer\Scripts\python.exe -m pip install -e ".[tokenizer,dev]"
.\.venv-tokenizer\Scripts\python.exe scripts/tokenizer_demo.py
.\.venv-tokenizer\Scripts\python.exe -m pytest -q tests/test_m1_tokenizer.py
```

Output ở `artifacts/tokenizer/`: tokenizer, metadata và báo cáo khôi phục text.
Fixture tổng hợp gồm **24 train / 12 validation / 12 test**, không chứa căn cứ pháp lý thật.
Một tokenizer chạy được chưa có nghĩa language/vision model đã được huấn luyện.

## Tự mở và sửa

| Muốn sửa | File/thư mục |
|---|---|
| Vocabulary size, BPE/Unigram, giới hạn dữ liệu | `configs/tokenizer/h2lm_tokenizer_dev.yaml` |
| Nguồn corpus/split/checksum | `data/tokenizer_sample/manifest.json` hoặc manifest riêng |
| Code tokenizer | `src/h2lm/tokenization/` |
| Kích thước và kiến trúc neural model | `configs/model/`, `src/h2lm/modeling/h2lm.py` |
| Tiêu chí kiểm thử | `tests/` |

Source Python, cấu hình YAML/JSON và tài liệu Markdown đều mở được bằng VS Code/PyCharm.
Tokenizer tools không import PyTorch; phần model cài môi trường riêng theo editing guide.
Không tự đổi tokenizer đã gắn với model checkpoint vì IDs phải khớp embedding.

## Tài liệu nguồn

- [Product spec](docs/00_PRODUCT_SPEC.md), [Architecture](docs/01_ARCHITECTURE.md).
- [Data & teachers](docs/02_DATA_AND_TEACHERS.md), [Roadmap](docs/03_ROADMAP.md).
- [Cách tự sửa](docs/04_EDITING_GUIDE.md), [Tokenizer M1-A](docs/05_TOKENIZER.md).
- [Task status / SESSION HANDOFF](docs/06_IMPLEMENTATION_TASKS.md).

Không commit dataset lớn, hồ sơ riêng tư hoặc checkpoints vào Git.
Kết quả phải có code SHA/config/dataset hash; không dùng demo nhỏ để tuyên bố mạnh hơn model khác.

## M1-B1 mới: chuẩn bị corpus và so sánh tokenizer

Windows mở `RUN_CORPUS_BENCHMARK.cmd`. Demo import TXT **tổng hợp**, tách test khỏi đầu vào
so sánh rồi huấn luyện bốn ứng viên BPE/Unigram trên CPU. Không phải model H2LM đã hiểu PDF.
Có timeout từng ứng viên, `--resume` giữa ứng viên, kiểm tra hash và không tự ghi đè artifact.

Để dùng tài liệu thật, đọc [hướng dẫn corpus/benchmark](docs/07_CORPUS_AND_BENCHMARK.md),
chuẩn bị registry có người duyệt bên ngoài Git. Mẫu chưa được duyệt nằm ở
`data/corpus_registry/registry.example.json`. Không tự gắn dữ liệu chưa kiểm chứng nhãn đã duyệt.
Cấu hình so sánh: `configs/tokenizer/h2lm_comparison_pilot.yaml`.
Chưa gọi teacher API, thuê GPU, kết nối PC runner hoặc huấn luyện neural model sản phẩm.

## M1-B2: nguồn Công báo thật, không phải fixture

Mở `RUN_PUBLIC_LEGAL_SEED.cmd` để tải danh sách 8 văn bản Công báo đã chọn và chạy tokenizer
trên CPU. Lưu PDF nguồn, text từng trang, kiểm trùng, freeze và holdout sau lựa chọn validation.
Đây là **seed**, chưa đạt corpus sản phẩm; text-layer chưa được kiểm chứng trực quan không được
gắn nhãn người duyệt. Xem [phạm vi và cách chạy](docs/08_PUBLIC_LEGAL_SEED.md).
Kết quả thực chạy phải đối chiếu Actions/PR đúng SHA. Không có checkpoint neural hiểu PDF.
