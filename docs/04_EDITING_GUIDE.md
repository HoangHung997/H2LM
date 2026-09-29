# Cách tự mở và sửa H2LM

Mục tiêu của repository là không khóa người dùng vào một công cụ riêng.

## 1. Mở project

Có thể clone repository và mở trực tiếp bằng VS Code hoặc PyCharm.
Nhánh triển khai hiện tại là `h2lm-m1-tokenizer`; main chưa tự động nhận PR chưa merge.

Windows PowerShell:

    git clone --branch h2lm-m1-tokenizer https://github.com/HoangHung997/H2LM.git
    cd H2LM
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -e ".[tokenizer,dev]"

Chỉ làm tokenizer thì không cần PyTorch/GPU. Hướng dẫn đầy đủ: `docs/05_TOKENIZER.md`.
Cách đơn giản trên Windows là mở `RUN_TOKENIZER_DEMO.cmd`.

## 2. Những nơi nên sửa trước

### Muốn đổi tokenizer

Sửa `configs/tokenizer/h2lm_tokenizer_dev.yaml`. Dữ liệu dùng manifest JSONL có checksum,
nguồn và split. Không sửa tokenizer đã gắn với checkpoint mà bỏ qua embedding tương ứng.

### Muốn đổi kích thước model

Sửa YAML trong `configs/model/`.
Không cần sửa Python nếu chỉ đổi số layer, hidden size, head, image size hoặc patch size.

### Muốn đổi kiến trúc

Sửa `src/h2lm/modeling/h2lm.py`.
Reference implementation cố tình nhỏ để dễ đọc. Encoder/fusion/decoder sẽ được tách khi cần.

### Muốn đổi chiến lược dữ liệu

Đọc `docs/02_DATA_AND_TEACHERS.md` và `docs/05_TOKENIZER.md`.
M1-A mới có corpus contract cho tokenizer, chưa có full PDF/teacher training pipeline.

## 3. Chạy thử model chưa train

Phần model cần cài PyTorch phù hợp môi trường, độc lập với tokenizer.
Trên CPU, có thể cài bản PyTorch đã dùng trong CI:

    python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
    python -m pip install -e ".[tokenizer,dev]"
    python -m pytest -q
    python scripts/smoke_train.py --device cpu

Chỉ dùng `--device cuda` khi đã kiểm tra build PyTorch/driver tương thích GPU thực tế.
Không coi một bản wheel CUDA mới nhất là đã được xác nhận chạy trên GTX 1070.
Script smoke dùng dữ liệu ngẫu nhiên, không tạo checkpoint H2LM thông minh.

## 4. Quy tắc để dễ sửa lâu dài

- Hyperparameter nằm trong YAML; đọc error thay vì bỏ qua config sai.
- Không hard-code API key hoặc commit hồ sơ riêng tư/weights lớn.
- Mọi format có version; benchmark có config/hash/source revision.
- Tạo branch riêng trước khi sửa; chạy tests nhỏ trước khi train tốn tiền.

Không có source generator kín hoặc binary bắt buộc để chỉnh mã H2LM.
Artifact tokenizer là đầu ra được tạo từ source/config/corpus có trong hướng dẫn.
