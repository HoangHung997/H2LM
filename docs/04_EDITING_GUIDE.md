# Cách tự mở và sửa H2LM

Mục tiêu của repository là không khóa người dùng vào một công cụ riêng.

## 1. Mở project

Có thể clone repository và mở trực tiếp bằng VS Code hoặc PyCharm.

Windows PowerShell:

    git clone https://github.com/HoangHung997/H2LM.git
    cd H2LM
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -e ".[dev]"

## 2. Những nơi nên sửa trước

### Muốn đổi kích thước model

Sửa YAML trong:

    configs/model/

Không cần sửa Python nếu chỉ đổi số layer, hidden size, head, image size hoặc patch size.

### Muốn đổi kiến trúc

Sửa:

    src/h2lm/modeling/h2lm.py

Reference implementation cố tình giữ trong một file nhỏ ở bootstrap để dễ đọc. Khi kiến trúc ổn định sẽ tách encoder/fusion/decoder thành module riêng.

### Muốn đổi chiến lược dữ liệu

Đọc:

    docs/02_DATA_AND_TEACHERS.md

Pipeline dataset sẽ được bổ sung ở milestone tiếp theo.

## 3. Chạy thử model chưa train

    python scripts/smoke_train.py --device cuda

Script tạo input ngẫu nhiên, forward, backward và optimizer step. Đây không phải huấn luyện thật; mục tiêu là xác nhận code model hoạt động trên máy.

## 4. Quy tắc để dễ sửa lâu dài

- Hyperparameter nằm trong YAML.
- Không hard-code API key.
- Không commit weights/dataset lớn.
- Mọi format dữ liệu phải có schema.
- Mọi benchmark phải lưu config và commit SHA.
- Kiến trúc mới phải có test nhỏ trước khi train tốn tiền.

## 5. Khi tự sửa lỗi

Tạo branch riêng rồi commit. Nếu sửa YAML mà model không load được, test config sẽ phát hiện một số lỗi cấu trúc cơ bản.

Không có source generator kín hoặc binary bắt buộc để chỉnh H2LM; mã model và pipeline phải đọc được trong repository.
