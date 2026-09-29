# H2LM

**H2LM V1** là dự án xây dựng một mô hình AI multimodal **Vision-First / Document-First** do chính dự án huấn luyện trọng số từ đầu, tập trung trước hết vào **tài liệu tiếng Việt**, đặc biệt là **văn bản pháp lý**: luật, nghị định, thông tư, quyết định, phụ lục, văn bản sửa đổi/bổ sung/thay thế và hồ sơ nhiều trang.

## Mục tiêu V1

H2LM không được thiết kế như một OCR rồi chuyển kết quả sang một AI khác. Mục tiêu là một mô hình duy nhất có khả năng:

- nhìn trang PDF/ảnh scan ở độ phân giải cao;
- đọc tiếng Việt có dấu, chữ nhỏ, bảng, biểu mẫu và bố cục phức tạp;
- hiểu cấu trúc Chương → Điều → Khoản → Điểm;
- nhận biết dẫn chiếu, sửa đổi, bổ sung, bãi bỏ, thay thế, hiệu lực và điều khoản chuyển tiếp;
- suy luận trên nhiều trang/nhiều văn bản;
- trả lời có căn cứ và vị trí nguồn, đồng thời biết từ chối kết luận khi thiếu dữ kiện;
- chạy local ở bản quantized trên máy đích tham chiếu **NVIDIA GTX 1070 8 GB VRAM**.

## Nguyên tắc sở hữu model

H2LM V1 chọn phương án **B**:

1. Trọng số H2LM được khởi tạo ngẫu nhiên và huấn luyện bởi pipeline của dự án.
2. Có thể dùng các model mạnh làm **teacher** để tạo, gợi ý hoặc kiểm tra dữ liệu huấn luyện.
3. Teacher model không được nhúng trọng số vào H2LM.
4. Dữ liệu do teacher sinh phải có provenance, bộ lọc và bước kiểm chứng; không coi output của teacher là ground truth mặc định.

## Thiết kế để người dùng tự sửa được

Repository ưu tiên **Python + PyTorch + YAML/JSON + Markdown**. Những thứ thường xuyên cần chỉnh (kích thước model, tokenizer, độ phân giải, batch size, dataset mix, loss weight, teacher provider, benchmark threshold...) phải đặt trong file cấu hình, không hard-code rải rác.

Mục tiêu là có thể mở repository bằng VS Code/PyCharm, đọc tài liệu và sửa từng phần mà không cần công cụ nội bộ bí mật.

## Trạng thái

Repository đang ở giai đoạn **V1 bootstrap / architecture lock**. Chưa có checkpoint H2LM V1 hoàn chỉnh. Mọi kết quả benchmark phải gắn với commit SHA + config + dataset manifest để có thể tái lập.

## Quy ước chính

- `docs/` — đặc tả và quyết định kiến trúc.
- `configs/` — cấu hình model/train/data; ưu tiên chỉnh ở đây.
- `src/h2lm/` — mã nguồn model và pipeline.
- `tests/` — smoke/unit tests.
- `data/` — chỉ chứa manifest/mẫu nhỏ; không commit dataset lớn hay tài liệu có hạn chế bản quyền.
- `artifacts/` và checkpoints — không commit trực tiếp vào Git.

Chi tiết triển khai sẽ được bổ sung theo từng milestone H2LM V1.
