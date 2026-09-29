# H2LM V1 - Product Specification

## 1. Định nghĩa

H2LM V1 là mô hình multimodal Vision-Language được huấn luyện trọng số từ đầu, ưu tiên hiểu tài liệu tiếng Việt, đặc biệt là văn bản pháp lý.

H2LM không được định nghĩa là OCR + agent. OCR, layout understanding, document parsing, language understanding và reasoning là các năng lực của cùng hệ thống H2LM.

## 2. Use cases ưu tiên

1. PDF có text layer.
2. PDF scan không có text layer.
3. Hồ sơ pháp lý nhiều trang.
4. Luật, nghị định, thông tư, quyết định và phụ lục.
5. Văn bản sửa đổi, bổ sung, thay thế, bãi bỏ.
6. Hỏi đáp có căn cứ, trang và vùng nguồn.
7. So sánh hiệu lực và quan hệ giữa nhiều văn bản.

## 3. Năng lực bắt buộc

- OCR tiếng Việt có dấu.
- Đọc chữ nhỏ và trang độ phân giải cao.
- Layout: tiêu đề, đoạn, bảng, footnote, header/footer.
- Nhận diện Chương/Điều/Khoản/Điểm.
- Table structure recovery.
- Multi-page reasoning.
- Temporal legal reasoning.
- Citation grounding.
- Uncertainty / abstention khi thiếu căn cứ.

## 4. Nguyên tắc huấn luyện

- H2LM khởi tạo random weights.
- Teacher models có thể tạo hoặc kiểm tra dữ liệu.
- Teacher output không tự động trở thành ground truth.
- Mọi mẫu huấn luyện phải có provenance.
- Benchmark phải tách khỏi training data.

## 5. Máy đích tham chiếu

Inference quantized phải hướng tới NVIDIA GTX 1070 8 GB VRAM.

Training model sản phẩm không bị giới hạn bởi GPU này; có thể dùng GPU cloud/server trong giai đoạn huấn luyện.

## 6. Không thuộc V1

- Audio.
- Video dài.
- Mục tiêu trở thành chatbot tổng quát mạnh nhất mọi lĩnh vực.
- Tự động đưa ra tư vấn pháp lý không có căn cứ nguồn.

## 7. Tiêu chí sản phẩm

H2LM chỉ được coi là tiến bộ khi benchmark chứng minh cải thiện. Không dùng cảm giác chat hay demo đơn lẻ để tuyên bố model tốt hơn.
