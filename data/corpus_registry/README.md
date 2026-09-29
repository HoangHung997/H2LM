# Registry nguồn corpus — M1-B1

Đây là mẫu khai báo, **không phải corpus pháp lý đã duyệt**. Các mục mặc định `approved: false`.
Sao chép mẫu ra thư mục local riêng ngoài Git, đặt các file TXT UTF-8 bên dưới thư mục đó,
kiểm tra nội dung/quyền sử dụng rồi mới điền thông tin duyệt. Không tự đổi trạng thái duyệt
chỉ để vượt qua lỗi. Các mẫu `train.txt`/`validation.txt`/`test.txt` không được cung cấp ở đây.

Một điểm bắt đầu tìm nguồn chính thức đã đối chiếu trong phiên triển khai:
- Trang Công báo cho bản công bố Nghị định 214/2025/NĐ-CP:
  https://congbao.chinhphu.vn/van-ban/nghi-dinh-so-214-2025-nd-cp-45774/58101.htm

Liên kết chỉ là ứng viên để thu thập ở M1-B2, chưa được tải vào corpus, chưa kiểm định bản trích xuất,
chưa xác nhận quyền phân phối lại, và không khẳng định văn bản hiện còn hiệu lực toàn bộ.
Không đưa bài báo tóm tắt, phần bình luận hay output AI vào cùng nhãn với nguyên văn chính thức.

Nhóm sửa đổi/thay thế liên quan phải được rà soát để dùng cùng `family_id` trước khi chia tập.
M1-B1 kiểm tra ID và nội dung trùng chính xác/NFC; chưa tự khám phá quan hệ văn bản hoặc chống
trùng gần đúng, không thể xác nhận chất lượng OCR bằng hash.

Xem `docs/07_CORPUS_AND_BENCHMARK.md` để chuẩn bị, so sánh, tiếp tục run và hiểu giới hạn.
