# M1-EXP-02 — Học đọc trường số/ngày trên ảnh chưa dùng để học

## Phạm vi

Tiếp tục theo yêu cầu chủ dự án: tự tìm cách huấn luyện, không chờ họ gán nhãn, cài runner
hoặc bật PC. Giữ vision-first và mục tiêu đọc PDF scan lại tiếng Việt. Task này là thử nghiệm
neural có giới hạn trên **một dòng/vùng số hoặc ngày**, không phải OCR toàn trang hay model
suy luận pháp lý hoàn chỉnh. Không thay kiến trúc sản phẩm, tokenizer chính hoặc checkpoint
EXP-01; các mã cũ vẫn giữ nguyên. Không thuê GPU, dùng API trả phí hoặc gửi thêm tài liệu ra ngoài.

## Thay đổi có thể tự sửa

`crop2_model.py`: CNN đọc crop chữ nhật giữ tỷ lệ -> chuỗi đặc trưng thị giác -> decoder
cross-attention sinh byte UTF-8. 524.852 tham số với config mặc định, khởi tạo ngẫu nhiên.
Không lấy trọng số ngoài. Đầu CTC dùng **loss phụ khi train**, không giải mã thành text rồi
đưa cho model khác; inference chỉ dùng ảnh, câu hỏi và decoder của cùng model.
Kiến trúc gộp chiều cao đang giả định một dòng; không dùng trực tiếp cho bảng/trang nhiều cột.

Ảnh đầu vào được contain vào canvas 48x320 có nền trắng, không kéo dãn thành hình vuông;
không sửa file ảnh gốc. Mọi phép resize vẫn có thể mất chi tiết. Đây không phải siêu phân giải,
không khôi phục nét chữ đã mất. Các crop thật vẫn giữ hash/provenance của bước scan trước.

`crop2_data.py`: tự tạo 1.200 ảnh train và 96 validation với số hiệu nhiều ký tự/ngày tháng.
Font serif/sans/italic lấy từ máy chạy, lưu tên + SHA256; không phân phối font. Có lệch nhẹ,
làm mờ/nén JPEG có giới hạn. Nhãn tạo cùng ảnh, không phải căn cứ pháp lý thật.
Các giá trị đáp án đầy đủ không trùng giữa train/validation, không dùng sáu giá trị số/ngày
đã biết của ba bản scan thật để tạo dữ liệu train. Font và template vẫn dùng chung giữa hai
tập; không gọi đây là kiểm thử font chưa thấy hoặc benchmark pháp lý độc lập.

`crop2_train.py`: train 1.800 bước CPU; checkpoint model/optimizer/RNG, hash cấu hình/dataset/
source và phiên bản môi trường. Kiểm loss/gradient hữu hạn, cổng độ lớn, khóa writer, không
âm thầm ghi đè checkpoint. Resume chỉ dùng cùng code/config/data; có test chạy liền và chia
đợt ra cùng trọng số trong môi trường thử. Hard kill có thể để lại lock, không hứa xử lý mọi
crash/NAS. Timer giới hạn vòng train; đọc dữ liệu/đánh giá/lưu checkpoint có thời gian riêng.

`crop2_probe.py`: chỉ đánh giá sáu crop số/ngày đã được người dùng cho phép. Chặn thay nguồn,
checksum sai và tự nâng nhãn AI thành nhãn người duyệt. Không update trọng số trong probe.
`scan_adapt.py`: thí nghiệm tùy chọn thêm 300 bước bằng 4 crop DT391/Thạch Bích dạng silver;
DA700 không vào loss. Phải bật allow_silver, giữ nguyên registry nháp. Không hỗ trợ resume
riêng cho stage này. Chỉ tiếp tục từ checkpoint tự train của EXP-02, không nhập model khác.

## Bằng chứng local trước khi đẩy GitHub

Linux, Python 3.13.5, torch 2.10.0+cpu. Local source harness giữ các dependency EXP-01 đã đối
chiếu Git blob với base 5bb0ea2; không giả vờ đã clone toàn repository (git DNS không truy cập).
27 tests mới đạt, compile đạt. Full repository và lint phải đọc CI đúng head sau push.
Không thay bằng kết quả CI cũ hoặc nói đã thử GTX 1070.

Lượt synthetic-only: 1.800 bước / khoảng 112,89 giây train; NLL trên 64 mẫu train kiểm tra
cố định từ 5,58759 xuống 0,001716. Vision weights thay đổi, checkpoint reload cho cùng dự đoán.
Đây không phải loss trung bình toàn corpus hoặc độ chính xác hiểu pháp luật.

| Đánh giá local | Kết quả |
|---|---:|
| 96 giá trị validation không có trong train | 88/96 khớp chuỗi hoàn toàn |
| Cùng câu hỏi, che trắng ảnh | 0/96 |
| Đổi ảnh sang mẫu khác cùng loại câu hỏi, còn trả đáp án cũ | 0/96 |
| Đổi ảnh và đối chiếu đúng đáp án của ảnh thay thế | 88/96 |
| Sáu crop scan thật không train trong lượt này | 0/6 theo nhãn nháp tham chiếu |

Tỷ lệ lỗi ký tự là trên **chuỗi đáp án trường ngắn**, không phải CER OCR cả trang.
Không so trực tiếp 88/96 với 16/32 ở EXP-01: nhiệm vụ, dữ liệu và cách chia đã khác.
Sinh đáp án tự do, không truyền target/tên file vào predict, không sửa số/ngày bằng regex.
Giá trị đúng vẫn phải có EOS và UTF-8 hợp lệ. Lỗi số không được bỏ khỏi báo cáo.

Thử adaptation riêng: 300 bước có 4 crop silver dùng để fit, rồi đánh giá lại. Kết quả thực chạy local: 4/4 crop đã học khớp, 0/2 DA700 chưa fit vẫn sai; synthetic
validation 89/96. Cải thiện 1 mẫu không
được xem là bằng chứng ưu thế thống kê. Tuyệt đối không gọi 4/4 fit là chính xác scan 100%.

## Điều đã học được và chưa đạt

Tự tạo dữ liệu rộng hơn giúp model học tổ hợp số thay vì chỉ thuộc vài đáp án. Các phép
che/đổi ảnh kiểm tra phụ thuộc thị giác; chúng không chứng minh hiểu câu hỏi ngoài hai dạng
đã học, không chứng minh suy luận pháp lý hay khả năng đọc tài liệu mới bất kỳ.

Trên scan thật, số/ngày viết tay, dòng địa danh nhiều từ và ảnh quét khác phân bố render
chưa tổng quát hóa tốt. Giữ các lỗi này công khai trong báo cáo; không dùng tên file, lớp
OCR ẩn hoặc đáp án gán trước để lấp lỗi. Cả sáu crop đã được quan sát từ những lượt trước,
vẫn là real development probe chứ không phải test niêm phong. Nhãn AI tham chiếu chưa duyệt
độc lập. Lỗi thu thập Công báo TLS vẫn OPEN; không tắt kiểm chứng TLS trong task này.

## Tự chạy / chỉnh sửa

Nhánh: `h2lm-crop-generalization-v2`. Các lệnh dưới đây là tùy chọn, không yêu cầu người dùng
thao tác để tôi chạy thí nghiệm. Có thể mở source bằng VS Code hoặc PyCharm.

```powershell
python -m h2lm.experiments.crop2_train --output artifacts/crop-v2
python -m h2lm.experiments.crop2_train --output artifacts/crop-segment --segment-steps 300
python -m h2lm.experiments.crop2_train --output artifacts/crop-segment --resume
python scripts/predict_crop_v2.py --run artifacts/crop-v2/run --image crop.png --question "Số?"
```

Cài torch CPU 2.10.0 và `pip install -e ".[scan,dev]"` như README. Config ở
`configs/training/crop_v2.yaml`. Không tăng steps/width/data vô hạn; code chặn ngoài giới hạn.
Checkpoint không tương thích với EXP-01/Ollama/GGUF; phải dùng đúng source kèm hash. Giữ
checkpoint ở nơi đáng tin; weights_only/checksum không thay chữ ký xác thực nguồn.

Workflow `crop-neural.yml` chạy một đợt giới hạn trên standard Ubuntu CPU cho PR cùng repo.
Chỉ tạo ảnh tổng hợp, không upload ba PDF người dùng vào job. Lưu actual checkpoint,
prediction reports và git archive HEAD. CI kỹ thuật thành công không tự phê duyệt chất lượng.

## Bước tiếp tục tự chủ

Tăng độ giống scan thật: bố cục dòng ngày đầy đủ, số viết tay/nét bút, font/chất lượng khác,
phân biệt trích nguyên văn với chuẩn hóa ngày. Thử dữ liệu tự sinh hoặc silver có nguồn nhưng
báo riêng, giữ DA700 là development probe đã xem. Tiếp tục nguyên tắc che/đổi ảnh và giữ lỗi.
Không vội tăng lên hàng tỷ tham số hay chốt vocabulary sản phẩm từ một demo trường ngắn.

## Tài liệu API dùng trong triển khai

- https://docs.pytorch.org/docs/2.10/generated/torch.nn.TransformerDecoderLayer.html
- https://docs.pytorch.org/docs/2.10/generated/torch.nn.CTCLoss.html
- https://docs.pytorch.org/docs/2.10/notes/randomness.html
- https://pillow.readthedocs.io/en/stable/reference/ImageDraw.html
