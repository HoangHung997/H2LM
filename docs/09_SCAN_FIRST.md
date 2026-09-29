# M1-SCAN-A — PDF scan lại là đầu vào chính của H2LM

## 1. Yêu cầu của chủ dự án

Tài liệu chủ yếu tiếng Việt, thường in rồi đưa qua máy scan lại. H2LM phải học từ hình
trang thật, không chỉ lấy text layer của PDF điện tử. Giữ mục tiêu một model vision-language
đọc và suy luận; bộ render bên dưới chỉ chuẩn bị ảnh, không phải agent khác thay H2LM hiểu.

Task này làm **đường vào dữ liệu scan local**. Chưa huấn luyện encoder, chưa có OCR/model trả
lời, chưa tự deskew hoặc phát hiện hướng chữ. M1/corpus vẫn ACTIVE, M2 pretraining chưa bắt đầu.
Không dùng pipeline mới để tuyên bố vượt Gemma hoặc đã hiểu luật.

## 2. Chạy trên Windows

Cài sẵn Python 3.11 x64 có `py` launcher. Mở repo bằng VS Code/PyCharm tùy ý.
Mở `RUN_SCAN_PREPARE.cmd`, chọn PDF; hoặc kéo một file PDF vào script. Lần cài dependency
cần Internet để tải thư viện, nhưng bước xử lý không gửi tài liệu ra ngoài. Không cài torch,
CUDA, runner, teacher API hoặc sửa driver. CLI cũng chạy offline sau khi cài xong.

Mặc định chỉ chuẩn bị **5 trang đầu**. Để chỉ định đợt tiếp theo:

```powershell
.\.venv-scan\Scripts\python.exe -m h2lm.scans.cli --pdf "D:\Ho so\ban scan.pdf" --start-page 6 --page-count 5 --output "D:\H2Scan\dot-02"
```

Chỉnh tham số bằng `configs/scan/scan_v1.yaml`. Mặc định render ở scale tương ứng 300 dpi,
ảnh vùng 768 px, chồng 96 px, preview tối đa 1000 px. Đó là giá trị phát triển để đo, không
phải độ phân giải/chất lượng máy scan đã được nhận diện. Render ảnh nguồn mờ ở 300 dpi
không làm những chữ đã mất trở lại. Khi quá giới hạn, công cụ dừng rõ lý do, không âm thầm
hạ chất lượng/cắt mất trang để báo thành công.

`--rotate 90` nghĩa là xoay thêm 90 độ theo chiều kim đồng hồ **tất cả các trang được chọn**.
Rotation có sẵn trong PDF được giữ. Với một trang lệch hướng, chạy riêng trang đó ra output mới.
Chưa có sửa hướng tự động; chưa tự thay đổi nhãn bbox hay xoay theo OCR. Không ghi đè output cũ.

Để chạy demo mà không gửi tài liệu cá nhân:

```powershell
.\.venv-scan\Scripts\python.exe -m h2lm.scans.cli --demo --page-count 8 --output artifacts\scan-demo-moi
```

Demo tạo PDF ảnh-only từ chữ giả lập tiếng Việt, sau đó mô phỏng giảm phân giải/mờ, nghiêng,
nén JPEG/nền tối, xoay ngang, ngược đầu, chữ rất nhạt và trang trắng. Không phải scan thật
qua máy, không phải điều khoản pháp luật thật, không phải nhãn OCR đã được người kiểm chứng.
Font lấy từ máy đang chạy; không có font file phân phối kèm repository/gói tải.

## 3. Kết quả và kiểm tra

`manifest.json` chỉ xuất khi worker thành công, nguồn không thay đổi và thông tin chuẩn bị
đã được kiểm tra. Nếu có `FAILED.json` hoặc thiếu manifest thì không dùng output như dataset
hoàn chỉnh. Thư mục lỗi được giữ để chẩn đoán, không ghi đè hoặc tự xóa để che lỗi.

| File | Nội dung |
|---|---|
| `source.pdf` | Bản sao bytes gốc, giữ nguyên toàn văn và nguồn; không sửa file đầu vào |
| `request.json` | Cấu hình, số trang, hash, document/family/split đã khai báo |
| `pages/00001/page.png` | Ảnh trang ở độ phân giải render; không tự denoise/binarize/crop |
| `overview.png`, `tile-*.png` | Preview giảm kích thước và các vùng ảnh **không resize** |
| `manifest.json` | Toàn bộ số trang gốc, phạm vi thực xử lý, hash ảnh, bbox và trạng thái nhãn |
| `review.html` | Mở bằng trình duyệt local để xem trang và nhấp vào ảnh độ phân giải đầy đủ |
| `worker.log` | Nhật ký local, có thể chứa đường dẫn/tên file; không gửi public tự động |

Số trang là thứ tự vật lý trong PDF, bắt đầu từ 1; không phải số in ở footer. Báo cáo phân
biệt xử lý một khoảng trang với hoàn tất cả tài liệu. Trang trắng/mờ vẫn được giữ; các chỉ số
độ sáng/độ tương phản là cảnh báo xem lại, không phải điểm tin cậy khả năng đọc chữ.
Dấu mộc, chữ ký, chú thích/mép trang không bị pipeline tự loại bỏ. Forms tương tác chưa hỗ
trợ sẽ báo lỗi yêu cầu flatten có kiểm chứng, không bỏ qua nội dung form rồi báo xong.

## 4. Tuyệt đối không mặc định chữ OCR cũ là đúng

Mọi trang đều có ảnh nguồn và `vision_required: true`, kể cả trang có lớp chữ.
Chỉ đếm ký tự có sẵn để báo `none` hoặc `present_untrusted`; không trích lớp chữ đó vào
input/nhãn. `none` cũng có thể là trang vẽ bằng vector, không khẳng định chắc chắn đã scan.

Các trường hiện tại: `expected_text: null`, `label_status: unlabeled`,
`eligible_for_supervised_training: false`, `readability: not_assessed`.
Chưa có thuật toán so sánh hình với OCR ẩn, nhận ra số bị OCR sai hoặc xuất đáp án pháp lý.
Test có PDF chứa ảnh và lớp chữ ẩn **cố tình sai**, kiểm rằng text này không lọt vào nhãn.
Đây là guard về đường đi dữ liệu, không phải mô hình đã phát hiện đúng/sai về nội dung.

## 5. Tọa độ và tính toàn vẹn

Vùng crop dùng `[x0,y0,x1,y1)` theo biên pixel trên ảnh render, gốc trên-trái. Mỗi tile giữ
quad bốn góc trong hệ tọa độ PDF nguồn nhờ `FPDF_DeviceToPage`, có tính đến CropBox và rotation
của trang/lệnh. Không suy tọa độ PDF bằng tỷ lệ từ preview đã thu nhỏ. Test đưa quad trở lại
pixel bằng `FPDF_PageToDevice` với bốn hướng xoay, CropBox khác MediaBox và trang đã xoay sẵn.
Các quad hiện là vị trí **vùng ảnh**, chưa phải vị trí từ/điều khoản do model nhận dạng.

PDFium không cung cấp hệ số UserUnit trong helper render hiện tại: số dpi trong báo cáo là
requested scale với giả định 1/72 inch cho một canvas unit, không xác minh kích thước bản giấy.
Mỗi PNG và nguồn có SHA256 để đối chiếu. SHA256 không chứng minh nội dung đúng hay quyền sử dụng.

## 6. Giới hạn và an toàn

Render từng trang tuần tự trong subprocess có timeout; không gọi PDFium đồng thời từ nhiều
thread. Mặc định file <=128 MiB, tối đa 20 trang/đợt, 16 triệu pixel/trang, 180 triệu pixel/đợt,
64 tile/trang và ngân sách output 512 MiB. Cấu hình có chặn giá trị vô hạn/âm/quá lớn.
Ngân sách là giới hạn logic; không phải sandbox hệ điều hành hoặc bảo đảm chặn mọi PDF độc hại.
Parser native vẫn có thể dùng nhiều RAM để giải nén object; chỉ mở file không tin cậy trong
máy/tài khoản tách biệt khi triển khai thật. Không hỗ trợ password hay bypass mã hóa.

Dữ liệu xử lý thật và mọi bản sao/preview/tiles có thể chứa thông tin nhạy cảm. Giữ ngoài Git
hoặc trong `artifacts/` đã ignore; không commit ra repo public. CI chỉ tạo fixture giả lập,
không tải hồ sơ người dùng hoặc cài self-hosted runner. Output local không tự đồng bộ với GitHub.

## 7. Gán nhãn và nghiệm thu tiếp theo — chưa triển khai trong task này

Cần PDF scan thật được người dùng cho phép, gồm bản scan lần một/lần hai, nhiều scanner,
chất lượng thấp, bảng, dấu mộc, trang xoay và văn bản nhiều trang. Giữ bản gốc cùng hash;
người/teacher hỗ trợ gán transcript, bố cục, bbox, answer và evidence sau khi đối chiếu vùng ảnh.
Vùng không nhìn rõ phải gắn `unreadable`/`uncertain`/`missing`, không ép người gán nhãn đoán số,
ngày, dấu hoặc điều khoản. Output teacher vẫn cần kiểm chứng trước khi cho vào dữ liệu học.

`family_id`/`document_id`/`split` theo request áp dụng toàn bộ trang/tiles; chưa auto dò hai PDF
khác nhau là hai lần scan của cùng bản. Thiếu family thì gắn `family_review_required: true`.
Có khai báo family không có nghĩa đã kiểm định; output vẫn chưa train-ready. Các bản quét,
biến thể mờ và crop cùng nguồn bắt buộc nằm cùng split. Không tạo biến thể của bản test để train.

Đo riêng lỗi ký tự tiếng Việt (CER), dấu/chữ số/số hiệu, vùng bảng, grounding, trang không đọc
được, và đáp án có căn cứ trên bộ scan chưa xuất hiện khi học. Chỉ số trên PDF có text layer
không thay chỉ số scan. Phần kiểm định này chưa có bộ dữ liệu đủ đại diện và không được tuyên bố
đã đạt từ 8 trang mô phỏng hoặc từ 54 unit tests. Chưa có nghiệm thu GTX 1070 hay neural checkpoint.

## 8. Trạng thái nguồn / lỗi đang mở

PR #4 tại c0049c7 có CI mã xanh nhưng acquisition run 36562685281 lỗi xác minh SSL issuer.
Không tắt TLS verification để lấy dữ liệu. M1-SCAN-A là nhánh tiếp tục chuẩn bị dữ liệu offline
được ưu tiên theo yêu cầu mới; không tuyên bố đã sửa lỗi thu thập Công báo hoặc hoàn tất M1-B2.
Bản ZIP M1-B2 cũ và PR #4 khác nhau: dùng code GitHub đang có làm base, không chép ngược ZIP.

## Tham khảo triển khai

- PDFium Python API, render/rotation/threading: https://pypdfium2-team.github.io/pypdfium2/python_api.html
- pypdf text extraction (text layer không phải OCR): https://pypdf.readthedocs.io/en/6.18.1/user/extract-text.html
- Pillow Image API: https://pillow.readthedocs.io/en/stable/reference/Image.html
