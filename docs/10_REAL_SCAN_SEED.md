# M1-SCAN-B — Ba bản scan thật do chủ dự án cung cấp

## Trạng thái

Đã nhận và chạy chuẩn bị ảnh trên ba PDF / 10 trang thật, không phải ảnh tổng hợp.
Người dùng nói: “Đây dùng tạm 3 bản này về cơ bản k bảo mật nhạy cảm gì có thể công khai được”.
Quyền công khai này chỉ được ghi nhận cho ba file có SHA256 trong `draft_labels.json`,
không suy thành giấy phép chung, xác thực chữ ký hoặc duyệt nội dung nhãn.

File nhãn: `data/scan_seed_20260929/draft_labels.json`. Có 28 vùng nhãn nháp và 12 câu hỏi
bám ảnh, bao gồm câu hỏi nhiều trang/nhiều tài liệu và câu không đủ căn cứ. Đây là phần
trích vùng/field, không phải bản chép toàn bộ 10 trang. Nguồn nhãn là assistant nhìn ảnh;
chưa có người độc lập kiểm tra. `training_eligible` luôn false. Không tính CER từ việc tự
so đáp án nháp với chính đáp án đó; chưa chạy neural training/inference.

## Những trường hợp thật đã phát hiện

| File | Trang | Trường hợp quan trọng |
|---|---:|---|
| DA700, số nhìn thấy 1542/QĐ-TCT | 2 | PDF có rotation 270; số/ngày viết tay; bảng có TT 1–4 ở trang 1, TT 5–6 ở trang 2; vệt ngang scan và dấu mộc |
| ĐT391 Hải Dương, số nhìn thấy 3464/QĐ-BQP | 4 | Chữ in nghiêng nhỏ; số/ngày viết tay; khoản nối nhiều trang; ghi chú viết tay lề để uncertain, không đoán |
| Thạch Bích, số nhìn thấy 3154/QĐ-BQP | 4 | Scan kèm AcroForm trường chữ ký số; phải render phần hiển thị chữ ký, không chỉ ảnh nền |

Không lấy số/ngày từ tên file làm nhãn. Đã nhìn số và dòng ngày tháng trong ảnh. Bản xem trước
của một hệ thống có thể thiếu vùng đầu trang; không kết luận PDF gốc bị cắt chỉ từ preview.
Dữ liệu DA700 thật có đầy đủ tiêu đề và hai hàng tiếp ở trang 2; không dạy model chỉ đếm 4 người.

Với Thạch Bích, dòng ngày tháng nội dung là 16/06/2026, phần hiển thị chữ ký số ghi
18-06-2026 09:27:58 +07:00. Hai trường khác nhau được gán nhãn riêng; không suy ra hiệu lực
pháp lý hay tính hợp lệ mật mã của chữ ký. Đã kiểm tra phần hiển thị bằng PDFium và MuPDF.
MuPDF chỉ dùng kiểm tra độc lập local; không thêm làm dependency của ứng dụng.

Ví dụ QA đối chiếu: Tổng công ty Thành An là đơn vị thi công trong ĐT391, nhưng là đơn vị
giám sát trong Thạch Bích. Chỉ mô tả những gì hai tài liệu ghi, không kết luận quy định hiện hành.
Hai quyết định giao nhiệm vụ có cùng nhóm mẫu để chống việc tách các đoạn giống nhau sang
train/test. Cả ba file vẫn là `development_seed`, chưa phải holdout độc lập.

## Lỗi đã sửa khi chạy file thật

Bản cũ từ chối mọi PDF có AcroForm. File Thạch Bích chứa trường `/Sig`, khiến worker báo
“Interactive forms need separately reviewed flattening”. Không sửa/flatten/xóa chữ ký
để lách lỗi. Thay vào đó khởi tạo form environment trước khi lấy page/length và render
appearance cùng annotations, giữ nguyên bytes PDF nguồn.

`form_policy.py` chỉ chấp nhận NONE/ACRO_FORM; XFA/loại không rõ vẫn bị chặn. Cấu hình không
đăng ký nền tảng JavaScript/callback tương tác và không gọi document actions. Vẫn là native
parser, không phải sandbox hệ điều hành. `digital_signature_validation: not_performed`.
Test mới tạo widget appearance nhìn thấy được: kiểm pixel có hình, khác khi bỏ forms,
source không đổi; đồng thời kiểm các đường chặn và lỗi init. Không dùng việc test giả có
hình chữ nhật để khẳng định mọi loại chữ ký PDF đều đúng.

## Chạy lại bằng ba file gốc

Lấy nhánh `h2lm-real-scan-seed` khi PR chưa merge. Giữ ba tên file như trong registry ở một
thư mục riêng; chạy với Python 3.11 x64:

```powershell
python -m venv .venv-scan
.\.venv-scan\Scripts\python.exe -m pip install -e ".[scan]"
.\.venv-scan\Scripts\python.exe -m h2lm.scans.review_seed --seed data/scan_seed_20260929/draft_labels.json --input "D:\H2ScanInputs" --output "D:\H2ScanReview\run-01"
```

Hoặc kéo thư mục chứa ba file vào `RUN_REAL_SCAN_REVIEW.cmd`. Input được kiểm hash trước
khi tạo output; output đã có sẽ bị từ chối, không ghi đè. Công cụ chạy local sau khi cài
thư viện. Không gọi API teacher, OCR ngoài, GPU cloud hay upload tài liệu.

Output gồm các bản sao PDF, 10 ảnh trang, ảnh xem trước, 224 tile, 28 ảnh crop vùng nhãn,
`draft_review.json` gắn hash ảnh trang/crop và `review.html` để đối chiếu bằng mắt. Tọa độ
nhãn là vùng trên toàn trang đã xoay đúng, chuẩn hóa 0–1000. Đó không phải bbox nhận dạng chữ.
Cùng nguồn phải giữ document/family/leakage group khi phát sinh crop hoặc scan lần khác.

Những gì được đưa lên Git trong task: code, nhãn nháp, checksum và ghi nhận quyền sử dụng.
PDF gốc và ảnh lớn nằm trong gói dữ liệu của cuộc trò chuyện, không nhúng base64 vào code
hoặc tự đăng lên website khác. Mọi mẫu khác vẫn cần quyền sử dụng riêng.

## Kiểm thử và điểm tiếp tục

Local là scan harness dựng từ patch M1-SCAN-A đã kiểm từng blob với HEAD 902d16c, không phải
full repository checkout (container không phân giải DNS để git clone). Local: 54 regression
scan + 8 tests form + 27 tests nhãn, chạy theo ba nhóm do giới hạn thời gian công cụ; compile
đạt. Lượt gộp đầu bị timeout nên không dùng làm bằng chứng đạt. CI mới chạy full repo trên
đúng PR head; trạng thái CI cuối được ghi trong PR conversation, không lấy CI cũ thay mới.

Cần tiếp tục: người độc lập kiểm nhãn/độ đầy đủ; mở rộng scan thật đa dạng; kiểm tách family
và gần trùng; bổ sung transcript vùng khó/bảng đầy đủ; xây bộ test độc lập rồi mới huấn luyện
vision ở milestone M2. Các draft hiện không được tự đánh dấu verified. Chưa đánh giá độ
chính xác H2LM hay hiệu năng GTX 1070. Lỗi TLS của luồng Công báo từ M1-B2 chưa sửa trong task này.

Tham khảo API render forms: https://pypdfium2-team.github.io/pypdfium2/python_api.html
