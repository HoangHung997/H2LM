# M1-EXP-01 — Huấn luyện neural thật, không chờ chủ dự án vận hành

## Quyết định theo yêu cầu mới

Người dùng yêu cầu tự tìm cách huấn luyện, không tiếp tục chờ họ gán nhãn/cài máy.
Thực hiện một thử nghiệm neural nhỏ bằng CPU đang có, đồng thời tạo workflow GitHub chạy
trên standard Ubuntu runner. Không thuê GPU, gọi teacher API, sửa quyền repo hay cài runner
vào PC của người dùng. Đây là nhánh nghiên cứu có giới hạn, **không bỏ qua gate của model
sản phẩm và không đánh dấu M1/M2 sản phẩm đã hoàn tất**.

Hai luồng dữ liệu tách biệt:
1. Hình tổng hợp có nhãn tạo cùng lúc bằng chương trình, không cần người chép lại.
2. Một số nhãn ảnh thật do AI đề xuất, chỉ dùng khi bật allow_silver trong thử nghiệm riêng.
   Không đổi draft_labels.json, không tự gắn human_verified, không đưa thành ground truth
   của bộ nghiệm thu. Các vùng uncertain/unreadable vẫn không dùng làm đáp án đoán.

## Thực sự đã huấn luyện cái gì?

Model thử nghiệm 710.616 tham số, bắt đầu từ random weights. Kế thừa lõi H2LM hiện có,
vision encoder nhỏ + resampler học được (hai dải ngang) + causal text decoder. Tất cả cùng
được tối ưu bằng gradient. Hai dải ngang là thiết kế thí nghiệm hẹp cho crop/ảnh hai dòng,
không phải kiến trúc sản phẩm đọc trang PDF phức tạp đã khóa.

Có auxiliary head đọc hai số trên ảnh tổng hợp để giúp encoder học thông tin nhìn thấy.
Head này **không được dùng khi sinh đáp án**; sinh đáp án vẫn qua H2LM decoder. Đây không
phải một OCR bên ngoài rồi chuyển text cho agent. Không nhập weights từ model khác.

Bộ mã hóa thí nghiệm là UTF-8 byte codec 266 IDs (10 reserved + 256 bytes), không phải
quyết định thay tokenizer sản phẩm. Tokenizer/corpus production M1 vẫn chưa khóa. Chỉ dùng
crop 128x128, không giả vờ đã train đầy đủ 10 trang scan độ phân giải cao.

Dữ liệu mặc định: 64 ảnh tổng hợp train/16 validation, mỗi ảnh hỏi hai câu Điều?/Khoản?,
tương ứng 128/32 mẫu QA. Split theo cặp giá trị/ảnh; giá trị và template vẫn có thể giống
nhau giữa split. Mẫu tổng hợp được tạo trong phần mềm, không phải scan từ máy.

Lượt local có bật silver: thêm số/ngày từ ĐT391 và Thạch Bích (4 crop train); hai crop
DA700 giữ làm **real_probe** theo tài liệu khác. Cả hai quyết định giao nhiệm vụ giữ chung
nhóm mẫu. Real probe đã được tác giả nhìn ở lượt trước, không phải benchmark độc lập/niêm phong.
Chỉ sáu vùng được dùng; các nhãn/bảng/QA khác chưa được train trong lần này.

## Những lỗi học đã phát hiện thay vì che bằng loss

Lượt đầu loss giảm nhưng che ảnh vẫn ra cùng đáp án: model học prior của câu hỏi.
Lượt thứ hai cân bằng loss theo từng đáp án vẫn chưa khắc phục. Lượt thứ ba thêm giám sát
thị giác cải thiện fit nhưng còn yếu. Lượt thứ tư thêm resampler hẹp và kiểm lại việc phụ
thuộc ảnh. Giữ báo cáo của các lượt kém trong evidence, không chỉ chọn số đẹp để báo.
Các lượt dùng cùng bộ development để chẩn đoán; không gọi kết quả validation này là final test.

Kết quả local trước push của lượt thứ tư (Python 3.13.5, torch 2.10.0+cpu):
- 1.000 bước, khoảng 50,44 giây phần train; 710.616 tham số.
- Token-averaged train NLL từ 5,7093 xuống 0,2854. Đây khác loss tổng (có auxiliary) trong log.
- Synthetic train: đúng 98/128; che ảnh: 10/128.
- Synthetic validation: đúng 15/32; che ảnh: 4/32.
- Bốn crop thật đã học: đúng 4/4 nhãn AI nháp; che ảnh đúng 2/4.
- Hai crop DA700 không train: **0/2**, chưa tổng quát hóa được sang tài liệu thật khác.

Các số trên là free-running generation (không truyền target vào predict), không phải
teacher-forced token accuracy. char_error_rate chỉ đo chuỗi đáp án ngắn, không phải CER
OCR toàn trang. Fit 4/4 không chứng minh hiểu pháp luật; không dùng checkpoint để kết luận
pháp lý, không hứa tương đương Gemma. production_ready=false.

## Chạy và tiếp tục

Workflow `.github/workflows/micro-neural.yml` tự chạy trên PR cùng repo khi code thử nghiệm
thay đổi; không cần nhờ người dùng bấm chạy. Nó chỉ sinh dữ liệu tổng hợp, không tải PDF cá
nhân. Cài torch CPU 2.10.0, chạy kiểm thử rồi train 1.000 bước và xuất checkpoint/report.
Job giới hạn 15 phút; code giới hạn 300 giây vòng train mặc định. Không lách giới hạn Actions
hoặc tạo chuỗi job vô hạn. Workflow phải được kiểm kết quả thật trên đúng SHA sau push.

Chạy thủ công (không phải yêu cầu chủ dự án phải làm để tôi tiếp tục):

```powershell
python -m h2lm.experiments.cli --output artifacts/my-micro-run
python -m h2lm.experiments.cli --output artifacts/my-segmented-run --segment-steps 100
python -m h2lm.experiments.cli --output artifacts/my-segmented-run --resume
```

Để dùng bundle scan thật đã tạo ở bước trước, truyền `--real-review` trỏ tới thư mục có
`draft_review.json` và `regions/`, thêm `--allow-silver`. Công cụ kiểm hash crop/nguồn khai
báo; không coi checksum là chứng minh nhãn đúng. Nguồn nhãn và phạm vi cho phép công khai
được giữ trong provenance; không tự mở rộng sang các tài liệu khác.

Mọi config chính nằm ở `configs/training/micro_vlm.yaml`. Nhánh thử nghiệm riêng không sửa
kích thước model TARGET và không tự đổi IDs của tokenizer đã gắn với checkpoint khác.

## Checkpoint và bằng chứng

Lưu state_dict của model, optimizer, số bước, RNG của torch và bộ lấy mẫu, hash dataset/
code, config và phiên bản torch. Ghi file tạm rồi thay con trỏ latest.json; kiểm checksum
trước torch.load(weights_only=True). Không lưu đối tượng Python model để unpickle tùy ý.
Chỉ dùng checkpoint có nguồn đáng tin; checksum không thay xác thực nguồn.

Có khóa một writer, từ chối ghi đè thư mục cũ, chặn config vượt giới hạn, NaN gradient/loss,
ảnh vượt kích thước, nhãn quá dài và trùng group/image giữa split. Resume từ checkpoint cùng
code/config/data; hard kill có thể để lại lock cần xử lý riêng, không hứa phục hồi mọi crash.
Timer là giới hạn vòng train, không phải quota OS; đánh giá/lưu file có thêm thời gian.

24 tests mới ở local: byte codec tiếng Việt, prompt/pad mask, nhân quả, độc lập khởi tạo
layer, kiểm hash/group/provenance, save/load, khóa writer và chạy liền so với chia hai đợt
cho trọng số giống từng bit trong môi trường thử. Đây không phải test trên GTX 1070.
Local là harness có các module sử dụng khớp blob trên base 6151fa0, không full git checkout;
CI full repository phải kiểm riêng. Kết quả remote cuối ghi ở PR conversation theo SHA.

Inference trên một ảnh crop, không cần target/tên tài liệu làm đầu vào:

```powershell
python scripts/predict_micro.py --run artifacts/my-micro-run/run --image crop.png --question "Số?"
```

## Điểm tiếp tục không cần chờ người dùng

Mở rộng dữ liệu có nhãn chương trình, thử thêm font/chất lượng và layout; tự đối chiếu các
vùng scan được phép; dùng silver để nghiên cứu nhưng đo riêng, không nâng thành gold. Ưu
tiên giảm lỗi trên real_probe và thay resampler hai dải bằng bộ nén phù hợp nhiều vùng;
không tăng lên hàng tỷ tham số trước khi đường học/thị giác/tổng quát hóa được chứng minh.
Lỗi SSL acquisition Công báo và corpus sản phẩm vẫn OPEN; không sửa TLS bằng verify=false.

Tham khảo triển khai:
- https://docs.pytorch.org/tutorials/beginner/saving_loading_models
- https://docs.pytorch.org/docs/2.14/generated/torch.nn.modules.transformer.TransformerEncoder.html
- https://docs.github.com/en/actions/reference/limits
