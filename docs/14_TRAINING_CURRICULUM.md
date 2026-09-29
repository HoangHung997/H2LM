# CQ-01 — Đào tạo H2LM 1B nhìn, tìm căn cứ và suy luận

## 1. Cam kết kỹ thuật và giới hạn của lời hứa

Giữ model khoảng 1B. Không thể gọi một phương pháp là “tốt nhất/mạnh nhất” trước khi so sánh
trên bộ đánh giá đủ đại diện. Mục tiêu là tiến bộ đo được trên scan thật tiếng Việt, không
phải tăng tham số, giảm loss trên vài mẫu hay viết lời giải dài cho có vẻ thông minh.

Nguồn tiếp tục: GitHub PR #9, commit 138ed6ea82299eeb77fbab67621fe822a38271f9, tree
810703d380c54e36ebdf38aba8c4b2e1630bf33e. **Kiến trúc hiện tại có 1.001.571.584 tham số**.
Nó KHÁC gói local trước đây có 1.000.032.768 tham số / 420 tensor. Không trộn checkpoint,
không chép gói đó đè code trên GitHub. CQ-01 giữ nguyên toàn bộ module `h2lm.scale` hiện tại.

Artifact của PR #9 chỉ có báo cáo/index/source, không có trọng số full model. Vì vậy lượt
CQ-01 tạo một run mới từ random weights của kiến trúc GitHub, không báo là resume từ những
weights chưa tải được. Checkpoint CQ-01 thật được lưu riêng sau khi chạy. Chưa nhập weights
của Gemma/Qwen/GPT, chưa sử dụng teacher API trả phí, GPU thuê hoặc PC của người dùng.

## 2. Phương án đào tạo cho sản phẩm — chưa phải tất cả đã thực hiện

### A. Nền ngôn ngữ và thị giác cùng phục vụ tài liệu

Model khởi tạo từ số 0 cần học tiếng Việt, cấu trúc câu, số, ngày tháng và quan hệ điều kiện;
chỉ học cặp ảnh -> vài chữ không tạo nền suy luận pháp lý. Thiết kế hai luồng pretraining:
ngôn ngữ tiếng Việt/kiến thức nền có kiểm soát nguồn và thị giác tài liệu. Luân phiên/ghép
hai luồng sau kiểm tra hội tụ, thay vì chỉ đào tạo một bên rồi hy vọng phần kia tự xuất hiện.
Không giữ language core ngẫu nhiên bị đóng băng rồi gọi đó là một model đã biết suy nghĩ.

Corpus sản phẩm cần lớn và đa dạng hơn những fixture hiện tại. Quyền sử dụng, thời điểm,
nhóm văn bản sửa đổi và phiên bản phải được lưu. Lỗi TLS tải Công báo vẫn còn mở, không
được bỏ `verify` để chạy cho xong. Nhãn là kiến thức trong tài liệu, không tự bảo đảm đang
có hiệu lực pháp lý. Các nguồn mới cần duyệt phạm vi/quyền sử dụng trước khi tải/train.

### B. Nhìn rõ trước: chữ, bố cục, bảng, dấu và số

Đầu vào chính là scan thật, kể cả PDF có lớp OCR cũ sai. Luôn giữ ảnh gốc/bbox/hash. Dùng
ảnh toàn trang + vùng ảnh độ phân giải cao, biết tọa độ và trang của từng vùng. Tăng từ
ký tự/cụm chữ -> dòng -> đoạn/bảng -> nhiều trang; không nén toàn bộ trang nhiều chữ thành
một ảnh 128px rồi mong phần ngôn ngữ đoán đúng. Render độ phân giải cao không phục hồi nét
đã mất. Binarize/denoise phải là biến thể có thể đối chiếu, không thay bằng chứng gốc.

Tự tạo tài liệu từ cấu trúc nguồn và render ra ảnh để có transcript, thứ tự đọc, hàng/cột,
box và đáp án. Thêm nhiều font, nét bút, scan lại, nhiễu nền, nghiêng, bóng và mộc. Các biến
đổi hình học phải biến đổi box tương ứng. Biến đổi xóa chữ phải đổi nhãn sang thiếu/không
đọc được, không giữ đáp án cũ để ép model đoán. Ảnh tổng hợp không thay thế đánh giá scan thật.

### C. Ghép nhìn với ngôn ngữ mà không bỏ qua ảnh

Đào tạo cùng mục tiêu: chép đúng nội dung nhìn thấy, chọn vùng bằng chứng, hiểu cấu trúc và
trả lời. Bằng chứng ảnh không đi qua OCR/agent ngoài bắt buộc tại inference. Có thể dùng
OCR/teacher như nguồn NHÁP lúc tạo data nếu được phép, nhưng không mặc định nhãn đó đúng.
Kiểm bottleneck resampler/image tokens riêng bằng ablation; tăng độ phân giải/token cần
đo lại RAM, tốc độ và khả năng giữ chữ nhỏ. CQ-01 chưa khóa lượng nén tối ưu cho mọi trang.

### D. Suy luận bám căn cứ, không thi viết “suy nghĩ” thật dài

Mẫu đào tạo cần: câu hỏi + trang/vùng -> kết luận + vị trí căn cứ + tóm tắt quy tắc áp dụng.
Đầu tiên tập sao chép và liên kết đúng, sau đó so sánh, cộng/đếm bảng, nối nội dung qua trang,
đối chiếu vai trò, sửa đổi, ngoại lệ và thời gian. Dùng quy tắc/tính toán kiểm chứng được
trước; nghiệp vụ luật thật phải dựa vào hồ sơ đủ căn cứ, không lấy mẫu giả lập làm luật.

Teacher tạo ứng viên answer/evidence/rule-summary; verifier kiểm số, ngày, box, trích dẫn,
phạm vi tài liệu và mâu thuẫn. Hai AI đồng ý không biến nhãn thành human_verified. Giữ các
mức programmatic / silver AI / người kiểm riêng. Cần ngân sách/quyền trước khi gọi API;
CQ-01 chỉ dùng giáo viên dạng chương trình và chưa gọi model bên ngoài.

Không huấn luyện theo tiêu chí “càng dài càng thông minh”. Với model nhỏ, tỉ lệ/phong cách
lời giải cần thử riêng; ưu tiên chứng cứ và giải thích ngắn kiểm chứng được. Không huấn luyện
model đoán phần bị che từ tên file hoặc văn bản tương tự trong tập khác.

### E. Tự sửa lỗi và học ưu tiên sau khi có năng lực nền

Sau SFT đủ chất lượng mới thử preference tuning/RL với verifier. Đáp án đúng nhưng sai
nguồn, lẫn ngày nội dung/ngày hiển thị ký số, hoặc bỏ qua ngoại lệ phải bị chấm không đạt.
Không thưởng độ dài, giọng tự tin hay JSON hợp lệ mà bỏ qua sự thật. RL không thay thế
pretraining của một model ngẫu nhiên. DPO/RL chưa được triển khai/chạy trong CQ-01.

### F. Chốt chất lượng bằng bộ test chưa dùng để chỉnh model

Chia nhóm document/family/phiên bản trước augmentation. Mọi scan/crop của cùng nguồn ở
cùng split. Giữ bộ scan thật độc lập, nhiều scanner/chất lượng/template và task. Đánh giá
đọc đúng dấu/chữ số, cấu trúc bảng, answer+evidence đồng thời, độ hợp lý khi từ chối và thời
gian xử lý. Đưa text chuẩn thay ảnh chỉ dùng chẩn đoán lỗi vision vs reasoning, không đổi
sản phẩm thành OCR ngoài rồi agent trả lời.

Ba PDF người dùng đưa đã được xem nhiều lần nên chỉ là development. Ví dụ ĐT391 và Thạch
Bích gán khác vai trò cho một đơn vị là mẫu tốt để đối chiếu nguồn, nhưng chưa phải test
niêm phong. Ngày trên ảnh nội dung và trên appearance chữ ký vẫn là hai trường khác nhau.
CQ-01 không âm thầm thêm nhãn vàng hay công bố đã hiểu các quyết định thật này.

## 3. Phần đã triển khai trong CQ-01

Factory có **96 family, 1.152 bài/biến thể**, mỗi bài hai ảnh 256x256. Đây là các thẻ văn bản
ngắn tiếng Việt, không phải trang A4 dày chữ. Sáu task: đọc mã, tìm trang, so sánh mã hai
trang, cộng bảng hai trang, phân biệt thi công/giám sát và từ chối khi mã đã bị xóa khỏi ảnh.
Các quy tắc là giả lập có solver xác định, không chứa chỉ dẫn rà phá vật nổ hoặc tư vấn luật.

Mỗi cặp giữ nguyên câu hỏi nhưng thay ảnh và thay đáp án theo oracle. Ví dụ đổi nhiệm vụ
thi công thành giám sát; hoán đổi mã giữa trang; thay một số lượng trong bảng; che/mở vùng mã.
Bộ kiểm tra chặn answer/evidence sai so với scene, pair sai, hash hỏng, path vượt thư mục,
family/full-document-pixel trùng giữa split. Hash/solver không chứng minh ảnh nào cũng dễ
đọc như scan thật hoặc ngăn mọi dữ liệu gần trùng ngoài factory có nguồn này.

Phân chia: 768 train / 192 validation / 192 holdout. Font thứ ba không dùng ở train/validation
và dành cho holdout. Không phân phối font. Holdout chỉ kiểm tính toàn vẹn, không chấm điểm
hay dùng để chỉnh lịch học trong runner. Validation cuối chỉ lấy sáu cặp định trước, không
phải chấm đủ 192 mẫu. Cần báo mẫu số thật, không dùng số lượng dataset thay số đã train/test.

Chuỗi mục tiêu tối giản `answer|pages` (ví dụ `16|1,2` hoặc `?|`) giúp phân biệt chấm đáp án
với chấm căn cứ, không ép viết lý luận dài khi nền ngôn ngữ chưa học. Proof có cấu trúc và
box trong metadata, **không đưa vào model lúc dự đoán**. Đây là chứng cứ theo trang/thẻ,
chưa là word-level grounding do model sinh. Byte codec 266 chỉ là prototype chưa khóa tokenizer.

Runner mặc định 48 bước: đầu tập đọc; giữa thêm định vị/vai trò; cuối giữ đọc và trộn các
task kết hợp. Warmup + cosine, loss answer/evidence nặng hơn dấu phân cách/EOS. Tỉ lệ và
hyperparameter là giả thuyết thử nghiệm, không phải kết luận tối ưu. Cấu hình tách YAML.

Toàn bộ 1B được cập nhật bằng SGD pilot. Local BF16 có thể làm tròn mất cập nhật, nên audit
đếm giá trị thực sự thay đổi; GitHub chạy FP32. **Không gọi SGD pilot là optimizer tốt nhất
cho pretrain dài hạn.** AdamW, tích lũy gradient, global clipping, phân tán GPU và corpus
streaming dài hạn cần triển khai/đo riêng; không đưa ra tuyên bố đã có ở đây.

Checkpoint tại cuối segment/run gồm weights, schedule/data/source contract và step; kiểm
hash trước resume. Có single-writer lock, chặn kế hoạch ngoài giới hạn, resume cùng plan,
không âm thầm đổi dataset. Streaming SGD có thể cập nhật một phần RAM trước lỗi: bỏ RAM,
chỉ resume checkpoint hoàn chỉnh. Không có phục hồi mọi crash hoặc checkpoint từng bước.

## 4. Cách tự mở/sửa/chạy

Các lệnh là tùy chọn của chủ dự án, không phải điều kiện để tác giả tiếp tục thử nghiệm:

```powershell
python -m h2lm.curriculum.train --output artifacts/cq1 --allow-large
python -m h2lm.curriculum.train --output artifacts/cq1-segment --allow-large --segment-steps 12
python -m h2lm.curriculum.train --output artifacts/cq1-segment --allow-large --resume
```

PC CPU RAM thấp: `--dtype bfloat16 --mapped --streaming`. Đó là đường kỹ thuật dùng file
backing, không bảo đảm phù hợp mọi máy/driver. Không tự cài runner/đổi driver của người dùng.
`--tiny` chỉ dùng kiểm thử logic, luôn báo số tham số thực và không được thay vào bước 1B CI.

Source: `src/h2lm/curriculum/data.py`, `train.py`; config `configs/training/curriculum_q1.yaml`.
Workflow `curriculum-1b.yml` giới hạn một job CPU, không tự nối vô hạn, không chi tiền GPU/API.
Remote artifact chỉ chứa report/index/source; **không chứa full weights** đã xóa cùng runner.
Full weights thử local được lưu và cung cấp riêng khi tạo gói; không được gọi index là weights.

## 5. Báo cáo phải phân biệt ba mốc

(1) Dataset/công cụ đã tạo; (2) số optimizer step, mẫu và token thật đã được học;
(3) năng lực đo bằng sinh đáp án tự do. Không coi (1)/(2) là đã đạt (3).
Luôn giữ lỗi đọc/nguồn/từ chối trong report. Có ảnh trắng và ảnh phản chứng cùng câu hỏi.
Bộ test kỹ thuật xanh chỉ chứng minh contract/đường chạy, không chứng nhận model thông minh.
Chưa có nghiệm thu GTX 1070, Ollama/GGUF, pretraining đầy đủ hoặc kết luận pháp lý đáng tin.

## 6. Nguồn nghiên cứu và cách áp dụng có giới hạn

- Pix2Struct (2210.03347): screenshot -> cấu trúc với input độ phân giải thay đổi. Gợi ý tạo
  nhãn từ cấu trúc rồi render; không sao chép điểm benchmark sang H2LM.
  https://arxiv.org/abs/2210.03347
- Donut (2111.15664): hiểu tài liệu không phụ thuộc OCR tại inference, dùng dữ liệu tổng hợp.
  Gợi ý một model ảnh-ngôn ngữ; H2LM vẫn dùng renderer kỹ thuật để mở PDF.
  https://arxiv.org/abs/2111.15664
- Qwen2.5-VL (2502.13923): độ phân giải động, định vị và tài liệu có cấu trúc. Tham khảo
  thiết kế dữ liệu/định vị, không nhập weights hoặc giả định ngang chất lượng.
  https://arxiv.org/abs/2502.13923
- SmolVLM (2504.05299), mục 3.4: CoT quá nhiều có thể hại model nhỏ trong thí nghiệm của họ.
  Không bê tỉ lệ % đó thành hằng số H2LM; thử lời giải ngắn/bằng chứng và đo riêng.
  https://arxiv.org/html/2504.05299v1
- DPO (2305.18290): học từ cặp ưu tiên là một lựa chọn post-training. Chưa chạy trong CQ-01.
  https://arxiv.org/abs/2305.18290

Không có nguồn nào chứng minh công thức CQ-01 là tốt nhất. Đó là giả thuyết được thiết kế
để kiểm tra và sửa dựa trên bằng chứng, phù hợp quyền tự chủ/tài nguyên hiện có.
