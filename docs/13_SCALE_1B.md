# SC-1B — H2LM đạt quy mô tối thiểu một tỷ tham số

## Yêu cầu và phạm vi chính xác

Người dùng yêu cầu “Tiếp tục làm cho đến đủ 1 tỉ tham số đi”. Mốc này chuyển từ các thử
nghiệm crop khoảng 0,5M sang một kiến trúc multimodal **1.001.571.584 tham số độc lập**.
Không tăng số bằng tensor thừa, nhân bản chung trọng số, đếm cả optimizer hay cộng các
checkpoint. Số tham số do kiến trúc quyết định; thêm bước huấn luyện không tự tăng số đó.

**Đạt 1B không có nghĩa model đã thông minh hoặc pretrain xong.** Gate quy mô và gate chất
lượng được báo riêng. Lượt ở đây là tám bước kiểm tra cập nhật toàn bộ kiến trúc, không
phải huấn luyện đủ corpus tiếng Việt/pháp lý. Không dùng nó để trả lời pháp lý hoặc công
bố chất lượng ngang Gemma. Các EXP-01/02, dữ liệu và nhãn scan gốc không bị xóa/sửa đè.

## Kiến trúc có thể sửa

| Thành phần | Tham số |
|---|---:|
| Vision: patch embedding + 12 block 2D vision, width 768 | 85.543.680 |
| Fusion: learned queries/cross attention + geometry + projection | 8.485.632 |
| Language: 26 block, width 1792, SwiGLU 5120, GQA 14/2 heads | 907.542.272 |
| Tổng duy nhất, tất cả requires_grad | **1.001.571.584** |

361 tensor tham số, không alias trọng số. `named_parameters()` đếm một lần cho mỗi tham
số; optimizer/RNG/buffer/ảnh không tính vào quy mô. Kiểm thử có đếm cấu trúc trên meta;
lượt train phải cấp phát dữ liệu thật và chứng minh cả 361 tensor đều có gradient.
Không coi count trên meta là model đã được huấn luyện.

Vision nhận nhiều tile RGB, vị trí 2D trong tile, bbox/page nguồn. Learned resampler tạo
32 visual tokens mỗi tile, rồi ghép làm prefix. Decoder dùng RMSNorm/SwiGLU/GQA/RoPE:
image prefix không nhìn text đích, text nhìn ảnh và các token trước; mask text/tile padding
được kiểm thử. Không đưa lớp OCR ẩn hoặc đáp án vào ảnh. `max_pages` là mẫu số quy ước khi
chuẩn hóa tọa độ trang; caller phải bảo đảm cùng hệ quy chiếu cho toàn hồ sơ.

Kiến trúc nhận tile đã chuẩn bị, chưa nối thành sản phẩm tự đọc hồ sơ hàng nghìn trang.
Đầu vào lớn hơn smoke (nhiều tile, chữ rất nhỏ, context dài) chưa được đo GPU/VRAM. Mask có
chi phí bậc hai theo context; chưa có KV cache, dynamic batching hay export GGUF/Ollama.
Các mục tiêu này không được báo đã hoàn thành từ việc khai báo giới hạn config.

Bộ từ vựng tạm là **266 IDs: 10 reserved + 256 byte UTF-8**, không thêm hàng chục triệu
embedding chưa có tokenizer để đủ số. Nó mã hóa được tiếng Việt có dấu, nhưng có thể tốn
nhiều token; chưa phải quyết định tối ưu tokenizer sản phẩm. Đổi vocab/architecture sau
này là một cấu hình/checkpoint mới, không tự sửa checkpoint đang dùng.

## Chạy, bộ nhớ và checkpoint

Cấu hình: `configs/model/h2lm_v1_1b.yaml`; code: `src/h2lm/scale/`.
Đếm cấu trúc không cấp phát 1B weights:

```powershell
python -m h2lm.scale.runner count
```

Tự chạy khi đã có môi trường phù hợp (không phải yêu cầu chủ dự án phải thao tác để tiếp tục):

```powershell
python -m h2lm.scale.runner train --output artifacts/scale-1b --steps 8 --allow-large
python -m h2lm.scale.runner verify --output artifacts/scale-1b --allow-large
```

Chế độ CPU ít RAM, ví dụ đã thực thi local:

```powershell
python -m h2lm.scale.runner train --output artifacts/scale-bf16 --steps 8 --dtype bfloat16 --mapped --streaming --allow-large
python -m h2lm.scale.runner verify --output artifacts/scale-bf16 --mapped --allow-large
python -m h2lm.scale.runner train --output artifacts/scale-bf16 --steps 8 --dtype bfloat16 --mapped --streaming --allow-large --resume
```

Mặc định train bị chặn nếu chưa có `--allow-large`. Count/tiny tests không có yêu cầu này.
BF16 cần CPU/runtime hỗ trợ; **không chỉ dẫn chạy BF16 trên GTX 1070**. Đích GPU 8 GB chưa
nghiệm thu; trọng số đủ nhỏ không có nghĩa tổng bộ nhớ infer/train cũng vừa.
Weights thô FP32 là 4.006.286.336 bytes; BF16 là 2.003.143.168 bytes. Activations, gradients,
workspace, optimizer và file metadata cần thêm bộ nhớ. Standard SGD giữ gradient của toàn
model; Adam còn có trạng thái lớn hơn. Pilot này dùng SGD không momentum/weight decay,
không giả vờ đây là cấu hình pretraining tối ưu.

Mapped mode dùng file thật với các vùng storage **không chồng lấn**, mỗi parameter có bộ
đếm version tensor riêng. Lỗi ban đầu do các view chung version counter đã được phát hiện
ở test nhỏ, sửa và thêm regression trước khi chạy model 1B. Đây không phải meta/giả trọng
số: mỗi phần được khởi tạo ngẫu nhiên, đọc khi forward và cập nhật khi backward. OS có thể
đưa các trang file ra/vào RAM, đánh đổi tốc độ; không hứa tính toán 1B miễn phí hay nhanh.

Streaming SGD cập nhật khi gradient của một parameter hoàn tất để giảm peak RAM. Có
kiểm thử so với torch.optim.SGD chuẩn trên cấu hình nhỏ, cùng trọng số sau hai bước.
Không hỗ trợ gradient accumulation, shared-weight aliases, distributed training,
activation recomputation hoặc global gradient clipping trong mode này. Nếu một hook lỗi,
trạng thái RAM có thể mới cập nhật một phần: bỏ trạng thái đó, chỉ resume checkpoint đã
hoàn tất. Không ghi “step hoàn thành” hoặc checkpoint của bước lỗi. Hard-kill có thể để
stale lock; không tự phá lock hoặc tuyên bố mọi crash đều tự phục hồi.

Checkpoint chia shard cỡ 64 MiB, lưu tensor riêng không mang cả arena vào mỗi shard,
kiểm hash/shape/dtype/số lượng và chỉ ghi index sau khi tất cả shard hoàn tất. Load dùng
`weights_only=True`, giới hạn file và kiểm nguồn; hash không thay chữ ký xác thực nguồn.
SGD không momentum có zero optimizer-state tensors; vẫn lưu tên optimizer/lr, số bước và
RNG. Resume yêu cầu cùng code/config/dtype/torch. Giới hạn 64 bước tích lũy dành cho pilot,
không là service tự nối job vô hạn. Một production trainer dài hạn là công việc tiếp theo.

## Đã kiểm chứng local trước khi đưa lên GitHub

Linux/Python 3.13.5/torch 2.10.0+cpu; không có GPU. Bộ nhớ cgroup của môi trường là 4 GiB,
có service dùng chung nên dùng BF16 mapped + streaming. Bản nguồn mới được phát triển
trong harness; package lõi lấy từ gói code trước, không giả có full git clone khi DNS lỗi.

- Cấu hình đúng 1.001.571.584 tham số thật đã khởi tạo.
- Tám bước SGD hoàn tất, mỗi bước cả 361 tensor có gradient khác 0, tổng số phần tử trong
  các tensor gradient là 1.001.571.584. Do BF16 làm tròn, **không phải cả một tỷ giá trị đều
  đổi sau mỗi bước**; audit ghi riêng tensor/element thực sự cập nhật.
- Lượt local ban đầu NLL trên fixture cố định 5,580708 -> 3,639687; tham khảo run log theo
  source fingerprint. Đây là loss của ví dụ tổng hợp nhỏ, không phải corpus/perplexity chuẩn.
- Đã lưu 36 shard (~2,003 GB) và tải lại độc lập trong process mới; số tham số và dự đoán
  khớp. Dự đoán fixture lúc này là chuỗi rỗng/EOS, **chưa đọc đúng**, không giấu kết quả.
- 43 unit/integration tests mới local đạt; CI đầy đủ và pilot FP32 phải đọc đúng SHA mới.

Lượt mô tả trước push có thể phải chạy lại sau sửa lint/code; bằng chứng cuối được gắn
source hash/commit ở PR conversation. Không đổi nhãn checkpoint cũ thành code mới.

## GitHub và tiếp tục

Nhánh `h2lm-scale-1b`, PR base `h2lm-crop-generalization-v2`. Không merge main hoặc đổi các
PR cũ. Workflow `scale-1b.yml` trên standard public Ubuntu CPU, không máy người dùng, không
thuê GPU/API. Nó chạy đầy đủ 1B FP32 tám bước, lưu rồi tải lại toàn bộ, không thay bằng tiny.
Tiny tests chạy riêng để kiểm lỗi. Có chặn public/same-repo PR và timeout 20 phút.

GitHub artifact chỉ giữ báo cáo/index/mã nguồn nhỏ (3 ngày), không tự upload nhiều GB vào
Actions storage hoặc mua dung lượng. Full weights được kiểm trong job nhưng không tồn tại
lâu dài sau khi runner bị xóa; checkpoint BF16 local được giao riêng trong hội thoại.
Không claim đã phát hành một checkpoint GitHub tải được nếu artifact chỉ chứa index.

Mốc quy mô đã được đo; các gate dữ liệu, chất lượng vision/pháp lý, tokenizer và GTX 1070
vẫn chưa hoàn tất. Tiếp theo giữ 1B làm kiến trúc để phát triển trainer/corpus, không tăng
thêm tham số để bù cho dữ liệu sai. Không bỏ các lỗi generalization của EXP-02 hay lỗi TLS
Công báo. Không đổi nhãn AI draft thành gold, không dùng probe đã xem làm benchmark độc lập.

## Tham khảo kỹ thuật

- https://docs.pytorch.org/docs/main/meta.html
- https://docs.pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html
- https://docs.pytorch.org/docs/stable/notes/autograd.html
- https://docs.github.com/en/actions/reference/runners/github-hosted-runners
