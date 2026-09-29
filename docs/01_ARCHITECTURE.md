# H2LM V1 - Architecture

## Trạng thái

Kiến trúc sản phẩm chưa khóa tuyệt đối. Mục tiêu hiện tại là xây một reference implementation có thể chạy, test và thay đổi từng thành phần.

## Luồng chính

PDF page / image
→ Document Vision Encoder
→ Visual token compressor / projector
→ H2LM language-reasoning core
→ structured output / answer / grounding

## Document Vision Encoder

Yêu cầu:

- hỗ trợ tile ảnh độ phân giải cao;
- giữ thông tin chữ nhỏ;
- giữ tọa độ không gian;
- không tối ưu chủ yếu cho natural-image classification;
- có thể pretrain riêng trước khi ghép language core.

## Fusion

V1 sử dụng prefix multimodal semantics:

- image tokens có thể tương tác với nhau;
- text tokens được nhìn toàn bộ image prefix;
- text tokens chỉ được nhìn các text token trước nó.

Reference implementation ban đầu dùng projector + prefix attention. Sau benchmark có thể thay bằng resampler, cross-attention hoặc kiến trúc khác mà không đổi dataset contract.

## Language / Reasoning Core

Yêu cầu:

- causal decoder;
- tokenizer tối ưu tiếng Việt + pháp lý + số + ký hiệu;
- hỗ trợ structured outputs;
- grounding token / coordinate representation;
- reasoning trên quan hệ Điều-Khoản-Điểm và thời gian hiệu lực.

## Quy mô

Target 2-3B parameters hiện chỉ là envelope thiết kế, không phải con số đã khóa.

Repository có hai lớp cấu hình:

- DEV: model rất nhỏ để chạy smoke test trên máy cá nhân.
- TARGET: mô tả hướng model sản phẩm, chỉ được khóa sau benchmark/memory profiling.

## Long-document strategy

Không nhét hàng nghìn trang vào context cùng lúc.

H2LM sẽ tạo page/document representations và dùng document memory/retrieval để chọn bằng chứng cần thiết, sau đó reasoning sâu trên tập bằng chứng nhỏ hơn.

Document memory là thành phần của hệ thống H2LM, không phải một AI agent thay model suy luận.

## Khả năng thay đổi

Mọi hyperparameter chính phải nằm trong YAML. Người dùng có thể thay encoder depth, d_model, số layer, số head, patch size, image size và vocabulary mà không sửa logic lõi.
