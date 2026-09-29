# H2LM V1 - Dataset & Teacher Strategy

## Mục tiêu

Dataset phải dạy H2LM nhìn, đọc, hiểu và suy luận. Không chỉ dạy image-to-text.

## Nhóm dữ liệu

### A. Vision transcription
page image → exact text

### B. Layout
page image → regions + type + bbox + reading order

### C. Structured reconstruction
page image → Markdown / HTML / JSON

### D. Legal structure
text/page → Chương / Điều / Khoản / Điểm

### E. Legal relations
nhiều văn bản → sửa đổi / bổ sung / bãi bỏ / thay thế / dẫn chiếu

### F. Temporal reasoning
văn bản + thời điểm → quy định có hiệu lực tại thời điểm đó

### G. Grounded QA
question + document pages → answer + exact evidence location

### H. Abstention
question + insufficient evidence → không đủ căn cứ

## Teacher model

Teacher chỉ là công cụ tạo/kiểm tra nhãn.

Mỗi record sinh bởi teacher phải lưu tối thiểu:

- teacher provider/model;
- prompt template version;
- source document hash;
- timestamp hoặc run id;
- verifier state;
- confidence / disagreement nếu có nhiều teacher.

## Verification

Ưu tiên các nhãn kiểm chứng được bằng máy:

- exact text đối chiếu text layer đáng tin cậy;
- Điều/Khoản/Điểm kiểm bằng parser quy tắc;
- ngày hiệu lực kiểm từ metadata/nguồn chính thức;
- bbox kiểm bằng rendering;
- teacher disagreement đưa vào hàng chờ review.

## Không commit dataset lớn vào Git

Git chỉ chứa schema, manifest, sample nhỏ và pipeline. Dataset thật nằm ngoài repository và được tham chiếu bằng manifest + checksum.
