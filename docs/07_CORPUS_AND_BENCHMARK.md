# M1-B1 — Chuẩn bị corpus có duyệt và so sánh tokenizer

## Trạng thái

M1-B1 bổ sung **công cụ**, không phải tuyên bố đã có corpus sản phẩm hoặc H2LM thông minh.
Đã có importer TXT có nguồn, tách development/holdout và so sánh nhiều tokenizer bằng CPU.
Bộ demo vẫn là fixture tổng hợp M1-A, không phải luật thật. M1-B2 còn phải thu thập/kiểm định
corpus đại diện, khóa quy trình test độc lập và chọn vocabulary trước khi huấn luyện neural model.
Không dùng số token của vài chục câu để kết luận mô hình hiểu pháp luật tốt hơn.

## 1. Chạy demo

Windows: mở `RUN_CORPUS_BENCHMARK.cmd`, cần Python 3.11 x64 (bản dùng trong CI).
Lần cài dependency cần Internet; các bước xử lý dữ liệu/huấn luyện tokenizer không gọi Web/API.
Chỉ cài môi trường tokenizer, không cài CUDA, không thuê GPU, không đăng ký self-hosted runner.

```powershell
python -m venv .venv-tokenizer
.\.venv-tokenizer\Scripts\python.exe -m pip install -e ".[tokenizer,dev]"
.\.venv-tokenizer\Scripts\python.exe scripts/corpus_benchmark_demo.py
```

Mỗi lần chạy tạo thư mục mới trong `artifacts/corpus-benchmark/`. Muốn đường dẫn cố định:

```powershell
python scripts/corpus_benchmark_demo.py --output artifacts/my-benchmark
```

Thư mục output đã tồn tại sẽ bị từ chối. Demo tạo TXT tổng hợp để kiểm thử importer rồi chạy
4 ứng viên: BPE 512/1024 và Unigram 512/1024. Các con số này không phải vocabulary sản phẩm.

## 2. Chuẩn bị tài liệu thật

Dùng `data/corpus_registry/registry.example.json` làm mẫu ngoài repository. Mỗi văn bản cần
ID, family ID, split, đường dẫn TXT tương đối, hash bytes gốc, nguồn gốc, quyền sử dụng,
người/thời điểm kiểm tra, độ nhạy dữ liệu và `approved: true` sau khi thật sự duyệt.
`human_verified` là khai báo có người kiểm tra, không được dùng cho dữ liệu chỉ do AI đoán nhãn.
Với `teacher_verified`, vẫn phải có đủ thông tin teacher và trạng thái kiểm chứng như M1-A.
Tool không tạo người duyệt, không chứng minh quyền sử dụng và không kiểm tra độ đúng của teacher.

TXT phải được đối chiếu với tài liệu nguồn. Tool không tự OCR, không tự tải PDF và không tự
xóa dấu/sửa chính tả. PDF render/vision sẽ thuộc milestone riêng. Với tài liệu scan chưa có bản
trích xuất đáng tin, giữ trong hàng chờ review, không mặc định OCR ngoài là ground truth.

```powershell
python -m h2lm.tokenization.preparation --registry D:\H2Data\reviewed\registry.json --output D:\H2Data\prepared-v1
```

Hash có thể lấy bằng `Get-FileHash -Algorithm SHA256 <file>`; ghi chữ thường vào registry.
Không đổi nội dung, encoding hoặc xuống dòng sau khi lấy hash. Vẫn giữ bản gốc để truy nguyên.
Mọi văn bản trong cùng một nhóm sửa đổi/phiên bản phải dùng chung `family_id` và cùng split.
Tool chỉ kiểm tra family đã khai báo, chưa tự khám phá các quan hệ này.

## 3. Kết quả importer

`development_manifest.json` chỉ trỏ tới train/validation. `holdout_manifest.json` chỉ trỏ test.
`preparation_report.json` ghi số lượng, tổng ký tự, độ nhạy, hash và thông tin truy nguyên.
Mỗi shard có `source_char_start/end` để ghép lại đúng TXT; đây là offset ký tự Unicode,
**không phải tọa độ trên PDF**. Không làm mất CRLF, tab, dấu tiếng Việt hoặc metaspace literal.
Budget chunk tính cả codec H2, để không bị SentencePiece âm thầm bỏ qua câu quá dài.

Importer từ chối nguồn chưa duyệt, hash sai, đường dẫn vượt thư mục (kể cả symlink), NUL/ký tự
replacement, trùng ID, trùng văn bản/family giữa các split, thiếu split và metadata teacher sai.
Không tự loại bỏ một đoạn whitespace quá dài để dữ liệu trở nên hợp lệ; lỗi yêu cầu review.
Loader M1-A còn loại các đoạn trùng chính xác trong cùng split, có đếm trong báo cáo.
Giới hạn pilot: 1.000 văn bản, 8 MiB/văn bản, tổng TXT 32 MiB; shard JSONL qua loader 64 MiB.
Đây chưa phải pipeline streaming hàng tỷ token. Ngắt giữa lúc ghi có thể để lại thư mục dở;
chọn output mới, không ghi đè thư mục dở để giấu bằng chứng.

## 4. So sánh tokenizer

```powershell
python -m h2lm.tokenization.benchmark --config configs/tokenizer/h2lm_comparison_pilot.yaml --manifest D:\H2Data\prepared-v1\development_manifest.json --output D:\H2Runs\tokenizer-v1
```

`h2lm_comparison_pilot.yaml` mặc định so sánh BPE/Unigram, 8192/16384; sửa YAML để thay ứng viên.
Các minimum train/validation/characters/families chỉ là cổng kỹ thuật ban đầu, chưa phải kết quả
nghiên cứu cỡ mẫu. Chế độ `pilot` từ chối mọi nguồn gắn nhãn `synthetic_fixture`.
Nếu dữ liệu chưa đủ cổng, quy trình dừng trước khi train, không tự hạ tiêu chí hoặc đổi nhãn nguồn.

Chỉ sử dụng validation để so sánh; manifest có bất kỳ nguồn test nào bị từ chối **trước khi đọc shard**.
Test được tách vật lý khỏi đầu vào benchmark, không dùng đi dùng lại để chọn hyperparameter.
Đây chưa phải một kho test chống giả mạo: người có quyền sửa file vẫn sửa được. Quy trình khóa test
và nghiệm thu độc lập của corpus thật phải hoàn thiện ở M1-B2.

Báo cáo từng ứng viên: requested/actual vocabulary, hash model, số token trên cùng validation,
khôi phục text chính xác, unknown, ký tự/byte trên token và thống kê theo nhóm.
Unigram có thể tạo ít token hơn yêu cầu khi corpus nhỏ; dùng **actual_vocab_size**, không gán nhãn
"1024 thực" chỉ vì config yêu cầu 1024. Thời gian đánh giá chỉ là số đo tham khảo của máy chạy.

Lựa chọn tạm thời: ít token validation nhất, hòa thì actual vocabulary nhỏ hơn rồi tên ứng viên.
Không chọn nếu bất kỳ ứng viên nào lỗi hoặc roundtrip không đạt. Không tự đổi config neural model,
không tự phát hành vocabulary/checkpoint, luôn `production_ready: false` ở giai đoạn này.
Chỉ số này không đo legal QA, hiệu lực văn bản, OCR, GPU latency hoặc lượng RAM model sẽ dùng.

## 5. Gián đoạn, giới hạn và tiếp tục

Mỗi ứng viên chạy trong một subprocess với timeout riêng (1–900 giây); tối đa 8 ứng viên.
Demo mặc định 120 giây/ứng viên; pilot 300 giây. Không có job GPU trả phí hoặc tự chạy vô hạn.
Hoàn thành một ứng viên sẽ giữ artifact tokenizer; có thể tiếp tục cả đợt:

```powershell
python -m h2lm.tokenization.benchmark --config configs/tokenizer/h2lm_comparison_pilot.yaml --manifest D:\H2Data\prepared-v1\development_manifest.json --output D:\H2Runs\tokenizer-v1 --resume
```

Resume đối chiếu config, manifest, source code hash và phiên bản Python/SentencePiece; revalidate
hash corpus, load/kiểm tra artifact cũ rồi đánh giá lại. Đổi dữ liệu/cấu hình/code thì dùng run mới.
Ứng viên hoàn tất không train lại. Ứng viên timeout chưa xuất artifact có thể chạy lại.
Artifact có thư mục nhưng thiếu metadata hoặc checksum sai sẽ bị từ chối, không bị ghi đè.
Chưa resume **giữa chừng trong một thuật toán SentencePiece**, chỉ resume giữa các ứng viên.
Đây không phải neural checkpoint có optimizer state; tính năng đó thuộc bước huấn luyện model sau.

`RUNNING.lock` chặn hai tiến trình cùng thư mục local. Không tuyên bố khóa phân tán cho NAS/WebDAV.
Khi tiến trình bị kill, có thể còn lock; phải xác nhận không còn worker rồi mới xóa lock thủ công.
Báo cáo append-only `comparison-0001.json`, `comparison-0002.json`… giữ lịch sử các lần tiếp tục.
CLI: mã 0 thành công, 1 có ứng viên lỗi/không đạt, 2 lỗi dữ liệu/cấu hình/quyền/lock.

## 6. GitHub và quyền riêng tư

CI chỉ chạy bộ fixture công khai trên CPU Windows/Linux. Không auto-upload dữ liệu thật,
không gọi teacher API, không kích hoạt runner trên PC người dùng và không sửa quyền repository.
Tokenizer/model bytes và metadata cũng có thể lộ thông tin corpus; không phát hành chúng từ hồ sơ
riêng tư chỉ vì đã bỏ PDF gốc. Registry, TXT, shard, logs của corpus riêng phải ở ngoài Git/public CI.

PR M1-B1 kế thừa M1-A. Chưa gộp main và chưa bỏ qua dependency sang vision pretraining.
Bằng chứng nghiệm thu remote phải đọc trên đúng PR head SHA. Test local Linux không thay thế
Windows CI hoặc kiểm tra GTX 1070 thật.

## 7. Tài liệu kỹ thuật

- SentencePiece v0.2.1: https://github.com/google/sentencepiece/tree/v0.2.1
- Training parameters: https://github.com/google/sentencepiece/blob/v0.2.1/src/sentencepiece_model.proto
- Normalization: https://google.github.io/sentencepiece/doc/normalization/
- Source candidates chưa duyệt: `data/corpus_registry/README.md`.
