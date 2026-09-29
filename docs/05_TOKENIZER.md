# M1-A — Tokenizer tiếng Việt: công cụ phát triển

## Trạng thái và giới hạn

Đã có công cụ train/encode/decode/evaluate chạy trên CPU. Đây là **tokenizer phát triển**,
không phải checkpoint H2LM biết đọc PDF, hiểu luật hoặc suy luận. M1 chưa hoàn tất nghiên cứu:
chưa có corpus pháp lý thật đã kiểm chứng và chưa khóa vocabulary sản phẩm.

Corpus kèm repo gồm 24 mẫu train, 12 validation, 12 test do dự án tạo để kiểm thử kỹ thuật.
Số hiệu/ngày tháng trong đó là ví dụ giả lập, không phải văn bản hay căn cứ pháp lý thật.
Không dùng những ví dụ này để đưa ra kết luận pháp lý.

## 1. Chạy thử, không cần GPU

Từ thư mục repository, mở `RUN_TOKENIZER_DEMO.cmd` trên Windows.
Script tạo môi trường `.venv-tokenizer`, cài dependency qua pip rồi chạy demo.
Lần cài dependency cần Internet; train và đánh giá mẫu sau đó chạy offline, không gọi teacher API.
Script không cài PyTorch/CUDA, không tải weights Gemma/Qwen và không chỉnh driver GPU.
Mỗi lần chạy tạo một thư mục mới trong `artifacts/tokenizer/`, không ghi đè run cũ.

Cách chạy bằng terminal, không cần kích hoạt PowerShell:

```powershell
python -m venv .venv-tokenizer
.\.venv-tokenizer\Scripts\python.exe -m pip install -e ".[tokenizer,dev]"
.\.venv-tokenizer\Scripts\python.exe scripts/tokenizer_demo.py
.\.venv-tokenizer\Scripts\python.exe -m pytest -q tests/test_m1_tokenizer.py
```

Python 3.11 là phiên bản trong CI Windows/Linux. CI không thay thế thử nghiệm trên GTX 1070 thật.

## 2. Những gì chủ dự án có thể tự sửa

| Nội dung | Nơi chỉnh |
|---|---|
| Kích thước vocabulary, BPE/Unigram, giới hạn corpus | `configs/tokenizer/h2lm_tokenizer_dev.yaml` |
| Dữ liệu, nguồn, quyền sử dụng, split và checksum | `data/tokenizer_sample/manifest.json` hoặc manifest riêng |
| Kiểm tra dữ liệu / chống rò rỉ tập đánh giá | `src/h2lm/tokenization/corpus.py` |
| Train, encode/decode, special IDs, đánh giá | `src/h2lm/tokenization/tokenizer.py` |
| Giao diện dòng lệnh | `src/h2lm/tokenization/cli.py` |
| Kiểm thử hồi quy | `tests/test_m1_tokenizer.py` |

Đổi `name`/`vocab_size`/`model_type` trong YAML và chọn thư mục output mới để thử một ứng viên.
Không sửa vocabulary của một model ngôn ngữ đã train mà không đồng bộ embedding/checkpoint.

## 3. Quyết định thiết kế

Dùng thư viện SentencePiece 0.2.1 để **tự train vocabulary**, không nhập vocabulary hay neural
weights của model khác. Hai ứng viên được hỗ trợ: `bpe` (mặc định) và `unigram`.
Đây là quyết định cho reference pipeline, chưa phải kết luận tokenizer nào tốt nhất.

Giữ nguyên ký tự: identity normalization; không lowercase; không xóa dấu; không NFKC;
không gộp khoảng trắng; không tự thêm khoảng trắng đầu câu. Dấu cách, tab, CRLF/LF,
chữ số và ký hiệu phải qua encode → decode không bị thay đổi.
`split_digits` là lựa chọn chia token, không phải bộ kiểm tra tính hợp lệ của ngày hay số hiệu.
Byte fallback xử lý ký tự chưa thấy khi train. Có kiểm thử văn bản tiếng Việt dạng Unicode
composed và decomposed, ký hiệu kỹ thuật, chữ ngoài corpus và emoji.

SentencePiece dùng `▁` (U+2581) làm ký hiệu khoảng trắng. H2 thêm codec `h2-exact-text-v1`:
literal `▁` được bảo vệ thành U+E000 + `1`, literal U+E000 thành U+E000 + `0` trước khi encode,
và khôi phục sau decode. Vì vậy **phải dùng H2Tokenizer**, không dùng `.model` trực tiếp rồi
cho rằng văn bản nào cũng khôi phục đúng. Codec này cũng phải được triển khai khi xuất runtime khác;
chưa tuyên bố tương thích Ollama/GGUF.

Các special IDs cố định: 0 `<unk>`, 1 `<s>`, 2 `</s>`, 3 `<pad>`, 4 `<image>`, 5 `<page>`,
6 `<bbox>`, 7 `<answer>`, 8 `<evidence>`, 9 `<abstain>`.
Chuỗi `<image>` xuất hiện trong tài liệu được mã hóa như **văn bản**, không tự biến thành lệnh.
Pipeline sau này chủ động chèn control IDs. BOS/EOS chỉ thêm khi gọi `encode(..., bos=True, eos=True)`.
`decode` loại control IDs theo SentencePiece và từ chối ID ngoài vocabulary hoặc `<unk>`.
Khôi phục chính xác được kiểm thử cho chuỗi encode hợp lệ; chưa bảo đảm mọi chuỗi ID do model
ngẫu nhiên sinh ra đều là văn bản/escape hợp lệ.

## 4. Corpus contract v1

Manifest JSON có `schema_version: 1` và danh sách `sources`. Mỗi source cần:
`id`, `path` tới JSONL, `split` (`train`/`validation`/`test`), `sha256` của **bytes gốc**,
`source_uri`, `rights_note`, `approved: true`, `origin`.
Origin chỉ nhận `synthetic_fixture`, `human_verified`, `teacher_verified`.
Teacher source còn cần `teacher.provider`, `model`, `prompt_version`, `run_id`,
`source_sha256` và `review_status: verified`. Đây là kiểm tra metadata đã khai báo,
**không phải bằng chứng tự động rằng teacher trả lời đúng hay nguồn được phép sử dụng**.
Người lập manifest phải kiểm chứng nội dung và điều khoản sử dụng trước.

Mỗi dòng JSONL là một object: `id`, `document_id`, `family_id`, `text`, `category`.
Các trang cùng văn bản giữ chung `document_id`; các phiên bản/sửa đổi liên quan cần được gán
cùng `family_id` trước khi chia train/evaluation. Tool không tự suy ra family từ tên file.

Các kiểm tra chặn: checksum sai, source chưa duyệt, đường dẫn ra ngoài thư mục manifest,
source/record ID trùng, teacher chưa xác nhận, cùng text/document/family ở hai split.
Hash text dùng NFC để nhận ra cả hai dạng Unicode tương đương. Nội dung trùng trong một split
được loại khỏi lần train/đánh giá và ghi số lượng. Chống trùng gần đúng/paraphrase chưa có.

Loader M1-A có giới hạn 64 MiB/100.000 record mặc định; đọc corpus trong RAM. Đây chưa phải
pipeline streaming cho hàng tỷ token. Dòng quá dài bị báo lỗi, không bị SentencePiece âm thầm bỏ qua.
Phải chủ động chia đoạn, giữ document/family ID; không sửa văn bản gốc để lách kiểm tra.
Không đưa PDF trực tiếp vào tokenizer: render/vision là milestone riêng.

## 5. Lệnh train và kiểm tra

```powershell
python -m h2lm.tokenization.cli train --config configs/tokenizer/h2lm_tokenizer_dev.yaml --manifest data/tokenizer_sample/manifest.json --output artifacts/my-tokenizer-run
python -m h2lm.tokenization.cli evaluate --tokenizer artifacts/my-tokenizer-run --manifest data/tokenizer_sample/manifest.json --split validation --report artifacts/my-tokenizer-run/validation.json
python -m h2lm.tokenization.cli inspect --tokenizer artifacts/my-tokenizer-run --text-file data/my-text.txt
```

CLI trả mã 0 khi thành công, 1 khi evaluation gate không đạt, 2 khi dữ liệu/cấu hình không hợp lệ.
Không ghi đè output/report cũ. Test split là fixture hồi quy công khai trong demo; bộ test sản phẩm
sau này phải được khóa độc lập, không dùng lặp đi lặp lại để chọn hyperparameter.

## 6. Artifact và bằng chứng

Mỗi run gồm `tokenizer.model`, `tokenizer.vocab`, `metadata.json`, bản sao manifest nguồn.
Demo thêm `validation_report.json` và `test_report.json`.
Metadata ghi phiên bản thư viện/Python, effective config, hash model, hash manifest, hash source code,
Git revision và trạng thái working tree nếu có Git. Không có Git thì revision để `null`, không bịa SHA.
Fingerprint của tập train lưu trong artifact để chặn cả việc dùng manifest khác đổi nhãn dữ liệu train
thành evaluation. Checksum bảo vệ tính toàn vẹn, không chứng minh nguồn đáng tin hay quyền sử dụng.

Báo cáo đo exact round-trip, unknown tokens, tokens/characters/bytes và từng nhóm case.
Không gọi đó là độ chính xác OCR hoặc suy luận pháp luật. Không suy ra ưu thế so với Gemma từ fixture nhỏ.
Vocabulary/model artifact cũng có thể lộ thông tin corpus; không phát hành artifact của hồ sơ riêng tư
chỉ vì không đính kèm PDF gốc. CI hiện chỉ xuất fixture tổng hợp công khai.

Test tích hợp lấy token IDs tiếng Việt thật, đưa vào model random-weight của M0, tính loss và backward
qua vision encoder. Nó kiểm tra kết nối kỹ thuật, không huấn luyện ra trí thông minh.
Embedding phải có `vocab_size` đúng actual vocabulary; dùng `require_model_vocab` để phát hiện sai.
Không thay âm thầm YAML model DEV 8192 bằng tokenizer fixture 1024.

## 7. Phần M1-B còn phải làm

Thu thập corpus tiếng Việt/pháp lý có nguồn và quyền sử dụng; rà soát dữ liệu nhạy cảm;
chia family chống nhiễm; có tập validation đại diện và bộ test khóa; so sánh BPE/Unigram và nhiều
kích thước vocabulary trên cùng corpus; khóa vocabulary/hash trước khi pretrain language model.
Chưa chuyển sang M2 chỉ vì fixture này đạt.

## Tài liệu kỹ thuật tham khảo

- SentencePiece source và API: https://github.com/google/sentencepiece/tree/v0.2.1
- Training parameters/schema: https://github.com/google/sentencepiece/blob/v0.2.1/src/sentencepiece_model.proto
- Python API: https://github.com/google/sentencepiece/blob/v0.2.1/python/README.md
