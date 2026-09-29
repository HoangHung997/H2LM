# H2LM V1 - Roadmap

## M0 - Bootstrap
- repository editable;
- config loader;
- reference multimodal model;
- smoke test random weights;
- CI cơ bản.

## M1 - Vietnamese tokenizer
- corpus sạch;
- tokenizer benchmark;
- legal-number/date/article token tests.

## M2 - Document vision pretraining
- OCR tiếng Việt;
- layout;
- bbox;
- reading order;
- page-to-markdown.

## M3 - Language base pretraining
- tiếng Việt ưu tiên;
- English hỗ trợ;
- pháp lý và kiến thức nền;
- continuation/reasoning objectives.

## M4 - Multimodal alignment
- page image + text;
- image + structure;
- grounding.

## M5 - Teacher-supervised reasoning
- grounded QA;
- legal relations;
- temporal reasoning;
- abstention;
- multi-document tasks.

## M6 - Evaluation gate
- frozen benchmarks;
- error taxonomy;
- regression suite.

## M7 - Local inference
- quantization;
- GTX 1070 8 GB memory profiling;
- chunk/tile scheduling;
- local PDF demo.

## Quy tắc

Không chuyển milestone chỉ vì code đã viết. Mỗi milestone phải có evidence và benchmark tương ứng.
