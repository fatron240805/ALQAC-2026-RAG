# run_pipeline.py — Runbook & Test Log (Khoa, T0-1)

## Cấu trúc file bàn giao

```
orchestration/
  __init__.py
  config.py              # PipelineConfig — cấu hình thống nhất (paths, alpha, top_k, rate limit, submission quota)
  interfaces.py           # ABC cho Reranker / CitationUsefulnessFilter / ReasoningAgent / Verifier / LLMClient
                          # + stub passthrough mặc định (PassthroughReranker, NoOpCitationFilter, PassthroughVerifier)
                          # + PromptTemplateReasoningAgent: implementation THẬT, render prompt_v0.md + gọi LLMClient
  rate_limiter.py         # RateLimiter — ép tối thiểu 5s/API call (luật thi)
  submission_tracker.py   # SubmissionTracker — giới hạn tối đa 3 submit/ngày, log CSV + config_hash
  data_adapters.py        # Adapter schema: chunks.jsonl (field phẳng) -> HybridIndexer ({doc_id, content, metadata})
                          # + load_test_cases() đọc ALQAC2026_public_test.json
  run_pipeline.py         # Entry point CLI: build-index / run / evaluate / submit

retrieval/
  indexing.py             # ĐÃ PATCH: thêm encode_query(), bm25_scores(), search(), load_index()
                          # (bản gốc chỉ có build_index()/save_index(), CHƯA có cách truy vấn)
```

## Gap quan trọng đã phát hiện & xử lý

1. **`HybridIndexer` không có method query/search.**
   Bản gốc chỉ build & lưu index, không có cách nào để hỏi "top-k tài liệu nào khớp với câu query mới".
   -> Đã bổ sung `search()`, `encode_query()`, `bm25_scores()`, `load_index()` vào `retrieval/indexing.py`,
   tái dùng đúng logic `_tokenize`/fallback embedding đã có để đảm bảo query vector và corpus vector
   cùng không gian. Công thức fusion đúng theo Plan.md: `score = alpha*BM25_norm + (1-alpha)*dense`, alpha=0.45.

2. **Schema mismatch giữa `chunks.jsonl` thật và interface của `HybridIndexer`.**
   `HybridIndexer._document_metadata()` kỳ vọng field `metadata` lồng bên trong mỗi document.
   Nhưng `chunks.jsonl` thật có field phẳng (`aid`, `law_id`, `chunk_id`, `deprecated`, ...) không có key `metadata`.
   Nếu không xử lý, mọi `law_id`/`aid` sẽ biến mất khỏi index -> `law_evidence` trong output luôn rỗng
   -> vỡ luôn 10% Micro Law F1 của điểm ALQAC.
   -> Đã viết `data_adapters.chunk_to_indexer_doc()` để map đúng field trước khi đưa vào `HybridIndexer.build_index()`.
   Đã test và xác nhận `law_evidence` ra đúng `[{"law_id": "47/2010/QH12", "aid": 270}, ...]`.

3. **`case_evidence` trong `metrics.py` chưa rõ nguồn dữ liệu.**
   Theo `metrics.py`, `case_evidence` là list segment ID của *văn bản vụ án* (không phải luật) — có vẻ
   là một sub-task "citation usefulness" khác, tách biệt với `law_evidence`. Chưa thấy schema/corpus cho
   phần "case segments" này trong các file đã review. Đã để placeholder `[]` để `evaluate_alqac_system()`
   không crash, nhưng **cần Thịnh/Tuấn Anh xác nhận schema thật** trước khi tính điểm `Penalized_Case_Recall`
   có ý nghĩa.

4. **Schema `ALQAC2026_public_test.json` chưa xác nhận 100%.**
   `load_test_cases()` đang giả định key `case_id`/`id` và `case_query`/`query`/`query_text`. Cần đối chiếu
   với file thật khi có (chưa được gửi trong đợt review này).

## Test log (dữ liệu mẫu, dry-run)

Input: 3 chunks (2 active + 1 `deprecated: true`), 1 test case, 1 gold label.

```
$ python -m orchestration.run_pipeline build-index
Indexed 2 active chunks -> data/index/     # <- đúng: chunk deprecated bị loại

$ python -m orchestration.run_pipeline run --dry-run
[0] case_id=0001 -> PARTIAL_B_WIN           # <- DryRunLLMClient, KHÔNG phải suy luận thật

$ cat experiments/run_v0_baseline.json
[{"case_id": "0001", "prediction": "PARTIAL_B_WIN", "confidence": 0.1,
  "law_evidence": [{"law_id": "47/2010/QH12", "aid": 270}, {"law_id": "47/2010/QH12", "aid": 271}],
  "case_evidence": [], "api_calls": 2}]

$ python -m orchestration.run_pipeline evaluate --pred experiments/run_v0_baseline.json
ALQAC_Final_Score = 0.0667                  # <- thấp vì DryRunLLMClient không suy luận thật, đúng như kỳ vọng

$ python -m orchestration.run_pipeline submit --score 0.42 --notes "test dry-run"
Submission logged (config_hash=6b226f2f2d). Còn lại hôm nay: 2/3.
```

Toàn bộ 4 lệnh CLI chạy sạch, không lỗi. Chưa test với `SentenceTransformer`/`rank_bm25` thật cài đủ
(môi trường test chỉ có numpy/pandas/scikit-learn/rank_bm25 tối thiểu) — nhưng code có fallback an toàn
cho cả 2 trường hợp (có/không có các thư viện optional này), theo đúng thiết kế gốc của `indexing.py`.

## Việc cần làm tiếp (theo owner)

- **Hưng (T2-1/T2-2):** cân nhắc di chuyển `search()`/`load_index()` từ `retrieval/indexing.py` sang
  `retrieval/router.py` khi làm query decomposition; implement `Reranker` thật (cross-encoder) theo
  interface trong `orchestration/interfaces.py`.
- **Thịnh (T2-3/T3-1/T3-2):** implement `CitationUsefulnessFilter`, hoàn thiện fast/deep mode trong
  `PromptTemplateReasoningAgent.answer()` (hiện chỉ có fast mode), implement `Verifier` thật.
  Xác nhận schema `case_evidence`/`n_segments` cho case-level citation task.
- **Tuấn Anh:** xác nhận công thức `Penalized_Case_Recall` hoạt động đúng khi `case_evidence` có dữ liệu thật.
- **Khoa (bạn):** chốt LLM backend (HF transformers local / vLLM / Ollama) -> implement 1 class kế thừa
  `LLMClient` trong `orchestration/interfaces.py`, thay `NotConfiguredLLMClient` bằng backend thật trong
  `run_pipeline.py::run_pipeline()`. Xác nhận schema thật của `ALQAC2026_public_test.json`.
