from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import os
from pathlib import Path
from typing import Any

# --- KÍCH HOẠT ĐỌC BIẾN MÔI TRƯỜNG TỪ FILE .ENV NGAY KHI KHỞI CHẠY ---
from dotenv import load_dotenv
load_dotenv()
# --------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import evaluate_alqac_system
from orchestration.case_retrieval_client import CaseRetrievalClient
from orchestration.config import PROJECT_ROOT, PipelineConfig
from orchestration.data_adapters import load_case_id_filter, load_test_cases, stream_active_indexer_docs
from orchestration.interfaces import (
    CitationUsefulnessFilter,
    DryRunLLMClient,
    NoOpCitationFilter,
    LocalOllamaClient,
    PassthroughReranker,
    PassthroughVerifier,
    PromptTemplateReasoningAgent,
    Reranker,
    ReasoningAgent,
    Verifier,
)
from orchestration.rate_limiter import RateLimiter
from orchestration.submission_tracker import SubmissionGuardrailError, SubmissionTracker
from retrieval.deprecated_filter import DeprecatedFilter
from retrieval.indexing import HybridIndexer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("run_pipeline")


class _DryRunCaseRetrievalClient:
    def __init__(self) -> None:
        self.call_count = 0

    def retrieve_multi(self, queries: list[str], case_id: str) -> list[Any]:
        from orchestration.case_retrieval_client import CaseEvidenceHit
        self.call_count += len(queries)
        return [
            CaseEvidenceHit(chunk_id=f"{case_id}_dryrun_chunk_0", text="[dry-run] no real retrieval.", score=0.0)
            for _ in queries[:1]
        ]


def build_index(config: PipelineConfig) -> HybridIndexer:
    logger.info("Building index from %s", config.chunks_path)
    deprecated_filter = DeprecatedFilter()
    docs = stream_active_indexer_docs(config.chunks_path, deprecated_filter)

    indexer = HybridIndexer(model_name=config.embedding_model)
    indexer.build_index(docs)
    indexer.save_index(config.index_path)
    logger.info("Indexed %d active chunks -> %s", len(indexer.doc_ids), config.index_path)
    return indexer


def load_or_build_index(config: PipelineConfig, force_rebuild: bool = False) -> HybridIndexer:
    index_dir = Path(config.index_path)
    required_files = ("documents.jsonl", "doc_ids.json", "bm25_model.pkl")
    if not force_rebuild and all((index_dir / name).exists() for name in required_files):
        logger.info("Loading existing index from %s", index_dir)
        return HybridIndexer.load_index(index_dir, model_name=config.embedding_model)
    return build_index(config)


def process_case(
    case: dict[str, Any],
    *,
    indexer: HybridIndexer,
    reranker: Reranker,
    citation_filter: CitationUsefulnessFilter,
    case_retrieval_client: CaseRetrievalClient,
    reasoning_agent: ReasoningAgent,
    verifier: Verifier,
    config: PipelineConfig,
) -> dict[str, Any]:
    case_id = case["case_id"]
    case_query = case["case_query"]

    # 1a. Tìm kiếm dữ liệu Luật pháp (Local Indexer - Không tính vào c_i)
    law_candidates = indexer.search(
        case_query, top_k=config.top_k_before_rerank, alpha=config.hybrid_alpha
    )
    law_reranked = reranker.rerank(case_query, law_candidates, top_k=config.top_k_after_rerank)
    law_evidence = citation_filter.filter(case_query, law_reranked)

    # 1b. Tìm kiếm dữ liệu Vụ án (Gọi API Ban tổ chức - Tính vào c_i)
    case_evidence_hits = case_retrieval_client.retrieve_multi(
        queries=[case_query][: config.max_case_retrieval_calls_per_case],
        case_id=case_id,
    )

    # 2. Thực hiện suy luận lập luận vụ án dựa trên mô hình ngôn ngữ
    answer = reasoning_agent.answer(case_id, case_query, law_evidence, case_evidence_hits)

    # 3. Thẩm định kết quả dự đoán
    verified = verifier.verify(answer, law_evidence)

    # 4. Đóng gói cấu trúc dữ liệu đầu ra
    law_evidence_out = [
        {"law_id": e["metadata"].get("law_id"), "aid": e["metadata"].get("aid")}
        for e in law_evidence
        if e["metadata"].get("law_id") is not None
    ]
    case_evidence_out = [hit.chunk_id for hit in case_evidence_hits if hit.chunk_id]

    return {
        "case_id": case_id,
        "prediction": verified.get("label"),
        "confidence": verified.get("confidence"),
        "justification": verified.get("justification"),
        "law_evidence": law_evidence_out,
        "case_evidence": case_evidence_out,
        "api_calls": case_retrieval_client.call_count,
    }


def run_pipeline(
    config: PipelineConfig,
    *,
    dry_run: bool = False,
    limit: int | None = None,
    force_rebuild_index: bool = False,
    case_id_filter: set[str] | None = None,
) -> Path:
    indexer = load_or_build_index(config, force_rebuild=force_rebuild_index)

    if dry_run:
        llm_client = DryRunLLMClient()
    else:
        # Nạp client tự động hoàn toàn bằng các tham số môi trường trong .env
        llm_client = LocalOllamaClient.from_env()
        
    reasoning_agent = PromptTemplateReasoningAgent(llm_client=llm_client, prompt_path=config.prompt_path)
    reranker = PassthroughReranker()
    citation_filter = NoOpCitationFilter()
    verifier = PassthroughVerifier()

    if dry_run:
        case_retrieval_client = _DryRunCaseRetrievalClient()
    else:
        # Rate limit 5s/request chuẩn hóa từ file cấu hình hệ thống
        rate_limiter = RateLimiter(config.seconds_between_api_calls)
        case_retrieval_client = CaseRetrievalClient.from_env(rate_limiter=rate_limiter)

    config.experiments_dir.mkdir(parents=True, exist_ok=True)
    debug_log_path = config.experiments_dir / f"run_{config.run_tag}.debug.jsonl"
    final_output_path = config.experiments_dir / f"run_{config.run_tag}.json"

    start_time = time.monotonic()
    predictions: list[dict[str, Any]] = []

    with debug_log_path.open("w", encoding="utf-8") as debug_handle:
        matched_so_far = 0
        for i, case in enumerate(load_test_cases(config.public_test_path)):
            if case_id_filter is not None:
                if case["case_id"] not in case_id_filter:
                    continue
                matched_so_far += 1
            elif limit is not None and i >= limit:
                break

            try:
                pred = process_case(
                    case,
                    indexer=indexer,
                    reranker=reranker,
                    citation_filter=citation_filter,
                    case_retrieval_client=case_retrieval_client,
                    reasoning_agent=reasoning_agent,
                    verifier=verifier,
                    config=config,
                )
            except Exception:
                logger.exception(f"Lỗi nghiêm trọng không mong muốn tại Case {case.get('case_id')}")
                # Đã loại bỏ hoàn toàn time.sleep() sai ngữ nghĩa tại đây
                continue

            debug_handle.write(json.dumps(pred, ensure_ascii=False) + "\n")
            predictions.append(pred)
            logger.info("[%d] case_id=%s -> %s", i, pred["case_id"], pred["prediction"])

            if case_id_filter is not None and matched_so_far >= len(case_id_filter):
                break

    final_output_path.write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    elapsed = time.monotonic() - start_time
    logger.info(
        "Done: %d cases in %.1fs (%.2fs/case) -> %s",
        len(predictions),
        elapsed,
        elapsed / max(1, len(predictions)),
        final_output_path,
    )
    return final_output_path


_ALQAC_VALID_LABELS = {"A_WIN", "PARTIAL_A_WIN", "PARTIAL_B_WIN", "B_WIN"}


def _validate_gold_labels(gold_data: list[dict[str, Any]]) -> None:
    if not gold_data:
        raise ValueError(
            "data/local_validation_gold.json rỗng (0 case). Chạy "
            "'python -m evaluation.scaffold_gold_labels' rồi điền tay trước."
        )
    invalid = [
        (item.get("case_id"), item.get("prediction"))
        for item in gold_data
        if item.get("prediction") not in _ALQAC_VALID_LABELS
    ]
    if invalid:
        raise ValueError(
            f"{len(invalid)}/{len(gold_data)} case trong gold có 'prediction' không hợp lệ "
            f"(phải là 1 trong {sorted(_ALQAC_VALID_LABELS)}). "
            f"Case lỗi (tối đa 5 case đầu): {invalid[:5]}"
        )


def build_submission_json(internal_pred_path: Path, output_path: Path) -> Path:
    internal = json.loads(internal_pred_path.read_text(encoding="utf-8"))

    seen_case_ids: set[str] = set()
    submission: list[dict[str, Any]] = []
    for item in internal:
        case_id = str(item["case_id"])
        if case_id in seen_case_ids:
            raise ValueError(f"case_id trùng lặp trong predictions: {case_id}")
        seen_case_ids.add(case_id)

        prediction = item.get("prediction")
        if prediction not in _ALQAC_VALID_LABELS:
            raise ValueError(
                f"case_id={case_id}: prediction '{prediction}' không hợp lệ "
                f"(phải là 1 trong {sorted(_ALQAC_VALID_LABELS)})"
            )

        submission.append(
            {
                "case_id": case_id,
                "prediction": prediction,
                "case_evidence": item.get("case_evidence", []),
                "law_evidence": item.get("law_evidence", []),
            }
        )

    output_path.write_text(json.dumps(submission, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Đã build %d case -> %s (đúng schema chính thức)", len(submission), output_path)
    return output_path


def run_local_evaluation(config: PipelineConfig, pred_path: Path) -> dict[str, Any]:
    gold_data = json.loads(Path(config.gold_path).read_text(encoding="utf-8"))
    pred_data = json.loads(Path(pred_path).read_text(encoding="utf-8"))
    _validate_gold_labels(gold_data)
    report, cm_df = evaluate_alqac_system(gold_data, pred_data)

    report_path = config.experiments_dir / f"report_{config.run_tag}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("ALQAC_Final_Score = %.4f (report -> %s)", report["ALQAC_Final_Score"], report_path)
    logger.info("\n%s", cm_df)
    return report


def submit(config: PipelineConfig, score: float | None, notes: str) -> None:
    tracker = SubmissionTracker(config.submission_tracker_path, config.max_submissions_per_day)
    tracker.register_submission(config.config_hash(), public_score=score, notes=notes)
    logger.info(
        "Submission logged (config_hash=%s). Còn lại hôm nay: %d/%d.",
        config.config_hash(),
        tracker.remaining_today(),
        config.max_submissions_per_day,
    )


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="ALQAC 2026 pipeline orchestrator")
    parser.add_argument("--config", type=Path, default=None, help="JSON config override file")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("build-index", help="Build/rebuild the hybrid BM25+dense index")

    run_parser = subparsers.add_parser("run", help="Run the full pipeline over the public test set")
    run_parser.add_argument("--dry-run", action="store_true", help="Use stub LLM, no real inference")
    run_parser.add_argument("--limit", type=int, default=None, help="Only process the first N cases")
    run_parser.add_argument("--rebuild-index", action="store_true")
    run_parser.add_argument(
        "--case-ids-file",
        type=Path,
        default=None,
        help="Chỉ xử lý đúng tập case_id có trong file này.",
    )

    eval_parser = subparsers.add_parser("evaluate", help="Score a predictions file against gold")
    eval_parser.add_argument("--pred", type=Path, default=None)

    export_parser = subparsers.add_parser(
        "export-submission", help="Build submission.json từ predictions nội bộ"
    )
    export_parser.add_argument("--pred", type=Path, default=None)
    export_parser.add_argument("--output", type=Path, default=Path("submission.json"))

    submit_parser = subparsers.add_parser("submit", help="Log a leaderboard submission attempt")
    submit_parser.add_argument("--score", type=float, default=None)
    submit_parser.add_argument("--notes", type=str, default="")

    args = parser.parse_args()
    config = PipelineConfig.from_file(args.config) if args.config else PipelineConfig()

    if args.command == "build-index":
        build_index(config)
    elif args.command == "run":
        case_id_filter = load_case_id_filter(args.case_ids_file) if args.case_ids_file else None
        if case_id_filter is not None:
            logger.info("Chỉ xử lý %d case_id từ %s", len(case_id_filter), args.case_ids_file)
        run_pipeline(
            config,
            dry_run=args.dry_run,
            limit=args.limit,
            force_rebuild_index=args.rebuild_index,
            case_id_filter=case_id_filter,
        )
    elif args.command == "evaluate":
        pred_path = args.pred or (config.experiments_dir / f"run_{config.run_tag}.json")
        if not pred_path.exists():
            parser.error(f"Không tìm thấy file predictions tại {pred_path}.")
        try:
            run_local_evaluation(config, pred_path)
        except ValueError as exc:
            logger.error(str(exc))
            raise SystemExit(1) from exc
    elif args.command == "export-submission":
        pred_path = args.pred or (config.experiments_dir / f"run_{config.run_tag}.json")
        if not pred_path.exists():
            parser.error(f"Không tìm thấy {pred_path}.")
        try:
            build_submission_json(pred_path, args.output)
        except ValueError as exc:
            logger.error(str(exc))
            raise SystemExit(1) from exc
    elif args.command == "submit":
        try:
            submit(config, args.score, args.notes)
        except SubmissionGuardrailError as exc:
            logger.error(str(exc))
            raise SystemExit(1) from exc


if __name__ == "__main__":
    main()