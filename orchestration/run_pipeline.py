from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import os
from functools import partial
from pathlib import Path
from typing import Any

# --- KÍCH HOẠT ĐỌC BIẾN MÔI TRƯỜNG TỪ FILE .ENV NGAY KHI KHỞI CHẠY ---
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional local convenience dependency
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()
else:
    env_path = Path(".env")
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
# --------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import evaluate_alqac_system
from orchestration.case_retrieval_client import CaseRetrievalClient
from orchestration.config import PROJECT_ROOT, PipelineConfig
from orchestration.data_adapters import load_case_id_filter, load_test_cases, stream_active_indexer_docs
from orchestration.interfaces import (
    CitationUsefulnessFilter,
    DryRunLLMClient,
    LocalOllamaClient,
    PromptTemplateReasoningAgent,
    Reranker,
    ReasoningAgent,
    StatutoryConsistencyVerifier,
    Verifier,
)
from orchestration.rate_limiter import RateLimiter
from orchestration.submission_tracker import SubmissionGuardrailError, SubmissionTracker
from retrieval.deprecated_filter import DeprecatedFilter
from retrieval.citation_usefulness import HeuristicCitationUsefulnessFilter
from retrieval.graph_retriever import GraphRetrieverConfig, LegalGraphRetriever
from retrieval.indexing import HybridIndexer
from retrieval.reranker import LexicalOverlapReranker
from retrieval.router import DocumentRouter
from graph_construct import Neo4jGraphStore, build_graph_records, write_graph_artifacts

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

    indexer = HybridIndexer(
        model_name=config.embedding_model,
        use_bge_m3=config.use_bge_m3,
        use_sentence_transformer=config.use_sentence_transformer,
        device=config.retrieval_device,
        batch_size=config.retrieval_batch_size,
        max_length=config.retrieval_max_length,
    )
    indexer.build_index(docs)
    indexer.save_index(config.index_path)
    logger.info("Indexed %d active chunks -> %s", len(indexer.doc_ids), config.index_path)
    return indexer


def load_or_build_index(config: PipelineConfig, force_rebuild: bool = False) -> HybridIndexer:
    index_dir = Path(config.index_path)
    required_files = ("documents.jsonl", "doc_ids.json", "bm25_model.pkl", "index_meta.json")
    if not force_rebuild and all((index_dir / name).exists() for name in required_files):
        try:
            meta = json.loads((index_dir / "index_meta.json").read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        if isinstance(meta, dict):
            if bool(meta.get("use_bge_m3")) != bool(config.use_bge_m3):
                logger.info("Index retrieval backend changed; rebuilding %s", index_dir)
                return build_index(config)
            if bool(meta.get("use_sentence_transformer")) != bool(config.use_sentence_transformer):
                logger.info("Index semantic mode changed; rebuilding %s", index_dir)
                return build_index(config)
            if str(meta.get("model_name") or "") != str(config.embedding_model):
                logger.info("Index embedding model changed; rebuilding %s", index_dir)
                return build_index(config)
        logger.info("Loading existing index from %s", index_dir)
        return HybridIndexer.load_index(
            index_dir,
            model_name=config.embedding_model,
            use_bge_m3=config.use_bge_m3,
            use_sentence_transformer=config.use_sentence_transformer,
            device=config.retrieval_device,
            batch_size=config.retrieval_batch_size,
            max_length=config.retrieval_max_length,
        )
    return build_index(config)


def build_graph_retriever(
    config: PipelineConfig,
    indexer: HybridIndexer,
    *,
    dry_run: bool = False,
    require_graph: bool = False,
) -> LegalGraphRetriever | None:
    if dry_run:
        return None
    if not config.use_graph_retrieval:
        if require_graph:
            raise RuntimeError("Graph retrieval is disabled in config but --require-graph was requested.")
        return None

    graph_store: Neo4jGraphStore | None = None
    try:
        graph_store = Neo4jGraphStore()
        graph_store.verify_connectivity()
        graph_node_count = graph_store.count_legal_nodes()
        if graph_node_count <= 0:
            raise RuntimeError(
                "Neo4j LegalNode graph is empty. Run "
                "`python -m orchestration.run_pipeline build-graph --import-neo4j --clear-graph` first."
            )
    except Exception as exc:
        if graph_store is not None:
            graph_store.close()
        if require_graph:
            raise RuntimeError(f"Neo4j graph retrieval required but unavailable: {exc}") from exc
        logger.warning("Neo4j graph retrieval unavailable, falling back to flat retrieval: %s", exc)
        return None

    logger.info("Using Neo4j graph retrieval with %d LegalNode nodes", graph_node_count)
    graph_config = GraphRetrieverConfig(
        seed_top_k=config.top_k_before_rerank,
        chain_top_k=config.top_k_after_rerank,
        expansion_depth_fast=config.graph_expansion_depth_fast,
        expansion_depth_deep=config.graph_expansion_depth_deep,
        bm25_weight=config.graph_bm25_weight,
        dense_weight=config.graph_dense_weight,
        graph_weight=config.graph_weight,
        exact_citation_weight=config.graph_exact_citation_weight,
        legal_issue_weight=config.graph_legal_issue_weight,
        freshness_weight=config.graph_freshness_weight,
        hybrid_alpha=config.hybrid_alpha,
    )
    graph_reranker = LexicalOverlapReranker(
        use_gpu_reranker=config.use_gpu_reranker,
        reranker_model_name=config.reranker_model_name,
        device=config.reranker_device,
        batch_size=config.reranker_batch_size,
    )
    return LegalGraphRetriever(indexer, graph_store=graph_store, config=graph_config, reranker=graph_reranker)


def build_graph(
    config: PipelineConfig,
    *,
    import_to_neo4j: bool = False,
    clear_first: bool = False,
    batch_size: int | None = None,
) -> dict[str, Any]:
    logger.info("Building graph from %s", config.chunks_path)
    node_rows, edge_rows, stats = build_graph_records(config.chunks_path)
    write_graph_artifacts(config.graph_path, node_rows, edge_rows, stats)
    logger.info(
        "Built graph artifacts -> %s (nodes=%d, edges=%d)",
        config.graph_path,
        len(node_rows),
        len(edge_rows),
    )

    if import_to_neo4j:
        store = Neo4jGraphStore()
        effective_batch_size = batch_size or config.neo4j_import_batch_size
        import_stats = store.import_graph(
            node_rows,
            edge_rows,
            clear_first=clear_first,
            batch_size=effective_batch_size,
        )
        store.close()
        stats = {
            **stats,
            **import_stats,
            "neo4j_imported": True,
            "import_batch_size": effective_batch_size,
        }
        logger.info(
            "Imported graph into Neo4j (%d nodes, %d edges)",
            import_stats.get("nodes_upserted", 0),
            import_stats.get("edges_upserted", 0),
        )
    else:
        stats["neo4j_imported"] = False

    return stats


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
    law_routed = DocumentRouter().apply(case_query, law_candidates)
    law_reranked = reranker.rerank(case_query, law_routed, top_k=config.top_k_after_rerank)
    law_evidence = citation_filter.filter(case_query, law_reranked)

    # 1b. Tìm kiếm dữ liệu Vụ án (Gọi API Ban tổ chức - Tính vào c_i)
    calls_before_case = case_retrieval_client.call_count
    case_evidence_hits = case_retrieval_client.retrieve_multi(
        queries=[case_query][: config.max_case_retrieval_calls_per_case],
        case_id=case_id,
    )
    api_calls_for_case = case_retrieval_client.call_count - calls_before_case

    # 2. Thực hiện suy luận lập luận vụ án dựa trên mô hình ngôn ngữ
    answer = reasoning_agent.answer(case_id, case_query, law_evidence, case_evidence_hits)

    # 3. Thẩm định kết quả dự đoán
    verified = verifier.verify(answer, law_evidence)

    # 4. Đóng gói cấu trúc dữ liệu đầu ra
    law_evidence_out: list[dict[str, Any]] = []
    seen_law_evidence: set[tuple[str, str]] = set()
    for e in law_evidence:
        law_id = e["metadata"].get("law_id")
        aid = e["metadata"].get("aid")
        if law_id is None or aid is None:
            continue
        key = (str(law_id), str(aid))
        if key in seen_law_evidence:
            continue
        seen_law_evidence.add(key)
        law_evidence_out.append({"law_id": law_id, "aid": aid})
    case_evidence_out = [hit.chunk_id for hit in case_evidence_hits if hit.chunk_id]

    return {
        "case_id": case_id,
        "prediction": verified.get("label"),
        "confidence": verified.get("confidence"),
        "justification": verified.get("justification"),
        "parser_status": verified.get("parser_status"),
        "verifier_status": verified.get("verifier_status"),
        "law_evidence": law_evidence_out,
        "case_evidence": case_evidence_out,
        "graph_path": [],
        "api_calls": api_calls_for_case,
    }


def process_case_graph(
    case: dict[str, Any],
    *,
    indexer: HybridIndexer,
    graph_retriever: LegalGraphRetriever,
    reranker: Reranker,
    citation_filter: CitationUsefulnessFilter,
    case_retrieval_client: CaseRetrievalClient,
    reasoning_agent: ReasoningAgent,
    verifier: Verifier,
    config: PipelineConfig,
) -> dict[str, Any]:
    case_id = case["case_id"]
    case_query = case["case_query"]

    law_candidates = graph_retriever.retrieve(case_query)
    law_reranked = reranker.rerank(case_query, law_candidates, top_k=config.top_k_after_rerank)
    law_evidence = citation_filter.filter(case_query, law_reranked)

    calls_before_case = case_retrieval_client.call_count
    case_evidence_hits = case_retrieval_client.retrieve_multi(
        queries=[case_query][: config.max_case_retrieval_calls_per_case],
        case_id=case_id,
    )
    api_calls_for_case = case_retrieval_client.call_count - calls_before_case

    answer = reasoning_agent.answer(case_id, case_query, law_evidence, case_evidence_hits)
    verified = verifier.verify(answer, law_evidence)

    law_evidence_out: list[dict[str, Any]] = []
    graph_path_out: list[str] = []
    seen_law_evidence: set[tuple[str, str]] = set()
    for e in law_evidence:
        metadata = e.get("metadata") if isinstance(e.get("metadata"), dict) else {}
        law_id = metadata.get("law_id")
        aid = metadata.get("aid")
        if law_id is None or aid is None:
            continue
        key = (str(law_id), str(aid))
        if key in seen_law_evidence:
            continue
        seen_law_evidence.add(key)
        law_evidence_out.append({"law_id": law_id, "aid": aid})
        graph_path = e.get("graph_path") or metadata.get("graph_path") or []
        if isinstance(graph_path, list):
            for node_id in graph_path:
                node_text = str(node_id).strip()
                if node_text and node_text not in graph_path_out:
                    graph_path_out.append(node_text)

    case_evidence_out = [hit.chunk_id for hit in case_evidence_hits if hit.chunk_id]

    return {
        "case_id": case_id,
        "prediction": verified.get("label"),
        "confidence": verified.get("confidence"),
        "justification": verified.get("justification"),
        "parser_status": verified.get("parser_status"),
        "verifier_status": verified.get("verifier_status"),
        "law_evidence": law_evidence_out,
        "case_evidence": case_evidence_out,
        "graph_path": graph_path_out,
        "api_calls": api_calls_for_case,
    }


def run_pipeline(
    config: PipelineConfig,
    *,
    dry_run: bool = False,
    limit: int | None = None,
    force_rebuild_index: bool = False,
    case_id_filter: set[str] | None = None,
    resume: bool = False,
    require_graph: bool = False,
) -> Path:
    indexer = load_or_build_index(config, force_rebuild=force_rebuild_index)

    if dry_run:
        llm_client = DryRunLLMClient()
    else:
        # Nạp client tự động hoàn toàn bằng các tham số môi trường trong .env
        llm_client = LocalOllamaClient.from_env()
        
    reasoning_agent = PromptTemplateReasoningAgent(llm_client=llm_client, prompt_path=config.prompt_path)
    reranker = LexicalOverlapReranker(
        use_gpu_reranker=config.use_gpu_reranker,
        reranker_model_name=config.reranker_model_name,
        device=config.reranker_device,
        batch_size=config.reranker_batch_size,
    )
    citation_filter = HeuristicCitationUsefulnessFilter(max_results=config.top_k_after_rerank)
    verifier = StatutoryConsistencyVerifier()
    graph_retriever = build_graph_retriever(
        config,
        indexer,
        dry_run=dry_run,
        require_graph=require_graph,
    )
    process_case_impl = (
        partial(process_case_graph, graph_retriever=graph_retriever)
        if graph_retriever is not None
        else process_case
    )

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
    predictions = _load_existing_predictions(final_output_path, debug_log_path) if resume else []
    completed_case_ids = {str(item.get("case_id")) for item in predictions if item.get("case_id")}
    if completed_case_ids:
        logger.info("Resume enabled: skipping %d completed cases.", len(completed_case_ids))

    debug_mode = "a" if resume and debug_log_path.exists() else "w"
    with debug_log_path.open(debug_mode, encoding="utf-8") as debug_handle:
        matched_so_far = 0
        for i, case in enumerate(load_test_cases(config.public_test_path)):
            if case_id_filter is not None:
                if case["case_id"] not in case_id_filter:
                    continue
                matched_so_far += 1
            elif limit is not None and i >= limit:
                break
            if str(case["case_id"]) in completed_case_ids:
                continue

            try:
                pred = process_case_impl(
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
            debug_handle.flush()
            predictions.append(pred)
            completed_case_ids.add(str(pred["case_id"]))
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


def _load_existing_predictions(final_output_path: Path, debug_log_path: Path) -> list[dict[str, Any]]:
    source_path = final_output_path if final_output_path.exists() else debug_log_path
    if not source_path.exists():
        return []

    loaded: list[dict[str, Any]] = []
    if source_path.suffix == ".json":
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            loaded = [item for item in payload if isinstance(item, dict)]
    else:
        for line in source_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                loaded.append(item)

    deduped: dict[str, dict[str, Any]] = {}
    for item in loaded:
        case_id = item.get("case_id")
        if case_id:
            deduped[str(case_id)] = item
    return list(deduped.values())


_ALQAC_VALID_LABELS = {"A_WIN", "PARTIAL_A_WIN", "PARTIAL_B_WIN", "B_WIN"}
_PLACEHOLDER_MARKERS = ("replace_", "<", ">", "your_", "changeme", "todo")


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


def _clean_case_evidence(value: Any) -> list[str]:
    if not isinstance(value, list):
        value = [value] if value else []

    cleaned: list[str] = []
    seen: set[str] = set()
    for item in value:
        evidence_id = str(item).strip()
        if not evidence_id or evidence_id in seen:
            continue
        cleaned.append(evidence_id)
        seen.add(evidence_id)
    return cleaned


def _clean_law_evidence(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    cleaned: list[dict[str, Any]] = []
    seen: set[tuple[str, int | str]] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        law_id = str(item.get("law_id") or "").strip()
        if not law_id or "aid" not in item:
            continue
        raw_aid = item["aid"]
        try:
            aid: int | str = int(raw_aid)
        except (TypeError, ValueError):
            aid = str(raw_aid).strip()
        if aid == "":
            continue
        key = (law_id, aid)
        if key in seen:
            continue
        cleaned.append({"law_id": law_id, "aid": aid})
        seen.add(key)
    return cleaned


def _case_order_from_public_test(public_test_path: Path) -> list[str]:
    if not public_test_path.exists():
        return []
    payload = json.loads(public_test_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        return []
    return [str(item.get("case_id")) for item in payload if isinstance(item, dict) and item.get("case_id")]


def build_submission_json(
    internal_pred_path: Path,
    output_path: Path,
    public_test_path: Path | None = None,
) -> Path:
    internal = json.loads(internal_pred_path.read_text(encoding="utf-8"))
    if not isinstance(internal, list):
        raise ValueError(f"{internal_pred_path} phải là JSON array.")

    seen_case_ids: set[str] = set()
    by_case_id: dict[str, dict[str, Any]] = {}
    for item in internal:
        if not isinstance(item, dict):
            raise ValueError("Mỗi prediction phải là JSON object.")
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

        by_case_id[case_id] = {
            "case_id": case_id,
            "prediction": prediction,
            "case_evidence": _clean_case_evidence(item.get("case_evidence", [])),
            "law_evidence": _clean_law_evidence(item.get("law_evidence", [])),
        }

    ordered_case_ids = _case_order_from_public_test(public_test_path) if public_test_path else []
    if ordered_case_ids:
        missing = [case_id for case_id in ordered_case_ids if case_id not in by_case_id]
        if missing:
            raise ValueError(
                f"Thiếu {len(missing)} case_id so với public test. Ví dụ: {missing[:5]}"
            )
        submission = [by_case_id[case_id] for case_id in ordered_case_ids]
    else:
        submission = list(by_case_id.values())

    output_path.write_text(json.dumps(submission, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Đã build %d case -> %s (đúng schema chính thức)", len(submission), output_path)
    return output_path


def _is_real_env_value(value: str | None) -> bool:
    if value is None:
        return False
    text = value.strip()
    if not text:
        return False
    lowered = text.lower()
    return not any(marker in lowered for marker in _PLACEHOLDER_MARKERS)


def validate_prediction_output(pred_path: Path, *, require_real_reasoning: bool = False) -> dict[str, Any]:
    """Validate internal prediction JSON before evaluation/submission export."""
    if not pred_path.exists():
        raise FileNotFoundError(f"Không tìm thấy prediction file: {pred_path}")
    payload = json.loads(pred_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{pred_path} phải là JSON array.")

    seen_case_ids: set[str] = set()
    invalid_labels: list[str] = []
    duplicate_case_ids: list[str] = []
    dry_run_cases: list[str] = []
    invalid_json_cases: list[str] = []
    missing_law_evidence: list[str] = []
    missing_case_evidence: list[str] = []

    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Mọi phần tử prediction phải là JSON object.")
        case_id = str(item.get("case_id") or "")
        if not case_id:
            raise ValueError("Có prediction thiếu case_id.")
        if case_id in seen_case_ids:
            duplicate_case_ids.append(case_id)
        seen_case_ids.add(case_id)

        if item.get("prediction") not in _ALQAC_VALID_LABELS:
            invalid_labels.append(case_id)
        if item.get("parser_status") == "invalid_json":
            invalid_json_cases.append(case_id)
        if "[dry-run]" in str(item.get("justification", "")):
            dry_run_cases.append(case_id)
        if not item.get("law_evidence"):
            missing_law_evidence.append(case_id)
        if not item.get("case_evidence"):
            missing_case_evidence.append(case_id)

    report = {
        "path": str(pred_path),
        "case_count": len(payload),
        "valid_label_count": len(payload) - len(invalid_labels),
        "invalid_label_cases": invalid_labels,
        "duplicate_case_ids": duplicate_case_ids,
        "invalid_json_cases": invalid_json_cases,
        "dry_run_cases": dry_run_cases,
        "missing_law_evidence_cases": missing_law_evidence,
        "missing_case_evidence_cases": missing_case_evidence,
        "ready_for_submission_shape": not (
            invalid_labels or duplicate_case_ids or invalid_json_cases
        ),
    }
    report["real_reasoning_ready"] = report["ready_for_submission_shape"] and not dry_run_cases

    if require_real_reasoning and not report["real_reasoning_ready"]:
        raise ValueError(
            "Prediction chưa đạt điều kiện real reasoning. Xem report để biết case lỗi."
        )
    return report


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


def check_runtime(
    config: PipelineConfig,
    *,
    ping_llm: bool = False,
    ping_case_api: bool = False,
    ping_neo4j: bool = False,
) -> dict[str, Any]:
    """Check whether the environment is ready for non-dry-run inference."""
    llm_provider = os.environ.get("ALQAC_LLM_PROVIDER", "ollama")
    llm_base_url = os.environ.get("ALQAC_LLM_BASE_URL")
    llm_model = os.environ.get("ALQAC_LLM_MODEL_NAME")
    team_token = os.environ.get("ALQAC_TEAM_TOKEN")
    case_api_url = os.environ.get("ALQAC_API_URL") or os.environ.get("ALQAC_RETRIEVAL_API_BASE_URL")
    neo4j_uri = os.environ.get(config.neo4j_uri_env)
    neo4j_username = os.environ.get(config.neo4j_username_env)
    neo4j_password = os.environ.get(config.neo4j_password_env)
    neo4j_database = os.environ.get(config.neo4j_database_env)
    report: dict[str, Any] = {
        "paths": {
            "chunks_path": config.chunks_path.exists(),
            "public_test_path": config.public_test_path.exists(),
            "prompt_path": config.prompt_path.exists(),
            "index_path": Path(config.index_path).exists(),
            "graph_path": Path(config.graph_path).exists(),
        },
        "env": {
            "ALQAC_LLM_PROVIDER": _is_real_env_value(llm_provider),
            "ALQAC_LLM_BASE_URL": _is_real_env_value(llm_base_url),
            "ALQAC_LLM_MODEL_NAME": _is_real_env_value(llm_model),
            "NEO4J_URI": _is_real_env_value(neo4j_uri),
            "NEO4J_USERNAME": _is_real_env_value(neo4j_username),
            "NEO4J_PASSWORD": _is_real_env_value(neo4j_password),
            # Aura can select the instance's default database when this is blank.
            "NEO4J_DATABASE": not neo4j_database or _is_real_env_value(neo4j_database),
            "ALQAC_TEAM_TOKEN": _is_real_env_value(team_token),
            "ALQAC_API_URL_or_RETRIEVAL_BASE": _is_real_env_value(case_api_url),
        },
        "llm_ping": "skipped",
        "case_api_ping": "skipped",
        "neo4j_ping": "skipped",
    }

    if ping_llm:
        try:
            client = LocalOllamaClient.from_env()
            raw = client.generate(
                (
                    "/no_think\n"
                    "Return JSON only: label=B_WIN, confidence=0.2, "
                    "evidence_ids=[], justification=ping."
                ),
                max_tokens=512,
                temperature=0.0,
            )
            parsed = PromptTemplateReasoningAgent._parse_json_output("runtime_ping", raw)
            report["llm_ping"] = {
                "ok": parsed.get("parser_status") == "ok",
                "label": parsed.get("label"),
                "confidence": parsed.get("confidence"),
            }
        except Exception as exc:
            report["llm_ping"] = {"ok": False, "error": str(exc)}

    if ping_case_api:
        try:
            rate_limiter = RateLimiter(0.0)
            client = CaseRetrievalClient.from_env(rate_limiter=rate_limiter)
            first_case = next(load_test_cases(config.public_test_path))
            hit = client.retrieve(first_case["case_query"], first_case["case_id"])
            report["case_api_ping"] = {
                "ok": bool(hit.chunk_id),
                "case_id": first_case["case_id"],
                "chunk_id": hit.chunk_id,
                "score": hit.score,
            }
        except Exception as exc:
            report["case_api_ping"] = {"ok": False, "error": str(exc)}

    if ping_neo4j:
        store: Neo4jGraphStore | None = None
        try:
            store = Neo4jGraphStore()
            store.verify_connectivity()
            report["neo4j_ping"] = {
                "ok": True,
                "backend": "aura" if store.config.is_aura else "self_managed",
                "database": store.config.database or "default",
                "legal_node_count": store.count_legal_nodes(),
            }
        except Exception as exc:
            report["neo4j_ping"] = {"ok": False, "error": str(exc)}
        finally:
            if store is not None:
                store.close()

    report["ready_for_real_run"] = all(report["paths"].values()) and all(report["env"].values())
    report["ready_for_graph_benchmark"] = (
        all(report["paths"][key] for key in ("chunks_path", "public_test_path", "index_path", "graph_path"))
        and all(report["env"][key] for key in ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "NEO4J_DATABASE"))
    )
    if ping_llm:
        report["ready_for_real_run"] = report["ready_for_real_run"] and bool(
            isinstance(report["llm_ping"], dict) and report["llm_ping"].get("ok")
        )
    if ping_case_api:
        report["ready_for_real_run"] = report["ready_for_real_run"] and bool(
            isinstance(report["case_api_ping"], dict) and report["case_api_ping"].get("ok")
        )
    if ping_neo4j:
        report["ready_for_graph_benchmark"] = report["ready_for_graph_benchmark"] and bool(
            isinstance(report["neo4j_ping"], dict) and report["neo4j_ping"].get("ok")
        )
    return report


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="ALQAC 2026 pipeline orchestrator")
    parser.add_argument("--config", type=Path, default=None, help="JSON config override file")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("build-index", help="Build/rebuild the hybrid BM25+dense index")
    build_graph_parser = subparsers.add_parser("build-graph", help="Build graph artifacts and optionally import Neo4j")
    build_graph_parser.add_argument("--import-neo4j", action="store_true", help="Import the graph into Neo4j")
    build_graph_parser.add_argument("--clear-graph", action="store_true", help="Clear existing LegalNode graph before import")
    build_graph_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Neo4j import batch size. Default: config value (200, suitable for Aura Free).",
    )

    run_parser = subparsers.add_parser("run", help="Run the full pipeline over the public test set")
    run_parser.add_argument("--dry-run", action="store_true", help="Use stub LLM, no real inference")
    run_parser.add_argument("--limit", type=int, default=None, help="Only process the first N cases")
    run_parser.add_argument("--rebuild-index", action="store_true")
    run_parser.add_argument("--resume", action="store_true", help="Skip case_ids already present in run output/debug log")
    run_parser.add_argument("--require-graph", action="store_true", help="Fail instead of falling back when Neo4j graph retrieval is unavailable")
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

    check_parser = subparsers.add_parser("check-runtime", help="Check env/path readiness for real inference")
    check_parser.add_argument("--ping-llm", action="store_true", help="Send a tiny JSON prompt to the configured LLM")
    check_parser.add_argument("--ping-case-api", action="store_true", help="Call the official case retrieval API once")
    check_parser.add_argument("--ping-neo4j", action="store_true", help="Verify Neo4j/Aura connectivity and count LegalNode records")
    check_parser.add_argument(
        "--graph-only",
        action="store_true",
        help="Validate only Neo4j graph-benchmark prerequisites; do not require the LLM or case API.",
    )

    validate_parser = subparsers.add_parser("validate-output", help="Validate internal prediction JSON")
    validate_parser.add_argument("--pred", type=Path, default=None)
    validate_parser.add_argument("--require-real", action="store_true")

    args = parser.parse_args()
    config = PipelineConfig.from_file(args.config) if args.config else PipelineConfig()

    if args.command == "build-index":
        build_index(config)
    elif args.command == "build-graph":
        try:
            build_graph(
                config,
                import_to_neo4j=args.import_neo4j,
                clear_first=args.clear_graph,
                batch_size=args.batch_size,
            )
        except Exception as exc:
            logger.error(str(exc))
            raise SystemExit(1) from exc
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
            resume=args.resume,
            require_graph=args.require_graph,
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
            build_submission_json(pred_path, args.output, public_test_path=config.public_test_path)
        except ValueError as exc:
            logger.error(str(exc))
            raise SystemExit(1) from exc
    elif args.command == "submit":
        try:
            submit(config, args.score, args.notes)
        except SubmissionGuardrailError as exc:
            logger.error(str(exc))
            raise SystemExit(1) from exc
    elif args.command == "check-runtime":
        report = check_runtime(
            config,
            ping_llm=args.ping_llm,
            ping_case_api=args.ping_case_api,
            ping_neo4j=args.ping_neo4j,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        ready_key = "ready_for_graph_benchmark" if args.graph_only else "ready_for_real_run"
        if not report[ready_key]:
            raise SystemExit(1)
    elif args.command == "validate-output":
        pred_path = args.pred or (config.experiments_dir / f"run_{config.run_tag}.json")
        try:
            report = validate_prediction_output(pred_path, require_real_reasoning=args.require_real)
        except (FileNotFoundError, ValueError) as exc:
            logger.error(str(exc))
            raise SystemExit(1) from exc
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if args.require_real and not report["real_reasoning_ready"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
