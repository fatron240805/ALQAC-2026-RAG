from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from orchestration.config import PipelineConfig
from orchestration.data_adapters import load_test_cases
from orchestration.run_pipeline import build_graph_retriever, load_or_build_index
from retrieval.citation_usefulness import HeuristicCitationUsefulnessFilter
from retrieval.reranker import LexicalOverlapReranker
from retrieval.router import DocumentRouter


DEFAULT_PUBLIC_TEST_PATH = PROJECT_ROOT / "data" / "ALQAC2026_public_test.json"
DEFAULT_CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks.jsonl"
DEFAULT_INDEX_PATH = PROJECT_ROOT / "data" / "index"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "experiments" / "retrieval_graph_benchmark_report.json"
DEFAULT_CASES_PATH = PROJECT_ROOT / "experiments" / "retrieval_graph_benchmark_cases.csv"
DEFAULT_TOP_K = (1, 3, 5, 8, 10, 20)


LAW_CODE_RE = re.compile(r"^\d+/\d{4}/[A-Z0-9-]+$", re.IGNORECASE)
ARTICLE_RE = re.compile(r"\bdieu\s+([0-9]+[a-z]?)\b", re.IGNORECASE)


@dataclass(frozen=True)
class GoldProvision:
    case_id: str
    law_title: str
    article_number: str
    law_code: str | None
    source: str

    @property
    def key(self) -> str | None:
        if not self.law_code or not self.article_number:
            return None
        return provision_key(self.law_code, self.article_number)


def strip_accents(value: Any) -> str:
    text = unicodedata.normalize("NFD", str(value or ""))
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("Đ", "D").replace("đ", "d")


def normalize_text(value: Any) -> str:
    text = strip_accents(value).lower()
    text = re.sub(r"[^0-9a-z/]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_article_number(value: Any) -> str | None:
    numbers = extract_article_numbers(value)
    return numbers[0] if numbers else None


def extract_article_numbers(value: Any) -> list[str]:
    """Extract every article number from one public-test provision reference."""
    if value in (None, ""):
        return []
    text = normalize_text(value)
    numbers = list(dict.fromkeys(match.lower() for match in ARTICLE_RE.findall(text)))
    if numbers:
        return numbers
    # `aid` and `article_number` can be bare numeric values in corpus metadata.
    return [text] if re.fullmatch(r"[0-9]+[a-z]?", text) else []


def provision_key(law_code: str, article_number: Any) -> str:
    return f"{str(law_code).strip()}:{str(article_number).strip().lower()}"


def canonical_law_code(value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    if LAW_CODE_RE.match(raw):
        return raw

    text = normalize_text(raw)
    if "91/2015/qh13" in text or ("bo luat dan su" in text and "2005" not in text and "1995" not in text):
        return "91/2015/QH13"
    if "92/2015/qh13" in text or "to tung dan su" in text:
        return "92/2015/QH13"
    if "45/2013/qh13" in text or ("dat dai" in text and "2003" not in text and "1987" not in text):
        return "45/2013/QH13"
    if "26/2008/qh12" in text or "thi hanh an dan su" in text:
        return "26/2008/QH12"
    if "326/2016" in text or ("an phi" in text and "nghi quyet" in text):
        return "326/2016/UBTVQH14"
    if "47/2010/qh12" in text or "cac to chuc tin dung" in text:
        return "47/2010/QH12"
    if "52/2014/qh13" in text or "hon nhan" in text:
        return "52/2014/QH13"
    if "50/2014/qh13" in text or ("xay dung" in text and "hop dong" not in text):
        return "50/2014/QH13"
    if "37/2015/nd cp" in text or "hop dong xay dung" in text:
        return "37/2015/NĐ-CP"
    if "66/2014/qh13" in text or "kinh doanh bat dong san" in text:
        return "66/2014/QH13"
    if "60/2014/qh13" in text or "ho tich" in text:
        return "60/2014/QH13"
    if "52/2010/qh12" in text or "nuoi con nuoi" in text:
        return "52/2010/QH12"
    if "100/2015/qh13" in text or "hinh su" in text:
        return "100/2015/QH13"
    if "93/2015/qh13" in text or "to tung hanh chinh" in text:
        return "93/2015/QH13"
    if "02/2011/qh13" in text or "khieu nai" in text:
        return "02/2011/QH13"
    if "39/2009/qh12" in text or "nguoi cao tuoi" in text:
        return "39/2009/QH12"
    if "24/2012/nd cp" in text or "kinh doanh vang" in text:
        return "24/2012/NĐ-CP"
    if "19/2011/nd cp" in text or "ho tro nhan dao" in text:
        return "19/2011/NĐ-CP"
    return None


class ProvisionNormalizer:
    def __init__(self, chunks_path: Path = DEFAULT_CHUNKS_PATH) -> None:
        self.chunk_to_key: dict[str, str] = {}
        self.raw_aid_to_article: dict[tuple[str, str], str] = {}
        self._load_chunks(chunks_path)

    def _load_chunks(self, chunks_path: Path) -> None:
        if not chunks_path.exists():
            return
        with chunks_path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                law_code = canonical_law_code(chunk.get("law_id"))
                if not law_code:
                    continue
                article_number = self._article_number_from_chunk(chunk)
                if not article_number:
                    continue
                key = provision_key(law_code, article_number)
                chunk_id = str(chunk.get("chunk_id") or "").strip()
                if chunk_id:
                    self.chunk_to_key[chunk_id] = key
                raw_aid = str(chunk.get("aid") or "").strip()
                if raw_aid:
                    self.raw_aid_to_article[(law_code, raw_aid)] = article_number

    @staticmethod
    def _article_number_from_chunk(chunk: dict[str, Any]) -> str | None:
        # In the supplied law corpus, `aid` / `article_number` are global
        # source-record IDs (for example 53354), while `article_index` is the
        # 1-based legal article ordinal (for example Article 584).
        article_index = chunk.get("article_index")
        try:
            return str(int(article_index))
        except (TypeError, ValueError):
            pass
        for field in ("article_number", "article_label", "unit_path", "aid"):
            article_number = normalize_article_number(chunk.get(field))
            if article_number:
                return article_number
        return None

    def candidate_key(self, candidate: dict[str, Any]) -> str | None:
        metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
        for field in ("chunk_id", "source_chunk_id", "seed_chunk_id"):
            chunk_id = str(metadata.get(field) or candidate.get(field) or "").strip()
            if chunk_id and chunk_id in self.chunk_to_key:
                return self.chunk_to_key[chunk_id]

        law_code = canonical_law_code(metadata.get("law_id") or candidate.get("law_id"))
        if not law_code:
            return None

        raw_aid = str(metadata.get("aid") or candidate.get("aid") or "").strip()
        article_number = None
        if raw_aid:
            article_number = self.raw_aid_to_article.get((law_code, raw_aid))
        if not article_number:
            article_number = self._article_number_from_chunk(metadata)
        if not article_number:
            return None
        return provision_key(law_code, article_number)

    def gold_key(self, item: GoldProvision) -> str | None:
        return item.key


def load_public_test_cases(path: Path) -> dict[str, str]:
    return {case["case_id"]: case["case_query"] for case in load_test_cases(path)}


def load_public_test_gold(path: Path) -> dict[str, list[GoldProvision]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    items = payload if isinstance(payload, list) else payload.get("cases", payload.get("data", []))
    gold: dict[str, list[GoldProvision]] = {}
    for case in items:
        case_id = str(case.get("case_id") or case.get("id") or "").strip()
        if not case_id:
            continue
        rows: list[GoldProvision] = []
        related = str(case.get("related_law_provisions") or "")
        for line in related.splitlines():
            if "|" not in line:
                continue
            law_title, provision_ref = [part.strip() for part in line.split("|", 1)]
            law_code = canonical_law_code(law_title)
            for article_number in extract_article_numbers(provision_ref):
                rows.append(
                    GoldProvision(
                        case_id=case_id,
                        law_title=law_title,
                        article_number=article_number,
                        law_code=law_code,
                        source="public_test.related_law_provisions",
                    )
                )
        gold[case_id] = dedupe_gold(rows)
    return gold


def load_json_gold(path: Path) -> dict[str, list[GoldProvision]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    items = payload if isinstance(payload, list) else payload.get("cases", payload.get("data", []))
    gold: dict[str, list[GoldProvision]] = {}
    for case in items:
        case_id = str(case.get("case_id") or "").strip()
        rows: list[GoldProvision] = []
        for evidence in case.get("law_evidence", []) or []:
            if not isinstance(evidence, dict):
                continue
            law_title = str(evidence.get("law_id") or "")
            law_code = canonical_law_code(law_title)
            for article_number in extract_article_numbers(evidence.get("aid")):
                rows.append(
                    GoldProvision(
                        case_id=case_id,
                        law_title=law_title,
                        article_number=article_number,
                        law_code=law_code,
                        source=str(path),
                    )
                )
        gold[case_id] = dedupe_gold(rows)
    return gold


def load_csv_gold(path: Path) -> dict[str, list[GoldProvision]]:
    gold: dict[str, list[GoldProvision]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            case_id = str(row.get("case_id") or "").strip()
            law_title = str(row.get("law_title") or row.get("law_id") or "").strip()
            article_numbers = extract_article_numbers(row.get("aid") or row.get("provision_ref"))
            if not case_id or not law_title or not article_numbers:
                continue
            law_code = canonical_law_code(law_title)
            for article_number in article_numbers:
                gold.setdefault(case_id, []).append(
                    GoldProvision(
                        case_id=case_id,
                        law_title=law_title,
                        article_number=article_number,
                        law_code=law_code,
                        source=str(path),
                    )
                )
    return {case_id: dedupe_gold(rows) for case_id, rows in gold.items()}


def load_gold(path: Path | None, public_test_path: Path) -> tuple[dict[str, list[GoldProvision]], str]:
    if path is None:
        return load_public_test_gold(public_test_path), str(public_test_path)
    if path.suffix.lower() == ".csv":
        return load_csv_gold(path), str(path)
    return load_json_gold(path), str(path)


def dedupe_gold(items: Iterable[GoldProvision]) -> list[GoldProvision]:
    seen: set[tuple[str | None, str, str]] = set()
    deduped: list[GoldProvision] = []
    for item in items:
        identity = (item.law_code, item.law_title, item.article_number)
        if identity in seen:
            continue
        seen.add(identity)
        deduped.append(item)
    return deduped


def unique_ranked_keys(candidates: list[dict[str, Any]], normalizer: ProvisionNormalizer) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = normalizer.candidate_key(candidate)
        if not key or key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return keys


def is_graph_candidate(candidate: dict[str, Any]) -> bool:
    """True only for a candidate emitted by Neo4j traversal, not a seed fallback."""
    metadata = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
    graph_path = candidate.get("graph_path") or metadata.get("graph_path")
    return isinstance(graph_path, list) and len(graph_path) >= 2


def per_case_metrics(gold_keys: set[str], predicted_keys: list[str], k: int) -> dict[str, float]:
    top = predicted_keys[:k]
    top_set = set(top)
    hits = gold_keys & top_set
    precision = len(hits) / len(top) if top else 0.0
    recall = len(hits) / len(gold_keys) if gold_keys else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    first_hit_rank = 0
    for index, key in enumerate(top, start=1):
        if key in gold_keys:
            first_hit_rank = index
            break
    return {
        "hit": 1.0 if hits else 0.0,
        "full_recall": 1.0 if gold_keys and gold_keys <= top_set else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mrr": 1.0 / first_hit_rank if first_hit_rank else 0.0,
    }


def summarize_metrics(case_rows: list[dict[str, Any]], top_ks: list[int]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for k in top_ks:
        suffix = f"@{k}"
        rows = [row for row in case_rows if row["mapped_gold_count"] > 0]
        denom = max(1, len(rows))
        total_gold = sum(row["mapped_gold_count"] for row in rows)
        total_pred = sum(min(k, len(row["predicted_keys"])) for row in rows)
        total_correct = 0
        for row in rows:
            total_correct += len(set(row["predicted_keys"][:k]) & set(row["gold_keys"]))
        micro_precision = total_correct / total_pred if total_pred else 0.0
        micro_recall = total_correct / total_gold if total_gold else 0.0
        micro_f1 = (
            2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            if micro_precision + micro_recall
            else 0.0
        )
        summary[suffix] = {
            "Case_Hit_Accuracy": round(sum(row[f"hit{suffix}"] for row in rows) / denom, 4),
            "Exact_Set_Accuracy": round(sum(row[f"full_recall{suffix}"] for row in rows) / denom, 4),
            "Macro_Precision": round(sum(row[f"precision{suffix}"] for row in rows) / denom, 4),
            "Macro_Recall": round(sum(row[f"recall{suffix}"] for row in rows) / denom, 4),
            "Macro_F1": round(sum(row[f"f1{suffix}"] for row in rows) / denom, 4),
            "Micro_Precision": round(micro_precision, 4),
            "Micro_Recall": round(micro_recall, 4),
            "Micro_F1": round(micro_f1, 4),
            "MRR": round(sum(row[f"mrr{suffix}"] for row in rows) / denom, 4),
        }
    return summary


def build_retriever(
    args: argparse.Namespace,
    config: PipelineConfig,
) -> tuple[Callable[[str], list[dict[str, Any]]], dict[str, Any], Callable[[], None]]:
    indexer = load_or_build_index(config, force_rebuild=args.rebuild_index)
    if args.retriever == "graph":
        retriever = build_graph_retriever(config, indexer, require_graph=True)
        if retriever is None:
            raise RuntimeError("Graph retriever was requested but is unavailable.")
        graph_store = retriever.graph_store
        if graph_store is None:
            raise RuntimeError("Graph retriever was requested without a Neo4j graph store.")
        return (
            retriever.retrieve,
            {
                "backend": "neo4j",
                "legal_node_count": graph_store.count_legal_nodes(),
                "expansion_depth_fast": retriever.config.expansion_depth_fast,
                "expansion_depth_deep": retriever.config.expansion_depth_deep,
            },
            graph_store.close,
        )

    router = DocumentRouter()
    reranker = LexicalOverlapReranker()
    citation_filter = HeuristicCitationUsefulnessFilter(max_results=max(args.top_k))

    def retrieve(query: str) -> list[dict[str, Any]]:
        candidates = indexer.search(query, top_k=max(args.seed_top_k, max(args.top_k)), alpha=args.alpha)
        routed = router.apply(query, candidates)
        reranked = reranker.rerank(query, routed, top_k=max(args.seed_top_k, max(args.top_k)))
        return citation_filter.filter(query, reranked)

    return retrieve, {"backend": "flat_hybrid"}, lambda: None


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    top_ks = sorted({int(k) for k in args.top_k if int(k) > 0})
    public_queries = load_public_test_cases(args.public_test)
    gold_by_case, gold_source = load_gold(args.gold, args.public_test)
    normalizer = ProvisionNormalizer(args.chunks)

    config = PipelineConfig(
        chunks_path=args.chunks,
        index_path=args.index,
        public_test_path=args.public_test,
        top_k_before_rerank=max(args.seed_top_k, max(top_ks)),
        top_k_after_rerank=max(top_ks),
        hybrid_alpha=args.alpha,
    )
    retrieve, retriever_metadata, close_retriever = build_retriever(args, config)

    case_rows: list[dict[str, Any]] = []
    raw_gold_items = 0
    mapped_gold_items = 0
    unmapped_gold_items = 0
    case_ids = [case_id for case_id in public_queries if case_id in gold_by_case]
    if args.limit is not None:
        case_ids = case_ids[: args.limit]

    try:
        for case_id in case_ids:
            query = public_queries[case_id]
            gold_items = gold_by_case.get(case_id, [])
            raw_gold_items += len(gold_items)
            gold_keys = {item.key for item in gold_items if item.key}
            mapped_gold_items += len(gold_keys)
            unmapped_gold_items += max(0, len(gold_items) - len(gold_keys))
            candidates = retrieve(query)
            graph_candidates = [candidate for candidate in candidates if is_graph_candidate(candidate)]
            score_candidates = (
                candidates
                if args.retriever != "graph" or args.include_flat_fallback
                else graph_candidates
            )
            predicted_keys = unique_ranked_keys(score_candidates, normalizer)

            row: dict[str, Any] = {
                "case_id": case_id,
                "gold_count": len(gold_items),
                "mapped_gold_count": len(gold_keys),
                "unmapped_gold_count": max(0, len(gold_items) - len(gold_keys)),
                "returned_candidate_count": len(candidates),
                "graph_candidate_count": len(graph_candidates),
                "flat_fallback_candidate_count": len(candidates) - len(graph_candidates),
                "graph_expansion_used": bool(graph_candidates),
                "retrieved_count": len(predicted_keys),
                "gold_keys": sorted(gold_keys),
                "predicted_keys": predicted_keys,
                "matched_keys": sorted(gold_keys & set(predicted_keys)),
                "missing_keys": sorted(gold_keys - set(predicted_keys)),
                "unmapped_gold_provisions": [
                    f"{item.law_title} | Article {item.article_number}"
                    for item in gold_items
                    if item.key is None
                ],
            }
            for k in top_ks:
                metrics = per_case_metrics(gold_keys, predicted_keys, k)
                for name, value in metrics.items():
                    row[f"{name}@{k}"] = value
            case_rows.append(row)
    finally:
        close_retriever()

    report = {
        "Retrieval_Benchmark": {
            "retriever": args.retriever,
            "score_scope": (
                "neo4j_traversal_only"
                if args.retriever == "graph" and not args.include_flat_fallback
                else "full_retrieval_pipeline"
            ),
            "gold_source": gold_source,
            "cases_evaluated": len(case_rows),
            "top_k": top_ks,
            "seed_top_k": args.seed_top_k,
            "alpha": args.alpha,
            **retriever_metadata,
        },
        "Graph_Execution": {
            "cases_with_neo4j_expansion": sum(1 for row in case_rows if row["graph_expansion_used"]),
            "cases_without_neo4j_expansion": sum(1 for row in case_rows if not row["graph_expansion_used"]),
            "returned_graph_candidates": sum(row["graph_candidate_count"] for row in case_rows),
            "returned_flat_fallback_candidates": sum(row["flat_fallback_candidate_count"] for row in case_rows),
        },
        "Gold_Coverage": {
            "raw_gold_items": raw_gold_items,
            "mapped_gold_items": mapped_gold_items,
            "unmapped_gold_items": unmapped_gold_items,
            "mapping_rate": round(mapped_gold_items / raw_gold_items, 4) if raw_gold_items else 0.0,
            "cases_with_mapped_gold": sum(1 for row in case_rows if row["mapped_gold_count"] > 0),
        },
        "Metrics": summarize_metrics(case_rows, top_ks),
        "Case_Rows": case_rows,
    }
    return report


def write_case_csv(path: Path, case_rows: list[dict[str, Any]], top_ks: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_fields: list[str] = []
    for k in top_ks:
        metric_fields.extend([f"hit@{k}", f"full_recall@{k}", f"precision@{k}", f"recall@{k}", f"f1@{k}", f"mrr@{k}"])
    fieldnames = [
        "case_id",
        "gold_count",
        "mapped_gold_count",
        "unmapped_gold_count",
        "returned_candidate_count",
        "graph_candidate_count",
        "flat_fallback_candidate_count",
        "graph_expansion_used",
        "retrieved_count",
        *metric_fields,
        "gold_keys",
        "predicted_keys",
        "matched_keys",
        "missing_keys",
        "unmapped_gold_provisions",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in case_rows:
            output = {field: row.get(field, "") for field in fieldnames}
            for field in ("gold_keys", "predicted_keys", "matched_keys", "missing_keys", "unmapped_gold_provisions"):
                output[field] = " ".join(row.get(field, []))
            writer.writerow(output)


def parse_top_k(value: str) -> list[int]:
    top_ks: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        top_ks.append(int(part))
    if not top_ks:
        raise argparse.ArgumentTypeError("top-k must contain at least one positive integer")
    if any(k <= 0 for k in top_ks):
        raise argparse.ArgumentTypeError("top-k values must be positive")
    return top_ks


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Neo4j legal-graph retrieval against public-test law labels")
    parser.add_argument("--public-test", type=Path, default=DEFAULT_PUBLIC_TEST_PATH)
    parser.add_argument("--gold", type=Path, default=None, help="Optional JSON/CSV gold. Default: public_test.related_law_provisions")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--retriever", choices=("graph", "hybrid"), default="graph")
    parser.add_argument("--top-k", type=parse_top_k, default=list(DEFAULT_TOP_K), help="Comma-separated k values, e.g. 1,3,5,8,10,20")
    parser.add_argument("--seed-top-k", type=int, default=80)
    parser.add_argument("--alpha", type=float, default=0.50)
    parser.add_argument(
        "--include-flat-fallback",
        action="store_true",
        help="For graph retrieval, score seed-only fallback candidates too. Default scores Neo4j traversal candidates only.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--cases-output", type=Path, default=DEFAULT_CASES_PATH)
    args = parser.parse_args()

    report = run_benchmark(args)
    report_to_write = dict(report)
    case_rows = report_to_write.pop("Case_Rows")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report_to_write, ensure_ascii=False, indent=2), encoding="utf-8")
    write_case_csv(args.cases_output, case_rows, report["Retrieval_Benchmark"]["top_k"])

    print(json.dumps(report_to_write, ensure_ascii=False, indent=2))
    print(f"case diagnostics: {args.cases_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
