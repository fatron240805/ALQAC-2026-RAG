from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from legal_features import extract_legal_features
from retrieval.deprecated_filter import DeprecatedFilter


ARTICLE_REF_RE = re.compile(
    r"(?:điều|dieu|Äiá»u|Ä‘iá»u)\s+([0-9]+[a-zA-Z]?)",
    re.IGNORECASE,
)


CONCEPT_ALIASES: dict[str, list[str]] = {
    "ontology:issue:tort_damage": [
        "bồi thường thiệt hại",
        "thiệt hại ngoài hợp đồng",
        "trách nhiệm bồi thường",
        "xâm phạm tài sản",
        "xâm phạm sức khỏe",
    ],
    "ontology:liability:animal_damage": [
        "súc vật",
        "vật nuôi",
        "chó",
        "trâu",
        "bò",
        "gây thiệt hại",
    ],
    "ontology:issue:contract_dispute": [
        "hợp đồng",
        "vi phạm hợp đồng",
        "nghĩa vụ thanh toán",
        "phạt vi phạm",
        "bồi thường do vi phạm",
    ],
    "ontology:issue:credit_contract": [
        "tín dụng",
        "ngân hàng",
        "cho vay",
        "lãi suất",
        "thế chấp",
        "xử lý tài sản bảo đảm",
    ],
    "ontology:issue:land_dispute": [
        "quyền sử dụng đất",
        "đất đai",
        "giấy chứng nhận",
        "chuyển nhượng quyền sử dụng đất",
        "tranh chấp đất",
    ],
    "ontology:issue:inheritance": [
        "thừa kế",
        "di sản",
        "di chúc",
        "hàng thừa kế",
        "chia di sản",
    ],
    "ontology:issue:marriage_family": [
        "hôn nhân",
        "gia đình",
        "ly hôn",
        "nuôi con",
        "tài sản chung vợ chồng",
    ],
    "ontology:issue:court_fee": [
        "án phí",
        "lệ phí tòa án",
        "tạm ứng án phí",
        "miễn án phí",
    ],
    "ontology:issue:civil_procedure": [
        "tố tụng dân sự",
        "thẩm quyền",
        "kháng cáo",
        "triệu tập hợp lệ",
        "xét xử vắng mặt",
    ],
    "ontology:issue:enforcement": [
        "thi hành án",
        "người được thi hành án",
        "người phải thi hành án",
        "cưỡng chế thi hành",
    ],
    "ontology:issue:construction_contract": [
        "xây dựng",
        "hợp đồng xây dựng",
        "thi công",
        "nghiệm thu",
    ],
}


def normalize_text(value: Any) -> str:
    text = str(value or "")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.replace("đ", "d").replace("Đ", "D")
    return re.sub(r"\s+", " ", text.lower()).strip()


def coerce_aid(value: Any) -> int | str | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        text = str(value).strip()
        return text or None


def safe_node_part(value: Any) -> str:
    return re.sub(r"[^0-9A-Za-z._/-]+", "_", str(value or "unknown")).strip("_") or "unknown"


def legal_article_number(chunk: dict[str, Any]) -> str | None:
    article_index = chunk.get("article_index")
    try:
        return str(int(article_index))
    except (TypeError, ValueError):
        pass
    for field in ("article_number", "article_label", "unit_path", "aid"):
        value = chunk.get(field)
        if value in (None, ""):
            continue
        match = ARTICLE_REF_RE.search(str(value))
        if match:
            return match.group(1).lower()
        try:
            return str(int(value))
        except (TypeError, ValueError):
            continue
    return None


@dataclass
class GraphNode:
    node_id: str
    labels: set[str]
    properties: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "labels": sorted(self.labels),
            "properties": clean_properties({"node_id": self.node_id, **self.properties}),
        }


@dataclass(frozen=True)
class GraphEdge:
    src: str
    dst: str
    edge_type: str
    properties: tuple[tuple[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "src": self.src,
            "dst": self.dst,
            "edge_type": self.edge_type,
            "properties": clean_properties(dict(self.properties)),
        }


def clean_properties(properties: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in properties.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        elif isinstance(value, list):
            cleaned[key] = [
                item for item in value if isinstance(item, (str, int, float, bool))
            ]
        else:
            cleaned[key] = str(value)
    return cleaned


def merge_node(nodes: dict[str, GraphNode], node: GraphNode) -> None:
    existing = nodes.get(node.node_id)
    if existing is None:
        nodes[node.node_id] = node
        return

    existing.labels.update(node.labels)
    for key, value in node.properties.items():
        if key not in existing.properties or existing.properties[key] in (None, "", []):
            existing.properties[key] = value
        elif key == "aliases":
            current = existing.properties.get(key)
            if not isinstance(current, list):
                current = [current]
            incoming = value if isinstance(value, list) else [value]
            existing.properties[key] = sorted({str(item) for item in current + incoming if item})


def edge_key(edge: GraphEdge) -> tuple[str, str, str, tuple[tuple[str, Any], ...]]:
    return (edge.src, edge.dst, edge.edge_type, edge.properties)


def concept_nodes() -> Iterable[GraphNode]:
    for node_id, aliases in CONCEPT_ALIASES.items():
        yield GraphNode(
            node_id=node_id,
            labels={"LegalNode", "Concept"},
            properties={
                "layer": "ontology",
                "node_type": "concept",
                "text": aliases[0],
                "aliases": aliases,
                "normalized_alias": normalize_text(" ".join(aliases)),
            },
        )


def matched_concepts(text: str) -> list[str]:
    normalized = normalize_text(text)
    matches: list[str] = []
    for node_id, aliases in CONCEPT_ALIASES.items():
        if any(normalize_text(alias) in normalized for alias in aliases):
            matches.append(node_id)
    return matches


def chunk_to_graph_records(chunk: dict[str, Any]) -> tuple[list[GraphNode], list[GraphEdge]]:
    chunk_id = str(chunk.get("chunk_id") or chunk.get("id") or "").strip()
    law_id = str(chunk.get("law_id") or "").strip()
    aid = coerce_aid(chunk.get("aid"))
    text = str(chunk.get("text") or chunk.get("content") or "")
    if not chunk_id or not law_id or aid is None:
        return [], []

    article_number = legal_article_number(chunk) or str(aid)
    article_part = safe_node_part(article_number)
    article_node_id = f"rule:{law_id}:{article_part}"
    law_node_id = f"law:{law_id}"
    unit_type = str(chunk.get("unit_type") or "chunk").strip() or "chunk"
    chunk_node_id = f"chunk:{chunk_id}"
    unit_path = str(chunk.get("unit_path") or "")
    extracted_features = extract_legal_features(
        f"{unit_path}\n{text}",
        concept_aliases=CONCEPT_ALIASES,
    )
    ontology_concepts = list(chunk.get("ontology_concepts") or extracted_features["ontology_concepts"])
    rule_signals = list(chunk.get("rule_signals") or extracted_features["rule_signals"])
    article_references = list(chunk.get("article_references") or extracted_features["article_references"])
    clause_number = str(chunk.get("clause_number") or "").strip() or None
    container_node_id = article_node_id

    nodes = [
        GraphNode(
            node_id=law_node_id,
            labels={"LegalNode", "Law"},
            properties={
                "layer": "rule",
                "node_type": "law",
                "law_id": law_id,
                "doc_id": chunk.get("doc_id"),
                "source_type": chunk.get("source_type"),
                "text": law_id,
            },
        ),
        GraphNode(
            node_id=article_node_id,
            labels={"LegalNode", "Article"},
            properties={
                "layer": "rule",
                "node_type": "article",
                "law_id": law_id,
                "aid": aid,
                "source_aid": aid,
                "article_number": article_number,
                "article_index": chunk.get("article_index"),
                "article_label": chunk.get("article_label"),
                "ontology_concepts": ontology_concepts,
                "rule_signals": rule_signals,
                "text": unit_path or f"{law_id} Dieu {article_number}",
                "source_chunk_id": chunk_id if unit_type == "article" else None,
            },
        ),
        GraphNode(
            node_id=chunk_node_id,
            labels={"LegalNode", "SourceChunk", _unit_label(unit_type)},
            properties={
                "layer": "rule",
                "node_type": unit_type,
                "law_id": law_id,
                "aid": aid,
                "source_aid": aid,
                "article_number": article_number,
                "article_index": chunk.get("article_index"),
                "source_chunk_id": chunk_id,
                "chunk_id": chunk_id,
                "article_node_id": article_node_id,
                "article_id": chunk.get("article_id"),
                "unit_path": unit_path,
                "clause_number": chunk.get("clause_number"),
                "point_label": chunk.get("point_label"),
                "ontology_concepts": ontology_concepts,
                "rule_signals": rule_signals,
                "article_references": article_references,
                "text": text,
            },
        ),
    ]

    edges = [
        GraphEdge(
            src=law_node_id,
            dst=article_node_id,
            edge_type="CONTAINS",
            properties=(("weight", 1.0), ("evidence", "same law_id"),),
        ),
    ]

    if clause_number:
        clause_node_id = f"rule:{law_id}:{article_part}:clause:{safe_node_part(clause_number)}"
        container_node_id = clause_node_id
        nodes.append(
            GraphNode(
                node_id=clause_node_id,
                labels={"LegalNode", "Clause"},
                properties={
                    "layer": "rule",
                    "node_type": "clause",
                    "law_id": law_id,
                    "aid": aid,
                    "article_number": article_number,
                    "clause_number": clause_number,
                    "source_chunk_id": chunk_id if unit_type.startswith("clause") else None,
                    "text": f"{law_id} Dieu {article_number} khoan {clause_number}",
                },
            )
        )
        edges.append(
            GraphEdge(
                src=article_node_id,
                dst=clause_node_id,
                edge_type="CONTAINS",
                properties=(("weight", 1.0), ("evidence", "clause belongs to article")),
            )
        )

    edges.append(
        GraphEdge(
            src=container_node_id,
            dst=chunk_node_id,
            edge_type="CONTAINS",
            properties=(("weight", 1.0), ("evidence", "source chunk belongs to legal unit")),
        )
    )

    for concept_id in ontology_concepts:
        edges.append(
            GraphEdge(
                src=concept_id,
                dst=article_node_id,
                edge_type="GOVERNED_BY",
                properties=(("weight", 0.8), ("evidence", "concept alias matched source text"),),
            )
        )
        edges.append(
            GraphEdge(
                src=concept_id,
                dst=chunk_node_id,
                edge_type="RELATED_TO",
                properties=(("weight", 0.5), ("evidence", "concept alias matched source chunk"),),
            )
        )

    for rule_signal in rule_signals:
        signal_node_id = f"ontology:rule_signal:{safe_node_part(rule_signal)}"
        nodes.append(
            GraphNode(
                node_id=signal_node_id,
                labels={"LegalNode", "RuleSignal"},
                properties={
                    "layer": "ontology",
                    "node_type": "rule_signal",
                    "text": rule_signal,
                    "normalized_alias": rule_signal,
                },
            )
        )
        edges.append(
            GraphEdge(
                src=signal_node_id,
                dst=article_node_id,
                edge_type="GOVERNED_BY",
                properties=(("weight", 0.7), ("evidence", "rule signal extracted from source text")),
            )
        )

    for referenced_aid in article_references:
        if str(referenced_aid) == str(article_number):
            continue
        edges.append(
            GraphEdge(
                src=article_node_id,
                dst=f"rule:{law_id}:{safe_node_part(referenced_aid)}",
                edge_type="CITES",
                properties=(("weight", 0.4), ("evidence", f"article reference Điều {referenced_aid}"),),
            )
        )

    return nodes, edges


def _unit_label(unit_type: str) -> str:
    lowered = unit_type.lower()
    if "clause" in lowered:
        return "Clause"
    if "point" in lowered:
        return "Point"
    if "article" in lowered:
        return "ArticleUnit"
    return "RuleUnit"


def build_graph_records(
    chunks_path: str | Path,
    *,
    deprecated_filter: DeprecatedFilter | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    deprecated_filter = deprecated_filter or DeprecatedFilter()
    nodes: dict[str, GraphNode] = {}
    edges: dict[tuple[str, str, str, tuple[tuple[str, Any], ...]], GraphEdge] = {}
    skipped = 0
    chunk_count = 0

    for concept in concept_nodes():
        merge_node(nodes, concept)

    path = Path(chunks_path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            chunk = json.loads(line)
            if deprecated_filter.is_deprecated(chunk):
                skipped += 1
                continue
            chunk_count += 1
            chunk_nodes, chunk_edges = chunk_to_graph_records(chunk)
            if not chunk_nodes:
                skipped += 1
                continue
            for node in chunk_nodes:
                merge_node(nodes, node)
            for edge in chunk_edges:
                edges.setdefault(edge_key(edge), edge)

    node_rows = [node.as_dict() for node in nodes.values()]
    edge_rows = [edge.as_dict() for edge in edges.values()]
    stats = {
        "chunks_seen": chunk_count,
        "chunks_skipped": skipped,
        "node_count": len(node_rows),
        "edge_count": len(edge_rows),
        "node_labels": dict(Counter(label for node in node_rows for label in node["labels"])),
        "edge_types": dict(Counter(edge["edge_type"] for edge in edge_rows)),
    }
    return node_rows, edge_rows, stats


def write_graph_artifacts(
    output_dir: str | Path,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    stats: dict[str, Any],
) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    with (path / "nodes.jsonl").open("w", encoding="utf-8") as handle:
        for node in nodes:
            handle.write(json.dumps(node, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    with (path / "edges.jsonl").open("w", encoding="utf-8") as handle:
        for edge in edges:
            handle.write(json.dumps(edge, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    manifest = {
        "schema_version": 1,
        "backend": "neo4j",
        **stats,
    }
    (path / "graph_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def build_legal_graph(
    chunks_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    nodes, edges, stats = build_graph_records(chunks_path)
    write_graph_artifacts(output_dir, nodes, edges, stats)
    return stats
