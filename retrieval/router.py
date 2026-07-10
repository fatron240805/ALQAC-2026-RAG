from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Mapping


_TOKEN_RE = re.compile(r"\b[\wÀ-ỹ]+\b", re.UNICODE)
_STATUTE_RE = re.compile(
    r"\b(?:điều|dieu)\s+([0-9]+[a-zA-Z]?)|"
    r"\b([0-9]{1,3}/[0-9]{4}/[A-ZĐ\-0-9]+)\b",
    re.IGNORECASE,
)


DOMAIN_KEYWORDS = {
    "civil": {
        "dân sự",
        "bồi thường",
        "hợp đồng",
        "thừa kế",
        "tài sản",
        "đất",
        "vay",
        "nợ",
        "nguyên đơn",
        "bị đơn",
    },
    "procedure": {"tố tụng", "án phí", "thẩm quyền", "kháng cáo", "thi hành án"},
    "banking": {"tín dụng", "ngân hàng", "cho vay", "bảo lãnh", "lãi suất"},
    "gold_business": {"vàng", "vàng miếng", "kinh doanh vàng", "trang sức"},
}


@dataclass(frozen=True)
class QueryAnalysis:
    raw_query: str
    normalized_query: str
    tokens: list[str]
    domains: list[str]
    statute_references: list[str]
    retrieval_depth: str
    routed_law_ids: list[str]


def normalize_query(text: str) -> str:
    value = unicodedata.normalize("NFC", text or "")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(normalize_query(text))]


class QueryAnalyzer:
    """Lightweight legal query analyzer for Phase 1 retrieval routing."""

    def analyze(self, query: str) -> QueryAnalysis:
        normalized = normalize_query(query)
        lowered = normalized.lower()
        tokens = tokenize(normalized)

        domains = [
            domain
            for domain, keywords in DOMAIN_KEYWORDS.items()
            if any(keyword in lowered for keyword in keywords)
        ]
        statute_references = [
            match.group(1) or match.group(2)
            for match in _STATUTE_RE.finditer(normalized)
            if match.group(1) or match.group(2)
        ]

        retrieval_depth = "deep" if statute_references or len(tokens) > 80 else "fast"
        routed_law_ids = [ref for ref in statute_references if "/" in ref]
        return QueryAnalysis(
            raw_query=query,
            normalized_query=normalized,
            tokens=tokens,
            domains=domains or ["general"],
            statute_references=statute_references,
            retrieval_depth=retrieval_depth,
            routed_law_ids=routed_law_ids,
        )


class DocumentRouter:
    """Score candidates with small metadata/domain boosts before final fusion."""

    def __init__(self, analyzer: QueryAnalyzer | None = None) -> None:
        self.analyzer = analyzer or QueryAnalyzer()

    def route_boost(self, analysis: QueryAnalysis, metadata: Mapping[str, Any]) -> float:
        boost = 0.0
        law_id = str(metadata.get("law_id") or "")
        unit_path = str(metadata.get("unit_path") or "").lower()

        if law_id and law_id in analysis.routed_law_ids:
            boost += 0.25
        if analysis.statute_references and any(ref.lower() in unit_path for ref in analysis.statute_references):
            boost += 0.15
        if "civil" in analysis.domains and any(word in unit_path for word in ("dân sự", "dan su")):
            boost += 0.05
        return boost

    def apply(self, query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        analysis = self.analyzer.analyze(query)
        routed: list[dict[str, Any]] = []
        for candidate in candidates:
            item = dict(candidate)
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            boost = self.route_boost(analysis, metadata)
            item["routing_boost"] = boost
            item["routed_score"] = float(item.get("fused_score", 0.0)) + boost
            item["query_domains"] = analysis.domains
            routed.append(item)
        routed.sort(key=lambda item: item.get("routed_score", 0.0), reverse=True)
        return routed
