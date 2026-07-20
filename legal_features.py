"""Deterministic legal-feature extraction shared by preprocessing and graph build.

The source LegalGraphRAG project uses an LLM to turn case narratives into a
small set of legally meaningful features before graph construction.  ALQAC's
corpus is statutory text, so this module extracts transparent rule features
instead of inventing case facts or relying on another model call.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Mapping


ARTICLE_REFERENCE_RE = re.compile(r"\b(?:điều|dieu)\s+([0-9]+[a-z]?)\b", re.IGNORECASE)

CONCEPT_ALIASES: dict[str, list[str]] = {
    "ontology:issue:tort_damage": [
        "bồi thường thiệt hại",
        "thiệt hại ngoài hợp đồng",
        "trách nhiệm bồi thường",
        "xâm phạm tài sản",
        "xâm phạm sức khỏe",
    ],
    "ontology:liability:animal_damage": ["súc vật", "vật nuôi", "chó", "trâu", "bò", "gây thiệt hại"],
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
    "ontology:issue:inheritance": ["thừa kế", "di sản", "di chúc", "hàng thừa kế", "chia di sản"],
    "ontology:issue:marriage_family": ["hôn nhân", "gia đình", "ly hôn", "nuôi con", "tài sản chung vợ chồng"],
    "ontology:issue:court_fee": ["án phí", "lệ phí tòa án", "tạm ứng án phí", "miễn án phí"],
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
    "ontology:issue:construction_contract": ["xây dựng", "hợp đồng xây dựng", "thi công", "nghiệm thu"],
}

RULE_SIGNAL_PATTERNS: dict[str, tuple[str, ...]] = {
    "obligation": ("phải", "có nghĩa vụ", "chịu trách nhiệm"),
    "prohibition": ("không được", "nghiêm cấm"),
    "entitlement": ("có quyền", "được quyền", "được yêu cầu"),
    "condition": ("trường hợp", "nếu", "khi"),
    "exception": ("trừ trường hợp", "trừ khi"),
    "procedure": ("thẩm quyền", "thủ tục", "thời hạn", "khởi kiện", "kháng cáo"),
}


def normalize_legal_text(value: Any) -> str:
    text = unicodedata.normalize("NFD", str(value or "").lower())
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    text = text.replace("đ", "d")
    return re.sub(r"\s+", " ", text).strip()


def matched_concepts(
    text: Any,
    concept_aliases: Mapping[str, list[str]] = CONCEPT_ALIASES,
) -> list[str]:
    normalized = normalize_legal_text(text)
    return [
        node_id
        for node_id, aliases in concept_aliases.items()
        if any(normalize_legal_text(alias) in normalized for alias in aliases)
    ]


def extract_legal_features(
    text: Any,
    *,
    concept_aliases: Mapping[str, list[str]] = CONCEPT_ALIASES,
) -> dict[str, list[str]]:
    """Return explainable, source-grounded legal features for one text unit."""
    raw_text = str(text or "")
    normalized = normalize_legal_text(raw_text)
    rule_signals = [
        signal
        for signal, patterns in RULE_SIGNAL_PATTERNS.items()
        if any(normalize_legal_text(pattern) in normalized for pattern in patterns)
    ]
    article_references = list(dict.fromkeys(match.lower() for match in ARTICLE_REFERENCE_RE.findall(raw_text)))
    return {
        "ontology_concepts": matched_concepts(raw_text, concept_aliases),
        "rule_signals": rule_signals,
        "article_references": article_references,
    }
