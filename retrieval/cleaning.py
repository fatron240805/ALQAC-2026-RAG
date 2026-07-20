"""Clean and normalize the ALQAC law corpus for retrieval.

This module is intentionally dependency-light so the first corpus pipeline can
run anywhere Python 3.10+ is available. It converts the raw ALQAC JSON law
corpus into:

- document-level JSONL records
- article-level JSONL records
- retrieval chunk JSONL records
- a small cleaning audit report

The code does not remove documents automatically. It marks possible deprecated
or replaced laws and keeps deprecation signals for human audit.

Changelog (v3 vs v2)
---------------------
1. INLINE_CLAUSE_RE no longer splits a line just because it sees " N. ".
   Vietnamese statutes constantly *reference* other clauses/articles
   ("... quy định tại khoản 2. Trường hợp khác ...") and the old regex
   mistook that reference for the start of a brand-new "khoản", corrupting
   article structure and chunk boundaries. A context guard now checks the
   word immediately before the number and skips the split when it is a
   reference keyword (khoản/điểm/điều/mục/chương/phần/số/...).
2. The same context guard is applied to INLINE_POINT_RE for symmetry/safety,
   in addition to its existing ";"/":" anchor requirement.
3. infer_article_number() now prefers the source `aid` field (when present
   and non-empty) over regex/positional guesses, since it is normally the
   most reliable signal for the true "Điều N" number.
4. ARTICLE_HEADING_RE is now only searched against the first couple of lines
   of an article's text, not the whole body — otherwise a cross-reference to
   another article inside the body text ("Điều 15 của luật này quy định
   ...") could be mistaken for the article's own heading.
5. article_index is now a *continuous* counter over kept (non-empty)
   articles only. The original raw JSON position is preserved separately as
   raw_position so debugging/traceability is not lost, but numbering
   fallbacks no longer have artificial gaps when an empty article is
   skipped.
6. Removed dead code (split_paragraphs, tail_by_paragraph_tokens) that was
   never invoked by the pipeline.
7. schema_version bumped to 3 and the audit report records which cleaning
   rules were active, for reproducibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from legal_features import extract_legal_features

RAW_CORPUS_PATH = Path("data/raw/ALQAC/corpus_law_pub.json")
DOCS_OUTPUT_PATH = Path("data/cleaned/corpus.jsonl")
ARTICLES_OUTPUT_PATH = Path("data/cleaned/articles.jsonl")
CHUNKS_OUTPUT_PATH = Path("data/chunks.jsonl")
AUDIT_OUTPUT_PATH = Path("data/cleaned/cleaning_audit.json")

SOURCE_TYPE = "statute"
SCHEMA_VERSION = 5


CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MULTI_SPACE_RE = re.compile(r"[ \t]+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
PAGE_NUMBER_RE = re.compile(r"(?m)^\s*\d+\s*$")
BROKEN_HYPHEN_RE = re.compile(r"(\w)-\n(\w)")
ARTICLE_HEADING_RE = re.compile(
    r"(?im)^\s*(?:điều|dieu)\s+([0-9]+[a-zA-Z]?)\s*[\.:)\-–]?\s*(.*)$"
)
# How many leading lines of an article's text we are willing to search for
# its own "Điều N" heading. Keeping this small avoids matching a
# cross-reference to a *different* article buried in the body.
ARTICLE_HEADING_SEARCH_LINES = 2

LAW_YEAR_RE = re.compile(r"/([0-9]{4})/")
CLAUSE_START_RE = re.compile(r"(?m)^\s*([1-9][0-9]*)\.\s+")
INLINE_CLAUSE_RE = re.compile(r"(?<![0-9])\s+([1-9][0-9]*)\.\s+")
POINT_START_RE = re.compile(r"(?m)^\s*([a-zđ])\)\s+", re.IGNORECASE)
INLINE_POINT_RE = re.compile(r"(?<=[;:\.])\s+([a-zđ])\)\s+", re.IGNORECASE)

# Words that, when they appear right before an inline "N." or "x)" pattern,
# indicate the number/letter is a *cross-reference* (e.g. "tại khoản 2.",
# "quy định tại điểm a)") rather than the start of a new clause/point.
CLAUSE_REFERENCE_KEYWORDS = {
    "khoan",
    "diem",
    "dieu",
    "muc",
    "chuong",
    "phan",
    "so",
    "phu",
    "luc",
}

DEPRECATED_PATTERNS = (
    r"\b(?:luật|nghị định|thông tư|quyết định|pháp lệnh|văn bản|điều|khoản)\b[^.\n]{0,160}\b(?:hết hiệu lực|chấm dứt hiệu lực|không còn hiệu lực|bị bãi bỏ|được bãi bỏ|được thay thế)\b",
    r"\b(?:hết hiệu lực|chấm dứt hiệu lực|không còn hiệu lực|bị bãi bỏ|được bãi bỏ|được thay thế)\b[^.\n]{0,160}\b(?:luật|nghị định|thông tư|quyết định|pháp lệnh|văn bản|điều|khoản|số)\b",
    r"\b(?:luật|nghị định|thông tư|quyết định|pháp lệnh|văn bản)\b[^.\n]{0,160}\bthay thế\b[^.\n]{0,160}\b(?:luật|nghị định|thông tư|quyết định|pháp lệnh|văn bản|số)\b",
)
DEPRECATED_RE = re.compile("|".join(DEPRECATED_PATTERNS), re.IGNORECASE)


@dataclass(frozen=True)
class DocumentRecord:
    doc_id: str
    law_id: str
    source_type: str
    source_path: str
    raw_sha256: str
    content_sha256: str
    article_count: int
    char_count: int
    token_count_estimate: int
    issue_year: int | None
    deprecated: bool
    deprecation_signals: list[str]
    cleaned_at: str
    doc_content: str


@dataclass(frozen=True)
class ArticleRecord:
    article_id: str
    doc_id: str
    law_id: str
    aid: int | str | None
    article_index: int
    raw_position: int
    article_number: str | None
    article_number_source: str
    article_label: str
    source_type: str
    deprecated: bool
    deprecation_signals: list[str]
    content_sha256: str
    char_count: int
    token_count_estimate: int
    article_text: str


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    article_id: str
    doc_id: str
    law_id: str
    aid: int | str | None
    article_index: int
    article_number: str | None
    article_label: str
    unit_type: str
    unit_path: str
    clause_number: str | None
    point_label: str | None
    source_type: str
    deprecated: bool
    chunk_index: int
    char_start: int
    char_end: int
    token_count_estimate: int
    content_sha256: str
    text: str
    ontology_concepts: list[str]
    rule_signals: list[str]
    article_references: list[str]


@dataclass(frozen=True)
class LegalUnit:
    unit_type: str
    char_start: int
    char_end: int
    text: str
    clause_number: str | None = None
    point_label: str | None = None


def read_json(path: Path) -> Any:
    """Read JSON with explicit UTF-8 to preserve Vietnamese text."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            file.write("\n")
            count += 1
    return count


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_id(*parts: Any, prefix: str) -> str:
    body = "::".join("" if part is None else str(part) for part in parts)
    digest = sha256_text(body)[:16]
    return f"{prefix}_{digest}"


def strip_accents(text: str) -> str:
    """ASCII-fold Vietnamese text for keyword matching regardless of diacritics.

    NFD decomposition strips combining diacritics (the circumflex/grave/etc.
    in "ề", "ầ", ...) but does NOT decompose "Đ"/"đ" — in Unicode that is a
    distinct base letter (U+0110/U+0111), not "d" plus a combining mark. It
    is folded to ASCII "d" explicitly here so keyword comparisons (e.g.
    "Điều" -> "dieu") work regardless of case or diacritics.
    """
    text = text.replace("Đ", "D").replace("đ", "d")
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def normalize_text(text: Any) -> str:
    """Normalize legal text while preserving article boundaries and accents."""
    if text is None:
        return ""
    value = str(text)
    value = value.replace("\ufeff", "")
    value = unicodedata.normalize("NFC", value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = BROKEN_HYPHEN_RE.sub(r"\1\2", value)
    value = CONTROL_CHARS_RE.sub(" ", value)
    value = PAGE_NUMBER_RE.sub("", value)

    normalized_lines: list[str] = []
    for line in value.split("\n"):
        line = MULTI_SPACE_RE.sub(" ", line).strip()
        normalized_lines.append(line)

    value = "\n".join(normalized_lines)
    value = MULTI_NEWLINE_RE.sub("\n\n", value)
    return value.strip()


def _is_clause_reference(text: str, match_start: int) -> bool:
    """True if the number/letter at match_start is a cross-reference.

    Looks at the word immediately preceding the match (e.g. "tại khoản",
    "quy định tại điểm") and, if it is a known legal-reference keyword,
    treats the number/letter as a reference rather than a new
    clause/point boundary.
    """
    prefix = text[:match_start].rstrip()
    word_match = re.search(r"([^\W\d_]+)$", prefix, re.UNICODE)
    if not word_match:
        return False
    word = strip_accents(word_match.group(1)).lower()
    return word in CLAUSE_REFERENCE_KEYWORDS


def _split_inline_markers(
    value: str,
    pattern: re.Pattern[str],
    formatter: Callable[[re.Match[str]], str],
) -> str:
    """Insert a newline before inline clause/point markers, skipping references."""
    result: list[str] = []
    cursor = 0
    for match in pattern.finditer(value):
        if _is_clause_reference(value, match.start()):
            continue
        result.append(value[cursor : match.start()])
        result.append(formatter(match))
        cursor = match.end()
    result.append(value[cursor:])
    return "".join(result)


def normalize_legal_lines(text: str) -> str:
    """Put Vietnamese legal clauses and points on stable lines.

    Raw ALQAC articles often contain inline legal points such as
    ``bao gồm: a) ...; b) ...``. Retrieval works better when those units are
    line-addressable, so this keeps article text readable while exposing
    ``khoản`` and ``điểm`` boundaries. Inline numbers/letters that are really
    *cross-references* to other clauses/points/articles (e.g. "tại khoản 2.")
    are left untouched instead of being treated as new boundaries.
    """
    value = normalize_text(text)
    value = _split_inline_markers(
        value, INLINE_POINT_RE, lambda m: f"\n{m.group(1).lower()}) "
    )

    lines: list[str] = []
    for line in value.split("\n"):
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue

        clause_match = None
        for candidate in INLINE_CLAUSE_RE.finditer(stripped):
            if _is_clause_reference(stripped, candidate.start()):
                continue
            clause_match = candidate
            break

        if clause_match and not CLAUSE_START_RE.match(stripped):
            prefix = stripped[: clause_match.start()].strip()
            suffix = stripped[clause_match.start() :].strip()
            if prefix:
                lines.append(prefix)
            lines.append(suffix)
            continue

        lines.append(stripped)

    value = "\n".join(lines)
    value = MULTI_NEWLINE_RE.sub("\n\n", value)
    return value.strip()


def estimate_tokens(text: str) -> int:
    """Cheap Vietnamese-friendly token estimate for chunk sizing."""
    if not text:
        return 0
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def infer_issue_year(law_id: str) -> int | None:
    match = LAW_YEAR_RE.search(law_id or "")
    if not match:
        return None
    return int(match.group(1))


def infer_article_number(
    article_text: str,
    aid: int | str | None,
    article_index: int,
) -> tuple[str | None, str]:
    """Return (article_number, source) where source explains provenance.

    Preference order:
    1. The raw `aid` field, when present and non-empty — it is normally the
       most reliable signal for the true "Điều N" number.
    2. A regex match of "Điều N" restricted to the first couple of lines of
       the article text (searching the whole body risks matching a
       cross-reference to a *different* article mentioned further down).
    3. The continuous kept-article index, as a last resort.
    """
    if aid is not None:
        aid_str = str(aid).strip()
        if aid_str and aid_str.lower() not in {"none", "null"}:
            return aid_str, "aid"

    heading_scope = "\n".join(article_text.split("\n")[:ARTICLE_HEADING_SEARCH_LINES])
    match = ARTICLE_HEADING_RE.search(heading_scope)
    if match:
        return match.group(1), "heading_regex"

    return str(article_index), "fallback_index"


def find_deprecation_signals(text: str, *, max_signals: int = 8) -> list[str]:
    """Return short snippets that suggest a document may be deprecated.

    These are *candidate* signals for human audit only — a clause that
    merely explains when legal documents lose effect in general can also
    match. Do not treat `deprecated=True` as a conclusive determination.
    """
    signals: list[str] = []
    for match in DEPRECATED_RE.finditer(text):
        start = max(0, match.start() - 90)
        end = min(len(text), match.end() + 140)
        snippet = text[start:end]
        snippet = " ".join(snippet.split())
        if snippet not in signals:
            signals.append(snippet)
        if len(signals) >= max_signals:
            break
    return signals


def iter_raw_documents(raw: Any) -> Iterator[dict[str, Any]]:
    if not isinstance(raw, list):
        raise TypeError("Expected the raw ALQAC law corpus to be a JSON array.")
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise TypeError(f"Expected corpus item {index} to be an object.")
        yield item


def iter_raw_articles(raw_doc: dict[str, Any]) -> Iterator[dict[str, Any]]:
    content = raw_doc.get("content", [])
    if not isinstance(content, list):
        return
    for article in content:
        if isinstance(article, dict):
            yield article


def build_document_record(
    raw_doc: dict[str, Any],
    *,
    source_path: Path,
    cleaned_at: str,
) -> tuple[DocumentRecord, list[ArticleRecord]]:
    law_id = normalize_text(raw_doc.get("law_id", ""))
    raw_doc_json = json.dumps(raw_doc, ensure_ascii=False, sort_keys=True)
    raw_doc_hash = sha256_text(raw_doc_json)
    doc_id = stable_id(law_id, raw_doc.get("id"), raw_doc_hash, prefix="doc")

    article_records: list[ArticleRecord] = []
    article_texts: list[str] = []
    kept_index = 0

    for position, raw_article in enumerate(iter_raw_articles(raw_doc)):
        raw_position = position + 1
        aid = raw_article.get("aid")
        article_text = normalize_legal_lines(raw_article.get("content_Article", ""))
        if not article_text:
            continue

        kept_index += 1
        article_number, article_number_source = infer_article_number(
            article_text, aid, kept_index
        )
        article_label = (
            f"Điều {article_number}" if article_number else f"Điều {kept_index}"
        )
        article_id = stable_id(doc_id, aid, position, article_text, prefix="article")
        article_signals = find_deprecation_signals(article_text)

        article_records.append(
            ArticleRecord(
                article_id=article_id,
                doc_id=doc_id,
                law_id=law_id,
                aid=aid,
                article_index=kept_index,
                raw_position=raw_position,
                article_number=article_number,
                article_number_source=article_number_source,
                article_label=article_label,
                source_type=SOURCE_TYPE,
                deprecated=bool(article_signals),
                deprecation_signals=article_signals,
                content_sha256=sha256_text(article_text),
                char_count=len(article_text),
                token_count_estimate=estimate_tokens(article_text),
                article_text=article_text,
            )
        )
        article_texts.append(article_text)

    doc_content = "\n\n".join(article_texts)
    doc_signals = find_deprecation_signals(doc_content)

    document_record = DocumentRecord(
        doc_id=doc_id,
        law_id=law_id,
        source_type=SOURCE_TYPE,
        source_path=source_path.as_posix(),
        raw_sha256=raw_doc_hash,
        content_sha256=sha256_text(doc_content),
        article_count=len(article_records),
        char_count=len(doc_content),
        token_count_estimate=estimate_tokens(doc_content),
        issue_year=infer_issue_year(law_id),
        deprecated=bool(doc_signals),
        deprecation_signals=doc_signals,
        cleaned_at=cleaned_at,
        doc_content=doc_content,
    )

    return document_record, article_records


def split_long_text_by_sentence(
    text: str,
    *,
    target_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    sentences = re.split(r"(?<=[.!?;:])\s+|\n+", text)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence_tokens = estimate_tokens(sentence)
        if current and current_tokens + sentence_tokens > target_tokens:
            chunks.append(" ".join(current).strip())
            current = tail_by_tokens(current, overlap_tokens)
            current_tokens = estimate_tokens(" ".join(current))
        current.append(sentence)
        current_tokens += sentence_tokens

    if current:
        chunks.append(" ".join(current).strip())

    return [chunk for chunk in chunks if chunk]


def split_by_line_pattern(
    text: str,
    pattern: re.Pattern[str],
) -> list[tuple[re.Match[str] | None, int, int, str]]:
    matches = list(pattern.finditer(text))
    if not matches:
        return [(None, 0, len(text), text.strip())] if text.strip() else []

    blocks: list[tuple[re.Match[str] | None, int, int, str]] = []
    if matches[0].start() > 0:
        prefix = text[: matches[0].start()].strip()
        if prefix:
            blocks.append((None, 0, matches[0].start(), prefix))

    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block_text = text[start:end].strip()
        if block_text:
            blocks.append((match, start, end, block_text))
    return blocks


def split_clause_points(
    clause_text: str,
    *,
    clause_start: int,
    clause_number: str | None,
) -> list[LegalUnit]:
    point_blocks = split_by_line_pattern(clause_text, POINT_START_RE)
    if len(point_blocks) == 1 and point_blocks[0][0] is None:
        return [
            LegalUnit(
                unit_type="clause" if clause_number else "article",
                char_start=clause_start,
                char_end=clause_start + len(clause_text),
                text=clause_text.strip(),
                clause_number=clause_number,
            )
        ]

    intro = ""
    units: list[LegalUnit] = []
    for match, start, end, block_text in point_blocks:
        absolute_start = clause_start + start
        absolute_end = clause_start + end
        if match is None:
            intro = block_text.strip()
            if intro and estimate_tokens(intro) >= 8:
                units.append(
                    LegalUnit(
                        unit_type="clause_intro",
                        char_start=absolute_start,
                        char_end=absolute_end,
                        text=intro,
                        clause_number=clause_number,
                    )
                )
            continue

        point_text = block_text.strip()
        text = f"{intro}\n{point_text}".strip() if intro else point_text
        units.append(
            LegalUnit(
                unit_type="point",
                char_start=absolute_start,
                char_end=absolute_end,
                text=text,
                clause_number=clause_number,
                point_label=match.group(1).lower(),
            )
        )

    return units


def split_article_units(article: ArticleRecord, *, target_tokens: int) -> list[LegalUnit]:
    """Split one article with multi-granular, pattern-based legal chunking.

    Article-level units are the default because legal context is usually held
    across the whole article. Only dense/oversized articles are split by legal
    section markers such as clauses and points. Sentence-level splitting is
    left to split_oversized_unit as the last-resort size fallback.
    """
    if estimate_tokens(article.article_text) <= target_tokens:
        return [
            LegalUnit(
                unit_type="article",
                char_start=0,
                char_end=len(article.article_text),
                text=article.article_text,
            )
        ]

    has_structured_units = bool(
        CLAUSE_START_RE.search(article.article_text)
        or POINT_START_RE.search(article.article_text)
    )
    if not has_structured_units:
        return [
            LegalUnit(
                unit_type="article",
                char_start=0,
                char_end=len(article.article_text),
                text=article.article_text,
            )
        ]

    clause_blocks = split_by_line_pattern(article.article_text, CLAUSE_START_RE)
    units: list[LegalUnit] = []
    for match, start, end, block_text in clause_blocks:
        clause_number = match.group(1) if match else None
        if match is None and estimate_tokens(block_text) <= target_tokens:
            units.append(
                LegalUnit(
                    unit_type="article_intro",
                    char_start=start,
                    char_end=end,
                    text=block_text,
                )
            )
            continue

        units.extend(
            split_clause_points(
                block_text,
                clause_start=start,
                clause_number=clause_number,
            )
        )

    if not units:
        return [
            LegalUnit(
                unit_type="article",
                char_start=0,
                char_end=len(article.article_text),
                text=article.article_text,
            )
        ]
    return units


def split_oversized_unit(
    unit: LegalUnit,
    *,
    target_tokens: int,
    overlap_tokens: int,
) -> list[LegalUnit]:
    if estimate_tokens(unit.text) <= target_tokens:
        return [unit]

    split_texts = split_long_text_by_sentence(
        unit.text,
        target_tokens=target_tokens,
        overlap_tokens=overlap_tokens,
    )
    split_units: list[LegalUnit] = []
    cursor = 0
    for text in split_texts:
        # Best-effort character offset recovery: the sentence splitter joins
        # on single spaces, which may not exactly match the original
        # whitespace/newlines, so this alignment is approximate.
        local_start = unit.text.find(text[:60], cursor)
        if local_start == -1:
            local_start = unit.text.find(text[:20], cursor)
        if local_start == -1:
            local_start = cursor
        local_end = min(len(unit.text), local_start + len(text))
        cursor = local_end
        split_units.append(
            LegalUnit(
                unit_type=f"{unit.unit_type}_part",
                char_start=unit.char_start + local_start,
                char_end=unit.char_start + local_end,
                text=text,
                clause_number=unit.clause_number,
                point_label=unit.point_label,
            )
        )
    return split_units


def tail_by_tokens(parts: list[str], token_limit: int) -> list[str]:
    if token_limit <= 0:
        return []
    selected: list[str] = []
    total = 0
    for part in reversed(parts):
        part_tokens = estimate_tokens(part)
        if selected and total + part_tokens > token_limit:
            break
        selected.append(part)
        total += part_tokens
    selected.reverse()
    return selected


def chunk_article(
    article: ArticleRecord,
    *,
    target_tokens: int = 600,
    overlap_tokens: int = 100,
) -> list[ChunkRecord]:
    """Chunk one article using legal markers before size-based fallback."""
    legal_units: list[LegalUnit] = []
    for unit in split_article_units(article, target_tokens=target_tokens):
        legal_units.extend(
            split_oversized_unit(
                unit,
                target_tokens=target_tokens,
                overlap_tokens=overlap_tokens,
            )
        )

    chunks: list[ChunkRecord] = []
    for index, unit in enumerate(legal_units):
        text = unit.text.strip()
        if not text:
            continue
        unit_path = build_unit_path(article, unit)
        features = extract_legal_features(f"{unit_path}\n{text}")
        chunks.append(
            ChunkRecord(
                chunk_id=stable_id(article.article_id, index, unit_path, text, prefix="chunk"),
                article_id=article.article_id,
                doc_id=article.doc_id,
                law_id=article.law_id,
                aid=article.aid,
                article_index=article.article_index,
                article_number=article.article_number,
                article_label=article.article_label,
                unit_type=unit.unit_type,
                unit_path=unit_path,
                clause_number=unit.clause_number,
                point_label=unit.point_label,
                source_type=article.source_type,
                deprecated=article.deprecated,
                chunk_index=len(chunks),
                char_start=unit.char_start,
                char_end=unit.char_end,
                token_count_estimate=estimate_tokens(text),
                content_sha256=sha256_text(text),
                text=text,
                ontology_concepts=features["ontology_concepts"],
                rule_signals=features["rule_signals"],
                article_references=features["article_references"],
            )
        )
    return chunks


def build_unit_path(article: ArticleRecord, unit: LegalUnit) -> str:
    parts = [article.law_id, article.article_label]
    if unit.clause_number:
        parts.append(f"khoản {unit.clause_number}")
    if unit.point_label:
        parts.append(f"điểm {unit.point_label}")
    if unit.unit_type.endswith("_part"):
        parts.append("phần dài")
    return " ".join(parts)


def build_clean_corpus(
    raw_path: Path,
    *,
    target_tokens: int = 600,
    overlap_tokens: int = 100,
) -> tuple[list[DocumentRecord], list[ArticleRecord], list[ChunkRecord], dict[str, Any]]:
    raw = read_json(raw_path)
    cleaned_at = datetime.now(timezone.utc).isoformat()

    documents: list[DocumentRecord] = []
    articles: list[ArticleRecord] = []
    chunks: list[ChunkRecord] = []

    for raw_doc in iter_raw_documents(raw):
        document, document_articles = build_document_record(
            raw_doc,
            source_path=raw_path,
            cleaned_at=cleaned_at,
        )
        documents.append(document)
        articles.extend(document_articles)
        for article in document_articles:
            chunks.extend(
                chunk_article(
                    article,
                    target_tokens=target_tokens,
                    overlap_tokens=overlap_tokens,
                )
            )

    law_ids = [document.law_id for document in documents]
    duplicate_law_ids = sorted(
        law_id for law_id, count in Counter(law_ids).items() if count > 1
    )
    chunk_unit_counts = Counter(chunk.unit_type for chunk in chunks)
    article_number_source_counts = Counter(
        article.article_number_source for article in articles
    )
    concept_counts = Counter(concept for chunk in chunks for concept in chunk.ontology_concepts)
    rule_signal_counts = Counter(signal for chunk in chunks for signal in chunk.rule_signals)
    citation_reference_count = sum(len(chunk.article_references) for chunk in chunks)

    audit = {
        "raw_path": raw_path.as_posix(),
        "cleaned_at": cleaned_at,
        "document_count": len(documents),
        "article_count": len(articles),
        "chunk_count": len(chunks),
        "deprecated_document_count": sum(document.deprecated for document in documents),
        "deprecated_article_count": sum(article.deprecated for article in articles),
        "duplicate_law_ids": duplicate_law_ids,
        "empty_article_count": sum(1 for article in articles if not article.article_text),
        "chunk_unit_counts": dict(sorted(chunk_unit_counts.items())),
        "article_number_source_counts": dict(sorted(article_number_source_counts.items())),
        "ontology_concept_counts": dict(sorted(concept_counts.items())),
        "rule_signal_counts": dict(sorted(rule_signal_counts.items())),
        "article_reference_count": citation_reference_count,
        "target_tokens": target_tokens,
        "overlap_tokens": overlap_tokens,
        "avg_article_tokens": round(
            sum(article.token_count_estimate for article in articles) / max(1, len(articles)),
            2,
        ),
        "avg_chunk_tokens": round(
            sum(chunk.token_count_estimate for chunk in chunks) / max(1, len(chunks)),
            2,
        ),
        "schema_version": SCHEMA_VERSION,
        "chunking_policy": (
            "multi_granular_pattern_based: article-level by default; "
            "clause/point fallback for oversized dense articles; sentence "
            "fallback only for oversized legal units"
        ),
        "cleaning_notes": (
            "deprecated=True is a candidate signal for human audit, not a "
            "conclusive determination; inline clause/point splitting skips "
            "cross-references (e.g. 'tại khoản 2.')."
        ),
    }

    return documents, articles, chunks, audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean the ALQAC public law corpus.")
    parser.add_argument("--raw", type=Path, default=RAW_CORPUS_PATH)
    parser.add_argument("--docs-out", type=Path, default=DOCS_OUTPUT_PATH)
    parser.add_argument("--articles-out", type=Path, default=ARTICLES_OUTPUT_PATH)
    parser.add_argument("--chunks-out", type=Path, default=CHUNKS_OUTPUT_PATH)
    parser.add_argument("--audit-out", type=Path, default=AUDIT_OUTPUT_PATH)
    parser.add_argument("--target-tokens", type=int, default=600)
    parser.add_argument("--overlap-tokens", type=int, default=100)
    args = parser.parse_args()

    documents, articles, chunks, audit = build_clean_corpus(
        args.raw,
        target_tokens=args.target_tokens,
        overlap_tokens=args.overlap_tokens,
    )

    doc_count = write_jsonl(args.docs_out, (asdict(record) for record in documents))
    article_count = write_jsonl(args.articles_out, (asdict(record) for record in articles))
    chunk_count = write_jsonl(args.chunks_out, (asdict(record) for record in chunks))
    write_json(
        args.audit_out,
        {
            **audit,
            "written": {
                "documents": doc_count,
                "articles": article_count,
                "chunks": chunk_count,
            },
        },
    )

    print(
        json.dumps(
            {
                "documents": doc_count,
                "articles": article_count,
                "chunks": chunk_count,
                "audit": args.audit_out.as_posix(),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
