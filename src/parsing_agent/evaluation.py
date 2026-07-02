from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
import re
from typing import Any

from parsing_agent.config import WorkflowConfig, WorkflowWeights
from parsing_agent.interfaces import CandidateEvaluator, CandidateJudge
from parsing_agent.models import (
    DocumentSource,
    EvaluationIssue,
    EvaluationMetrics,
    JudgeResult,
    ParseCandidate,
    load_document_source_text,
)

_WORD_RE = re.compile(r"\w+")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_LIST_RE = re.compile(r"^([-*+]|\d+\.)\s+")
_PDF_CHAPTER_CUE_RE = re.compile(r"\uC81C\s*\d+\s*\uC7A5")
_PDF_SECTION_CUE_RE = re.compile(
    r"(?<![\d|])((?:[1-9]|1\d|20)\.\d{1,2})(?=\s*[A-Za-z\uAC00-\uD7A3])"
)
_PDF_SUBSECTION_CUE_RE = re.compile(r"(?<!\w)([\uAC00-\uD558])\.\s*")
_PDF_TABLE_LABEL_RE = re.compile(
    r"(?:\uD45C|表)\s*(?:<\s*)?(\d+(?:\.\d+)*(?:-\d+)?)(?:\s*>)?",
    re.IGNORECASE,
)
TABLE_ISSUE_MISSING_HEADER = "missing_header"
TABLE_ISSUE_SPLIT_MULTIPAGE = "split_multipage_table"
TABLE_ISSUE_MERGED_CELL_LOSS = "merged_cell_loss"
TABLE_ISSUE_NUMERIC_TOKEN_BREAK = "numeric_token_break"
TABLE_ISSUE_TEXT_DUPLICATION = "table_text_duplication"
TABLE_ISSUE_TAXONOMY = (
    TABLE_ISSUE_MISSING_HEADER,
    TABLE_ISSUE_SPLIT_MULTIPAGE,
    TABLE_ISSUE_MERGED_CELL_LOSS,
    TABLE_ISSUE_NUMERIC_TOKEN_BREAK,
    TABLE_ISSUE_TEXT_DUPLICATION,
)
_GENERIC_TABLE_TOKENS = {
    "구분",
    "항목",
    "비고",
    "내용",
    "값",
    "항목명",
    "단위",
    "소계",
    "합계",
}
_GENERIC_TABLE_TOKENS_LOWER = {token.lower() for token in _GENERIC_TABLE_TOKENS}
_TABLE_ISSUE_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    TABLE_ISSUE_MISSING_HEADER: (
        re.compile(r"\bmissing(?:\s+\w+){0,2}\s+header\b", re.IGNORECASE),
        re.compile(r"\bheader\s+missing\b", re.IGNORECASE),
        re.compile(r"\bmissing\s+column\s+headers?\b", re.IGNORECASE),
        re.compile(r"머리글", re.IGNORECASE),
        re.compile(r"헤더", re.IGNORECASE),
        re.compile(r"표 형태로 렌더링되지", re.IGNORECASE),
        re.compile(r"행/열 구조", re.IGNORECASE),
        re.compile(r"구조 보존이 미흡", re.IGNORECASE),
    ),
    TABLE_ISSUE_SPLIT_MULTIPAGE: (
        re.compile(r"\bsplit(?:\s+\w+){0,2}\s+table\b", re.IGNORECASE),
        re.compile(r"\bmulti[\s-]?page\s+table\b", re.IGNORECASE),
        re.compile(r"\btable\s+(?:continues|continuation)\b", re.IGNORECASE),
        re.compile(r"\bcontinued\s+on\s+next\s+page\b", re.IGNORECASE),
        re.compile(r"다음 페이지", re.IGNORECASE),
        re.compile(r"계속", re.IGNORECASE),
    ),
    TABLE_ISSUE_MERGED_CELL_LOSS: (
        re.compile(r"\bmerged?\s+cells?\b", re.IGNORECASE),
        re.compile(r"\browspan\b", re.IGNORECASE),
        re.compile(r"\bcolspan\b", re.IGNORECASE),
        re.compile(r"\bspanning\s+cells?\b", re.IGNORECASE),
        re.compile(r"병합\s*셀", re.IGNORECASE),
        re.compile(r"열 배치가 어색", re.IGNORECASE),
    ),
    TABLE_ISSUE_NUMERIC_TOKEN_BREAK: (
        re.compile(r"\bnumeric\s+token\s+break\b", re.IGNORECASE),
        re.compile(r"\bnumber\s+token\s+break\b", re.IGNORECASE),
        re.compile(r"\bdecimal\s+split\b", re.IGNORECASE),
        re.compile(r"\bbroken\s+numbers?\b", re.IGNORECASE),
        re.compile(r"\bsplit\s+digits?\b", re.IGNORECASE),
        re.compile(r"숫자.*깨", re.IGNORECASE),
        re.compile(r"소수점", re.IGNORECASE),
        re.compile(r"단위.*오", re.IGNORECASE),
        re.compile(r"값이 비어", re.IGNORECASE),
        re.compile(r"누락된\s*셀", re.IGNORECASE),
        re.compile(r"누락된\s*값", re.IGNORECASE),
    ),
    TABLE_ISSUE_TEXT_DUPLICATION: (
        re.compile(r"\btable\s+text\s+duplication\b", re.IGNORECASE),
        re.compile(r"\bduplicated?\s+table\s+text\b", re.IGNORECASE),
        re.compile(r"\brepeated\s+table\s+text\b", re.IGNORECASE),
        re.compile(r"\btable\s+text\s+repeated\b", re.IGNORECASE),
        re.compile(r"중복", re.IGNORECASE),
        re.compile(r"반복", re.IGNORECASE),
    ),
}


def _clamp(score: float) -> float:
    return max(0.0, min(1.0, score))


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _non_empty_lines(text: str) -> list[str]:
    return [line for line in (part.rstrip() for part in text.splitlines()) if line.strip()]


def calculate_text_coverage(source_text: str, candidate_text: str) -> float:
    """원문 대비 후보 텍스트의 토큰 보존율을 계산한다.

    source 쪽 단어 토큰을 기준으로 candidate 안에서 몇 개가 다시
    등장하는지 본다. 순서 변경 자체보다는 "빠진 단어가 얼마나 많은가"를
    강하게 보는 지표다.
    """
    source_tokens = Counter(_WORD_RE.findall(source_text.lower()))
    if not source_tokens:
        return 1.0 if not candidate_text.strip() else 0.0
    candidate_tokens = Counter(_WORD_RE.findall(candidate_text.lower()))
    matched_tokens = sum(min(count, candidate_tokens[token]) for token, count in source_tokens.items())
    return matched_tokens / sum(source_tokens.values())


def calculate_normalized_similarity(source_text: str, candidate_text: str) -> float:
    """정규화된 전체 문서 문자열 유사도를 계산한다.

    공백과 대소문자를 정리한 뒤 `SequenceMatcher`로 비교한다. 단순 토큰
    보존율보다 더 엄격해서, 문장 순서 변경이나 큰 폭의 재작성도 점수에
    반영된다.
    """
    normalized_source = _normalize_text(source_text)
    normalized_candidate = _normalize_text(candidate_text)
    if not normalized_source and not normalized_candidate:
        return 1.0
    if not normalized_source or not normalized_candidate:
        return 0.0
    return SequenceMatcher(None, normalized_source, normalized_candidate).ratio()


def _line_kind(line: str) -> str | None:
    stripped = line.strip()
    if not stripped:
        return None
    if stripped.startswith("#"):
        return "heading"
    if _LIST_RE.match(stripped):
        return "list"
    if _looks_like_table_line(stripped):
        return "table"
    return None


def _looks_like_table_line(line: str) -> bool:
    stripped = line.strip().lower()
    return (line.count("|") >= 2 and (line.startswith("|") or line.endswith("|"))) or stripped.startswith("<table")


def _count_structural_lines(text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for line in text.splitlines():
        kind = _line_kind(line)
        if kind is not None:
            counts[kind] += 1
    return counts


def _count_similarity(expected: int, actual: int) -> float:
    if expected == 0:
        return 1.0 if actual == 0 else 0.0
    return _clamp(1.0 - (abs(expected - actual) / expected))


def _count_paragraphs(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return len([part for part in re.split(r"\n\s*\n+", stripped) if part.strip()])


def _calculate_plain_text_structure_similarity(source_text: str, candidate_text: str) -> float:
    source_lines = _non_empty_lines(source_text)
    candidate_lines = _non_empty_lines(candidate_text)

    if not source_lines:
        return 1.0 if not candidate_lines else 0.0

    line_count_score = _count_similarity(len(source_lines), len(candidate_lines))
    paragraph_count_score = _count_similarity(_count_paragraphs(source_text), _count_paragraphs(candidate_text))
    return (line_count_score + paragraph_count_score) / 2


def calculate_structure_retention(source_text: str, candidate_text: str) -> float:
    """문서 구조가 얼마나 유지됐는지 점수화한다.

    일반 텍스트에서는 heading, list, table 개수를 비교한다. 이런 구조
    마커가 원문에 거의 없으면 문단 수와 비어 있지 않은 줄 수를 비교하는
    완화된 방식으로 계산한다.
    """
    source_counts = _count_structural_lines(source_text)
    candidate_counts = _count_structural_lines(candidate_text)
    kinds = ("heading", "list", "table")
    relevant_kinds = [kind for kind in kinds if source_counts[kind] > 0]
    if not relevant_kinds:
        return _calculate_plain_text_structure_similarity(source_text, candidate_text)
    scores = [_count_similarity(source_counts[kind], candidate_counts[kind]) for kind in relevant_kinds]
    return sum(scores) / len(scores)


def calculate_table_preservation(source_text: str, candidate_text: str) -> float:
    """표 형태 블록이 얼마나 보존됐는지 계산한다.

    PDF가 아닌 일반 문서에서는 표처럼 보이는 블록 수를 비교하는 단순한
    구조 지표다. PDF는 아래의 전용 표 보존 계산 함수를 사용한다.
    """
    source_tables = _count_structural_lines(source_text)["table"]
    candidate_tables = _count_structural_lines(candidate_text)["table"]
    return _count_similarity(source_tables, candidate_tables)


def _is_pdf_source(source: DocumentSource) -> bool:
    return source.media_type == "application/pdf" or source.path.suffix.lower() == ".pdf"


def _extract_pdf_structure_cues(text: str) -> Counter[str]:
    cues: Counter[str] = Counter()
    flattened_text = " ".join(_non_empty_lines(text))
    for chapter_match in _PDF_CHAPTER_CUE_RE.finditer(flattened_text):
        cues[f"chapter:{_normalize_text(chapter_match.group(0))}"] += 1
    for section_match in _PDF_SECTION_CUE_RE.finditer(flattened_text):
        cues[f"section:{section_match.group(1)}"] += 1
    for subsection_match in _PDF_SUBSECTION_CUE_RE.finditer(flattened_text):
        cues[f"subsection:{subsection_match.group(1)}"] += 1
    return cues


def _calculate_pdf_structure_retention(source_text: str, candidate_text: str) -> float:
    """PDF 문서의 장/절 번호 cue를 기준으로 구조 보존율을 계산한다.

    PDF 원문 텍스트는 markdown heading이 없을 수 있으므로, 먼저 장/절/소절
    번호 패턴을 찾는다. 그런 cue가 없을 때만 일반 구조 보존 계산으로
    fallback 한다.
    """
    source_cues = _extract_pdf_structure_cues(source_text)
    if not source_cues:
        return calculate_structure_retention(source_text, candidate_text)
    candidate_cues = _extract_pdf_structure_cues(candidate_text)
    matched_cues = sum(min(count, candidate_cues[cue]) for cue, count in source_cues.items())
    return matched_cues / sum(source_cues.values())


def _extract_pdf_table_label_ids(text: str) -> list[str]:
    label_ids: list[str] = []
    seen: set[str] = set()
    for match in _PDF_TABLE_LABEL_RE.finditer(text):
        label_id = match.group(1)
        if label_id in seen:
            continue
        seen.add(label_id)
        label_ids.append(label_id)
    return label_ids


def extract_table_issue_types(issue_text: str) -> list[str]:
    normalized_issue = str(issue_text or "").strip()
    if not normalized_issue:
        return []
    matches: list[str] = []
    for issue_type in TABLE_ISSUE_TAXONOMY:
        if any(pattern.search(normalized_issue) for pattern in _TABLE_ISSUE_PATTERNS[issue_type]):
            matches.append(issue_type)
    return matches


def _candidate_table_regions(candidate: ParseCandidate) -> list[dict[str, Any]]:
    raw_regions = candidate.metadata.get("table_regions")
    if isinstance(raw_regions, list):
        regions = [region for region in raw_regions if isinstance(region, dict)]
        if regions:
            return regions

    support_metadata = candidate.metadata.get("support_parser_metadata")
    if not isinstance(support_metadata, dict):
        return []
    for metadata in support_metadata.values():
        if not isinstance(metadata, dict):
            continue
        raw_regions = metadata.get("table_regions")
        if not isinstance(raw_regions, list):
            continue
        regions = [region for region in raw_regions if isinstance(region, dict)]
        if regions:
            return regions
    return []


def _has_repeated_table_block(content: str) -> bool:
    normalized_blocks: Counter[str] = Counter()
    for table_lines in _extract_markdown_table_blocks(content):
        normalized = _normalize_text("\n".join(table_lines))
        if normalized:
            normalized_blocks[normalized] += 1
    return any(count > 1 for count in normalized_blocks.values())


def classify_table_issues(source: DocumentSource, candidate: ParseCandidate, judge_result: JudgeResult | None) -> list[str]:
    """judge 결과와 parser metadata를 표 이슈 타입으로 정규화한다.

    judge는 "헤더 누락", "셀 병합 손실" 같은 의미적 힌트를 주고,
    candidate metadata는 "다음 페이지로 이어진 표" 같은 결정적 단서를
    제공한다. 둘을 합쳐 표 수리에 필요한 이슈 타입 목록을 만든다.
    """
    if not _is_pdf_source(source):
        return []

    detected: list[str] = []
    seen: set[str] = set()

    def add(issue_type: str) -> None:
        if issue_type in seen:
            return
        seen.add(issue_type)
        detected.append(issue_type)

    if judge_result is not None:
        for finding in judge_result.table_findings:
            issue_type = finding.get("issue_type")
            if isinstance(issue_type, str) and issue_type in TABLE_ISSUE_TAXONOMY:
                if issue_type == TABLE_ISSUE_TEXT_DUPLICATION and not _has_repeated_table_block(candidate.content):
                    continue
                add(issue_type)
        for issue in judge_result.issues:
            for issue_type in extract_table_issue_types(issue):
                if issue_type == TABLE_ISSUE_TEXT_DUPLICATION and not _has_repeated_table_block(candidate.content):
                    continue
                add(issue_type)

    for table_region in _candidate_table_regions(candidate):
        if table_region.get("continued_from_page") is not None:
            add(TABLE_ISSUE_SPLIT_MULTIPAGE)
            break

    return detected


def _build_pdf_table_label_pattern(label_id: str) -> re.Pattern[str]:
    return re.compile(
        rf"(?:\uD45C|表)\s*(?:<\s*)?{re.escape(label_id)}(?:\s*>)?",
        re.IGNORECASE,
    )


def _find_label_line_index(lines: list[str], label_id: str) -> int | None:
    label_pattern = _build_pdf_table_label_pattern(label_id)
    for index, line in enumerate(lines):
        if label_pattern.search(line):
            return index
    return None


def _extract_nearby_markdown_tables(
    lines: list[str],
    label_line_index: int,
    window: int = 12,
) -> list[tuple[int, list[str]]]:
    upper_bound = min(len(lines), label_line_index + window + 1)
    tables: list[tuple[int, list[str]]] = []
    index = label_line_index + 1
    while index < upper_bound:
        if not _looks_like_table_line(lines[index].strip()):
            index += 1
            continue
        start_index = index
        table_lines: list[str] = []
        if lines[index].strip().lower().startswith("<table"):
            while index < upper_bound:
                table_lines.append(lines[index])
                if "</table>" in lines[index].lower():
                    index += 1
                    break
                index += 1
        else:
            while index < upper_bound and _looks_like_table_line(lines[index].strip()):
                table_lines.append(lines[index])
                index += 1
        if table_lines:
            tables.append((start_index, table_lines))
    return tables


def _extract_source_table_context(text: str, label_id: str, window: int = 240) -> str:
    label_pattern = _build_pdf_table_label_pattern(label_id)
    if match := label_pattern.search(text):
        return text[match.end() : match.end() + window]
    return ""


def _extract_table_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for token in _WORD_RE.findall(text):
        normalized = token.lower()
        if normalized.isdigit() or normalized in _GENERIC_TABLE_TOKENS_LOWER:
            continue
        tokens.add(normalized)
    return tokens


def _extract_table_numbers(text: str) -> set[str]:
    return set(_NUMBER_RE.findall(text))


def _score_markdown_table_match(source_context: str, table_lines: list[str]) -> float:
    if len(table_lines) < 2:
        return 0.0

    if any("<table" in line.lower() for line in table_lines):
        return _score_html_table_match(source_context, table_lines)

    header_line = table_lines[0]
    body_text = "\n".join(table_lines[2:]) if len(table_lines) > 2 else ""
    table_text = "\n".join(table_lines)
    source_tokens = _extract_table_tokens(source_context)
    header_tokens = _extract_table_tokens(header_line)
    body_tokens = _extract_table_tokens(body_text)
    source_numbers = _extract_table_numbers(source_context)
    candidate_numbers = _extract_table_numbers(table_text)

    header_score = 1.0 if source_tokens & header_tokens else 0.0
    body_score = 1.0 if source_tokens & body_tokens else 0.0
    number_score = 1.0 if source_numbers & candidate_numbers else 0.0
    shape_score = 1.0
    return (header_score + body_score + number_score + shape_score) / 4


def _score_html_table_match(source_context: str, table_lines: list[str]) -> float:
    table_text = "\n".join(table_lines)
    source_tokens = _extract_table_tokens(source_context)
    candidate_tokens = _extract_table_tokens(table_text)
    source_numbers = _extract_table_numbers(source_context)
    candidate_numbers = _extract_table_numbers(table_text)

    component_scores = [1.0]
    if source_tokens:
        component_scores.append(len(source_tokens & candidate_tokens) / len(source_tokens))
    if source_numbers:
        component_scores.append(len(source_numbers & candidate_numbers) / len(source_numbers))
    return sum(component_scores) / len(component_scores)


def _count_markdown_tables(text: str) -> int:
    return len(_extract_markdown_table_blocks(text))


def _extract_markdown_table_blocks(text: str) -> list[list[str]]:
    lines = text.splitlines()
    tables: list[list[str]] = []
    index = 0
    while index < len(lines):
        if not _looks_like_table_line(lines[index].strip()):
            index += 1
            continue
        table_lines: list[str] = []
        if lines[index].strip().lower().startswith("<table"):
            while index < len(lines):
                table_lines.append(lines[index])
                if "</table>" in lines[index].lower():
                    index += 1
                    break
                index += 1
        else:
            while index < len(lines) and _looks_like_table_line(lines[index].strip()):
                table_lines.append(lines[index])
                index += 1
        tables.append(table_lines)
    return tables


def _looks_like_unlabeled_pdf_table_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    number_count = len(_NUMBER_RE.findall(stripped))
    token_count = len(_WORD_RE.findall(stripped))
    return number_count >= 2 or (number_count >= 1 and token_count >= 4)


def _count_unlabeled_pdf_table_blocks(text: str) -> int:
    return len(_extract_unlabeled_pdf_table_blocks(text))


def _extract_unlabeled_pdf_table_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    current_block: list[str] = []
    previous_non_empty_line: str | None = None
    for line in text.splitlines():
        if _looks_like_unlabeled_pdf_table_line(line):
            if not current_block and previous_non_empty_line is not None:
                previous_number_count = len(_NUMBER_RE.findall(previous_non_empty_line))
                previous_token_count = len(_WORD_RE.findall(previous_non_empty_line))
                if previous_number_count == 0 and previous_token_count >= 2:
                    current_block.append(previous_non_empty_line)
            current_block.append(line)
            continue
        if current_block:
            blocks.append("\n".join(current_block))
            current_block = []
        previous_non_empty_line = line if line.strip() else None
    if current_block:
        blocks.append("\n".join(current_block))
    return blocks


def _calculate_pdf_unlabeled_table_preservation(source_text: str, candidate_text: str) -> float:
    source_blocks = _extract_unlabeled_pdf_table_blocks(source_text)
    if not source_blocks:
        return calculate_table_preservation(source_text, candidate_text)

    candidate_tables = _extract_markdown_table_blocks(candidate_text)
    count_score = _count_similarity(len(source_blocks), len(candidate_tables))
    if not candidate_tables:
        return count_score / 2

    used_table_indexes: set[int] = set()
    matched_score = 0.0
    for source_block in source_blocks:
        best_index: int | None = None
        best_score = 0.0
        for index, table_lines in enumerate(candidate_tables):
            if index in used_table_indexes:
                continue
            score = _score_unlabeled_markdown_table_match(source_block, table_lines)
            if score > best_score:
                best_score = score
                best_index = index
        if best_index is not None and best_score > 0:
            used_table_indexes.add(best_index)
        matched_score += best_score
    content_score = matched_score / len(source_blocks)
    return (count_score + content_score) / 2


def _score_unlabeled_markdown_table_match(source_block: str, table_lines: list[str]) -> float:
    table_text = "\n".join(table_lines)
    source_tokens = _extract_table_tokens(source_block)
    candidate_tokens = _extract_table_tokens(table_text)
    source_numbers = _extract_table_numbers(source_block)
    candidate_numbers = _extract_table_numbers(table_text)

    component_scores: list[float] = []
    if source_tokens:
        component_scores.append(len(source_tokens & candidate_tokens) / len(source_tokens))
    if source_numbers:
        component_scores.append(len(source_numbers & candidate_numbers) / len(source_numbers))
    if not component_scores:
        return 1.0 if len(table_lines) >= 2 else 0.0
    return sum(component_scores) / len(component_scores)


def _calculate_pdf_table_preservation(source_text: str, candidate_text: str) -> float:
    """PDF 표 보존율을 표 라벨과 주변 표 블록 매칭으로 계산한다.

    원문에 `표 4.2-2` 같은 라벨이 있으면, candidate도 그 라벨을 유지하고
    근처에 실제로 쓸 수 있는 표 블록을 만들어야 점수를 얻는다. 라벨이
    없으면 비라벨 표 블록 heuristic으로 fallback 한다.
    """
    source_label_ids = _extract_pdf_table_label_ids(source_text)
    if not source_label_ids:
        return _calculate_pdf_unlabeled_table_preservation(source_text, candidate_text)

    candidate_lines = candidate_text.splitlines()
    represented_labels = 0
    usable_tables = 0.0
    used_table_starts: set[int] = set()

    for label_id in source_label_ids:
        label_line_index = _find_label_line_index(candidate_lines, label_id)
        if label_line_index is None:
            continue
        represented_labels += 1
        source_context = _extract_source_table_context(source_text, label_id)
        best_table_start: int | None = None
        best_match_score = 0.0
        for table_start, table_lines in _extract_nearby_markdown_tables(candidate_lines, label_line_index):
            if table_start in used_table_starts:
                continue
            score = _score_markdown_table_match(source_context, table_lines)
            if score > best_match_score:
                best_match_score = score
                best_table_start = table_start
        if best_table_start is not None and best_match_score > 0:
            used_table_starts.add(best_table_start)
        usable_tables += best_match_score

    label_score = represented_labels / len(source_label_ids)
    usable_table_score = usable_tables / len(source_label_ids)
    return (label_score + usable_table_score) / 2


def calculate_empty_block_penalty(candidate_text: str) -> float:
    """과도한 빈 줄 구간에 대한 패널티를 계산한다.

    연속된 빈 줄에서 첫 줄을 제외한 나머지를 노이즈로 보고, 전체 줄 수로
    정규화해 `[0, 1]` 범위의 패널티 값으로 만든다.
    """
    total_lines = max(len(candidate_text.splitlines()), 1)
    extra_blank_lines = 0
    blank_run = 0

    for line in candidate_text.splitlines():
        if line.strip():
            if blank_run > 1:
                extra_blank_lines += blank_run - 1
            blank_run = 0
            continue
        blank_run += 1

    if blank_run > 1:
        extra_blank_lines += blank_run - 1

    return _clamp(extra_blank_lines / total_lines)


def calculate_repetition_penalty(candidate_text: str) -> float:
    """중복된 비어 있지 않은 줄에 대한 패널티를 계산한다.

    정규화 후 같은 줄이 반복되면 패널티를 준다. 주로 header/footer 중복,
    표 조각 반복 같은 parser 실패 패턴을 잡기 위한 지표다.
    """
    normalized_lines = [_normalize_text(line) for line in _non_empty_lines(candidate_text)]
    if not normalized_lines:
        return 0.0
    repeated_lines = sum(count - 1 for count in Counter(normalized_lines).values() if count > 1)
    return _clamp(repeated_lines / len(normalized_lines))


def aggregate_score(
    metrics: EvaluationMetrics,
    weights: WorkflowWeights,
    judge_weight: float = 0.0,
) -> float:
    """긍정 지표, 패널티, 선택적 judge 점수를 합쳐 최종 점수를 만든다.

    기본 점수는 coverage/similarity/structure/table 가중합에서
    blank/repetition 패널티 평균을 뺀 값이다. judge가 있으면 그 점수를
    `judge_weight` 비율로 선형 혼합한다.
    """
    positive_score = (
        metrics.text_coverage * weights.text_coverage
        + metrics.normalized_similarity * weights.normalized_similarity
        + metrics.structure_retention * weights.structure_retention
        + metrics.table_preservation * weights.table_preservation
    )
    penalty_score = (metrics.empty_block_penalty + metrics.repetition_penalty) / 2
    base_score = _clamp(positive_score - penalty_score)
    judge_score = metrics.llm_judge_score
    if judge_score is None and metrics.judge_result is not None:
        judge_score = metrics.judge_result.overall_score
    if judge_score is None or judge_weight <= 0:
        return base_score
    effective_weight = _clamp(judge_weight)
    return _clamp((base_score * (1 - effective_weight)) + (judge_score * effective_weight))


def _build_notes(metrics: EvaluationMetrics) -> list[str]:
    notes: list[str] = []
    if metrics.text_coverage < 0.75:
        notes.append("Low text coverage against source.")
    if metrics.structure_retention < 0.75:
        notes.append("Structural markers were not retained well.")
    if metrics.table_preservation < 0.75:
        notes.append("Table formatting appears incomplete.")
    if metrics.empty_block_penalty > 0:
        notes.append("Candidate contains oversized blank sections.")
    if metrics.repetition_penalty > 0:
        notes.append("Candidate contains repeated non-empty lines.")
    return notes


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _coerce_table_findings(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    findings: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        finding: dict[str, Any] = {}
        issue_type = item.get("issue_type")
        if issue_type is not None:
            finding["issue_type"] = str(issue_type)
        table_label = item.get("table_label")
        if table_label is not None:
            finding["table_label"] = str(table_label)
        page_number = item.get("page_number")
        if page_number is not None:
            try:
                finding["page_number"] = int(page_number)
            except (TypeError, ValueError):
                pass
        if finding:
            findings.append(finding)
    return findings


def _coerce_judge_score(value: Any) -> float | None:
    if value is None:
        return None
    return _clamp(float(value))


def _coerce_judge_result(raw_result: Any) -> JudgeResult:
    if isinstance(raw_result, JudgeResult):
        return raw_result
    overall_score = _coerce_judge_score(getattr(raw_result, "overall_score", None))
    if overall_score is None:
        overall_score = _clamp(float(getattr(raw_result, "score")))
    metadata = getattr(raw_result, "metadata", {}) or {}
    return JudgeResult(
        overall_score=overall_score,
        coverage_score=_coerce_judge_score(getattr(raw_result, "coverage_score", None)),
        structure_score=_coerce_judge_score(getattr(raw_result, "structure_score", None)),
        table_score=_coerce_judge_score(getattr(raw_result, "table_score", None)),
        hallucination_risk=_coerce_judge_score(getattr(raw_result, "hallucination_risk", None)),
        editorial_readiness=_coerce_judge_score(getattr(raw_result, "editorial_readiness", None)),
        notes=_coerce_string_list(getattr(raw_result, "notes", [])),
        issues=_coerce_string_list(getattr(raw_result, "issues", [])),
        table_findings=_coerce_table_findings(
            getattr(raw_result, "table_findings", None)
            if getattr(raw_result, "table_findings", None) is not None
            else metadata.get("table_findings")
        ),
        metadata=dict(metadata),
    )


def _issue_severity(score: float, *, high_below: float = 0.55, medium_below: float = 0.75) -> str:
    if score < high_below:
        return "high"
    if score < medium_below:
        return "medium"
    return "low"


def _excerpt_for_issue(text: str, limit: int = 240) -> str | None:
    stripped = " ".join(text.split())
    if not stripped:
        return None
    if len(stripped) <= limit:
        return stripped
    return f"{stripped[: limit - 3]}..."


def _metric_confidence(score: float) -> float:
    return _clamp(1.0 - score)


def _build_metric_issues(
    *,
    source_text: str,
    candidate_text: str,
    metrics: EvaluationMetrics,
) -> list[EvaluationIssue]:
    issues: list[EvaluationIssue] = []
    source_excerpt = _excerpt_for_issue(source_text)
    candidate_excerpt = _excerpt_for_issue(candidate_text)

    if metrics.text_coverage < 0.75:
        issues.append(
            EvaluationIssue(
                issue_type="text_coverage_missing_lines",
                metric_name="text_coverage",
                severity=_issue_severity(metrics.text_coverage),
                confidence=_metric_confidence(metrics.text_coverage),
                description="Candidate is missing source text tokens or structured lines.",
                source_excerpt=source_excerpt,
                candidate_excerpt=candidate_excerpt,
                repairability="heuristic",
            )
        )
    if metrics.structure_retention < 0.75:
        issues.append(
            EvaluationIssue(
                issue_type="structure_heading_noise",
                metric_name="structure_retention",
                severity=_issue_severity(metrics.structure_retention),
                confidence=_metric_confidence(metrics.structure_retention),
                description="Candidate structure markers differ from the source.",
                source_excerpt=source_excerpt,
                candidate_excerpt=candidate_excerpt,
                repairability="heuristic",
            )
        )
    if metrics.table_preservation < 0.75:
        issues.append(
            EvaluationIssue(
                issue_type="table_layout_noise",
                metric_name="table_preservation",
                severity=_issue_severity(metrics.table_preservation),
                confidence=_metric_confidence(metrics.table_preservation),
                description="Candidate table structure appears incomplete or damaged.",
                source_excerpt=source_excerpt,
                candidate_excerpt=candidate_excerpt,
                repairability="visual",
            )
        )
    if metrics.empty_block_penalty > 0:
        issues.append(
            EvaluationIssue(
                issue_type="blank_line_noise",
                metric_name="empty_block_penalty",
                severity="medium" if metrics.empty_block_penalty >= 0.1 else "low",
                confidence=_clamp(metrics.empty_block_penalty),
                description="Candidate contains oversized blank-line runs.",
                candidate_excerpt=candidate_excerpt,
                repairability="heuristic",
            )
        )
    if metrics.repetition_penalty > 0:
        issues.append(
            EvaluationIssue(
                issue_type="line_repetition_noise",
                metric_name="repetition_penalty",
                severity="medium" if metrics.repetition_penalty >= 0.1 else "low",
                confidence=_clamp(metrics.repetition_penalty),
                description="Candidate contains repeated non-empty lines.",
                candidate_excerpt=candidate_excerpt,
                repairability="heuristic",
            )
        )
    return issues


def _build_table_finding_issues(
    *,
    candidate_text: str,
    judge_result: JudgeResult | None,
    table_issues: list[str],
) -> list[EvaluationIssue]:
    issues: list[EvaluationIssue] = []
    candidate_excerpt = _excerpt_for_issue(candidate_text)
    seen: set[tuple[str, str | None, int | None]] = set()

    if judge_result is not None:
        for finding in judge_result.table_findings:
            issue_type = str(finding.get("issue_type") or "").strip()
            if not issue_type:
                continue
            table_label = finding.get("table_label")
            page_number = finding.get("page_number")
            key = (
                issue_type,
                table_label if isinstance(table_label, str) else None,
                page_number if isinstance(page_number, int) else None,
            )
            if key in seen:
                continue
            seen.add(key)
            issues.append(
                EvaluationIssue(
                    issue_type=issue_type,
                    metric_name="table_preservation",
                    severity="high",
                    confidence=0.85,
                    description=f"Judge identified table issue: {issue_type}.",
                    candidate_excerpt=candidate_excerpt,
                    page_number=key[2],
                    table_label=key[1],
                    repairability="visual",
                )
            )

    for issue_type in table_issues:
        key = (issue_type, None, None)
        if key in seen:
            continue
        seen.add(key)
        issues.append(
            EvaluationIssue(
                issue_type=issue_type,
                metric_name="table_preservation",
                severity="medium",
                confidence=0.65,
                description=f"Table issue detected by deterministic evidence: {issue_type}.",
                candidate_excerpt=candidate_excerpt,
                repairability="visual",
            )
        )
    return issues


class DeterministicEvaluator(CandidateEvaluator):
    def __init__(self, config: WorkflowConfig, judge: CandidateJudge | None = None) -> None:
        self._config = config
        self._judge = judge

    def evaluate(self, source: DocumentSource, candidate: ParseCandidate) -> EvaluationMetrics:
        """candidate 하나를 평가해서 workflow가 쓰는 최종 metric을 만든다.

        source text를 불러와 핵심 지표를 계산하고, notes와 table issue를
        만든 뒤, 필요하면 LLM judge 결과를 반영한다. 마지막으로 모든 값을
        합쳐 workflow가 수리 여부를 판단할 총점을 계산한다.
        """
        source_text = load_document_source_text(source)
        candidate_text = candidate.content
        if _is_pdf_source(source):
            structure_retention = _calculate_pdf_structure_retention(source_text, candidate_text)
            table_preservation = _calculate_pdf_table_preservation(source_text, candidate_text)
        else:
            structure_retention = calculate_structure_retention(source_text, candidate_text)
            table_preservation = calculate_table_preservation(source_text, candidate_text)
        metrics = EvaluationMetrics(
            text_coverage=calculate_text_coverage(source_text, candidate_text),
            normalized_similarity=calculate_normalized_similarity(source_text, candidate_text),
            structure_retention=structure_retention,
            table_preservation=table_preservation,
            empty_block_penalty=calculate_empty_block_penalty(candidate_text),
            repetition_penalty=calculate_repetition_penalty(candidate_text),
            total_score=0.0,
        )
        metrics.notes.extend(_build_notes(metrics))
        metrics.table_issues = classify_table_issues(source, candidate, None)
        metrics.issues.extend(
            _build_metric_issues(
                source_text=source_text,
                candidate_text=candidate_text,
                metrics=metrics,
            )
        )
        if self._judge is not None:
            judge_result = _coerce_judge_result(self._judge.judge(source, candidate, metrics))
            metrics.judge_result = judge_result
            metrics.llm_judge_score = judge_result.overall_score
            metrics.table_issues = classify_table_issues(source, candidate, judge_result)
            metrics.notes.extend(judge_result.notes)
            metrics.notes.extend(f"Judge issue: {issue}" for issue in judge_result.issues)
            if metrics.table_issues:
                metrics.notes.append(f"Table issues: {', '.join(metrics.table_issues)}")
            metrics.issues.extend(
                _build_table_finding_issues(
                    candidate_text=candidate_text,
                    judge_result=judge_result,
                    table_issues=metrics.table_issues,
                )
            )
        elif metrics.table_issues:
            metrics.issues.extend(
                _build_table_finding_issues(
                    candidate_text=candidate_text,
                    judge_result=None,
                    table_issues=metrics.table_issues,
                )
            )
        metrics.total_score = aggregate_score(metrics, self._config.weights, self._config.judge_weight)
        return metrics
