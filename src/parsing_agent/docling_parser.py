"""IBM Docling 어댑터 — 학습된 레이아웃 모델을 후보 풀에 편입한다.

docling은 무거운 옵셔널 의존성(`uv sync --extra bench-docling`)이다.
설치돼 있지 않으면 어댑터는 조용히 빈 목록을 반환하고, 융합은 남은
후보들로 계속 간다 — xgboost와 같은 우아한 저하 패턴.

경쟁 엔진을 이기는 대신 흡수한다: docling의 표 인식이 좋은 문서에서는
표 융합이 docling의 표를 채택하고, 나쁜 문서에서는 기존 후보를 지킨다.
심판은 언제나 PDF 자신의 괘선 그리드(TEDS-lite)다.
"""
from __future__ import annotations

from parsing_agent.config import WorkflowConfig
from parsing_agent.filetype import is_pdf_source
from parsing_agent.interfaces import ParserAdapter
from parsing_agent.models import DocumentSource, ParseCandidate

try:
    from docling.document_converter import DocumentConverter
except ImportError:  # pragma: no cover - 옵셔널 의존성 가드
    DocumentConverter = None

# DocumentConverter는 모델 로딩이 비싸므로 프로세스당 한 번만 만든다.
_converter = None


def docling_available() -> bool:
    return DocumentConverter is not None


def _get_converter():
    global _converter
    if _converter is None and DocumentConverter is not None:
        _converter = DocumentConverter()
    return _converter


class DoclingPdfParserAdapter(ParserAdapter):
    name = "docling-pdf"

    def parse(self, source: DocumentSource, config: WorkflowConfig) -> list[ParseCandidate]:
        if not is_pdf_source(source) or not docling_available():
            return []
        converter = _get_converter()
        try:
            markdown = converter.convert(str(source.path)).document.export_to_markdown()
        except Exception:  # noqa: BLE001 - 옵셔널 엔진의 실패는 후보 하나가 빠질 뿐이다
            return []
        if not markdown.strip():
            return []
        return [
            ParseCandidate(
                parser_name=self.name,
                content=markdown,
                format_name=config.output_format,
                metadata={"media_type": source.media_type, "engine": "docling"},
                source_path=source.path,
            )
        ]
