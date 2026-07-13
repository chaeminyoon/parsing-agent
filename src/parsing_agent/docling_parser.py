"""IBM Docling 어댑터 — 학습된 레이아웃 모델을 후보 풀에 편입한다.

docling은 무거운 옵셔널 의존성(`uv sync --extra bench-docling`)이다.
설치돼 있지 않으면 어댑터는 조용히 빈 목록을 반환하고, 융합은 남은
후보들로 계속 간다 — xgboost와 같은 우아한 저하 패턴.

경쟁 엔진을 이기는 대신 흡수한다: docling의 표 인식이 좋은 문서에서는
표 융합이 docling의 표를 채택하고, 나쁜 문서에서는 기존 후보를 지킨다.
심판은 언제나 PDF 자신의 괘선 그리드(TEDS-lite)다.
"""
from __future__ import annotations

import sys

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


def _pdf_format_options():
    """macOS에서는 Apple Vision OCR을 쓴다 — 한국어 래스터 그림표 실측에서
    기본 RapidOCR(중국어 모델)이 전멸("10년"→"10H")한 반면 Vision은 완벽 판독.
    ocrmac 미설치·비macOS면 None을 반환해 docling 기본으로 우아하게 저하."""
    if sys.platform != "darwin":
        return None
    try:
        import ocrmac  # noqa: F401 - 가용성 확인용
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import OcrMacOptions, PdfPipelineOptions
        from docling.document_converter import PdfFormatOption

        options = PdfPipelineOptions(ocr_options=OcrMacOptions(lang=["ko-KR", "en-US"]))
        return {InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    except Exception:  # noqa: BLE001 - OCR 백엔드 선택 실패는 기본으로 저하
        return None


def _get_converter():
    global _converter
    if _converter is None and DocumentConverter is not None:
        format_options = _pdf_format_options()
        _converter = DocumentConverter(format_options=format_options) if format_options else DocumentConverter()
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


# 문서별 그리드 캐시 — 변환이 비싸다 (모델 추론).
_grid_cache: dict[str, list] = {}


def docling_reference_grids(path) -> list[list[list[str]]]:
    """docling TableFormer가 인식한 표들을 기준 그리드 형태로 반환한다.

    괘선 없는 표·그림 표는 fitz find_tables가 못 보므로 기준 그리드가 없어
    심판 부재 상태가 된다 (골든 파일럿에서 사람이 확정한 맹점). 시각 모델의
    그리드를 그 자리에 세운다.

    주의 — 편향: docling은 융합 후보이기도 하므로 이 그리드는 docling에
    유리한 심판이다. 그래서 호출부는 괘선 그리드가 '아예 없을 때만' 쓴다
    (심판이 없는 것보다는 편향된 심판이 낫다는 실측 판단 — borderless
    골든 케이스에서 5×4 표를 정확히 복원).
    """
    key = str(path)
    if key in _grid_cache:
        return _grid_cache[key]
    grids: list[list[list[str]]] = []
    if docling_available():
        try:
            document = _get_converter().convert(key).document
            for table in document.tables:
                frame = table.export_to_dataframe(document)
                rows: list[list[str]] = []
                columns = list(frame.columns)
                # 열 이름이 실제 헤더면(0..n 정수 나열이 아니면) 첫 행으로 승격
                if not all(isinstance(c, int) for c in columns):
                    rows.append([" ".join(str(c).split()).lower() for c in columns])
                for record in frame.itertuples(index=False):
                    rows.append([" ".join(str(v).split()).lower() for v in record])
                rows = [r for r in rows if any(cell.strip() for cell in r)]
                if len(rows) >= 2 and max(len(r) for r in rows) >= 2:
                    grids.append(rows)
        except Exception:  # noqa: BLE001 - 시각 그리드 실패는 심판 부재로 돌아갈 뿐이다
            grids = []
    _grid_cache[key] = grids
    return grids
