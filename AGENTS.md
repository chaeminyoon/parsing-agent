# AGENTS.md

## Project

Self-healing document parsing workflow: parse → evaluate → inspect → route →
repair → finalize (LangGraph). Parsers are swappable adapters behind
`ParserRegistry`; the LLM judge and visual repair are optional and fail open.

## Commands (verified)

```bash
uv sync --extra dev          # install (Python >= 3.12 recommended; CI uses 3.12)
uv run pytest -q             # full test suite — must stay green
uvx ruff check src tests     # lint (pyproject [tool.ruff])
uv run python -m parsing_agent <input> <output_dir>   # run the workflow CLI
```

CI (`.github/workflows/ci.yml`) runs `uv sync --extra dev` + `uv run pytest -q`.

## Layout

- `src/parsing_agent/`
  - `workflow.py` — LangGraph orchestration (`WorkflowRunner`); base-parser
    routing lives in `_base_parser_name_for_source`
  - `parsers.py` — PDF adapters + text fallback + `ParserRegistry`
  - `format_parsers.py` — docx/pptx/csv/html/json/yaml structured adapters
    (stdlib OOXML parsing; suffix→adapter map `STRUCTURED_SUFFIX_PARSERS`)
  - `filetype.py` — 파일 타입 판별 단일 소스 (`is_pdf*`, `is_image*`,
    `is_text_like*`, suffix 상수). 새 포맷은 여기부터 갱신
  - `textutil.py` — 인코딩 폴백 읽기(utf-8→cp949→euc-kr)·NFC 정규화·마크다운 표 렌더
  - `ingestion.py` — `DocumentSource` 구성, `extracted_text` 채우기, OCR 게이트
  - `evaluation.py` / `judge.py` / `repair.py` / `visual_repair.py` — 품질 루프
- `tests/` — pytest; 새 파서 테스트는 가짜 `DocumentSource` + monkeypatch 패턴
  (`test_layout_first_parser.py`, `test_format_parsers.py` 참고)
- `golden/`, `benchmarks/` — 원본 PDF·API 키가 필요한 별도 프로토콜 (CI 밖)

## Conventions

- 주석·docstring은 한국어 위주, 코드 식별자는 영어.
- 파서 어댑터는 자기 suffix/media-type을 스스로 가드하고 해당 없으면 `[]` 반환
  — 예외를 던지지 않는다 (workflow 폴백 체인이 다음 파서로 넘어간다).
- 비-PDF 입력은 `extracted_text`가 반드시 채워져야 평가/수리 루프가 돈다
  (`load_document_source_text`가 없으면 ValueError).
- 새 의존성 추가는 신중히: docx/pptx는 의도적으로 stdlib로만 파싱한다.
- 커밋 전 `uv run pytest -q` + `uvx ruff check src tests` 필수.

## Do not edit

- `uv.lock` 수동 편집 금지 (`uv add`/`uv sync`가 관리).
- `golden/labels/*.json` — 사람이 라벨링한 골든셋.
