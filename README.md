<h1 align="center">parsing-agent</h1>

<p align="center">파싱 결과를 스스로 채점하고, 고칠 가치가 있는 것만 고치고, 망치면 되돌리는 PDF 파싱 루프</p>

<p align="center">
  <a href="https://github.com/chaeminyoon/parsing-agent/actions/workflows/ci.yml"><img src="https://github.com/chaeminyoon/parsing-agent/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/Python-3.12%2B-blue.svg" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/LangGraph-state%20machine-green.svg" alt="LangGraph">
</p>

---

기존 파서(opendataloader, PyMuPDF 계열)에 한국 환경영향평가 보고서를 넣어보니 표가 절반쯤 깨져 나왔다. 파서를 바꿔도 깨지는 위치만 달라졌다. 그래서 파서를 고르는 대신, 파싱 결과가 깨졌다는 걸 알아채고 고치는 쪽을 만들었다.

50~200페이지짜리 정부 문서가 대상이다. 병합셀, 다중 페이지 표, 스캔 페이지가 섞여 있는 부류.

## 특징

- 결정적 메트릭 + 멀티모달 LLM judge로 품질 게이트를 통과할 때까지 수리 루프를 돈다 (최대 3라운드)
- 수리 전략이 3단계다: 무료 휴리스틱 → 정체된 이슈만 LLM 승격 → 깨진 표는 비전 모델로 재구성
- 수리마다 재채점하고, 점수가 떨어지면 롤백한다. 최종 결과가 수리 전보다 나빠질 수 없다
- 노드 사이에는 enum과 수치만 오간다. judge의 문장은 사람용 리포트에만 남는다
- 모든 실패에 사유가 남는다: `low_confidence(0.2)`, `patch_target_not_found`, `recover_exception: TimeoutError`
- LLM 비용이 스테이지별로 집계된다 (호출 수, 재시도, 소요시간, 토큰)
- 한국어 특화: 종결 어미 기반 문장 병합, 한국어 표 라벨 매칭
- API 키가 없으면 judge와 LLM 수리 없이 결정적 메트릭만으로 동작한다

## 설치

Python 3.12+와 [uv](https://docs.astral.sh/uv/)가 필요하다.

```bash
git clone https://github.com/chaeminyoon/parsing-agent
cd parsing-agent
uv sync
```

## 사용법

```bash
export OPENAI_API_KEY=sk-...   # 선택. 없어도 동작한다

uv run python -m parsing_agent.cli "문서.pdf" --output-dir outputs/run-1
```

문서마다 두 파일이 나온다.

```
outputs/run-1/문서/
├── 문서.md      # 수리된 마크다운
└── 문서.json    # 의사결정 기록 전체
```

JSON 리포트에는 라운드별 점수 궤적, 진단된 이슈, 수리 계획과 스킵 사유, 롤백 이벤트, 비전 수리 거부 사유, LLM 사용량이 담긴다. 설정은 `PARSING_AGENT_*` 환경변수로 조정한다. `config.py` 참고.

```bash
uv run pytest    # 190개, 1초 미만
```

## 동작 방식

```mermaid
graph LR
    A[parse] --> B[evaluate]
    B -->|게이트 실패| C[inspect]
    B -->|통과| F[finalize]
    C --> D[route]
    D -->|수리 계획| E[repair]
    D -->|고칠 게 없음| F
    E -->|재평가| B
```

parse가 마크다운 후보를 만들고, evaluate가 채점한다. 게이트(기본 0.7)를 못 넘으면 inspect가 깨진 지점을 진단하고, route가 전략과 비용을 판단해서 repair가 실행한다. 다시 evaluate로 돌아간다.

evaluate에는 함정이 하나 있는데, 수리 후 점수가 이전보다 떨어지는 경우다. 이때는 최고 점수 후보로 되돌리고 `rollback_events`에 기록한다. 실문서에서 실제로 라운드당 한두 번씩 발동한다.

노드 간 계약은 전부 구조화된 값이다. 처음에는 judge가 내려주는 문장을 정규식으로 파싱해서 라우팅했는데, "반복"이라는 단어 하나에 엉뚱한 수리가 발동하는 걸 보고 뜯어냈다. 지금은 judge가 taxonomy enum(`table_findings`)으로만 수리를 요청할 수 있고, 자유 문장은 기계 판단에 쓰이지 않는다.

| 노드 | 내보내는 것 |
|---|---|
| parse | `candidate`, `parse_errors` |
| evaluate | `metrics` (점수, `table_issues` enum, `table_cell_similarity`), `rollback_events` |
| inspect | `repair_targets` (`issue_type`, `route_name`, `severity`, `confidence`) |
| route | `repair_plan` (`strategy`, `expected_gain`, `estimated_cost`, `skip_reason`) |
| repair | `repairs`, `attempted_repair_routes`, `visual_repair_rejections` |

## 수리 전략

| 상황 | 전략 | 비용 |
|---|---|---|
| 중복 줄, 빈 줄, 잘린 문장 | 휴리스틱 | 무료 |
| 휴리스틱이 시도했지만 점수 정체 | LLM 텍스트 수리로 승격 | LLM 1회/이슈 |
| 본문 누락 (커버리지 < 0.72) | LLM 텍스트 수리 직행 | LLM 1회/이슈 |
| 표 파손 (병합셀, 다중 페이지, 헤더 누락) | 비전 표 재구성 | vision 1회/표 |

LLM 수리는 문제 구간을 라인 윈도우로 잘라 원문 근거와 함께 보낸다. confidence 임계값과 길이 제한을 걸고, 모델이 확신 없으면 그대로 반환하게 했다. 비전 수리는 원본 페이지를 crop해서 보내는데, 다중 페이지 표면 다음 페이지 상단도 같이 보낸다. 같은 표를 같은 이슈로 두 번 시도하지 않는다.

## 프레임워크 구성

```mermaid
graph TB
    subgraph 오케스트레이션
        LG[LangGraph StateGraph<br/>조건부 엣지 루프, TypedDict 상태]
    end
    subgraph LLM
        HTTP[OpenAI 호환 REST<br/>urllib 직접 호출 + 자체 재시도]
        HTTP --> J[judge<br/>gpt-4.1-mini]
        HTTP --> T[텍스트 수리<br/>gpt-4.1-mini]
        HTTP --> V[표 재구성<br/>gpt-5-mini vision]
    end
    subgraph 문서 처리
        ODL[opendataloader-pdf<br/>Java 파서]
        FITZ[PyMuPDF<br/>렌더링 · 표 감지 · crop]
        OCR[Surya OCR<br/>subprocess]
    end
    subgraph 관측
        LS[LangSmith 트레이싱]
        USE[llm_usage 비용 집계]
    end
    LG --> HTTP
    LG --> ODL
    LG --> FITZ
    LG --> OCR
    LG -.-> LS
    HTTP -.-> USE
```

| 레이어 | 선택 | 역할 |
|---|---|---|
| 오케스트레이션 | LangGraph | 6개 노드 상태 머신. 조건부 엣지로 수리 루프를 돌리고, 상태는 `TypedDict`로 노드 간 계약을 고정 |
| LLM 호출 | OpenAI 호환 REST (urllib) | judge, 텍스트 수리, 비전 재구성이 같은 HTTP 경로를 탄다. base URL만 바꾸면 호환 서버로 교체 가능 |
| PDF 처리 | PyMuPDF | 페이지 렌더링(judge 그라운딩, 비전 crop), 괘선 표 감지, TEDS-lite 기준 그리드 추출 |
| 기본 파서 | opendataloader-pdf | Java 기반. 파서 어댑터 레지스트리 뒤에 있어서 다른 파서로 교체하거나 추가할 수 있다 |
| OCR | Surya (subprocess) | 스캔 페이지용. 실패해도 파이프라인은 계속 간다 (fail-open) |
| 트레이싱 | LangSmith | 노드 입출력을 구조화 요약으로만 내보낸다. 문서 원문은 트레이스에 나가지 않는다 |
| 테스트/패키징 | pytest, uv, GitHub Actions | 190개 테스트가 1초 안에 돈다. 전부 모킹 기반이라 API 키 없이 CI에서 돈다 |

openai SDK 대신 urllib를 직접 쓰는 건 의도한 선택이다. 재시도 정책과 비용 계측을 호출 지점 한 곳(`_call_with_retry`)에서 통제하고 싶었고, SDK 버전 업그레이드에 끌려다니고 싶지 않았다. LangGraph를 쓴 이유는 반대로 직접 만들기 싫어서다. 조건부 엣지와 상태 병합을 손으로 짜면 그게 또 하나의 버그 표면이 된다.

## 벤치마크

두 종류를 잰다. 채점기는 자체 결정적 메트릭이고, parsing-agent는 이 채점기를 내부에서 최적화하므로 유리하다는 점은 감안하고 봐야 한다. 중립 검증은 사람 라벨로 하는 게 맞고, `golden/`에 그 프로토콜이 있다.

수리 루프가 더하는 가치. 1라운드 점수가 곧 기존 파서 출력이다.

| 시나리오 | 파서 출력 | 루프 종료 |
|---|---|---|
| 노이즈 문서 (중복 헤딩, 잘린 문장, 깨진 표) | 0.862 | 0.981 |
| 본문 절반 누락 | 0.406 | 0.930 |
| 고장난 수리기 주입 | 0.862 | 0.862 (롤백이 차단) |

외부 파서와 같은 잣대로 쟀을 때. 실제 환경영향평가 PDF 3종.

| 엔진 | 평균 | 협의내용 | 사업개요 | 대상지역 | 시간/문서 |
|---|---|---|---|---|---|
| parsing-agent | 0.732 | 0.812 | 0.630 | 0.755 | 186~260s |
| markitdown | 0.666 | 0.785 | 0.680 | 0.531 | 0.1~1.4s |
| docling | 0.657 | 0.783 | 0.426 | 0.761 | 5~19s |
| opendataloader | 0.655 | 0.744 | 0.583 | 0.638 | 1.1~5.3s |
| pymupdf4llm | 0.358 | 0.405 | 0.000 | 0.669 | 0.7~16.5s |

문서별 1위는 셋 다 다르다. 협의내용은 우리, 사업개요는 markitdown, 대상지역은 docling이다. 우리가 평균 1위인 건 최고점 때문이 아니라 최저점이 0.630으로 제일 높아서다. 단발 파서들은 문서에 따라 0.0까지 무너진다. 수리 루프가 파는 건 바닥이다.

재현:

```bash
uv sync --extra bench --extra bench-docling
uv run python benchmarks/run_head_to_head.py data/*.pdf
```

## 프로젝트 구조

```
src/parsing_agent/
├── workflow.py        # LangGraph 상태 머신, 롤백, 시도 추적
├── evaluation.py      # 결정적 메트릭, judge 통합, 이슈 분류
├── judge.py           # 멀티모달 LLM judge (재시도, JSON 폴백, fail-open)
├── repair.py          # 휴리스틱 수리, 수리 대상 진단
├── llm_repair.py      # 이슈 단위 LLM 텍스트 수리
├── visual_repair.py   # 비전 표 재구성, crop 전략
├── table_metrics.py   # TEDS-lite 셀 단위 표 유사도
├── llm_usage.py       # 스테이지별 LLM 비용 집계
└── parsers.py         # 파서 어댑터 (opendataloader, layout-first, fallback)

benchmarks/            # 외부 파서 head-to-head
golden/                # 사람 라벨 골든셋 (라벨링 가이드, 상관 분석)
tests/                 # 190 tests
```

## 로드맵

- [ ] 골든셋 라벨 수집. 지금 점수는 전부 자체 메트릭이라 사람 기준과의 상관이 검증 안 됐다
- [ ] 라벨 확보 후 judge 캘리브레이션, head-to-head 중립 채점 재계산
- [x] 외부 파서 head-to-head (5개 엔진)
- [x] TEDS-lite 셀 단위 표 메트릭
- [x] 다중 페이지 표 crop

## 라이선스

[MIT](LICENSE)
