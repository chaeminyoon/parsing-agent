# Structured Node Rules PRD

## 목적

문서마다 달라지는 자유문장 해석 의존을 줄이고, `evaluate -> inspect -> route -> repair` 노드가
구조화된 입력을 같은 우선순위 규칙으로 처리하도록 고정한다.

핵심 목표는 아래 두 가지다.

1. 같은 유형의 입력이면 같은 판단과 같은 수리 루프가 반복되도록 만든다.
2. 한 문서에 우연히 맞는 규칙이 아니라, 여러 PDF에서도 재현 가능한 처리 규칙을 유지한다.

## 범위

이번 문서의 범위는 상위 그래프의 아래 네 노드다.

- `evaluate`
- `inspect`
- `route`
- `repair`

`parse`와 `finalize`는 입력 구성과 결과 조립의 성격이 강하므로 본 문서에서는 보조 범위로 본다.

## 공통 처리 원칙

1. structured 필드가 있으면 자유문장보다 우선한다.
2. structured 필드가 비어 있거나 불완전할 때만 자유문장 파싱을 fallback으로 사용한다.
3. `page_number`는 `1 <= page_number <= source.page_count`를 만족할 때만 유효하다.
4. `table_label`이 비어 있거나 `issue_type`이 미등록 taxonomy면 해당 structured 항목은 보조 정보로만 본다.
5. 노드 출력은 다음 노드가 그대로 재사용할 수 있는 structured 필드를 유지해야 한다.
6. 한 번 실패한 visual repair task는 같은 루프 안에서 반복 시도하지 않는다.

## Structured 입력 정의

### `JudgeResult.table_findings`

judge가 표 관련 문제를 구조화해서 전달하는 필드다.

```json
{
  "issue_type": "missing_header | split_multipage_table | merged_cell_loss | numeric_token_break | table_text_duplication",
  "table_label": "표 4.2-2",
  "page_number": 12
}
```

### 허용 규칙

| 필드 | 규칙 |
|---|---|
| `issue_type` | 등록된 table issue taxonomy만 허용 |
| `table_label` | 비어 있지 않은 문자열일 때만 사용 |
| `page_number` | 정수이면서 문서 페이지 범위 안일 때만 우선 사용 |

### 공통 fallback 순서

structured finding이 비어 있거나 page 근거가 부족하면 아래 순서로 보정한다.

1. `candidate.metadata.table_label_pages`
2. `candidate.metadata.table_regions`
3. PDF 본문 검색 기반 `_find_page_number()`
4. judge 자유문장 `issues`

## 노드별 처리 규칙

### 1. `evaluate`

| 항목 | 규칙 |
|---|---|
| 입력 | `source`, `candidate` |
| structured 입력 | judge 응답의 `table_findings` |
| 우선순위 | `table_findings` -> 자유문장 `issues` |
| 출력 | `metrics`, `metrics.table_issues`, `metrics.judge_result` |

세부 규칙:

1. judge가 `table_findings`를 반환하면 `JudgeResult.table_findings`에 그대로 보존한다.
2. `metrics.table_issues`는 `JudgeResult.table_findings.issue_type`를 우선 사용한다.
3. structured finding이 없거나 불완전할 때만 `judge_result.issues`를 정규식으로 해석한다.
4. `metrics.notes`는 설명용이며, 제어 신호로 직접 사용하지 않는다.

### 2. `inspect`

| 항목 | 규칙 |
|---|---|
| 입력 | `source`, `candidate`, `metrics` |
| structured 입력 | `metrics.table_issues`, `metrics.judge_result.table_findings` |
| 우선순위 | structured table finding -> heuristic text pattern |
| 출력 | `repair_targets` |

세부 규칙:

1. text 문제는 기존 heuristic 규칙으로 탐지한다.
   - `wrapped_line_noise`
   - `blank_line_noise`
   - `line_repetition_noise`
   - `boundary_repetition_noise`
   - `structure_heading_noise`
   - `image_link_noise`
2. table 문제는 `metrics.table_issues`를 기반으로 target을 만든다.
3. `JudgeResult.table_findings`가 있으면 `RepairTarget`에 아래 필드를 같이 보존한다.
   - `table_label`
   - `page_number`
   - `source_name="judge_table_finding"`
4. structured finding이 없는 table issue는 `source_name="metrics_table_issue"`로 남긴다.

### 3. `route`

| 항목 | 규칙 |
|---|---|
| 입력 | `repair_targets` |
| structured 입력 | `issue_type`, `route_name`, `table_label`, `page_number`, `source_name` |
| 우선순위 | issue taxonomy 고정 매핑 |
| 출력 | `repair_plan` |

세부 규칙:

1. 아래 issue는 기본적으로 `visual_table_repair`로 라우팅한다.
   - `missing_header`
   - `split_multipage_table`
   - `merged_cell_loss`
   - `numeric_token_break`
   - `table_text_duplication`
2. 나머지 text/layout 계열 issue는 `heuristic`으로 라우팅한다.
3. 같은 `(strategy, route_name)` 조합은 하나의 `RepairPlanStep`으로 묶는다.
4. `RepairPlanStep.targets`에는 `RepairTarget`의 structured 필드를 그대로 유지한다.

### 4. `repair`

| 항목 | 규칙 |
|---|---|
| 입력 | `repair_plan`, `candidate`, `metrics`, `source`, `failed_visual_task_keys` |
| structured 입력 | `table_label`, `page_number`, `source_name`, candidate metadata |
| 우선순위 | valid raw page -> metadata -> `_find_page_number()` -> 자유문장 fallback |
| 출력 | 수정된 `candidate`, 누적 `repairs`, 증가된 `iteration_count`, 갱신된 `failed_visual_task_keys` |

#### heuristic repair 규칙

1. `route_name`에 매핑된 transform만 실행한다.
2. `RepairTarget`에 없는 heuristic은 적용하지 않는다.
3. 변환 결과가 기존과 같으면 action을 만들지 않는다.

#### visual repair 규칙

1. route에서 전달된 structured `RepairTarget`이 있으면 judge 자유문장보다 먼저 사용한다.
2. `page_number`가 유효하면 그대로 사용한다.
3. `page_number`가 비어 있거나 범위를 벗어나면 아래 순서로 보정한다.
   - `candidate.metadata.table_label_pages`
   - `_find_page_number()`
4. `repair_targets`에서 만든 visual task는 `table_label|page_number|issue_types` 형태의 stable key로 식별한다.
5. 이미 `failed_visual_task_keys`에 있는 task는 다음 라운드에서 다시 실행하지 않는다.
6. visual recoverer 예외, 저신뢰 결과, 빈 markdown 결과는 workflow 전체 실패로 올리지 않고 task 실패로만 처리한다.

## 노드별 필수 검증

| 노드 | 검증 |
|---|---|
| `evaluate` | `table_findings.issue_type`가 등록 taxonomy인지 확인 |
| `inspect` | structured table finding이 `RepairTarget`으로 유지되는지 확인 |
| `route` | `RepairTarget`의 structured 필드가 `RepairPlanStep.targets`에 유지되는지 확인 |
| `repair` | invalid page skip, page 보정, 실패 task 재시도 금지, recover 예외 흡수 |

## 현재 반영 상태

### 반영 완료

- `JudgeResult.table_findings` 추가
- judge 응답 schema에 `table_findings` 반영
- `evaluate`에서 structured finding 우선 사용
- `inspect`에서 structured finding을 `RepairTarget`으로 보존
- `route`에서 structured `RepairTarget`을 그대로 `RepairPlanStep.targets`에 유지
- `repair`에서 structured visual target 우선 사용
- invalid `page_number` 보정 규칙 반영
- visual recoverer 예외 흡수
- 동일 visual task 재시도 금지

### 남은 개선 과제

- `route`가 `iteration_count`와 이전 실패 이력을 반영해 전략을 승격/중단하는 규칙
- judge가 `table_findings`를 더 안정적으로 채우도록 prompt 보강
- visual repair 실패 사유를 report에 더 구조적으로 남기기

## 성공 기준

1. judge가 structured `table_findings`를 주면 `inspect -> route -> repair`가 같은 payload를 유지한다.
2. invalid page, recover 예외, 응답 파싱 실패가 workflow 전체 실패로 번지지 않는다.
3. 동일 visual task는 한 번 실패하면 다음 라운드에서 반복 시도하지 않는다.
4. 여러 PDF에서 `evaluate -> inspect -> route -> repair`가 같은 taxonomy와 같은 우선순위 규칙으로 동작한다.
