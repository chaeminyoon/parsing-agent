"""사람 라벨과 파이프라인 점수의 상관을 분석한다.

사용:
    uv run python golden/run_golden_eval.py --run-dir outputs/golden-run --labels golden/labels

동작:
- labels/*.json (schema.json 형식)을 읽어 평가자 평균 라벨을 만든다
- run-dir에서 각 문서의 파이프라인 리포트(문서.json)를 찾아 자체 점수를 뽑는다
- 평가자 간 일치도(문서×항목 단위 절대 일치율과 ±1 일치율)를 출력한다
- 라벨 5건 이상이면 메트릭별 Spearman 상관을, 미만이면 개별 대조표를 출력한다

외부 의존성 없이 표준 라이브러리로만 동작한다 (scipy 불필요).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LABEL_FIELDS = ["overall_quality", "text_coverage", "table_correctness", "structure_preservation"]
# 사람 라벨 항목 ↔ 파이프라인 메트릭 대응
METRIC_FOR_LABEL = {
    "overall_quality": "total_score",
    "text_coverage": "text_coverage",
    "table_correctness": "table_preservation",
    "structure_preservation": "structure_retention",
}


def load_labels(labels_dir: Path) -> list[dict]:
    records = []
    for path in sorted(labels_dir.glob("*.json")):
        if path.name in {"schema.json", "example.json"}:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data.get("ratings"):
            continue
        records.append(data)
    return records


def mean_rating(record: dict, field: str) -> float:
    values = [rating[field] for rating in record["ratings"] if field in rating]
    return sum(values) / len(values)


def rater_agreement(records: list[dict]) -> dict[str, float]:
    """평가자 2인 이상인 문서에서 항목별 절대 일치율과 ±1 일치율."""
    exact = 0
    within_one = 0
    total = 0
    for record in records:
        ratings = record["ratings"]
        if len(ratings) < 2:
            continue
        for field in LABEL_FIELDS:
            values = [rating[field] for rating in ratings if field in rating]
            if len(values) < 2:
                continue
            total += 1
            if max(values) == min(values):
                exact += 1
            if max(values) - min(values) <= 1:
                within_one += 1
    if total == 0:
        return {}
    return {"pairs": total, "exact_agreement": exact / total, "within_one_agreement": within_one / total}


def find_pipeline_metrics(run_dir: Path, document: str) -> dict | None:
    stem = Path(document).stem
    for report_path in run_dir.rglob(f"{stem}.json"):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        metrics = report.get("metrics")
        if isinstance(metrics, dict):
            return metrics
    return None


def spearman(xs: list[float], ys: list[float]) -> float:
    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        result = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                result[order[k]] = rank
            i = j + 1
        return result

    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mean_rx, mean_ry = sum(rx) / n, sum(ry) / n
    cov = sum((a - mean_rx) * (b - mean_ry) for a, b in zip(rx, ry))
    var_x = sum((a - mean_rx) ** 2 for a in rx) ** 0.5
    var_y = sum((b - mean_ry) ** 2 for b in ry) ** 0.5
    if var_x == 0 or var_y == 0:
        return 0.0
    return cov / (var_x * var_y)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("golden/labels"))
    args = parser.parse_args()

    records = load_labels(args.labels)
    if not records:
        print("라벨이 없다. golden/labels/에 schema.json 형식으로 라벨을 추가한 뒤 다시 실행.")
        return 1

    agreement = rater_agreement(records)
    if agreement:
        print(
            f"평가자 일치도: 절대 {agreement['exact_agreement']:.0%}, "
            f"±1 {agreement['within_one_agreement']:.0%} (항목쌍 {agreement['pairs']}개)"
        )
        if agreement["within_one_agreement"] < 0.7:
            print("경고: ±1 일치율이 70% 미만이다. 라벨링 기준을 재합의하고 재라벨링을 권장.")
    else:
        print("평가자 2인 이상 라벨이 없어 일치도를 계산하지 못했다.")

    paired: dict[str, list[tuple[float, float]]] = {field: [] for field in LABEL_FIELDS}
    hallucination_flags: list[tuple[str, bool]] = []
    missing = []
    for record in records:
        metrics = find_pipeline_metrics(args.run_dir, record["document"])
        if metrics is None:
            missing.append(record["document"])
            continue
        for field in LABEL_FIELDS:
            metric_value = metrics.get(METRIC_FOR_LABEL[field])
            if metric_value is None:
                continue
            paired[field].append((mean_rating(record, field), float(metric_value)))
        hallucination_flags.append(
            (record["document"], any(rating.get("hallucination") for rating in record["ratings"]))
        )

    if missing:
        print(f"리포트를 찾지 못한 문서 {len(missing)}건: {missing}")

    sample_size = len(paired["overall_quality"])
    print(f"\n대조 가능한 문서: {sample_size}건")
    if sample_size >= 5:
        print("\n사람 라벨 vs 파이프라인 메트릭 (Spearman):")
        for field in LABEL_FIELDS:
            pairs = paired[field]
            if len(pairs) < 5:
                continue
            rho = spearman([p[0] for p in pairs], [p[1] for p in pairs])
            verdict = "OK" if rho >= 0.6 else ("약함" if rho >= 0.3 else "불일치 — 메트릭 재검토 필요")
            print(f"  {field:26s} ↔ {METRIC_FOR_LABEL[field]:20s} rho={rho:+.3f}  [{verdict}]")
    else:
        print("\n표본이 5건 미만이라 상관계수 대신 개별 대조를 출력한다:")
        for field in LABEL_FIELDS:
            for human, machine in paired[field]:
                print(f"  {field:26s} 사람={human:.1f}/5  기계={machine:.3f}")

    flagged = [doc for doc, flag in hallucination_flags if flag]
    if flagged:
        print(f"\n환각 라벨 문서 {len(flagged)}건: {flagged}")
        print("→ 해당 문서의 judge hallucination_risk와 대조해 max_hallucination_risk 게이트 설정을 검토할 것.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
