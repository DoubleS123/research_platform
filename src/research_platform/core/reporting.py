"""Markdown report export from AnalysisResult."""

from research_platform.core.insights import format_p_value
from research_platform.core.schemas import AnalysisResult


def result_to_markdown(result: AnalysisResult) -> str:
    """Convert analysis result to a Markdown report string."""
    lines = [
        f"# Отчёт: {result.method.value}",
        "",
        "## Краткий вывод",
        result.conclusion or result.summary,
        "",
        "## Оценка эффекта",
    ]
    if result.effect_estimate is not None:
        lines.append(f"- Point estimate: **{result.effect_estimate:.4f}**")
    if result.ci_lower is not None and result.ci_upper is not None:
        lines.append(f"- 95% interval: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
    if result.p_value is not None:
        lines.append(f"- p-value: {format_p_value(result.p_value)}")
    if result.relative_effect_pct is not None:
        lines.append(f"- Relative lift: {result.relative_effect_pct:.2f}%")
    if result.decision:
        lines.append(f"- Decision hint: {result.decision}")
    if result.risk_level:
        lines.append(f"- Risk level: {result.risk_level}")

    if result.statistical_metrics:
        lines.extend(["", "## Статистические метрики"])
        for metric in result.statistical_metrics:
            suffix = f" — {metric.description}" if metric.description else ""
            lines.append(f"- {metric.name}: **{metric.value}**{suffix}")

    lines.extend(["", "## Предпосылки"])
    for a in result.assumptions:
        lines.append(f"- {a}")

    lines.extend(["", "## Диагностики"])
    for d in result.diagnostics:
        status = "✅ Пройдено" if d.passed else "⚠️ Требует внимания"
        lines.append(f"- [{status}] {d.name}: {d.message}")

    lines.extend(
        [
            "",
            "## Интерпретация",
            result.interpretation,
            "",
            "## Продуктовая рекомендация",
            result.recommendation,
        ]
    )
    return "\n".join(lines)
