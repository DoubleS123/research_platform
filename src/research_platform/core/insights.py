"""Helpers for product-oriented interpretation of statistical results."""

import math


def is_statistically_significant(
    *,
    p_value: float | None = None,
    ci_lower: float | None = None,
    ci_upper: float | None = None,
) -> bool | None:
    """Infer statistical significance from p-value or confidence interval."""
    if p_value is not None and not math.isnan(p_value):
        return p_value < 0.05
    if ci_lower is not None and ci_upper is not None:
        return ci_lower > 0 or ci_upper < 0
    return None


def format_effect(effect: float | None) -> str:
    """Format point estimate for Russian UI text."""
    if effect is None or math.isnan(effect):
        return "не удалось оценить"
    return f"{effect:.4f}"


def format_relative_effect(relative_effect_pct: float | None) -> str:
    """Format relative effect for Russian UI text."""
    if relative_effect_pct is None or math.isnan(relative_effect_pct):
        return "не рассчитан"
    sign = "+" if relative_effect_pct > 0 else ""
    return f"{sign}{relative_effect_pct:.2f}%"


def format_p_value(p_value: float | None) -> str:
    """Format p-value without displaying tiny values as zero."""
    if p_value is None or math.isnan(p_value):
        return "n/a"
    if p_value < 0.0001:
        return "< 0.0001"
    return f"{p_value:.4f}"


def build_product_conclusion(
    *,
    metric_name: str,
    effect: float | None,
    relative_effect_pct: float | None,
    p_value: float | None = None,
    ci_lower: float | None = None,
    ci_upper: float | None = None,
    method_label: str = "метод",
) -> tuple[str, str, str]:
    """
    Build conclusion, recommendation and risk level for product decisions.

    Returns:
        Tuple of conclusion, recommendation and risk level.
    """
    significance = is_statistically_significant(
        p_value=p_value,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )
    effect_text = format_effect(effect)
    relative_text = format_relative_effect(relative_effect_pct)

    if effect is None or math.isnan(effect):
        return (
            "Недостаточно информации для устойчивого вывода об эффекте.",
            "Проверьте качество данных, окна анализа и выбранные колонки.",
            "high",
        )

    direction = "увеличилась" if effect > 0 else "снизилась" if effect < 0 else "не изменилась"
    stat_text = (
        "статистически значимым"
        if significance is True
        else "не является статистически значимым"
        if significance is False
        else "требует дополнительной проверки статистической значимости"
    )

    conclusion = (
        f"По методу {method_label} продуктовые изменения связаны с эффектом {effect_text} "
        f"по метрике `{metric_name}`. Целевая метрика {direction} примерно на {relative_text}; "
        f"эффект {stat_text}."
    )

    if significance is True and effect > 0:
        recommendation = (
            "Рекомендация: рассмотреть масштабирование изменения, сохранив мониторинг guardrail-метрик "
            "и проверку устойчивости эффекта на альтернативных окнах."
        )
        risk = "medium"
    elif significance is True and effect < 0:
        recommendation = (
            "Рекомендация: не масштабировать изменение без доработки; эффект отрицательный и статистически "
            "подтверждён."
        )
        risk = "high"
    elif significance is False:
        recommendation = (
            "Рекомендация: не делать сильный causal claim; увеличить период наблюдения, проверить мощность "
            "и качество контрольной группы."
        )
        risk = "medium"
    else:
        recommendation = (
            "Рекомендация: интерпретировать результат как предварительный и провести robustness checks."
        )
        risk = "medium"

    return conclusion, recommendation, risk
