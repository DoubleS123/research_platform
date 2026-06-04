"""Rule-based method recommendation."""

from pydantic import BaseModel, Field

from research_platform.core.schemas import AnalysisMethod, DatasetFormat, DatasetProfile


class MethodRecommendation(BaseModel):
    """Recommended method with rationale."""

    method: AnalysisMethod
    confidence: str
    reason: str
    alternatives: list[AnalysisMethod] = Field(default_factory=list)


def recommend_method(profile: DatasetProfile) -> MethodRecommendation:
    """
    Suggest an analysis method from dataset profile.

    Uses simple rules, no ML.
    """
    fmt = profile.detected_format

    if fmt == DatasetFormat.LONG_METRICS:
        if profile.suggested_group_col and profile.suggested_unit_col:
            return MethodRecommendation(
                method=AnalysisMethod.DID,
                confidence="low",
                reason=(
                    "Long-формат (metric_name + metric_value) с группами treated/control. "
                    "Перед запуском отфильтруйте одну метрику по metric_name — методы ожидают "
                    "один числовой столбец."
                ),
                alternatives=[
                    AnalysisMethod.EVENT_STUDY,
                    AnalysisMethod.CAUSAL_IMPACT,
                    AnalysisMethod.ITS,
                ],
            )
        return MethodRecommendation(
            method=AnalysisMethod.CAUSAL_IMPACT,
            confidence="low",
            reason=(
                "Long-формат (metric_name + metric_value) без явных групп. Перед запуском "
                "выберите одну метрику по metric_name и приведите данные к одной точке на дату."
            ),
            alternatives=[
                AnalysisMethod.ITS,
                AnalysisMethod.FORECAST_BASELINE,
                AnalysisMethod.ANOMALY_DETECTION,
            ],
        )

    if fmt == DatasetFormat.TIME_SERIES:
        return MethodRecommendation(
            method=AnalysisMethod.CAUSAL_IMPACT,
            confidence="high",
            reason="Обнаружен агрегированный временной ряд (date + метрика).",
            alternatives=[
                AnalysisMethod.ITS,
                AnalysisMethod.FORECAST_BASELINE,
                AnalysisMethod.ANOMALY_DETECTION,
            ],
        )

    if fmt == DatasetFormat.PANEL and profile.suggested_group_col:
        alts = [AnalysisMethod.EVENT_STUDY] if profile.has_treatment_start else []
        alts.extend([AnalysisMethod.ITS, AnalysisMethod.ANOMALY_DETECTION])
        return MethodRecommendation(
            method=AnalysisMethod.DID,
            confidence="high",
            reason="Есть panel-данные с группами treated/control — подходит Diff-in-Diff.",
            alternatives=alts,
        )

    if profile.has_treatment_start and profile.suggested_unit_col:
        return MethodRecommendation(
            method=AnalysisMethod.EVENT_STUDY,
            confidence="medium",
            reason="Есть unit и дата начала воздействия — можно оценить динамику Event Study.",
            alternatives=[AnalysisMethod.DID],
        )

    if profile.suggested_time_col and profile.suggested_metric_cols:
        return MethodRecommendation(
            method=AnalysisMethod.CAUSAL_IMPACT,
            confidence="low",
            reason="Есть время и метрика, но нет явных групп — возможен Causal Impact или ITS на агрегате.",
            alternatives=[
                AnalysisMethod.ITS,
                AnalysisMethod.FORECAST_BASELINE,
                AnalysisMethod.ANOMALY_DETECTION,
                AnalysisMethod.DID,
            ],
        )

    return MethodRecommendation(
        method=AnalysisMethod.DID,
        confidence="low",
        reason="Формат не распознан полностью — укажите колонки вручную.",
        alternatives=[
            AnalysisMethod.EVENT_STUDY,
            AnalysisMethod.CAUSAL_IMPACT,
            AnalysisMethod.ITS,
            AnalysisMethod.FORECAST_BASELINE,
            AnalysisMethod.ANOMALY_DETECTION,
        ],
    )
