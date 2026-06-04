"""Jupyter notebook export for reproducible manual analysis."""

import base64
import gzip
import json
from typing import Any

import pandas as pd
from pydantic import BaseModel

from research_platform.core.schemas import AnalysisMethod, AnalysisResult


def _source(lines: list[str]) -> list[str]:
    """Convert plain lines to notebook source lines."""
    return [f"{line}\n" for line in lines]


def _params_to_dict(params: BaseModel | object) -> dict[str, Any]:
    """Serialize pydantic params to a plain dict."""
    if isinstance(params, BaseModel):
        return params.model_dump(mode="json")
    return {}


def build_reproduction_notebook(
    *,
    df: pd.DataFrame,
    method: AnalysisMethod,
    params: BaseModel | object,
    result: AnalysisResult,
) -> bytes:
    """
    Build an ipynb file that reproduces the selected analysis.

    The notebook embeds the uploaded CSV as gzip+base64 so it can be executed standalone.
    """
    csv_payload = df.to_csv(index=False).encode("utf-8")
    encoded_csv = base64.b64encode(gzip.compress(csv_payload)).decode("ascii")
    params_json = json.dumps(_params_to_dict(params), ensure_ascii=False, indent=2)

    method_import = {
        AnalysisMethod.DID: "from research_platform.methods.did import run_did",
        AnalysisMethod.EVENT_STUDY: "from research_platform.methods.event_study import run_event_study",
        AnalysisMethod.CAUSAL_IMPACT: "from research_platform.methods.causal_impact import run_causal_impact",
        AnalysisMethod.ITS: "from research_platform.methods.interrupted_time_series import run_interrupted_time_series",
        AnalysisMethod.FORECAST_BASELINE: "from research_platform.methods.forecast_baseline import run_forecast_baseline",
        AnalysisMethod.ANOMALY_DETECTION: "from research_platform.methods.anomaly_detection import run_anomaly_detection",
    }[method]
    params_import = {
        AnalysisMethod.DID: "from research_platform.core.schemas import DidParams",
        AnalysisMethod.EVENT_STUDY: "from research_platform.core.schemas import EventStudyParams",
        AnalysisMethod.CAUSAL_IMPACT: "from research_platform.core.schemas import CausalImpactParams",
        AnalysisMethod.ITS: "from research_platform.core.schemas import InterruptedTimeSeriesParams",
        AnalysisMethod.FORECAST_BASELINE: "from research_platform.core.schemas import ForecastBaselineParams",
        AnalysisMethod.ANOMALY_DETECTION: "from research_platform.core.schemas import AnomalyDetectionParams",
    }[method]
    params_class = {
        AnalysisMethod.DID: "DidParams",
        AnalysisMethod.EVENT_STUDY: "EventStudyParams",
        AnalysisMethod.CAUSAL_IMPACT: "CausalImpactParams",
        AnalysisMethod.ITS: "InterruptedTimeSeriesParams",
        AnalysisMethod.FORECAST_BASELINE: "ForecastBaselineParams",
        AnalysisMethod.ANOMALY_DETECTION: "AnomalyDetectionParams",
    }[method]
    run_call = {
        AnalysisMethod.DID: "run_did(df, params)",
        AnalysisMethod.EVENT_STUDY: "run_event_study(df, params)",
        AnalysisMethod.CAUSAL_IMPACT: "run_causal_impact(df, params)",
        AnalysisMethod.ITS: "run_interrupted_time_series(df, params)",
        AnalysisMethod.FORECAST_BASELINE: "run_forecast_baseline(df, params)",
        AnalysisMethod.ANOMALY_DETECTION: "run_anomaly_detection(df, params)",
    }[method]

    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": _source(
                    [
                        "# Воспроизведение causal-анализа",
                        "",
                        f"Метод: `{method.value}`",
                        "",
                        "Notebook содержит исходный CSV, параметры анализа и код запуска метода.",
                        "Не пересылайте файл наружу, если данные содержат персональную или коммерчески чувствительную информацию.",
                    ]
                ),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": _source(
                    [
                        "import base64",
                        "import gzip",
                        "from io import BytesIO",
                        "",
                        "import pandas as pd",
                        "",
                        method_import,
                        params_import,
                    ]
                ),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": _source(
                    [
                        f"encoded_csv = '''{encoded_csv}'''",
                        "csv_bytes = gzip.decompress(base64.b64decode(encoded_csv))",
                        "df = pd.read_csv(BytesIO(csv_bytes))",
                        "df.head()",
                    ]
                ),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": _source(
                    [
                        f"params_dict = {params_json}",
                        f"params = {params_class}(**params_dict)",
                        "params",
                    ]
                ),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": _source(
                    [
                        f"result = {run_call}",
                        "print(result.summary)",
                        "print(result.conclusion)",
                        "print(result.recommendation)",
                    ]
                ),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": _source(
                    [
                        "pd.DataFrame([metric.model_dump() for metric in result.statistical_metrics])",
                    ]
                ),
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": _source(
                    [
                        "for plot in result.plots:",
                        "    fig = plot.figure",
                        "    fig.show()",
                    ]
                ),
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook, ensure_ascii=False).encode("utf-8")
