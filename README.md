# Research Platform

Платформа для квази-экспериментов: **Diff-in-Diff**, **Event Study**, **Causal Impact**,
**Interrupted Time Series**, **Forecast + baseline**, **Anomaly Detection**.

## Быстрый старт

```bash
git clone https://github.com/DoubleS123/research_platform.git
cd research_platform
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
streamlit run src/research_platform/app/streamlit_app.py
```

После `pip install` доступна и консольная команда `research-app` — она сама поднимает
Streamlit-сервер (эквивалент `streamlit run …`).

## Workflow

1. Загрузить CSV (panel или time series).
2. Получить рекомендацию метода (rule-based, без ML).
3. Заполнить минимальные параметры для выбранного метода.
4. Получить эффект, диагностики, графики и Markdown-отчёт.

## Типовые CSV

**Panel (DiD / Event Study):**

```text
unit_id,date,group,treatment_start,metric_value
```

**Time series (Causal Impact):**

```text
date,target_metric,covariate_1
```

Базовые примеры: `tests/fixtures/synthetic_panel.csv`, `tests/fixtures/synthetic_time_series.csv`.

### Демонстрационные датасеты

В `tests/fixtures/` есть готовые сценарии (генерируются `_generate_examples.py`, seed зафиксирован).
Эффекты подобраны так, чтобы p-value покрывал весь спектр — от уверенной значимости до
пограничных и незначимых результатов (не «лабораторные» p≈0 во всех файлах):

| Файл | Тип | Сценарий | Ожидаемый результат |
|------|-----|----------|---------------------|
| `panel_staggered_rollout.csv` | panel | Раскатка по городам тремя волнами | DiD/Event Study — «+» эффект, p≈0.01 (умеренно значим) |
| `panel_pricing_negative.csv` | panel | Повышение цены снизило заказы | DiD — отрицательный ATE, p≈0.004, риск high |
| `panel_null_effect.csv` | panel | Фича раскатана, но эффекта почти нет | DiD — p≈0.7, результат **не значим** |
| `ts_campaign_uplift.csv` | time series | Кампания + трафик-ковариата | Causal Impact / ITS — uplift ~2%, p≈0.003 |
| `ts_marginal_effect.csv` | time series | Слабый uplift на коротком окне | Causal Impact — **пограничный** результат, p≈0.06 |
| `ts_anomaly_incidents.csv` | time series | Сбой трекинга и всплеск разной силы | Anomaly Detection — severity 2–4σ, число аномалий зависит от порога |

Перегенерировать: `python tests/fixtures/_generate_examples.py`.

## Тесты

```bash
pytest
```

## Структура

- `src/research_platform/core` — профилирование, рекомендации, схемы
- `src/research_platform/methods` — отдельные модули по методам
- `src/research_platform/app` — Streamlit UI
