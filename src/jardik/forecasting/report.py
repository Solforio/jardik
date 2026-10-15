"""The forecast backtest report (PDF). Numbers follow the Spanish convention (1.234,5)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from jardik.core.formatting import format_percent_es
from jardik.core.pdfkit import (
    H1,
    SMALL,
    TITLE,
    build_document,
    bullets,
    image,
    kv,
    num,
    para,
    table,
)
from jardik.forecasting.backtest import BacktestResult

SCENARIO_TEXT = {
    "val": "validation buildings, validation period (only stops the boosting early)",
    "test_time": "training buildings, test period: known buildings, unseen future",
    "test_building": "test buildings, training period: unseen buildings, known period",
    "test_both": "test buildings, test period: unseen buildings AND unseen future (main result)",
}

DEFINITIONS = [
    "MAE: mean absolute error in kWh per hour. RMSE: root mean square error, punishes large "
    "misses.",
    "CV(RMSE): RMSE divided by the mean load, as a percentage; comparable between buildings.",
    "NMBE: normalised mean bias, ASHRAE Guideline 14 sign: positive = the model under-forecasts, "
    "negative = it over-forecasts.",
    "Pinball loss: error of the quantile forecasts (average over 10 %, 50 % and 90 %). "
    "Coverage 80 %: share of actual values inside the 10-90 % band (ideal: 80 %).",
    "Skill: 1 - model error / baseline error; positive means the model is better than the best "
    "baseline of that scenario.",
]


def _pct(value: Any) -> str:
    return format_percent_es(None if pd.isna(value) else float(value))


def _metric_rows(metrics: pd.DataFrame) -> list[list[object]]:
    head = ["Scenario", "Model", "Rows", "MAE", "RMSE", "CV(RMSE)", "NMBE"]
    rows: list[list[object]] = [head]
    for r in metrics.itertuples(index=False):
        rows.append(
            [
                r.scenario,
                r.model,
                num(r.n),
                num(r.mae, 2),
                num(r.rmse, 2),
                _pct(r.cv_rmse),
                _pct(r.nmbe),
            ]
        )
    return rows


def build_forecast_pdf(
    path: Path,
    manifest: dict[str, Any],
    backtest: BacktestResult,
    figures_dir: Path,
    importance: pd.DataFrame,
) -> None:
    gate = backtest.gate
    primary = gate["scenario"]
    extraction = manifest["extraction"]
    metrics = backtest.metrics
    story: list[Any] = [
        para("Forecast backtest report", TITLE),
        para(f"Run ID: {manifest['run_id']}"),
        para(
            f"Created: {manifest['created_at']}   |   Jardik {manifest['environment']['jardik']}, "
            f"LightGBM {manifest['environment']['lightgbm']}",
            SMALL,
        ),
    ]

    verdict = "PASSED" if gate["passed"] else "NOT PASSED"
    story += [
        para("1. Summary", H1),
        kv(
            [
                ("Gate", f"{verdict}: {gate['message']}"),
                ("Data", f"{extraction['dataset']} ({extraction['id']})"),
                ("Period", f"{extraction['period'][0]} to {extraction['period'][1]}"),
                (
                    "Buildings",
                    f"{num(manifest['split']['roles']['train'])} train / "
                    f"{num(manifest['split']['roles']['val'])} validation / "
                    f"{num(manifest['split']['roles']['test'])} test",
                ),
                ("Models", ", ".join(manifest["models_fitted"])),
                ("Forecast task", "day-ahead hourly load (kWh per hour), issued at local midnight"),
            ]
        ),
    ]

    story += [
        para("2. Scenarios", H1),
        para("Evaluation never uses random rows: buildings and time are both held out."),
        table(
            [["Scenario", "Meaning", "Rows"]]
            + [
                [s, SCENARIO_TEXT[s], num(manifest["split"]["scenarios"].get(s, 0))]
                for s in SCENARIO_TEXT
            ],
            [26, 120, 26],
        ),
        para(
            "All models are scored on exactly the same rows (rows where every model produced a "
            "forecast and the load was measured).",
            SMALL,
        ),
    ]

    main = metrics[metrics["scenario"] == primary]
    story += [
        para(f"3. Results on the main scenario ({primary})", H1),
        table(_metric_rows(main), [26, 52, 20, 20, 20, 20, 14]),
        image(figures_dir / "metric_bars.png", 150),
        para("All scenarios", SMALL),
        table(_metric_rows(metrics), [26, 52, 20, 20, 20, 20, 14]),
    ]

    skill = backtest.skill
    story += [para("4. Skill against the best baseline", H1)]
    skill_rows = [
        ["Scenario", "Model", "Best baseline", "MAE", "RMSE", "CV(RMSE)", "Median CV(RMSE)"]
    ]
    for r in skill.itertuples(index=False):
        skill_rows.append(
            [
                r.scenario,
                r.model,
                r.vs,
                _pct(r.mae_skill),
                _pct(r.rmse_skill),
                _pct(r.cv_rmse_skill),
                _pct(r.median_cv_rmse_skill),
            ]
        )
    story += [table(skill_rows, [24, 22, 46, 18, 18, 20, 24])]
    for name, check in gate["checks"].items():
        story.append(
            para(
                f"Gate check {name}: challenger {num(check['challenger'], 3)} vs "
                f"baseline {num(check['baseline'], 3)} ({_pct(check['relative'])} better)",
                SMALL,
            )
        )

    macro = backtest.macro[backtest.macro["scenario"] == primary]
    macro_rows = [
        ["Model", "Buildings", "MAE mean", "MAE median", "CV(RMSE) mean", "CV(RMSE) median"]
    ]
    for r in macro.itertuples(index=False):
        macro_rows.append(
            [
                r.model,
                num(r.buildings),
                num(r.mae_mean, 2),
                num(r.mae_median, 2),
                _pct(r.cv_rmse_mean),
                _pct(r.cv_rmse_median),
            ]
        )
    story += [
        para("5. Building by building", H1),
        para(
            "Every building counts equally here (macro average), so a few large buildings cannot "
            "hide a bad result on the many small ones."
        ),
        table(macro_rows, [48, 20, 24, 26, 28, 26]),
        image(figures_dir / "building_scatter.png", 90),
    ]

    prob = main[main["pinball"].notna()]
    prob_rows = [["Model", "Pinball loss", "Coverage 80 %", "Mean band width (kWh/h)"]]
    for r in prob.itertuples(index=False):
        prob_rows.append(
            [r.model, num(r.pinball, 3), _pct(r.coverage_80), num(r.interval_width, 2)]
        )
    story += [para("6. Probabilistic forecast", H1), table(prob_rows, [50, 36, 36, 50])]

    story += [
        para("7. When the model is wrong", H1),
        image(figures_dir / "error_by_hour.png", 140),
        image(figures_dir / "example_week.png", 150),
        para(
            f"Example: {manifest['figures']['example_week']['building_id']}, week starting "
            f"{manifest['figures']['example_week']['week_start']}.",
            SMALL,
        ),
    ]

    top = importance.head(10)
    story += [
        para("8. What drives the model", H1),
        image(figures_dir / "feature_importance.png", 120),
        table(
            [["Feature", "Gain"]]
            + [[r.feature, num(r.gain, 1)] for r in top.itertuples(index=False)],
            [80, 40],
        ),
    ]

    story += [para("9. Metric definitions", H1), *bullets(DEFINITIONS)]
    story += [para("10. Limitations", H1), *bullets(manifest["limitations"])]
    story += [
        para("11. Reproducibility", H1),
        kv(
            [
                ("Config hash", manifest["config_hash"]),
                ("Seed", manifest["seed"]),
                ("Data checksum (merged dataset)", extraction["merged_sha256"][:32] + "..."),
                ("Licence of the data", extraction["licence"].split(":")[0]),
            ]
        ),
        para(extraction["attribution"], SMALL),
    ]
    build_document(path, story, f"Forecast backtest report {manifest['run_id']}")
