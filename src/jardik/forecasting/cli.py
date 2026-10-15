"""`jardik forecast ...` commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from jardik.core.catalogue import Catalogue
from jardik.core.formatting import format_number_es, format_percent_es
from jardik.core.paths import JardikPaths
from jardik.forecasting.review import build_run_review
from jardik.forecasting.run import (
    ExtractionNotFoundError,
    ForecastRunConfig,
    RunExistsError,
    run_forecast,
)

forecast_app = typer.Typer(
    help="Forecasting: backtest the model ladder on an extraction.", no_args_is_help=True
)


def _fail(message: str, code: int = 1) -> typer.Exit:
    typer.echo(f"Error: {message}", err=True)
    return typer.Exit(code=code)


@forecast_app.command("run")
def run(
    ctx: typer.Context,
    config_file: Annotated[Path, typer.Option("--config", "-c", help="Run configuration YAML.")],
    force: Annotated[bool, typer.Option("--force", help="Replace an existing run.")] = False,
) -> None:
    """Backtest baselines and gradient boosting on an extraction and write the results."""
    paths: JardikPaths = ctx.obj["paths"]
    try:
        config = ForecastRunConfig.from_yaml(config_file)
    except (ValidationError, OSError, ValueError) as exc:
        raise _fail(str(exc), 2) from exc
    try:
        with Catalogue(paths.catalogue_path) as catalogue:
            result = run_forecast(config, paths, catalogue, force=force)
    except (ExtractionNotFoundError, RunExistsError, ValueError) as exc:
        raise _fail(str(exc)) from exc

    gate = result.manifest["gate"]
    typer.echo(f"run {result.run_id}")
    typer.echo(f"gate {'PASSED' if gate['passed'] else 'NOT PASSED'}: {gate['message']}")
    metrics = result.backtest.metrics
    main = metrics[metrics["scenario"] == gate["scenario"]]
    typer.echo(
        f"\n{gate['scenario']} ({format_number_es(int(main['n'].iloc[0]), 0)} rows per model)"
    )
    typer.echo(f"{'model':26s}{'MAE':>9s}{'RMSE':>9s}{'CV(RMSE)':>10s}{'NMBE':>9s}")
    for r in main.itertuples(index=False):
        typer.echo(
            f"{r.model:26s}{format_number_es(r.mae, 2):>9s}{format_number_es(r.rmse, 2):>9s}"
            f"{format_percent_es(r.cv_rmse):>10s}{format_percent_es(r.nmbe):>9s}"
        )
    typer.echo(f"\noutput {result.output_dir}")


@forecast_app.command("runs")
def runs(ctx: typer.Context) -> None:
    """List forecast runs registered in the catalogue, newest first."""
    paths: JardikPaths = ctx.obj["paths"]
    with Catalogue(paths.catalogue_path) as catalogue:
        rows = catalogue.list_runs()
    if not rows:
        typer.echo("No forecast runs yet.")
    for r in rows:
        verdict = "PASSED" if r["gate_passed"] else "NOT PASSED"
        typer.echo(
            f"{r['created_at'][:19]}  {r['run_id']}  {verdict}  "
            f"{r['challenger']} vs {r['best_baseline']}  on {r['extraction_id']}"
        )


@forecast_app.command("review")
def review(
    ctx: typer.Context,
    run_id: Annotated[str | None, typer.Argument(help="Run id; default: the newest.")] = None,
    out: Annotated[
        Path | None, typer.Option(help="Folder to write into (default: review/).")
    ] = None,
) -> None:
    """Build a readable copy of a run: Excel workbook, PDF, figures and JSON files."""
    paths: JardikPaths = ctx.obj["paths"]
    with Catalogue(paths.catalogue_path) as catalogue:
        if run_id is None:
            rows = catalogue.list_runs()
            if not rows:
                raise _fail("No forecast runs yet. Run `jardik forecast run` first.")
            run_id = rows[0]["run_id"]
        entry = catalogue.get_run(run_id)
    if entry is None:
        raise _fail(f"Unknown run {run_id!r}. See `jardik forecast runs`.")
    folder = Path(entry["output_dir"])
    if not (folder / "run_manifest.json").is_file():
        raise _fail(f"The run folder is missing or incomplete: {folder}")
    dest = build_run_review(folder, out or paths.review_dir)
    typer.echo(f"readable copy of {run_id}:")
    for item in sorted(dest.rglob("*")):
        if item.is_file():
            size = format_number_es(item.stat().st_size / 1e3, 0)
            typer.echo(f"  {item.relative_to(dest)}  ({size} kB)")
    typer.echo(f"folder: {dest}")
