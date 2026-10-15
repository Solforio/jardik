"""Top-level command line interface. Module CLIs (dataset, forecast, ...) are added as sub-apps."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from jardik import __version__
from jardik.core.config import AppConfig, ConfigError, config_hash, load_config, load_env
from jardik.core.logging import get_logger, setup_logging
from jardik.core.paths import JardikPaths
from jardik.dataset_manager.cli import dataset_app
from jardik.providers.cli import weather_app

app = typer.Typer(
    name="jardik",
    help="Project Jardik: dataset management, forecasting and NILM for energy consulting.",
    no_args_is_help=True,
    add_completion=False,
)

app.add_typer(weather_app, name="weather")
app.add_typer(dataset_app, name="dataset")

log = get_logger("cli")


@app.callback()
def main(
    ctx: typer.Context,
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="Path to a YAML config file.")
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,
) -> None:
    """Load configuration and logging before any command runs."""
    load_env()
    try:
        cfg = load_config(config)
    except ConfigError as exc:
        typer.echo(f"Configuration error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    paths = cfg.build_paths()
    level = "DEBUG" if verbose else cfg.logging.level
    setup_logging(level, paths.log_file() if cfg.logging.log_to_file else None)
    ctx.obj = {"config": cfg, "paths": paths}
    log.debug("Config loaded (hash %s)", config_hash(cfg))


@app.command()
def version() -> None:
    """Print the Jardik version."""
    typer.echo(f"jardik {__version__}")


@app.command()
def info(ctx: typer.Context) -> None:
    """Show the active configuration, key paths and providers."""
    cfg: AppConfig = ctx.obj["config"]
    paths: JardikPaths = ctx.obj["paths"]
    typer.echo(f"version        : {__version__}")
    typer.echo(f"config hash    : {config_hash(cfg)}")
    typer.echo(f"project root   : {paths.root}")
    typer.echo(f"data dir       : {paths.data_dir}")
    typer.echo(f"outputs dir    : {paths.outputs_dir}")
    typer.echo(f"weather provider: {cfg.providers.weather}")
    typer.echo(
        f"forecast       : {cfg.forecast.horizon_h} h, {cfg.forecast.granularity_s} s, "
        f"{cfg.forecast.target_unit}"
    )


@app.command("check-config")
def check_config(
    path: Annotated[Path, typer.Argument(help="YAML config file to validate.")],
) -> None:
    """Validate a config file and print its hash."""
    try:
        cfg = load_config(path)
    except Exception as exc:  # pydantic ValidationError or ConfigError: show it, exit non-zero
        typer.echo(f"INVALID: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"OK (hash {config_hash(cfg)})")


if __name__ == "__main__":
    app()
