"""`jardik weather ...` commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from jardik.core.config import AppConfig
from jardik.core.formatting import format_number_es
from jardik.core.interfaces import Location, TimeWindow
from jardik.core.paths import JardikPaths
from jardik.core.schemas import WEATHER_VARIABLES
from jardik.providers.base import ProviderError
from jardik.providers.registry import build_weather_provider

weather_app = typer.Typer(help="Weather providers.", no_args_is_help=True)


def _utc(text: str) -> pd.Timestamp:
    stamp = pd.Timestamp(text)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


@weather_app.command("fetch")
def fetch(
    ctx: typer.Context,
    lat: Annotated[float, typer.Option(help="Latitude in degrees.")],
    lon: Annotated[float, typer.Option(help="Longitude in degrees.")],
    start: Annotated[str, typer.Option(help="Start (UTC unless an offset is given), inclusive.")],
    end: Annotated[str, typer.Option(help="End (UTC unless an offset is given), exclusive.")],
    variables: Annotated[
        str, typer.Option(help="Comma-separated canonical variables.")
    ] = "temp_c,rh_pct",
    forecast: Annotated[bool, typer.Option("--forecast", help="Request a forecast.")] = False,
    out: Annotated[Path | None, typer.Option(help="Write the result to this CSV file.")] = None,
) -> None:
    """Fetch weather through the provider selected in the config (providers.weather)."""
    cfg: AppConfig = ctx.obj["config"]
    paths: JardikPaths = ctx.obj["paths"]
    names = [v.strip() for v in variables.split(",") if v.strip()]
    unknown = [v for v in names if v not in WEATHER_VARIABLES]
    if unknown:
        typer.echo(f"Unknown variable(s) {unknown}. Valid: {sorted(WEATHER_VARIABLES)}", err=True)
        raise typer.Exit(code=2)

    try:
        provider = build_weather_provider(cfg, paths)
        window = TimeWindow(_utc(start), _utc(end))
        location = Location(lat, lon)
        frame = (
            provider.get_forecast(location, window, names)
            if forecast
            else provider.get_history(location, window, names)
        )
    except (ProviderError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"provider {provider.name}: {format_number_es(len(frame), 0)} rows")
    if len(frame):
        typer.echo(frame.head(5).to_string(index=False))
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out, index=False)
        typer.echo(f"written: {out}")
