"""`jardik dataset ...` commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from jardik.core.catalogue import Catalogue
from jardik.core.config import AppConfig
from jardik.core.formatting import format_number_es
from jardik.core.paths import JardikPaths
from jardik.dataset_manager.adapters.base import DatasetError
from jardik.dataset_manager.adapters.bdg2_download import fetch_bdg2
from jardik.dataset_manager.adapters.registry import (
    available_datasets,
    build_adapter,
    raw_data_dir,
)
from jardik.dataset_manager.export.review import build_review
from jardik.dataset_manager.interactive import build_request_interactively
from jardik.dataset_manager.pipeline import (
    ExtractionExistsError,
    ExtractionRequestError,
    NoBuildingsError,
    run_extraction,
)
from jardik.dataset_manager.request import ExtractionRequest

dataset_app = typer.Typer(
    help="Dataset Manager: fetch, inspect and extract datasets.", no_args_is_help=True
)


def _fail(message: str, code: int = 1) -> typer.Exit:
    typer.echo(f"Error: {message}", err=True)
    return typer.Exit(code=code)


def _state(ctx: typer.Context) -> tuple[AppConfig, JardikPaths]:
    return ctx.obj["config"], ctx.obj["paths"]


@dataset_app.command("list")
def list_datasets(ctx: typer.Context) -> None:
    """List the supported datasets and those registered in the local catalogue."""
    _, paths = _state(ctx)
    typer.echo("Supported datasets: " + ", ".join(available_datasets()))
    with Catalogue(paths.catalogue_path) as catalogue:
        registered = catalogue.list_datasets()
    if not registered:
        typer.echo("Registered: none yet (run `jardik dataset fetch bdg2`)")
    for row in registered:
        typer.echo(
            f"  {row['dataset_id']:8s} {row['name']}  licence={row['licence_id']}  "
            f"registered {row['registered_at'][:19]}"
        )


@dataset_app.command("fetch")
def fetch(
    ctx: typer.Context,
    dataset_id: Annotated[str, typer.Argument(help="Dataset to download (bdg2).")],
    variant: Annotated[str, typer.Option(help="cleaned or raw electricity file.")] = "cleaned",
) -> None:
    """Download a dataset with checksum verification and register it in the catalogue."""
    cfg, paths = _state(ctx)
    if dataset_id != "bdg2":
        raise _fail(f"No downloader for {dataset_id!r}. Supported: bdg2")
    if variant not in ("cleaned", "raw"):
        raise _fail("--variant must be 'cleaned' or 'raw'", 2)
    try:
        files = fetch_bdg2(raw_data_dir(paths, dataset_id), variant)  # type: ignore[arg-type]
    except DatasetError as exc:
        raise _fail(str(exc)) from exc
    for f in files:
        typer.echo(
            f"{f['status']:10s} {f['name']}  {format_number_es(f['size_bytes'] / 1e6, 1)} MB  "
            f"sha256 {f['sha256'][:16]}"
        )
    try:
        info = build_adapter(dataset_id, cfg, paths).info()
        with Catalogue(paths.catalogue_path) as catalogue:
            catalogue.register_dataset(info, info["files"])
        typer.echo(f"Registered {dataset_id} in {paths.catalogue_path}")
    except DatasetError as exc:
        typer.echo(f"Downloaded, but not registered: {exc}", err=True)


@dataset_app.command("describe")
def describe(
    ctx: typer.Context,
    dataset_id: Annotated[str, typer.Argument()] = "bdg2",
) -> None:
    """Show licence, clock convention and what the dataset offers."""
    cfg, paths = _state(ctx)
    try:
        adapter = build_adapter(dataset_id, cfg, paths)
        info, catalogue = adapter.info(), adapter.discover()
    except DatasetError as exc:
        raise _fail(str(exc)) from exc
    typer.echo(f"{info['name']}\n  licence: {info['licence']}\n  source:  {info['source_url']}")
    typer.echo(
        f"  granularity: {info['native_granularity_s']} s   clock: {info['timestamp_clock']}"
    )
    typer.echo(
        f"  coverage (local): {info['time_coverage_local'][0]} .. {info['time_coverage_local'][1]}"
    )
    typer.echo(f"  tasks: {', '.join(info['supported_tasks'])}")
    typer.echo(f"  coordinates: {info['coordinate_precision']}")
    typer.echo(f"  buildings with electricity: {format_number_es(len(catalogue), 0)}")
    typer.echo(
        "  by usage: "
        + ", ".join(f"{k}={v}" for k, v in catalogue["usage_category"].value_counts().items())
    )
    typer.echo(f"  sites: {catalogue['site_id'].nunique()}")
    for correction in adapter.corrections:
        typer.echo(
            f"  correction: {correction['site_id']} {correction['field']} "
            f"{correction['old']} -> {correction['new']}"
        )


@dataset_app.command("validate-request")
def validate_request(
    request_file: Annotated[Path, typer.Argument(help="Request YAML file.")],
) -> None:
    """Check a request file and print its hash and extraction id."""
    try:
        request = ExtractionRequest.from_yaml(request_file)
    except (ValidationError, OSError, ValueError) as exc:
        raise _fail(str(exc), 2) from exc
    typer.echo(f"OK  extraction id {request.extraction_id}")


@dataset_app.command("extract")
def extract(
    ctx: typer.Context,
    request_file: Annotated[
        Path | None, typer.Option("--request", "-r", help="Request YAML file.")
    ] = None,
    interactive: Annotated[
        bool, typer.Option("--interactive", "-i", help="Answer questions instead.")
    ] = False,
    force: Annotated[bool, typer.Option("--force", help="Replace an existing extraction.")] = False,
) -> None:
    """Run an extraction from a request file or an interactive dialogue."""
    cfg, paths = _state(ctx)
    if (request_file is None) == (not interactive):
        raise _fail("give either --request FILE or --interactive", 2)

    try:
        if request_file is not None:
            request = ExtractionRequest.from_yaml(request_file)
            adapter = build_adapter(request.dataset_id, cfg, paths)
        else:
            adapter = build_adapter(typer.prompt("Dataset", default="bdg2"), cfg, paths)
            request = build_request_interactively(
                adapter.info(), adapter.discover(), typer.prompt, typer.echo
            )
            saved = paths.requests_dir / f"{request.name}.yaml"
            saved.parent.mkdir(parents=True, exist_ok=True)
            saved.write_text(request.to_yaml(), encoding="utf-8")
            typer.echo(f"\nRequest saved to {saved} (replay with --request)\n")
            if not typer.confirm("Run the extraction now?", default=True):
                raise typer.Exit(code=0)
    except (ValidationError, ValueError, DatasetError) as exc:
        raise _fail(str(exc), 2) from exc

    try:
        with Catalogue(paths.catalogue_path) as catalogue:
            result = run_extraction(request, adapter, paths, catalogue, force=force)
    except (NoBuildingsError, ExtractionRequestError, ExtractionExistsError, DatasetError) as exc:
        raise _fail(str(exc)) from exc

    counts = result.manifest["counts"]
    typer.echo(f"extraction {result.extraction_id}")
    typer.echo(
        f"  buildings {format_number_es(counts['buildings'], 0)}   "
        f"load rows {format_number_es(counts['load_rows'], 0)}"
    )
    typer.echo(f"  output    {result.output_dir}")
    for warning in result.manifest["warnings"]:
        typer.echo(f"  warning: {warning}")


@dataset_app.command("extractions")
def list_extractions(ctx: typer.Context) -> None:
    """List extractions registered in the catalogue, newest first."""
    _, paths = _state(ctx)
    with Catalogue(paths.catalogue_path) as catalogue:
        rows = catalogue.list_extractions()
    if not rows:
        typer.echo("No extractions yet.")
    for r in rows:
        typer.echo(
            f"{r['created_at'][:19]}  {r['extraction_id']}  {r['dataset_id']}  "
            f"{format_number_es(r['n_buildings'], 0)} buildings  "
            f"{format_number_es(r['n_rows'], 0)} rows"
        )


@dataset_app.command("review")
def review(
    ctx: typer.Context,
    extraction_id: Annotated[
        str | None, typer.Argument(help="Extraction id; default: the newest one.")
    ] = None,
    out: Annotated[
        Path | None, typer.Option(help="Folder to write into (default: review/).")
    ] = None,
) -> None:
    """Build a readable copy of an extraction: Excel workbook, PDF, sample CSV and JSON reports."""
    _, paths = _state(ctx)
    with Catalogue(paths.catalogue_path) as catalogue:
        if extraction_id is None:
            rows = catalogue.list_extractions()
            if not rows:
                raise _fail("No extractions yet. Run `jardik dataset extract` first.")
            extraction_id = rows[0]["extraction_id"]
        entry = catalogue.get_extraction(extraction_id)
    if entry is None:
        raise _fail(f"Unknown extraction {extraction_id!r}. See `jardik dataset extractions`.")
    folder = Path(entry["output_dir"])
    if not (folder / "extraction_manifest.json").is_file():
        raise _fail(f"The extraction folder is missing or incomplete: {folder}")
    dest = build_review(folder, out or paths.review_dir)
    typer.echo(f"readable copy of {extraction_id}:")
    for item in sorted(dest.rglob("*")):
        if item.is_file():
            typer.echo(
                f"  {item.relative_to(dest)}  ({format_number_es(item.stat().st_size / 1e3, 0)} kB)"
            )
    typer.echo(f"folder: {dest}")
