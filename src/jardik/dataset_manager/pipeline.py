"""The extraction pipeline: select, screen, assemble, validate and export, in that order.

Outputs are written to a `<name>.partial` folder and moved into place only when everything,
including the manifest, succeeded. A failed run therefore never leaves a folder that looks
like a finished extraction, and a failed `force` rerun leaves the previous extraction intact.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from jardik import __version__
from jardik.core.catalogue import Catalogue
from jardik.core.interfaces import DatasetAdapter
from jardik.core.logging import get_logger
from jardik.core.paths import JardikPaths
from jardik.core.schemas import (
    WEATHER_CANONICAL_UNITS,
    BuildingMetadata,
    validate_load,
    validate_weather,
    validate_weather_match,
)
from jardik.dataset_manager.export import manifest as mf
from jardik.dataset_manager.export.merge import build_merged_dataset
from jardik.dataset_manager.export.pdf_report import build_pdf
from jardik.dataset_manager.export.reports import (
    build_distribution_report,
    build_quality_report,
    funnel_frame,
)
from jardik.dataset_manager.export.writer import write_json, write_report_csv, write_table
from jardik.dataset_manager.matching import MatchResult, match_weather, validate_join
from jardik.dataset_manager.quality.geo import check_geographic_consistency
from jardik.dataset_manager.request import ExtractionRequest
from jardik.dataset_manager.selection.assemble import Assembled, assemble_selection, padded_window
from jardik.dataset_manager.selection.filters import apply_metadata_filters
from jardik.dataset_manager.selection.quality_screen import screen_quality
from jardik.dataset_manager.selection.sampling import order_candidates

log = get_logger("dataset_manager.pipeline")

TABLES = (
    "selected_buildings",
    "building_metadata",
    "load_timeseries",
    "weather_timeseries",
    "weather_match",
    "merged_training_dataset",
    "excluded_buildings",
)
FIXED_FILES = (
    "debug_counts.csv",
    "distribution_report.json",
    "quality_report.json",
    "extraction_manifest.json",
    "extraction_report.pdf",
)


class ExtractionRequestError(ValueError):
    """The request cannot be served by this dataset."""


class NoBuildingsError(RuntimeError):
    """No building survives the selection."""


class ExtractionExistsError(RuntimeError):
    """The extraction folder already holds outputs; pass force to replace them."""


@dataclass
class ExtractionResult:
    extraction_id: str
    output_dir: Path
    manifest: dict[str, Any]
    files: dict[str, Path]


@dataclass
class Selection:
    """Outcome of the selection steps (filters, quality screen, weather, sampling)."""

    eligible: pd.DataFrame
    funnel: list[dict[str, Any]]
    excluded: list[dict[str, Any]]
    pool: pd.DataFrame
    stats: pd.DataFrame
    assembled: Assembled
    warnings: list[str]


@dataclass
class Frames:
    """Every table that is written, plus the validation results that go into the reports."""

    metadata: pd.DataFrame
    selected: pd.DataFrame
    load: pd.DataFrame
    weather: pd.DataFrame
    matched: MatchResult
    join_report: dict[str, Any]
    merged: pd.DataFrame
    contracts: dict[str, str]
    duplicates: dict[str, int]


# --- request and folder checks ---------------------------------------------------------------


def _known_outputs() -> list[str]:
    names = list(FIXED_FILES)
    for stem in TABLES:
        names += [f"{stem}.parquet", f"{stem}.csv"]
    return names


def _check_request(request: ExtractionRequest, info: dict[str, Any]) -> None:
    if request.dataset_id != info["dataset_id"]:
        raise ExtractionRequestError(
            f"The request is for dataset '{request.dataset_id}' but the adapter serves "
            f"'{info['dataset_id']}'"
        )
    supported = info["supported_tasks"]
    if request.task == "both" and "nilm" not in supported:
        raise ExtractionRequestError(
            f"Task 'both' needs NILM data, which dataset {info['dataset_id']} does not have"
        )
    if request.task != "both" and request.task not in supported:
        raise ExtractionRequestError(
            f"Task '{request.task}' is not supported by dataset {info['dataset_id']} "
            f"(supported: {', '.join(supported)})"
        )
    if request.granularity_s != info["native_granularity_s"]:
        raise ExtractionRequestError(
            f"granularity_s={request.granularity_s} is not available: this dataset is "
            f"{info['native_granularity_s']} s and resampling is not implemented yet"
        )


def _check_output(directory: Path, force: bool) -> None:
    existing = [n for n in _known_outputs() if (directory / n).exists()]
    if existing and not force:
        raise ExtractionExistsError(
            f"{directory} already contains extraction outputs. Use force to replace them."
        )


def _publish(work_dir: Path, final_dir: Path) -> None:
    """Move the finished files into place, replacing only files this tool creates."""
    final_dir.mkdir(parents=True, exist_ok=True)
    for name in _known_outputs():
        (final_dir / name).unlink(missing_ok=True)
    for path in sorted(work_dir.iterdir()):
        os.replace(path, final_dir / path.name)
    work_dir.rmdir()


def _empty_message(funnel: list[dict[str, Any]]) -> str:
    for step in funnel:
        if step["remaining"] == 0:
            return (
                f"No building is left after step '{step['step']}' ({step['detail']}). "
                "Relax that filter or check the dataset."
            )
    return "No building is left after selection."


# --- selection -------------------------------------------------------------------------------


def _select(
    request: ExtractionRequest, adapter: DatasetAdapter, available: pd.DataFrame
) -> Selection:
    eligible, funnel = apply_metadata_filters(available, request)
    if eligible.empty:
        raise NoBuildingsError(_empty_message(funnel))

    stats, quality_funnel, excluded = screen_quality(
        adapter, eligible["building_id"].tolist(), request
    )
    funnel += quality_funnel
    if stats.empty:
        raise NoBuildingsError(_empty_message(funnel))
    pool = eligible[eligible["building_id"].isin(stats["building_id"])].reset_index(drop=True)

    order = order_candidates(pool, request.sampling, request.seed)
    assembled = assemble_selection(adapter, request, pool, order, request.n_buildings or len(order))
    excluded = excluded + assembled.excluded

    p, kept = len(pool), len(assembled.accepted)
    after_assignment = p - assembled.rejected_assignment
    after_coverage = after_assignment - assembled.rejected_coverage
    reasons = ", ".join(f"{k}: {v}" for k, v in sorted(assembled.reject_reasons.items())) or "none"
    funnel += [
        {
            "step": "weather_assignment",
            "remaining": after_assignment,
            "removed": assembled.rejected_assignment,
            "detail": f"{assembled.evaluated} candidates evaluated; reasons: {reasons}",
        },
        {
            "step": "weather_coverage",
            "remaining": after_coverage,
            "removed": assembled.rejected_coverage,
            "detail": f"coverage >= {request.weather.matching.min_coverage}",
        },
        {
            "step": "sampling",
            "remaining": kept,
            "removed": after_coverage - kept,
            "detail": f"{request.sampling} order, seed {request.seed}, "
            f"n_buildings {request.n_buildings}",
        },
    ]
    if not assembled.accepted:
        raise NoBuildingsError(_empty_message(funnel))

    warnings: list[str] = []
    if request.n_buildings and kept < request.n_buildings:
        warnings.append(
            f"requested {request.n_buildings} buildings but only {kept} satisfy all criteria"
        )
    return Selection(eligible, funnel, excluded, pool, stats, assembled, warnings)


def _selected_tables(
    assembled: Assembled, pool: pd.DataFrame, stats: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Metadata of the chosen buildings and the `selected_buildings` table, in rank order."""
    rank = {b: i + 1 for i, b in enumerate(assembled.accepted)}
    metadata = (
        pool[pool["building_id"].isin(rank)]
        .assign(_rank=lambda d: d["building_id"].map(rank))
        .sort_values("_rank")
        .drop(columns="_rank")
        .reset_index(drop=True)
    )
    link = assembled.assignment.set_index("building_id")[
        ["weather_point_id", "spatial_link", "distance_km"]
    ]
    columns = [
        "building_id",
        "site_id",
        "usage_category",
        "usage_raw",
        "location_quality",
        "floor_area_m2",
        "timezone",
    ]
    selected = (
        metadata[columns]
        .merge(link, left_on="building_id", right_index=True, how="left")
        .merge(stats, on="building_id", how="left")
    )
    selected.insert(0, "rank", selected["building_id"].map(rank).astype("int64"))
    return metadata, selected


# --- frames and validation -------------------------------------------------------------------


def _validate_outputs(
    metadata: pd.DataFrame, load: pd.DataFrame, weather: pd.DataFrame, match: pd.DataFrame
) -> dict[str, str]:
    fields = set(BuildingMetadata.model_fields)
    for row in metadata.to_dict("records"):
        BuildingMetadata(**{k: (None if pd.isna(v) else v) for k, v in row.items() if k in fields})
    validate_load(load)
    validate_weather(weather)
    validate_weather_match(match)
    return {
        "building_metadata": "passed",
        "load_timeseries": "passed",
        "weather_timeseries": "passed",
        "weather_match": "passed",
    }


def _build_frames(request: ExtractionRequest, sel: Selection) -> Frames:
    assembled = sel.assembled
    metadata, selected = _selected_tables(assembled, sel.pool, sel.stats)
    policy, variables = request.weather.matching, request.weather.variables
    buildings = metadata[["building_id", "timezone", "location_quality"]]
    keys = assembled.load[["building_id", "ts_utc"]]

    # One more match over the whole selection: its statistics go into the reports.
    matched = match_weather(
        load_keys=keys,
        weather=assembled.weather,
        assignment=assembled.assignment,
        buildings=buildings,
        policy=policy,
        variables=variables,
    )
    join_report = validate_join(matched, keys, policy, buildings)

    window = padded_window(request)
    weather = assembled.weather[
        (assembled.weather["ts_utc"] >= window.start) & (assembled.weather["ts_utc"] < window.end)
    ].reset_index(drop=True)
    contracts = _validate_outputs(metadata, assembled.load, weather, matched.match)
    merged = build_merged_dataset(
        assembled.load, matched.values, matched.match, metadata, variables
    )
    duplicates = {
        "load_rows": int(
            assembled.load.duplicated(["building_id", "meter_type", "appliance", "ts_utc"]).sum()
        ),
        "merged_rows": int(merged.duplicated(["building_id", "ts_utc"]).sum()),
    }
    return Frames(
        metadata,
        selected,
        assembled.load,
        weather,
        matched,
        join_report,
        merged,
        contracts,
        duplicates,
    )


# --- manifest helpers ------------------------------------------------------------------------


def _export_info(export: Any) -> dict[str, str]:
    if export.format == "parquet":
        return {"format": "parquet"}
    sep, decimal = (";", ",") if export.csv_style == "spanish" else (",", ".")
    return {"format": "csv", "csv_style": export.csv_style, "sep": sep, "decimal": decimal}


def _geographic_coverage(metadata: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "site_id": site,
            "buildings": len(group),
            "lat": group["lat"].iloc[0],
            "lon": group["lon"].iloc[0],
            "timezone": group["timezone"].iloc[0],
            "location_quality": group["location_quality"].iloc[0],
        }
        for site, group in metadata.groupby("site_id")
    ]


def _build_manifest(
    request: ExtractionRequest,
    adapter: DatasetAdapter,
    info: dict[str, Any],
    sel: Selection,
    frames: Frames,
    final_dir: Path,
    created_at: str,
) -> dict[str, Any]:
    policy, variables = request.weather.matching, request.weather.variables
    corrections = list(adapter.corrections)
    screening = sel.assembled.weather_screening
    unmapped = getattr(adapter, "unmapped_usage_labels", lambda: set())()
    return {
        "extraction_id": request.extraction_id,
        "name": request.name,
        "created_at": created_at,
        "request_hash": request.request_hash(),
        "request": request.model_dump(mode="json"),
        "dataset_id": request.dataset_id,
        "dataset": info,
        "environment": mf.environment(),
        "jardik_version": __version__,
        "output_dir": str(final_dir),
        "counts": {
            "buildings": len(sel.assembled.accepted),
            "load_rows": len(frames.load),
            "weather_rows": len(frames.weather),
            "matched_rows": len(frames.matched.match),
            "available_buildings": int(sel.funnel[0]["remaining"]),
            "eligible_buildings": len(sel.eligible),
            "excluded_buildings": len(sel.excluded),
        },
        "funnel": sel.funnel,
        "selected_buildings": frames.selected[
            ["rank", "building_id", "site_id", "usage_category", "residual_missing_ratio"]
        ].to_dict("records"),
        "geographic_coverage": _geographic_coverage(frames.metadata),
        "corrections": corrections,
        "units": {
            "target_kwh": "kWh per hour",
            **{v: WEATHER_CANONICAL_UNITS[v] for v in variables},
        },
        "time_convention": (
            "ts_utc is the key (UTC). ts_local is local wall-clock time of the building without "
            "timezone, with utc_offset_min (civil time, DST included). The source clock is "
            f"'{info['timestamp_clock']}' and is converted to UTC on load."
        ),
        "weather": {
            "source": "dataset",
            "variables": variables,
            "units": {v: WEATHER_CANONICAL_UNITS[v] for v in variables},
            "matching": policy.model_dump(mode="json"),
            "value_rules": screening,
        },
        "export": _export_info(request.export),
        "columns": mf.column_schema(frames.merged),
        "limitations": mf.build_limitations(
            request, info, frames.metadata, screening, corrections, unmapped
        ),
        "ml_use_cases": mf.ML_USE_CASES[request.task],
        "attribution": mf.attribution(info),
        "warnings": sel.warnings,
    }


# --- execution -------------------------------------------------------------------------------


def _execute(
    request: ExtractionRequest,
    adapter: DatasetAdapter,
    info: dict[str, Any],
    work_dir: Path,
    final_dir: Path,
    created_at: str,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """Do all the work, writing into `work_dir`. Returns the manifest and the files written."""
    available = adapter.discover()
    sel = _select(request, adapter, available)
    frames = _build_frames(request, sel)

    window = request.window
    distribution = build_distribution_report(
        available=len(available),
        eligible=len(sel.eligible),
        selected=frames.selected,
        merged=frames.merged,
    )
    quality = build_quality_report(
        extraction_id=request.extraction_id,
        request_window=(window.start.isoformat(), window.end.isoformat()),
        granularity_s=request.granularity_s,
        selected_stats=frames.selected,
        contracts=frames.contracts,
        duplicates=frames.duplicates,
        geo_issues=check_geographic_consistency(frames.metadata, info.get("bbox_by_timezone", {})),
        corrections=list(adapter.corrections),
        policy=request.weather.matching,
        match_stats=frames.matched.stats,
        join_report=frames.join_report,
        weather_screening=sel.assembled.weather_screening,
        excluded_count=len(sel.excluded),
    )

    export = request.export
    files: dict[str, Path] = {}
    rows: dict[str, int] = {}
    excluded_df = pd.DataFrame(sel.excluded, columns=["building_id", "step", "detail"])
    for stem, frame in (
        ("selected_buildings", frames.selected),
        ("building_metadata", frames.metadata),
        ("load_timeseries", frames.load),
        ("weather_timeseries", frames.weather),
        ("weather_match", frames.matched.match),
        ("merged_training_dataset", frames.merged),
        ("excluded_buildings", excluded_df),
    ):
        files[stem] = write_table(frame, work_dir, stem, export)
        rows[stem] = len(frame)
    files["debug_counts"] = write_report_csv(
        funnel_frame(sel.funnel), work_dir / "debug_counts.csv", export
    )
    files["distribution_report"] = write_json(distribution, work_dir / "distribution_report.json")
    files["quality_report"] = write_json(quality, work_dir / "quality_report.json")

    # The manifest is written last because it lists the checksum of every other file.
    manifest = _build_manifest(request, adapter, info, sel, frames, final_dir, created_at)
    if export.report_pdf:
        files["extraction_report"] = work_dir / "extraction_report.pdf"
        build_pdf(files["extraction_report"], manifest, quality, distribution)
    manifest["files"] = [
        mf.file_entry(path, work_dir, rows.get(stem)) for stem, path in sorted(files.items())
    ]
    files["extraction_manifest"] = write_json(manifest, work_dir / "extraction_manifest.json")
    return manifest, files


def run_extraction(
    request: ExtractionRequest,
    adapter: DatasetAdapter,
    paths: JardikPaths,
    catalogue: Catalogue | None = None,
    *,
    force: bool = False,
    now: Callable[[], datetime] | None = None,
) -> ExtractionResult:
    """Run one extraction end to end and return where everything was written."""
    clock = now or (lambda: datetime.now(UTC))
    info = adapter.info()
    _check_request(request, info)

    final_dir = request.export.output_dir or paths.extraction_dir(request.extraction_id)
    if not final_dir.is_absolute():
        final_dir = paths.root / final_dir
    _check_output(final_dir, force)
    work_dir = final_dir.with_name(final_dir.name + ".partial")
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True)
    log.info("extraction %s -> %s", request.extraction_id, final_dir)

    try:
        manifest, written = _execute(
            request, adapter, info, work_dir, final_dir, clock().isoformat()
        )
        _publish(work_dir, final_dir)
    except BaseException:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise

    files = {stem: final_dir / path.name for stem, path in written.items()}
    if catalogue is not None:
        catalogue.register_extraction(manifest)
    counts = manifest["counts"]
    log.info(
        "extraction %s done: %d buildings, %d rows",
        request.extraction_id,
        counts["buildings"],
        counts["load_rows"],
    )
    return ExtractionResult(request.extraction_id, final_dir, manifest, files)
