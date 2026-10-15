"""The extraction report (PDF). Numbers follow the Spanish convention (1.234,5)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.shapes import Drawing
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import KeepTogether, Paragraph, Spacer

from jardik.core.formatting import format_percent_es
from jardik.core.pdfkit import BODY, BULLET, H1, SMALL, TITLE, build_document, kv, num, para, table

_p, _table, _kv, _n = para, table, kv, num
MAX_ROWS = 60


def _bar_chart(counts: dict[str, int]) -> Drawing:
    drawing = Drawing(172 * mm, 55 * mm)
    chart = VerticalBarChart()
    chart.x, chart.y, chart.width, chart.height = 12 * mm, 14 * mm, 150 * mm, 36 * mm
    chart.data = [list(counts.values())]
    chart.categoryAxis.categoryNames = list(counts.keys())
    chart.categoryAxis.labels.fontSize = 6.5
    chart.categoryAxis.labels.angle = 30
    chart.categoryAxis.labels.boxAnchor = "ne"
    chart.valueAxis.valueMin = 0
    chart.valueAxis.labels.fontSize = 7
    chart.bars[0].fillColor = colors.HexColor("#4f81bd")
    drawing.add(chart)
    return drawing


def _q(block: dict[str, float] | None, key: str = "median", pct: bool = False) -> str:
    if not block:
        return "n/a"
    return format_percent_es(block[key]) if pct else _n(block[key], 2)


def build_pdf(
    path: Path,
    manifest: dict[str, Any],
    quality: dict[str, Any],
    distribution: dict[str, Any],
) -> None:
    """Write the report. Deterministic: the same inputs give byte-identical output."""
    build_document(
        path,
        _story(manifest, quality, distribution),
        f"Extraction report {manifest['extraction_id']}",
    )


def _story(manifest: dict[str, Any], quality: dict[str, Any], dist: dict[str, Any]) -> list[Any]:
    request, dataset = manifest["request"], manifest["dataset"]
    counts = manifest["counts"]
    weather = manifest["weather"]
    env = manifest["environment"]
    story: list[Any] = [
        _p("Extraction report", TITLE),
        _p(f"Extraction ID: {manifest['extraction_id']}"),
        _p(
            f"Created: {manifest['created_at']}   |   Jardik {env['jardik']}, "
            f"pandas {env['pandas']}",
            SMALL,
        ),
    ]

    # 1 summary
    story += [
        _p("1. Summary", H1),
        _kv(
            [
                ("Dataset", dataset["name"]),
                ("Task", request["task"]),
                ("Period (UTC)", f"{request['start']} to {request['end']} (end excluded)"),
                ("Granularity", f"{_n(request['granularity_s'])} s"),
                (
                    "Buildings available / eligible / selected",
                    f"{_n(dist['buildings']['available'])} / "
                    f"{_n(dist['buildings']['eligible_after_filters'])} / "
                    f"{_n(counts['buildings'])}",
                ),
                ("Load rows", _n(counts["load_rows"])),
                ("Weather variables", ", ".join(weather["variables"])),
                ("Output format", manifest["export"]["format"]),
            ]
        ),
    ]
    for warning in manifest["warnings"]:
        story.append(_p(f"Warning: {warning}", BODY))

    # 2 dataset and licence
    licence_head, _, licence_tail = dataset["licence"].partition(":")
    story += [
        _p("2. Datasets used and licence", H1),
        _p(f"Dataset: {dataset['name']}"),
        _p(f"Source: {dataset['source_url']}"),
        _p(f"Licence: {licence_head}"),
        _p(licence_tail.strip(), SMALL),
        _p(f"Citation: {dataset['citation']}", SMALL),
        _p(f"Timestamp clock of the source: {dataset['timestamp_clock']}"),
        Spacer(1, 3 * mm),
        _table(
            [["File", "Size (MB)", "SHA-256 (start)"]]
            + [
                [f["name"], _n(f["size_bytes"] / 1e6, 1), f["sha256"][:16]]
                for f in dataset["files"]
            ],
            [70, 30, 72],
        ),
    ]

    # 3 filters + funnel
    area = request["area"]
    story += [
        _p("3. Selection filters and funnel", H1),
        _kv(
            [
                ("Usage categories", ", ".join(request["usage_categories"]) or "all"),
                ("Sites", ", ".join(area["site_ids"]) or "all"),
                ("Countries / regions", ", ".join(area["countries"] + area["regions"]) or "all"),
                ("Minimum location quality", request["min_location_quality"] or "any"),
                ("Contracted power (kW)", request["contracted_power_kw"] or "any"),
                ("Number of buildings", request["n_buildings"] or "all that qualify"),
                ("Sampling", f"{request['sampling']} (seed {request['seed']})"),
                (
                    "Max missing ratio after imputation",
                    format_percent_es(request["quality"]["max_missing_ratio"]),
                ),
                ("Max imputed ratio", format_percent_es(request["quality"]["max_imputed_ratio"])),
                ("Imputation", request["quality"]["imputation"]),
            ]
        ),
        Spacer(1, 3 * mm),
        _table(
            [["Step", "Remaining", "Removed", "Detail"]]
            + [
                [f["step"], _n(f["remaining"]), _n(f["removed"]), f["detail"]]
                for f in manifest["funnel"]
            ],
            [38, 20, 20, 94],
        ),
    ]

    # 4 buildings
    selected = manifest["selected_buildings"]
    shown = selected[:MAX_ROWS]
    rows = [["Rank", "Building", "Site", "Usage", "Missing after imputation"]] + [
        [
            b["rank"],
            b["building_id"],
            b["site_id"],
            b["usage_category"],
            format_percent_es(b["residual_missing_ratio"]),
        ]
        for b in shown
    ]
    story += [
        _p("4. Buildings available and selected", H1),
        _p(
            f"{_n(dist['buildings']['available'])} buildings are available in the dataset; "
            f"{_n(counts['buildings'])} were selected."
        ),
        _table(rows, [14, 70, 26, 26, 36]),
    ]
    if len(selected) > MAX_ROWS:
        story.append(
            _p(f"... and {_n(len(selected) - MAX_ROWS)} more (see selected_buildings).", SMALL)
        )

    # 5 period
    tc = quality["temporal_coverage"]
    story += [
        _p("5. Period and granularity", H1),
        _kv(
            [
                ("Window (UTC, half-open)", f"{tc['window_start']} to {tc['window_end']}"),
                ("Slots per building", _n(tc["slots_per_building"])),
                ("First / last value", f"{tc['first_value']} / {tc['last_value']}"),
                ("Time storage", manifest["time_convention"]),
            ]
        ),
    ]

    # 6 schema
    story += [
        _p("6. Column schema of merged_training_dataset", H1),
        _table(
            [["Column", "Type", "Unit", "Description"]]
            + [[c["name"], c["dtype"], c["unit"], c["description"]] for c in manifest["columns"]],
            [34, 22, 22, 94],
        ),
    ]

    # 7 quality
    s = quality["summary"]
    story += [
        _p("7. Data quality", H1),
        _kv(
            [
                ("Contracts", ", ".join(f"{k}: {v}" for k, v in quality["contracts"].items())),
                (
                    "Duplicated rows",
                    ", ".join(f"{k}: {_n(v)}" for k, v in quality["duplicates"].items()),
                ),
                (
                    "Missing ratio (median / max)",
                    f"{_q(s['missing_ratio'], pct=True)} / "
                    f"{_q(s['missing_ratio'], 'max', pct=True)}",
                ),
                (
                    "Imputed ratio (median / max)",
                    f"{_q(s['imputed_ratio'], pct=True)} / "
                    f"{_q(s['imputed_ratio'], 'max', pct=True)}",
                ),
                (
                    "Longest remaining gap, h (median / max)",
                    f"{_q(s['longest_gap_h'])} / {_q(s['longest_gap_h'], 'max')}",
                ),
                ("Buildings excluded by quality or weather", _n(quality["excluded_buildings"])),
            ]
        ),
    ]

    # 8 weather matching
    m, stats = weather["matching"], quality["weather_matching"]["stats"]
    coverage = list(quality["weather_matching"]["coverage_by_building"].values())
    dst = quality["weather_matching"]["dst_days"]
    dst_ok = all(d["load_rows"] == d["matched_rows"] for d in dst)
    story += [
        _p("8. Weather matching", H1),
        _p(
            "Each load record is matched to a weather point and then to a weather record. "
            "Space: the weather series that the dataset links to the building, otherwise the "
            "nearest point by great-circle distance from the best available location "
            "(coordinates, postal centroid, municipality, province, region). "
            "Time: all timestamps are compared in UTC, so daylight saving changes can neither "
            "repeat nor drop a record. Policy and results:"
        ),
        _kv(
            [
                ("Temporal policy", f"{m['method']} (tolerance {_n(stats['tolerance_min'])} min)"),
                ("Maximum distance", f"{_n(m['max_distance_km'], 1)} km"),
                ("Minimum coverage per building", format_percent_es(m["min_coverage"])),
                (
                    "Matched / unmatched rows",
                    f"{_n(stats['matched_rows'])} / {_n(stats['unmatched_rows'])}",
                ),
                (
                    "Coverage per building (min)",
                    format_percent_es(min(coverage)) if coverage else "n/a",
                ),
                (
                    "Time offset, min (min / max)",
                    f"{_n((stats['offset_min'] or {}).get('min'))} / "
                    f"{_n((stats['offset_min'] or {}).get('max'))}",
                ),
                (
                    "Distance, km (min / max)",
                    f"{_n((stats['distance_km'] or {}).get('min'), 1)} / "
                    f"{_n((stats['distance_km'] or {}).get('max'), 1)}",
                ),
                (
                    "DST days audited",
                    f"{_n(len(dst))}; every record kept: {'yes' if dst_ok else 'NO'}",
                ),
                (
                    "Stored per record",
                    "weather_source, matched_lat, matched_lon, distance_km, "
                    "weather_time_offset_min, timezone_used, matching_method, "
                    "spatial_link, location_quality",
                ),
            ]
        ),
    ]

    # 9 weather variables
    rules = weather["value_rules"]
    story += [
        _p("9. Weather variables", H1),
        _table(
            [["Variable", "Unit", "Share missing", "Value rules applied"]]
            + [
                [
                    v,
                    weather["units"].get(v, ""),
                    format_percent_es(stats["value_missing_ratio"].get(v)),
                    "; ".join(f"{k}: {_n(c)}" for k, c in rules.get(v, {}).items()) or "none",
                ]
                for v in weather["variables"]
            ],
            [38, 26, 30, 78],
        ),
    ]

    # 10 geography
    geo = manifest["geographic_coverage"]
    block: list[Any] = [
        _p("10. Geographic coverage", H1),
        _table(
            [["Site", "Buildings", "Latitude", "Longitude", "Timezone", "Location quality"]]
            + [
                [
                    g["site_id"],
                    _n(g["buildings"]),
                    _n(g["lat"], 2),
                    _n(g["lon"], 2),
                    g["timezone"],
                    g["location_quality"],
                ]
                for g in geo
            ],
            [28, 20, 22, 22, 40, 40],
        ),
    ]
    if 1 < len(dist["by_site"]) <= 25:
        block += [Spacer(1, 2 * mm), _p("Buildings per site", SMALL), _bar_chart(dist["by_site"])]
    story.append(KeepTogether(block))
    issues = quality["geographic_consistency"]["issues"]
    story.append(_p(f"Geographic consistency issues flagged: {_n(len(issues))}.", SMALL))
    for c in manifest["corrections"]:
        story.append(
            _p(
                f"Correction applied: site {c['site_id']}, {c['field']} "
                f"{c['old']} -> {c['new']} ({c['reason']}).",
                SMALL,
            )
        )

    # 11-12 limitations, ML
    story.append(_p("11. Limitations", H1))
    story += [Paragraph(escape(n), BULLET, bulletText="-") for n in manifest["limitations"]]
    story.append(_p("12. Possible ML use cases", H1))
    story += [Paragraph(escape(n), BULLET, bulletText="-") for n in manifest["ml_use_cases"]]

    story += [_p("Attribution", H1), _p(manifest["attribution"], SMALL)]
    return story
