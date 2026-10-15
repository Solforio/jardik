# Data contracts

Canonical tables used by every module. Implemented in `src/jardik/core/schemas/`. A violation raises
`ContractError` listing every problem; data is never repaired silently.

General rules (from CLAUDE.md): timestamps are timezone-aware, UTC is the key and local time is
derived; units are explicit and canonical; every extraction produces a manifest and a quality report.

## Canonical units

| Quantity | Unit | Column examples |
|---|---|---|
| Energy | kWh | `load_timeseries.value` (electricity_total) |
| Power | W (NILM appliance signals), kW elsewhere | `load_timeseries.value` (electricity_appliance) |
| Temperature | degC | `temp_c`, `dewpoint_c` |
| Wind speed | m/s | `wind_ms` |
| Pressure | hPa | `pressure_hpa` |
| Irradiance | W/m2 | `ghi_wm2`, `dni_wm2`, `dhi_wm2` |
| Percent | % (0-100) | `rh_pct`, `cloud_pct` |
| Precipitation | mm per interval | `precip_mm` |

Conversions live only in `jardik.core.units`. Unknown units raise; they are never guessed.
Forecast target: kWh per hour (energy in the hour). Mean kW over an hour is numerically equal but
is only produced through `energy_to_mean_power_kw`.

## building_metadata (one row per building)

Pydantic model `BuildingMetadata`. Key: `building_id` = `<dataset_id>:<source_id>`.
`location_quality` is computed from the available fields (coordinates, postal_centroid,
municipality, province, region, unknown) and cannot be supplied by the user. Unknown values are null.

## load_timeseries (long format)

Contract `LOAD_CONTRACT`. Unique key: `building_id, meter_type, appliance, ts_utc`.

- `ts_utc` is tz-aware UTC (the key). `ts_local` is the local wall-clock time of the building **without a
  timezone** and `utc_offset_min` is its civil UTC offset (DST included): `ts_local = ts_utc + offset`.
  A pandas/Arrow column holds one timezone only, while a dataset spans many; the offset keeps the
  repeated autumn hour unambiguous and the IANA zone lives in `building_metadata.timezone`.
- `meter_type` fixes the unit: `electricity_total` and `electricity_pv_generation` use `kWh`,
  `electricity_appliance` uses `W`.
- `is_missing` implies a null `value`, and a null `value` requires `is_missing`. A row cannot be both
  `is_missing` and `is_imputed`.
- `granularity_s` > 0.

## weather_timeseries

Contract `WEATHER_CONTRACT`. Variable columns are optional but, when present, are range-checked
(for example `rh_pct` in [0, 100]). `product_type` is observation, reanalysis or forecast; forecast
rows must carry `issued_at_utc` so models can be trained on what was known at issue time.

## weather_match

Contract `WEATHER_MATCH_CONTRACT`. Unique key: `building_id, ts_utc`. One row per load record that got
weather. Records `weather_source`, `matched_lat`, `matched_lon`, `distance_km` (>= 0),
`weather_time_offset_min` (weather time minus slot time), `timezone_used`, `matching_method`
(exact, nearest_previous, nearest, interpolate), `spatial_link` and `location_quality`.
`spatial_link` says how the weather point was chosen: `nearest_point` (by distance from known
coordinates, geometry must be known) or `dataset_link` (the dataset itself ties the building to a
weather series; coordinates and distance may be unknown and are then null).

## Weather providers (M1)

Providers return frames that satisfy `WEATHER_CONTRACT` and are selected only by `providers.weather`
in the config (`file` or `open_meteo`).

- Canonical variable names and units are in `WEATHER_CANONICAL_UNITS`. Providers convert to them.
- `issued_at_utc` is always present: null for observation and reanalysis, required for forecast.
- Windows are half-open `[start, end)` and timezone-aware.
- **File provider** (`providers.file_weather`): every mapped column needs a declared unit. Naive
  timestamps need `timezone` and follow `time_policy`; a file mixing naive and offset timestamps is
  rejected. A file can hold one fixed point (`lat`, `lon`) or many (`point_id_col`, `lat_col`, `lon_col`);
  `get_history` returns the point nearest to the requested location and `points()` lists all points.
  Forecast files need `issued_at_col`.
- **Open-Meteo provider**: requests UTC and SI units, then converts using the units the response
  declares; a missing or unknown unit stops the run. The API snaps requests to its own grid, so the
  returned point id carries the grid coordinates (`open_meteo:<lat>:<lon>`). The archive is
  `product_type = reanalysis`. Forecast `issued_at_utc` is the request time floored to the hour (the
  API does not expose the model run), which is an approximation.
- Responses are cached on disk (`data/cache/providers/<provider>/<hash>.json`). History never expires;
  forecasts expire after `open_meteo.forecast_ttl_hours`.

## DST policy

Naive local timestamps are rejected unless localised with an explicit policy
(`time_policy.ambiguous`: raise, dst, standard, nat; `time_policy.nonexistent`: raise,
shift_forward, nat). Default is `raise`. Hourly grids are built in UTC; DST days have 23 or 25 hours.

## Dataset adapters (M2)

Every adapter implements `DatasetAdapter` (`jardik.core.interfaces`): `info()`, `discover()`,
`load_metadata()`, `load_wide()` (hourly energy on one UTC grid, one column per building, NaN = missing),
`load_timeseries()` (the same in long form, satisfying `LOAD_CONTRACT`), `weather_points()` and
`load_weather()` (satisfying `WEATHER_CONTRACT`). Rows of `discover()` carry the `BuildingMetadata`
fields plus `usage_sub_raw`, `weather_link` and `location_quality`. Adapters also expose `corrections`
(declared fixes of the source data) and `weather_screening` (counts of weather values changed by the
adapter's value rules); both always end up in the manifest.

A missing slot is never dropped: the long frame keeps one row per hourly slot of the requested window
with `is_missing = True`, so gaps can be counted. Imputed slots carry `is_imputed = True` and
`quality_flag = imputed`.

## Extraction request (M2)

`ExtractionRequest` (`jardik.dataset_manager.request`), loadable from YAML (example:
`configs/requests/example_offices_2016.yaml`) or built interactively. The request hash covers everything
that influences the result except the output folder; the extraction id is `<name>-<hash>`.

- `start` / `end`: UTC calendar days, half-open `[start, end)`. `granularity_s` must equal the
  dataset's native granularity (resampling is not implemented yet).
- Filters: `usage_categories`, `area` (site_ids, countries, regions, provinces, municipalities,
  postal_codes, bbox), `min_location_quality`, `contracted_power_kw`. A building with an unknown value
  for a filtered field is excluded, and the funnel says how many.
- `quality.max_missing_ratio` is the share of slots still empty AFTER the optional imputation;
  `quality.max_imputed_ratio` is the share of imputed slots. Imputation (`linear_short_gaps`) fills only
  interior gaps of at most `max_gap_hours`.
- `quality.max_near_zero_ratio` (default 0.1): max share of readings at or below 1 % of the building's
  median. Catches partial or dead meters that exact-zero checks miss; the statistic `near_zero_ratio` is
  in the quality report.
- `weather.matching`: `method`, `max_time_offset_min`, `max_distance_km`, `min_coverage` (a building
  below it is dropped and replaced by the next candidate).
- Candidates are ordered deterministically (`random` seeded, `stratified_site`, `first`), so a request
  always selects the same buildings.

## Extraction outputs (M2)

In `outputs/extractions/<extraction_id>/` (written to a `.partial` folder and moved into place only on
success): `selected_buildings`, `building_metadata`, `load_timeseries`, `weather_timeseries`,
`weather_match`, `merged_training_dataset`, `excluded_buildings` (Parquet or CSV), `debug_counts.csv` (the
funnel), `distribution_report.json`, `quality_report.json`, `extraction_manifest.json` (request, dataset
info and licence, checksums of every file, units, corrections, limitations, attribution) and
`extraction_report.pdf`.

`merged_training_dataset` has one row per hourly slot: keys, `target_kwh` (kWh per hour), flags,
the matched weather variables and `weather_time_offset_min`, static building fields and calendar
columns computed from the local wall clock. Column documentation is in the manifest.

CSV convention (`export.csv_style`): `spanish` = `;` separator and decimal comma (default, as required for
user-facing exports); `plain` = `,` and decimal point. Parquet always uses plain numbers. The manifest
records the dialect.

## Building Data Genome 2 specifics

- Licence CC BY-SA 4.0: the manifest and PDF carry the attribution; extractions are adaptations.
- Timestamps are local standard time with DST ignored (one continuous clock, no gap or repeat);
  converted with `standard_time_to_utc` (Irish winter time is GMT, not IST).
- Coordinates are per site and city-level; four sites have none (matched through the dataset link).
  Site Wolf has a longitude with the wrong sign, corrected by a declared override.
- Weather rules (counted in the manifest): precipitation -1 (trace) becomes 0; cloud cover in oktas
  becomes percent with 9 (sky obscured) mapped to 100; values outside the plausible range become NaN.
  Sea-level pressure is `pressure_msl_hpa`, not `pressure_hpa`.
- No country is given, so holiday flags are not included; contracted power is unknown.

## Forecasting (M3)

Input: the `merged_training_dataset` of an extraction (its SHA-256 is verified before use). Code in
`src/jardik/forecasting/`.

- **Feature frame** (`build_features`): one row per hourly slot. Keys `building_id`, `site_id`, `ts_utc`,
  `ts_local`; `target_kwh`; flags `is_measured` (measured, not imputed) and `has_history` (the scale
  exists); `hours_ahead` (slots since local midnight); `scale_kwh`; raw lags `lag_24 ... lag_672` and
  `base_kwh` (for baselines); normalised model features `n_lag_*`, `n_last`, `n_mean_24`, `n_std_168`,
  `n_max_24`, `n_min_24`, `n_base`; calendar, weather and static features. `model_feature_columns` lists
  what the models see (never raw kWh).
- **Splits** (`SplitSpec`): UTC dates `train_end`, `val_end`, optional `test_end`; fractions of buildings
  for validation and test; `by: building | site`; `seed`. Scenarios: train, val, test_time,
  test_building, test_both.
- **Run outputs** in `outputs/runs/<run_id>/` (`.partial` folder, moved into place on success):
  `predictions.parquet` (point and quantile forecasts of every model), `metrics.csv`,
  `macro_metrics.csv`, `per_building_metrics.csv`, `skill.csv`, `feature_importance.csv`,
  `split_roles.csv`, `gate.json`, `run_manifest.json` (config hash, extraction checksum, licence,
  limitations, checksums), `forecast_report.pdf`, `figures/`, `models/lightgbm/`.
- **Metrics** (`forecasting/metrics.py`): MAE, RMSE, CV(RMSE) = RMSE / mean load, NMBE with the ASHRAE
  Guideline 14 sign (positive = under-forecast), pinball loss, 80 % interval coverage and width, skill
  score = 1 - model error / baseline error. Pairs with a missing value are ignored; no pair gives NaN.

## Added in later milestones

Client file contract (M4) will be documented here when implemented.
