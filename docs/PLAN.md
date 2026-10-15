# Project Jardik: Implementation Plan

Status: approved on 2026-10-15. Source of truth for implementation. Changes require an explicit decision logged in `docs/DECISIONS.md`.
Progress: M0, M1 and M2 done (2026-10-16). Deviations from this plan are listed in `docs/DECISIONS.md`.

## 1. Context and problem

Jardik is a Python modular monolith for energy consulting (homes, condominiums, offices, shops). Dataset Manager, NILM and Forecasting live in one codebase with clear interfaces. Models are trained on public datasets, so selecting, cleaning, weather-matching and exporting them is a first-class module. Prototype phase, one technical internal user, limited budget, free tools first, Windows mandatory.

The repository currently holds only `CLAUDE.md` and `docs/BRIEF.md`. Everything below is greenfield.

## 2. Decisions closed (from the planning questions)

| # | Decision | Choice |
|---|---|---|
| 1 | MVP forecast horizon | 24 h, hourly, day-ahead. 48 h, 7 d and rolling are config parameters added later |
| 2 | Target | kWh per hour (energy in the hour). 15 min data is summed to hourly |
| 3 | First real data formats | CSV and Excel, one data contract with explicit column mapping and units |
| 4 | First what-if scenario | Weather (temperature offset), it reuses the trained model directly |
| 5 | Anomaly v1 | Residual-based: actual minus forecast beyond a statistical threshold, plus interpretable rules |
| 6 | MVP UI | Typer CLI first, Streamlit only at the dashboard milestone |
| 7 | Data schema | Proposed in section 6, canonical and unit-explicit |
| 8 | Storage | Parquet files plus DuckDB catalogue. PostgreSQL deferred until an API or multi-user need appears |
| 9 | First dataset family | Building Data Genome 2 (BDG2) |
| 10 | First building types | Offices, then homes (homes needed for NILM) |

Note on decision 9: BDG2 is not Spanish. It is chosen to validate the architecture on a clean, documented, open dataset. Spain-first stays the priority for geography filters, and a Spanish adapter (GoiEner, licence and access to be verified) is milestone M2b.

## 3. Assumptions

- Python 3.11 or 3.12, `pip` plus `venv` (no conda requirement), packaging via `pyproject.toml`.
- One developer, one machine, Windows 11. All paths via `pathlib`, no shell-only assumptions.
- Internal units: energy in kWh, power in kW, temperature in degC, irradiance in W/m2, wind in m/s, humidity in %, pressure in hPa. Any other unit must be declared and converted at ingestion, never guessed.
- Canonical time grid for the MVP is hourly. Native granularity is stored in the raw layer.
- Missing-value policy: never impute silently. Gaps are flagged (`is_missing`, `is_imputed`), imputation is optional, declared in config and counted in the quality report.
- Weather is matched by nearest point in space and time, with declared thresholds. Interpolation is opt-in.
- BDG2 ships its own weather and site coordinates. The first matching implementation still goes through the `WeatherProvider` interface (a file/BDG2 adapter), so Open-Meteo and others plug in later without touching models.
- Spanish number format (comma decimal, period thousands) applies to user-facing reports and exports only. Internal files and Parquet use plain numeric types.
- Reports and docs are in English. Chat with the user is in Italian.
- Licence terms of each source are verified at implementation time and recorded in `LICENSES.md`. Trained-weight restrictions from non-commercial data are noted there too.
- Deep learning libraries (PyTorch, Darts or PyTorch Forecasting) are optional extras, installed from milestone M7.

## 4. Architecture

Modular monolith. One installable package `jardik`. Modules communicate only through typed interfaces and shared schemas in `jardik.core`. No module imports another module's internals.

```
            +-----------------------+
            |        CLI (Typer)    |   later: Streamlit dashboard
            +-----------+-----------+
                        |
   +---------+----------+----------+-----------+-----------+
   |         |          |          |           |           |
 dataset   ingest     features  forecasting   nilm      analytics -> scenarios -> reporting
 manager   (clients)                                      
   |         |          ^          ^           ^
   +----+----+          |          |           |
        v               |          |           |
   providers (weather, solar, market)  <-- cached, chosen in config
        |
        v
   core: schemas (Pydantic), units, time/DST, config, logging, paths, io, catalogue (DuckDB)
```

Dependency rule: `core` depends on nothing. `providers` and `io` depend on `core`. Feature modules (`dataset_manager`, `ingest`, `features`, `forecasting`, `nilm`, ...) depend on `core` and `providers`, and on each other only through the public interfaces below.

Public interfaces (Python `Protocol` classes in `jardik.core.interfaces`):

- `WeatherProvider.get_history(location, start, end, variables) -> WeatherFrame`; `get_forecast(...)` where supported.
- `SolarProvider`, `MarketProvider` with the same shape.
- `DatasetAdapter.discover() / load_metadata() / load_timeseries()` returning canonical frames. One adapter per dataset family (BDG2 first).
- `Forecaster.fit(train) / predict(horizon) / predict_quantiles(...)`. The model ladder implements one interface so backtesting is model-agnostic.
- `Disaggregator.fit / disaggregate` for NILM.

Why this shape (learning note): interfaces make the model ladder honest. Baseline, LightGBM and deep models are swapped behind one `Forecaster` contract and scored by the same backtest code, so "each step must beat the previous" is a mechanical check, not a manual comparison.

Configuration: Pydantic models loaded from YAML (`configs/`). Providers, paths, thresholds, horizons and model parameters live in config. Secrets only via environment variables, documented in `.env.example`.

Reproducibility: every run writes a `run_manifest.json` (config hash, code version, input dataset ids, seeds, timestamps). Every dataset extraction writes `extraction_manifest.json` plus a quality report.

## 5. Repository structure

```
Jardik project/
  CLAUDE.md
  README.md
  LICENSES.md                 # every source, terms, commercial status
  pyproject.toml
  .env.example
  .gitignore                  # data/, .env, *.parquet, caches
  configs/
    default.yaml
    datasets/bdg2.yaml
    providers.yaml
  docs/
    BRIEF.md
    PLAN.md
    DATA_CONTRACTS.md
    DECISIONS.md              # short ADR log
  src/jardik/
    core/
      config.py               # Pydantic settings
      paths.py                # single place for all output paths
      logging.py
      units.py                # explicit unit registry and conversions
      timeutils.py            # UTC/local, DST handling
      schemas/                # Pydantic + pandera-style frame contracts
      interfaces.py           # Protocols
      catalogue.py            # DuckDB catalogue
      io.py                   # Parquet/CSV/Excel readers and writers
    providers/
      base.py
      weather/ (bdg2_file.py, open_meteo.py, aemet.py ...)   # adapters added per milestone
      solar/ market/
      cache.py                # response cache keyed by request hash
    dataset_manager/
      adapters/ (bdg2.py, goiener.py later, ukdale.py ...)
      selection/              # filters, criteria models, selection logic
      matching/               # weather-building-time matching
      quality/                # checks and quality report
      export/                 # parquet/csv, manifest, PDF report
      cli.py
    ingest/                   # client CSV/Excel readers, mapping, validation
    features/                 # calendar, lags, rolling, weather, static
    forecasting/
      baselines.py gbm.py deep/ backtest.py metrics.py
    nilm/
      events.py signatures.py seq2point.py metrics.py
    integration/              # component-based forecasting
    analytics/                # KPIs, residuals, anomalies
    scenarios/
    reporting/
    cli.py                    # top-level Typer app composing module CLIs
  tests/
    unit/ contract/ integration/
    data/                     # tiny synthetic fixtures only, no real datasets
  data/                       # gitignored: raw/ interim/ processed/ cache/ catalogue.duckdb
  outputs/                    # gitignored: extractions/ runs/ reports/
```

File rules from CLAUDE.md applied: no monolithic files (CLI, selection, matching, reporting, IO are separate), output paths centralised in `core/paths.py`, no raw data or keys committed.

## 6. Data schema (canonical layer)

Storage: Parquet files under `data/processed/`, registered in a DuckDB catalogue (`data/catalogue.duckdb`). All timestamps are timezone-aware.

**`building_metadata`** (one row per building)
- `building_id` (str, stable key, `<dataset>:<source_id>`), `dataset_id`, `site_id`
- `usage_category` (enum: home, condominium, office, shop, other), `usage_raw` (original label)
- `country`, `region`, `province`, `municipality`, `postal_code`
- `lat`, `lon`, `location_quality` (enum: coordinates, postal_centroid, municipality, province, region, unknown)
- `timezone` (IANA), `floor_area_m2`, `contracted_power_kw`, `year_built`
- `has_pv`, `has_battery`, `has_ac`, `has_heating` (nullable bool), `heating_fuel`
- `licence_id` (FK to LICENSES.md entry)

**`load_timeseries`** (long format)
- `building_id`, `meter_type` (enum: electricity_total, appliance, ...), `appliance` (nullable, NILM)
- `ts_utc` (timestamp[UTC]), `ts_local` (timestamp with tz), `utc_offset_min` (int)
- `value` (float), `unit` (fixed `kWh` for energy, `W` for NILM power), `granularity_s` (int)
- `is_missing` (bool), `is_imputed` (bool), `quality_flag` (enum)

**`weather_timeseries`**
- `weather_point_id`, `ts_utc`, `temp_c`, `dewpoint_c`, `rh_pct`, `wind_ms`, `wind_dir_deg`, `pressure_hpa`, `precip_mm`, `cloud_pct`, `ghi_wm2`, `dni_wm2`, `dhi_wm2`
- `source`, `product_type` (enum: observation, reanalysis, forecast), `issued_at_utc` (nullable, required for forecasts)

**`weather_match`** (one row per load record or per building-period, as declared)
- `building_id`, `ts_utc`, `weather_source`, `matched_lat`, `matched_lon`, `distance_km`, `weather_time_offset_min`, `timezone_used`, `matching_method`, `location_quality`

**`merged_training_dataset`** = load joined with weather via `weather_match`, plus static building fields and calendar columns (hour, weekday, holiday flag).

**Catalogue tables (DuckDB):** `datasets`, `extractions`, `extraction_buildings`, `runs`, `models`. Each row links to its manifest path and config hash.

**Time and DST rules:** store `ts_utc` as the key. Derive `ts_local` from the building timezone. Repeated local hour in autumn and missing hour in spring are handled explicitly and covered by tests. The hourly grid is built in UTC and never from naive local time.

## 7. Input data contracts

Defined as Pydantic models (config and metadata) and frame validators (data). Documented in `docs/DATA_CONTRACTS.md`. Violations raise typed errors and are listed in the quality report. They are not silently repaired.

1. **Client load file (CSV/Excel)**
   - Required: timestamp column, value column. Optional: building id (else taken from config), meter id.
   - Config declares: column mapping, timestamp format and timezone (or UTC offset), decimal separator, unit (`Wh`, `kWh`, `W`, `kW`, `MWh`), whether the value is interval energy or cumulative register, and granularity.
   - Checks: parseable and timezone-aware timestamps, no duplicate timestamps per meter, monotonic after sort, regular grid or declared irregular, non-negative values (negative allowed only if the building declares PV export), DST-consistent, gap and coverage report.
   - Cumulative registers are converted to interval energy with reset/rollover detection.
2. **Building metadata file (CSV/Excel):** mapped to `building_metadata`. Required: `building_id`, `usage_category`, and at least one location level. `location_quality` is derived, never user-typed.
3. **Weather file (for provider `file`):** variables mapped to canonical names with declared units. Forecast files must carry `issued_at_utc`.
4. **Dataset adapter output contract:** every adapter returns the three canonical frames, which then pass the same validators. This keeps BDG2, GoiEner, UK-DALE and client data on one path.
5. **Extraction request (selection criteria):** Pydantic model covering task, dataset family, country and area, number of buildings, usage category, period, granularity, max missing/imputed ratio, contracted power range, weather variables, export format and output directory. Same model drives the interactive prompt and a YAML file for repeatable runs.

## 8. Weather-building-time matching (critical path)

Algorithm, in `dataset_manager/matching/`:
1. Resolve the best building location by the ladder coordinates, postal centroid, municipality, province, region. Store `location_quality`.
2. Find the nearest weather point in space (haversine), reject beyond `max_distance_km`.
3. Convert load timestamps to UTC and join to weather by policy: `exact`, `nearest_previous` within threshold, `nearest` within threshold, or `interpolate` (opt-in, declared, flagged).
4. Write the match columns listed in the schema, then run a final validation of the join (coverage, offset distribution, distance distribution, DST days).

Tests written first (pytest): exact match, previous/nearest thresholds, offset signs, spring-forward and fall-back days, location fallback ladder, distance rejection, interpolation flagged, missing weather. Metric tests (MAE, RMSE, CV(RMSE), NMBE, pinball) are written before the metrics module.

## 9. Milestones

Each milestone ends with: tests green, `python -m py_compile` on delivered files, a minimal smoke test, and complete files delivered (no fragments). Order follows the brief roadmap.

**M0. Foundations**
- Scaffold, `pyproject.toml`, `core/` (config, paths, logging, units, timeutils, schemas, interfaces), `LICENSES.md` skeleton, `.env.example`, `.gitignore`, `docs/DATA_CONTRACTS.md`, CLI skeleton, pytest set-up, fill in the Commands section of `CLAUDE.md`.
- Tests: units conversion, UTC/local and DST, schema validators.
- Exit: `jardik --help` and `pytest` work on Windows.

**M1. Providers and cache**
- Provider interfaces, response cache, `file` weather adapter. Open-Meteo adapter behind the same interface (non-commercial licence recorded).
- Exit: provider swap by config only; cache hit proven by test.

**M2. Dataset Manager v1 (BDG2)**
- DuckDB catalogue, BDG2 adapter, selection criteria model, interactive CLI plus YAML mode, weather matching, quality checks, export (selected_buildings, building_metadata, load_timeseries, weather_timeseries, merged_training_dataset, debug_counts, distribution_report, extraction_manifest.json), PDF report with Spanish number formatting.
- Exit: one reproducible office extraction end to end, with manifest and report.
- **M2b (optional, after M2):** Spanish dataset adapter (GoiEner) once access and licence are verified.

**M3. Forecasting baselines and GBM**
- Features (calendar, lags, rolling, weather, static), time-and-building split, backtest engine, metrics, baselines (naive lag, hour-by-weekday mean, seasonal naive), LightGBM with quantiles.
- Gate: GBM must beat the best baseline on held-out buildings and periods, otherwise iterate features before moving on.

**M4. Client ingestion and weather enrichment**
- CSV/Excel readers with mapping config, validation, unit standardisation, geocoding hook, enrichment via providers, forecast run on a real building.

**M5. NILM v1**
- Dataset Manager adapter for one open dataset (UK-DALE or REFIT, licence checked), event-based baseline and macro-estimates, seq2point CNN per appliance, evaluation on unseen houses (MAE, SAE, on/off F1).

**M6. Analytics v1**
- KPIs and residual-based anomalies with statistical thresholds and interpretable rules.

**M7. Deep forecasting benchmarks**
- N-HiTS and TFT (Darts or PyTorch Forecasting), probabilistic output, compared with the tabular model on identical splits.

**M8. Integration**
- Component-based forecast (NILM per-appliance forecasts summed) versus single-series forecast.

**M9. Dashboard and consulting report**
- Streamlit dashboard, HTML/PDF consulting report.

**M10. Scenario engine**
- Weather what-if (temperature offset) using the trained model, then tariff.

**M11. Optimisation v1** (peak shaving or tariff cost), only after M3 and M6 are stable.

**M12. Hardening**
- Test coverage review, logging, docs, packaging, Docker evaluation, licence review of data and trained weights.

Note: the brief's order puts deep benchmarks (5) before integration and analytics. Here analytics (M6) comes before deep benchmarks (M7) because residual-based anomalies need only the GBM and deliver client value sooner. This is a swap of the brief's items 5 and 7. Say so if you prefer the original order.

## 10. Risks and mitigations

- Scope creep: one milestone at a time, hard gates on the model ladder, optimisation and scenarios locked behind M3/M6.
- Thin or missing building metadata: `location_quality` and nullable fields are explicit; models report which static features were available.
- Unit and DST errors: central `units.py` and `timeutils.py`, test-first, no inline conversions elsewhere.
- Train/production weather mismatch: `product_type` and `issued_at_utc` are in the schema from day one, so forecast-weather training is possible when a provider supplies it. BDG2 weather is observed, so the first models are trained on observations and this limitation is stated in the extraction report.
- Licences: `LICENSES.md` is updated in the same change that adds any source or dataset.
- Overengineering: no orchestrator, no database server, no UI before M9.

## 11. Verification (applies to every milestone)

- `pytest` (unit, contract, integration) passes. Data-contract, matching and metric tests exist before their implementation.
- `python -m py_compile` on every delivered `.py` file, plus a smoke run of the relevant CLI command.
- M2 acceptance: run the BDG2 office extraction, then check that the manifest lists dataset, filters, counts and config hash, and that `distribution_report` and `debug_counts` reconcile with the selected buildings. Also check that every row of `weather_match` has all required match fields and that DST days show no duplicated or missing hours.
- M3 acceptance: backtest table (MAE, RMSE, CV(RMSE), NMBE, pinball) for every rung of the ladder on buildings and periods unseen in training.

## 12. Next step

Start M0 (scaffold, core modules, tests, CLI skeleton), delivered as complete files.

