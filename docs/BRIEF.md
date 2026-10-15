# Project Jardik: Energy Analysis and Optimisation Software

Status: prototype / MVP planning. This brief replaces earlier handoff notes and is the single source of truth for scope. Language of all docs and code: English.

## 1. Vision

Python software that supports energy-efficiency consulting for homes, condominiums, offices and shops. It analyses consumption, forecasts demand, disaggregates loads (NILM) and identifies optimisation opportunities.

- First user: an internal technical user with an energy-engineering background (hybrid wind-PV-BESS systems, energyPRO, WindPRO).
- Not a commercial product yet. Goal: a robust technical prototype that validates the value of the software, teaches real ML and software engineering, and sets a serious architecture for later versions.
- Budget is limited. Prefer free and open-source tools.

## 2. Core principle

The Dataset Manager, the NILM engine and the Forecasting engine are all part of the same base software. The Dataset Manager is not a side tool: the models are trained on public datasets, so selecting, cleaning, weather-enriching and exporting those datasets is a first-class module.

## 3. Modules

1. **Dataset Manager (core)**: discover, select, download or register, clean, weather-enrich and export public datasets for training. See section 4.
2. **Client data ingestion**: load smart-meter data (CSV, Excel, later databases) with validation, timezone-aware timestamps and unit standardisation, plus building metadata.
3. **External data layer**: weather (history and forecast), solar radiation, geocoding, prices. Provider-agnostic, see section 8.
4. **Feature engineering**: calendar and time features, lags, rolling statistics, weather features, static building features.
5. **Forecasting engine**: baselines, gradient boosting, deep learning, backtesting, metrics. See section 5.
6. **NILM engine**: load disaggregation, trained and evaluated through the Dataset Manager. See section 6.
7. **Integration**: component-based forecasting (NILM output feeds per-appliance forecasts, summed and compared with the single-series forecast).
8. **Analytics**: energy KPIs, residual analysis, anomaly detection.
9. **Scenario engine**: simple what-if (weather, tariff, setpoint, occupancy).
10. **Reporting and UI**: technical dashboard, consulting report export (PDF/HTML).
11. **Optimisation (later)**: peak shaving, tariff cost, self-consumption, PV + battery. Only after forecasting and KPIs are stable.

## 4. Dataset Manager requirements

Purpose: build curated, ML-ready training datasets from public sources, with full traceability.

**Interactive selection** (CLI/TUI first):
- task: forecasting, NILM or both
- dataset or dataset family
- priority country (Spain first, then Europe, then others)
- area: region, province, municipality, postal code when available
- number of buildings or meters
- building type or usage category
- period and target duration
- temporal granularity
- maximum missing or imputed data ratio
- contracted power range, when available
- weather variables to include
- export format and output directory

**Weather-building-time matching** (critical, must be tested):
- Use the best available location: real coordinates, then postal-code centroid, municipality, province, region. Store a location-quality indicator.
- Timezone-aware timestamps. Handle DST explicitly. Keep both UTC and local timestamps.
- Match each load record to the nearest weather record in space and time.
- Store: weather_source, matched_lat, matched_lon, distance_km, weather_time_offset_min, timezone_used, matching_method.
- Temporal policies: exact match; nearest previous within threshold; nearest within threshold; interpolation only if optional and declared; final validation of the temporal join.

**Quality controls**: input schema validation, required columns, duplicates, temporal coverage, missing ratio, percentage of imputed data, geographic consistency, final quality report.

**Outputs**: selected_buildings, building_metadata, load_timeseries (CSV or Parquet), weather_timeseries, merged_training_dataset, debug_counts, distribution_report, extraction_manifest.json, extraction_report.pdf. The PDF documents datasets used, filters, buildings available and selected, period, granularity, column schema, data quality, weather-matching logic, weather variables, geographic coverage, limitations and possible ML use cases.

**Local catalogue**: DuckDB (or SQLite) catalogue of datasets and extractions; Pydantic for config and validation; standard Python logging.

## 5. Forecasting strategy

Target: hourly electrical demand (kWh or mean kW per hour, to be decided), 15-minute data supported later. Horizons under evaluation: 24 h, 48 h, 7 days, rolling.

Model ladder (each step must beat the previous one on held-out data before moving on):
1. Baselines: naive lag, historical mean by hour and weekday, seasonal naive.
2. Gradient boosting (LightGBM or XGBoost) with serious feature engineering.
3. Deep learning benchmarks: Temporal Fusion Transformer, N-HiTS, LSTM/GRU/TCN. Probabilistic output (quantiles).
4. Hybrid option: grey-box RC thermal model plus a neural network when few buildings are available.

Inputs: consumption history; known-future covariates (weather forecast, calendar); static building data (type, floor area, coordinates, insulation quality, installed equipment such as AC, fridge, heating, PV, battery).

Evaluation: hold-out by building and by time, never random splits. Metrics: MAE, RMSE, CV(RMSE), NMBE, pinball loss for quantiles. Train with historical weather forecasts where available, not only measurements, to avoid train/production mismatch.

## 6. NILM strategy

NILM is part of the base software, built in stages:
1. Simple baselines: event-based detection and signature matching, macro-estimates of major loads (heating/cooling, lighting, base load).
2. Deep learning: seq2point CNN, one model per appliance (seq2seq as comparison), trained on open European datasets prepared by the Dataset Manager.
3. Evaluation on houses never seen in training. Metrics per appliance (MAE, SAE, F1 on on/off state).

Known limits to design around: poor generalisation to unseen houses; small loads are not detectable at 1-8 s sampling; open NILM datasets are residential, so offices, shops and condominiums need fine-tuning or own data.

## 7. Analytics and anomalies

Operational definition: real behaviour significantly different from the behaviour expected by the forecast or the normal profile. v1 is residual-based with statistical thresholds and interpretable rules. Isolation Forest only as a later layer. Deep autoencoders deferred.

## 8. Data sources (prototype phase: free, non-commercial use allowed)

Every source must be tagged with its licence and commercial status. Verify terms and current access before use.

- Weather: Open-Meteo (free for non-commercial use only), AEMET OpenData (Spain, free API key), DWD and ECMWF open data, NASA POWER.
- Solar: PVGIS (EU JRC).
- Reanalysis: ERA5 via Copernicus CDS (free account).
- Market and grid: ESIOS (Red Electrica, Spain, free token), ENTSO-E Transparency Platform (free token).
- Client smart-meter data in Spain: Datadis API (for the customer's own supply, requires customer authorisation).
- NILM datasets: UK-DALE, REFIT, ECO, GREEND, DRED, BLOND.
- Forecasting datasets: Building Data Genome 2, Low Carbon London, Spanish open smart-meter datasets (for example GoiEner, to be verified).
- Review the papers already collected in the project space before choosing datasets.

**Provider abstraction (required from the start)**:
- Common interfaces (WeatherProvider, SolarProvider, MarketProvider) with one adapter per source.
- Provider chosen in config, never hard-coded in models.
- Cache all API responses locally for reproducibility and to limit calls.
- Maintain LICENSES.md with each source, its terms and its commercial status.
- Before going commercial: review licences of data and of models trained on them (non-commercial data can restrict the use of trained weights).
- Paid APIs are out of scope now but will be evaluated for production.

## 9. Architecture and stack

Modular monolith. No microservices, orchestrators or distributed systems in the prototype.

- Python, Pandas (Polars optional later), Pydantic, Typer for the CLI.
- Local storage for the prototype: Parquet files plus DuckDB. PostgreSQL with SQLAlchemy and Alembic only when an API or multi-user use appears.
- ML: scikit-learn, LightGBM/XGBoost, PyTorch with a forecasting library (PyTorch Forecasting or Darts). NILMTK for dataset handling only if it is maintained and compatible; otherwise own loaders.
- Interface: technical dashboard. Streamlit for the prototype is the working assumption; React + TypeScript with FastAPI later (open decision).
- Quality: pytest, Git, Docker later. Strong Windows compatibility.

## 10. Non-functional requirements

Modular architecture, consistent naming, clean repository, reproducible pipelines, run and configuration traceability, easy testing and debugging, readable code, simple deployment, low operating cost, easy evolution from prototype to robust internal tool.

## 11. Roadmap

0. **Foundations**: data schema, units, timezone policy, repository structure, config, logging, test setup.
1. **Dataset Manager v1**: catalogue, selection CLI, weather matching, quality report, export, manifest and PDF.
2. **Forecasting baselines**: baselines and gradient boosting with backtesting on Dataset Manager outputs.
3. **Client ingestion and weather enrichment** for real buildings.
4. **NILM v1**: baselines, then seq2point on one open dataset, evaluated on unseen houses.
5. **Deep forecasting benchmarks** (TFT, N-HiTS) against the tabular model.
6. **Integration**: component-based forecasting.
7. **Analytics**: KPIs and residual-based anomalies.
8. **Dashboard and consulting report.**
9. **Scenario engine.**
10. **Optimisation v1** (peak shaving or tariff cost).
11. **Hardening**: tests, logging, documentation, packaging, repeatable deployment.

## 12. Out of scope for now

Real-time control, BMS/SCADA/IoT live integration, multi-tenant and user management, complex MILP optimisation, microservices, reinforcement learning, digital twin.

## 13. Risks

- Scope too wide (forecasting, NILM, dataset manager, scenarios and optimisation all at once).
- Incomplete building metadata reduces accuracy and interpretability.
- Overengineering too early.
- Vague definition of anomaly produces noise, not insight.
- Premature optimisation on a weak baseline.
- Heterogeneous data: mixed frequencies, gaps, inconsistent units, DST errors.
- Licence restrictions on datasets and trained weights.

## 14. Open decisions (to close during planning)

1. Forecast horizon for the MVP (24 h, 48 h, 7 days, rolling).
2. Exact target: kWh per hour or mean kW per hour.
3. Real data formats available first (CSV, Excel, database).
4. First what-if scenario (tariff, weather, setpoint, occupancy).
5. Anomaly definition for v1 (residual-based, KPI rules, semi-supervised).
6. UI for the MVP (Streamlit, React, simple dashboard).
7. Data schema: tables, naming, units, timezone, missing-value policy.
8. Local storage choice confirmation (DuckDB + Parquet versus PostgreSQL).
9. Which dataset family to implement first in the Dataset Manager, given Spain-first priority.
10. Which building types (home, condominium, office, shop) to target first.

## 15. Instructions for the planner

- Do not write implementation code until the plan is approved.
- Start by restating the problem, then propose architecture, repository structure, data schema, data contracts, milestones and assumptions.
- Ask at most 8 targeted questions on the open decisions above before finalising.
- Keep the Dataset Manager and NILM inside the same codebase, as separate modules with clear interfaces.
- Save the approved plan to docs/PLAN.md.
