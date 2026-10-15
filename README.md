# Project Jardik

Python software for energy consulting: dataset management, forecasting and NILM in one modular
monolith. Prototype phase.

- Scope and requirements: `docs/BRIEF.md`
- Approved plan and milestones: `docs/PLAN.md`
- Data contracts: `docs/DATA_CONTRACTS.md`
- Decision log: `docs/DECISIONS.md`
- Data source licences: `LICENSES.md`

## Setup (Windows, PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
Copy-Item .env.example .env   # then fill in the keys you need
```

## Everyday commands

```powershell
.\.venv\Scripts\python.exe -m pytest              # tests
.\.venv\Scripts\ruff.exe check src tests          # lint
.\.venv\Scripts\ruff.exe format src tests         # format
.\.venv\Scripts\jardik.exe --help                 # CLI
.\.venv\Scripts\jardik.exe info                   # active config and paths
```

## Status

- M0 (foundations) done: config, paths, logging, units, DST-safe time utilities, data contracts,
  provider and model interfaces, CLI skeleton, tests.
- M1 (providers) done: response cache, file weather adapter, Open-Meteo adapter (non-commercial use
  only, see `LICENSES.md`), provider chosen by config, `jardik weather fetch`.
- M2 (Dataset Manager v1) done: DuckDB catalogue, verified BDG2 download, selection filters and quality
  screening, weather-building-time matching, merged training dataset, manifest, quality and distribution
  reports, PDF report, interactive and YAML requests (`jardik dataset ...`).
- M3 (forecasting) done: causal features, baselines, LightGBM with quantiles, split by building and
  time, gate, PDF report and readable review (`jardik forecast ...`). On two years of offices LightGBM
  beats the best baseline by about 32 % MAE on unseen buildings and period.
- Next: M4 client data ingestion and weather enrichment.

## Dataset Manager quick start

```powershell
.\.venv\Scripts\jardik.exe dataset fetch bdg2                 # download and verify (about 195 MB)
.\.venv\Scripts\jardik.exe dataset describe bdg2
.\.venv\Scripts\jardik.exe dataset extract --request configs\requests\example_offices_2016.yaml
.\.venv\Scripts\jardik.exe dataset extract --interactive      # answer questions instead
.\.venv\Scripts\jardik.exe dataset extractions                # what has been extracted
```

## Forecasting quick start

```powershell
.\.venv\Scripts\jardik.exe dataset extract --request configs\requests\offices_2016_2017.yaml
.\.venv\Scripts\jardik.exe forecast run --config configs\forecast\offices_baselines_gbm.yaml
.\.venv\Scripts\jardik.exe forecast runs                       # list runs
.\.venv\Scripts\jardik.exe forecast review                     # readable copy in review\
```

The run config names the extraction (`extraction_id`, see `jardik dataset extractions`).
