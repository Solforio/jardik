# CLAUDE.md

## Project
Project Jardik: Python software for energy consulting (forecasting, NILM, dataset management, analytics, scenarios) for homes, condominiums, offices and shops. Prototype phase, internal technical user, limited budget.

Full scope, requirements, roadmap and open decisions: `docs/BRIEF.md`. Read it before planning or coding. The approved plan lives in `docs/PLAN.md` once it exists.

## Working agreement
- Plan before code. Do not implement until the plan is approved.
- The user writes in Italian: reply in Italian. All code, comments, commit messages and docs are in English.
- The user wants to learn ML and software engineering: explain key design choices briefly, do not hide decisions behind "magic" code.
- Ask at most 8 targeted questions at a time. Prefer stating an assumption over asking about minor details.
- Deliver progressively, one milestone at a time, with complete files.

## Architecture rules
- Modular monolith. No microservices or orchestrators.
- Dataset Manager, NILM and Forecasting are modules of the same codebase with clear interfaces.
- External data goes through provider interfaces (weather, solar, market). Providers are chosen in config, never hard-coded. Cache all API responses.
- Free and open-source tools first. Non-commercial data is allowed in the prototype but every source and licence must be recorded in `LICENSES.md`.
- Model ladder: baselines, then gradient boosting, then deep learning. Each step must beat the previous one on held-out data.
- Evaluation splits are by time and by building, never random.

## Data rules
- Timestamps are timezone-aware. Store UTC and local time. Handle DST explicitly.
- Units are explicit and standardised. Never guess units silently.
- Every dataset extraction produces a manifest and a quality report.
- Never commit raw datasets, API keys or tokens. Use environment variables and a `.env.example`.

## Code rules
- Windows compatibility is mandatory (paths with `pathlib`, no shell-only assumptions).
- No monolithic files. Separate CLI, selection logic, weather matching, reporting and IO.
- Centralise output paths in one place.
- When fixing a file, rewrite it completely instead of patching fragments.
- Before delivering Python files, run `python -m py_compile` on them and a minimal smoke test.
- Write pytest tests for data contracts, weather matching and metric calculations first.
- Logging via the standard `logging` module. Config and validation with Pydantic.

## Reporting conventions
- User-facing reports and exports use Spanish number format: comma for decimals, period for thousands.
- Documents are written in English unless the user asks for another language.

## Commands
Run from the project root in PowerShell. The virtual environment lives in `.venv`.
- Setup: `python -m venv .venv` then `.\.venv\Scripts\python.exe -m pip install -e ".[dev]"`
- Test: `.\.venv\Scripts\python.exe -m pytest` (single file: add its path; single test: add `-k name`)
- Compile check: `.\.venv\Scripts\python.exe -m compileall -q src tests`
- Lint: `.\.venv\Scripts\ruff.exe check src tests`
- Format: `.\.venv\Scripts\ruff.exe format src tests`
- CLI: `.\.venv\Scripts\jardik.exe --help` (or `python -m jardik`)
- Datasets: `jardik dataset fetch bdg2`, `jardik dataset extract --request FILE` (or `--interactive`)
- Weather: `jardik weather fetch --lat .. --lon .. --start .. --end ..` (provider chosen in the config)
