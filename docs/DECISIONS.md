# Decision log

Short records of decisions that shape the project. Changes to `docs/PLAN.md` are logged here first.

## 2026-10-15: planning decisions (approved with PLAN.md)

| # | Decision | Choice | Why |
|---|---|---|---|
| 1 | MVP forecast horizon | 24 h hourly, day-ahead | Most common use, easiest to validate. Longer horizons are config. |
| 2 | Target | kWh per hour | Matches billing and sums cleanly from 15 min data. |
| 3 | First real data formats | CSV and Excel | One contract with explicit column mapping and units. |
| 4 | First what-if scenario | Weather (temperature offset) | Reuses the trained model directly. |
| 5 | Anomaly v1 | Residual-based | Needs only the forecast model; interpretable. |
| 6 | MVP UI | Typer CLI, Streamlit later | Effort goes to the core first. |
| 8 | Storage | Parquet plus DuckDB | No server, good on Windows. PostgreSQL only if an API appears. |
| 9 | First dataset family | Building Data Genome 2 | Clean, documented, open. Not Spanish; Spanish adapter is M2b. |
| 10 | First building types | Offices, then homes | Offices for forecasting, homes for NILM. |

## 2026-10-15: M0 implementation choices

- Plain pandas contracts (`FrameContract`) instead of pandera: fewer dependencies, errors listed together,
  easy to read while learning. Revisit if contracts grow complicated.
- DuckDB, PyArrow and openpyxl are not installed in M0; they are added in the milestone that first uses them.
- `StrEnum` for controlled vocabularies, so values serialise as plain strings in Parquet and YAML.
- Deviation from the brief's roadmap order: analytics (M6) before deep forecasting (M7), see PLAN.md section 9.

## 2026-10-15: M1 implementation choices

- HTTP uses the standard library (`urllib`) behind an injectable `fetch_json(url, params)` function, so
  tests never touch the network and no HTTP dependency is needed yet.
- The cache stores plain JSON keyed by a hash of (provider, endpoint, parameters). History entries never
  expire, forecasts use a TTL. Writes are atomic (temp file then replace). Corrupt entries are refetched.
- Open-Meteo is used with `timezone=UTC` and `wind_speed_unit=ms`, but conversion still relies on the
  units in the response, so a silent API default change cannot corrupt data.
- Open-Meteo archive data is tagged `reanalysis`, not `observation`, because it is model-derived.
- Providers keep a fixed contract: a frame with `issued_at_utc` always present, so the duplicate-key
  check in the weather contract works for every product type.

## 2026-10-16: M2 implementation choices (deviations from PLAN.md are marked)

- **Deviation: `ts_local` is naive wall-clock time plus `utc_offset_min`**, not a tz-aware timestamp. A
  column can hold one timezone only and a dataset spans several; the offset keeps the repeated autumn
  hour unambiguous. (PLAN.md section 6 said "timestamp with tz".)
- **BDG2 clock = local standard time.** Evidence: the documentation says DST is ignored; the file is one
  continuous 17,544-hour grid, so it has no DST gap or repeat; office load starts at plausible local
  hours (not UTC). Not proven: profile-shift tests at DST changes were noisy and mixed across sites, so
  this rests on the documentation plus grid continuity. `timestamp_clock` is a declared setting.
- **Standard offset = the smaller of the offsets on 1 January and 1 July**, not `utcoffset() - dst()`,
  because tzdata models Irish winter time as negative DST (that formula gives UTC+1 for Dublin).
- **Declared site override for Wolf** (longitude +6.26 -> -6.2603): Europe/Dublin at latitude 53.35 cannot
  be at +6.26. Applied through `configs/datasets/bdg2.yaml`, listed in the manifest and PDF.
- **Weather value rules for BDG2** are declared and counted, never silent: trace precipitation (-1) -> 0,
  oktas -> percent (9 -> 100), implausible values -> NaN. `pressure_msl_hpa` added as its own variable.
- **Deviation: weather in extractions comes from the dataset only** (`weather.source: dataset`). Provider
  sourced weather (Open-Meteo, AEMET) for datasets without weather arrives with the Spanish adapter (M2b).
- **Deviation: granularity must equal the native one** (hourly for BDG2); resampling is not implemented.
- **Deviation: holiday flags are not included** because BDG2 has no country; calendar columns are
  computed from the local wall clock.
- **Matching spatial rule:** the dataset's own weather link wins over distance (`dataset_link`);
  otherwise the nearest point with known coordinates within `max_distance_km`.
- **Quality ratios:** `max_missing_ratio` applies after imputation (what the user cannot use);
  `max_imputed_ratio` limits how much of a series is invented. Imputation fills interior gaps only,
  entirely or not at all.
- **Selection with replacement:** candidates are taken in deterministic order; one rejected for weather
  reasons is replaced by the next, so the requested number is reached whenever enough exist.
- **Atomic publishing:** outputs go to `<id>.partial` and are moved into place only on success; a failed
  `--force` rerun keeps the previous extraction intact.
- **PDF is deterministic** (reportlab `invariant` mode), so the same inputs give identical bytes.
- Real-data acceptance (offices 2016, 30 buildings, 12 sites): 263,520 rows = 30 x 8,784 h; 60 DST days
  audited with every record kept; weather coverage at least 99.7 %.
