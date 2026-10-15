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

## 2026-10-16: M3 forecasting baselines and gradient boosting

- **Task definition:** day-ahead hourly load forecast issued at LOCAL MIDNIGHT for the coming local day
  (23, 24 or 25 slots on DST days). Target: kWh per hour.
- **No look-ahead, proven by test:** a feature of slot T may use load strictly before the origin
  (lags are masked when `lag_k <= hours_ahead`, statistics end at the origin) plus the weather of slot T
  and of the day. The decisive test overwrites ALL load from the origin onwards and checks that no feature
  of the forecast day changes. Imputed values are never used as features (interpolation looks ahead of a
  gap).
- **Normalisation:** load-like features are divided by `scale_kwh` (mean load of the 168 h before the
  origin) and the model predicts load / scale, so one model serves buildings of very different size.
- **Evaluation:** buildings AND time are held out. Scenarios: `train`, `val` (early stopping only),
  `test_time` (known buildings, unseen future), `test_building` (unseen buildings, known period) and
  `test_both` (unseen buildings AND future: the main result). All models are scored on exactly the same
  rows. `by: site` holds out whole sites (harsher).
- **Gate (each model beats the previous rung):** the challenger must beat the best baseline of the main
  scenario on MAE, RMSE and the MEDIAN building CV(RMSE). The median, not the mean: a building with a tiny
  mean load has an enormous CV(RMSE) (one test building had 802) and dominated the mean.
- **NMBE uses the ASHRAE Guideline 14 sign** (positive = under-forecast), as the user works with energy
  models; documented in the metrics module, the PDF and the workbook.
- **Data-quality finding made by the model (not by the screen):** about 14 of 239 candidate offices
  (mostly site Lamb) had 20-45 % of their readings at ~0.0001 kWh and 3-7 kWh/m2/year against a median of
  105: partial or dead meters that the exact-zero check missed. Added `quality.max_near_zero_ratio`
  (default 0.1; share of readings at or below 1 % of the building's median). This changes the request hash
  of every extraction that does not set it.
- **The plain L2 loss failed on real data.** With it the boosting stopped after 13-18 trees (early stopping
  on validation) and LightGBM lost to the baselines (MAE +24 % worse, better than the baselines in 3 of 25
  buildings). Cause: the load/scale ratio has heavy tails, so the squared loss is dominated by a few
  extreme rows. Robust losses fixed it.
- **How the defaults were chosen (on VALIDATION, never on the test set).** Grid on 100 offices, train 2016,
  validate Jan-Mar 2017 (unseen buildings), test Apr-Dec 2017:

  | target | loss | trees | val MAE | val RMSE | test MAE (shown, not used) |
  |---|---|---|---|---|---|
  | ratio | L2 | 16 | 10.23 | - | 18.52 |
  | ratio | L1 | 130 | 6.523 | 11.079 | 10.77 |
  | ratio | Huber | 287 | 6.536 | 11.000 | 10.79 |
  | residual | L2 | 18 | 28.11 | - | 40.39 |
  | residual | L1 | 46 | 6.94 | - | 13.86 |
  | residual | Huber | 232 | 6.642 | 11.076 | 11.16 |

  Rule declared before looking: lowest validation MAE; ties within 1 % broken by validation RMSE. L1 and
  Huber tie (0.19 %), Huber has the lower RMSE: **default = ratio target, Huber loss**. The residual mode
  (learn a correction of the mean of the last four same-slot values) is kept as an option; it was not
  better here. Best baseline on the same data: 8.46 (val) and 15.83 (test).
- **Result (offices 2016-2017, 100 buildings, run `offices_gbm`, test_both = unseen buildings and period):**
  MAE 10.79 vs 15.83 for the best baseline (naive_lag_24): -31.9 %; RMSE -32.8 %; median CV(RMSE) -31.7 %;
  wins in 24 of 25 test buildings. Holding out whole sites (`offices_gbm_by_site`): MAE -24.4 %, RMSE
  -16.5 %, median CV(RMSE) -37.6 %, wins in 29 of 30 buildings. Gate passed in both. The 10-90 % band covers
  77.5 % / 76.4 % of the values (nominal 80 %: slightly narrow).
- **What drives the model:** the base forecast (mean of the last weeks, 48 %), yesterday's value (20 %) and
  last week's (10 %); weather ranks low because the lags already carry its effect.
- **Limitation that matters:** the weather is OBSERVED and stands in for the forecast, so these errors are
  optimistic. One assignment of buildings to roles per run; variability across assignments not measured;
  hyperparameters other than loss and target mode are defaults.
