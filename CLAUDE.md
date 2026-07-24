# CLAUDE.md

Guidance for working in this repo. For setup/usage narrative, see [README.md](README.md);
this file focuses on things that aren't obvious from reading a single file.

## What this is

A Singapore HDB resale-flat valuation project. It estimates a flat's fair market value
from public [data.gov.sg](https://data.gov.sg/datasets/d_8b84c4ee58e3cfc0ece0d773c8ca6abc/view)
transaction data using three independent approaches, then blends them.

## Single source of truth: the `hdb_valuation` package

The valuation math lives **once**, in the repo-root [hdb_valuation/](hdb_valuation/) package.
Both front-ends are thin consumers that import it:

- [scripts/hdb_resale_trends.ipynb](scripts/hdb_resale_trends.ipynb) — the annotated,
  exploratory narrative (EDA plots + `great_tables` displays around the pipeline calls).
- [scripts/dashboard.py](scripts/dashboard.py) — the interactive Streamlit UI.

Change pipeline math (feature engineering, similarity scoring, regression setup, blending)
**in the package only** — both front-ends and the tests pick it up automatically. Do not
re-implement pipeline steps inline in a notebook cell or the dashboard.

Package layout:

- `config.py` — constants + `Subject` / `PipelineConfig` dataclasses (every tunable number).
- `data.py` — `fetch_raw_csv` (network) and `engineer_features` (pure) — I/O split from transforms.
- `pipeline.py` — `build_universe`, `fit_factors`, `valuation_comparables`,
  `valuation_regression`, `blend`.
- `plots.py` — the shared matplotlib helpers + `apply_style`.

## Commands

Use the project virtualenv (`.venv`, Python 3.9):

```bash
pip install -r requirements.txt              # dependencies
pytest tests/                                # unit tests (offline, uses data/fixture.csv)
streamlit run scripts/dashboard.py           # interactive dashboard (needs network)
jupyter notebook scripts/hdb_resale_trends.ipynb  # notebook (needs network)
```

## Pipeline

Both front-ends call the same package functions in this order:

`fetch_raw_csv` → `engineer_features` → `build_universe` → `fit_factors` →
`valuation_comparables` + `valuation_regression` (with time, then without) → `blend`

The three approaches:
1. **Historical comparables** — 20 most similar recent txns, each adjusted to the subject's
   exact attributes via univariate slopes from `fit_factors`.
2. **Regression with time trend** — OLS on 2024+ data including `months_since_2024`.
3. **Regression without time trend** — same model, time feature dropped (sensitivity check).

Blended point estimate = simple average of the three; range = widest of the three intervals.

## Data source

- Fetched **live** by `data.fetch_raw_csv` from the data.gov.sg async API: `initiate-download`
  then `poll-download` in a loop (`PipelineConfig.poll_max_tries`, default 8, 3s sleep).
  Requires network; only `data/fixture.csv` (the test sample) is committed under `data/`.
- `data.engineer_features` derives the engineered columns: `remaining_lease_years` (parsed
  from a `"NN years MM months"` string), `storey_mid`, `price_per_sqft`, `months_since_2024`.
- The dashboard wraps `fetch_raw_csv`+`engineer_features` in an `@st.cache_data` loader (24h TTL).

## Conventions

- Plot styling lives in the `DJQC_COLORS` / `DJQC_RC` constants in `config.py`; call
  `plots.apply_style()` and reuse `DJQC_COLORS` for new charts.
- `INCLUDED_MODELS` filters the comparable universe; `SQM_TO_SQFT = 10.7639` for $/sqft
  (both in `config.py`). Tunable numbers (similarity weights, `n_comps`, `min_universe`,
  poll settings) live in `PipelineConfig`.
- The subject property is a `Subject` dataclass (`.sqft`, `.flat_type_label` are derived).
- Input validation is a **front-end** concern: bad values raise `ValueError` (notebook)
  or surface via `st.error` and `st.stop()` (dashboard). The package assumes valid inputs.

## Testing

`pytest tests/` — runs offline against `data/fixture.csv` (a ~430-row committed sample).
These cover feature engineering, factor fitting, comparable selection/adjustment, regression
outputs, and blend arithmetic. Also verify end-to-end by running the dashboard (needs network)
or executing the notebook top-to-bottom; both must reproduce the same blended estimate.

## Gotchas

- Subject-property specifics are **deliberately censored** in the README (`XX # Censored`).
  Don't commit real personal values.
- `data/fixture.csv` is the only committed data; live snapshots (`data/_snapshot.csv` etc.)
  are gitignored via `/data/*` + `!/data/fixture.csv`.
- The notebook cleared its outputs during the package refactor — re-run it in Jupyter to
  repopulate the `great_tables` displays and plots.
- `.venv/` and `.DS_Store` are gitignored; `.claude/` (local prefs) is too.
