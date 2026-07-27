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

The **hosted GitHub Pages build is not a third front-end** — [web/index.html](web/index.html)
mounts the *same* `scripts/dashboard.py` + `hdb_valuation` package via stlite (Streamlit compiled
to WebAssembly). There is no web-specific app code to keep in sync; see "Hosting" below.

Package layout:

- `config.py` — constants + `Subject` / `PipelineConfig` dataclasses (every tunable number).
- `data.py` — `fetch_raw_csv` (network) and `engineer_features` (pure) — I/O split from transforms.
  Also `suggest_subject_from_address` (pure) — maps a block/street to typical town/model/area/lease
  for the dashboard's address auto-fill. Note: `import requests` is **lazy** (inside `fetch_raw_csv`)
  so the package imports in Pyodide/stlite, where `requests` is unavailable.
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
- The dashboard's `@st.cache_data` loader (24h TTL) **prefers a bundled `data/snapshot.csv`** when
  present, else calls `fetch_raw_csv`. Locally the snapshot is absent (gitignored) → live fetch; the
  hosted build ships one → offline load. `engineer_features` runs either way, so both paths match.

## Hosting (static GitHub Pages build)

- [web/index.html](web/index.html) is a stlite (`@stlite/browser`, pinned CDN version) loader that
  mounts `scripts/dashboard.py` + the `hdb_valuation/*.py` modules + `data/snapshot.csv` in the
  browser via Pyodide/WebAssembly. Files are mirrored under the repo layout so `dashboard.py`'s
  `sys.path` insert and the `data/snapshot.csv` path resolve unchanged.
- The stlite `requirements` list excludes `streamlit` (stlite provides it) and `requests` (unused on
  web after the lazy import) — only compute deps (pandas/numpy/scipy/scikit-learn/matplotlib/seaborn).
- [.github/workflows/deploy.yml](.github/workflows/deploy.yml) fetches the snapshot server-side
  (where `requests` works), assembles `site/`, and deploys to Pages on push to `main`, weekly (cron,
  for fresh data), or manual dispatch. **One-time:** enable Settings → Pages → Source = "GitHub Actions".
- Data.gov.sg's async API is **not** browser-reachable (CORS); this is why the snapshot is built in
  CI rather than fetched client-side. Don't try to re-enable live fetch in the web build.

## Conventions

- Plot styling lives in the `DJQC_COLORS` / `DJQC_RC` constants in `config.py`; call
  `plots.apply_style()` and reuse `DJQC_COLORS` for new charts.
- `build_universe` matches comparables to the **subject's own `flat_model`** (apples-to-apples:
  the approach-1 adjustments correct for floor/lease/area/time but **not** model, so pooling
  models would inject unadjusted bias). A rare model may make the universe too small — widen the
  comparable towns, don't mix models. `SQM_TO_SQFT = 10.7639` for $/sqft (`config.py`). Tunable
  numbers (similarity weights, `n_comps`, `min_universe`, poll settings) live in `PipelineConfig`.
- The subject property is a `Subject` dataclass (`.sqft`, `.flat_type_label` are derived).
- Input validation is a **front-end** concern; the package assumes valid inputs. The notebook
  raises `ValueError` on bad values. The dashboard mostly **prevents** invalid input at the source
  (town/type/model/comparables are dropdowns sourced from the data), and still uses `st.error` +
  `st.stop()` for the runtime guard that the comparable universe is large enough.

## Testing

`pytest tests/` — runs offline against `data/fixture.csv` (a ~430-row committed sample).
These cover feature engineering, factor fitting, comparable selection/adjustment, regression
outputs, and blend arithmetic. Also verify end-to-end by running the dashboard (needs network)
or executing the notebook top-to-bottom; both must reproduce the same blended estimate.

## Gotchas

- Subject-property specifics are **deliberately censored** in the README (`XX # Censored`) and the
  notebook. Don't commit real personal values. The dashboard opens on a **neutral, data-derived
  default** subject (`_default_subject`) precisely so the public GitHub Pages build exposes nothing
  personal — keep it that way when editing defaults.
- `data/fixture.csv` is the only committed data; live snapshots (`data/_snapshot.csv` etc.)
  are gitignored via `/data/*` + `!/data/fixture.csv`.
- The notebook cleared its outputs during the package refactor — re-run it in Jupyter to
  repopulate the `great_tables` displays and plots.
- `.venv/` and `.DS_Store` are gitignored; `.claude/` (local prefs) is too.
