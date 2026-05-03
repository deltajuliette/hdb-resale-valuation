# HDB Resale Valuation

A Jupyter notebook and Streamlit dashboard that estimate the fair market value of a Singapore HDB resale flat using public [data.gov.sg](https://data.gov.sg/datasets/d_8b84c4ee58e3cfc0ece0d773c8ca6abc/view) transaction data.

Three independent valuation approaches are computed and blended:

1. **Historical comparables** — top 20 most similar recent transactions, each adjusted to the subject property's exact attributes
2. **Regression with time trend** — OLS on 2024+ data including a `months_since_2024` feature
3. **Regression without time trend** — same model with the time feature removed (sensitivity check against (2))

The blended point estimate is the simple average of the three; the conservative range spans the widest of the three confidence intervals.

## Setup

```bash
pip install -r requirements.txt
```

Tested on Python 3.9+.

## Usage

### Notebook

```bash
jupyter notebook scripts/hdb_resale_trends.ipynb
```

Edit the configuration block near the top of the notebook to set the subject property and comparable universe:

```python
SUBJECT_TOWN       = "QUEENSTOWN"
SUBJECT_FLAT_TYPE  = "4 ROOM"
SUBJECT_FLAT_MODEL = "Premium Apartment"
SUBJECT_FLOOR      = XX # Censored
SUBJECT_AREA_SQM   = XX # Censored
SUBJECT_LEASE_LEFT = XX # Censored
SUBJECT_STREET     = "DAWSON"
COMPARABLE_TOWNS   = ["QUEENSTOWN", "BUKIT MERAH", "CENTRAL AREA"]
```

Inputs are validated against the loaded dataset; typos surface a `ValueError` listing the valid values. The subject town is auto-added to the comparable list if missing.

### Dashboard

```bash
streamlit run scripts/dashboard.py
```

Same pipeline, exposed as a sidebar of widgets — useful for exploring what-if scenarios without editing code. The dataset is fetched once per session and cached for 24 hours.

## Project layout

| Path | Purpose |
|---|---|
| `scripts/hdb_resale_trends.ipynb` | Annotated notebook walking through each valuation approach |
| `scripts/dashboard.py` | Streamlit re-implementation of the same pipeline |
| `requirements.txt` | Python dependencies |

## Data caveats

Per HDB:

- The dataset excludes transactions that may not reflect the full market price (resale between relatives, resale of part shares, etc.).
- Approximate floor area includes any recess area purchased, space-adding additions under upgrading programmes, roof terraces, etc.
- Resale prices are indicative only — actual sale prices depend on many factors not captured here.

## Disclaimer

This is exploratory analysis, not financial advice. The point estimates are model-driven and depend heavily on the chosen comparable universe and the assumption that 2024+ market dynamics extrapolate cleanly to the valuation date.
