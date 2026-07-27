"""
The valuation pipeline: the single source of truth for the math.

Stages (mirrors the notebook narrative):
    build_universe -> fit_factors ->
    valuation_comparables + valuation_regression (with / without time) -> blend

Every function here is pure (no Streamlit, no I/O), so the dashboard, the
notebook, and the test suite all exercise identical code.
"""

from __future__ import annotations

import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    CAT_COLS,
    DEFAULT_CONFIG,
    NUM_COLS_NO_TIME,
    NUM_COLS_WITH_TIME,
    PipelineConfig,
    Subject,
)


def build_universe(df, subject_town, flat_type, comp_towns, flat_model):
    """Split the comparable pool into the full history and the 2024+ subset.

    Comparables are restricted to the subject's own `flat_model`: the comparables
    approach adjusts for floor/lease/area/time but *not* model, so pooling models
    (e.g. valuing an Executive "Apartment" against pricier "Premium Apartment"
    sales) would inject unadjusted, model-driven bias. Same-model keeps it
    apples-to-apples; if that leaves too few comps, the front-end widens the
    comparable towns rather than mixing models.
    """
    df_comp = df.loc[
        (df["flat_type"] == flat_type)
        & (df["town"].isin(comp_towns))
        & (df["flat_model"] == flat_model)
    ].copy()
    df_comp_recent = df_comp.loc[df_comp["month"].dt.year >= 2024].copy()
    return df_comp, df_comp_recent


def fit_factors(df_comp_recent):
    """Univariate OLS slopes ($ per unit) for each adjustable attribute."""
    factors = {}
    for label, x_col in [
        ("floor-level", "storey_mid"),
        ("lease", "remaining_lease_years"),
        ("area", "floor_area_sqm"),
    ]:
        subset = df_comp_recent.dropna(subset=[x_col])
        slope, *_ = stats.linregress(subset[x_col], subset["resale_price"])
        factors[label] = slope
    slope_t, *_ = stats.linregress(
        df_comp_recent["months_since_2024"], df_comp_recent["resale_price"]
    )
    factors["time"] = slope_t
    return factors


def valuation_comparables(df_comp_recent, subject: Subject, factors, valuation_month,
                          config: PipelineConfig = DEFAULT_CONFIG):
    """Approach 1: the N most similar recent comps, each adjusted to the subject."""
    w = config.similarity_weights
    df_candidates = df_comp_recent.dropna(subset=["remaining_lease_years"]).copy()
    df_candidates["similarity"] = (
        abs(df_candidates["storey_mid"] - subject.floor) * w["floor"]
        + abs(df_candidates["remaining_lease_years"] - subject.lease_left) * w["lease"]
        + abs(df_candidates["floor_area_sqm"] - subject.area_sqm) * w["area"]
    )
    top = df_candidates.nsmallest(config.n_comps, "similarity").copy()
    for attr, col, key in [
        ("floor",      "storey_mid",            "floor-level"),
        ("lease_left", "remaining_lease_years", "lease"),
        ("area_sqm",   "floor_area_sqm",        "area"),
    ]:
        top[f"adj_{key}"] = (getattr(subject, attr) - top[col]) * factors[key]
    top["adj_time"] = (valuation_month - top["months_since_2024"]) * factors["time"]
    top["total_adjustment"] = top[["adj_floor-level", "adj_lease", "adj_area", "adj_time"]].sum(axis=1)
    top["adjusted_price"] = top["resale_price"] + top["total_adjustment"]
    return {
        "top_comps": top,
        "estimate":  float(top["adjusted_price"].median()),
        "q25":       float(top["adjusted_price"].quantile(0.25)),
        "q75":       float(top["adjusted_price"].quantile(0.75)),
    }


def valuation_regression(df_comp_recent, subject: Subject, valuation_month, with_time):
    """Approach 2: OLS on the recent universe, with or without the time feature."""
    df_reg = df_comp_recent.dropna(subset=["remaining_lease_years"]).copy()
    num_cols = NUM_COLS_WITH_TIME if with_time else NUM_COLS_NO_TIME

    pre = ColumnTransformer(transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), CAT_COLS),
    ])
    model = Pipeline([("preprocessor", pre), ("regressor", LinearRegression())])
    model.fit(df_reg[num_cols + CAT_COLS], df_reg["resale_price"])
    y_pred = model.predict(df_reg[num_cols + CAT_COLS])
    residuals = df_reg["resale_price"] - y_pred

    subject_row = {
        "floor_area_sqm":        [subject.area_sqm],
        "storey_mid":            [float(subject.floor)],
        "remaining_lease_years": [subject.lease_left],
        "town":                  [subject.town],
        "flat_model":            [subject.flat_model],
    }
    if with_time:
        subject_row["months_since_2024"] = [valuation_month]

    return {
        "estimate":     float(model.predict(pd.DataFrame(subject_row))[0]),
        "residual_std": float(residuals.std()),
        "r2":           float(r2_score(df_reg["resale_price"], y_pred)),
        "mae":          float(mean_absolute_error(df_reg["resale_price"], y_pred)),
        "n":            len(df_reg),
        # Fitted artefacts for diagnostics (coefficient tables, residual plots);
        # the dashboard ignores these, the notebook inspects them.
        "model":        model,
        "y_pred":       y_pred,
        "df_reg":       df_reg,
    }


def blend(comp, reg_a, reg_b, subject: Subject):
    """Blend the three approaches: mean point estimate, widest interval."""
    point = (comp["estimate"] + reg_a["estimate"] + reg_b["estimate"]) / 3
    low   = min(comp["q25"], reg_a["estimate"] - reg_a["residual_std"], reg_b["estimate"] - reg_b["residual_std"])
    high  = max(comp["q75"], reg_a["estimate"] + reg_a["residual_std"], reg_b["estimate"] + reg_b["residual_std"])
    return {
        "blended": point,
        "low":     low,
        "high":    high,
        "psf":     point / subject.sqft,
    }
