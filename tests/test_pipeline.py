"""
Unit tests for the hdb_valuation pipeline — the machine-checked version of the
old hand-enforced "keep the two implementations in sync" rule. All run offline
against data/fixture.csv.
"""

import math

import numpy as np
import pandas as pd
import pytest

from hdb_valuation import (
    Subject,
    SQM_TO_SQFT,
    blend,
    build_universe,
    engineer_features,
    fit_factors,
    months_since_2024,
    valuation_comparables,
    valuation_regression,
)
from hdb_valuation.config import DEFAULT_CONFIG


# ── config / Subject ─────────────────────────────────────────────────────────
def test_subject_derived_properties():
    s = Subject("QUEENSTOWN", "4 ROOM", "Premium Apartment", 10, 83.0, 94.99)
    assert s.sqft == pytest.approx(83.0 * SQM_TO_SQFT)
    assert s.flat_type_label == "4-Room"


def test_months_since_2024():
    from datetime import datetime
    assert months_since_2024(datetime(2024, 1, 15)) == 1
    assert months_since_2024(datetime(2026, 7, 1)) == 31


# ── engineer_features ────────────────────────────────────────────────────────
def test_engineer_features_is_pure(raw_df):
    before = raw_df.copy()
    engineer_features(raw_df)
    pd.testing.assert_frame_equal(raw_df, before)  # input untouched


def test_lease_string_parsing():
    df = engineer_features(pd.DataFrame({
        "month": ["2024-06"],
        "remaining_lease": ["94 years 06 months"],
        "floor_area_sqm": [100.0],
        "storey_range": ["07 TO 09"],
        "resale_price": [1_000_000.0],
    }))
    assert df.loc[0, "remaining_lease_years"] == pytest.approx(94 + 6 / 12)
    assert df.loc[0, "storey_mid"] == pytest.approx(8.0)
    assert df.loc[0, "price_per_sqft"] == pytest.approx(1_000_000 / (100 * SQM_TO_SQFT))
    assert df.loc[0, "months_since_2024"] == 6


# ── build_universe ───────────────────────────────────────────────────────────
def test_build_universe_filters(df):
    df_comp, df_recent = build_universe(
        df, "QUEENSTOWN", "4 ROOM", ["QUEENSTOWN", "BUKIT MERAH", "CENTRAL AREA"], "Model A"
    )
    assert (df_comp["flat_type"] == "4 ROOM").all()
    # Comparables are restricted to the subject's own flat model.
    assert (df_comp["flat_model"] == "Model A").all()
    assert (df_recent["month"].dt.year >= 2024).all()
    assert len(df_recent) <= len(df_comp)
    # Enough recent rows to exceed the trust threshold (guards the fixture).
    assert len(df_recent) >= DEFAULT_CONFIG.min_universe


def test_build_universe_matches_subject_model(df):
    """A model absent from the comp towns yields an empty universe (not other models)."""
    df_comp, _ = build_universe(
        df, "QUEENSTOWN", "4 ROOM", ["QUEENSTOWN", "BUKIT MERAH", "CENTRAL AREA"], "Maisonette"
    )
    # Only same-model rows are ever included; none leak in from other models.
    assert df_comp["flat_model"].eq("Maisonette").all()


def test_build_universe_unknown_town_is_empty(df):
    df_comp, _ = build_universe(df, "NOWHERE", "4 ROOM", ["NOWHERE"], "Model A")
    assert len(df_comp) == 0


# ── suggest_subject_from_address (dashboard "find by address" auto-fill) ──────
def test_suggest_from_address_matches_data(df):
    from hdb_valuation import suggest_subject_from_address
    # Pick a real block/street straight from the fixture so the test is data-driven.
    row = df.iloc[0]
    street, block = row["street_name"], str(row["block"])
    expected_town = df.loc[df["street_name"] == street, "town"].iloc[0]

    sug = suggest_subject_from_address(df, street, block)
    assert sug["town"] == expected_town
    assert sug["flat_model"] in set(df["flat_model"])
    assert sug["area_sqm"] > 0
    assert 0 <= sug["lease_left"] <= 99
    assert sug["n_txns"] >= 1
    assert sug["scope"] in {"block+type", "block", "street+type", "street"}


def test_suggest_from_address_flat_type_and_fallback(df):
    from hdb_valuation import suggest_subject_from_address
    street = df.iloc[0]["street_name"]
    ft = df.loc[df["street_name"] == street, "flat_type"].iloc[0]
    # With a matching flat_type but no block, we fall back to a street-level scope.
    sug = suggest_subject_from_address(df, street, block="", flat_type=ft)
    assert sug["town"] == df.loc[df["street_name"] == street, "town"].iloc[0]
    assert sug["scope"] in {"street+type", "street"}


def test_suggest_from_address_unknown_is_empty(df):
    from hdb_valuation import suggest_subject_from_address
    assert suggest_subject_from_address(df, "NO SUCH STREET", "999") == {}
    assert suggest_subject_from_address(df, "", "") == {}


# ── fit_factors ──────────────────────────────────────────────────────────────
def test_fit_factors_keys_and_signs(df_comp_recent):
    factors = fit_factors(df_comp_recent)
    assert set(factors) == {"floor-level", "lease", "area", "time"}
    assert all(math.isfinite(v) for v in factors.values())
    # Higher floors and longer leases command higher prices.
    assert factors["floor-level"] > 0
    assert factors["lease"] > 0


# ── valuation_comparables ────────────────────────────────────────────────────
def test_comparables_selection_and_adjustment(df_comp_recent, subject):
    factors = fit_factors(df_comp_recent)
    res = valuation_comparables(df_comp_recent, subject, factors, valuation_month=31)
    top = res["top_comps"]
    assert len(top) == DEFAULT_CONFIG.n_comps
    # adjusted_price = resale_price + sum of the four adjustments.
    recomputed = top["resale_price"] + top[
        ["adj_floor-level", "adj_lease", "adj_area", "adj_time"]
    ].sum(axis=1)
    np.testing.assert_allclose(top["adjusted_price"], recomputed)
    # estimate is the median; IQR brackets it.
    assert res["estimate"] == pytest.approx(top["adjusted_price"].median())
    assert res["q25"] <= res["estimate"] <= res["q75"]


def test_comparables_are_the_nearest(df_comp_recent, subject):
    """The chosen comps must be the lowest-similarity-score rows."""
    factors = fit_factors(df_comp_recent)
    res = valuation_comparables(df_comp_recent, subject, factors, valuation_month=31)
    cand = df_comp_recent.dropna(subset=["remaining_lease_years"]).copy()
    w = DEFAULT_CONFIG.similarity_weights
    cand["sim"] = (
        (cand["storey_mid"] - subject.floor).abs() * w["floor"]
        + (cand["remaining_lease_years"] - subject.lease_left).abs() * w["lease"]
        + (cand["floor_area_sqm"] - subject.area_sqm).abs() * w["area"]
    )
    expected_max = cand["sim"].nsmallest(DEFAULT_CONFIG.n_comps).max()
    chosen_max = res["top_comps"]["similarity"].max()
    assert chosen_max == pytest.approx(expected_max)


# ── valuation_regression ─────────────────────────────────────────────────────
@pytest.mark.parametrize("with_time", [True, False])
def test_regression_outputs(df_comp_recent, subject, with_time):
    res = valuation_regression(df_comp_recent, subject, valuation_month=31, with_time=with_time)
    assert res["estimate"] > 0
    assert 0.0 <= res["r2"] <= 1.0
    assert res["residual_std"] > 0
    assert res["n"] == len(df_comp_recent.dropna(subset=["remaining_lease_years"]))
    # Diagnostic artefacts are exposed for the notebook.
    assert res["model"] is not None
    assert len(res["y_pred"]) == res["n"]


# ── blend ────────────────────────────────────────────────────────────────────
def test_blend_arithmetic(subject):
    comp = {"estimate": 1_000_000.0, "q25": 950_000.0, "q75": 1_050_000.0}
    reg_a = {"estimate": 1_100_000.0, "residual_std": 80_000.0}
    reg_b = {"estimate": 900_000.0, "residual_std": 60_000.0}
    res = blend(comp, reg_a, reg_b, subject)
    assert res["blended"] == pytest.approx((1_000_000 + 1_100_000 + 900_000) / 3)
    # widest interval: min of lows, max of highs
    assert res["low"] == pytest.approx(min(950_000, 1_100_000 - 80_000, 900_000 - 60_000))
    assert res["high"] == pytest.approx(max(1_050_000, 1_100_000 + 80_000, 900_000 + 60_000))
    assert res["psf"] == pytest.approx(res["blended"] / subject.sqft)


# ── end-to-end smoke on the fixture ──────────────────────────────────────────
def test_full_pipeline_runs(df_comp_recent, subject):
    factors = fit_factors(df_comp_recent)
    comp = valuation_comparables(df_comp_recent, subject, factors, 31)
    reg_a = valuation_regression(df_comp_recent, subject, 31, with_time=True)
    reg_b = valuation_regression(df_comp_recent, subject, 31, with_time=False)
    res = blend(comp, reg_a, reg_b, subject)
    assert res["low"] <= res["blended"] <= res["high"]
    assert res["blended"] > 0
