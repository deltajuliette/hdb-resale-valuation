"""Shared pytest fixtures — load the committed sample offline (no network)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from hdb_valuation import Subject, build_universe, engineer_features  # noqa: E402

FIXTURE = REPO_ROOT / "data" / "fixture.csv"
COMP_TOWNS = ["QUEENSTOWN", "BUKIT MERAH", "CENTRAL AREA"]


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    """The raw fixture CSV, exactly as a fresh download would look."""
    return pd.read_csv(FIXTURE)


@pytest.fixture(scope="session")
def df(raw_df) -> pd.DataFrame:
    """Fixture with engineered features."""
    return engineer_features(raw_df)


@pytest.fixture
def subject() -> Subject:
    # Model A is the dominant 4-room model in the fixture, so same-model
    # comparables (build_universe now matches the subject's flat_model) leave a
    # universe comfortably above min_universe.
    return Subject(
        town="QUEENSTOWN", flat_type="4 ROOM", flat_model="Model A",
        floor=10, area_sqm=83.0, lease_left=94.99, street="DAWSON",
    )


@pytest.fixture
def universe(df):
    return build_universe(df, "QUEENSTOWN", "4 ROOM", COMP_TOWNS, "Model A")


@pytest.fixture
def df_comp_recent(universe):
    return universe[1]
