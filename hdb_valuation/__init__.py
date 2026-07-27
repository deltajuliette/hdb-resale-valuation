"""
hdb_valuation — shared valuation pipeline for the HDB resale project.

Single source of truth imported by both the Streamlit dashboard
(scripts/dashboard.py) and the notebook (scripts/hdb_resale_trends.ipynb).
"""

from .config import (
    CAT_COLS,
    DATASET_ID,
    DEFAULT_CONFIG,
    DJQC_COLORS,
    DJQC_RC,
    NUM_COLS_NO_TIME,
    NUM_COLS_WITH_TIME,
    SQM_TO_SQFT,
    PipelineConfig,
    Subject,
    months_since_2024,
)
from .data import engineer_features, fetch_raw_csv, suggest_subject_from_address
from .pipeline import (
    blend,
    build_universe,
    fit_factors,
    valuation_comparables,
    valuation_regression,
)
from . import plots

__all__ = [
    # config
    "CAT_COLS", "DATASET_ID", "DEFAULT_CONFIG", "DJQC_COLORS", "DJQC_RC",
    "NUM_COLS_NO_TIME", "NUM_COLS_WITH_TIME", "SQM_TO_SQFT",
    "PipelineConfig", "Subject", "months_since_2024",
    # data
    "engineer_features", "fetch_raw_csv", "suggest_subject_from_address",
    # pipeline
    "blend", "build_universe", "fit_factors",
    "valuation_comparables", "valuation_regression",
    # plots
    "plots",
]
