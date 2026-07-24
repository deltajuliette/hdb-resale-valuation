"""
Constants and typed configuration for the HDB resale valuation pipeline.

This is the single source of truth for every tunable number and column list.
Both the Streamlit dashboard and the notebook import from here, so the two
front-ends can never drift on the underlying math.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

# ── Data source ──────────────────────────────────────────────────────────────
DATASET_ID = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"

# ── Unit conversion ──────────────────────────────────────────────────────────
SQM_TO_SQFT = 10.7639

# ── Comparable universe ──────────────────────────────────────────────────────
# Flat models eligible to enter the comparable pool.
INCLUDED_MODELS = ["Model A", "Premium Apartment"]

# ── Regression feature sets ──────────────────────────────────────────────────
NUM_COLS_WITH_TIME = ["floor_area_sqm", "storey_mid", "remaining_lease_years", "months_since_2024"]
NUM_COLS_NO_TIME   = ["floor_area_sqm", "storey_mid", "remaining_lease_years"]
CAT_COLS           = ["town", "flat_model"]

# ── Plot styling ─────────────────────────────────────────────────────────────
DJQC_COLORS = ["grey", "#785EF0", "#DC267F", "#FE6100", "#FFB000", "#44AA99", "#117733"]
DJQC_RC = {
    "axes.facecolor": "#EBF8FF",
    "axes.edgecolor": "#333333",
    "axes.grid": False,
    "grid.color": "#FFFFFF",
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
}


def months_since_2024(when: datetime) -> int:
    """Month index used as the time feature: Jan 2024 = 1, Feb 2024 = 2, …"""
    return (when.year - 2024) * 12 + when.month


@dataclass
class Subject:
    """The unit being valued."""

    town: str
    flat_type: str
    flat_model: str
    floor: int
    area_sqm: float
    lease_left: float
    street: str = ""

    @property
    def sqft(self) -> float:
        return self.area_sqm * SQM_TO_SQFT

    @property
    def flat_type_label(self) -> str:
        """Display-friendly label, e.g. "4 ROOM" -> "4-Room"."""
        return self.flat_type.title().replace(" ", "-")


@dataclass
class PipelineConfig:
    """Tunable knobs for the valuation pipeline (formerly magic numbers)."""

    # valuation_comparables: weights on |Δstorey|, |Δlease|, |Δarea| in the
    # similarity score, and how many nearest comps to keep.
    similarity_weights: dict = field(
        default_factory=lambda: {"floor": 2.0, "lease": 1.0, "area": 0.5}
    )
    n_comps: int = 20

    # Minimum recent (2024+) transactions required before a valuation is trusted.
    min_universe: int = 30

    # fetch_raw_csv: how many times to poll data.gov.sg before giving up.
    poll_max_tries: int = 8
    poll_sleep_seconds: float = 3.0


DEFAULT_CONFIG = PipelineConfig()
