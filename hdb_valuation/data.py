"""
Data acquisition and feature engineering.

`fetch_raw_csv` is the only function that touches the network; `engineer_features`
is pure and deterministic, so tests (and the notebook) can run the whole pipeline
offline against a saved CSV.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .config import DATASET_ID, SQM_TO_SQFT, DEFAULT_CONFIG, PipelineConfig


def fetch_raw_csv(dataset_id: str = DATASET_ID, config: PipelineConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Download the raw resale CSV from data.gov.sg's async download API.

    Issues `initiate-download`, then polls `poll-download` until a URL appears.
    Requires network. Returns the untouched CSV as a DataFrame — call
    `engineer_features` to derive the modelling columns.

    `requests` is imported lazily so the rest of the package (and the web build,
    which reads a bundled snapshot instead of fetching) can import cleanly in
    environments where `requests` is unavailable — notably Pyodide/stlite.
    """
    import requests

    s = requests.Session()
    s.headers.update({"referer": "https://colab.research.google.com"})
    s.get(
        f"https://api-open.data.gov.sg/v1/public/api/datasets/{dataset_id}/initiate-download",
        headers={"Content-Type": "application/json"},
        json={},
    )
    for _ in range(config.poll_max_tries):
        resp = s.get(
            f"https://api-open.data.gov.sg/v1/public/api/datasets/{dataset_id}/poll-download",
            headers={"Content-Type": "application/json"},
            json={},
        )
        url = resp.json()["data"].get("url")
        if url:
            return pd.read_csv(url)
        time.sleep(config.poll_sleep_seconds)
    raise RuntimeError("Download timed out — try again or check data.gov.sg status.")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the modelling columns from the raw resale data.

    Adds: `month` (datetime), `remaining_lease_years`, `price_per_sqft`,
    `storey_mid`, `months_since_2024`. Pure — returns a new frame, does not
    mutate the input.
    """
    df = df.copy()
    df["month"] = pd.to_datetime(df["month"])
    df["remaining_lease_years"] = (
        df["remaining_lease"].str.extract(r"(\d+)").astype(float)
        + df["remaining_lease"].str.extract(r"\d+\s+years\s+(\d+)").astype(float) / 12
    )
    df["price_per_sqft"] = df["resale_price"] / (df["floor_area_sqm"] * SQM_TO_SQFT)
    df["storey_mid"] = df["storey_range"].apply(
        lambda x: np.mean([int(s) for s in x.split(" TO ")])
    )
    df["months_since_2024"] = (df["month"].dt.year - 2024) * 12 + df["month"].dt.month
    return df


def suggest_subject_from_address(
    df: pd.DataFrame, street_name: str, block: str = "", flat_type: str = ""
) -> dict:
    """Suggest typical subject attributes for a given HDB address, from past txns.

    Pure and offline — powers the dashboard's "find by address" auto-fill so the
    user picks a real block/street instead of typing town/model/area/lease by hand.

    Narrows `df` to the given `street_name` (and `block`, and `flat_type` when
    provided), falling back to broader scopes if that slice is too thin, then
    reports the deterministic `town` plus the modal `flat_model` and median
    `floor_area_sqm` / `remaining_lease_years` / `storey_mid`. `remaining_lease`
    is reported as of *today* — each txn's lease minus the years elapsed since it
    transacted — since a decade-old lease figure would overstate what's left.

    Returns a dict with keys ``town, flat_model, area_sqm, lease_left, storey_mid,
    n_txns, scope`` (``scope`` names which fallback matched), or ``{}`` if nothing
    matches. Assumes `df` has engineered features (`remaining_lease_years`,
    `storey_mid`, `month`).
    """
    if not street_name:
        return {}

    street = df[df["street_name"] == street_name]
    if street.empty:
        return {}

    # Progressively broaden until we have a usable slice: block+type -> block ->
    # street+type -> street. `town` is constant within a street, so it is always
    # taken from the street-level slice.
    candidates = []
    if block:
        blk = street[street["block"].astype(str) == str(block)]
        if flat_type:
            candidates.append(("block+type", blk[blk["flat_type"] == flat_type]))
        candidates.append(("block", blk))
    if flat_type:
        candidates.append(("street+type", street[street["flat_type"] == flat_type]))
    candidates.append(("street", street))

    scope, sample = next(((s, c) for s, c in candidates if not c.empty), ("street", street))

    now = pd.Timestamp.now()
    years_elapsed = (now - sample["month"]).dt.days / 365.25
    lease_today = (sample["remaining_lease_years"] - years_elapsed).clip(lower=0)

    return {
        "town": street["town"].iloc[0],
        "flat_model": sample["flat_model"].mode().iloc[0],
        "area_sqm": round(float(sample["floor_area_sqm"].median()), 1),
        "lease_left": round(float(lease_today.median()), 2),
        "storey_mid": int(round(float(sample["storey_mid"].median()))),
        "n_txns": int(len(sample)),
        "scope": scope,
    }
