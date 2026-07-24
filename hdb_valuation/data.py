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
import requests

from .config import DATASET_ID, SQM_TO_SQFT, DEFAULT_CONFIG, PipelineConfig


def fetch_raw_csv(dataset_id: str = DATASET_ID, config: PipelineConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Download the raw resale CSV from data.gov.sg's async download API.

    Issues `initiate-download`, then polls `poll-download` until a URL appears.
    Requires network. Returns the untouched CSV as a DataFrame — call
    `engineer_features` to derive the modelling columns.
    """
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
