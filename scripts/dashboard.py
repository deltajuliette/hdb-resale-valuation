"""
HDB Resale Valuation Dashboard
==============================
Interactive front-end for the same valuation pipeline used in
hdb_resale_trends.ipynb. Configure the subject property and comparable
universe in the sidebar; the three valuation approaches and supporting plots
re-render on every change.

Run with:
    streamlit run scripts/dashboard.py
"""

from __future__ import annotations

import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import streamlit as st
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ── Constants ───────────────────────────────────────────────────────────────
DATASET_ID = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"
SQM_TO_SQFT = 10.7639
INCLUDED_MODELS = ["Model A", "Premium Apartment"]

NUM_COLS_WITH_TIME = ["floor_area_sqm", "storey_mid", "remaining_lease_years", "months_since_2024"]
NUM_COLS_NO_TIME   = ["floor_area_sqm", "storey_mid", "remaining_lease_years"]
CAT_COLS           = ["town", "flat_model"]

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
sns.set_style("darkgrid", rc=DJQC_RC)
sns.set_palette(sns.color_palette(DJQC_COLORS))


def _months_since_2024(when: datetime) -> int:
    return (when.year - 2024) * 12 + when.month


# ── Data loading ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Downloading HDB resale dataset…", ttl=24 * 3600)
def load_data() -> pd.DataFrame:
    """Fetch the HDB resale dataset from data.gov.sg and engineer features."""
    s = requests.Session()
    s.headers.update({"referer": "https://colab.research.google.com"})
    s.get(
        f"https://api-open.data.gov.sg/v1/public/api/datasets/{DATASET_ID}/initiate-download",
        headers={"Content-Type": "application/json"},
        json={},
    )
    df = None
    for _ in range(8):
        resp = s.get(
            f"https://api-open.data.gov.sg/v1/public/api/datasets/{DATASET_ID}/poll-download",
            headers={"Content-Type": "application/json"},
            json={},
        )
        url = resp.json()["data"].get("url")
        if url:
            df = pd.read_csv(url)
            break
        time.sleep(3)
    if df is None:
        raise RuntimeError("Download timed out — try again or check data.gov.sg status.")

    df["month"] = pd.to_datetime(df["month"])
    df["remaining_lease_years"] = (
        df["remaining_lease"].str.extract(r"(\d+)").astype(float)
        + df["remaining_lease"].str.extract(r"\d+\s+years\s+(\d+)").astype(float) / 12
    )
    df["price_per_sqft"] = df["resale_price"] / (df["floor_area_sqm"] * SQM_TO_SQFT)
    df["storey_mid"] = df["storey_range"].apply(
        lambda x: np.mean([int(s) for s in x.split(" TO ")])
    )
    return df


# ── Pipeline ────────────────────────────────────────────────────────────────
def build_universe(df, subject_town, flat_type, comp_towns, included_models):
    df_comp = df.loc[
        (df["flat_type"] == flat_type)
        & (df["town"].isin(comp_towns))
        & (df["flat_model"].isin(included_models))
    ].copy()
    df_comp_recent = df_comp.loc[df_comp["month"].dt.year >= 2024].copy()
    df_comp_recent["months_since_2024"] = (
        (df_comp_recent["month"].dt.year - 2024) * 12 + df_comp_recent["month"].dt.month
    )
    return df_comp, df_comp_recent


def fit_factors(df_comp_recent):
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


def valuation_comparables(df_comp_recent, subject, factors, valuation_month):
    df_candidates = df_comp_recent.dropna(subset=["remaining_lease_years"]).copy()
    df_candidates["similarity"] = (
        abs(df_candidates["storey_mid"] - subject["floor"]) * 2
        + abs(df_candidates["remaining_lease_years"] - subject["lease_left"]) * 1
        + abs(df_candidates["floor_area_sqm"] - subject["area_sqm"]) * 0.5
    )
    top = df_candidates.nsmallest(20, "similarity").copy()
    for attr, col, key in [
        ("floor",      "storey_mid",            "floor-level"),
        ("lease_left", "remaining_lease_years", "lease"),
        ("area_sqm",   "floor_area_sqm",        "area"),
    ]:
        top[f"adj_{key}"] = (subject[attr] - top[col]) * factors[key]
    top["adj_time"] = (valuation_month - top["months_since_2024"]) * factors["time"]
    top["total_adjustment"] = top[["adj_floor-level", "adj_lease", "adj_area", "adj_time"]].sum(axis=1)
    top["adjusted_price"] = top["resale_price"] + top["total_adjustment"]
    return {
        "top_comps": top,
        "estimate":  float(top["adjusted_price"].median()),
        "q25":       float(top["adjusted_price"].quantile(0.25)),
        "q75":       float(top["adjusted_price"].quantile(0.75)),
    }


def valuation_regression(df_comp_recent, subject, valuation_month, with_time):
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
        "floor_area_sqm":        [subject["area_sqm"]],
        "storey_mid":            [float(subject["floor"])],
        "remaining_lease_years": [subject["lease_left"]],
        "town":                  [subject["town"]],
        "flat_model":            [subject["flat_model"]],
    }
    if with_time:
        subject_row["months_since_2024"] = [valuation_month]

    return {
        "estimate":     float(model.predict(pd.DataFrame(subject_row))[0]),
        "residual_std": float(residuals.std()),
        "r2":           float(r2_score(df_reg["resale_price"], y_pred)),
        "mae":          float(mean_absolute_error(df_reg["resale_price"], y_pred)),
        "n":            len(df_reg),
    }


def blend(comp, reg_a, reg_b, subject):
    point = (comp["estimate"] + reg_a["estimate"] + reg_b["estimate"]) / 3
    low   = min(comp["q25"], reg_a["estimate"] - reg_a["residual_std"], reg_b["estimate"] - reg_b["residual_std"])
    high  = max(comp["q75"], reg_a["estimate"] + reg_a["residual_std"], reg_b["estimate"] + reg_b["residual_std"])
    return {
        "blended": point,
        "low":     low,
        "high":    high,
        "psf":     point / (subject["area_sqm"] * SQM_TO_SQFT),
    }


# ── Plot helpers ────────────────────────────────────────────────────────────
def plot_towns_boxen(df_all_recent, subject_town, flat_type_label):
    fig, ax = plt.subplots(figsize=(8, 4))
    town_colors = {
        t: DJQC_COLORS[1] if t == subject_town else "grey"
        for t in df_all_recent["town"].unique()
    }
    sns.boxenplot(
        data=df_all_recent, x="town", y="resale_price",
        hue="town", palette=town_colors, dodge=False, ax=ax,
    )
    ax.set_title(f"Flat Prices across Towns from 2024 onwards ({flat_type_label} Flats)")
    ax.set_xlabel("Town")
    ax.set_ylabel("Resale Price")
    plt.setp(ax.get_xticklabels(), rotation=90)
    fig.tight_layout()
    return fig


def plot_subject_time_series(df_subject_full, subject_town, flat_type_label):
    ts = df_subject_full.groupby("month")["resale_price"].agg(
        ["median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
    )
    ts.columns = ["median", "Q1", "Q3"]
    ts["rolling_avg"] = ts["median"].rolling(window=6).mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ts.index, ts["median"],      marker="o", label="Median",          color=DJQC_COLORS[1], linewidth=2)
    ax.plot(ts.index, ts["Q1"],          marker="o", label="Q1",              color="grey", linestyle=":", linewidth=1.2)
    ax.plot(ts.index, ts["Q3"],          marker="o", label="Q3",              color="grey", linestyle=":", linewidth=1.2)
    ax.plot(ts.index, ts["rolling_avg"],             label="6-Mo Rolling Avg", color="red",  linestyle="--", linewidth=2)
    ax.set_title(f"Resale Price of {subject_town.title()} ({flat_type_label} Flats)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Resale Price")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_adjusted_distribution(top_comps):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(top_comps["adjusted_price"], bins=12, color=DJQC_COLORS[1], edgecolor="white", alpha=0.85)
    median = top_comps["adjusted_price"].median()
    ax.axvline(median, color=DJQC_COLORS[2], linestyle="--", linewidth=2,
               label=f"Median: ${median:,.0f}")
    ax.set_title("Adjusted Price Distribution (Top 20 Comps)")
    ax.set_xlabel("Adjusted Price ($)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_subject_vs_market(df_comp_recent, subject, subject_town, flat_type_label,
                           blended, low, high):
    fig, ax = plt.subplots(figsize=(8, 4))
    sub = df_comp_recent.loc[df_comp_recent["town"] == subject_town]
    ax.scatter(sub["floor_area_sqm"], sub["resale_price"],
               s=20, alpha=0.4, color="grey",
               label=f"{subject_town.title()} {flat_type_label} (2024+)")
    if subject["street"]:
        street_recent = sub.loc[sub["street_name"].str.contains(subject["street"], na=False)]
        if len(street_recent):
            ax.scatter(street_recent["floor_area_sqm"], street_recent["resale_price"],
                       s=30, alpha=0.7, color=DJQC_COLORS[1],
                       label=f"{subject['street'].title()} Road")
    ax.scatter([subject["area_sqm"]], [blended], s=200, color=DJQC_COLORS[2],
               marker="*", zorder=5, label=f"Subject (est. ${blended:,.0f})")
    ax.errorbar(subject["area_sqm"], blended,
                yerr=[[blended - low], [high - blended]],
                color=DJQC_COLORS[2], capsize=6, linewidth=2, zorder=4)
    ax.set_title(f"Subject vs {subject_town.title()} Market ({flat_type_label}, 2024+)")
    ax.set_xlabel("Floor Area (sqm)")
    ax.set_ylabel("Resale Price ($)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ── Streamlit UI ────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="HDB Resale Valuation", layout="wide")
    st.title("HDB Resale Valuation")

    df = load_data()
    available_towns      = sorted(df["town"].unique())
    available_models     = sorted(df["flat_model"].unique())
    available_flat_types = sorted(df["flat_type"].unique())

    with st.sidebar:
        st.header("Subject Property")
        town_in = st.text_input(
            "Town",
            value="QUEENSTOWN",
            help=f"Available: {', '.join(available_towns)}",
        )
        flat_type = st.selectbox(
            "Flat type",
            available_flat_types,
            index=available_flat_types.index("4 ROOM") if "4 ROOM" in available_flat_types else 0,
            format_func=lambda s: s.title().replace(" ", "-"),
        )
        flat_model_in = st.text_input(
            "Flat model",
            value="Premium Apartment",
            help=f"Available: {', '.join(available_models)}",
        )
        floor      = st.number_input("Floor (storey)",         min_value=1,    max_value=60,  value=10,    step=1)
        area_sqm   = st.number_input("Floor area (sqm)",       min_value=10.0, max_value=300.0, value=83.0, step=1.0)
        lease_left = st.number_input("Remaining lease (years)", min_value=1.0, max_value=99.0, value=94.99, step=0.01, format="%.2f")
        street_in  = st.text_input(
            "Street (substring)",
            value="DAWSON",
            help="Substring match against street_name; used only for the street-level highlight in plots.",
        )

        st.header("Comparable universe")
        comp_in = st.text_input(
            "Comparable towns (comma-separated)",
            value="QUEENSTOWN, BUKIT MERAH, CENTRAL AREA",
            help="Subject town is auto-added if missing.",
        )

    # Normalize
    subject_town = town_in.upper().strip()
    flat_model   = flat_model_in.strip()
    street       = street_in.upper().strip()
    comp_towns   = [t.upper().strip() for t in comp_in.split(",") if t.strip()]

    # Validate
    errors = []
    if subject_town not in available_towns:
        errors.append(f"Unknown town {subject_town!r}. Pick one of: {available_towns}")
    if flat_model not in available_models:
        errors.append(f"Unknown flat model {flat_model!r}. Pick one of: {available_models}")
    bad_comp = [t for t in comp_towns if t not in available_towns]
    if bad_comp:
        errors.append(f"Unknown comparable town(s): {bad_comp}. Pick from: {available_towns}")
    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    if subject_town not in comp_towns:
        st.info(f"Subject town **{subject_town}** not in comparable list — auto-adding.")
        comp_towns = [subject_town, *comp_towns]

    flat_type_label = flat_type.title().replace(" ", "-")
    subject = {
        "town":       subject_town,
        "flat_type":  flat_type,
        "flat_model": flat_model,
        "floor":      int(floor),
        "area_sqm":   float(area_sqm),
        "lease_left": float(lease_left),
        "street":     street,
    }

    st.caption(
        f"Valuing **{flat_type_label} {flat_model}** · "
        f"floor {subject['floor']} · {subject['area_sqm']:g} sqm · "
        f"{subject['lease_left']:.2f} yrs lease · in **{subject_town.title()}** "
        f"(comparables: {', '.join(t.title() for t in comp_towns)})"
    )

    # ── Build comparable universe ──
    df_comp, df_comp_recent = build_universe(df, subject_town, flat_type, comp_towns, INCLUDED_MODELS)
    if len(df_comp_recent) < 30:
        st.error(
            f"Comparable universe too small ({len(df_comp_recent)} txns since 2024). "
            "Widen the comparable towns or pick a more common flat type."
        )
        st.stop()

    val_month = _months_since_2024(datetime.now())

    # ── Universe stats ──
    st.subheader("Comparable universe")
    c1, c2, c3 = st.columns(3)
    c1.metric("Full universe (2017+)", f"{len(df_comp):,}")
    c2.metric("Recent (2024+)",        f"{len(df_comp_recent):,}")
    c3.metric("Towns",                 len(comp_towns))

    by_town  = df_comp_recent["town"].value_counts().rename_axis("Town").reset_index(name="Recent txns")
    by_model = df_comp_recent["flat_model"].value_counts().rename_axis("Flat model").reset_index(name="Recent txns")
    cc1, cc2 = st.columns(2)
    cc1.dataframe(by_town,  hide_index=True, use_container_width=True)
    cc2.dataframe(by_model, hide_index=True, use_container_width=True)

    # ── Run valuation ──
    factors = fit_factors(df_comp_recent)
    comp    = valuation_comparables(df_comp_recent, subject, factors, val_month)
    reg_a   = valuation_regression(df_comp_recent, subject, val_month, with_time=True)
    reg_b   = valuation_regression(df_comp_recent, subject, val_month, with_time=False)
    blended = blend(comp, reg_a, reg_b, subject)

    # ── Headline ──
    st.subheader("Blended fair value")
    h1, h2, h3 = st.columns(3)
    h1.metric("Point estimate",  f"${blended['blended']:,.0f}")
    h2.metric("Range",           f"${blended['low']:,.0f} – ${blended['high']:,.0f}")
    h3.metric("Price per sqft",  f"${blended['psf']:,.0f}")

    # ── Plots ──
    is_subj_type    = df["flat_type"] == flat_type
    df_all_recent   = df.loc[is_subj_type & (df["month"].dt.year >= 2024)]
    df_subject_full = df.loc[is_subj_type & (df["town"] == subject_town)]

    st.subheader("Plots")
    p1, p2 = st.columns(2)
    with p1:
        st.pyplot(plot_towns_boxen(df_all_recent, subject_town, flat_type_label))
        st.pyplot(plot_adjusted_distribution(comp["top_comps"]))
    with p2:
        if len(df_subject_full):
            st.pyplot(plot_subject_time_series(df_subject_full, subject_town, flat_type_label))
        else:
            st.info(f"No {flat_type_label} transactions in {subject_town.title()} — time-series omitted.")
        st.pyplot(plot_subject_vs_market(
            df_comp_recent, subject, subject_town, flat_type_label,
            blended["blended"], blended["low"], blended["high"],
        ))

    # ── Top 20 comparables ──
    st.subheader("Top 20 comparable transactions (adjusted)")
    cols = ["month", "town", "block", "street_name", "storey_range",
            "floor_area_sqm", "remaining_lease_years", "flat_model",
            "resale_price", "total_adjustment", "adjusted_price"]
    st.dataframe(
        comp["top_comps"][cols].sort_values("month", ascending=False).reset_index(drop=True),
        use_container_width=True, hide_index=True,
    )

    # ── Approach comparison ──
    st.subheader("Approach comparison")
    sqft = subject["area_sqm"] * SQM_TO_SQFT
    summary = pd.DataFrame({
        "Approach": [
            "1. Historical Comparables",
            "2a. Regression (with time)",
            "2b. Regression (no time)",
            "Blended Estimate",
        ],
        "Point estimate": [comp["estimate"], reg_a["estimate"], reg_b["estimate"], blended["blended"]],
        "Low":  [comp["q25"], reg_a["estimate"] - reg_a["residual_std"], reg_b["estimate"] - reg_b["residual_std"], blended["low"]],
        "High": [comp["q75"], reg_a["estimate"] + reg_a["residual_std"], reg_b["estimate"] + reg_b["residual_std"], blended["high"]],
        "$/sqft": [comp["estimate"] / sqft, reg_a["estimate"] / sqft, reg_b["estimate"] / sqft, blended["psf"]],
    })
    st.dataframe(
        summary.style.format({
            "Point estimate": "${:,.0f}",
            "Low":            "${:,.0f}",
            "High":           "${:,.0f}",
            "$/sqft":         "${:,.0f}",
        }),
        use_container_width=True, hide_index=True,
    )

    # ── Diagnostics ──
    with st.expander("Regression diagnostics & adjustment factors"):
        d1, d2 = st.columns(2)
        d1.markdown(
            f"**Approach 2a (with time)**\n\n"
            f"- R²: {reg_a['r2']:.3f}\n"
            f"- MAE: ${reg_a['mae']:,.0f}\n"
            f"- n: {reg_a['n']:,}\n"
            f"- Residual std: ${reg_a['residual_std']:,.0f}"
        )
        d2.markdown(
            f"**Approach 2b (no time)**\n\n"
            f"- R²: {reg_b['r2']:.3f}\n"
            f"- MAE: ${reg_b['mae']:,.0f}\n"
            f"- n: {reg_b['n']:,}\n"
            f"- Residual std: ${reg_b['residual_std']:,.0f}"
        )
        st.markdown("**Approach 1 univariate factors:**")
        st.dataframe(
            pd.DataFrame({
                "Factor": ["Floor ($/floor)", "Lease ($/year)", "Area ($/sqm)", "Time ($/month)"],
                "Slope":  [factors["floor-level"], factors["lease"], factors["area"], factors["time"]],
            }).style.format({"Slope": "${:,.0f}"}),
            use_container_width=True, hide_index=True,
        )
        st.caption(f"Valuation month index (months since Jan 2024): {val_month}")


if __name__ == "__main__":
    main()
