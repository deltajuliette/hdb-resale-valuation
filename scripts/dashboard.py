"""
HDB Resale Valuation Dashboard
==============================
Interactive front-end for the shared valuation pipeline in the `hdb_valuation`
package (the same code the notebook uses). This file is UI only: sidebar
widgets, input validation, orchestration, and rendering. Configure the subject
property and comparable universe in the sidebar; the three valuation approaches
and supporting plots re-render on every change.

Run with:
    streamlit run scripts/dashboard.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

# Make the repo-root `hdb_valuation` package importable when Streamlit runs this
# file from the scripts/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hdb_valuation as V


# ── Data loading (cached wrapper around the pure package functions) ───────────
@st.cache_data(show_spinner="Downloading HDB resale dataset…", ttl=24 * 3600)
def load_data() -> pd.DataFrame:
    """Fetch the HDB resale dataset from data.gov.sg and engineer features."""
    return V.engineer_features(V.fetch_raw_csv())


# ── Streamlit UI ─────────────────────────────────────────────────────────────
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

    subject = V.Subject(
        town=subject_town,
        flat_type=flat_type,
        flat_model=flat_model,
        floor=int(floor),
        area_sqm=float(area_sqm),
        lease_left=float(lease_left),
        street=street,
    )
    flat_type_label = subject.flat_type_label

    st.caption(
        f"Valuing **{flat_type_label} {flat_model}** · "
        f"floor {subject.floor} · {subject.area_sqm:g} sqm · "
        f"{subject.lease_left:.2f} yrs lease · in **{subject_town.title()}** "
        f"(comparables: {', '.join(t.title() for t in comp_towns)})"
    )

    # ── Build comparable universe ──
    df_comp, df_comp_recent = V.build_universe(df, subject_town, flat_type, comp_towns, V.INCLUDED_MODELS)
    if len(df_comp_recent) < V.DEFAULT_CONFIG.min_universe:
        st.error(
            f"Comparable universe too small ({len(df_comp_recent)} txns since 2024). "
            "Widen the comparable towns or pick a more common flat type."
        )
        st.stop()

    val_month = V.months_since_2024(datetime.now())

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
    factors = V.fit_factors(df_comp_recent)
    comp    = V.valuation_comparables(df_comp_recent, subject, factors, val_month)
    reg_a   = V.valuation_regression(df_comp_recent, subject, val_month, with_time=True)
    reg_b   = V.valuation_regression(df_comp_recent, subject, val_month, with_time=False)
    blended = V.blend(comp, reg_a, reg_b, subject)

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
        st.pyplot(V.plots.plot_towns_boxen(df_all_recent, subject_town, flat_type_label))
        st.pyplot(V.plots.plot_adjusted_distribution(comp["top_comps"]))
    with p2:
        if len(df_subject_full):
            st.pyplot(V.plots.plot_subject_time_series(df_subject_full, subject_town, flat_type_label))
        else:
            st.info(f"No {flat_type_label} transactions in {subject_town.title()} — time-series omitted.")
        st.pyplot(V.plots.plot_subject_vs_market(
            df_comp_recent, subject, blended["blended"], blended["low"], blended["high"],
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
    sqft = subject.sqft
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
