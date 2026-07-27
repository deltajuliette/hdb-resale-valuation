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
SNAPSHOT = Path(__file__).resolve().parent.parent / "data" / "snapshot.csv"


@st.cache_data(show_spinner="Loading HDB resale dataset…", ttl=24 * 3600)
def load_data() -> pd.DataFrame:
    """Load the HDB resale dataset and engineer features.

    Prefers a bundled `data/snapshot.csv` when present (the web/stlite build ships
    one, since live fetching from the browser is blocked by CORS); otherwise fetches
    live from data.gov.sg. `engineer_features` runs either way, so both paths yield
    identical modelling columns.
    """
    raw = pd.read_csv(SNAPSHOT) if SNAPSHOT.exists() else V.fetch_raw_csv()
    return V.engineer_features(raw)


def _default_subject(df, available_towns, available_models, available_flat_types) -> dict:
    """A neutral, data-derived default subject (no personal values on the page).

    Picks the most common town for a 4-room flat and dataset-median area/lease, so
    the page opens on a representative-but-anonymous example.
    """
    flat_type = "4 ROOM" if "4 ROOM" in available_flat_types else available_flat_types[0]
    sub = df[df["flat_type"] == flat_type]
    town = sub["town"].mode().iloc[0] if not sub.empty else available_towns[0]
    model = "Model A" if "Model A" in available_models else available_models[0]
    top_towns = list(sub["town"].value_counts().head(3).index) if not sub.empty else [town]
    return {
        "town": town,
        "flat_type": flat_type,
        "flat_model": model,
        "floor": 10,
        "area_sqm": round(float(sub["floor_area_sqm"].median()), 1) if not sub.empty else 90.0,
        "lease_left": round(float(sub["remaining_lease_years"].median()), 2) if not sub.empty else 90.0,
        "comp_towns": top_towns,
    }


# ── Streamlit UI ─────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="HDB Resale Valuation", layout="wide")
    st.title("HDB Resale Valuation")

    df = load_data()
    available_towns      = sorted(df["town"].unique())
    available_models     = sorted(df["flat_model"].unique())
    available_flat_types = sorted(df["flat_type"].unique())

    # ── Neutral, data-derived defaults (nothing personal on the public page) ──
    d = _default_subject(df, available_towns, available_models, available_flat_types)
    st.session_state.setdefault("town", d["town"])
    st.session_state.setdefault("flat_type", d["flat_type"])
    st.session_state.setdefault("flat_model", d["flat_model"])
    st.session_state.setdefault("floor", d["floor"])
    st.session_state.setdefault("area_sqm", d["area_sqm"])
    st.session_state.setdefault("lease_left", d["lease_left"])
    st.session_state.setdefault("comp_towns", d["comp_towns"])

    with st.sidebar:
        # ── Address auto-fill (optional convenience) ──
        # Placed above the subject widgets so a click updates session_state before
        # those widgets are drawn this run — no explicit rerun needed.
        st.header("📍 Find by address")
        st.caption("Pick a real block to auto-fill the fields below from its past sales.")
        streets = sorted(df["street_name"].dropna().unique())
        street_sel = st.selectbox("Street", streets, key="addr_street")
        blocks = sorted(df.loc[df["street_name"] == street_sel, "block"].astype(str).unique())
        block_sel = st.selectbox("Block", blocks, key="addr_block")
        if st.button("Use this address", use_container_width=True):
            sug = V.suggest_subject_from_address(
                df, str(street_sel), str(block_sel), st.session_state["flat_type"]
            )
            if sug:
                _clamp = lambda v, lo, hi: max(lo, min(hi, v))  # keep within widget ranges
                st.session_state["town"]       = sug["town"]
                st.session_state["flat_model"] = sug["flat_model"]
                st.session_state["floor"]      = int(_clamp(sug["storey_mid"], 1, 60))
                st.session_state["area_sqm"]   = float(_clamp(sug["area_sqm"], 10.0, 300.0))
                st.session_state["lease_left"] = float(_clamp(sug["lease_left"], 1.0, 99.0))
                if sug["town"] not in st.session_state["comp_towns"]:
                    st.session_state["comp_towns"] = [sug["town"], *st.session_state["comp_towns"]]
                st.success(
                    f"Filled from {sug['n_txns']} sale(s) on {street_sel} "
                    f"(basis: {sug['scope']})."
                )
            else:
                st.warning("No transactions found for that address.")

        st.header("Subject Property")
        st.selectbox("Town", available_towns, key="town")
        st.selectbox(
            "Flat type", available_flat_types, key="flat_type",
            format_func=lambda s: s.title().replace(" ", "-"),
        )
        st.selectbox("Flat model", available_models, key="flat_model")
        st.number_input("Floor (storey)",          min_value=1,    max_value=60,    step=1,    key="floor")
        st.number_input("Floor area (sqm)",        min_value=10.0, max_value=300.0, step=1.0,  key="area_sqm")
        st.number_input("Remaining lease (years)", min_value=1.0,  max_value=99.0,  step=0.01, format="%.2f", key="lease_left")

        st.header("Comparable universe")
        st.multiselect(
            "Comparable towns", available_towns, key="comp_towns",
            help="Subject town is auto-added if missing.",
        )

    # Read the (always-valid) selections back
    subject_town = st.session_state["town"]
    flat_type    = st.session_state["flat_type"]
    flat_model   = st.session_state["flat_model"]
    floor        = st.session_state["floor"]
    area_sqm     = st.session_state["area_sqm"]
    lease_left   = st.session_state["lease_left"]
    comp_towns   = list(st.session_state["comp_towns"])
    # Street highlight for plots: derive from the picked address (town-level match).
    street       = subject_town

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
    df_comp, df_comp_recent = V.build_universe(df, subject_town, flat_type, comp_towns, flat_model)
    if len(df_comp_recent) < V.DEFAULT_CONFIG.min_universe:
        st.error(
            f"Comparable universe too small ({len(df_comp_recent)} txns since 2024 "
            f"for **{flat_model}** {flat_type_label} in these towns). "
            "Comparables are matched to the subject's flat model — widen the "
            "comparable towns, or pick a more common flat type/model."
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
