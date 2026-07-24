"""
Matplotlib plot helpers. Each returns a Figure so the caller decides how to
render it (`st.pyplot`, `plt.show`, notebook display). Importing this module
applies the shared DJQC style so the dashboard and notebook look identical.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

from .config import DJQC_COLORS, DJQC_RC, Subject


def apply_style() -> None:
    """Apply the shared seaborn/matplotlib styling."""
    sns.set_style("darkgrid", rc=DJQC_RC)
    sns.set_palette(sns.color_palette(DJQC_COLORS))


apply_style()


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


def plot_subject_vs_market(df_comp_recent, subject: Subject, blended, low, high):
    fig, ax = plt.subplots(figsize=(8, 4))
    subject_town = subject.town
    flat_type_label = subject.flat_type_label
    sub = df_comp_recent.loc[df_comp_recent["town"] == subject_town]
    ax.scatter(sub["floor_area_sqm"], sub["resale_price"],
               s=20, alpha=0.4, color="grey",
               label=f"{subject_town.title()} {flat_type_label} (2024+)")
    if subject.street:
        street_recent = sub.loc[sub["street_name"].str.contains(subject.street, na=False)]
        if len(street_recent):
            ax.scatter(street_recent["floor_area_sqm"], street_recent["resale_price"],
                       s=30, alpha=0.7, color=DJQC_COLORS[1],
                       label=f"{subject.street.title()} Road")
    ax.scatter([subject.area_sqm], [blended], s=200, color=DJQC_COLORS[2],
               marker="*", zorder=5, label=f"Subject (est. ${blended:,.0f})")
    ax.errorbar(subject.area_sqm, blended,
                yerr=[[blended - low], [high - blended]],
                color=DJQC_COLORS[2], capsize=6, linewidth=2, zorder=4)
    ax.set_title(f"Subject vs {subject_town.title()} Market ({flat_type_label}, 2024+)")
    ax.set_xlabel("Floor Area (sqm)")
    ax.set_ylabel("Resale Price ($)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
