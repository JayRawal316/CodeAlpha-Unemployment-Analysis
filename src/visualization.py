"""
visualization.py
-----------------
All plotting functions for the project. Each function builds a matplotlib
Figure, saves it as a PNG into the `images/` folder, and returns the Figure
(so callers — e.g. a notebook — can also display it inline, or close it
immediately if running headless via `main.py`).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100

IMAGES_DIR = Path(__file__).resolve().parent.parent / "images"
IMAGES_DIR.mkdir(exist_ok=True)


def _save(fig: plt.Figure, filename: str) -> Path:
    path = IMAGES_DIR / filename
    fig.savefig(path, bbox_inches="tight", dpi=120)
    return path


def plot_distribution(df1: pd.DataFrame) -> plt.Figure:
    """Histogram of unemployment rate + rural vs urban boxplot."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.histplot(df1["Unemployment_Rate"], bins=40, kde=True, color="#2c7fb8", ax=axes[0])
    axes[0].set_title("Distribution of Unemployment Rate (%)")
    axes[0].set_xlabel("Unemployment Rate (%)")

    sns.boxplot(
        data=df1, x="Area", y="Unemployment_Rate", hue="Area",
        palette={"Rural": "#41ab5d", "Urban": "#e6550d"}, legend=False, ax=axes[1],
    )
    axes[1].set_title("Unemployment Rate: Rural vs Urban")
    fig.tight_layout()
    _save(fig, "01_distribution_rural_urban.png")
    return fig


def plot_state_ranking(state_avg: pd.Series) -> plt.Figure:
    """Horizontal bar chart of every state's average unemployment rate."""
    fig, ax = plt.subplots(figsize=(9, 8))
    colors = plt.cm.get_cmap("RdYlGn_r")(np.linspace(0.15, 0.95, len(state_avg)))
    state_avg.sort_values().plot(kind="barh", color=colors, ax=ax)
    ax.set_xlabel("Average Unemployment Rate (%)")
    ax.set_title("Average Unemployment Rate by State (May 2019 - Jun 2020)")
    fig.tight_layout()
    _save(fig, "02_state_ranking.png")
    return fig


def plot_zone_ranking(zone_avg: pd.Series) -> plt.Figure:
    """Bar chart of average unemployment rate by geographic zone."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    zone_avg.plot(kind="bar", color=sns.color_palette("viridis", len(zone_avg)), ax=ax)
    ax.set_ylabel("Average Unemployment Rate (%)")
    ax.set_title("Average Unemployment Rate by Zone (2020)")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    _save(fig, "03_zone_ranking.png")
    return fig


def plot_covid_trend(monthly_national: pd.Series) -> plt.Figure:
    """National monthly average unemployment rate with the lockdown highlighted."""
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(monthly_national.index, monthly_national.values, marker="o", color="#08519c", linewidth=2)
    ax.axvspan(
        pd.Timestamp("2020-03-25"), pd.Timestamp("2020-05-31"),
        color="red", alpha=0.12, label="Nationwide lockdown",
    )
    ax.set_title("National Average Unemployment Rate - the COVID-19 Shock")
    ax.set_ylabel("Unemployment Rate (%)")
    ax.legend()
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    _save(fig, "04_covid_trend.png")
    return fig


def plot_state_impact(impact: pd.Series, top_n: int = 10) -> plt.Figure:
    """Bar chart of the states with the largest COVID-driven unemployment spike."""
    fig, ax = plt.subplots(figsize=(9, 6))
    impact.head(top_n).sort_values().plot(kind="barh", color="#cb181d", ax=ax)
    ax.set_xlabel("Increase in Unemployment Rate, Apr 2020 vs. pre-COVID baseline (pp)")
    ax.set_title(f"{top_n} States Hit Hardest by the Lockdown")
    fig.tight_layout()
    _save(fig, "05_state_covid_impact.png")
    return fig


def plot_seasonal_pattern(pattern: pd.Series) -> plt.Figure:
    """Bar chart of average unemployment rate by calendar month."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    pattern.plot(kind="bar", color=sns.color_palette("crest", len(pattern)), ax=ax)
    ax.set_ylabel("Average Unemployment Rate (%)")
    ax.set_title("Average Unemployment Rate by Calendar Month (all years pooled)")
    ax.tick_params(axis="x", rotation=40)
    fig.tight_layout()
    _save(fig, "06_seasonal_pattern.png")
    return fig


def plot_recovery_curve(recovery: pd.Series) -> plt.Figure:
    """Line chart of the post-lockdown recovery, Jan-Oct 2020."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(recovery.index, recovery.values, marker="o", color="#238b45", linewidth=2)
    ax.axvspan(pd.Timestamp("2020-03-25"), pd.Timestamp("2020-05-31"), color="red", alpha=0.12, label="Lockdown")
    ax.set_title("Recovery After the Lockdown (Jan - Oct 2020)")
    ax.set_ylabel("Unemployment Rate (%)")
    ax.legend()
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    _save(fig, "07_recovery_curve.png")
    return fig


def plot_model_performance(reg_result) -> plt.Figure:
    """Feature importance + actual-vs-predicted scatter for the regression model."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    reg_result.importances.plot(kind="barh", color="#3182bd", ax=axes[0])
    axes[0].set_title("Random Forest - Feature Importance")

    axes[1].scatter(reg_result.y_test, reg_result.y_pred, alpha=0.5, color="#08519c")
    lims = [
        min(reg_result.y_test.min(), reg_result.y_pred.min()),
        max(reg_result.y_test.max(), reg_result.y_pred.max()),
    ]
    axes[1].plot(lims, lims, "r--", linewidth=1.5)
    axes[1].set_xlabel("Actual Unemployment Rate (%)")
    axes[1].set_ylabel("Predicted Unemployment Rate (%)")
    axes[1].set_title(f"Actual vs. Predicted (R\u00b2 = {reg_result.r2:.2f})")

    fig.tight_layout()
    _save(fig, "08_model_performance.png")
    return fig


def plot_clusters(state_features: pd.DataFrame) -> plt.Figure:
    """Scatter plot of states colored by K-Means risk cluster."""
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {"Low Risk": "#31a354", "Moderate Risk": "#fd8d3c", "High Risk": "#de2d26"}

    for level, group in state_features.groupby("Risk_Level"):
        ax.scatter(
            group["Avg_Participation"], group["Avg_Unemployment"], s=90,
            label=level, color=colors.get(level, "#999999"), edgecolor="white", linewidth=0.5,
        )

    for _, row in state_features.iterrows():
        if row["Risk_Level"] == "High Risk":
            ax.annotate(
                row["Region"], (row["Avg_Participation"], row["Avg_Unemployment"]),
                fontsize=8, xytext=(4, 4), textcoords="offset points",
            )

    ax.set_xlabel("Average Labour Participation Rate (%)")
    ax.set_ylabel("Average Unemployment Rate (%)")
    ax.set_title("State Clusters by Unemployment Risk Profile (2020)")
    ax.legend()
    fig.tight_layout()
    _save(fig, "09_state_clusters.png")
    return fig
