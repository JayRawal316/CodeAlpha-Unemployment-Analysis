"""
main.py
-------
Single entry point that runs the full pipeline:
    load -> clean -> analyze -> visualize -> print summary

Run from anywhere (paths are resolved relative to this file, not the
current working directory):

    python src/main.py
"""
import matplotlib.pyplot as plt
import pandas as pd

from data_cleaning import load_clean_data
from analysis import (
    area_comparison, state_ranking, zone_ranking,
    monthly_national_average, covid_period_summary, state_covid_impact,
    seasonal_pattern, recovery_curve,
    train_unemployment_regressor, cluster_states_by_risk,
)
from visualization import (
    plot_distribution, plot_state_ranking, plot_zone_ranking,
    plot_covid_trend, plot_state_impact, plot_seasonal_pattern,
    plot_recovery_curve, plot_model_performance, plot_clusters,
)

pd.set_option("display.float_format", lambda x: f"{x:.2f}")


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    section("1. LOADING & CLEANING DATA")
    df1, df2 = load_clean_data()
    print(f"Dataset 1 (India, Rural/Urban): {df1.shape} | "
          f"{df1['Date'].min().date()} -> {df1['Date'].max().date()}")
    print(f"Dataset 2 (2020, Zone/Geo):     {df2.shape} | "
          f"{df2['Date'].min().date()} -> {df2['Date'].max().date()}")

    section("2. EXPLORATORY DATA ANALYSIS")
    area_avg = area_comparison(df1)
    print("Rural vs Urban average unemployment rate:\n", area_avg)

    state_avg = state_ranking(df1)
    print("\nTop 5 highest-unemployment states:\n", state_avg.head())
    print("\nTop 5 lowest-unemployment states:\n", state_avg.tail())

    zone_avg = zone_ranking(df2)
    print("\nZone-wise average unemployment rate:\n", zone_avg)

    plot_distribution(df1)
    plot_state_ranking(state_avg)
    plot_zone_ranking(zone_avg)

    section("3. COVID-19 IMPACT ANALYSIS")
    monthly_natl = monthly_national_average(df1)
    covid_summary = covid_period_summary(df1)
    print("National average by lockdown period:\n", covid_summary)

    impact = state_covid_impact(df1)
    print("\nStates hit hardest (Apr-2020 vs pre-COVID baseline, pp increase):\n", impact.head())

    plot_covid_trend(monthly_natl)
    plot_state_impact(impact)

    section("4. SEASONAL TRENDS & POST-LOCKDOWN RECOVERY")
    pattern = seasonal_pattern(df1)
    print("Average unemployment rate by calendar month:\n", pattern)

    recovery = recovery_curve(df2)
    print("\nMonthly national average, Jan-Oct 2020 (recovery curve):\n", recovery)

    plot_seasonal_pattern(pattern)
    plot_recovery_curve(recovery)

    section("5. MACHINE LEARNING - RANDOM FOREST REGRESSION")
    reg = train_unemployment_regressor(df1)
    print(f"R-squared : {reg.r2:.3f}")
    print(f"RMSE      : {reg.rmse:.2f} pp")
    print(f"MAE       : {reg.mae:.2f} pp")
    print("\nFeature importances:\n", reg.importances.sort_values(ascending=False))

    plot_model_performance(reg)

    section("6. MACHINE LEARNING - K-MEANS CLUSTERING")
    state_features = cluster_states_by_risk(df2)
    print(state_features["Risk_Level"].value_counts())
    print(
        "\nHigh-risk states:\n",
        state_features.loc[state_features["Risk_Level"] == "High Risk"]
        .sort_values("Avg_Unemployment", ascending=False)[["Region", "Avg_Unemployment"]]
        .to_string(index=False),
    )

    plot_clusters(state_features)

    section("DONE")
    print("All charts saved to the images/ folder.")
    plt.close("all")


if __name__ == "__main__":
    main()
