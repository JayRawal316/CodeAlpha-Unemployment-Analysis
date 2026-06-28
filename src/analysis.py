"""
analysis.py
-----------
Core analysis functions: EDA summaries, COVID-19 impact quantification,
seasonal trend extraction, and the two machine learning components
(Random Forest regression + K-Means clustering).

Every function takes an already-cleaned dataframe (see `data_cleaning.py`)
and returns a pandas Series/DataFrame or a small result object — no
plotting happens here, that's `visualization.py`'s job.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

LOCKDOWN_START = "2020-03-25"
LOCKDOWN_END = "2020-05-31"

MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


# --------------------------------------------------------------------- #
# Exploratory data analysis
# --------------------------------------------------------------------- #
def area_comparison(df1: pd.DataFrame) -> pd.Series:
    """Average unemployment rate, Rural vs Urban (dataset 1 only)."""
    return df1.groupby("Area")["Unemployment_Rate"].mean().round(2)


def state_ranking(df1: pd.DataFrame) -> pd.Series:
    """States ranked by average unemployment rate, highest first."""
    return (
        df1.groupby("Region")["Unemployment_Rate"]
        .mean()
        .sort_values(ascending=False)
        .round(2)
    )


def zone_ranking(df2: pd.DataFrame) -> pd.Series:
    """Geographic zones ranked by average unemployment rate, highest first."""
    return (
        df2.groupby("Zone")["Unemployment_Rate"]
        .mean()
        .sort_values(ascending=False)
        .round(2)
    )


# --------------------------------------------------------------------- #
# COVID-19 impact
# --------------------------------------------------------------------- #
def monthly_national_average(df: pd.DataFrame) -> pd.Series:
    """National average unemployment rate, resampled to monthly."""
    series = df.groupby(df["Date"].dt.to_period("M"))["Unemployment_Rate"].mean()
    series.index = series.index.to_timestamp()
    return series


def covid_period_summary(df1: pd.DataFrame) -> pd.Series:
    """National average rate: pre-lockdown vs during vs post (Jun-20)."""
    pre = df1.loc[df1["Date"] < LOCKDOWN_START, "Unemployment_Rate"].mean()
    during = df1.loc[
        (df1["Date"] >= LOCKDOWN_START) & (df1["Date"] <= LOCKDOWN_END),
        "Unemployment_Rate",
    ].mean()
    post = df1.loc[df1["Date"] > LOCKDOWN_END, "Unemployment_Rate"].mean()
    return pd.Series(
        {"Pre-lockdown": pre, "During lockdown": during, "Post-lockdown (Jun-20)": post}
    ).round(2)


def state_covid_impact(df1: pd.DataFrame) -> pd.Series:
    """April-2020 average minus each state's pre-lockdown baseline (pp), sorted desc."""
    pre_by_state = (
        df1[df1["Date"] < LOCKDOWN_START].groupby("Region")["Unemployment_Rate"].mean()
    )
    peak_by_state = (
        df1[df1["Date"].dt.to_period("M") == "2020-04"]
        .groupby("Region")["Unemployment_Rate"]
        .mean()
    )
    return (peak_by_state - pre_by_state).dropna().sort_values(ascending=False).round(2)


# --------------------------------------------------------------------- #
# Seasonal trends
# --------------------------------------------------------------------- #
def seasonal_pattern(df1: pd.DataFrame) -> pd.Series:
    """Average unemployment rate by calendar month, all years pooled."""
    pattern = (
        df1.groupby("Month_Name")["Unemployment_Rate"]
        .mean()
        .reindex(MONTH_ORDER)
        .dropna()
    )
    return pattern.round(2)


def recovery_curve(df2: pd.DataFrame) -> pd.Series:
    """National monthly average for dataset 2 (Jan-Oct 2020): the recovery curve."""
    return monthly_national_average(df2).round(2)


# --------------------------------------------------------------------- #
# Machine Learning #1 — Random Forest regression
# --------------------------------------------------------------------- #
@dataclass
class RegressionResult:
    model: RandomForestRegressor
    feature_names: list
    r2: float
    rmse: float
    mae: float
    y_test: np.ndarray
    y_pred: np.ndarray
    importances: pd.Series


def train_unemployment_regressor(
    df1: pd.DataFrame, random_state: int = 42
) -> RegressionResult:
    """Train a Random Forest Regressor to predict Unemployment_Rate.

    Features: state, area (rural/urban), month, year, number employed,
    and labour participation rate.
    """
    data = df1.copy()
    data["Region_enc"] = LabelEncoder().fit_transform(data["Region"])
    data["Area_enc"] = LabelEncoder().fit_transform(data["Area"])

    features = [
        "Region_enc", "Area_enc", "Month", "Year",
        "Employed", "Labour_Participation_Rate",
    ]
    X, y = data[features], data["Unemployment_Rate"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return RegressionResult(
        model=model,
        feature_names=features,
        r2=r2_score(y_test, y_pred),
        rmse=float(np.sqrt(mean_squared_error(y_test, y_pred))),
        mae=mean_absolute_error(y_test, y_pred),
        y_test=y_test.values,
        y_pred=y_pred,
        importances=pd.Series(model.feature_importances_, index=features).sort_values(),
    )


# --------------------------------------------------------------------- #
# Machine Learning #2 — K-Means clustering
# --------------------------------------------------------------------- #
def cluster_states_by_risk(
    df2: pd.DataFrame, n_clusters: int = 3, random_state: int = 42
) -> pd.DataFrame:
    """Segment states into Low / Moderate / High unemployment-risk tiers.

    Returns a dataframe with one row per state: average unemployment rate,
    average employed, average labour participation rate, and a `Risk_Level`
    label derived from K-Means clustering on the standardized features.
    """
    state_features = (
        df2.groupby("Region")
        .agg(
            Avg_Unemployment=("Unemployment_Rate", "mean"),
            Avg_Employed=("Employed", "mean"),
            Avg_Participation=("Labour_Participation_Rate", "mean"),
        )
        .reset_index()
    )

    X = StandardScaler().fit_transform(
        state_features[["Avg_Unemployment", "Avg_Employed", "Avg_Participation"]]
    )
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    state_features["Cluster"] = kmeans.fit_predict(X)

    # Name clusters by their average unemployment rate, low -> high
    order = state_features.groupby("Cluster")["Avg_Unemployment"].mean().sort_values().index
    risk_labels = ["Low Risk", "Moderate Risk", "High Risk"]
    label_map = {cluster_id: risk_labels[i] if i < len(risk_labels) else f"Tier {i+1}"
                 for i, cluster_id in enumerate(order)}
    state_features["Risk_Level"] = state_features["Cluster"].map(label_map)

    return state_features
