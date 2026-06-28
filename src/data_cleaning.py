"""
data_cleaning.py
----------------
Loading and cleaning utilities for the two raw unemployment CSVs used in this
project.

Both source files (CMIE survey data, distributed via Kaggle) share the same
quirks straight from the source:
    - column names have leading/trailing whitespace (e.g. ' Date')
    - string values have stray whitespace (e.g. ' Rural')
    - dates are stored as text in DD-MM-YYYY format
    - `Unemployment in India.csv` has a block of fully blank rows
"""
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

INDIA_FILE = "Unemployment in India.csv"
RATE_2020_FILE = "Unemployment_Rate_upto_11_2020.csv"

RENAME_MAP = {
    "Estimated Unemployment Rate (%)": "Unemployment_Rate",
    "Estimated Employed": "Employed",
    "Estimated Labour Participation Rate (%)": "Labour_Participation_Rate",
}


def _basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace, drop blank rows, parse dates, add Year/Month helpers."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    df = df.dropna(how="all")

    for col in df.columns:
        if df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype(str).str.strip()

    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Month_Name"] = df["Date"].dt.month_name()
    df = df.rename(columns=RENAME_MAP)

    return df.sort_values("Date").reset_index(drop=True)


def load_raw_data(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the two raw CSVs exactly as provided (no cleaning applied yet)."""
    df1 = pd.read_csv(data_dir / INDIA_FILE)
    df2 = pd.read_csv(data_dir / RATE_2020_FILE)
    return df1, df2


def clean_india_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean `Unemployment in India.csv` (state x rural/urban, May'19-Jun'20)."""
    return _basic_clean(df)


def clean_2020_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean `Unemployment_Rate_upto_11_2020.csv` (state x zone, Jan-Oct 2020)."""
    df = _basic_clean(df)
    return df.rename(columns={"Region.1": "Zone"})


def load_clean_data(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper: load + clean both datasets in one call.

    Returns
    -------
    df1 : cleaned `Unemployment in India.csv` (has an `Area` column: Rural/Urban)
    df2 : cleaned `Unemployment_Rate_upto_11_2020.csv` (has a `Zone` column)
    """
    raw1, raw2 = load_raw_data(data_dir)
    return clean_india_dataset(raw1), clean_2020_dataset(raw2)


if __name__ == "__main__":
    india_df, rate_2020_df = load_clean_data()
    print("Dataset 1 (India, Rural/Urban):", india_df.shape)
    print("Dataset 2 (2020, Zone/Geo):    ", rate_2020_df.shape)
    print("\nDataset 1 preview:\n", india_df.head())
    print("\nDataset 2 preview:\n", rate_2020_df.head())
