import pandas as pd
import streamlit as st
from utils.data_loader import get_team_defensive_metrics

@st.cache_data(ttl=3600)
def build_feature_dataset(df):
    """
    Safely build feature dataset for XGBoost model training and prediction.
    Handles missing data and guarantees a valid DataFrame output.
    """
    import pandas as pd

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        print("⚠️ Warning: build_feature_dataset received an empty or invalid DataFrame.")
        return pd.DataFrame(columns=["PTS", "REB", "AST", "PRA"])

    df = df.copy()

    # Normalize column names
    df.columns = [c.upper().strip() for c in df.columns]

    # Fill missing key columns
    for col in ["PTS", "REB", "AST"]:
        if col not in df.columns:
            df[col] = 0

    # Create PRA (points + rebounds + assists)
    df["PRA"] = df["PTS"] + df["REB"] + df["AST"]

    # Rolling averages
    for col in ["PTS", "REB", "AST"]:
        df[f"{col}_ROLL5"] = df[col].rolling(5, min_periods=1).mean()

    # Basic efficiency metric
    df["EFF"] = (
        df.get("PTS", 0)
        + df.get("REB", 0)
        + df.get("AST", 0)
        + df.get("STL", 0)
        + df.get("BLK", 0)
        - df.get("TOV", 0)
    )

    df = df.fillna(0)
    return df
