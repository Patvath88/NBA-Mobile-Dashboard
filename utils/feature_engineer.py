import pandas as pd
import streamlit as st
from utils.data_loader import get_team_defensive_metrics

@st.cache_data(ttl=3600)
def build_feature_dataset(df):
    """
    Safely build feature dataset for XGBoost model training and prediction.
    Ensures DataFrame integrity and handles missing or malformed input.
    """
    import pandas as pd

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        print("⚠️ Warning: build_feature_dataset received an empty or invalid DataFrame.")
        return pd.DataFrame()  # Return an empty DataFrame to prevent crashes

    df = df.copy()

    # Add feature engineering safely
    try:
        if "PTS" in df.columns and "REB" in df.columns and "AST" in df.columns:
            df["PRA"] = df["PTS"] + df["REB"] + df["AST"]

        # Rolling averages (smooth trends)
        for col in ["PTS", "REB", "AST"]:
            if col in df.columns:
                df[f"{col}_roll5"] = df[col].rolling(5, min_periods=1).mean()

        # Basic efficiency
        df["EFF"] = df.get("PTS", 0) + df.get("REB", 0) + df.get("AST", 0) \
                    + df.get("STL", 0) + df.get("BLK", 0) - df.get("TOV", 0)

        df = df.fillna(0)
        return df

    except Exception as e:
        print(f"⚠️ Feature engineering failed: {e}")
        return df.fillna(0)
