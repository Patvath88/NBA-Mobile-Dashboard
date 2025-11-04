# utils/feature_engineer.py
import pandas as pd
import numpy as np
from utils.data_loader import (
    get_player_gamelog,
    get_team_defensive_metrics,
    get_player_id,
)
import streamlit as st

# ------------------------------
# 🧩 Rolling & Usage Features
# ------------------------------
def add_rolling_features(df: pd.DataFrame, window: int = 5):
    """Add rolling averages for recent form."""
    numeric_cols = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG_PCT", "FG3M"]
    for col in numeric_cols:
        if col in df.columns:
            df[f"{col}_roll{window}"] = df[col].rolling(window, min_periods=1).mean()
    return df


def add_usage_rate(df: pd.DataFrame):
    """Approximate player usage rate using FGA, FTA, and TOV."""
    if not {"FGA", "FTA", "TOV", "MIN"}.issubset(df.columns):
        return df
    df["USG"] = (
        (df["FGA"] + 0.44 * df["FTA"] + df["TOV"])
        / (df["MIN"].replace(0, np.nan))
        * 100
    )
    df["USG"] = df["USG"].fillna(df["USG"].mean())
    return df


def add_composite_metrics(df: pd.DataFrame):
    """Add combined metrics like PRA, Efficiency."""
    if {"PTS", "REB", "AST"}.issubset(df.columns):
        df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    if {"PTS", "REB", "AST", "STL", "BLK", "TOV"}.issubset(df.columns):
        df["EFF"] = (
            df["PTS"] + df["REB"] + df["AST"] + df["STL"] + df["BLK"] - df["TOV"]
        )
    return df


# ------------------------------
# 🧩 Opponent Defense Merge
# ------------------------------
def merge_opponent_defense(df: pd.DataFrame, season: str = "2024-25"):
    """Merge opponent defensive metrics into player game logs."""
    team_def = get_team_defensive_metrics(season)

    if team_def is None or team_def.empty:
        st.warning("⚠️ No defensive metrics found; skipping opponent merge.")
        df["Opp_Def_Rtg"] = np.nan
        df["Opp_Pace"] = np.nan
        return df

    # Normalize column names
    team_def = team_def.rename(
        columns={
            "team.full_name": "Team",
            "TEAM_NAME": "Team",
            "TEAM": "Team",
        }
    )

    if "Team" not in team_def.columns:
        st.warning("⚠️ Defensive metrics missing 'Team' column; skipping merge.")
        df["Opp_Def_Rtg"] = np.nan
        df["Opp_Pace"] = np.nan
        return df

    # Extract opponent abbreviation (e.g. 'LAL vs BOS' -> 'BOS')
    df["Opp_Team"] = df["MATCHUP"].str.extract(r"vs\. (\w+)|@ (\w+)").bfill(axis=1).iloc[:, 0]
    df["Opp_Team"] = df["Opp_Team"].str.strip()

    # Attempt fuzzy merge
    from difflib import get_close_matches
    def find_match(opp):
        if pd.isna(opp):
            return None
        match = get_close_matches(opp, team_def["Team"].tolist(), n=1, cutoff=0.4)
        return match[0] if match else None

    df["TeamMatch"] = df["Opp_Team"].apply(find_match)

    merged = df.merge(team_def, how="left", left_on="TeamMatch", right_on="Team")
    merged.drop(columns=["Team"], inplace=True, errors="ignore")
    return merged


# ------------------------------
# 🧩 Feature Dataset Builder
# ------------------------------
@st.cache_data(ttl=3600)
def build_feature_dataset(player_id: int, season: str = "2024-25"):
    """Full feature pipeline for model training and prediction."""
    try:
        df = get_player_gamelog(player_id, season)
        if df is None or df.empty:
            st.warning("No player game logs available.")
            return pd.DataFrame()

        # Feature engineering
        df = add_rolling_features(df)
        df = add_usage_rate(df)
        df = add_composite_metrics(df)
        df = merge_opponent_defense(df, season)

        # Drop irrelevant or text-heavy columns
        drop_cols = [
            "MATCHUP", "VIDEO_AVAILABLE", "TeamMatch",
            "Opp_Team", "Game_ID", "TEAM_ABBREVIATION"
        ]
        df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors="ignore")

        # Handle missing numeric values
        df = df.select_dtypes(include=[np.number]).fillna(0)

        return df

    except Exception as e:
        st.error(f"Error building feature dataset: {e}")
        return pd.DataFrame()
