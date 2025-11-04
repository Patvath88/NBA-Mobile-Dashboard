# utils/feature_engineer.py
import pandas as pd
import numpy as np
import streamlit as st
from utils.data_loader import get_player_gamelog, get_team_defensive_metrics
from difflib import get_close_matches

# ============================================================
# 🧠 PLAYER FEATURE ENGINEERING PIPELINE
# ============================================================

# ------------------------------
# 🧩 Rolling & Usage Features
# ------------------------------
def add_rolling_features(df: pd.DataFrame, window: int = 5):
    """Add rolling averages for key stats."""
    if df.empty:
        return df
    numeric_cols = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG_PCT", "FG3M"]
    for col in numeric_cols:
        if col in df.columns:
            df[f"{col}_roll{window}"] = df[col].rolling(window, min_periods=1).mean()
    return df


def add_usage_rate(df: pd.DataFrame):
    """Estimate usage rate using FGA, FTA, and TOV."""
    if {"FGA", "FTA", "TOV", "MIN"}.issubset(df.columns):
        df["USG"] = (
            (df["FGA"] + 0.44 * df["FTA"] + df["TOV"])
            / df["MIN"].replace(0, np.nan)
        ) * 100
        df["USG"] = df["USG"].fillna(df["USG"].mean())
    else:
        df["USG"] = np.nan
    return df


def add_composite_metrics(df: pd.DataFrame):
    """Add PRA and efficiency (EFF) metrics."""
    if {"PTS", "REB", "AST"}.issubset(df.columns):
        df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    if {"PTS", "REB", "AST", "STL", "BLK", "TOV"}.issubset(df.columns):
        df["EFF"] = (
            df["PTS"] + df["REB"] + df["AST"] + df["STL"] + df["BLK"] - df["TOV"]
        )
    return df


# ------------------------------
# 🛡️ Opponent Defense Merge
# ------------------------------
def merge_opponent_defense(df: pd.DataFrame, season: str = "2024-25"):
    """Merge opponent defensive metrics (pace + defensive rating)."""
    try:
        team_def = get_team_defensive_metrics(season)

        if team_def is None or team_def.empty:
            st.warning("⚠️ No team defensive data found; skipping opponent merge.")
            df["Opp_Def_Rtg"] = np.nan
            df["Opp_Pace"] = np.nan
            return df

        # Normalize naming for consistency
        team_def = team_def.rename(
            columns={
                "team.full_name": "Team",
                "TEAM_NAME": "Team",
                "TEAM": "Team"
            }
        )

        if "Team" not in team_def.columns:
            st.warning("⚠️ 'Team' column not found in defensive metrics.")
            df["Opp_Def_Rtg"] = np.nan
            df["Opp_Pace"] = np.nan
            return df

        # Extract opponent team abbreviation from MATCHUP (e.g. 'LAL vs BOS' -> 'BOS')
        df["Opp_Team"] = df["MATCHUP"].str.extract(r"vs\. (\w+)|@ (\w+)").bfill(axis=1).iloc[:, 0]
        df["Opp_Team"] = df["Opp_Team"].str.strip()

        # Fuzzy match opponent name
        def match_team(opp):
            if pd.isna(opp):
                return None
            match = get_close_matches(opp, team_def["Team"].tolist(), n=1, cutoff=0.4)
            return match[0] if match else None

        df["TeamMatch"] = df["Opp_Team"].apply(match_team)

        merged = df.merge(team_def, how="left", left_on="TeamMatch", right_on="Team")
        merged.drop(columns=["Team"], inplace=True, errors="ignore")

        # If no numeric metrics exist, fill placeholders
        if "DefRtg" not in merged.columns:
            merged["DefRtg"] = np.nan
        if "Pace" not in merged.columns:
            merged["Pace"] = np.nan

        merged.rename(columns={"DefRtg": "Opp_Def_Rtg", "Pace": "Opp_Pace"}, inplace=True)

        return merged

    except Exception as e:
        st.error(f"Error merging opponent defense: {e}")
        df["Opp_Def_Rtg"] = np.nan
        df["Opp_Pace"] = np.nan
        return df


# ------------------------------
# 🧩 Full Dataset Builder
# ------------------------------
@st.cache_data(ttl=3600)
def build_feature_dataset(player_id: int, season: str = "2024-25"):
    """End-to-end feature builder for model training."""
    try:
        df = get_player_gamelog(player_id, season)

        if df is None or df.empty:
            st.warning("No player game logs found.")
            return pd.DataFrame()

        # Step 1: Add rolling stats & derived features
        df = add_rolling_features(df)
        df = add_usage_rate(df)
        df = add_composite_metrics(df)

        # Step 2: Merge defensive context
        df = merge_opponent_defense(df, season)

        # Step 3: Drop non-numeric or unnecessary columns
        drop_cols = [
            "MATCHUP", "VIDEO_AVAILABLE", "Opp_Team",
            "TeamMatch", "TEAM_ABBREVIATION", "GAME_ID"
        ]
        df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors="ignore")

        # Step 4: Keep only numeric features for modeling
        df = df.select_dtypes(include=[np.number]).fillna(0)

        return df

    except Exception as e:
        st.error(f"Error building feature dataset: {e}")
        return pd.DataFrame()
