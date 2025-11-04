import pandas as pd
import numpy as np
import streamlit as st
from utils.data_loader import get_player_gamelog, get_team_defensive_metrics


def build_feature_dataset(player_id: int, season="2024-25"):
    """Builds clean feature dataset with rolling metrics."""
    try:
        df = get_player_gamelog(player_id, season)
        if df.empty:
            return pd.DataFrame()

        numeric_cols = [
            "MIN", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
            "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB", "AST",
            "STL", "BLK", "TOV", "PF", "PTS", "PLUS_MINUS"
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        df = df.sort_values("GAME_DATE").reset_index(drop=True)

        roll_features = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG_PCT", "FG3M"]
        for f in roll_features:
            df[f"{f}_roll5"] = df[f].rolling(5, min_periods=1).mean()

        df["USG"] = df["FGA"] + df["FTA"] + df["TOV"]
        df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
        df["EFF"] = df["PTS"] + df["REB"] + df["AST"] + df["STL"] + df["BLK"] - (
            (df["FGA"] - df["FGM"]) + (df["FTA"] - df["FTM"]) + df["TOV"]
        )
        df["Player_ID"] = player_id

        # Merge with opponent team defensive stats
        team_def = get_team_defensive_metrics()
        if not team_def.empty and "TEAM_NAME" in df.columns:
            df = df.merge(team_def, how="left", left_on="MATCHUP", right_on="TEAM_NAME")

        return df
    except Exception as e:
        st.error(f"Error building features: {e}")
        return pd.DataFrame()
