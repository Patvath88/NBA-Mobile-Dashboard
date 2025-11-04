# utils/feature_engineer.py
import pandas as pd
import numpy as np
from utils.data_loader import get_player_gamelog, get_team_defensive_metrics

# ------------------------------
# 🔹 HELPER FUNCTIONS
# ------------------------------

def add_rolling_features(df: pd.DataFrame, windows=[5, 10, 20]) -> pd.DataFrame:
    """Add rolling averages for key performance stats."""
    df = df.sort_values("GAME_DATE")

    key_stats = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M"]
    for stat in key_stats:
        for w in windows:
            df[f"{stat}_avg_{w}"] = df[stat].rolling(window=w, min_periods=1).mean()
    return df


def add_usage_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Approximate usage rate using field goals, free throws, and turnovers.
    True usage rate requires play-by-play data, but this gives a good proxy.
    """
    df["USG%"] = ((df["FGA"] + 0.44 * df["FTA"] + df["TOV"]) / df["MIN"].replace(0, np.nan)) * 100
    return df


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple contextual features such as home/away and rest days."""
    df["Home"] = df["MATCHUP"].apply(lambda x: 0 if "@" in x else 1)
    df["Rest_Days"] = df["GAME_DATE"].diff().dt.days.fillna(1)
    return df


ddef merge_opponent_defense(df: pd.DataFrame, season: str = "2024-25"):
    """Merge opponent defensive averages into player game logs."""
    team_def = get_team_defensive_metrics(season)

    # Safely rename if needed
    if "team.full_name" in team_def.columns:
        team_def.rename(columns={"team.full_name": "Team"}, inplace=True)

    # Extract opponent name substring from MATCHUP (e.g., "LAL vs BOS" -> "BOS")
    df["Opp_Team"] = df["MATCHUP"].str.extract(r'vs\. (\w+)|@ (\w+)').bfill(axis=1).iloc[:, 0]
    df["Opp_Team"] = df["Opp_Team"].str.strip()

    # Perform merge on cleaned team names
    df = df.merge(team_def, how="left", left_on="Opp_Team", right_on="Team")
    df.drop(columns=["Team"], inplace=True, errors="ignore")

    return df



def add_composite_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Create combined metrics like PRA and fantasy points."""
    df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    df["Fantasy_Pts"] = (
        df["PTS"] + 1.2 * df["REB"] + 1.5 * df["AST"] + 3 * (df["STL"] + df["BLK"]) - df["TOV"]
    )
    return df


# ------------------------------
# 🔹 MAIN PIPELINE FUNCTION
# ------------------------------

def build_feature_dataset(player_id: int, season: str = "2024-25") -> pd.DataFrame:
    """
    Full pipeline to prepare a player's feature dataset for modeling or predictions.
    Returns DataFrame with engineered features.
    """
    df = get_player_gamelog(player_id, season, last_n=20)

    df = add_context_features(df)
    df = add_rolling_features(df)
    df = add_usage_rate(df)
    df = add_composite_metrics(df)
    df = merge_opponent_defense(df, season)

    # Drop irrelevant columns and clean
    keep_cols = [
        "GAME_DATE", "MATCHUP", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M",
        "PRA", "Fantasy_Pts", "USG%", "Home", "Rest_Days"
    ] + [col for col in df.columns if "_avg_" in col]

    df_final = df[keep_cols].copy().fillna(0)
    return df_final.round(2)
