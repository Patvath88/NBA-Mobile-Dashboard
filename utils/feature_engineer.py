import pandas as pd
import streamlit as st
from utils.data_loader import get_team_defensive_metrics

@st.cache_data(ttl=3600)
def build_feature_dataset(df: pd.DataFrame, season: str = "2024-25"):
    df = df.copy()

    # Rolling averages for recent form
    for col in ["PTS", "REB", "AST", "STL", "BLK", "TOV"]:
        df[f"{col}_roll5"] = df[col].rolling(5, min_periods=1).mean()

    df["FG_PCT_roll5"] = df["FG_PCT"].rolling(5, min_periods=1).mean()
    df["FG3M_roll5"] = df["FG3M"].rolling(5, min_periods=1).mean()
    df["USG"] = (df["FGA"] + 0.44 * df["FTA"] + df["TOV"]) / df["MIN"]

    df["EFF"] = (df["PTS"] + df["REB"] + df["AST"] + df["STL"] + df["BLK"] -
                 ((df["FGA"] - df["FGM"]) + (df["FTA"] - df["FTM"]) + df["TOV"]))

    # Merge opponent defensive data
    team_def = get_team_defensive_metrics(season)
    if not team_def.empty and "MATCHUP" in df.columns:
        df["Opponent"] = df["MATCHUP"].apply(lambda x: x.split(" ")[-1])
        df = df.merge(team_def, left_on="Opponent", right_on="TEAM_NAME", how="left")
        df.rename(columns={"DEF_RATING": "Opp_Def_Rtg", "PACE": "Opp_Pace"}, inplace=True)
    else:
        df["Opp_Def_Rtg"] = 110
        df["Opp_Pace"] = 99

    df = df.fillna(0)
    return df
