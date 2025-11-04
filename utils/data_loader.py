import pandas as pd
import numpy as np
import requests
import streamlit as st

NBA_BASE_URL = "https://stats.nba.com/stats/"
HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}


def build_feature_dataset(player_id: int, season="2024-25"):
    """Builds rolling feature dataset for model training."""
    try:
        url = f"https://stats.nba.com/stats/playergamelog?PlayerID={player_id}&Season={season}&SeasonType=Regular+Season"
        resp = requests.get(url, headers=HEADERS)
        result = resp.json()["resultSets"][0]
        df = pd.DataFrame(result["rowSet"], columns=result["headers"])

        # Convert numeric fields
        numeric_cols = [
            "MIN", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
            "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB", "AST", "STL",
            "BLK", "TOV", "PF", "PTS", "PLUS_MINUS"
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        df = df.sort_values("GAME_DATE").reset_index(drop=True)

        # Rolling features
        roll_features = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG_PCT", "FG3M"]
        for f in roll_features:
            df[f"{f}_roll5"] = df[f].rolling(5, min_periods=1).mean()

        df["USG"] = df["FGA"] + df["FTA"] + df["TOV"]
        df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
        df["EFF"] = df["PTS"] + df["REB"] + df["AST"] + df["STL"] + df["BLK"] - ((df["FGA"] - df["FGM"]) + (df["FTA"] - df["FTM"]) + df["TOV"])

        df["Player_ID"] = player_id
        return df

    except Exception as e:
        st.error(f"Error building features: {e}")
        return pd.DataFrame()
