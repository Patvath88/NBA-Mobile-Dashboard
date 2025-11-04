import requests
import pandas as pd
import streamlit as st

HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/plain, */*",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}


def get_player_id(player_name: str):
    """Fetch NBA player ID by name."""
    try:
        resp = requests.get(
            "https://stats.nba.com/stats/commonallplayers?LeagueID=00&Season=2024-25&IsOnlyCurrentSeason=1",
            headers=HEADERS,
        )
        data = resp.json()["resultSets"][0]["rowSet"]
        df = pd.DataFrame(data, columns=resp.json()["resultSets"][0]["headers"])
        match = df[df["DISPLAY_FIRST_LAST"].str.lower() == player_name.lower()]
        if not match.empty:
            return int(match.iloc[0]["PERSON_ID"])
        else:
            st.warning(f"⚠️ Player '{player_name}' not found.")
            return None
    except Exception as e:
        st.error(f"Error fetching player ID: {e}")
        return None


def get_player_context(player_name: str):
    """Return player's general info."""
    pid = get_player_id(player_name)
    if not pid:
        return {}
    try:
        resp = requests.get(
            f"https://stats.nba.com/stats/commonplayerinfo?PlayerID={pid}&LeagueID=00",
            headers=HEADERS,
        )
        result = resp.json()["resultSets"][0]["rowSet"][0]
        headers = resp.json()["resultSets"][0]["headers"]
        return dict(zip(headers, result))
    except Exception as e:
        st.error(f"Error fetching player context: {e}")
        return {}


def get_player_gamelog(player_id: int, season="2024-25"):
    """Fetches player game logs."""
    try:
        url = f"https://stats.nba.com/stats/playergamelog?PlayerID={player_id}&Season={season}&SeasonType=Regular+Season"
        resp = requests.get(url, headers=HEADERS)
        result = resp.json()["resultSets"][0]
        df = pd.DataFrame(result["rowSet"], columns=result["headers"])
        return df
    except Exception as e:
        st.error(f"Error fetching player gamelog: {e}")
        return pd.DataFrame()


def get_team_defensive_metrics():
    """Fetches basic team defensive stats (for opponent merge)."""
    try:
        url = "https://stats.nba.com/stats/leaguedashteamstats?Season=2024-25&SeasonType=Regular+Season&PerMode=PerGame"
        resp = requests.get(url, headers=HEADERS)
        result = resp.json()["resultSets"][0]
        df = pd.DataFrame(result["rowSet"], columns=result["headers"])
        df = df[["TEAM_ID", "TEAM_NAME", "DEF_RATING", "PACE"]].rename(
            columns={"DEF_RATING": "Opp_Def_Rtg", "PACE": "Opp_Pace"}
        )
        return df
    except Exception as e:
        st.warning(f"⚠️ No team defensive data found: {e}")
        return pd.DataFrame()
