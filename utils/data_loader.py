import pandas as pd
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
from nba_api.stats.endpoints import leaguedashteamstats
import time
import streamlit as st

@st.cache_data(ttl=3600)
def get_player_id(player_name: str):
    all_players = players.get_players()
    match = next((p for p in all_players if player_name.lower() in p["full_name"].lower()), None)
    return match["id"] if match else None

@st.cache_data(ttl=3600)
def get_player_context(player_id: int, season: str = "2024-25"):
    try:
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
        gamelog["GAME_DATE"] = pd.to_datetime(gamelog["GAME_DATE"])
        gamelog = gamelog.sort_values("GAME_DATE").reset_index(drop=True)
        gamelog["PRA"] = gamelog["PTS"] + gamelog["REB"] + gamelog["AST"]
        return {"recent_games": gamelog.tail(30)}
    except Exception as e:
        st.warning(f"⚠️ Error fetching player data: {e}")
        return None

@st.cache_data(ttl=86400)
def get_team_defensive_metrics(season: str = "2024-25"):
    try:
        df = leaguedashteamstats.LeagueDashTeamStats(season=season, per_mode_detailed="PerGame").get_data_frames()[0]
        return df[["TEAM_NAME", "DEF_RATING", "PACE"]]
    except Exception as e:
        st.warning(f"⚠️ Could not load defensive metrics: {e}")
        return pd.DataFrame()
