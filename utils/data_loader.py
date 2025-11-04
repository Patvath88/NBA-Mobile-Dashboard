import pandas as pd
import numpy as np
import streamlit as st
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, leaguedashteamstats
from requests.exceptions import RequestException

# ------------------------------------------------------
# 🏀 Player Data
# ------------------------------------------------------
@st.cache_data(ttl=3600)
def get_player_id(player_name: str):
    """Return the player ID using fuzzy match."""
    try:
        all_players = players.get_active_players()
        match = next((p for p in all_players if player_name.lower() in p["full_name"].lower()), None)
        if match:
            return match["id"]
        else:
            # Try closest fuzzy match
            from difflib import get_close_matches
            names = [p["full_name"] for p in all_players]
            close = get_close_matches(player_name, names, n=1, cutoff=0.4)
            if close:
                st.info(f"Using closest match: {close[0]}")
                pid = next((p["id"] for p in all_players if p["full_name"] == close[0]), None)
                return pid
        raise ValueError(f"Player '{player_name}' not found in NBA API.")
    except Exception as e:
        st.error(f"Error finding player: {e}")
        return None


@st.cache_data(ttl=3600)
def get_player_gamelog(player_id: int, season: str = "2024-25"):
    """Return player game logs."""
    try:
        res = playergamelog.PlayerGameLog(player_id=player_id, season=season)
        df = res.get_data_frames()[0]
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        df.sort_values("GAME_DATE", inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading player gamelog: {e}")
        return pd.DataFrame()


# ------------------------------------------------------
# 🛡️ Team Defense (fast + cached fallback)
# ------------------------------------------------------
@st.cache_data(ttl=604800)  # refresh weekly
def get_team_defensive_metrics(season: str = "2024-25"):
    """Get team defensive rating and pace from NBA API or cached fallback."""
    try:
        res = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            per_mode_detailed="PerGame",
            measure_type_detailed_defense="Defense"
        )
        df = res.get_data_frames()[0][["TEAM_NAME", "DEF_RATING", "PACE"]]
        df.rename(columns={"TEAM_NAME": "Team", "DEF_RATING": "DefRtg"}, inplace=True)
        st.success("✅ Team defensive metrics loaded from NBA API.")
        return df
    except (RequestException, Exception) as e:
        st.warning(f"⚠️ Using fallback defensive data ({e})")
        fallback_data = {
            "Team": [
                "Boston Celtics", "Milwaukee Bucks", "Denver Nuggets",
                "Dallas Mavericks", "Golden State Warriors", "Los Angeles Lakers",
                "Miami Heat", "Phoenix Suns"
            ],
            "DefRtg": [108.4, 110.1, 109.3, 112.0, 114.7, 113.9, 111.2, 110.9],
            "PACE": [97.4, 99.1, 98.5, 100.2, 101.8, 99.9, 96.7, 100.5]
        }
        return pd.DataFrame(fallback_data)


# ------------------------------------------------------
# 🔄 Context Builder
# ------------------------------------------------------
@st.cache_data(ttl=1800)
def get_player_context(player_name: str, season: str = "2024-25"):
    """Return player game context + season averages."""
    player_id = get_player_id(player_name)
    if not player_id:
        return None

    gamelog_df = get_player_gamelog(player_id, season)
    if gamelog_df.empty:
        return None

    recent = gamelog_df.tail(10).copy()
    recent["PRA"] = recent["PTS"] + recent["REB"] + recent["AST"]

    season_avg = gamelog_df.mean(numeric_only=True).round(1).to_dict()

    return {
        "recent_games": recent,
        "season_avg": season_avg
    }
