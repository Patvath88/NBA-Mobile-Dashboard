# utils/data_loader.py
import pandas as pd
import streamlit as st
from nba_api.stats.endpoints import playergamelog, teamgamelog
from nba_api.stats.static import players, teams
from difflib import get_close_matches
from datetime import datetime
from pathlib import Path
import time

# --------------------------
# 🔹 PLAYER DATA FETCHING
# --------------------------

@st.cache_data(ttl=3600)
def get_player_id(player_name: str):
    """Return the NBA API player ID with fuzzy matching for similar names."""
    all_players = players.get_players()
    names = [p["full_name"] for p in all_players]

    # Exact match first
    exact = next((p for p in all_players if p["full_name"].lower() == player_name.lower()), None)
    if exact:
        return exact["id"]

    # Fuzzy match for accents or minor typos
    close = get_close_matches(player_name, names, n=1, cutoff=0.6)
    if close:
        match = next(p for p in all_players if p["full_name"] == close[0])
        st.info(f"Using closest match: **{match['full_name']}**")
        return match["id"]

    raise ValueError(f"Player '{player_name}' not found in NBA API.")


@st.cache_data(ttl=600)
def get_player_gamelog(player_id: int, season: str = "2024-25", last_n: int = 20):
    """Fetch last N games for the specified player."""
    try:
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season)
        df = gamelog.get_data_frames()[0]
        df = df.head(last_n)
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        df = df.sort_values("GAME_DATE", ascending=True)
        return df
    except Exception as e:
        st.error(f"Error fetching player game logs: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=600)
def get_season_averages(player_id: int, season: str = "2024-25"):
    """Compute player's current season average stats."""
    df = get_player_gamelog(player_id, season, last_n=82)
    if df.empty:
        return pd.DataFrame()
    return df.describe().loc[["mean"]].round(1)


# --------------------------
# 🔹 TEAM DEFENSIVE METRICS
# --------------------------

@st.cache_data(ttl=3600)
def get_team_defensive_metrics(season: str = "2024-25"):
    """
    Derive opponent defensive metrics from team game logs using nba_api.
    Opponent averages = average points, rebounds, assists allowed per game.
    """
    try:
        all_teams = teams.get_teams()
        team_defense = []

        for t in all_teams:
            team_id = t["id"]
            name = t["full_name"]

            # Avoid rate limit: nba_api can throttle
            time.sleep(0.5)

            gamelog = teamgamelog.TeamGameLog(team_id=team_id, season=season)
            df = gamelog.get_data_frames()[0]
            df["PTS_ALLOWED"] = df["PTS"].shift(-1)  # Simplistic: next opponent
            team_defense.append({
                "Team": name,
                "Opp_PPG": df["PTS_ALLOWED"].mean(),
                "Opp_RPG": df["REB"].mean(),
                "Opp_APG": df["AST"].mean(),
            })

        defense_df = pd.DataFrame(team_defense)
        return defense_df.round(1)

    except Exception as e:
        st.error(f"Error fetching team defensive metrics: {e}")
        return pd.DataFrame(columns=["Team", "Opp_PPG", "Opp_RPG", "Opp_APG"])


# --------------------------
# 🔹 TEAM + PLAYER CONTEXT
# --------------------------

@st.cache_data(ttl=300)
def get_player_context(player_name: str, opponent_team: str, season: str = "2024-25"):
    """Combine player recent stats and opponent defensive averages."""
    try:
        player_id = get_player_id(player_name)
        gamelog_df = get_player_gamelog(player_id, season)
        team_def_df = get_team_defensive_metrics(season)

        # Match opponent (fuzzy)
        opp = get_close_matches(opponent_team, team_def_df["Team"].tolist(), n=1, cutoff=0.6)
        opponent_df = team_def_df[team_def_df["Team"] == opp[0]] if opp else pd.DataFrame()

        context = {
            "player": player_name,
            "season_avg": get_season_averages(player_id, season).to_dict() if not gamelog_df.empty else {},
            "recent_games": gamelog_df.tail(10) if not gamelog_df.empty else pd.DataFrame(),
            "opponent_defense": opponent_df.to_dict(orient="records")[0] if not opponent_df.empty else {},
        }

        return context

    except Exception as e:
        st.error(f"Error building player context: {e}")
        return {"player": player_name, "season_avg": {}, "recent_games": pd.DataFrame(), "opponent_defense": {}}
