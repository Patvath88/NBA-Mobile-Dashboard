# utils/data_loader.py
import requests
import pandas as pd
import streamlit as st
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players, teams
from datetime import datetime
from difflib import get_close_matches
from pathlib import Path

# --------------------------
# 🔹 PLAYER DATA FETCHING
# --------------------------

@st.cache_data(ttl=3600)
def get_player_id(player_name: str):
    """Return the NBA API player ID with fuzzy matching for similar names."""
    all_players = players.get_players()
    names = [p["full_name"] for p in all_players]

    # Try exact match first
    exact = next((p for p in all_players if p["full_name"].lower() == player_name.lower()), None)
    if exact:
        return exact["id"]

    # Fuzzy match (handles accents and slight typos)
    close = get_close_matches(player_name, names, n=1, cutoff=0.6)
    if close:
        match = next(p for p in all_players if p["full_name"] == close[0])
        st.info(f"Using closest match: **{match['full_name']}**")
        return match["id"]

    raise ValueError(f"Player '{player_name}' not found in NBA API.")


@st.cache_data(ttl=600)
def get_player_gamelog(player_id: int, season: str = "2025", last_n: int = 20):
    """Fetch the last N games for a specific player."""
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
def get_season_averages(player_id: int, season: str = "2025"):
    """Compute player's current season averages."""
    df = get_player_gamelog(player_id, season, last_n=82)
    if df.empty:
        return pd.DataFrame()
    return df.describe().loc[["mean"]].round(1)


# --------------------------
# 🔹 TEAM + OPPONENT DATA
# --------------------------

FALLBACK_PATH = Path("data/fallback_defense.csv")
FALLBACK_PATH.parent.mkdir(parents=True, exist_ok=True)

@st.cache_data(ttl=3600)
def get_team_defensive_metrics(season: str = "2025"):
    """
    Safely pull team defensive metrics using balldontlie.io.
    Handles API downtime and returns fallback if needed.
    """
    url = f"https://www.balldontlie.io/api/v1/stats?season={season}"

    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()

        try:
            data = res.json()
        except Exception:
            st.warning("⚠️ API returned non-JSON response. Using fallback defensive metrics.")
            if FALLBACK_PATH.exists():
                return pd.read_csv(FALLBACK_PATH)
            return pd.DataFrame(columns=["team.full_name", "Opp_PPG", "Opp_RPG", "Opp_APG"])

        if "data" not in data or len(data["data"]) == 0:
            st.warning(f"No defensive data available for season {season}.")
            return pd.DataFrame(columns=["team.full_name", "Opp_PPG", "Opp_RPG", "Opp_APG"])

        df = pd.json_normalize(data["data"])
        defense_df = (
            df.groupby("team.full_name")
              .agg({"pts": "mean", "reb": "mean", "ast": "mean"})
              .rename(columns={"pts": "Opp_PPG", "reb": "Opp_RPG", "ast": "Opp_APG"})
              .reset_index()
        )
        defense_df = defense_df.round(1)

        # Cache fallback locally
        defense_df.to_csv(FALLBACK_PATH, index=False)
        return defense_df

    except Exception as e:
        st.error(f"Error fetching defensive metrics: {e}")
        if FALLBACK_PATH.exists():
            st.info("Using locally cached defensive metrics.")
            return pd.read_csv(FALLBACK_PATH)
        return pd.DataFrame(columns=["team.full_name", "Opp_PPG", "Opp_RPG", "Opp_APG"])


@st.cache_data(ttl=3600)
def get_team_id(team_name: str):
    """Return the NBA team ID with fuzzy match fallback."""
    all_teams = teams.get_teams()
    team = next((t for t in all_teams if team_name.lower() in t["full_name"].lower()), None)
    if team:
        return team["id"]

    # Fuzzy match fallback
    names = [t["full_name"] for t in all_teams]
    close = get_close_matches(team_name, names, n=1, cutoff=0.6)
    if close:
        st.info(f"Using closest team match: **{close[0]}**")
        match = next(t for t in all_teams if t["full_name"] == close[0])
        return match["id"]

    raise ValueError(f"Team '{team_name}' not found.")


# --------------------------
# 🔹 MERGED CONTEXT DATA
# --------------------------

@st.cache_data(ttl=300)
def get_player_context(player_name: str, opponent_team: str, season: str = "2025"):
    """Combine player recent stats and opponent defense into one contextual dataset."""
    try:
        player_id = get_player_id(player_name)
        gamelog_df = get_player_gamelog(player_id, season)
        team_def_df = get_team_defensive_metrics(season)

        if gamelog_df.empty:
            st.warning(f"No game log data available for {player_name}")
        if team_def_df.empty:
            st.warning(f"No team defensive data available for {opponent_team}")

        opponent_df = team_def_df[
            team_def_df["team.full_name"].str.contains(opponent_team, case=False, na=False)
        ]

        context = {
            "player": player_name,
            "season_avg": get_season_averages(player_id, season).to_dict() if not gamelog_df.empty else {},
            "recent_games": gamelog_df.tail(10) if not gamelog_df.empty else pd.DataFrame(),
            "opponent_defense": opponent_df.to_dict(orient="records")[0]
            if not opponent_df.empty else {}
        }

        return context

    except Exception as e:
        st.error(f"Context fetch failed: {e}")
        return {"player": player_name, "season_avg": {}, "recent_games": pd.DataFrame(), "opponent_defense": {}}
