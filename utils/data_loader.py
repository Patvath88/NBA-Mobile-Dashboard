# utils/data_loader.py
import pandas as pd
import streamlit as st
from nba_api.stats.endpoints import playergamelog, teamgamelog
from nba_api.stats.static import players, teams
from difflib import get_close_matches
from datetime import datetime
import time

# --------------------------
# 🔹 PLAYER FUNCTIONS
# --------------------------

@st.cache_data(ttl=3600)
def get_player_id(player_name: str):
    """Return the NBA API player ID with fuzzy matching."""
    all_players = players.get_players()
    names = [p["full_name"] for p in all_players]

    # Exact match first
    exact = next((p for p in all_players if p["full_name"].lower() == player_name.lower()), None)
    if exact:
        return exact["id"]

    # Fuzzy match
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
# 🔹 TEAM METRICS (NBA API)
# --------------------------

@st.cache_data(ttl=3600)
def get_team_defensive_metrics(season: str = "2024-25"):
    """
    Build team defensive and pace metrics using NBA team game logs.
    Derives Opp_PPG, Opp_RPG, Opp_APG, Pace, and DefRtg.
    """
    all_teams = teams.get_teams()
    team_stats = []

    for t in all_teams:
        team_id = t["id"]
        team_name = t["full_name"]

        # Avoid hitting NBA API rate limits
        time.sleep(0.4)

        try:
            logs = teamgamelog.TeamGameLog(team_id=team_id, season=season)
            df = logs.get_data_frames()[0]

            if df.empty:
                continue

            # Convert numeric columns
            df["PTS"] = pd.to_numeric(df["PTS"], errors="coerce")
            df["REB"] = pd.to_numeric(df["REB"], errors="coerce")
            df["AST"] = pd.to_numeric(df["AST"], errors="coerce")
            df["TOV"] = pd.to_numeric(df["TOV"], errors="coerce")
            df["FGA"] = pd.to_numeric(df["FGA"], errors="coerce")
            df["FTA"] = pd.to_numeric(df["FTA"], errors="coerce")

            # Estimate possessions (pace formula)
            df["Possessions"] = df["FGA"] + 0.44 * df["FTA"] - df["OREB"] + df["TOV"]

            opp_ppg = df["PTS"].mean()
            opp_rpg = df["REB"].mean()
            opp_apg = df["AST"].mean()
            pace = df["Possessions"].mean()
            def_rtg = (df["PTS"].mean() / (df["Possessions"].mean() / 100)) if df["Possessions"].mean() > 0 else None

            team_stats.append({
                "Team": team_name,
                "Opp_PPG": round(opp_ppg, 1),
                "Opp_RPG": round(opp_rpg, 1),
                "Opp_APG": round(opp_apg, 1),
                "Pace": round(pace, 1),
                "DefRtg": round(def_rtg, 1) if def_rtg else None
            })

        except Exception as e:
            st.warning(f"Could not fetch {team_name} metrics: {e}")

    df_defense = pd.DataFrame(team_stats)
    return df_defense


# --------------------------
# 🔹 CONTEXT BUILDER
# --------------------------

@st.cache_data(ttl=300)
def get_player_context(player_name: str, opponent_team: str, season: str = "2024-25"):
    """Combine player recent stats and opponent team context with pace and defensive rating."""
    try:
        player_id = get_player_id(player_name)
        gamelog_df = get_player_gamelog(player_id, season)
        team_metrics_df = get_team_defensive_metrics(season)

        # Safety: handle missing or malformed team_metrics_df
        if team_metrics_df is None or team_metrics_df.empty:
            st.warning("Team defensive metrics unavailable. Using empty opponent context.")
            opponent_df = pd.DataFrame()
        elif "Team" not in team_metrics_df.columns:
            st.warning("Unexpected defensive metrics format (no 'Team' column).")
            opponent_df = pd.DataFrame()
        else:
            # Fuzzy match opponent name
            from difflib import get_close_matches
            team_names = team_metrics_df["Team"].tolist()
            opp_match = get_close_matches(opponent_team, team_names, n=1, cutoff=0.5)
            opponent_df = (
                team_metrics_df[team_metrics_df["Team"] == opp_match[0]]
                if opp_match
                else pd.DataFrame()
            )

        context = {
            "player": player_name,
            "season_avg": (
                get_season_averages(player_id, season).to_dict()
                if not gamelog_df.empty
                else {}
            ),
            "recent_games": gamelog_df.tail(10) if not gamelog_df.empty else pd.DataFrame(),
            "opponent_metrics": opponent_df.to_dict(orient="records")[0]
            if not opponent_df.empty
            else {},
        }

        return context

    except Exception as e:
        st.error(f"Error building player context: {e}")
        return {
            "player": player_name,
            "season_avg": {},
            "recent_games": pd.DataFrame(),
            "opponent_metrics": {},
        }
