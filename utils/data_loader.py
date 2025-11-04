# utils/data_loader.py
import requests
import pandas as pd
import streamlit as st
from nba_api.stats.endpoints import playergamelog, teamgamelog
from nba_api.stats.static import players, teams
from datetime import datetime, timedelta

# --------------------------
# 🔹 PLAYER DATA FETCHING
# --------------------------

@st.cache_data(ttl=3600)
def get_player_id(player_name: str):
    """Return the NBA API player ID given their full name."""
    all_players = players.get_players()
    player = next((p for p in all_players if p['full_name'].lower() == player_name.lower()), None)
    if player:
        return player['id']
    else:
        raise ValueError(f"Player '{player_name}' not found in NBA API.")


@st.cache_data(ttl=600)
def get_player_gamelog(player_id: int, season: str = '2024-25', last_n: int = 20):
    """Fetch last N games for the specified player."""
    gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season)
    df = gamelog.get_data_frames()[0]
    df = df.head(last_n)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df = df.sort_values('GAME_DATE', ascending=True)
    return df


@st.cache_data(ttl=600)
def get_season_averages(player_id: int, season: str = '2024-25'):
    """Compute player's current season average stats."""
    df = get_player_gamelog(player_id, season, last_n=82)
    return df.describe().loc[['mean']].round(1)


# --------------------------
# 🔹 TEAM + OPPONENT DATA
# --------------------------

@st.cache_data(ttl=3600)
def get_team_id(team_name: str):
    """Return the NBA API team ID."""
    all_teams = teams.get_teams()
    team = next((t for t in all_teams if team_name.lower() in t['full_name'].lower()), None)
    if team:
        return team['id']
    else:
        raise ValueError(f"Team '{team_name}' not found.")


@st.cache_data(ttl=900)
def get_team_defensive_metrics(season: str = '2024-25'):
    """Pull team defensive stats using balldontlie.io."""
    url = f"https://www.balldontlie.io/api/v1/stats?season={season}"
    res = requests.get(url)
    data = res.json()

    df = pd.json_normalize(data['data'])
    defense_df = (
        df.groupby('team.full_name')
        .agg({
            'pts': 'mean',
            'reb': 'mean',
            'ast': 'mean'
        })
        .rename(columns={'pts': 'Opp_PPG', 'reb': 'Opp_RPG', 'ast': 'Opp_APG'})
        .reset_index()
    )
    return defense_df.round(1)


# --------------------------
# 🔹 MERGED PLAYER CONTEXT DATA
# --------------------------

@st.cache_data(ttl=300)
def get_player_context(player_name: str, opponent_team: str, season: str = '2024-25'):
    """Combine player recent stats and opponent defense into one contextual dataset."""
    player_id = get_player_id(player_name)
    gamelog_df = get_player_gamelog(player_id, season)
    team_def_df = get_team_defensive_metrics(season)

    opponent_df = team_def_df[team_def_df['team.full_name'] == opponent_team]

    context = {
        "player": player_name,
        "season_avg": get_season_averages(player_id, season).to_dict(),
        "recent_games": gamelog_df.tail(10),
        "opponent_defense": opponent_df.to_dict(orient='records')[0]
    }
    return context
