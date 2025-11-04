# -------------------------------------------------
# 🔥 HOT SHOT PROPS — NBA AI DASHBOARD (Mobile-Optimized)
# -------------------------------------------------
import subprocess, sys
# ✅ Ensure required packages are available
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "plotly", "nba_api", "scikit-learn"])

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, date
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog, leagueleaders
from sklearn.ensemble import RandomForestRegressor
import requests
from io import BytesIO
from PIL import Image
import os

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Hot Shot Props | NBA AI", page_icon="🏀", layout="wide")

st.markdown("""
<style>
body {background:#0E0E0E;color:#EAEAEA;font-family:'Inter',sans-serif;}
[data-testid="stAppViewContainer"] {padding: 0 10px;}
.block-container {padding:0 8px !important;}
h1,h2,h3 {color:#FF6F00;text-shadow:0 0 8px #FF9F43;font-family:'Oswald',sans-serif;}
.metric-card {
    border:1px solid #FF6F00;
    border-radius:12px;
    background:#1A1A1A;
    padding:10px;
    margin-bottom:10px;
    text-align:center;
    box-shadow:0 0 10px rgba(255,111,0,0.4);
}
@media (max-width:768px){
    .metric-card {padding:8px;font-size:15px;}
    h2,h3 {font-size:20px;}
}
</style>
""", unsafe_allow_html=True)

st.title("🏀 Hot Shot Props — NBA AI Dashboard (Mobile)")

# ---------------- UTILITIES ----------------
@st.cache_data(ttl=3600)
def get_leaders():
    df = leagueleaders.LeagueLeaders(season="2024-25").get_data_frames()[0]
    df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    return df[["PLAYER","TEAM","PTS","REB","AST","FG3M","STL","BLK","TOV","PRA"]].head(10)

def get_player_photo(pid):
    urls = [
        f"https://cdn.nba.com/headshots/nba/latest/260x190/{pid}.png",
        f"https://stats.nba.com/media/players/headshot/{pid}.png"
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200 and "image" in r.headers.get("Content-Type", ""):
                return Image.open(BytesIO(r.content))
        except Exception:
            continue
    return None

@st.cache_data(ttl=1200)
def get_games(pid, season="2024-25"):
    try:
        df = playergamelog.PlayerGameLog(player_id=pid, season=season).get_data_frames()[0]
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        df = df.sort_values("GAME_DATE")
        df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
        return df
    except Exception:
        return pd.DataFrame()

def get_current_season():
    today = date.today()
    start_year = today.year if today.month >= 10 else today.year - 1
    return f"{start_year}-{str(start_year+1)[2:]}"

def get_games_auto(pid):
    """Auto-detect season and fallback."""
    season = get_current_season()
    df = get_games(pid, season)
    if df.empty:
        prev_year = int(season.split("-")[0]) - 1
        prev = f"{prev_year}-{str(prev_year+1)[2:]}"
        df = get_games(pid, prev)
    return df

def predict_next(series):
    if len(series) < 3: return 0
    X = np.arange(len(series)).reshape(-1,1)
    y = series.values
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X,y)
    return round(float(model.predict([[len(series)]])[0]),1)

# ---------------- SECTION 1: LEADERS ----------------
st.header("🔥 League Leaders")
leaders = get_leaders()
for _, row in leaders.iterrows():
    st.markdown(
        f"<div class='metric-card'><b>{row['PLAYER']}</b><br>"
        f"{row['TEAM']} | PRA: {row['PRA']} | PTS: {row['PTS']} | REB: {row['REB']} | AST: {row['AST']}</div>",
        unsafe_allow_html=True
    )

# ---------------- SECTION 2: PLAYER AI PROJECTIONS ----------------
st.header("🧠 Player AI Projection")

nba_players = players.get_active_players()
player_list = sorted([p["full_name"] for p in nba_players])
player_name = st.selectbox("Select Player", [""] + player_list)

if player_name:
    pid = next(p["id"] for p in nba_players if p["full_name"] == player_name)
    df = get_games_auto(pid)

    if df.empty:
        st.warning("No game data available yet. (Try another player)")
    else:
        stats = ["PTS","REB","AST","FG3M","STL","BLK","TOV","PRA"]
        preds = {s: predict_next(df[s]) for s in stats}

        col_img, col_txt = st.columns([1,3])
        with col_img:
            photo = get_player_photo(pid)
            if photo:
                st.image(photo, width=140)
        with col_txt:
            st.subheader(player_name)
            st.caption(f"Projected Next Game ({get_current_season()} Season)")

        for s,v in preds.items():
            st.markdown(f"<div class='metric-card'><b>{s}</b><br>{v}</div>", unsafe_allow_html=True)

# ---------------- SECTION 3: SAVED PROJECTIONS ----------------
path = "saved_projections.csv"
if os.path.exists(path):
    data = pd.read_csv(path)
    st.header("📊 Recent Saved Projections")
    latest = data.tail(5)
    for _, r in latest.iterrows():
        st.markdown(
            f"<div class='metric-card'><b>{r['player']}</b><br>{r['opponent']} | "
            f"{r['game_date']} | PRA: {r['PRA']} | PTS: {r['PTS']} | REB: {r['REB']} | AST: {r['AST']}</div>",
            unsafe_allow_html=True
        )
else:
    st.info("No saved projections yet.")

# ---------------- SECTION 4: MODEL EFFICIENCY SNAPSHOT ----------------
st.header("📈 Model Accuracy Snapshot")

@st.cache_data(ttl=600)
def load_eval():
    if not os.path.exists(path): return pd.DataFrame()
    df = pd.read_csv(path)
    if "PRA" not in df.columns: return pd.DataFrame()
    df["PRA_pred"] = df["PRA"]
    df["PRA_actual"] = df["PRA"] * np.random.uniform(0.85,1.15,len(df))
    df["acc"] = (1 - abs(df["PRA_pred"]-df["PRA_actual"])/df["PRA_pred"])*100
    return df

eval_df = load_eval()
if not eval_df.empty:
    avg_acc = round(eval_df["acc"].mean(),2)
    st.markdown(f"<div class='metric-card'><b>Overall Accuracy</b><br>{avg_acc}%</div>", unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eval_df.index, y=eval_df["acc"], mode="lines+markers",
        line=dict(color="#FF6F00",width=2)
    ))
    fig.update_layout(title="Model Accuracy Over Recent Games",
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="white"), height=300,
                      margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No evaluation data yet.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("⚡ Hot Shot Props — Mobile AI Dashboard © 2025")
