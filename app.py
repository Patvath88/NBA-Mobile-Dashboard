# =============================================
# 🏀 FUTURECOURT NBA AI DASHBOARD — Elite Build
# =============================================

import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_loader import get_player_context, get_player_id
from utils.feature_engineer import build_feature_dataset
from utils.model_utils import train_xgboost_models, predict_next_game
from utils.helpers import sanitize_dataframe_for_streamlit
from nba_api.stats.static import teams
import time

# ======================================================
# 🌌 FUTURECOURT THEME & PAGE SETTINGS
# ======================================================
st.set_page_config(
    page_title="FutureCourt NBA AI",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500&display=swap');

        .stApp {
            background: radial-gradient(circle at top left, #091020, #000000 85%);
            color: #FFFFFF;
            font-family: 'Orbitron', sans-serif;
        }
        .metric-card {
            background: rgba(255,255,255,0.08);
            border-radius: 20px;
            padding: 18px;
            text-align: center;
            box-shadow: 0 0 30px rgba(0,255,255,0.25);
            backdrop-filter: blur(8px);
            transition: transform 0.2s ease-in-out;
        }
        .metric-card:hover {
            transform: scale(1.05);
        }
        .fade-in {
            animation: fadeIn 1s ease-in-out;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        h1, h2, h3 {
            color: #00FFFF !important;
            text-shadow: 0 0 12px #00FFFF;
        }
        .block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

# ======================================================
# 🚀 HEADER
# ======================================================
st.markdown("<h1 class='fade-in'>🏀 FutureCourt NBA AI Dashboard</h1>", unsafe_allow_html=True)
st.caption("AI-Powered NBA Player Performance Predictor — built with XGBoost + NBA API")

# ======================================================
# 🎯 PLAYER INPUT
# ======================================================
player = st.text_input("Enter Player Name", value="Luka Doncic")

if st.button("🚀 Generate Prediction"):
    with st.spinner("Fetching player data and generating predictions..."):
        try:
            player_id = get_player_id(player)
            if not player_id:
                st.error(f"Player '{player}' not found.")
            else:
                context = get_player_context(player_id)
                if not context or "recent_games" not in context:
                    st.warning("Unable to fetch recent games.")
                else:
                    df = context["recent_games"]
                    df = build_feature_dataset(df)
                    df = sanitize_dataframe_for_streamlit(df)

                    models = train_xgboost_models(df)
                    preds = predict_next_game(models, df)

                    # ==============================================
                    # 📸 Fetch Player & Team Images
                    # ==============================================
                    headshot_url = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"
                    team_name = df.iloc[-1]["TEAM_ABBREVIATION"] if "TEAM_ABBREVIATION" in df.columns else None

                    team_logo_url = None
                    if team_name:
                        nba_teams = teams.get_teams()
                        team_info = next((t for t in nba_teams if t["abbreviation"] == team_name), None)
                        if team_info:
                            team_logo_url = f"https://cdn.nba.com/logos/nba/{team_info['id']}/primary/L/logo.svg"

                    col_logo, col_stats = st.columns([1, 3])
                    with col_logo:
                        if team_logo_url:
                            st.image(team_logo_url, width=100)
                        st.image(headshot_url, width=240, caption=player, use_container_width=False)

                    with col_stats:
                        st.markdown("### 🔮 Predicted Next Game Stats")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.markdown(f"<div class='metric-card fade-in'><h3>PTS</h3><h2>{preds['PTS']:.1f}</h2></div>", unsafe_allow_html=True)
                        col2.markdown(f"<div class='metric-card fade-in'><h3>REB</h3><h2>{preds['REB']:.1f}</h2></div>", unsafe_allow_html=True)
                        col3.markdown(f"<div class='metric-card fade-in'><h3>AST</h3><h2>{preds['AST']:.1f}</h2></div>", unsafe_allow_html=True)
                        col4.markdown(f"<div class='metric-card fade-in'><h3>PRA</h3><h2>{preds['PRA']:.1f}</h2></div>", unsafe_allow_html=True)

                    # ==============================================
                    # 📈 Performance Trend Chart
                    # ==============================================
                    st.markdown("### 📊 Recent Performance Trend")
                    chart = px.line(
                        df.tail(10),
                        x="GAME_DATE",
                        y=["PTS", "REB", "AST", "PRA"],
                        title=f"{player}'s Recent Game Stats",
                        markers=True
                    )
                    chart.update_layout(
                        template="plotly_dark",
                        title_font=dict(size=20, color="#00FFFF"),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="white"),
                        legend=dict(bgcolor="rgba(0,0,0,0.2)")
                    )
                    st.plotly_chart(chart, use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")

# ======================================================
# 🌙 FOOTER
# ======================================================
st.markdown("""
<hr style='border:1px solid #00FFFF'>
<div style='text-align:center; color:gray;'>
🚀 FutureCourt AI © 2025 — Built by Patvath88 | Powered by Streamlit, XGBoost & NBA API
</div>
""", unsafe_allow_html=True)
