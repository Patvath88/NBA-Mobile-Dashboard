# =============================================
# 🏀 FUTURECOURT NBA AI DASHBOARD — Final Build
# =============================================

import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_loader import get_player_context, get_player_id
from utils.feature_engineer import build_feature_dataset
from utils.model_utils import train_xgboost_models, predict_next_game
from utils.helpers import sanitize_dataframe_for_streamlit
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
        /* Futuristic Neon Theme */
        .stApp {
            background: radial-gradient(circle at top left, #0a0f1c, #000000 80%);
            color: #FFFFFF;
            font-family: 'Orbitron', sans-serif;
        }
        .stMetric {
            background: rgba(255,255,255,0.06);
            border-radius: 16px;
            padding: 18px;
            text-align: center;
            box-shadow: 0 0 20px rgba(0,255,255,0.2);
            backdrop-filter: blur(6px);
        }
        div[data-testid="stHeader"] {background: none;}
        h1, h2, h3 {color: #00FFFF !important; text-shadow: 0 0 10px #00FFFF;}
        .block-container {padding-top: 2rem;}
        .css-1v3fvcr {background: none;}
    </style>
""", unsafe_allow_html=True)

# ======================================================
# 🚀 HEADER
# ======================================================
st.title("🏀 FutureCourt NBA AI Dashboard")
st.caption("AI-Powered NBA Player Performance Predictor — built with XGBoost + NBA API")

# ======================================================
# 🎯 PLAYER INPUT
# ======================================================
player = st.text_input("Enter Player Name", value="Luka Doncic")

if st.button("🚀 Generate Prediction"):
    with st.spinner("Fetching player data and training models..."):
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

                    # Display metrics with glow animation
                    st.markdown("### 🔮 Predicted Next Game Stats")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("PTS", f"{preds['PTS']:.1f}")
                    col2.metric("REB", f"{preds['REB']:.1f}")
                    col3.metric("AST", f"{preds['AST']:.1f}")
                    col4.metric("PRA", f"{preds['PRA']:.1f}")

                    # Trend visualization
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
🚀 FutureCourt AI © 2025 — Powered by Streamlit, XGBoost, and the NBA API
</div>
""", unsafe_allow_html=True)
