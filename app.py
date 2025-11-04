import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_loader import get_player_context, get_player_id
from utils.feature_engineer import build_feature_dataset
from utils.model_utils import train_xgboost_models, predict_next_game
from utils.helpers import sanitize_dataframe_for_streamlit

st.set_page_config(page_title="FutureCourt NBA AI", page_icon="🏀", layout="wide")

# ======== STYLE & THEME ========
st.markdown("""
    <style>
    body {background-color: #0E1117; color: #FFFFFF;}
    .stApp {background: radial-gradient(circle at top left, #111827, #0E1117);}
    h1, h2, h3 {color: #FFFFFF;}
    .metric-card {
        background: rgba(255,255,255,0.07);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 0 12px rgba(0, 255, 255, 0.1);
        transition: 0.3s;
    }
    .metric-card:hover {
        transform: scale(1.05);
        box-shadow: 0 0 18px rgba(0, 255, 255, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# ======== TITLE ========
st.markdown("<h1 style='text-align:center;'>🤖 AI Model Predictions</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#9CA3AF;'>Predict next game performance using advanced XGBoost models</p>", unsafe_allow_html=True)
st.divider()

# ======== INPUT ========
player_name = st.text_input("Enter player name", placeholder="e.g. Luka Doncic", value="Luka Doncic")

if st.button("Run Predictions"):
    with st.spinner("Training model and generating predictions..."):
        try:
            # === Player ID + context ===
            player_id = get_player_id(player_name)
            if not player_id:
                st.error("Player not found.")
                st.stop()

            context = get_player_context(player_id)
            if context is None or "recent_games" not in context:
                st.warning("No recent game data found.")
                st.stop()

            df = context["recent_games"]
            df = sanitize_dataframe_for_streamlit(df)

            # === Build features & train ===
            dataset = build_feature_dataset(df)
            models = train_xgboost_models(dataset)

            # === Predict ===
            pred = predict_next_game(models, dataset)

            st.success(f"✅ Predictions generated for **{player_name}**")

            # === METRICS ===
            col1, col2, col3, col4 = st.columns(4)
            col1.markdown(f"<div class='metric-card'><h3>Points</h3><h2>{pred['PTS']:.1f}</h2></div>", unsafe_allow_html=True)
            col2.markdown(f"<div class='metric-card'><h3>Rebounds</h3><h2>{pred['REB']:.1f}</h2></div>", unsafe_allow_html=True)
            col3.markdown(f"<div class='metric-card'><h3>Assists</h3><h2>{pred['AST']:.1f}</h2></div>", unsafe_allow_html=True)
            col4.markdown(f"<div class='metric-card'><h3>PRA</h3><h2>{pred['PRA']:.1f}</h2></div>", unsafe_allow_html=True)

            st.divider()

            # === TREND CHART ===
            st.subheader("📊 Last 10 Games Trend")
            fig = px.line(df.tail(10), x="GAME_DATE", y=["PTS", "REB", "AST", "PRA"], markers=True)
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#FFFFFF",
                xaxis_title="Game Date",
                yaxis_title="Stat Value"
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")
