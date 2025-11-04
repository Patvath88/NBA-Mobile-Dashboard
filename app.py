import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_loader import get_player_context, get_player_id
from utils.feature_engineer import build_feature_dataset
from utils.model_utils import train_xgboost_models, predict_next_game
from utils.helpers import sanitize_dataframe_for_streamlit
from streamlit_extras.metric_cards import style_metric_cards
import requests
import warnings
warnings.filterwarnings("ignore")

# ======================================================
# 🌌 FUTURECOURT NBA AI DASHBOARD CONFIG
# ======================================================
st.set_page_config(
    page_title="FutureCourt NBA AI",
    page_icon="🏀",
    layout="wide",
)

# ======================================================
# 🎨 STYLES
# ======================================================
st.markdown("""
<style>
body {
    background: radial-gradient(circle at 20% 20%, #0d0f22, #05060e);
    color: #E0E0E0;
}
[data-testid="stMetricValue"] {
    font-size: 34px !important;
    font-weight: 700;
    color: #00E0FF !important;
}
.metric-card {
    background: rgba(255,255,255,0.05);
    border-radius: 16px;
    padding: 20px;
    backdrop-filter: blur(15px);
    transition: all 0.3s ease;
}
.metric-card:hover {
    transform: scale(1.03);
    box-shadow: 0 0 20px rgba(0,255,255,0.3);
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# 🔍 PLAYER INPUT
# ======================================================
st.title("🏀 FutureCourt NBA AI")
st.caption("AI-powered NBA player performance predictor — powered by GPT-5 + NBA API")

player_name = st.text_input("Enter Player Name", "LeBron James")

if player_name:
    with st.spinner("Fetching player data..."):
        player_context = get_player_context(player_name)
        player_id = get_player_id(player_name)

    if player_context and player_id:
        st.subheader(f"📊 {player_context.get('DISPLAY_FIRST_LAST', player_name)} — {player_context.get('TEAM_NAME', '')}")

        headshot_url = f"https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"

        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(headshot_url, caption=player_name, use_container_width=True)
        with col2:
            st.markdown(f"""
            <div style="font-size:22px; line-height:1.6;">
            <b>Team:</b> {player_context.get("TEAM_NAME")}<br>
            <b>Position:</b> {player_context.get("POSITION")}<br>
            <b>Height / Weight:</b> {player_context.get("HEIGHT")} / {player_context.get("WEIGHT")} lbs<br>
            <b>Experience:</b> {player_context.get("SEASON_EXP")} years
            </div>
            """, unsafe_allow_html=True)

        # ======================================================
        # ⚙️ BUILD FEATURES & TRAIN MODEL
        # ======================================================
        with st.spinner("Building AI model..."):
            df = build_feature_dataset(player_id)
            df = sanitize_dataframe_for_streamlit(df)

            if not df.empty:
                metrics = train_xgboost_models(df)
                preds = predict_next_game(df, metrics)

                if preds:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.subheader("🔮 Predicted Next Game Stats")

                    c1, c2, c3 = st.columns(3)
                    c1.metric("PTS", preds.get("PTS", 0))
                    c2.metric("REB", preds.get("REB", 0))
                    c3.metric("AST", preds.get("AST", 0))
                    style_metric_cards(border_left_color="#00FFFF", border_color="#00FFFF")

                    # ======================================================
                    # 📈 HISTORICAL TREND CHART
                    # ======================================================
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.subheader("📉 Recent Performance (Last 10 Games)")
                    recent = df.tail(10)
                    fig = px.line(
                        recent,
                        x="GAME_DATE",
                        y=["PTS", "REB", "AST"],
                        markers=True,
                        title=f"{player_name} - Last 10 Games",
                    )
                    fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#00E0FF")
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Prediction failed. Please retry.")
            else:
                st.error("No data available for this player.")
