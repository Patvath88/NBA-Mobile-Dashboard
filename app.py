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
# 🌌 FUTURECOURT NBA AI CONFIG
# ======================================================
st.set_page_config(
    page_title="FutureCourt NBA AI",
    page_icon="🏀",
    layout="wide",
)

# ======================================================
# 🎨 CUSTOM STYLE
# ======================================================
st.markdown("""
<style>
body {
    background: radial-gradient(circle at 25% 25%, #0b0e23, #02030a);
    color: #E0E0E0;
}
h1, h2, h3, h4 {
    color: #00E0FF !important;
}
[data-testid="stMetricValue"] {
    font-size: 36px !important;
    font-weight: 800;
    color: #00E0FF !important;
}
.metric-card {
    background: linear-gradient(145deg, rgba(0,255,255,0.08), rgba(0,0,0,0.3));
    border: 1px solid rgba(0,255,255,0.25);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    transition: all 0.3s ease;
    backdrop-filter: blur(20px);
    box-shadow: 0 0 20px rgba(0,255,255,0.1);
}
.metric-card:hover {
    transform: scale(1.03);
    box-shadow: 0 0 25px rgba(0,255,255,0.4);
}
@keyframes shimmer {
  0% {background-position: -1000px 0;}
  100% {background-position: 1000px 0;}
}
.loading-shimmer {
  width: 100%;
  height: 16px;
  border-radius: 8px;
  background: linear-gradient(to right, #0d0f22 4%, #1f2335 25%, #0d0f22 36%);
  background-size: 1000px 100%;
  animation: shimmer 1.8s infinite linear;
  margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# 🏀 FUTURECOURT MAIN DASHBOARD
# ======================================================
st.title("🏀 FutureCourt NBA AI")
st.caption("Futuristic AI-powered NBA player performance predictor — built with GPT-5 & NBA Stats API")

player_name = st.text_input("Enter Player Name", "Luka Doncic")

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
            <div style="font-size:20px; line-height:1.6;">
            <b>Team:</b> {player_context.get("TEAM_NAME")}<br>
            <b>Position:</b> {player_context.get("POSITION")}<br>
            <b>Height / Weight:</b> {player_context.get("HEIGHT")} / {player_context.get("WEIGHT")} lbs<br>
            <b>Experience:</b> {player_context.get("SEASON_EXP")} years
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr style='border: 1px solid rgba(0,255,255,0.2);'>", unsafe_allow_html=True)

        # ======================================================
        # ⚙️ BUILD MODEL & MAKE PREDICTIONS
        # ======================================================
        with st.spinner("Training model and generating predictions..."):
            st.markdown('<div class="loading-shimmer"></div>' * 4, unsafe_allow_html=True)
            df = build_feature_dataset(player_id)
            df = sanitize_dataframe_for_streamlit(df)

            if not df.empty:
                metrics = train_xgboost_models(df)
                preds = predict_next_game(df, metrics)

                if preds:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.subheader("🔮 Predicted Next Game Stats")

                    # Team logo (if available)
                    team_name = player_context.get("TEAM_NAME", "")
                    logo_url = f"https://cdn.ssref.net/req/202301041/tlogo/bbr/{team_name.lower().replace(' ', '')}.png" if team_name else ""

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <img src="{logo_url}" width="40"><br>
                                <h3>PTS</h3>
                                <h2 style="color:#00E0FF;">{preds.get("PTS", 0)}</h2>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with c2:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <img src="{logo_url}" width="40"><br>
                                <h3>REB</h3>
                                <h2 style="color:#00E0FF;">{preds.get("REB", 0)}</h2>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with c3:
                        st.markdown(
                            f"""
                            <div class="metric-card">
                                <img src="{logo_url}" width="40"><br>
                                <h3>AST</h3>
                                <h2 style="color:#00E0FF;">{preds.get("AST", 0)}</h2>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # ======================================================
                    # 📈 HISTORICAL PERFORMANCE
                    # ======================================================
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.subheader("📉 Recent Performance (Last 10 Games)")
                    recent = df.tail(10)
                    fig = px.line(
                        recent,
                        x="GAME_DATE",
                        y=["PTS", "REB", "AST"],
                        markers=True,
                        title=f"{player_name} — Last 10 Games",
                    )
                    fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#00E0FF"),
                        title_x=0.3,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Prediction failed. Please retry.")
            else:
                st.error("No data available for this player.")
