# ======================================================
# 🏀 FUTURECOURT NBA AI — Futuristic Animated Dashboard
# ======================================================

import streamlit as st  # ✅ Import first
import pandas as pd
import plotly.express as px

# Import utilities
from utils.data_loader import get_player_context, get_player_id
from utils.feature_engineer import build_feature_dataset
from utils.model_utils import train_xgboost_models, predict_next_game
from utils.helpers import sanitize_dataframe_for_streamlit

# UI helpers
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_lottie import st_lottie
import requests
import warnings

warnings.filterwarnings("ignore")

# ======================================================
# 🌌 FUTURECOURT DASHBOARD CONFIG
# ======================================================

st.set_page_config(
    page_title="FutureCourt NBA AI",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom futuristic styling
st.markdown("""
<style>
body {
    background: radial-gradient(circle at 10% 20%, #0a0f1a 0%, #09121b 100%);
    color: white;
    font-family: 'Segoe UI', sans-serif;
}
.stApp {
    background: transparent;
}
h1, h2, h3, h4 {
    color: #00FFFF;
}
div[data-testid="stMetricValue"] {
    font-size: 2rem;
    font-weight: bold;
    color: #00FFFF;
}
.metric-card {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(0,255,255,0.2);
    border-radius: 15px;
    padding: 1rem;
    box-shadow: 0 0 20px rgba(0,255,255,0.2);
    transition: all 0.3s ease-in-out;
    text-align: center;
}
.metric-card:hover {
    transform: scale(1.03);
    box-shadow: 0 0 40px rgba(0,255,255,0.4);
}
@keyframes pulse {
    0% { box-shadow: 0 0 10px rgba(0,255,255,0.2); }
    50% { box-shadow: 0 0 25px rgba(0,255,255,0.6); }
    100% { box-shadow: 0 0 10px rgba(0,255,255,0.2); }
}
.metric-animate {
    animation: pulse 2s infinite;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# 🧠 HEADER
# ======================================================

col1, col2 = st.columns([0.7, 0.3])
with col1:
    st.title("🏀 FutureCourt NBA AI")
    st.markdown("#### Predict player performance with futuristic AI precision.")
with col2:
    try:
        lottie_url = "https://assets4.lottiefiles.com/packages/lf20_x62chJ.json"
        lottie_json = requests.get(lottie_url).json()
        st_lottie(lottie_json, height=150, key="nba_lottie")
    except Exception:
        pass

st.divider()

# ======================================================
# 🎯 PLAYER INPUT SECTION
# ======================================================

player_name = st.text_input("Enter player name", placeholder="e.g. Luka Doncic")

if st.button("🚀 Generate Prediction"):
    if not player_name:
        st.warning("Please enter a valid player name.")
    else:
        with st.spinner("🤖 Crunching numbers and running AI models..."):
            try:
                player_id = get_player_id(player_name)
                context = get_player_context(player_name)

                if not context:
                    st.error("Unable to fetch player data. Try another name.")
                else:
                    df = build_feature_dataset(player_id)
                    if df is None or df.empty:
                        st.error("No sufficient data found for this player.")
                    else:
                        results = train_xgboost_models(df)
                        preds = predict_next_game(df)

                        # ======================================================
                        # ⚡️ Animated Metric Cards
                        # ======================================================
                        st.subheader(f"✨ Predicted Next Game — {player_name}")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(
                                f"""
                                <div class="metric-card metric-animate">
                                    <h3>Points</h3>
                                    <p style="font-size:2rem;">{preds.get('PTS', 0):.1f}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        with col2:
                            st.markdown(
                                f"""
                                <div class="metric-card metric-animate">
                                    <h3>Rebounds</h3>
                                    <p style="font-size:2rem;">{preds.get('REB', 0):.1f}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        with col3:
                            st.markdown(
                                f"""
                                <div class="metric-card metric-animate">
                                    <h3>Assists</h3>
                                    <p style="font-size:2rem;">{preds.get('AST', 0):.1f}</p>
                                </div>
                                """, unsafe_allow_html=True)

                        st.divider()

                        # ======================================================
                        # 🧩 Model Metrics
                        # ======================================================
                        st.markdown("### 🧮 Model Performance Metrics")
                        metrics_df = sanitize_dataframe_for_streamlit(pd.DataFrame(results).T)
                        st.dataframe(metrics_df, width="stretch")

                        # ======================================================
                        # 📊 Visual Trends
                        # ======================================================
                        st.markdown("### 📈 Last 10 Games Performance")
                        df_vis = sanitize_dataframe_for_streamlit(df.tail(10))
                        fig = px.line(
                            df_vis,
                            x="GAME_DATE",
                            y=["PTS", "REB", "AST"],
                            markers=True,
                            title=f"{player_name} — Last 10 Games Trend",
                            template="plotly_dark"
                        )
                        st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error during prediction: {e}")

else:
    st.info("💡 Enter a player name and click **Generate Prediction** to begin.")
