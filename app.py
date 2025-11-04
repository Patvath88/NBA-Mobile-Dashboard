import streamlit as st
import pandas as pd
import plotly.express as px
import time

from utils.data_loader import get_player_context, get_player_id
from utils.feature_engineer import build_feature_dataset
from utils.model_utils import train_xgboost_models, predict_next_game
from utils.helpers import sanitize_dataframe_for_streamlit

# ======================================================
# 🌌 FUTURECOURT DASHBOARD THEME & PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="FutureCourt NBA AI",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {
        background: radial-gradient(circle at 25% 25%, #0e0e1a, #000);
        color: #fff;
    }
    .stMetric {
        background: linear-gradient(145deg, #101020, #1e1e2f);
        border-radius: 16px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 0 10px rgba(0,255,255,0.15);
        transition: all 0.3s ease-in-out;
    }
    .stMetric:hover {
        transform: scale(1.02);
        box-shadow: 0 0 20px rgba(0,255,255,0.4);
    }
    .metric-label {
        color: #aaa;
        font-size: 0.9rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 600;
        color: #00e0ff;
    }
    h1, h2, h3, h4 {
        color: #00e0ff;
        text-shadow: 0px 0px 6px rgba(0,255,255,0.3);
    }
</style>
""", unsafe_allow_html=True)

# ======================================================
# 🧠 HEADER SECTION
# ======================================================
st.markdown("<h1 style='text-align:center;'>🏀 FutureCourt NBA AI Dashboard</h1>", unsafe_allow_html=True)
st.caption("AI-powered predictive insights for NBA players — points, rebounds, assists, and beyond.")

# ======================================================
# 🎯 PLAYER INPUT
# ======================================================
player_name = st.text_input("Enter NBA Player Name:", placeholder="e.g. Luka Doncic, Jayson Tatum")

if player_name:
    with st.spinner("Fetching player data..."):
        try:
            player_id = get_player_id(player_name)
            if not player_id:
                st.error("Player not found. Please check the name spelling.")
                st.stop()

            context = get_player_context(player_id)
            if not context or "recent_games" not in context:
                st.error("Could not load player stats.")
                st.stop()

            df = context["recent_games"]
            df = build_feature_dataset(df)
            df = sanitize_dataframe_for_streamlit(df)

            if df.empty:
                st.warning("No recent game data available for this player.")
                st.stop()

            st.success(f"Data loaded for **{player_name}** — {len(df)} games found.")

        except Exception as e:
            st.error(f"Error fetching player data: {e}")
            st.stop()

    # ======================================================
    # 📊 PERFORMANCE TREND CHART
    # ======================================================
    st.subheader(f"📈 Recent Performance — {player_name}")
    try:
        chart_df = df.tail(10)
        fig = px.line(
            chart_df,
            x="GAME_DATE",
            y=["PTS", "REB", "AST", "PRA"],
            title="Last 10 Games: Points / Rebounds / Assists / PRA",
            markers=True
        )
        fig.update_layout(
            template="plotly_dark",
            height=400,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Chart unavailable: {e}")

    # ======================================================
    # 🚀 PREDICTION SECTION
    # ======================================================
    if st.button("🚀 Train & Predict Next Game"):
        with st.spinner("Training AI models and generating predictions..."):
            try:
                models = train_xgboost_models(df)

                if not models:
                    st.warning("Not enough data to train reliable models.")
                    st.stop()

                preds = predict_next_game(models, df)

                st.success("✅ AI Predictions Ready!")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Predicted Points", f"{preds['PTS']:.1f}")
                col2.metric("Predicted Rebounds", f"{preds['REB']:.1f}")
                col3.metric("Predicted Assists", f"{preds['AST']:.1f}")
                col4.metric("Predicted PRA", f"{preds['PRA']:.1f}")

                # Data readiness indicator
                st.markdown("""
                    <div style='margin-top:20px;'>
                        <div style='background:#111; border-radius:10px; height:20px; width:100%; position:relative;'>
                            <div style='background:linear-gradient(90deg, #00e0ff, #009fff);
                                        width:{0}%; height:100%; border-radius:10px; box-shadow:0 0 10px #00e0ff;'>
                            </div>
                        </div>
                        <p style='font-size:12px; color:#aaa; margin-top:4px;'>Data readiness: {0}% based on recent games</p>
                    </div>
                """.format(min(len(df)*10, 100)), unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error during prediction: {e}")

else:
    st.info("👆 Enter a player’s name above to start exploring predictions.")
