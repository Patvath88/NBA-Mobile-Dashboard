import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_loader import get_player_id
from utils.feature_engineer import build_feature_dataset
from utils.model_utils import predict_next_game
from streamlit_lottie import st_lottie
import requests
import time

st.set_page_config(page_title="Live Tracker", page_icon="📡", layout="wide")

def load_custom_css():
    with open("assets/theme.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_custom_css()

def load_lottieurl(url):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None

st.title("📡 Live Prediction Tracker")
st.caption("Compare model predictions with real-time performance updates")
st_lottie(load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_uh1s7bsv.json"), height=140)

player = st.text_input("Track player", "Stephen Curry")

if st.button("Start Live Tracker"):
    st.info("Fetching data every 30 seconds...")
    with st.spinner("Initializing tracker..."):
        try:
            player_id = get_player_id(player)
            df = build_feature_dataset(player_id)
            preds = predict_next_game(df)

            # Simulated live update loop
            progress = st.empty()
            chart = st.empty()

            for i in range(10):
                progress.text(f"🔄 Refresh cycle {i+1}/10 — checking stats...")
                time.sleep(3)  # simulate 30s API refresh cycle (replace with live call)

                live_stats = {
                    "PTS": preds["PTS"] * (0.7 + 0.3 * i / 10),
                    "REB": preds["REB"] * (0.7 + 0.3 * i / 10),
                    "AST": preds["AST"] * (0.7 + 0.3 * i / 10),
                }

                df_live = pd.DataFrame([live_stats])
                fig = px.bar(df_live, x=df_live.columns, y=[0], title="Live vs Predicted Stats", text_auto=True)
                fig.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="rgba(10,10,10,0.6)",
                    paper_bgcolor="rgba(10,10,10,0.6)",
                    font=dict(color="#f8f8f8"),
                )
                chart.plotly_chart(fig, use_container_width=True)

            st.success("✅ Live tracking completed (demo mode).")

        except Exception as e:
            st.error(f"Error during live tracking: {e}")
