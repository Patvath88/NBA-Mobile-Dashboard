import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_loader import get_player_context, get_player_id
from streamlit_lottie import st_lottie
import requests

st.set_page_config(page_title="Player Stats", page_icon="📊", layout="wide")

# 🎨 Custom CSS
def load_custom_css():
    with open("assets/theme.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_custom_css()

# ✨ Animation
def load_lottieurl(url):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None

st.title("📊 Player Performance Viewer")
st.caption("Explore rolling trends, matchups, and game-by-game analytics")

st_lottie(load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_5ngs2ksb.json"), height=120)

player = st.text_input("Enter player name", "LeBron James")
opponent = st.text_input("Next Opponent", "Golden State Warriors")

if st.button("Load Player Data"):
    with st.spinner("Fetching stats..."):
        try:
            context = get_player_context(player, opponent)
            df = context["recent_games"]

            st.subheader(f"{player} — Season Averages")
            st.dataframe(pd.DataFrame(context["season_avg"]).T, use_container_width=True)

            st.subheader("📈 Recent Performance Trends")
            if not df.empty:
                fig = px.line(df, x="GAME_DATE", y=["PTS", "REB", "AST", "PRA"], markers=True)
                fig.update_layout(
                    template="plotly_dark",
                    plot_bgcolor="rgba(10,10,10,0.6)",
                    paper_bgcolor="rgba(10,10,10,0.6)",
                    hovermode="x unified",
                    font=dict(color="#f8f8f8"),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No recent data available.")

        except Exception as e:
            st.error(f"Error loading data: {e}")
