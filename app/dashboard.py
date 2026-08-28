"""
Streamlit dashboard for FPL analysis.

Run:
    streamlit run app/dashboard.py
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.optimizer import select_squad, select_best_xi  # noqa: E402

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

st.set_page_config(page_title="FPL Dashboard", layout="wide")
st.title("⚽ Fantasy Premier League Dashboard")


@st.cache_data
def load_predictions() -> pd.DataFrame:
    path = PROCESSED_DIR / "predictions.parquet"
    if not path.exists():
        st.error(
            "No predictions found. Run `python src/fetch_data.py`, "
            "`python src/features.py`, then `python src/train_model.py` first."
        )
        st.stop()
    return pd.read_parquet(path)


df = load_predictions()

tab1, tab2, tab3 = st.tabs(
    ["📊 Efficiency Rankings", "🔮 Predicted Top Scorers", "🏆 Optimal Team Builder"]
)

with tab1:
    st.subheader("Most Efficient Players")
    positions = st.multiselect(
        "Position", options=sorted(df["position"].unique()), default=None
    )
    filtered = df if not positions else df[df["position"].isin(positions)]

    sort_metric = st.selectbox(
        "Sort by", ["points_per_million", "points_per_90", "form"], index=0
    )
    top_n = st.slider("Show top N", 5, 50, 20)
    ranked = filtered.sort_values(sort_metric, ascending=False).head(top_n)

    st.dataframe(
        ranked[
            ["web_name", "team_name", "position", "price", "total_points",
             "points_per_million", "points_per_90", "form"]
        ],
        use_container_width=True,
    )
    fig = px.bar(
        ranked, x="web_name", y=sort_metric, color="position",
        title=f"Top {top_n} by {sort_metric}",
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Predicted Top Scorers — Next Gameweek")
    top_n2 = st.slider("Show top N players", 5, 50, 15, key="pred_n")
    preds = df.sort_values("predicted_points", ascending=False).head(top_n2)

    st.dataframe(
        preds[
            ["web_name", "team_name", "position", "price", "predicted_points",
             "next_fixture_difficulty"]
        ],
        use_container_width=True,
    )
    fig2 = px.bar(
        preds, x="web_name", y="predicted_points", color="position",
        title="Predicted Points — Next Gameweek",
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.subheader("Build the Optimal Squad")
    budget = st.slider("Budget (£m)", 80.0, 100.0, 100.0, step=0.5)
    max_per_team = st.slider("Max players per team", 1, 3, 3)

    all_names = sorted(df["web_name"].unique())
    must_include = st.multiselect("Must include", options=all_names)
    must_exclude = st.multiselect("Must exclude", options=all_names)

    if st.button("Optimize Squad"):
        with st.spinner("Solving..."):
            squad = select_squad(
                df,
                budget=budget,
                max_per_team=max_per_team,
                must_include=must_include or None,
                must_exclude=must_exclude or None,
            )
            xi, formation, captain = select_best_xi(squad)

        st.success(f"Optimal squad found! Formation: {formation}")
        st.metric("Total Cost", f"£{squad['price'].sum():.1f}m")
        st.metric("Predicted XI Points", f"{xi['predicted_points'].sum():.1f}")

        st.markdown("### Starting XI")
        st.dataframe(
            xi[["web_name", "position", "team_name", "price", "predicted_points"]],
            use_container_width=True,
        )
        st.info(f"🅲 Captain: **{captain['web_name']}** ({captain['predicted_points']:.1f} pts)")

        bench = squad[~squad["id"].isin(xi["id"])]
        st.markdown("### Bench")
        st.dataframe(
            bench[["web_name", "position", "team_name", "price", "predicted_points"]],
            use_container_width=True,
        )
