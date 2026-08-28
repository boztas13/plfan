"""
Build engineered features from raw FPL data:
- Efficiency metrics (points per million, points per 90)
- Rolling form (avg points over last N gameweeks)
- Fixture difficulty for upcoming gameweek
- Consistency (minutes played variance)

Run:
    python src/features.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

POSITION_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}


def load_raw():
    with open(RAW_DIR / "bootstrap_static.json") as f:
        bootstrap = json.load(f)
    with open(RAW_DIR / "fixtures.json") as f:
        fixtures = json.load(f)
    with open(RAW_DIR / "player_histories.json") as f:
        histories = json.load(f)
    return bootstrap, fixtures, histories


def build_player_table(bootstrap: dict) -> pd.DataFrame:
    df = pd.DataFrame(bootstrap["elements"])
    teams = {t["id"]: t["name"] for t in bootstrap["teams"]}

    df["team_name"] = df["team"].map(teams)
    df["position"] = df["element_type"].map(POSITION_MAP)
    df["price"] = df["now_cost"] / 10.0
    df["points_per_million"] = df["total_points"] / df["price"].replace(0, np.nan)

    df["minutes"] = df["minutes"].astype(float)
    df["points_per_90"] = np.where(
        df["minutes"] > 0, df["total_points"] / (df["minutes"] / 90.0), 0.0
    )

    keep_cols = [
        "id", "web_name", "team_name", "position", "price", "total_points",
        "points_per_million", "points_per_90", "minutes", "form",
        "selected_by_percent", "chance_of_playing_next_round",
        "ict_index", "influence", "creativity", "threat",
        "expected_goals", "expected_assists", "expected_goal_involvements",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols].copy()


def build_form_features(histories: dict, n_last: int = 5) -> pd.DataFrame:
    """Rolling average points/minutes over last N gameweeks per player."""
    rows = []
    for pid, hist in histories.items():
        gw_history = hist.get("history", [])
        if not gw_history:
            continue
        gw_df = pd.DataFrame(gw_history).sort_values("round")
        last_n = gw_df.tail(n_last)
        rows.append({
            "id": int(pid),
            "avg_points_last_n": last_n["total_points"].mean(),
            "avg_minutes_last_n": last_n["minutes"].mean(),
            "std_points_last_n": last_n["total_points"].std(ddof=0),
            "games_played_last_n": len(last_n),
        })
    return pd.DataFrame(rows)


def build_fixture_difficulty(fixtures: list, bootstrap: dict) -> pd.DataFrame:
    """Next-fixture difficulty per team (lower = easier)."""
    next_event = next(
        (e["id"] for e in bootstrap["events"] if e.get("is_next")), None
    )
    if next_event is None:
        return pd.DataFrame(columns=["team", "next_fixture_difficulty"])

    rows = []
    for fx in fixtures:
        if fx.get("event") != next_event:
            continue
        rows.append({"team": fx["team_h"], "next_fixture_difficulty": fx["team_h_difficulty"]})
        rows.append({"team": fx["team_a"], "next_fixture_difficulty": fx["team_a_difficulty"]})
    return pd.DataFrame(rows)


def main():
    bootstrap, fixtures, histories = load_raw()

    players = build_player_table(bootstrap)
    form = build_form_features(histories)
    fixture_diff = build_fixture_difficulty(fixtures, bootstrap)

    df = players.merge(form, on="id", how="left")

    team_id_map = {t["name"]: t["id"] for t in bootstrap["teams"]}
    df["team_id"] = df["team_name"].map(team_id_map)
    df = df.merge(
        fixture_diff.rename(columns={"team": "team_id"}), on="team_id", how="left"
    )

    out_path = PROCESSED_DIR / "player_features.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved {out_path} with {len(df)} players and {len(df.columns)} features")


if __name__ == "__main__":
    main()
