"""
Train a model to predict each player's points for the next gameweek.

NOTE: This baseline trains on a single snapshot of current-season aggregates
(data/processed/player_features.parquet). For a real time-series model you'll
want to accumulate one row per (player, gameweek) over multiple weeks -
see build_training_history() below for the recommended approach once you
have several weeks of data collected.

Run:
    python src/train_model.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "price", "points_per_million", "points_per_90", "form",
    "avg_points_last_n", "avg_minutes_last_n", "std_points_last_n",
    "ict_index", "influence", "creativity", "threat",
    "expected_goal_involvements", "next_fixture_difficulty",
]
TARGET_COL = "avg_points_last_n"  # proxy target for the baseline model


def build_training_history():
    """
    Placeholder for building a proper (player, gameweek) -> next_gw_points
    training set once you have collected several weeks of snapshots.
    Each week, save data/processed/player_features_gw{N}.parquet, then
    concatenate them here with the label = points scored in gw{N+1}.
    """
    raise NotImplementedError(
        "Collect multiple weekly snapshots before building a true time-series dataset."
    )


def load_features() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / "player_features.parquet")
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    df[FEATURE_COLS] = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)
    return df


def train_baseline(df: pd.DataFrame):
    """
    Baseline: predict avg_points_last_n (a stand-in for 'expected next GW
    points') from ICT/fixture/price features. Once you have real next-GW
    labels collected week over week, swap TARGET_COL for actual outcomes.
    """
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"Validation MAE: {mae:.3f}")

    model_path = MODELS_DIR / "gw_predictor.pkl"
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")

    with open(MODELS_DIR / "feature_cols.json", "w") as f:
        json.dump(FEATURE_COLS, f)

    return model


def predict_next_gw(df: pd.DataFrame, model) -> pd.DataFrame:
    X = df[FEATURE_COLS]
    df = df.copy()
    df["predicted_points"] = model.predict(X)
    return df.sort_values("predicted_points", ascending=False)


def main():
    df = load_features()
    model = train_baseline(df)
    preds = predict_next_gw(df, model)

    out_path = PROCESSED_DIR / "predictions.parquet"
    preds.to_parquet(out_path, index=False)
    print(f"Saved predictions to {out_path}")
    print(preds[["web_name", "position", "price", "predicted_points"]].head(10))


if __name__ == "__main__":
    main()
