"""
Fetch raw data from the official Fantasy Premier League API.

Endpoints used:
- bootstrap-static: players, teams, positions, current gameweek info
- fixtures: full season fixture list with difficulty ratings
- element-summary/{id}: per-player gameweek-by-gameweek history

Run:
    python src/fetch_data.py
"""
import json
import os
import time
from pathlib import Path

import requests

BASE_URL = "https://fantasy.premierleague.com/api"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def _get(url: str) -> dict:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_bootstrap_static() -> dict:
    """Players, teams, positions, gameweeks (events)."""
    data = _get(f"{BASE_URL}/bootstrap-static/")
    _save(data, "bootstrap_static.json")
    return data


def fetch_fixtures() -> list:
    """Full season fixtures with difficulty ratings."""
    data = _get(f"{BASE_URL}/fixtures/")
    _save(data, "fixtures.json")
    return data


def fetch_player_history(player_ids: list[int], sleep: float = 0.2) -> dict:
    """Per-player gameweek history (points, minutes, xG, etc.)."""
    histories = {}
    for pid in player_ids:
        try:
            data = _get(f"{BASE_URL}/element-summary/{pid}/")
            histories[pid] = data
        except requests.RequestException as exc:
            print(f"Failed to fetch player {pid}: {exc}")
        time.sleep(sleep)  # be polite to the API
    _save(histories, "player_histories.json")
    return histories


def _save(data, filename: str) -> None:
    path = RAW_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"Saved {path}")


def main():
    bootstrap = fetch_bootstrap_static()
    fetch_fixtures()

    player_ids = [p["id"] for p in bootstrap["elements"]]
    print(f"Fetching history for {len(player_ids)} players...")
    fetch_player_history(player_ids)


if __name__ == "__main__":
    main()
