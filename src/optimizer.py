"""
Integer Linear Programming optimizer for building the optimal FPL squad.

Constraints (standard FPL rules):
- 15 total players: 2 GKP, 5 DEF, 5 MID, 3 FWD
- Budget: 100.0m (configurable)
- Max 3 players per real-life team
- Objective: maximize total predicted_points (or any score column)

Also selects the best starting XI (valid formation) and captain
(2x highest predicted scorer in the XI) from the chosen 15.

Run:
    python src/optimizer.py
"""
from pathlib import Path

import pandas as pd
import pulp

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

SQUAD_RULES = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
FORMATIONS = [
    # (GKP, DEF, MID, FWD) - valid starting XI formations
    (1, 3, 4, 3), (1, 3, 5, 2), (1, 4, 4, 2),
    (1, 4, 3, 3), (1, 5, 4, 1), (1, 5, 3, 2), (1, 4, 5, 1),
]


def select_squad(
    df: pd.DataFrame,
    score_col: str = "predicted_points",
    budget: float = 100.0,
    max_per_team: int = 3,
    must_include: list[str] | None = None,
    must_exclude: list[str] | None = None,
) -> pd.DataFrame:
    """Solve ILP to pick the 15-man squad maximizing score_col under budget."""
    df = df.reset_index(drop=True)
    prob = pulp.LpProblem("fpl_squad", pulp.LpMaximize)

    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in df.index}

    prob += pulp.lpSum(x[i] * df.loc[i, score_col] for i in df.index)

    prob += pulp.lpSum(x[i] * df.loc[i, "price"] for i in df.index) <= budget
    prob += pulp.lpSum(x[i] for i in df.index) == 15

    for pos, count in SQUAD_RULES.items():
        prob += pulp.lpSum(
            x[i] for i in df.index if df.loc[i, "position"] == pos
        ) == count

    for team in df["team_name"].unique():
        prob += pulp.lpSum(
            x[i] for i in df.index if df.loc[i, "team_name"] == team
        ) <= max_per_team

    if must_include:
        for name in must_include:
            idxs = df.index[df["web_name"] == name].tolist()
            if idxs:
                prob += x[idxs[0]] == 1

    if must_exclude:
        for name in must_exclude:
            idxs = df.index[df["web_name"] == name].tolist()
            if idxs:
                prob += x[idxs[0]] == 0

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"Solver status: {pulp.LpStatus[prob.status]}")

    chosen = [i for i in df.index if x[i].value() == 1]
    return df.loc[chosen].sort_values("position").reset_index(drop=True)


def select_best_xi(squad: pd.DataFrame, score_col: str = "predicted_points"):
    """Given a 15-man squad, pick the highest-scoring valid starting XI + captain."""
    best_xi, best_score, best_formation = None, -1, None

    for gkp, dfc, mid, fwd in FORMATIONS:
        counts = {"GKP": gkp, "DEF": dfc, "MID": mid, "FWD": fwd}
        picks = []
        for pos, n in counts.items():
            pool = squad[squad["position"] == pos].sort_values(
                score_col, ascending=False
            )
            picks.append(pool.head(n))
        xi = pd.concat(picks)
        total = xi[score_col].sum()
        if total > best_score:
            best_xi, best_score, best_formation = xi, total, (gkp, dfc, mid, fwd)

    captain = best_xi.sort_values(score_col, ascending=False).iloc[0]
    return best_xi, best_formation, captain


def main():
    df = pd.read_parquet(PROCESSED_DIR / "predictions.parquet")
    squad = select_squad(df)
    xi, formation, captain = select_best_xi(squad)

    print(f"Formation: {formation}")
    print(f"Total squad cost: {squad['price'].sum():.1f}m")
    print("\nStarting XI:")
    print(xi[["web_name", "position", "team_name", "price", "predicted_points"]])
    print(f"\nCaptain: {captain['web_name']} ({captain['predicted_points']:.1f} pts)")


if __name__ == "__main__":
    main()
