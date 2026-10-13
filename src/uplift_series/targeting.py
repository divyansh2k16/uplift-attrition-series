"""Choosing which clients RMs should call, and scoring that choice.

The key identity used throughout the series:

    expected clients saved = sum of the true uplift of the clients you act on

because each client's uplift is the drop in their probability of leaving
caused by the call. Sleeping dogs have negative uplift, so calling them
subtracts from the total.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def top_k(df: pd.DataFrame, score: str, k: int) -> pd.DataFrame:
    """The k clients with the highest value of ``score``."""
    return df.nlargest(k, score)


def random_k(df: pd.DataFrame, k: int, seed: int = 0) -> pd.DataFrame:
    """k clients chosen at random (the baseline any strategy must beat)."""
    return df.sample(n=k, random_state=seed)


def expected_clients_saved(selected: pd.DataFrame) -> float:
    """Expected number of clients kept because they were called."""
    return float(selected["true_uplift"].sum())


def expected_clients_pushed_out(selected: pd.DataFrame) -> float:
    """Expected clients lost *because* they were called (sleeping dogs)."""
    uplift = selected["true_uplift"]
    return float(-uplift[uplift < 0].sum()) + 0.0  # + 0.0 avoids printing -0.0


def compare_strategies(df: pd.DataFrame, k: int, strategies: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarise several selections of k clients side by side."""
    rows = []
    for name, selected in strategies.items():
        if len(selected) != k:
            raise ValueError(f"strategy {name!r} selected {len(selected)} clients, expected {k}")
        types = selected["true_client_type"].value_counts()
        rows.append(
            {
                "strategy": name,
                "clients_called": k,
                "expected_saved": expected_clients_saved(selected),
                "expected_pushed_out": expected_clients_pushed_out(selected),
                "avg_churn_risk": float(selected["true_p_churn_no_action"].mean()),
                "avg_uplift": float(selected["true_uplift"].mean()),
                "persuadables": int(types.get("persuadable", 0)),
                "sure_things": int(types.get("sure_thing", 0)),
                "lost_causes": int(types.get("lost_cause", 0)),
                "sleeping_dogs": int(types.get("sleeping_dog", 0)),
            }
        )
    return pd.DataFrame(rows).set_index("strategy")


def saves_per_100_calls(selected: pd.DataFrame) -> float:
    """Expected clients saved for every 100 calls made."""
    return 100 * float(np.mean(selected["true_uplift"]))
