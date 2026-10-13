"""Levelling the playing field when RMs chose who got pricing.

The propensity score is the probability that a client like this one was
chosen for pricing. Two ways to use it:

1. **Stratification (counting):** sort clients into cells of look-alikes,
   compare treated and control churn only inside each cell, and average the
   cell results by cell size. Cells without both treated and control
   clients are dropped (no common support).
2. **Inverse probability weighting (IPW):** weight treated clients by
   1 / propensity and control clients by 1 / (1 - propensity), so both
   groups end up with the same mix of client types.

Both only balance what is recorded in the data. A hidden reason for the
RM's choice (the "golf with the treasurer" problem) is not fixed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from uplift_series.data import FEATURES


# --------------------------------------------------------------------------
# Stratification
# --------------------------------------------------------------------------

def default_strata(data: pd.DataFrame) -> pd.Series:
    """Cells built from five drivers of both RM choice and churn.

    Tenure (3 bands), RM meetings (3 bands), revenue (3 bands), any complaint
    (yes/no) and corporate event (yes/no): up to 108 cells.
    """
    tenure = pd.qcut(data["tenure_years"], 3, labels=["short", "mid", "long"])
    meetings = pd.cut(data["rm_meetings_6m"], [-1, 1, 3, np.inf], labels=["0-1", "2-3", "4+"])
    revenue = pd.qcut(data["revenue_ttm"], 3, labels=["small", "medium", "large"])
    complaints = np.where(data["complaints_6m"] > 0, "complained", "no complaint")
    event = np.where(data["corporate_event"] == 1, "event", "no event")
    return (tenure.astype(str) + " | " + meetings.astype(str) + " | " + revenue.astype(str)
            + " | " + complaints + " | " + event)


def stratified_effect(data: pd.DataFrame, strata: pd.Series, treatment: str = "treated",
                      outcome: str = "churned_6m") -> dict:
    """Average treatment effect by comparing within cells of look-alikes.

    Returns the estimate, the number of clients dropped for lack of overlap,
    and a per-cell table with each cell's propensity (treated share).
    """
    rows = []
    for cell, g in data.groupby(strata):
        t = g[g[treatment] == 1]
        c = g[g[treatment] == 0]
        rows.append({
            "cell": cell,
            "clients": len(g),
            "treated": len(t),
            "control": len(c),
            "propensity": len(t) / len(g),
            "effect": (c[outcome].mean() - t[outcome].mean()) if len(t) and len(c) else np.nan,
        })
    cells = pd.DataFrame(rows).set_index("cell")
    usable = cells.dropna(subset=["effect"])
    estimate = float(np.average(usable["effect"], weights=usable["clients"]))
    return {
        "estimate": estimate,
        "clients_dropped": int(cells["clients"].sum() - usable["clients"].sum()),
        "cells_used": len(usable),
        "cells_total": len(cells),
        "cells": cells.sort_values("clients", ascending=False),
    }


# --------------------------------------------------------------------------
# Propensity model and weights
# --------------------------------------------------------------------------

def _design(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    X = data[features].astype(float).copy()
    if "revenue_ttm" in X:
        X["revenue_ttm"] = np.log(X["revenue_ttm"])
    if "tenure_years" in X:
        X["tenure_years"] = np.log1p(X["tenure_years"])
    return X


class PropensityModel:
    """Logistic regression for P(chosen for pricing | features)."""

    def __init__(self, features: list[str] | None = None):
        self.features = features or FEATURES
        self.model = LogisticRegression(max_iter=2000)

    def fit(self, data: pd.DataFrame, treatment: str = "treated") -> "PropensityModel":
        X = _design(data, self.features)
        self.mean_, self.std_ = X.mean(), X.std().replace(0, 1)
        self.model.fit((X - self.mean_) / self.std_, data[treatment])
        return self

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        X = (_design(data, self.features) - self.mean_) / self.std_
        return self.model.predict_proba(X)[:, 1]


def ipw_weights(treated, propensity, clip: tuple[float, float] = (0.02, 0.98)) -> np.ndarray:
    """1 / e for treated clients, 1 / (1 - e) for control clients.

    Propensities are clipped (trimmed) so that a single client with a
    propensity near 0 or 1 cannot receive an enormous weight.
    """
    e = np.clip(np.asarray(propensity, dtype=float), *clip)
    t = np.asarray(treated)
    return np.where(t == 1, 1 / e, 1 / (1 - e))


def weighted_effect(data: pd.DataFrame, weight: str, treatment: str = "treated",
                    outcome: str = "churned_6m") -> float:
    """Weighted control churn minus weighted treated churn."""
    t = data[data[treatment] == 1]
    c = data[data[treatment] == 0]
    return float(np.average(c[outcome], weights=c[weight]) - np.average(t[outcome], weights=t[weight]))


def standardized_mean_differences(data: pd.DataFrame, columns: list[str], treatment: str = "treated",
                                  weight: str | None = None) -> pd.Series:
    """(treated mean - control mean) / pooled standard deviation, per column.

    Values within +/-0.1 are conventionally taken as balanced.
    """
    t = data[data[treatment] == 1]
    c = data[data[treatment] == 0]
    out = {}
    for col in columns:
        wt = t[weight] if weight else None
        wc = c[weight] if weight else None
        mt = np.average(t[col], weights=wt)
        mc = np.average(c[col], weights=wc)
        pooled = np.sqrt((t[col].var() + c[col].var()) / 2)
        out[col] = (mt - mc) / pooled if pooled > 0 else 0.0
    return pd.Series(out)
