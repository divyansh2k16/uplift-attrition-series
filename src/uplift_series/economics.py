"""Turning uplift into money.

For one client, compare the two choices:

- negotiate pricing: expected value = P(stay | pricing) x (revenue - concession)
- do nothing:        expected value = P(stay | no action) x revenue

The difference, minus the RM's effort, is the client's **net value**:

    net value = uplift x revenue at risk
                - P(stay | pricing) x concession
                - RM cost

because P(stay | pricing) - P(stay | no action) is exactly the uplift. The
middle term is why discounting loyal clients is expensive: the concession is
paid by every treated client who stays, including those who were never
leaving.

All amounts are illustrative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON_YEARS = 2          # a saved client keeps paying beyond one year
CONCESSION_RATE = 0.03     # 3% price concession over the same horizon
RM_COST = 10_000           # cost of one negotiation (RM and pricing desk time)


def revenue_at_risk(revenue_ttm, years: float = HORIZON_YEARS):
    """Revenue lost over the horizon if the client leaves."""
    return np.asarray(revenue_ttm, dtype=float) * years


def net_value(uplift, p_churn_pricing, rev_at_risk, concession_rate: float = CONCESSION_RATE,
              rm_cost: float = RM_COST):
    """Expected dollars gained by negotiating pricing with a client."""
    uplift = np.asarray(uplift, dtype=float)
    p_stay_pricing = 1 - np.asarray(p_churn_pricing, dtype=float)
    rar = np.asarray(rev_at_risk, dtype=float)
    return uplift * rar - p_stay_pricing * concession_rate * rar - rm_cost


def true_net_value(data: pd.DataFrame, concession_rate: float = CONCESSION_RATE,
                   rm_cost: float = RM_COST, years: float = HORIZON_YEARS):
    """Net value computed from the oracle columns (synthetic data only)."""
    return net_value(data["true_uplift"], data["true_p_churn_pricing"],
                     revenue_at_risk(data["revenue_ttm"], years), concession_rate, rm_cost)


def fit_uplift_calibration(bins_table: pd.DataFrame) -> tuple[float, float]:
    """Straight-line correction of predicted uplift, from a holdout bin table.

    Fits actual uplift = intercept + slope x predicted uplift across the bins
    of ``evaluation.uplift_by_bin`` (weighted by bin size). A slope below 1
    means the model's predictions are too spread out, which is typical at the
    top of a ranking.
    """
    slope, intercept = np.polyfit(bins_table["predicted_uplift"], bins_table["actual_uplift"], 1,
                                  w=np.sqrt(bins_table["clients"]))
    return float(intercept), float(slope)


def apply_uplift_calibration(uplift, calibration: tuple[float, float]):
    intercept, slope = calibration
    return intercept + slope * np.asarray(uplift, dtype=float)


def profit_curve(data: pd.DataFrame, score: str, value: str) -> pd.DataFrame:
    """Cumulative ``value`` when clients are called in order of ``score``."""
    d = data.sort_values(score, ascending=False)
    return pd.DataFrame({
        "clients_called": np.arange(1, len(d) + 1),
        "cumulative_value": d[value].cumsum().to_numpy(),
    })
