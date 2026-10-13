"""One retention cycle, end to end.

    history (RM-chosen pricing outcomes)
      -> propensity model and IPW weights
      -> weighted T-learner
      -> score this cycle's flagged clients
      -> net value per client
      -> RM list: top clients by net value, within capacity, net value > 0
      -> random holdout: a share of the list is not contacted this cycle

``simulate_outcomes`` then plays the cycle forward using the synthetic
bank's true probabilities, so the holdout's measurement can be compared
with the truth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from uplift_series.data import analysis_pool
from uplift_series.economics import net_value, revenue_at_risk
from uplift_series.modeling import TLearner
from uplift_series.propensity import PropensityModel, ipw_weights


def run_cycle(history: pd.DataFrame, live: pd.DataFrame, capacity: int = 500,
              holdout_share: float = 0.2, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """Build this cycle's RM list from biased history.

    Returns the RM list (one row per selected client, with an ``action``
    column of "negotiate" or "hold back") and a dict of fitted parts.
    """
    pool = analysis_pool(history)
    propensity = PropensityModel().fit(pool)
    pool["propensity"] = propensity.predict(pool)
    pool["ipw"] = ipw_weights(pool["treated"], pool["propensity"])
    model = TLearner().fit(pool, sample_weight="ipw")

    candidates = live[live["flagged"] == 1].copy()
    candidates["pred_uplift"] = model.predict_uplift(candidates)
    candidates["pred_p_churn_pricing"] = model.predict_churn_with_pricing(candidates)
    candidates["revenue_at_risk"] = revenue_at_risk(candidates["revenue_ttm"])
    candidates["pred_net_value"] = net_value(candidates["pred_uplift"], candidates["pred_p_churn_pricing"],
                                             candidates["revenue_at_risk"])

    rm_list = candidates[candidates["pred_net_value"] > 0].nlargest(capacity, "pred_net_value").copy()
    rng = np.random.default_rng(seed)
    held = rng.random(len(rm_list)) < holdout_share
    rm_list["action"] = np.where(held, "hold back", "negotiate")
    return rm_list, {"propensity_model": propensity, "uplift_model": model, "training_pool": pool}


def simulate_outcomes(rm_list: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Play the cycle forward with the true churn probabilities."""
    # A separate random stream from the holdout draw in run_cycle, so the
    # outcomes are independent of who was held back.
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    p = np.where(rm_list["action"] == "negotiate", rm_list["true_p_churn_pricing"],
                 rm_list["true_p_churn_no_action"])
    out = rm_list.copy()
    out["churned_next_6m"] = (rng.random(len(out)) < p).astype(int)
    return out


def holdout_readout(outcomes: pd.DataFrame) -> dict:
    """What the business sees: churn among negotiated vs held-back clients."""
    neg = outcomes[outcomes["action"] == "negotiate"]
    held = outcomes[outcomes["action"] == "hold back"]
    measured = held["churned_next_6m"].mean() - neg["churned_next_6m"].mean()
    return {
        "negotiated": len(neg),
        "held_back": len(held),
        "churn_negotiated": float(neg["churned_next_6m"].mean()),
        "churn_held_back": float(held["churned_next_6m"].mean()),
        "measured_uplift": float(measured),
        "predicted_uplift": float(outcomes["pred_uplift"].mean()),
        "true_uplift": float(outcomes["true_uplift"].mean()),
        "clients_saved_estimate": float(measured * len(neg)),
    }
