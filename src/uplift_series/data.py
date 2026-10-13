"""Synthetic institutional-banking data for the uplift modelling series.

Every number in this module is invented. The generator exists so that each
article can compare an estimate against a *known* truth, something real data
can never offer.

Design (see docs/data_spec.md for the full write-up):

1. 40,000 institutional clients with pre-period features.
2. A baseline six-month churn probability, ``p0``, if the bank does nothing.
3. A pricing negotiation shifts churn on the log-odds scale. The size and
   sign of the shift depend on the client, which creates the four client
   types: persuadables, sure things, lost causes and sleeping dogs.
4. An existing churn model flags the riskiest 6,000 clients for RMs.
5. Among flagged clients, 1,000 get a service fix and 1,000 get a pricing
   negotiation. Who gets pricing depends on ``assignment``:
   - ``"random"``: a coin flip (a clean experiment, used in Act 2).
   - ``"rm_biased"``: RMs pick "easy" clients (used in Act 3). Part of what
     makes a client easy is a hidden relationship-strength variable that is
     not in the observed data, so propensity methods can only partly fix it.
6. Outcomes are drawn from the probability that matches what actually
   happened to each client.

Columns starting with ``true_`` (and ``hidden_relationship``) are oracle
columns: they are never available in real life and must never be used as
model features. ``observed_view`` drops them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Public constants
# --------------------------------------------------------------------------

#: Pre-period features a model may use.
FEATURES = [
    "revenue_ttm",
    "tenure_years",
    "n_products",
    "txn_volume_trend",
    "pricing_gap",
    "rm_meetings_6m",
    "complaints_6m",
    "corporate_event",
    "years_since_price_review",
    "segment_large_corporate",
    "segment_mid_market",
]

#: Measured *after* the start date. Included on purpose to demonstrate
#: data leakage in Part 3. Never use it as a feature.
LEAKY_COLUMNS = ["txn_change_after_start"]

#: Oracle columns: known only because the data is synthetic.
ORACLE_COLUMNS = [
    "hidden_relationship",
    "true_p_churn_no_action",
    "true_p_churn_pricing",
    "true_uplift",
    "true_client_type",
    "true_propensity",
]

#: Seed used throughout the series. Chosen because its headline numbers sit
#: close to the average over many seeds (tests check the averages).
SERIES_SEED = 54

SEGMENTS = ["large_corporate", "mid_market", "commercial"]
CYCLE_DATES = pd.to_datetime(["2025-01-01", "2025-04-01", "2025-07-01", "2025-10-01"])
OUTCOME_WINDOW_MONTHS = 6


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _zscore(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / x.std()


def _calibrate_intercept(logit_without_intercept, target_mean):
    """Find the intercept so that mean(sigmoid(logit + b)) == target_mean."""
    lo, hi = -20.0, 20.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if _sigmoid(logit_without_intercept + mid).mean() < target_mean:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def classify_client_type(p0, uplift):
    """Label the four uplift client types from oracle quantities.

    - persuadable: pricing cuts churn by at least 5 points
    - sleeping_dog: pricing *raises* churn by at least 2 points
    - lost_cause: little effect and churn risk of 30% or more either way
    - sure_thing: little effect and low churn risk
    """
    p0 = np.asarray(p0)
    uplift = np.asarray(uplift)
    out = np.where(uplift >= 0.05, "persuadable", "")
    out = np.where((out == "") & (uplift <= -0.02), "sleeping_dog", out)
    out = np.where((out == "") & (p0 >= 0.30), "lost_cause", out)
    out = np.where(out == "", "sure_thing", out)
    return out


# --------------------------------------------------------------------------
# Generator
# --------------------------------------------------------------------------

def generate_bank(
    n_clients: int = 40_000,
    n_flagged: int = 6_000,
    n_pricing: int = 1_000,
    n_service_fix: int = 1_000,
    assignment: str = "random",
    hidden_confounding: float = 0.5,
    selection_strength: float = 0.7,
    base_churn_rate: float = 0.08,
    seed: int = SERIES_SEED,
) -> pd.DataFrame:
    """Generate one cohort of synthetic institutional clients.

    Parameters
    ----------
    n_clients : total clients scored by the churn model.
    n_flagged : clients the churn model sends to RMs (highest churn score).
    n_pricing : flagged clients who receive a pricing negotiation.
    n_service_fix : flagged clients who receive a different intervention
        (a service fix). They are excluded from the pricing analysis.
    assignment : ``"random"`` or ``"rm_biased"``.
    hidden_confounding : how strongly the unobserved relationship strength
        drives both RM choice and churn (0 = no hidden confounding).
        Only matters when ``assignment="rm_biased"``.
    selection_strength : how strongly RMs favour easy clients in
        ``"rm_biased"`` mode. Higher values push propensities towards 0 and 1.
    base_churn_rate : average six-month churn probability with no action.
    seed : random seed; the same seed always gives the same data.
    """
    if assignment not in {"random", "rm_biased"}:
        raise ValueError("assignment must be 'random' or 'rm_biased'")
    if n_pricing + n_service_fix > n_flagged:
        raise ValueError("n_pricing + n_service_fix cannot exceed n_flagged")

    rng = np.random.default_rng(seed)
    n = n_clients

    # ---- 1. Pre-period features ------------------------------------------
    segment = rng.choice(SEGMENTS, size=n, p=[0.15, 0.35, 0.50])
    median_revenue = np.select(
        [segment == "large_corporate", segment == "mid_market"], [4_000_000, 700_000], 150_000
    )
    revenue_ttm = np.round(median_revenue * rng.lognormal(0, 0.6, n), -3)

    # Hidden: the strength of the personal relationship (golf with the treasurer).
    hidden_relationship = rng.normal(0, 1, n)

    tenure_years = np.round(np.clip(rng.gamma(2.2, 3.5, n) + 0.8 * hidden_relationship, 0.5, 30), 1)
    n_products = np.clip(
        rng.poisson(1.5 + 0.08 * tenure_years + 0.6 * (segment == "large_corporate")), 1, 10
    )
    txn_volume_trend = np.round(rng.normal(0.0, 0.15, n), 3)
    pricing_gap = np.round(rng.normal(0.03, 0.07, n), 3)  # >0: we are dearer than peers
    rm_meetings_6m = rng.poisson(np.exp(0.6 + 0.35 * hidden_relationship + 0.4 * (segment == "large_corporate")))
    complaints_6m = rng.poisson(0.4, n)
    corporate_event = (rng.random(n) < 0.03).astype(int)  # merger / bank consolidation
    years_since_price_review = rng.integers(0, 9, n)

    # ---- 2. Baseline churn probability with no action (p0) ---------------
    observed_risk = (
        3.0 * pricing_gap
        - 4.0 * txn_volume_trend
        + 0.35 * complaints_6m
        - 0.25 * (n_products - 3)
        - 0.45 * np.log1p(tenure_years)
        - 0.30 * _zscore(np.log(revenue_ttm))
        - 0.30 * _zscore(rm_meetings_6m)
        + 2.8 * corporate_event
    )
    hidden_risk = -0.6 * hidden_confounding * hidden_relationship
    intercept = _calibrate_intercept(observed_risk + hidden_risk, base_churn_rate)
    logit_p0 = intercept + observed_risk + hidden_risk
    p0 = _sigmoid(logit_p0)

    # ---- 3. Effect of a pricing negotiation (on the log-odds scale) ------
    price_sensitivity = 2.2 * _sigmoid((pricing_gap - 0.04) / 0.03)  # 0 to 2.2
    # Client-level variation the features do not explain, so the effect is
    # realistic rather than a clean formula.
    effect_noise = rng.normal(0, 0.35, n)
    pricing_effect = price_sensitivity + effect_noise
    # Lost causes: a corporate decision that no price will reverse.
    pricing_effect = np.where(corporate_event == 1, 0.05 + 0.3 * effect_noise, pricing_effect)
    # Sleeping dogs: happy, cheaply priced, not reviewed in years; raising
    # pricing prompts them to run an RFP and shop around.
    is_sleeper = (years_since_price_review >= 5) & (pricing_gap <= 0.0) & (complaints_6m == 0)
    pricing_effect = np.where(is_sleeper, -1.0 + effect_noise, pricing_effect)
    p1 = _sigmoid(logit_p0 - pricing_effect)

    # A service fix helps everyone a little, and complainers the most.
    service_effect = 0.5 + 0.5 * np.minimum(complaints_6m, 3)
    p_service = _sigmoid(logit_p0 - service_effect)

    true_uplift = p0 - p1

    # ---- 4. Existing churn model flags the riskiest clients --------------
    # The bank's churn model only sees observed features, imperfectly.
    churn_score = _sigmoid(intercept + observed_risk + rng.normal(0, 0.5, n))
    flagged = np.zeros(n, dtype=bool)
    flagged[np.argsort(-churn_score)[:n_flagged]] = True
    flagged_idx = np.flatnonzero(flagged)

    # ---- 5. Who got which intervention ----------------------------------
    treatment = np.array(["none"] * n, dtype=object)
    service_idx = rng.choice(flagged_idx, size=n_service_fix, replace=False)
    treatment[service_idx] = "service_fix"
    eligible = flagged & (treatment == "none")  # candidates for pricing

    true_propensity = np.zeros(n)
    if assignment == "random":
        true_propensity[eligible] = n_pricing / eligible.sum()
    else:
        # RMs favour clients they find easy: long tenure, frequent meetings,
        # big revenue, a strong (hidden) relationship; they avoid complainers
        # and clients in the middle of a corporate event.
        easiness = selection_strength * (
            0.7 * _zscore(np.log1p(tenure_years))
            + 0.6 * _zscore(rm_meetings_6m)
            + 0.5 * _zscore(np.log(revenue_ttm))
            - 0.6 * complaints_6m
            - 2.5 * corporate_event
            + 0.9 * hidden_confounding * hidden_relationship
        )
        e = easiness[eligible]
        b = _calibrate_intercept(e, n_pricing / eligible.sum())
        true_propensity[eligible] = _sigmoid(e + b)

    gets_pricing = eligible & (rng.random(n) < true_propensity)
    treatment[gets_pricing] = "pricing"

    # ---- 6. Outcomes -----------------------------------------------------
    p_actual = np.select([treatment == "pricing", treatment == "service_fix"], [p1, p_service], p0)
    churned = (rng.random(n) < p_actual).astype(int)

    # A leaky post-start feature: clients who left stopped transacting.
    txn_change_after_start = np.round(
        np.where(churned == 1, rng.normal(-0.6, 0.2, n), rng.normal(0.02, 0.15, n)), 3
    )

    start_date = CYCLE_DATES[rng.integers(0, len(CYCLE_DATES), n)]

    df = pd.DataFrame(
        {
            "client_id": [f"C{i:05d}" for i in range(1, n + 1)],
            "segment": segment,
            "revenue_ttm": revenue_ttm,
            "tenure_years": tenure_years,
            "n_products": n_products,
            "txn_volume_trend": txn_volume_trend,
            "pricing_gap": pricing_gap,
            "rm_meetings_6m": rm_meetings_6m,
            "complaints_6m": complaints_6m,
            "corporate_event": corporate_event,
            "years_since_price_review": years_since_price_review,
            "churn_score": np.round(churn_score, 4),
            "flagged": flagged.astype(int),
            "start_date": start_date,
            "treatment": treatment,
            "churned_6m": churned,
            "txn_change_after_start": txn_change_after_start,
            "hidden_relationship": np.round(hidden_relationship, 3),
            "true_p_churn_no_action": p0,
            "true_p_churn_pricing": p1,
            "true_uplift": true_uplift,
            "true_client_type": classify_client_type(p0, true_uplift),
            "true_propensity": true_propensity,
        }
    )
    df["segment_large_corporate"] = (df["segment"] == "large_corporate").astype(int)
    df["segment_mid_market"] = (df["segment"] == "mid_market").astype(int)
    df.attrs.update(
        assignment=assignment,
        seed=seed,
        hidden_confounding=hidden_confounding,
        selection_strength=selection_strength,
    )
    return df


# --------------------------------------------------------------------------
# Helpers used across notebooks
# --------------------------------------------------------------------------

def observed_view(df: pd.DataFrame) -> pd.DataFrame:
    """Drop oracle columns: what an analyst at the bank would actually see."""
    return df.drop(columns=[c for c in ORACLE_COLUMNS if c in df.columns])


def analysis_pool(df: pd.DataFrame) -> pd.DataFrame:
    """Flagged clients who got pricing (treated) or nothing (control).

    Service-fix clients are excluded: they are not "no intervention".
    Adds ``treated`` = 1 for pricing, 0 for no action.
    """
    pool = df[(df["flagged"] == 1) & (df["treatment"].isin(["pricing", "none"]))].copy()
    pool["treated"] = (pool["treatment"] == "pricing").astype(int)
    return pool


def naive_uplift(pool: pd.DataFrame) -> float:
    """Control churn rate minus treated churn rate, with no adjustment."""
    treated = pool.loc[pool["treated"] == 1, "churned_6m"].mean()
    control = pool.loc[pool["treated"] == 0, "churned_6m"].mean()
    return control - treated


def true_average_uplift(pool: pd.DataFrame) -> float:
    """Average true uplift over the pool (the ATE we are trying to recover)."""
    return pool["true_uplift"].mean()
