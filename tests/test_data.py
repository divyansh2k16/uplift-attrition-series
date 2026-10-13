"""Tests for the synthetic data generator.

These tests protect the facts every article relies on:
- the data is reproducible,
- the oracle quantities are internally consistent,
- random assignment gives an unbiased naive estimate,
- RM-biased assignment overstates it, and
- weighting by the true propensity recovers the truth when nothing is hidden.
"""

import numpy as np
import pandas as pd
import pytest

from uplift_series.data import (
    SERIES_SEED,
    FEATURES,
    ORACLE_COLUMNS,
    analysis_pool,
    generate_bank,
    naive_uplift,
    observed_view,
    true_average_uplift,
)


@pytest.fixture(scope="module")
def random_df():
    return generate_bank(assignment="random", seed=SERIES_SEED)


@pytest.fixture(scope="module")
def biased_df():
    return generate_bank(assignment="rm_biased", seed=SERIES_SEED)


def test_reproducible():
    a = generate_bank(n_clients=5_000, n_flagged=1_000, n_pricing=200, n_service_fix=200, seed=1)
    b = generate_bank(n_clients=5_000, n_flagged=1_000, n_pricing=200, n_service_fix=200, seed=1)
    pd.testing.assert_frame_equal(a, b)


def test_pool_sizes(random_df):
    assert len(random_df) == 40_000
    assert random_df["flagged"].sum() == 6_000
    assert (random_df["treatment"] == "service_fix").sum() == 1_000
    n_pricing = (random_df["treatment"] == "pricing").sum()
    assert 900 <= n_pricing <= 1_100
    # Only flagged clients can receive an intervention.
    assert (random_df.loc[random_df["flagged"] == 0, "treatment"] == "none").all()


def test_probabilities_are_valid(random_df):
    for col in ["true_p_churn_no_action", "true_p_churn_pricing", "churn_score"]:
        assert random_df[col].between(0, 1).all()
    assert np.allclose(
        random_df["true_uplift"],
        random_df["true_p_churn_no_action"] - random_df["true_p_churn_pricing"],
    )


def test_all_four_client_types_present(random_df):
    flagged_types = set(random_df.loc[random_df["flagged"] == 1, "true_client_type"])
    assert flagged_types == {"persuadable", "sure_thing", "lost_cause", "sleeping_dog"}


def test_flagged_clients_are_riskier(random_df):
    flagged = random_df["flagged"] == 1
    assert (
        random_df.loc[flagged, "true_p_churn_no_action"].mean()
        > 2 * random_df.loc[~flagged, "true_p_churn_no_action"].mean()
    )


def test_observed_view_hides_oracle_columns(random_df):
    seen = observed_view(random_df)
    assert not set(ORACLE_COLUMNS) & set(seen.columns)
    assert set(FEATURES) <= set(seen.columns)


def test_analysis_pool_excludes_service_fix(random_df):
    pool = analysis_pool(random_df)
    assert set(pool["treatment"]) == {"pricing", "none"}
    assert len(pool) == 5_000
    assert pool["treated"].sum() == (pool["treatment"] == "pricing").sum()


def test_random_assignment_naive_is_close_to_truth():
    gaps = []
    for seed in range(5):
        pool = analysis_pool(generate_bank(assignment="random", seed=seed))
        gaps.append(naive_uplift(pool) - true_average_uplift(pool))
    assert abs(np.mean(gaps)) < 0.015


def test_rm_bias_overstates_uplift(biased_df):
    pool = analysis_pool(biased_df)
    assert naive_uplift(pool) > true_average_uplift(pool) + 0.05
    # RMs picked clients who were less likely to leave anyway.
    treated = pool[pool["treated"] == 1]
    control = pool[pool["treated"] == 0]
    assert treated["true_p_churn_no_action"].mean() < control["true_p_churn_no_action"].mean()


def test_true_propensity_weighting_recovers_truth_without_hidden_confounding():
    estimates, truths = [], []
    for seed in range(5):
        pool = analysis_pool(generate_bank(assignment="rm_biased", hidden_confounding=0.0, seed=seed))
        e = pool["true_propensity"].clip(0.01, 0.99).to_numpy()
        t = pool["treated"].to_numpy()
        y = pool["churned_6m"].to_numpy()
        w1, w0 = t / e, (1 - t) / (1 - e)
        estimates.append((w0 * y).sum() / w0.sum() - (w1 * y).sum() / w1.sum())
        truths.append(true_average_uplift(pool))
    assert abs(np.mean(estimates) - np.mean(truths)) < 0.02


def test_invalid_arguments():
    with pytest.raises(ValueError):
        generate_bank(assignment="coin_toss")
    with pytest.raises(ValueError):
        generate_bank(n_flagged=100, n_pricing=80, n_service_fix=80)
