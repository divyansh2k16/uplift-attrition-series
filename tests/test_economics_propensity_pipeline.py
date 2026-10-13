import numpy as np
import pandas as pd
import pytest

from uplift_series import analysis_pool, generate_bank, naive_uplift, true_average_uplift
from uplift_series.economics import (
    apply_uplift_calibration,
    fit_uplift_calibration,
    net_value,
    profit_curve,
    revenue_at_risk,
    true_net_value,
)
from uplift_series.pipeline import holdout_readout, run_cycle, simulate_outcomes
from uplift_series.propensity import (
    PropensityModel,
    default_strata,
    ipw_weights,
    standardized_mean_differences,
    stratified_effect,
    weighted_effect,
)


def test_net_value_worked_example():
    # Client A from the guide: $5M at risk, uplift 0.10, P(stay | pricing) 0.70,
    # 3% concession, $10K RM cost -> 500K - 105K - 10K = +385K.
    assert net_value(0.10, 0.30, 5_000_000) == pytest.approx(385_000)
    # A loyal client loses money: tiny uplift, almost everyone stays.
    assert net_value(0.01, 0.04, 8_000_000) < 0


def test_revenue_at_risk_and_true_value():
    df = generate_bank(n_clients=2_000, n_flagged=400, n_pricing=80, n_service_fix=80, seed=3)
    assert np.allclose(revenue_at_risk(df["revenue_ttm"]), 2 * df["revenue_ttm"])
    assert len(true_net_value(df)) == len(df)


def test_calibration_recovers_a_line():
    table = pd.DataFrame({"predicted_uplift": [0.3, 0.2, 0.1, 0.0],
                          "actual_uplift": [0.15, 0.10, 0.05, 0.0], "clients": [100] * 4})
    intercept, slope = fit_uplift_calibration(table)
    assert slope == pytest.approx(0.5) and intercept == pytest.approx(0.0, abs=1e-9)
    assert apply_uplift_calibration([0.4], (intercept, slope))[0] == pytest.approx(0.2)


def test_profit_curve_is_cumulative():
    d = pd.DataFrame({"s": [3, 1, 2], "v": [10.0, -5.0, 4.0]})
    curve = profit_curve(d, "s", "v")
    assert curve["cumulative_value"].tolist() == [10.0, 14.0, 9.0]


@pytest.fixture(scope="module")
def biased_pool():
    return analysis_pool(generate_bank(assignment="rm_biased"))


def test_weighting_moves_estimate_towards_truth(biased_pool):
    pool = biased_pool.copy()
    pool["e"] = PropensityModel().fit(pool).predict(pool)
    pool["w"] = ipw_weights(pool["treated"], pool["e"])
    naive_gap = abs(naive_uplift(pool) - true_average_uplift(pool))
    ipw_gap = abs(weighted_effect(pool, "w") - true_average_uplift(pool))
    assert ipw_gap < naive_gap / 2


def test_stratification_moves_estimate_towards_truth(biased_pool):
    result = stratified_effect(biased_pool, default_strata(biased_pool))
    assert abs(result["estimate"] - true_average_uplift(biased_pool)) < abs(
        naive_uplift(biased_pool) - true_average_uplift(biased_pool)) / 2
    assert result["clients_dropped"] >= 0


def test_weighting_improves_balance(biased_pool):
    pool = biased_pool.copy()
    pool["e"] = PropensityModel().fit(pool).predict(pool)
    pool["w"] = ipw_weights(pool["treated"], pool["e"])
    cols = ["tenure_years", "rm_meetings_6m", "complaints_6m"]
    before = standardized_mean_differences(pool, cols).abs()
    after = standardized_mean_differences(pool, cols, weight="w").abs()
    assert (after < before).all()


def test_ipw_weights_are_clipped():
    w = ipw_weights([1, 0], [0.0001, 0.9999], clip=(0.02, 0.98))
    assert w.max() == pytest.approx(50.0)


def test_pipeline_runs_and_holds_back():
    history = generate_bank(assignment="rm_biased")
    live = generate_bank(assignment="rm_biased", seed=2026)
    rm_list, parts = run_cycle(history, live, capacity=300, holdout_share=0.2, seed=1)
    assert len(rm_list) <= 300
    assert (rm_list["pred_net_value"] > 0).all()
    assert 0.1 < (rm_list["action"] == "hold back").mean() < 0.3
    readout = holdout_readout(simulate_outcomes(rm_list, seed=1))
    assert readout["negotiated"] + readout["held_back"] == len(rm_list)
