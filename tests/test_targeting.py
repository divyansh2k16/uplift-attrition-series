import pandas as pd
import pytest

from uplift_series.targeting import (
    compare_strategies,
    expected_clients_pushed_out,
    expected_clients_saved,
    random_k,
    saves_per_100_calls,
    top_k,
)


@pytest.fixture
def toy():
    # The four-client example from Part 1 of the series.
    return pd.DataFrame(
        {
            "client": ["A", "B", "C", "D"],
            "true_p_churn_no_action": [0.80, 0.40, 0.05, 0.10],
            "true_uplift": [0.02, 0.25, 0.00, -0.10],
            "true_client_type": ["lost_cause", "persuadable", "sure_thing", "sleeping_dog"],
        }
    )


def test_saved_is_sum_of_uplift(toy):
    assert expected_clients_saved(toy) == pytest.approx(0.17)


def test_pushed_out_counts_only_negative_uplift(toy):
    assert expected_clients_pushed_out(toy) == pytest.approx(0.10)


def test_ranking_by_risk_vs_uplift(toy):
    by_risk = top_k(toy, "true_p_churn_no_action", 1)
    by_uplift = top_k(toy, "true_uplift", 1)
    assert by_risk["client"].item() == "A"
    assert by_uplift["client"].item() == "B"
    assert expected_clients_saved(by_uplift) > expected_clients_saved(by_risk)


def test_compare_strategies_shape(toy):
    out = compare_strategies(
        toy,
        2,
        {"risk": top_k(toy, "true_p_churn_no_action", 2), "random": random_k(toy, 2, seed=1)},
    )
    assert list(out.index) == ["risk", "random"]
    assert out.loc["risk", "persuadables"] == 1


def test_compare_strategies_rejects_wrong_size(toy):
    with pytest.raises(ValueError):
        compare_strategies(toy, 3, {"risk": top_k(toy, "true_p_churn_no_action", 2)})


def test_saves_per_100_calls(toy):
    assert saves_per_100_calls(toy) == pytest.approx(100 * 0.17 / 4)
