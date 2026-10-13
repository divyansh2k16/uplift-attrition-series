import numpy as np
import pandas as pd
import pytest

from uplift_series.evaluation import (
    assign_bins,
    bootstrap_uplift_by_bin,
    qini_coefficient,
    qini_curve,
    uplift_at_k,
    uplift_by_bin,
)


@pytest.fixture(scope="module")
def experiment():
    """A randomised experiment where pricing helps only clients with x = 1."""
    rng = np.random.default_rng(0)
    n = 20_000
    x = rng.integers(0, 2, n)
    treated = rng.integers(0, 2, n)
    p_churn = np.where(x == 1, 0.40 - 0.25 * treated, 0.20)
    churned = (rng.random(n) < p_churn).astype(int)
    return pd.DataFrame({
        "x": x,
        "treated": treated,
        "churned_6m": churned,
        "perfect": x + rng.random(n) * 0.1,   # ranks helped clients first
        "random": rng.random(n),
        "backwards": -x + rng.random(n) * 0.1,
    })


def test_bins_are_equal_and_ordered(experiment):
    bins = assign_bins(experiment, "perfect", 10)
    assert bins.value_counts().nunique() == 1
    assert experiment.loc[bins == 1, "perfect"].min() >= experiment.loc[bins == 10, "perfect"].max()


def test_uplift_by_bin_finds_the_effect(experiment):
    table = uplift_by_bin(experiment, "perfect", n_bins=2)
    assert table.loc[1, "actual_uplift"] == pytest.approx(0.25, abs=0.03)
    assert table.loc[2, "actual_uplift"] == pytest.approx(0.0, abs=0.03)


def test_unit_weights_change_nothing(experiment):
    data = experiment.assign(w=1.0)
    a = uplift_by_bin(data, "perfect", 5)
    b = uplift_by_bin(data, "perfect", 5, weight="w")
    pd.testing.assert_frame_equal(a, b)


def test_qini_curve_ends_at_overall_effect(experiment):
    curve = qini_curve(experiment, "random")
    assert curve["fraction"].iloc[0] == 0 and curve["fraction"].iloc[-1] == 1
    n_t = experiment["treated"].sum()
    t = experiment[experiment["treated"] == 1]["churned_6m"].mean()
    c = experiment[experiment["treated"] == 0]["churned_6m"].mean()
    assert curve["qini"].iloc[-1] == pytest.approx((c - t) * n_t, rel=1e-6)


def test_qini_coefficient_orders_models(experiment):
    perfect = qini_coefficient(experiment, "perfect")
    random = qini_coefficient(experiment, "random")
    backwards = qini_coefficient(experiment, "backwards")
    assert perfect > abs(random) * 5
    assert backwards < 0


def test_uplift_at_full_population_is_naive_difference(experiment):
    t = experiment[experiment["treated"] == 1]["churned_6m"].mean()
    c = experiment[experiment["treated"] == 0]["churned_6m"].mean()
    assert uplift_at_k(experiment, "random", k=1.0) == pytest.approx(c - t)
    assert uplift_at_k(experiment, "perfect", k=0.5) == pytest.approx(0.25, abs=0.03)


def test_bootstrap_interval_contains_estimate(experiment):
    table = uplift_by_bin(experiment, "perfect", 4)
    ci = bootstrap_uplift_by_bin(experiment, "perfect", 4, n_boot=100)
    assert ((ci["lower"] <= table["actual_uplift"]) & (table["actual_uplift"] <= ci["upper"])).all()


def test_churn_metrics_and_gains_table():
    from uplift_series.evaluation import churn_metrics, gains_table

    rng = np.random.default_rng(1)
    y = (rng.random(5_000) < 0.1).astype(int)
    p = np.clip(0.1 + 0.3 * y + rng.normal(0, 0.15, 5_000), 0.001, 0.999)
    m = churn_metrics(y, p)
    assert 0.8 < m["roc_auc"] <= 1
    assert m["gini"] == pytest.approx(2 * m["roc_auc"] - 1)
    assert m["pr_auc"] > m["base_rate"]
    g = gains_table(y, p)
    assert g["recall_tpr"].iloc[-1] == pytest.approx(1.0)
    assert g["fpr"].iloc[-1] == pytest.approx(1.0)
    assert g["tpr_minus_fpr"].max() <= m["ks"] + 1e-9
