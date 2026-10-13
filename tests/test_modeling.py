import numpy as np
import pytest

from uplift_series import analysis_pool, generate_bank
from uplift_series.modeling import SLearner, TLearner, split_pool


@pytest.fixture(scope="module")
def split():
    pool = analysis_pool(generate_bank(assignment="random"))
    return split_pool(pool)


def test_split_keeps_treated_share(split):
    train, test = split
    assert len(test) == 1_500
    assert train["treated"].mean() == pytest.approx(test["treated"].mean(), abs=0.005)
    assert not set(train.index) & set(test.index)


@pytest.mark.parametrize("learner", [TLearner, SLearner])
def test_learner_ranks_better_than_chance(split, learner):
    train, test = split
    uplift = learner().fit(train).predict_uplift(test)
    assert uplift.shape == (len(test),)
    test = test.assign(pred=uplift)
    top = test.nlargest(300, "pred")["true_uplift"].mean()
    bottom = test.nsmallest(300, "pred")["true_uplift"].mean()
    assert top > bottom + 0.05
    assert np.corrcoef(uplift, test["true_uplift"])[0, 1] > 0.3


def test_uplift_is_no_action_minus_pricing(split):
    train, test = split
    model = TLearner().fit(train)
    expected = model.predict_churn_no_action(test) - model.predict_churn_with_pricing(test)
    assert np.allclose(model.predict_uplift(test), expected)


def test_sample_weight_is_accepted(split):
    train, test = split
    train = train.assign(w=1.0)
    TLearner().fit(train, sample_weight="w").predict_uplift(test)
    SLearner().fit(train, sample_weight="w").predict_uplift(test)
