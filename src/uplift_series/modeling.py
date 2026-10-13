"""Splitting the data and estimating uplift.

Two classic approaches, both built on XGBoost classifiers:

- TLearner ("two models"): one churn model trained only on treated clients,
  one trained only on control clients. Uplift is the difference between their
  predictions. Easy to explain: one model per parallel world.
- SLearner ("single model"): one churn model with the treatment flag as an
  extra input, scored twice per client (flag on, flag off).

Uplift is always defined as

    uplift = P(churn | no action) - P(churn | pricing)

so a positive value means pricing helps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

from uplift_series.data import FEATURES

#: Deliberately simple settings: shallow trees suit a treated group of only
#: ~700 training clients. Used for every model in the series unless stated.
DEFAULT_XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=2,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10,
    eval_metric="logloss",
    random_state=0,
)


def split_pool(pool: pd.DataFrame, test_size: float = 0.3, seed: int = 0):
    """Split into train and test, keeping the treated share equal in both.

    Stratifying on ``treated`` guarantees the test set holds both treated
    and control clients in the same ratio, which validation relies on.
    """
    train, test = train_test_split(pool, test_size=test_size, stratify=pool["treated"], random_state=seed)
    return train.copy(), test.copy()


def _classifier(params: dict | None) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(**(params or DEFAULT_XGB_PARAMS))


class TLearner:
    """Two models: one per parallel world."""

    def __init__(self, features: list[str] | None = None, params: dict | None = None):
        self.features = features or FEATURES
        self.params = params
        self.model_treated = None   # learns P(churn | pricing)
        self.model_control = None   # learns P(churn | no action)

    def fit(self, data: pd.DataFrame, treatment: str = "treated", outcome: str = "churned_6m",
            sample_weight: str | None = None) -> "TLearner":
        treated = data[data[treatment] == 1]
        control = data[data[treatment] == 0]
        w_t = treated[sample_weight] if sample_weight else None
        w_c = control[sample_weight] if sample_weight else None
        self.model_treated = _classifier(self.params).fit(
            treated[self.features], treated[outcome], sample_weight=w_t
        )
        self.model_control = _classifier(self.params).fit(
            control[self.features], control[outcome], sample_weight=w_c
        )
        return self

    def predict_churn_no_action(self, data: pd.DataFrame) -> np.ndarray:
        return self.model_control.predict_proba(data[self.features])[:, 1]

    def predict_churn_with_pricing(self, data: pd.DataFrame) -> np.ndarray:
        return self.model_treated.predict_proba(data[self.features])[:, 1]

    def predict_uplift(self, data: pd.DataFrame) -> np.ndarray:
        """No-action churn minus pricing churn. Positive means pricing helps."""
        return self.predict_churn_no_action(data) - self.predict_churn_with_pricing(data)


class SLearner:
    """One model with the treatment flag as an input, scored twice."""

    def __init__(self, features: list[str] | None = None, params: dict | None = None):
        self.features = features or FEATURES
        self.params = params
        self.model = None

    def _inputs(self, data: pd.DataFrame, flag: int) -> pd.DataFrame:
        return data[self.features].assign(treated=flag)

    def fit(self, data: pd.DataFrame, treatment: str = "treated", outcome: str = "churned_6m",
            sample_weight: str | None = None) -> "SLearner":
        X = data[self.features].assign(treated=data[treatment].to_numpy())
        w = data[sample_weight] if sample_weight else None
        self.model = _classifier(self.params).fit(X, data[outcome], sample_weight=w)
        return self

    def predict_churn_no_action(self, data: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._inputs(data, 0))[:, 1]

    def predict_churn_with_pricing(self, data: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._inputs(data, 1))[:, 1]

    def predict_uplift(self, data: pd.DataFrame) -> np.ndarray:
        return self.predict_churn_no_action(data) - self.predict_churn_with_pricing(data)
