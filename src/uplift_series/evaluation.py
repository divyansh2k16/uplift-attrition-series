"""Grading an uplift model without knowing the truth.

Uplift can never be observed for one client, because each client is either
treated or not. It *can* be observed for a group: among similar clients,
compare the churn rate of those who got pricing with those who did not.
Every function here builds on that idea.

Conventions:
- ``score`` is the model's predicted uplift; higher means "call first".
- The outcome is churn, which is bad, so actual uplift is
  control churn rate minus treated churn rate.
- ``weight`` (optional) is a column of sample weights. It is unused in Act 2
  and becomes the propensity weights in Act 3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rate(y: pd.Series, w: pd.Series | None) -> float:
    if len(y) == 0:
        return np.nan
    if w is None:
        return float(y.mean())
    return float(np.average(y, weights=w))


def assign_bins(data: pd.DataFrame, score: str, n_bins: int = 10) -> pd.Series:
    """Bin 1 holds the highest scores, bin ``n_bins`` the lowest. Equal sizes."""
    ranks = data[score].rank(method="first", ascending=False)
    return pd.qcut(ranks, n_bins, labels=range(1, n_bins + 1)).astype(int)


def uplift_by_bin(data: pd.DataFrame, score: str, n_bins: int = 10, treatment: str = "treated",
                  outcome: str = "churned_6m", weight: str | None = None,
                  bins: pd.Series | None = None) -> pd.DataFrame:
    """Actual uplift in each bin of clients ranked by predicted uplift.

    Returns one row per bin with the treated and control churn rates, the
    actual uplift (control minus treated) and the average predicted uplift.
    """
    bins = assign_bins(data, score, n_bins) if bins is None else bins
    rows = []
    for b, group in data.groupby(bins, sort=True):
        t = group[group[treatment] == 1]
        c = group[group[treatment] == 0]
        w_t = t[weight] if weight else None
        w_c = c[weight] if weight else None
        treated_churn = _rate(t[outcome], w_t)
        control_churn = _rate(c[outcome], w_c)
        rows.append({
            "bin": b,
            "clients": len(group),
            "treated": len(t),
            "control": len(c),
            "treated_churn": treated_churn,
            "control_churn": control_churn,
            "actual_uplift": control_churn - treated_churn,
            "predicted_uplift": float(group[score].mean()),
        })
    return pd.DataFrame(rows).set_index("bin")


def bootstrap_uplift_by_bin(data: pd.DataFrame, score: str, n_bins: int = 10, n_boot: int = 300,
                            seed: int = 0, treatment: str = "treated", outcome: str = "churned_6m",
                            ci: float = 0.9) -> pd.DataFrame:
    """How much each bin's actual uplift wobbles from sampling alone.

    Bins are fixed from the full data; clients are then resampled with
    replacement and each bin's actual uplift recomputed. Returns the lower
    and upper bounds of the ``ci`` interval per bin.
    """
    rng = np.random.default_rng(seed)
    bins = assign_bins(data, score, n_bins)
    t = data[treatment].to_numpy()
    y = data[outcome].to_numpy()
    b = bins.to_numpy()
    n = len(data)
    draws = np.full((n_boot, n_bins), np.nan)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        tb, yb, bb = t[idx], y[idx], b[idx]
        for j in range(1, n_bins + 1):
            in_bin = bb == j
            treated = in_bin & (tb == 1)
            control = in_bin & (tb == 0)
            if treated.any() and control.any():
                draws[i, j - 1] = yb[control].mean() - yb[treated].mean()
    alpha = (1 - ci) / 2
    return pd.DataFrame({
        "lower": np.nanquantile(draws, alpha, axis=0),
        "upper": np.nanquantile(draws, 1 - alpha, axis=0),
    }, index=pd.Index(range(1, n_bins + 1), name="bin"))


def qini_curve(data: pd.DataFrame, score: str, treatment: str = "treated", outcome: str = "churned_6m",
               weight: str | None = None) -> pd.DataFrame:
    """Extra clients kept if the top fraction of clients (by score) were treated.

    At each cut-off, the Qini value is

        control churners x (treated count / control count) - treated churners

    i.e. how many more treated clients would have left at the control
    group's churn rate. Returns ``fraction`` (share of clients targeted) and
    ``qini`` (extra clients kept, in treated-group units), starting at (0, 0).
    """
    d = data.sort_values(score, ascending=False)
    t = d[treatment].to_numpy()
    y = d[outcome].to_numpy()
    w = d[weight].to_numpy() if weight else np.ones(len(d))
    n_t = np.cumsum(w * t)
    n_c = np.cumsum(w * (1 - t))
    churn_t = np.cumsum(w * t * y)
    churn_c = np.cumsum(w * (1 - t) * y)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(n_c > 0, churn_c * n_t / n_c - churn_t, 0.0)
    fraction = np.arange(1, len(d) + 1) / len(d)
    return pd.DataFrame({"fraction": np.r_[0.0, fraction], "qini": np.r_[0.0, q]})


def qini_coefficient(data: pd.DataFrame, score: str, treatment: str = "treated",
                     outcome: str = "churned_6m", weight: str | None = None) -> float:
    """Area between the model's Qini curve and the random-targeting line.

    Positive means the ranking beats random targeting; zero means no better
    than random; negative means worse. Use it to compare models on the same
    test set (its scale depends on the test set's size).
    """
    curve = qini_curve(data, score, treatment, outcome, weight)
    random_line = curve["fraction"] * curve["qini"].iloc[-1]
    return float(np.trapezoid(curve["qini"] - random_line, curve["fraction"]))


def uplift_at_k(data: pd.DataFrame, score: str, k: float = 0.2, treatment: str = "treated",
                outcome: str = "churned_6m", weight: str | None = None) -> float:
    """Actual uplift (control minus treated churn) among the top ``k`` share."""
    top = data.nlargest(max(1, int(round(k * len(data)))), score)
    t = top[top[treatment] == 1]
    c = top[top[treatment] == 0]
    return _rate(c[outcome], c[weight] if weight else None) - _rate(t[outcome], t[weight] if weight else None)


def churn_metrics(y_true, p_churn) -> dict:
    """Standard health checks for a churn (not uplift) model.

    - roc_auc: probability a random churner is scored above a random stayer
    - pr_auc: average precision; compare it with ``base_rate``, not with 0.5
    - ks: the largest gap between the share of churners and the share of
      stayers captured at any cut-off (max of TPR - FPR)
    - gini: 2 x roc_auc - 1, common in banking scorecards
    - log_loss: binary cross-entropy; compare with ``base_log_loss``, the
      score of always predicting the base rate
    """
    from sklearn.metrics import average_precision_score, log_loss, roc_auc_score, roc_curve

    y = np.asarray(y_true)
    p = np.asarray(p_churn)
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)
    base = y.mean()
    return {
        "roc_auc": float(auc),
        "gini": float(2 * auc - 1),
        "ks": float(np.max(tpr - fpr)),
        "pr_auc": float(average_precision_score(y, p)),
        "base_rate": float(base),
        "log_loss": float(log_loss(y, p)),
        "base_log_loss": float(log_loss(y, np.full(len(y), base))),
    }


def gains_table(y_true, p_churn, n_bins: int = 10) -> pd.DataFrame:
    """Cumulative gains by decile of a churn score.

    Each row is a cut-off: "flag everyone in this decile and above". From
    it come recall (TPR), false positive rate, precision, lift and the
    TPR - FPR gap whose maximum approximates KS.
    """
    d = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(p_churn)})
    d["bin"] = assign_bins(d, "p", n_bins)
    g = d.groupby("bin")["y"].agg(clients="count", churners="sum")
    total_churners = d["y"].sum()
    total_stayers = len(d) - total_churners
    g["churners_in_bin_rate"] = g["churners"] / g["clients"]
    cum_clients = g["clients"].cumsum()
    cum_churners = g["churners"].cumsum()
    g["recall_tpr"] = cum_churners / total_churners
    g["fpr"] = (cum_clients - cum_churners) / total_stayers
    g["cum_precision"] = cum_churners / cum_clients
    g["cum_lift"] = g["cum_precision"] / d["y"].mean()
    g["tpr_minus_fpr"] = g["recall_tpr"] - g["fpr"]
    return g
