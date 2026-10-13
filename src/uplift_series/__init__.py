"""Code for the series "Who Will Stay Because of Us? Uplift Modelling and
Causal Inference for Client Attrition".

All data is synthetic. See docs/data_spec.md.
"""

from uplift_series.data import (
    FEATURES,
    LEAKY_COLUMNS,
    ORACLE_COLUMNS,
    analysis_pool,
    generate_bank,
    naive_uplift,
    observed_view,
    true_average_uplift,
)

__all__ = [
    "FEATURES",
    "LEAKY_COLUMNS",
    "ORACLE_COLUMNS",
    "analysis_pool",
    "generate_bank",
    "naive_uplift",
    "observed_view",
    "true_average_uplift",
]

__version__ = "0.1.0"
