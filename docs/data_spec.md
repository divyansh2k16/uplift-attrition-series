# Data specification: the synthetic bank

Every article in the series runs on one synthetic dataset produced by
`uplift_series.data.generate_bank()`. This document explains what is in it,
how it is generated, and why it is built that way.

All clients, figures and behaviours are invented. The data is not derived
from, and does not describe, any real bank or client.

## Why synthetic data

Uplift can never be observed for a real client: each client either received
the intervention or did not. With synthetic data we decide the true effect
for every client ourselves, so each notebook can check whether a method
recovers the truth. That is the main reason this series is reproducible and
checkable.

## The setting

A bank serves 40,000 institutional clients through relationship managers
(RMs). An existing churn model scores every client each cycle, and the
riskiest are sent to RMs. RMs can offer a **pricing negotiation** (the
treatment studied in the series) or a different intervention (a **service
fix**). The outcome is whether the client leaves within six months.

## Pipeline

| Step | What happens | Default |
|---|---|---|
| 1 | Generate client features | 40,000 clients |
| 2 | Baseline churn probability with no action, `p0` | average 8% over six months |
| 3 | Pricing changes churn on the log-odds scale, client by client | creates four client types |
| 4 | Existing churn model flags the riskiest clients | top 6,000 |
| 5 | Interventions among flagged clients | 1,000 service fix, about 1,000 pricing |
| 6 | Outcomes drawn from the matching probability | `churned_6m` |

## Features (observed, measured before the start date)

| Column | Meaning | How it is generated |
|---|---|---|
| `segment` | large_corporate, mid_market, commercial | 15% / 35% / 50% |
| `revenue_ttm` | Trailing 12-month revenue (USD) | log-normal; medians 4M / 700K / 150K by segment |
| `tenure_years` | Years as a client | gamma, plus a nudge from hidden relationship strength |
| `n_products` | Products held | Poisson, rises with tenure and size |
| `txn_volume_trend` | Change in transaction volume, last 6 months vs prior 6 | normal(0, 0.15) |
| `pricing_gap` | How much dearer the bank is than peers (0.05 = 5% dearer) | normal(0.03, 0.07) |
| `rm_meetings_6m` | RM meetings in the last 6 months | Poisson, rises with hidden relationship strength |
| `complaints_6m` | Complaints in the last 6 months | Poisson(0.4) |
| `corporate_event` | Merger or banking consolidation under way | 3% of clients |
| `years_since_price_review` | Years since pricing was last reviewed | 0 to 8 |
| `segment_large_corporate`, `segment_mid_market` | One-hot segment flags | from `segment` |

`uplift_series.data.FEATURES` lists the columns a model may use.

Other observed columns: `client_id`, `churn_score` (the existing churn
model's score), `flagged`, `start_date` (quarterly cycle dates in 2025),
`treatment` (`none`, `pricing` or `service_fix`) and `churned_6m`.

## Baseline churn (`p0`)

On the log-odds scale, churn risk rises with a higher pricing gap, falling
transaction volume, complaints and corporate events, and falls with more
products, longer tenure, larger revenue, more RM meetings and a stronger
hidden relationship. An intercept is calibrated so that average `p0` is 8%.

## The effect of pricing and the four client types

A pricing negotiation lowers the log-odds of churn by an amount that depends
on the client:

- **Price-sensitive clients** (high `pricing_gap`): up to a 2.2 log-odds
  reduction. These are mostly **persuadables**.
- **Corporate event clients**: almost no effect, because no price reverses a
  merger decision. Combined with high risk, these are **lost causes**.
- **Sleeping dogs**: clients priced at or below peers, with no complaints and
  no price review for five or more years. Raising pricing prompts them to run
  an RFP, so churn *rises* (an effect of minus 1 log-odds).
- Everyone else with low risk and little effect is a **sure thing**.

Every client's effect also carries random variation (normal, standard
deviation 0.35 log-odds) that the features do not explain, so the data looks
like real clients rather than a clean formula.

`true_uplift = p0 - p1`, where `p1` is churn probability after pricing.
Labels in `true_client_type` follow these rules:

| Type | Rule |
|---|---|
| persuadable | uplift of at least 5 points |
| sleeping_dog | uplift of minus 2 points or lower |
| lost_cause | otherwise, and `p0` of 30% or more |
| sure_thing | everything else |

A **service fix** lowers churn a little for every client who gets one, and
most for clients who have been complaining. It is there so Part 3 can show
why those clients must not be used as controls.

## Treatment assignment: the switch behind the whole series

`assignment="random"` (Act 2): every eligible flagged client has the same
chance of getting pricing. This is a clean experiment, and the naive
comparison of treated and control churn is unbiased.

`assignment="rm_biased"` (Act 3): RMs favour clients they find easy, meaning
long tenure, frequent meetings and large revenue. They avoid complainers and
clients in a corporate event. Those clients would have stayed more often
anyway, so the naive comparison overstates the effect of pricing.

Two parameters control the bias:

- `selection_strength` (default 0.7): how strongly RMs favour easy clients.
- `hidden_confounding` (default 0.5): how strongly an **unobserved**
  relationship strength (think "golf with the treasurer") drives both RM
  choice and churn. Propensity methods using observed features cannot remove
  this part. Set it to 0 for a world where weighting can fully recover the
  truth, or raise it to show the limits of weighting.

## Oracle columns (never use as features)

| Column | Meaning |
|---|---|
| `hidden_relationship` | Unobserved relationship strength |
| `true_p_churn_no_action` | `p0` |
| `true_p_churn_pricing` | `p1` |
| `true_uplift` | `p0 - p1` |
| `true_client_type` | Persuadable, sure thing, lost cause or sleeping dog |
| `true_propensity` | Real probability of being chosen for pricing |

`observed_view(df)` drops them, giving what an analyst would actually see.

## Leakage column (never use as a feature)

`txn_change_after_start` is measured after the start date: clients who left
stopped transacting. It is included on purpose so Part 3 can show how a
leaked feature makes a model look perfect and fail in practice.

## Headline numbers (series seed 54)

Produced by `python scripts/sanity_report.py`. The series uses
`SERIES_SEED = 54` because its numbers sit close to the average over many
seeds; the tests check averages across seeds, not this one draw.

| | Random | RM-biased |
|---|---|---|
| Flagged clients | 6,000 | 6,000 |
| Pricing (treated) | 1,000 | 976 |
| Service fix (excluded) | 1,000 | 1,000 |
| No action (control) | 4,000 | 4,024 |
| Churn among flagged | 20.8% | 21.0% |
| Treated churn | 16.0% | 10.6% |
| Control churn | 23.6% | 25.3% |
| **Naive uplift** | **7.6 points** | **14.7 points** |
| **True average uplift** | **7.7 points** | **7.7 points** |
| True no-action risk, treated vs control | 24.5% vs 23.8% | 17.4% vs 25.5% |

Client types among flagged clients: persuadable 60%, sure thing 18%, lost
cause 13%, sleeping dog 9%.

In the random world, the naive estimate matches the truth. In the RM-biased
world it roughly doubles it, because RMs chose clients whose no-action churn
risk was 17% instead of 26%. That gap is the illusion Part 8 reveals and
Part 9 corrects.

Checks run during development (20 seeds each, propensity estimated with a
logistic regression on observed features, then inverse probability
weighting):

| Setting | Naive | After weighting | Truth |
|---|---|---|---|
| Default (`hidden_confounding=0.5`) | 15.7 | 8.2 | 7.8 |
| No hidden confounding (`0.0`) | 14.6 | 7.4 | 7.5 |
| Strong hidden confounding (`1.5`) | 19.3 | 13.0 | 7.8 |

With the defaults, weighting on observed features removes almost all of the
bias, which is the result Part 9 builds on. Part 10 raises
`hidden_confounding` to show what weighting cannot fix.

## Targeting example used in Part 2 (series seed)

RMs make 500 pricing calls among the 6,000 flagged clients:

| Rule | Expected clients saved | Per 100 calls |
|---|---|---|
| Random | 35 | 7 |
| Riskiest first (churn score) | 10 | 2 |
| Highest true uplift first | 136 | 27 |

The riskiest-first list is 78% lost causes.

## Reference numbers for Parts 3 and 4 (series seed)

Part 3, estimated effect of pricing under three choices of control group
(true effect 7.7 points):

| Control group | Estimate |
|---|---|
| All clients with no pricing | -8.7 points |
| Flagged clients, including service fix | +5.7 points |
| Flagged clients with no action | +7.6 points |

Part 3 leakage demo: a churn model's test ROC-AUC is 0.77 with pre-start
features and 0.996 once the post-start column is added.

Part 4, 300 calls among the 1,500 test clients (T-learner and S-learner use
`DEFAULT_XGB_PARAMS`):

| Rule | Expected clients saved | Share of best possible |
|---|---|---|
| Random | 23 | 36% |
| Riskiest first | 11 | 18% |
| T-learner | 42 | 66% |
| Best possible (true uplift) | 64 | 100% |

## Using the generator

```python
from uplift_series import generate_bank, analysis_pool, observed_view

df = generate_bank(assignment="random")     # Act 2 (series seed by default)
df = generate_bank(assignment="rm_biased")  # Act 3
pool = analysis_pool(df)        # flagged clients with pricing or no action
seen = observed_view(df)        # without oracle columns
live = generate_bank(seed=2026)  # a new cohort for "this cycle" in Part 7
```
