# Who Will Stay Because of Us?

### Uplift modelling and causal inference for client attrition

Code for a 10-part Medium series. It starts with the simplest questions about
uplift modelling and works up to causal inference, all set in a synthetic
institutional bank where relationship managers (RMs) decide which at-risk
clients get a pricing negotiation.

> **New to churn or attrition modelling?** The series assumes you know what a
> churn model does. A good primer:
> [Causal Machine Learning for Customer Retention (Towards Data Science)](https://towardsdatascience.com/causal-machine-learning-for-customer-retention-a-practical-guide-with-python-6bd959b25741/).

## The series

| Part | Title | Notebook | Status |
|---|---|---|---|
| **Act 1: Foundations** | | | |
| 1 | What is uplift modelling? Your churn model is answering the wrong question | [`01`](notebooks/01_what_is_uplift_modelling.ipynb) | Notebook ready |
| 2 | The four kinds of clients, and why calling the riskiest ones wastes effort | [`02`](notebooks/02_four_kinds_of_clients.ipynb) | Notebook ready |
| **Act 2: Building on clean data** | | | |
| 3 | The data you need: treated, control and the start date | [`03`](notebooks/03_data_you_need.ipynb) | Notebook ready |
| 4 | Two models, two parallel worlds: your first uplift model | [`04`](notebooks/04_two_models_two_worlds.ipynb) | Notebook ready |
| 5 | How do you grade a prediction you can never check? | `05_` | Planned |
| 6 | Why AUC, KS and log loss won't save you here | `06_` | Planned |
| 7 | From uplift to dollars: stop discounting clients who were never leaving | `07_` | Planned |
| **Act 3: Causal inference in banking** | | | |
| 8 | Causal inference 101 for bankers: why your data is lying | `08_` | Planned |
| 9 | Levelling the playing field: propensity scores without the mystery | `09_` | Planned |
| 10 | The experiment nobody wanted to run: holdouts, hidden bias and the full pipeline | `10_` | Planned |

Article links will be added as each part is published.

## The one idea behind the code

Uplift, the change in a client's churn risk caused by an action, can never be
observed for a real client. So the series uses a synthetic bank where the true
uplift of every client is known. Every notebook can then check whether a
method gets the right answer.

The generator has a switch:

- `assignment="random"`: pricing offered at random (Act 2). Simple methods work.
- `assignment="rm_biased"`: RMs pick the easy clients (Act 3). The naive
  estimate roughly doubles the true effect, and causal inference is needed to
  correct it.

| Series seed | Random | RM-biased |
|---|---|---|
| Naive uplift | 7.6 points | 14.7 points |
| True uplift | 7.7 points | 7.7 points |

And the Act 1 punchline: with 500 calls, calling the riskiest clients first
saves about 10 clients, picking at random saves about 35, and calling the
clients with the highest true uplift saves about 136.

Full details are in [docs/data_spec.md](docs/data_spec.md).

## Quick start

```bash
git clone https://github.com/divyansh2k16/uplift-attrition-series.git
cd uplift-attrition-series
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest                          # checks the generator
python scripts/sanity_report.py  # prints the headline numbers
```

```python
from uplift_series import generate_bank, analysis_pool

df = generate_bank(assignment="rm_biased")
pool = analysis_pool(df)   # flagged clients who got pricing or nothing
```

## Repository layout

```
├── docs/data_spec.md        how the synthetic bank is generated, and why
├── src/uplift_series/       reusable code: data generator, plotting style,
│                            targeting, uplift models (evaluation and
│                            propensity code added part by part)
├── notebooks/               one notebook per article
├── scripts/sanity_report.py headline numbers for the default data
├── tests/                   checks that the generator tells the truth
└── figures/                 charts used in the articles
```

## Disclaimer

All data, clients and figures in this repository are synthetic and invented
for teaching. Nothing here describes or is derived from any real
institution's data, models or results.

## Licence

MIT. See [LICENSE](LICENSE).
