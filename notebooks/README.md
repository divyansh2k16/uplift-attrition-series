# Notebooks

One notebook per article, numbered to match the series:

| Notebook | Part |
|---|---|
| `01_what_is_uplift_modelling.ipynb` | 1. What is uplift modelling? |
| `02_four_kinds_of_clients.ipynb` | 2. The four kinds of clients |
| `03_data_you_need.ipynb` | 3. The data you need |
| `04_two_models_two_worlds.ipynb` | 4. Two models, two parallel worlds |
| `05_grading_uplift.ipynb` | 5. How do you grade a prediction you can never check? |
| `06_why_auc_wont_save_you.ipynb` | 6. Why AUC, KS and log loss won't save you here |
| `07_uplift_to_dollars.ipynb` | 7. From uplift to dollars |
| `08_causal_inference_101.ipynb` | 8. Causal inference 101 for bankers |
| `09_propensity_scores.ipynb` | 9. Propensity scores without the mystery |
| `10_holdouts_hidden_bias_pipeline.ipynb` | 10. Holdouts, hidden bias and the full pipeline |

Figures are saved to `figures/`. Every notebook runs top to bottom in a few
minutes at most.

Each notebook imports from `uplift_series` rather than repeating code, so
install the package first (`pip install -e .` from the repo root).
