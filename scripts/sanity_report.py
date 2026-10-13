"""Print the headline numbers the series relies on.

Run from the repo root:

    python scripts/sanity_report.py

It generates the default cohort in both assignment modes and reports pool
sizes, client types, and naive versus true uplift. If you change the
generator, rerun this and update docs/data_spec.md.
"""

from uplift_series.data import (
    SERIES_SEED,
    analysis_pool,
    generate_bank,
    naive_uplift,
    true_average_uplift,
)


def pct(x):
    return f"{100 * x:5.1f}%"


def report(assignment: str, seed: int = SERIES_SEED) -> None:
    df = generate_bank(assignment=assignment, seed=seed)
    flagged = df[df["flagged"] == 1]
    pool = analysis_pool(df)
    treated = pool[pool["treated"] == 1]
    control = pool[pool["treated"] == 0]

    print(f"\n=== assignment = {assignment!r} (seed {seed}) ===")
    print(f"clients                     {len(df):>7,}")
    print(f"flagged by churn model      {len(flagged):>7,}")
    print(f"  pricing (treated)         {(df['treatment'] == 'pricing').sum():>7,}")
    print(f"  service fix (excluded)    {(df['treatment'] == 'service_fix').sum():>7,}")
    print(f"  no action (control)       {len(control):>7,}")
    print(f"overall 6-month churn       {pct(df['churned_6m'].mean())}")
    print(f"churn among flagged         {pct(flagged['churned_6m'].mean())}")

    print("client types among flagged:")
    for name, share in flagged["true_client_type"].value_counts(normalize=True).items():
        print(f"  {name:<14} {pct(share)}")

    print(f"treated churn               {pct(treated['churned_6m'].mean())}")
    print(f"control churn               {pct(control['churned_6m'].mean())}")
    print(f"naive uplift                {pct(naive_uplift(pool))}")
    print(f"true average uplift         {pct(true_average_uplift(pool))}")
    print(
        "true no-action churn risk   treated "
        f"{pct(treated['true_p_churn_no_action'].mean())} | control "
        f"{pct(control['true_p_churn_no_action'].mean())}"
    )


if __name__ == "__main__":
    report("random")
    report("rm_biased")
