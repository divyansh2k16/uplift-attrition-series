"""Render the article tables as images (Medium's editor has no tables).

Run from the repo root:

    python scripts/make_table_images.py

Images are written to figures/ with names starting with "table_".
"""

import matplotlib.pyplot as plt

from uplift_series import plotting as P

HEADER_FILL = "#e8eef6"
ROW_FILL = ["#fcfcfb", "#f4f3ef"]


def table_image(name, columns, rows, col_widths, title=None, bold_first_col=True, font_size=11):
    P.set_style()
    n_rows = len(rows) + 1
    two_line = any("\n" in str(v) for row in rows for v in row)
    fig_h = (0.62 if two_line else 0.45) * n_rows + (0.5 if title else 0.15)
    fig, ax = plt.subplots(figsize=(sum(col_widths), fig_h))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=columns, colWidths=[w / sum(col_widths) for w in col_widths],
                   cellLoc="left", colLoc="left", loc="upper left", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(font_size)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(P.GRID)
        cell.set_linewidth(0.8)
        cell.PAD = 0.03
        if r == 0:
            cell.set_facecolor(HEADER_FILL)
            cell.set_text_props(weight="bold", color=P.INK)
        else:
            cell.set_facecolor(ROW_FILL[(r - 1) % 2])
            cell.set_text_props(color=P.INK, weight="bold" if (c == 0 and bold_first_col) else "normal")
    if title:
        ax.set_title(title, loc="left", pad=10)
    path = P.save(fig, name)
    plt.close(fig)
    return path


def main():
    # Part 2
    table_image(
        "table_02_four_kinds",
        ["Type", "If RMs do nothing", "If RMs call", "In institutional banking"],
        [
            ["Persuadable", "Likely to leave", "Likely to stay", "Pays well above peers; stays once\npricing is fixed"],
            ["Sure thing", "Stays", "Stays", "Loyal, well priced, operations run\non the bank's platforms"],
            ["Lost cause", "Leaves", "Leaves", "Being acquired by a group that\nbanks elsewhere"],
            ["Sleeping dog", "Stays", "More likely to leave", "Happy and cheaply priced; a pricing\ncall prompts a formal tender"],
        ],
        col_widths=[1.7, 2.2, 2.4, 4.0],
    )
    table_image(
        "table_02_four_client_example",
        ["Client", "Risk if no call", "Risk with call", "Uplift"],
        [
            ["A (lost cause)", "80%", "78%", "0.02"],
            ["B (persuadable)", "40%", "15%", "0.25"],
            ["C (sure thing)", "5%", "5%", "0.00"],
            ["D (sleeping dog)", "10%", "20%", "-0.10"],
        ],
        col_widths=[2.2, 1.7, 1.7, 1.2],
    )
    table_image(
        "table_02_500_calls",
        ["Rule", "Expected clients saved", "Per 100 calls"],
        [
            ["Riskiest first", "about 10", "2"],
            ["Random", "about 35", "7"],
            ["Highest uplift first", "about 136", "27"],
        ],
        col_widths=[2.4, 2.9, 1.7],
    )


if __name__ == "__main__":
    main()
