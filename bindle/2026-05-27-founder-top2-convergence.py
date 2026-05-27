import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell
def import_std():
    import pathlib

    return (pathlib,)


@app.cell
def import_pkg():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import requests
    import seaborn as sns
    from teeplot import teeplot as tp
    from watermark import watermark

    return mo, np, pd, plt, requests, sns, tp, watermark


@app.cell(hide_code=True)
def do_watermark(mo, watermark):
    mo.md(
        f"""
    ```Text
    {watermark(
        current_date=True,
        iso8601=True,
        machine=True,
        updated=True,
        python=True,
        iversions=True,
        globals_=globals(),
    )}
    ```
    """
    )
    return


@app.cell(hide_code=True)
def delimit_data(mo):
    mo.md("""
    ## Data

    Load the per-replicate Hamming-weight time series produced by the
    founder-sweep slurm job (`slurm/2026-05-20/2026-05-20-founder.sh`,
    notebook `bindle/2026-05-20-founder.py`), cached as a parquet on
    OSF. The sweep covers four `N_SITES` conditions
    (`2, 3, 4, 5`) at 20 replicates each, 5000 steps per replicate,
    POP_SIZE=100_000, on CPU (engine=numpy). The dataframe has one
    row per `(replicate_uid, Step, hw)` with the per-step number of
    cases (`n_cases`) at each Hamming weight, alongside the
    simulation parameters as constant-valued columns.

    The OSF slug is downloaded with `requests` and cached at
    `/tmp/<slug>` so re-runs hit the local copy.
    """)
    return


@app.cell
def configure_args(mo):
    # CLI args. Defaults pull the founder-sweep hw parquet that
    # backs this notebook.
    _args = mo.cli_args()
    OSF_SLUG = str(_args.get("osf-slug") or "xzusj")
    OSF_URL = str(
        _args.get("osf-url") or f"https://osf.io/{OSF_SLUG}/download",
    )
    print(f"args: OSF_SLUG={OSF_SLUG} OSF_URL={OSF_URL}")
    return OSF_SLUG, OSF_URL


@app.cell
def download_data(OSF_SLUG, OSF_URL, pathlib, pd, requests):
    cache_path = pathlib.Path("/tmp") / OSF_SLUG
    if not cache_path.exists():
        print(f"downloading {OSF_URL} -> {cache_path}")
        resp = requests.get(OSF_URL, allow_redirects=True, timeout=120)
        resp.raise_for_status()
        cache_path.write_bytes(resp.content)
    else:
        print(f"reusing cached {cache_path}")
    print(f"size: {cache_path.stat().st_size} bytes")

    hw_df = pd.read_parquet(cache_path)
    print(f"loaded hw dataframe: {hw_df.shape}")
    print(
        "n_sites x replicate counts:\n"
        + str(hw_df.groupby("n_sites")["replicate_uid"].nunique()),
    )
    return (hw_df,)


@app.cell(hide_code=True)
def delimit_analysis(mo):
    mo.md("""
    ## Top-2 Convergence Hamming Weights

    For each replicate, locate the run's last simulation step and
    identify the **top two Hamming-weight bins** by case count
    (`n_cases`) at that step. The Hamming-weight bin is used as a
    coarse strain identifier --- with only the aggregated `hw`
    dataframe in hand, each bin stands in for the set of genomes
    sharing a given mutational distance from the founder
    (all-zero wildtype) strain.

    We then tabulate, separately for each `N_SITES` condition, the
    **fraction of replicates** whose end-state top-two pair falls at
    each `(hw_top1, hw_top2)` combination. Convergence to a single
    Hamming weight shows up as mass on the `hw_top1 == hw_top2`
    diagonal; divergent convergence to two distinct evolutionary
    clusters shows up as off-diagonal mass.
    """)
    return


@app.cell
def compute_top2(hw_df, pd):
    # Use the final step of each replicate (n_steps is constant within a
    # replicate, but compute per-replicate to be safe).
    last_steps = hw_df.groupby("replicate_uid")["Step"].transform("max")
    last = hw_df[hw_df["Step"] == last_steps].copy()

    def _top2(group):
        ranked = group.sort_values("n_cases", ascending=False)
        ranked = ranked[ranked["n_cases"] > 0]
        if len(ranked) == 0:
            return None
        top1_hw = int(ranked["hw"].iloc[0])
        # If only one Hamming-weight bin is occupied at end (rare given
        # extinction prevention guarantees at least one circulating
        # case, but the founder allele can be the sole survivor), fall
        # back to top1 so the pair stays well-defined.
        top2_hw = int(ranked["hw"].iloc[1]) if len(ranked) >= 2 else top1_hw
        return top1_hw, top2_hw

    rows = []
    for (_uid, _n_sites), _sub in last.groupby(
        ["replicate_uid", "n_sites"],
    ):
        _pair = _top2(_sub)
        if _pair is None:
            continue
        rows.append(
            {
                "replicate_uid": _uid,
                "n_sites": int(_n_sites),
                "hw_top1": _pair[0],
                "hw_top2": _pair[1],
            },
        )

    top2_df = pd.DataFrame(rows)
    print(f"top-2 frame: {top2_df.shape}")
    print(
        "replicates per n_sites:\n" + str(top2_df.groupby("n_sites").size()),
    )
    return (top2_df,)


@app.cell
def compute_fractions(top2_df):
    # Fraction of replicates per (n_sites, hw_top1, hw_top2) cell.
    counts = (
        top2_df.groupby(["n_sites", "hw_top1", "hw_top2"])
        .size()
        .rename("n_reps")
        .reset_index()
    )
    totals = top2_df.groupby("n_sites").size().rename("n_total").reset_index()
    frac_df = counts.merge(totals, on="n_sites")
    frac_df["fraction"] = frac_df["n_reps"] / frac_df["n_total"]
    print(frac_df.to_string(index=False))
    return (frac_df,)


@app.cell(hide_code=True)
def delimit_plot(mo):
    mo.md("""
    ## Fraction-of-Runs Heatmap

    One panel per `N_SITES` condition. Axes are the top-two Hamming
    weights at the final step (`hw_top1` = most-populous bin,
    `hw_top2` = second-most). Cell color encodes the fraction of
    replicates (out of 20) whose end state lands in that pair; cells
    with no replicates are masked white. The fraction-as-text overlay
    makes it easy to read off counts directly.
    """)
    return


@app.cell
def plot_fraction_heatmap(frac_df, np, pathlib, plt, sns, tp):
    n_sites_vals = sorted(frac_df["n_sites"].unique().tolist())
    n_panels = len(n_sites_vals)

    with tp.teed(
        plt.subplots,
        nrows=1,
        ncols=n_panels,
        figsize=(3.6 * n_panels, 3.6),
        squeeze=False,
        teeplot_outattrs={"a": "top2-hw-fraction"},
        teeplot_show=True,
        teeplot_subdir=pathlib.Path(__file__).stem,
    ) as (fig, axes):
        axes = axes[0]
        for _ax, _ns in zip(axes, n_sites_vals):
            _sub = frac_df[frac_df["n_sites"] == _ns]
            _hw_max = int(_ns)
            grid = np.full((_hw_max + 1, _hw_max + 1), np.nan)
            for _, _row in _sub.iterrows():
                grid[int(_row["hw_top2"]), int(_row["hw_top1"])] = float(
                    _row["fraction"],
                )
            _masked = np.ma.masked_invalid(grid)
            _cmap = plt.get_cmap("rocket_r").copy()
            _cmap.set_bad("white")
            _im = _ax.imshow(
                _masked,
                origin="lower",
                cmap=_cmap,
                vmin=0.0,
                vmax=1.0,
                aspect="equal",
                interpolation="nearest",
            )
            for _i in range(_hw_max + 1):
                for _j in range(_hw_max + 1):
                    if not np.ma.is_masked(_masked[_i, _j]):
                        _ax.text(
                            _j,
                            _i,
                            f"{_masked[_i, _j]:.2f}",
                            ha="center",
                            va="center",
                            fontsize=8,
                            color=(
                                "white" if _masked[_i, _j] > 0.5 else "black"
                            ),
                        )
            _ax.set_xticks(range(_hw_max + 1))
            _ax.set_yticks(range(_hw_max + 1))
            _ax.set_xlabel("hw_top1 (most-populous bin)")
            _ax.set_ylabel("hw_top2 (second-most)")
            _ax.set_title(f"N_SITES = {_ns}")
            sns.despine(ax=_ax, left=True, bottom=True)
        cbar = fig.colorbar(
            _im,
            ax=axes,
            pad=0.02,
            shrink=0.85,
            label="fraction of replicates",
        )
        _ = cbar
    return


@app.cell(hide_code=True)
def delimit_diff(mo):
    mo.md("""
    ## Stacked Histogram by Top-2 Hamming-Weight Difference

    Per-replicate **Hamming-weight gap** between the top-two end-state
    bins, `hw_diff = |hw_top1 - hw_top2|`. A gap of 0 means the
    replicate converged to a single dominant Hamming weight (all
    sub-dominant cases share the same weight); a gap of 1 means the
    second-most-populous bin sits one mutation away from the dominant
    one; larger gaps mean the top-two cluster pair spans further
    apart. We plot the **percent of replicates** (out of 20) at each
    `hw_diff` value, stacked, per `N_SITES` condition.
    """)
    return


@app.cell
def plot_diff_stacked_hist(np, pathlib, plt, sns, top2_df, tp):
    # Possible Hamming-weight differences range from 0 to max(N_SITES)
    # since each top bin's hw is in [0, N_SITES]. Set the hue order
    # explicitly so stack ordering and the legend stay consistent
    # across N_SITES conditions (smaller diffs at the bottom).
    _max_n_sites = int(top2_df["n_sites"].max())
    _hue_order = list(range(_max_n_sites + 1))

    _diff_df = top2_df.assign(
        hw_diff=(top2_df["hw_top1"] - top2_df["hw_top2"]).abs(),
    )
    _totals = _diff_df.groupby("n_sites").size()
    _pct_wide = (
        _diff_df.groupby(["n_sites", "hw_diff"]).size().unstack(fill_value=0)
    )
    _pct_wide = _pct_wide.div(_totals, axis=0).mul(100.0)
    _pct_wide = _pct_wide.reindex(columns=_hue_order, fill_value=0.0)
    print("percent of replicates by (n_sites, hw_diff):")
    print(_pct_wide.round(1).to_string())

    _palette = sns.color_palette("rocket_r", n_colors=len(_hue_order))
    _n_conditions = len(_pct_wide.index)
    _bar_x = np.arange(_n_conditions)

    with tp.teed(
        plt.subplots,
        figsize=(1.4 * _n_conditions + 2.0, 4.2),
        teeplot_outattrs={"a": "top2-hw-diff-stacked-pct"},
        teeplot_show=True,
        teeplot_subdir=pathlib.Path(__file__).stem,
    ) as (_fig, _ax):
        _bottom = np.zeros(_n_conditions, dtype=float)
        for _hw_diff, _color in zip(_hue_order, _palette):
            _heights = _pct_wide[_hw_diff].to_numpy()
            _ax.bar(
                _bar_x,
                _heights,
                bottom=_bottom,
                color=_color,
                label=f"diff = {_hw_diff}",
                edgecolor="white",
                linewidth=0.5,
            )
            _bottom = _bottom + _heights
        _ax.set_xticks(_bar_x)
        _ax.set_xticklabels([str(int(_ns)) for _ns in _pct_wide.index])
        _ax.set_xlabel("N_SITES")
        _ax.set_ylabel("% of replicates")
        _ax.set_ylim(0, 100)
        # `reverse=True` (matplotlib >= 3.7) shows the largest hw_diff
        # entry at the top of the legend, matching the stack top-down.
        _handles, _labels = _ax.get_legend_handles_labels()
        _ax.legend(
            _handles[::-1],
            _labels[::-1],
            title="|hw_top1 - hw_top2|",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            handletextpad=0.4,
        )
        sns.despine(ax=_ax)
    return


@app.cell(hide_code=True)
def delimit_tables(mo):
    mo.md("""
    ## Per-N_SITES Outcome Tables

    One table per `N_SITES` condition listing every observed
    end-state `(hw_top1, hw_top2)` pair with the percent of
    replicates that converged there and the Hamming-weight gap
    `hw_diff`. Rows are sorted by decreasing percent.
    """)
    return


@app.cell
def show_tables(frac_df, mo):
    _pct_df = frac_df.assign(
        hw_diff=(frac_df["hw_top1"] - frac_df["hw_top2"]).abs(),
        pct=(frac_df["fraction"] * 100.0).round(2),
    )[
        [
            "n_sites",
            "hw_top1",
            "hw_top2",
            "hw_diff",
            "n_reps",
            "n_total",
            "pct",
        ]
    ]

    _panels = []
    for _ns in sorted(_pct_df["n_sites"].unique().tolist()):
        _sub = (
            _pct_df[_pct_df["n_sites"] == _ns]
            .sort_values("pct", ascending=False)
            .reset_index(drop=True)
        )
        print(f"\nN_SITES = {int(_ns)}")
        print(_sub.to_string(index=False))
        _panels.append(mo.md(f"### N_SITES = {int(_ns)}"))
        _panels.append(mo.ui.table(_sub, selection=None))
    mo.vstack(_panels)
    return


if __name__ == "__main__":
    app.run()
