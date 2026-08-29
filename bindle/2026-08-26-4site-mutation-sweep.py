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
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    import requests
    from scipy import stats as sps
    import seaborn as sns
    from teeplot import teeplot as tp
    from watermark import watermark

    return ds, mo, pc, pd, requests, sns, sps, tp, watermark


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

    Load the per-replicate Hamming-weight **end-state** table produced
    by the 4-site mutation-rate sweep slurm job
    (`slurm/2026-08-26/2026-08-26-4site-mutation-sweep.sh`, driven by
    notebook `bindle/2026-05-20-founder.py`), cached as a parquet on
    OSF (https://osf.io/96r2v). The sweep targets `N_SITES=4` across a
    12-condition geometric grid of `MUTATION_RATE` from `1e-5` down to
    `3e-11` (~2 points per decade) at 25 replicates per condition (300
    replicates planned), 150,000 steps per replicate,
    POP_SIZE=1,000,000, on CPU (engine=numpy).

    This sweep was still in flight when this notebook was written: not
    every planned condition has completed replicates yet, and
    completed conditions can have far fewer than 25 replicates apiece
    --- see the `mutation_rate x replicate counts` diagnostic printed
    below for the live picture, and treat the resulting confidence
    intervals as provisional (they should tighten, and any missing
    conditions fill in, as the sweep continues and this OSF file is
    refreshed).

    Unlike the smaller companion sweeps, the OSF file backing this
    notebook is the **full per-step trajectory table** (one row per
    `(replicate_uid, Step, hw)`) rather than a pre-filtered end-state
    export, so loading it wholesale with `pandas` risks exhausting
    memory. Instead, the final-step filter (`Step == max(Step)`) is
    pushed down through `pyarrow.dataset` before materializing to
    `pandas`, since every completed replicate is recorded densely for
    every step (no checkpointing).

    The OSF file is downloaded with `requests` and cached at
    `/tmp/<slug>` so re-runs hit the local copy.
    """)
    return


@app.cell
def configure_args(mo):
    # CLI args. Defaults pull the 4-site mutation-sweep hw parquet that
    # backs this notebook (OSF https://osf.io/96r2v).
    _args = mo.cli_args()
    OSF_SLUG = str(_args.get("osf-slug") or "96r2v")
    OSF_URL = str(
        _args.get("osf-url") or f"https://osf.io/{OSF_SLUG}/download",
    )
    print(f"args: OSF_SLUG={OSF_SLUG} OSF_URL={OSF_URL}")
    return OSF_SLUG, OSF_URL


@app.cell
def download_data(OSF_SLUG, OSF_URL, pathlib, requests):
    cache_path = pathlib.Path("/tmp") / OSF_SLUG
    if not cache_path.exists():
        print(f"downloading {OSF_URL} -> {cache_path}")
        resp = requests.get(OSF_URL, allow_redirects=True, timeout=300)
        resp.raise_for_status()
        cache_path.write_bytes(resp.content)
    else:
        print(f"reusing cached {cache_path}")
    print(f"size: {cache_path.stat().st_size} bytes")
    return (cache_path,)


@app.cell
def load_data(cache_path, ds, pc):
    # The parquet is the full per-step trajectory table (~75M rows for
    # this sweep), so avoid pd.read_parquet(cache_path) --- loading
    # every step of every replicate into pandas is enough on its own to
    # exhaust memory on a typical runner. Every completed replicate is
    # recorded densely for every step (no checkpointing), so pushing a
    # Step == max(Step) filter down through pyarrow before converting
    # to pandas recovers exactly the end-state rows at a fraction of
    # the memory cost.
    _dataset = ds.dataset(cache_path, format="parquet")
    _max_step = pc.max(_dataset.to_table(columns=["Step"])["Step"]).as_py()
    hw_df = _dataset.to_table(
        filter=(pc.field("Step") == _max_step),
        columns=[
            "hw",
            "n_cases",
            "replicate_uid",
            "mutation_rate",
            "n_sites",
        ],
    ).to_pandas()
    print(f"loaded hw end-state dataframe: {hw_df.shape}")
    print(
        "mutation_rate x replicate counts:\n"
        + str(hw_df.groupby("mutation_rate")["replicate_uid"].nunique()),
    )
    return (hw_df,)


@app.cell(hide_code=True)
def delimit_outcome(mo):
    mo.md("""
    ## Fixation Outcome per Replicate

    For each replicate, identify the **dominant Hamming-weight class**
    at its final simulation step by case count (`n_cases`) --- the
    Hamming-weight bin that has "fixed" as the most-populous strain
    cluster at end-state, following the coarse strain identifier used
    in the founder-convergence analyses.

    With `N_SITES=4` there are five Hamming-weight classes drawn from
    the sixteen possible genomes:

    - **founder strain (0000/1111)**: Hamming weights `0` and `4`,
      i.e. the all-zero founder/wildtype genome and its bitwise
      complement (`hw in {0, 4}`, 2 of the 16 genomes).
    - **HW 1/2/3 intermediate**: Hamming weights `1`, `2`, and `3`,
      the one-, two-, and three-mutation classes (`hw in {1, 2, 3}`,
      the other 14 of the 16 genomes) --- `hw` `1` and `3` are bitwise
      complements of one another, and `hw` `2` is self-complementary.

    We classify each replicate's end-state into one of these two
    categories, then ask how the **probability of the founder strain
    fixing** (vs. the HW 1/2/3 intermediate classes fixing) depends on
    the swept `mutation_rate`.
    """)
    return


@app.cell
def compute_outcome(hw_df):
    _n_sites = int(hw_df["n_sites"].iloc[0])

    # hw_df already holds one row per (replicate_uid, hw) at each
    # replicate's final step (filtered upstream in load_data), so the
    # dominant Hamming-weight class per replicate is just the hw with
    # the largest n_cases.
    _dom_idx = hw_df.groupby("replicate_uid")["n_cases"].idxmax()
    outcome_df = hw_df.loc[
        _dom_idx,
        ["replicate_uid", "mutation_rate", "hw"],
    ].copy()

    # Founder strain (0000/1111) == extreme Hamming weights {0, N_SITES};
    # HW 1/2/3 intermediate == the three middle weights {1, 2, 3}.
    outcome_df["group"] = outcome_df["hw"].map(
        lambda _hw: "founder (0000/1111)"
        if _hw in (0, _n_sites)
        else "HW 1/2/3 (intermediate)",
    )
    print(f"outcome frame: {outcome_df.shape}")
    print(
        "dominant hw class counts:\n"
        + str(outcome_df["hw"].value_counts().sort_index()),
    )
    print(
        "group counts:\n" + str(outcome_df["group"].value_counts()),
    )
    return (outcome_df,)


@app.cell(hide_code=True)
def delimit_stats(mo):
    mo.md("""
    ## Exact (Clopper-Pearson) 95% Confidence Intervals

    For each `mutation_rate` the fixation outcome is a Bernoulli trial
    per replicate (founder strain fixes or not), so the per-condition
    fraction of replicates is a **binomial proportion**. We summarize
    it with the **exact Clopper-Pearson interval** via
    `scipy.stats.binomtest(k, n).proportion_ci(method="exact")` --- an
    exact estimator CI inverted from the binomial CDF, **not** a
    bootstrap. The complementary "HW 1/2/3 (intermediate)" series uses
    the same exact construction on the `n - k` complementary
    successes, so its interval is the exact CI for that proportion
    (and is not merely a reflection of the founder interval).
    Per-condition replicate counts in this sweep snapshot are uneven
    and can be as low as single digits (vs. the planned 25), so expect
    some of these intervals --- especially at the sparser conditions
    --- to be wide.
    """)
    return


@app.cell
def compute_stats(outcome_df, pd, sps):
    _groups = ["founder (0000/1111)", "HW 1/2/3 (intermediate)"]
    _rows = []
    for _mr, _sub in outcome_df.groupby("mutation_rate"):
        _n = len(_sub)
        _k_founder = int((_sub["group"] == "founder (0000/1111)").sum())
        _k_by_group = {
            "founder (0000/1111)": _k_founder,
            "HW 1/2/3 (intermediate)": _n - _k_founder,
        }
        for _group in _groups:
            _k = _k_by_group[_group]
            _ci = sps.binomtest(_k, _n).proportion_ci(
                confidence_level=0.95,
                method="exact",
            )
            _rows.append(
                {
                    "mutation_rate": float(_mr),
                    "group": _group,
                    "n_fixed": _k,
                    "n_total": _n,
                    "p": _k / _n,
                    "ci_low": float(_ci.low),
                    "ci_high": float(_ci.high),
                },
            )

    summary_df = (
        pd.DataFrame(_rows)
        .sort_values(
            ["group", "mutation_rate"],
        )
        .reset_index(drop=True)
    )
    print(summary_df.to_string(index=False))
    return (summary_df,)


@app.cell(hide_code=True)
def delimit_plot(mo):
    mo.md("""
    ## Fixation Probability vs. Mutation Rate

    `seaborn` lineplot of the per-condition fixation probability
    against `mutation_rate` on a **log x-axis**, one line per outcome
    group. The shaded **band shows the exact (Clopper-Pearson) 95%
    confidence interval** computed above (seaborn's own bootstrap CI
    is disabled via `errorbar=None`; the band is drawn from the exact
    estimator). The dashed **horizontal rule at 12.5%** marks the
    chance expectation that one of the two founder-aligned genomes
    (0000/1111, 2 of the 16 possible 4-site genomes) fixes under a
    uniform-over-genomes null. Gaps along the x-axis reflect
    mutation-rate conditions with no completed replicates yet in this
    sweep snapshot.
    """)
    return


@app.cell
def plot_fixation(pathlib, sns, summary_df, tp):
    _groups = ["founder (0000/1111)", "HW 1/2/3 (intermediate)"]
    _palette = dict(
        zip(_groups, sns.color_palette("colorblind", n_colors=len(_groups))),
    )

    with tp.teed(
        sns.lineplot,
        data=summary_df,
        x="mutation_rate",
        y="p",
        hue="group",
        hue_order=_groups,
        palette=_palette,
        marker="o",
        errorbar=None,
        teeplot_outattrs={"a": "founder-vs-hw123-fixation-prob"},
        teeplot_show=True,
        teeplot_subdir=pathlib.Path(__file__).stem,
    ) as _ax:
        # Exact-CI band per group (fill_between, not bootstrap).
        for _group in _groups:
            _sub = summary_df[summary_df["group"] == _group].sort_values(
                "mutation_rate",
            )
            _ax.fill_between(
                _sub["mutation_rate"],
                _sub["ci_low"],
                _sub["ci_high"],
                color=_palette[_group],
                alpha=0.2,
                linewidth=0,
            )
        # Chance expectation for the founder pair (2 of 16 genomes).
        _ax.axhline(
            0.125,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="12.5% (chance)",
        )
        _ax.set_xscale("log")
        _ax.set_ylim(-0.02, 1.02)
        _ax.set_xlabel("mutation rate (log scale)")
        _ax.set_ylabel("P(strain class fixes)")
        _ax.legend(
            title=None,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
        )
        sns.despine(ax=_ax)
        _ax.figure.set_size_inches(7, 4)
    return


@app.cell(hide_code=True)
def delimit_table(mo):
    mo.md("""
    ## Probability & 95% CI Table

    Per-`mutation_rate` fixation probability with the exact
    Clopper-Pearson 95% confidence interval, for both outcome groups.
    `n_fixed` / `n_total` are the binomial counts behind each estimate.
    """)
    return


@app.cell
def show_table(mo, summary_df):
    _table_df = (
        summary_df.assign(
            p=summary_df["p"].round(4),
            ci_low=summary_df["ci_low"].round(4),
            ci_high=summary_df["ci_high"].round(4),
            ci_95=summary_df.apply(
                lambda _r: f"[{_r['ci_low']:.3f}, {_r['ci_high']:.3f}]",
                axis=1,
            ),
        )[
            [
                "group",
                "mutation_rate",
                "n_fixed",
                "n_total",
                "p",
                "ci_low",
                "ci_high",
                "ci_95",
            ]
        ]
        .sort_values(["group", "mutation_rate"])
        .reset_index(drop=True)
    )

    print(_table_df.to_string(index=False))
    mo.ui.table(_table_df, selection=None)
    return


if __name__ == "__main__":
    app.run()
