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

    Load the per-replicate Hamming-weight **end-state** table for the
    4-site mutation-rate sweep, combining three slurm jobs (all driven
    by notebook `bindle/2026-05-20-founder.py`, fixing `N_SITES=4` and
    `POP_SIZE=1,000,000` on CPU, engine=numpy) that together cover the
    swept range at two different equilibration depths:

    - **deep** sweep
      (`slurm/2026-08-26/2026-08-26-4site-mutation-sweep.sh`, OSF
      https://osf.io/96r2v): `MUTATION_RATE` from `1e-5` down to
      `3e-11` (~2 points per decade, 12 conditions planned), 25
      replicates per condition (300 planned), **`N_STEPS=150,000`**.
    - **wide / high-rate** sweep
      (`slurm/2026-08-29/2026-08-29-4site-mutation-sweep.sh`, OSF
      https://osf.io/m6hzn): `MUTATION_RATE` from `1e-1` down to
      `3e-6` (~2 points per decade, 10 conditions), 200 replicates per
      condition (2000 planned), **`N_STEPS=5,000`**.
    - **wide / low-rate** sweep
      (`slurm/2026-08-30/2026-08-30-4site-mutation-sweep.sh`, OSF
      https://osf.io/buz8e): `MUTATION_RATE` from `1e-6` down to
      `3e-11` (10 conditions), 200 replicates per condition (2000
      planned), **`N_STEPS=5,000`**, seeded from 201 to avoid
      replicate-uid collisions with the other two sweeps.

    Both wide sweeps use a much shorter run than the deep sweep because
    dynamics saturate quickly relative to `N_STEPS=150,000`; they trade
    that depth for ~8x the replicate count, deliberately **duplicating**
    the deep sweep's rate range rather than avoiding it, to get a
    larger-N shallow estimate at the same conditions. The wide/low-rate
    sweep's 10 rungs (`1e-6` .. `3e-11`) fully overlap the deep sweep's
    bottom 10 rungs, so at those `mutation_rate` values this notebook
    pools replicates from **two different equilibration depths**
    (150,000 vs. 5,000 steps) --- the `mutation_rate x n_steps x
    replicate counts` diagnostic printed below shows exactly where
    that overlap falls. Rather than keeping the two depths as separate
    series, a replicate that hasn't actually settled on an outcome by
    its final step is instead captured directly as its own **"not
    converged"** group in the outcome classification below, so a
    shallow run's under-equilibration shows up as a probability rather
    than being silently averaged away or requiring a second series.

    The deep sweep was still in flight when this notebook was written:
    not every planned condition has completed replicates yet, and
    completed conditions can have far fewer than 25 replicates apiece.
    Both wide sweeps are essentially complete (200 replicates at every
    condition, except the high-rate sweep's `3e-4`, which has 180).
    Treat the deep sweep's confidence intervals as provisional
    accordingly (they should tighten, and any missing conditions fill
    in, as the sweep continues and its OSF file is refreshed).

    All three OSF files back this notebook as **full per-step
    trajectory tables** (one row per `(replicate_uid, Step, hw)`)
    rather than pre-filtered end-state exports, so loading any of them
    wholesale with `pandas` risks exhausting memory. Instead, each
    file's final-step filter (`Step == max(Step)`, computed per file
    since `N_STEPS` differs between the deep and wide sweeps) is
    pushed down through `pyarrow.dataset` before materializing to
    `pandas` and concatenating, since every completed replicate is
    recorded densely for every step (no checkpointing).

    All three OSF files are downloaded with `requests` and cached at
    `/tmp/<slug>` so re-runs hit the local copies.
    """)
    return


@app.cell
def configure_args(mo):
    # CLI args. Defaults pull the three 4-site mutation-sweep hw
    # parquets that back this notebook: deep (OSF https://osf.io/96r2v),
    # wide/high-rate (OSF https://osf.io/m6hzn), and wide/low-rate (OSF
    # https://osf.io/buz8e) --- see the Data section above.
    _args = mo.cli_args()
    _default_slugs = {
        "deep": "96r2v",
        "wide_high": "m6hzn",
        "wide_low": "buz8e",
    }
    OSF_SOURCES = {}
    for _name, _default_slug in _default_slugs.items():
        _slug = str(_args.get(f"osf-slug-{_name}") or _default_slug)
        _url = str(
            _args.get(f"osf-url-{_name}")
            or f"https://osf.io/{_slug}/download",
        )
        OSF_SOURCES[_name] = (_slug, _url)
        print(f"args: OSF_SLUG_{_name}={_slug} OSF_URL_{_name}={_url}")
    return (OSF_SOURCES,)


@app.cell
def download_data(OSF_SOURCES, pathlib, requests):
    def _download(slug, url):
        cache_path = pathlib.Path("/tmp") / slug
        if not cache_path.exists():
            print(f"downloading {url} -> {cache_path}")
            resp = requests.get(url, allow_redirects=True, timeout=300)
            resp.raise_for_status()
            cache_path.write_bytes(resp.content)
        else:
            print(f"reusing cached {cache_path}")
        print(f"size: {cache_path.stat().st_size} bytes")
        return cache_path

    cache_paths = {
        _name: _download(_slug, _url)
        for _name, (_slug, _url) in OSF_SOURCES.items()
    }
    return (cache_paths,)


@app.cell
def load_data(cache_paths, ds, pc, pd):
    def _load_final_step(cache_path):
        # Each parquet is the full per-step trajectory table for its
        # sweep, so avoid pd.read_parquet(cache_path) --- loading every
        # step of every replicate into pandas is enough on its own to
        # exhaust memory on a typical runner. Every completed replicate
        # is recorded densely for every step (no checkpointing), so
        # pushing a Step == max(Step) filter down through pyarrow
        # before converting to pandas recovers exactly the end-state
        # rows at a fraction of the memory cost. max(Step) is computed
        # per file since N_STEPS differs between the deep and wide
        # sweeps.
        _dataset = ds.dataset(cache_path, format="parquet")
        _max_step = pc.max(
            _dataset.to_table(columns=["Step"])["Step"],
        ).as_py()
        return _dataset.to_table(
            filter=(pc.field("Step") == _max_step),
            columns=[
                "hw",
                "n_cases",
                "replicate_uid",
                "mutation_rate",
                "n_sites",
                "n_steps",
            ],
        ).to_pandas()

    hw_df = pd.concat(
        [_load_final_step(_p) for _p in cache_paths.values()],
        ignore_index=True,
    )
    print(f"loaded hw end-state dataframe: {hw_df.shape}")
    print(
        "mutation_rate x n_steps x replicate counts:\n"
        + str(
            hw_df.groupby(["mutation_rate", "n_steps"])[
                "replicate_uid"
            ].nunique(),
        ),
    )
    return (hw_df,)


@app.cell(hide_code=True)
def delimit_outcome(mo):
    mo.md("""
    ## Fixation Outcome per Replicate

    For each replicate, classify its final-simulation-step case counts
    (`n_cases`, summed into Hamming-weight bins) into one of four
    outcomes. A replicate has **converged on a complement pair** only
    if **both members of that pair each hold at least 1/3 of its
    total** final-step cases --- e.g. a replicate sitting at ~93%
    `hw=0` with `hw=4` still at 0% hasn't reached the founder pair, it
    simply hasn't had a mutation reach the far extreme yet, so it
    doesn't count as converged. Requiring 1/3 from *each* member (not
    just a combined majority) rules out one side dominating while the
    other is barely represented. Replicates from all three sweeps are
    pooled together at each `mutation_rate` (see the Data section
    above).

    With `N_SITES=4` there are five Hamming-weight classes drawn from
    the sixteen possible genomes, grouped into four outcomes:

    - **founder strain (0000/1111)**: both `hw=0` and `hw=4` --- the
      all-zero founder/wildtype genome and its bitwise complement,
      each a single genome --- individually hold >= 1/3 of cases.
    - **HW 1/3**: both `hw=1` and `hw=3` --- the one- and
      three-mutation classes, bitwise complements of one another ---
      individually hold >= 1/3 of cases. Each bin spans 4 genomes
      rather than 1, so this confirms *some* hw=1/hw=3 complement pair
      is jointly well-represented, without pinning down which specific
      pair (that would need per-genome data).
    - **HW 2**: `hw == 2` alone holds >= 2/3 of cases --- the
      combined-pair threshold (2 x 1/3) applied to the single bin,
      since `hw == 2` isn't a two-way complement split like the other
      groups (e.g. `1100` and `0011` are complements of each other,
      but both land in `hw == 2`, alongside two other complement pairs
      the aggregated bin can't distinguish without per-genome data).
    - **not converged**: none of the above thresholds are met.

    We classify each replicate's end-state into one of these four
    categories, then ask how the **probability of each outcome**
    depends on the swept `mutation_rate`. Splitting HW 1/3 from HW 2
    (rather than lumping all of `hw in {1, 2, 3}` together, as the
    2-/3-site companion notebooks do) matters here because the two
    groups carry different genome counts (8 vs. 6 of 16) and thus
    different chance baselines.
    """)
    return


@app.cell
def compute_outcome(hw_df):
    _n_sites = int(hw_df["n_sites"].iloc[0])
    # Converged on a complement pair requires each member of that pair
    # to individually hold >= 1/3 of the replicate's total final-step
    # cases; HW 2 isn't a two-way split, so its bar is 2 x 1/3 applied
    # to the single bin --- see the rationale in delimit_outcome above.
    _PAIR_THRESHOLD = 1 / 3

    _pivot = hw_df.pivot_table(
        index="replicate_uid",
        columns="hw",
        values="n_cases",
        aggfunc="sum",
        fill_value=0,
    )
    _total = _pivot.sum(axis=1)
    _frac0 = _pivot[0] / _total
    _frac1 = _pivot[1] / _total
    _frac2 = _pivot[_n_sites // 2] / _total
    _frac3 = _pivot[_n_sites - 1] / _total
    _frac4 = _pivot[_n_sites] / _total

    def _classify(_uid):
        if _frac0[_uid] >= _PAIR_THRESHOLD and _frac4[_uid] >= _PAIR_THRESHOLD:
            return "founder (0000/1111)"
        if _frac1[_uid] >= _PAIR_THRESHOLD and _frac3[_uid] >= _PAIR_THRESHOLD:
            return "HW 1/3"
        if _frac2[_uid] >= 2 * _PAIR_THRESHOLD:
            return "HW 2"
        return "not converged"

    outcome_df = (
        hw_df[["replicate_uid", "mutation_rate"]]
        .drop_duplicates("replicate_uid")
        .copy()
    )
    outcome_df["group"] = outcome_df["replicate_uid"].map(_classify)
    print(f"outcome frame: {outcome_df.shape}")
    print(
        "group counts:\n" + str(outcome_df["group"].value_counts()),
    )
    return (outcome_df,)


@app.cell(hide_code=True)
def delimit_stats(mo):
    mo.md("""
    ## Exact (Clopper-Pearson) 95% Confidence Intervals

    Each replicate's outcome label is one of four mutually exclusive,
    collectively exhaustive categories, so for each `mutation_rate` the
    "outcome X vs. not" indicator is a Bernoulli trial per replicate
    and the per-condition fraction is a **binomial proportion**.
    Replicates are pooled across all three sweeps at each
    `mutation_rate` regardless of `n_steps` (see the Data and Fixation
    Outcome sections above for how the differing equilibration depths
    are instead surfaced via the "not converged" group). We summarize
    each of the four group proportions independently with the **exact
    Clopper-Pearson interval** via `scipy.stats.binomtest(k,
    n).proportion_ci(method="exact")` --- an exact estimator CI
    inverted from the binomial CDF, **not** a bootstrap.
    Per-condition replicate counts in this combined dataset are uneven
    and can be as low as single digits for the deep sweep's sparser
    conditions (vs. its planned 25), up to 200 for the wide sweeps, so
    expect some intervals to be wide.
    """)
    return


@app.cell
def compute_stats(outcome_df, pd, sps):
    _groups = [
        "founder (0000/1111)",
        "HW 1/3",
        "HW 2",
        "not converged",
    ]
    _rows = []
    for _mr, _sub in outcome_df.groupby("mutation_rate"):
        _n = len(_sub)
        _counts = _sub["group"].value_counts()
        for _group in _groups:
            _k = int(_counts.get(_group, 0))
            _ci = sps.binomtest(_k, _n).proportion_ci(
                confidence_level=0.95,
                method="exact",
            )
            _rows.append(
                {
                    "mutation_rate": float(_mr),
                    "group": _group,
                    "n_group": _k,
                    "n_total": _n,
                    "p": _k / _n,
                    "ci_low": float(_ci.low),
                    "ci_high": float(_ci.high),
                },
            )

    summary_df = (
        pd.DataFrame(_rows)
        .sort_values(["group", "mutation_rate"])
        .reset_index(drop=True)
    )
    print(summary_df.to_string(index=False))
    return (summary_df,)


@app.cell(hide_code=True)
def delimit_plot(mo):
    mo.md("""
    ## Fixation Probability vs. Mutation Rate

    `seaborn` lineplot of the per-condition outcome probability against
    `mutation_rate` on a **log x-axis**, one line per outcome group
    (replicates pooled across all three sweeps at each `mutation_rate`
    --- see the Data section above). The shaded **band shows the exact
    (Clopper-Pearson) 95% confidence interval** computed above
    (seaborn's own bootstrap CI is disabled via `errorbar=None`; the
    band is drawn from the exact estimator). The three fixation groups
    each get a color-matched dashed **horizontal chance-expectation
    rule**, at the genome-count fraction of the 16 possible 4-site
    genomes they cover: **12.5%** for founder (0000/1111, 2 genomes),
    **50%** for HW 1/3 (8 genomes), and **37.5%** for HW 2
    (6 genomes) --- all under a uniform-over-genomes null. "not
    converged" has no genome-identity analog (it's about whether any
    class dominates at all, not which one), so it gets no chance line.
    Gaps along the x-axis reflect mutation-rate conditions with no
    completed replicates yet in this sweep snapshot.
    """)
    return


@app.cell
def plot_fixation(pathlib, sns, summary_df, tp):
    _groups = [
        "founder (0000/1111)",
        "HW 1/3",
        "HW 2",
        "not converged",
    ]
    _palette = dict(
        zip(_groups, sns.color_palette("colorblind", n_colors=len(_groups))),
    )
    # Genome-count fraction of the 16 possible 4-site genomes each
    # fixation group covers, under a uniform-over-genomes null: founder
    # {0000, 1111} (2), HW 1/3 complements (2 * C(4,1) = 8), HW 2
    # (C(4,2) = 6). "not converged" has no genome-identity analog, so
    # it's intentionally absent here and gets no chance line below.
    _chance = {
        "founder (0000/1111)": 2 / 16,
        "HW 1/3": 8 / 16,
        "HW 2": 6 / 16,
    }

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
        teeplot_outattrs={
            "a": "founder-vs-hw13-vs-hw2-vs-unconverged-fixation-prob",
        },
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
        # Color-matched chance-expectation rule per group with a defined
        # genome-count baseline (skips "not converged").
        for _group in _groups:
            if _group not in _chance:
                continue
            _ax.axhline(
                _chance[_group],
                color=_palette[_group],
                linestyle="--",
                linewidth=1.0,
                label=f"{_group} chance ({_chance[_group]:.1%})",
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

    Per-`mutation_rate` outcome probability with the exact
    Clopper-Pearson 95% confidence interval, for all four outcome
    groups. `n_group` / `n_total` are the binomial counts behind each
    estimate.
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
                "n_group",
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
