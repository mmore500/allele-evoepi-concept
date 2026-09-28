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
    import matplotlib.colors as mcolors
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np
    import pandas as pd
    import requests
    import seaborn as sns
    from teeplot import teeplot as tp
    from watermark import watermark

    return (
        mcolors,
        mlines,
        mo,
        mticker,
        np,
        pd,
        plt,
        requests,
        sns,
        tp,
        watermark,
    )


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

    Render a reproduction of Figure 3 from Koopman & Eisenberg,
    "An allele-based evolution model of the population spread of
    SARS-CoV-2" (medRxiv, <https://doi.org/10.1101/2025.09.15.25335781>).
    That figure follows a fully susceptible population through an
    initial outbreak of a 2-allele strain, its immune-driven mutational
    replacement, and the ensuing damped oscillations in strain
    prevalence.
    The underlying Madonna ODE-model trajectory is hosted on OSF as
    `Fig3.csv` (<https://osf.io/jy59v>) and downloaded below with no
    caching-key CLI arguments (this is a visualization notebook).
    Columns are keyed `TIME` plus one variable per 2-allele strain
    (named by the pair of mutated sites, e.g. `02`, `03`, `12`, `13`):
    a `J**` column gives that strain's prevalence, and a `SuscToI**`
    column gives the population's fraction still susceptible to
    (re)infection by that strain.
    In the reference dynamic, strain `02` seeds the initial wave.
    Immunity to it wanes unevenly across its two alleles, permitting
    mutations to intermediate strains `03` and `12` at vanishingly low
    prevalence.
    Both intermediates mutate onward to strain `13`, which faces little
    standing immunity and comes to dominate later waves.
    """)
    return


@app.cell
def def_fetch(pathlib, pd, requests):
    def fetch_osf(slug):
        cache_path = pathlib.Path("/tmp") / slug
        url = f"https://osf.io/{slug}/download"
        if not cache_path.exists():
            print(f"downloading {url} -> {cache_path}")
            resp = requests.get(url, allow_redirects=True, timeout=120)
            resp.raise_for_status()
            cache_path.write_bytes(resp.content)
        else:
            print(f"reusing cached {cache_path}")
        print(f"size: {cache_path.stat().st_size} bytes")
        return pd.read_csv(cache_path)

    return (fetch_osf,)


@app.cell
def download_data(fetch_osf):
    fig3_df = fetch_osf("jy59v")
    fig3_df.columns = [col.replace(":1", "") for col in fig3_df.columns]
    print(f"loaded Fig3 dataframe: {fig3_df.shape}")
    print(fig3_df.head().to_string(index=False))
    return (fig3_df,)


@app.cell(hide_code=True)
def delimit_plot(mo):
    mo.md("""
    ## Strain Dynamics Render

    Two panels stacked vertically, sharing a strain color hue.
    Top: per-strain prevalence over time (zero-prevalence stretches,
    i.e. before a strain has yet mutated into existence, are left as
    gaps), rendered three times with a different y-axis scale each
    time (symlog, log, and linear) to show the damped oscillations at
    both the strain-replacement scale and the many-orders-of-magnitude
    scale of a strain's initial emergence.
    Bottom: per-strain fraction of the population susceptible to
    reinfection, always on a linear scale.
    """)
    return


@app.cell
def prep_long(fig3_df):
    long_df = fig3_df.melt(
        id_vars="TIME",
        var_name="variable",
        value_name="value",
    )
    long_df["strain"] = long_df["variable"].str.slice(-2)
    is_susceptibility = long_df["variable"].str.startswith("Susc")
    long_df["quantity"] = is_susceptibility.map(
        {True: "susceptibility", False: "prevalence"},
    )
    zeroed_prevalence = (long_df["quantity"] == "prevalence") & (
        long_df["value"] == 0
    )
    long_df.loc[zeroed_prevalence, "value"] = float("nan")
    return (long_df,)


@app.cell
def plot_fig3(long_df, mcolors, mlines, mticker, np, pathlib, sns, tp):
    _strain_order = ["02", "03", "12", "13"]
    # Named colors ordered by mutational distance from founder strain
    # 02: wt is a subdued dark blue-purple, the two single-mutant
    # intermediates (03, 12) are medium-intensity green/blue, and
    # double-mutant 13 is the most intense (a warm coral-orange), all
    # checked for readable contrast against a white background.
    _palette = {
        "02": "#373061",
        "03": "#3cb371",
        "12": "#1e90ff",
        "13": "#ff8671",
    }
    _dashes = {
        "02": "",
        "03": "",
        "12": (0.8, 2.6),
        "13": (4, 1.5),
    }
    _dotted_rgb = mcolors.to_rgb(_palette["12"])

    def _fmt(y, _):
        if y == 0:
            return "0"
        return f"{y:.10f}".rstrip("0").rstrip(".")

    for _yscale in ["symlog", "log", "linear"]:
        with tp.teed(
            sns.relplot,
            data=long_df,
            x="TIME",
            y="value",
            hue="strain",
            hue_order=_strain_order,
            style="strain",
            style_order=_strain_order,
            dashes=_dashes,
            palette=_palette,
            linewidth=1.2,
            row="quantity",
            row_order=["prevalence", "susceptibility"],
            kind="line",
            facet_kws=dict(sharey=False),
            teeplot_outattrs={
                "a": "fig3-strain-dynamics",
                "yscale": _yscale,
            },
            teeplot_show=True,
            teeplot_subdir=pathlib.Path(__file__).stem,
        ) as g:
            g.axes_dict["prevalence"].set_ylabel("Strain\nPrevalence")
            g.axes_dict["susceptibility"].set_ylabel("Host\nSusceptibility")
            g.axes_dict["susceptibility"].set_xlabel("Time")
            g.axes_dict["susceptibility"].set_yticks([0.0, 0.5, 1.0])
            g.axes_dict["susceptibility"].set_ylim(bottom=0.0)
            g.set_titles("")

            if _yscale == "symlog":
                # Linear near zero (below 1e-4) so the exact-zero gaps
                # stay well-defined, log above it; tick decades are
                # spelled out explicitly since the default symlog
                # locator doesn't reach this far down on its own.
                g.axes_dict["prevalence"].set_yscale("symlog", linthresh=1e-4)
                # Small negative bottom so the y=0 line isn't flush
                # against the axis edge, without a tick below zero.
                g.axes_dict["prevalence"].set_ylim(bottom=-2e-5, top=1)
                g.axes_dict["prevalence"].set_yticks(
                    [0, 1e-4, 1e-3, 1e-2, 1e-1, 1]
                )
                g.axes_dict["prevalence"].yaxis.set_major_formatter(
                    mticker.FuncFormatter(_fmt)
                )
                # Minor ticks double as a visual cue for where the
                # scale is log (uneven, sub-decade spacing) versus
                # linear (evenly spaced) near zero.
                _log_minors = mticker.SymmetricalLogLocator(
                    base=10,
                    linthresh=1e-4,
                    subs=np.arange(2, 10),
                    transform=g.axes_dict["prevalence"].yaxis.get_transform(),
                ).tick_values(1e-4, 1)
                _linear_minors = np.linspace(0, 1e-4, 5)[1:-1]
                g.axes_dict["prevalence"].yaxis.set_minor_locator(
                    mticker.FixedLocator([*_linear_minors, *_log_minors])
                )
                g.axes_dict["prevalence"].tick_params(
                    axis="y", which="major", pad=-2, labelsize=7
                )
                g.axes_dict["prevalence"].tick_params(
                    axis="y", which="minor", length=2
                )
                # Angled labels take less horizontal room, so they can
                # sit closer to the axis.
                for _tick in g.axes_dict["prevalence"].get_yticklabels():
                    _tick.set_rotation(45)
                    _tick.set_ha("right")
            elif _yscale == "log":
                g.axes_dict["prevalence"].set_yscale("log")
                g.axes_dict["prevalence"].set_ylim(bottom=1e-9)

            # Widen 12's dotted line beyond the shared base linewidth
            # so its sparser dots stay legible.
            for _ax in g.axes.flat:
                for _line in _ax.lines:
                    if mcolors.to_rgb(_line.get_color()) == _dotted_rgb:
                        _line.set_linewidth(2.8)

            g.figure.set_size_inches(4.6, 2.6)
            g.figure.tight_layout()

            # Fold the "strain" title into the legend's single row as
            # a label-only dummy entry, rather than a separate title
            # line.
            _handles = g.legend.legend_handles
            _labels = [_t.get_text() for _t in g.legend.get_texts()]
            for _handle, _label in zip(_handles, _labels):
                if _label == "12":
                    # Denser dashes than the plotted line so the short
                    # legend key still reads clearly as dotted.
                    _handle.set_linewidth(2.8)
                    _handle.set_dashes([0.8, 1.2])
            _dummy = mlines.Line2D([], [], linestyle="none", label="Strain")
            g.legend.remove()
            g.figure.legend(
                handles=[_dummy, *_handles],
                labels=["Strain", *_labels],
                loc="upper center",
                bbox_to_anchor=(0.5, 1.09),
                ncol=len(_labels) + 1,
                frameon=False,
                handlelength=1.8,
                handletextpad=0.6,
                columnspacing=1.2,
            )
    return


@app.cell(hide_code=True)
def delimit_plot_3panel(mo):
    mo.md("""
    ## Strain Dynamics Render (3-Panel)

    An alternative single-figure layout stacking three panels top to
    bottom: host susceptibility, strain prevalence on a linear scale,
    and that same prevalence data again on a log scale (to show the
    low-prevalence dynamics the linear scale hides).
    """)
    return


@app.cell
def plot_fig3_3panel(long_df, mcolors, mlines, pathlib, plt, sns, tp):
    _strain_order = ["02", "03", "12", "13"]
    _palette = {
        "02": "#373061",
        "03": "#3cb371",
        "12": "#1e90ff",
        "13": "#ff8671",
    }
    _dashes = {
        "02": "",
        "03": "",
        "12": (0.8, 2.6),
        "13": (4, 1.5),
    }
    _dotted_rgb = mcolors.to_rgb(_palette["12"])

    _prevalence_df = long_df[long_df["quantity"] == "prevalence"]
    _susceptibility_df = long_df[long_df["quantity"] == "susceptibility"]

    with tp.teed(
        plt.subplots,
        nrows=3,
        ncols=1,
        figsize=(4.6, 3.9),
        teeplot_outattrs={"a": "fig3-strain-dynamics-3panel"},
        teeplot_show=True,
        teeplot_subdir=pathlib.Path(__file__).stem,
    ) as (fig, axes):
        _ax_lin_susc, _ax_lin_prev, _ax_log = axes

        for _ax in (_ax_lin_prev, _ax_log):
            sns.lineplot(
                data=_prevalence_df,
                x="TIME",
                y="value",
                hue="strain",
                hue_order=_strain_order,
                style="strain",
                style_order=_strain_order,
                dashes=_dashes,
                palette=_palette,
                linewidth=1.2,
                legend=False,
                ax=_ax,
            )
        sns.lineplot(
            data=_susceptibility_df,
            x="TIME",
            y="value",
            hue="strain",
            hue_order=_strain_order,
            style="strain",
            style_order=_strain_order,
            dashes=_dashes,
            palette=_palette,
            linewidth=1.2,
            legend=True,
            ax=_ax_lin_susc,
        )

        _ax_log.set_yscale("log")
        _ax_log.set_ylim(bottom=1e-9)
        _ax_log.set_ylabel("Log Strain\nPrevalence")
        _ax_log.set_xlabel("Time")

        _ax_lin_prev.set_ylabel("Strain\nPrevalence")
        _ax_lin_prev.set_xlabel("")
        _ax_lin_prev.set_xticklabels([])

        _ax_lin_susc.set_ylabel("Host\nSusceptibility")
        _ax_lin_susc.set_xlabel("")
        _ax_lin_susc.set_xticklabels([])
        _ax_lin_susc.set_yticks([0.0, 0.5, 1.0])
        _ax_lin_susc.set_ylim(bottom=0.0)

        # Widen 12's dotted line beyond the shared base linewidth so
        # its sparser dots stay legible.
        for _ax in axes:
            for _line in _ax.lines:
                if mcolors.to_rgb(_line.get_color()) == _dotted_rgb:
                    _line.set_linewidth(2.8)

        sns.despine(fig=fig)
        fig.tight_layout()

        # Fold the "strain" title into the legend's single row as a
        # label-only dummy entry, rather than a separate title line.
        _legend = _ax_lin_susc.get_legend()
        _handles = _legend.legend_handles
        _labels = [_t.get_text() for _t in _legend.get_texts()]
        _legend.remove()
        for _handle, _label in zip(_handles, _labels):
            if _label == "12":
                # Denser dashes than the plotted line so the short
                # legend key still reads clearly as dotted.
                _handle.set_linewidth(2.8)
                _handle.set_dashes([0.8, 1.2])
        _dummy = mlines.Line2D([], [], linestyle="none", label="Strain")
        fig.legend(
            handles=[_dummy, *_handles],
            labels=["Strain", *_labels],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.06),
            ncol=len(_labels) + 1,
            frameon=False,
            handlelength=1.8,
            handletextpad=0.6,
            columnspacing=1.2,
        )
    return


if __name__ == "__main__":
    app.run()
