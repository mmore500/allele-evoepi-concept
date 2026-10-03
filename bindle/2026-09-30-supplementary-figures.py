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
    import matplotlib.legend_handler as mlegend_handler
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt
    import pandas as pd
    import requests
    import seaborn as sns
    from teeplot import teeplot as tp
    from watermark import watermark

    return (
        mcolors,
        mlegend_handler,
        mlines,
        mo,
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

    Prototype redraws of the supplementary figures from Simon, Koopman,
    Eisenberg, Zaman, Moreno & Polanco, "An allele-based model of
    coronavirus evolution under population immunity" (medRxiv,
    <https://doi.org/10.1101/2025.09.15.25335781>, v3 supplementary
    material), reusing the strain-dynamics rendering style established
    in `2026-09-26-figrender.py` and `2026-09-29-manuscript-figures.py`.

    Most datasets share the same column convention as the main-text
    figures: `TIME` plus a `J**` prevalence column and a `SuscToI**`
    susceptibility column per 2-allele strain (`02`, `03`, `12`, `13`).
    These render as the standard two-panel susceptibility+prevalence
    figure (`FigS1a`-`FigS1d`, `FigS2`-`FigS8`, `FigS10a`-`FigS10b`,
    `FigS11a`-`FigS11b`, `FigS12a`-`FigS12c`).

    Two groups of figures extend this convention to track specific
    immune-history compartments. Per the manuscript's "Immunity Status
    Vector" definition, a recovered individual's history is tracked as
    `R(K0,K1,K2,K3)`, the immunity level (0-3, with 3 naive) against
    each of the model's four alleles. `R3333` is the fully-naive,
    never-infected pool; `R0000` is the pool that has been infected by
    every allele but whose immunity has fully waned back to
    susceptible. Neither is a strain, so each is rendered on its own
    "Compartment Occupancy" panel with its own "Host Compartment"
    legend row, separate from the "Strain" legend row:

    - `FigS9a`-`FigS9c` add a single `R0000` column to the standard
      4-strain dataset, rendered as a three-panel figure
      (susceptibility, strain prevalence, compartment occupancy).
    - `FigS13a`-`FigS13c` use a reduced 2-strain (`02`, `13`) dataset
      with both `R3333` and `R0000` columns, also rendered as a
      three-panel figure.

    As in the manuscript-figures notebook, strain `12`'s dotted line
    gets a thin solid underlay (and a matching legend key) so its
    trajectory reads as continuous between dots, and each figure's
    x-axis maximum is set to match the corresponding original figure.
    Figures are not titled.

    `TableS14`-`TableS24b` are additional OSF-hosted datasets using
    the same `J**`/`SuscToI**` column convention, rendered with the
    same `plot_figure` helper. No original figure/caption was
    available for these to match axis ranges against, so each simply
    spans its full simulated time range. `TableS18` holds two
    conditions in one file (columns suffixed `:1`/`:2`); these are
    split into separate `TableS18a`/`TableS18b` prevalence-only
    figures rather than merged into one.
    """)
    return


@app.cell
def def_fetch(pathlib, pd, requests):
    def fetch_osf(slug, sep=","):
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
        df = pd.read_csv(cache_path, sep=sep)
        df.columns = [col.replace(":1", "") for col in df.columns]
        return df

    return (fetch_osf,)


@app.function
def prep_long(df):
    long_df = df.melt(
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
    has_susceptibility = is_susceptibility.any()
    return long_df, has_susceptibility


@app.cell(hide_code=True)
def delimit_plot(mo):
    mo.md("""
    ## Renders

    One cell per figure: fetch its OSF dataset, then render with the
    shared plotting helpers below. The x-axis maximum for each is set
    to match the corresponding original figure.
    """)
    return


@app.cell
def def_plot(mcolors, mlegend_handler, mlines, plt, sns, tp):
    STRAIN_ORDER = ["02", "03", "12", "13"]
    # Same named-color palette as 2026-09-26-figrender.py and
    # 2026-09-29-manuscript-figures.py, so strain identity reads
    # consistently across notebooks.
    PALETTE = {
        "02": "#373061",
        "03": "#3cb371",
        "12": "#1e90ff",
        "13": "#ff8671",
    }
    DASHES = {
        "02": "",
        "03": "",
        "12": (0.8, 2.6),
        "13": (4, 1.5),
    }
    DOTTED_RGB = mcolors.to_rgb(PALETTE["12"])

    # Tableau brown/gold, distinct from the strain palette, for the
    # non-strain immune-history compartments R3333 and R0000.
    R0000_COLOR = "#EDC948"
    S13_STRAIN_ORDER = ["02", "13"]
    S13_COMPARTMENT_COLORS = {"R3333": "#9C755F", "R0000": "#EDC948"}
    S13_COMPARTMENT_STYLES = {"R3333": "-", "R0000": "-."}

    def style_dotted(ax):
        # Widen 12's dotted line beyond the shared base linewidth so
        # its sparser dots stay legible, and add a thin solid underlay
        # so the trajectory still reads as continuous between dots.
        for line in list(ax.lines):
            if mcolors.to_rgb(line.get_color()) == DOTTED_RGB:
                xdata, ydata = line.get_data()
                if len(xdata) == 0:
                    continue
                ax.plot(
                    xdata,
                    ydata,
                    color=line.get_color(),
                    linewidth=0.6,
                    zorder=line.get_zorder() - 0.1,
                    solid_capstyle="round",
                )
                line.set_linewidth(2.8)

    def finish_legend(fig, legend_ax, has_susceptibility):
        # Fold the "strain" title into the legend's single row as a
        # label-only dummy entry, rather than a separate title line.
        legend = legend_ax.get_legend()
        handles = legend.legend_handles
        labels = [t.get_text() for t in legend.get_texts()]
        legend.remove()
        for i, (handle, label) in enumerate(zip(handles, labels)):
            if label == "12":
                # Denser dashes than the plotted line so the short
                # legend key still reads clearly as dotted.
                handle.set_linewidth(2.8)
                handle.set_dashes([0.8, 1.2])
                # Pair with a thin solid proxy so the key matches the
                # plotted line's solid-underlay treatment.
                solid_proxy = mlines.Line2D(
                    [],
                    [],
                    color=handle.get_color(),
                    linewidth=0.6,
                    solid_capstyle="round",
                )
                handles[i] = (solid_proxy, handle)
        dummy = mlines.Line2D([], [], linestyle="none", label="Strain")
        fig.legend(
            handles=[dummy, *handles],
            labels=["Strain", *labels],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.06 if has_susceptibility else 1.1),
            ncol=len(labels) + 1,
            frameon=False,
            handlelength=1.8,
            handletextpad=0.6,
            columnspacing=1.2,
            handler_map={tuple: mlegend_handler.HandlerTuple(ndivide=1)},
        )

    def plot_figure(long_df, has_susceptibility, name, xmax, pathlib_stem):
        prevalence_df = long_df[long_df["quantity"] == "prevalence"]
        nrows = 2 if has_susceptibility else 1
        figsize = (4.6, 3.9) if has_susceptibility else (4.6, 2.2)

        with tp.teed(
            plt.subplots,
            nrows=nrows,
            ncols=1,
            squeeze=False,
            figsize=figsize,
            teeplot_outattrs={"a": f"{name.lower()}-strain-dynamics"},
            teeplot_show=True,
            teeplot_subdir=pathlib_stem,
        ) as (fig, axes_grid):
            axes = axes_grid[:, 0]
            if has_susceptibility:
                susceptibility_df = long_df[
                    long_df["quantity"] == "susceptibility"
                ]
                ax_susc, ax_prev = axes
                sns.lineplot(
                    data=susceptibility_df,
                    x="TIME",
                    y="value",
                    hue="strain",
                    hue_order=STRAIN_ORDER,
                    style="strain",
                    style_order=STRAIN_ORDER,
                    dashes=DASHES,
                    palette=PALETTE,
                    linewidth=1.2,
                    legend=True,
                    ax=ax_susc,
                )
                ax_susc.set_ylabel("Host\nSusceptibility")
                ax_susc.set_xlabel("")
                ax_susc.set_xticklabels([])
                ax_susc.set_yticks([0.0, 0.5, 1.0])
                ax_susc.set_ylim(bottom=0.0)
                legend_ax = ax_susc
            else:
                (ax_prev,) = axes
                legend_ax = None

            sns.lineplot(
                data=prevalence_df,
                x="TIME",
                y="value",
                hue="strain",
                hue_order=STRAIN_ORDER,
                style="strain",
                style_order=STRAIN_ORDER,
                dashes=DASHES,
                palette=PALETTE,
                linewidth=1.2,
                legend=(legend_ax is None),
                ax=ax_prev,
            )
            ax_prev.set_ylabel("Strain\nPrevalence")
            ax_prev.set_xlabel("Time")
            if legend_ax is None:
                legend_ax = ax_prev

            style_dotted(ax_prev)
            if has_susceptibility:
                style_dotted(ax_susc)
            for ax in axes:
                ax.set_xlim(left=0, right=xmax)

            sns.despine(fig=fig)
            fig.tight_layout()
            finish_legend(fig, legend_ax, has_susceptibility)
        return fig

    def prep_long_r0000(df):
        long_df, has_susceptibility = prep_long(
            df[[c for c in df.columns if c != "R0000"]]
        )
        r0000_df = df[["TIME", "R0000"]].rename(columns={"R0000": "value"})
        return long_df, r0000_df, has_susceptibility

    def plot_figure_r0000(
        long_df, r0000_df, has_susceptibility, name, xmax, pathlib_stem
    ):
        prevalence_df = long_df[long_df["quantity"] == "prevalence"]
        susceptibility_df = long_df[long_df["quantity"] == "susceptibility"]

        with tp.teed(
            plt.subplots,
            nrows=3,
            ncols=1,
            squeeze=False,
            figsize=(4.6, 5.4),
            teeplot_outattrs={"a": f"{name.lower()}-strain-dynamics"},
            teeplot_show=True,
            teeplot_subdir=pathlib_stem,
        ) as (fig, axes_grid):
            axes = axes_grid[:, 0]
            ax_susc, ax_prev, ax_comp = axes
            sns.lineplot(
                data=susceptibility_df,
                x="TIME",
                y="value",
                hue="strain",
                hue_order=STRAIN_ORDER,
                style="strain",
                style_order=STRAIN_ORDER,
                dashes=DASHES,
                palette=PALETTE,
                linewidth=1.2,
                legend=True,
                ax=ax_susc,
            )
            ax_susc.set_ylabel("Host\nSusceptibility")
            ax_susc.set_xlabel("")
            ax_susc.set_xticklabels([])
            ax_susc.set_yticks([0.0, 0.5, 1.0])
            ax_susc.set_ylim(bottom=0.0)

            sns.lineplot(
                data=prevalence_df,
                x="TIME",
                y="value",
                hue="strain",
                hue_order=STRAIN_ORDER,
                style="strain",
                style_order=STRAIN_ORDER,
                dashes=DASHES,
                palette=PALETTE,
                linewidth=1.2,
                legend=False,
                ax=ax_prev,
            )
            ax_prev.set_ylabel("Strain\nPrevalence")
            ax_prev.set_xlabel("")
            ax_prev.set_xticklabels([])

            ax_comp.plot(
                r0000_df["TIME"],
                r0000_df["value"],
                color=R0000_COLOR,
                linewidth=1.2,
                linestyle=S13_COMPARTMENT_STYLES["R0000"],
            )
            ax_comp.set_ylabel("Compartment\nOccupancy")
            ax_comp.set_xlabel("Time")

            style_dotted(ax_prev)
            style_dotted(ax_susc)
            for ax in axes:
                ax.set_xlim(left=0, right=xmax)

            sns.despine(fig=fig)
            fig.tight_layout()

            legend = ax_susc.get_legend()
            handles = legend.legend_handles
            labels = [t.get_text() for t in legend.get_texts()]
            legend.remove()
            for i, (handle, label) in enumerate(zip(handles, labels)):
                if label == "12":
                    handle.set_linewidth(2.8)
                    handle.set_dashes([0.8, 1.2])
                    solid_proxy = mlines.Line2D(
                        [],
                        [],
                        color=handle.get_color(),
                        linewidth=0.6,
                        solid_capstyle="round",
                    )
                    handles[i] = (solid_proxy, handle)
            dummy = mlines.Line2D([], [], linestyle="none", label="Strain")
            fig.legend(
                handles=[dummy, *handles],
                labels=["Strain", *labels],
                loc="upper center",
                bbox_to_anchor=(0.5, 1.1),
                ncol=len(labels) + 1,
                frameon=False,
                handlelength=1.8,
                handletextpad=0.6,
                columnspacing=1.2,
                handler_map={tuple: mlegend_handler.HandlerTuple(ndivide=1)},
            )

            r0000_handle = mlines.Line2D(
                [],
                [],
                color=R0000_COLOR,
                linewidth=1.2,
                linestyle=S13_COMPARTMENT_STYLES["R0000"],
            )
            comp_dummy = mlines.Line2D(
                [], [], linestyle="none", label="Host Compartment"
            )
            fig.legend(
                handles=[comp_dummy, r0000_handle],
                labels=["Host Compartment", "R0000"],
                loc="upper center",
                bbox_to_anchor=(0.5, 1.055),
                ncol=2,
                frameon=False,
                handlelength=1.8,
                handletextpad=0.6,
                columnspacing=1.2,
            )
        return fig

    def prep_long_s13(df):
        strain_df = df[["TIME", "J02", "J13", "SuscToI02"]]
        long_df, has_susceptibility = prep_long(strain_df)
        compartment_df = df[["TIME", *S13_COMPARTMENT_COLORS]].melt(
            id_vars="TIME", var_name="variable", value_name="value"
        )
        return long_df, compartment_df, has_susceptibility

    def plot_figure_s13(
        long_df, compartment_df, has_susceptibility, name, xmax, pathlib_stem
    ):
        prevalence_df = long_df[long_df["quantity"] == "prevalence"]
        susceptibility_df = long_df[long_df["quantity"] == "susceptibility"]

        with tp.teed(
            plt.subplots,
            nrows=3,
            ncols=1,
            squeeze=False,
            figsize=(4.6, 5.4),
            teeplot_outattrs={"a": f"{name.lower()}-strain-dynamics"},
            teeplot_show=True,
            teeplot_subdir=pathlib_stem,
        ) as (fig, axes_grid):
            axes = axes_grid[:, 0]
            ax_susc, ax_prev, ax_comp = axes
            sns.lineplot(
                data=susceptibility_df,
                x="TIME",
                y="value",
                hue="strain",
                hue_order=S13_STRAIN_ORDER,
                style="strain",
                style_order=S13_STRAIN_ORDER,
                dashes=DASHES,
                palette=PALETTE,
                linewidth=1.2,
                legend=False,
                ax=ax_susc,
            )
            ax_susc.set_ylabel("Host\nSusceptibility")
            ax_susc.set_xlabel("")
            ax_susc.set_xticklabels([])
            ax_susc.set_yticks([0.0, 0.5, 1.0])
            ax_susc.set_ylim(bottom=0.0)

            sns.lineplot(
                data=prevalence_df,
                x="TIME",
                y="value",
                hue="strain",
                hue_order=S13_STRAIN_ORDER,
                style="strain",
                style_order=S13_STRAIN_ORDER,
                dashes=DASHES,
                palette=PALETTE,
                linewidth=1.2,
                legend=True,
                ax=ax_prev,
            )
            ax_prev.set_ylabel("Strain\nPrevalence")
            ax_prev.set_xlabel("")
            ax_prev.set_xticklabels([])

            for comp_name, color in S13_COMPARTMENT_COLORS.items():
                comp_df = compartment_df[
                    compartment_df["variable"] == comp_name
                ]
                ax_comp.plot(
                    comp_df["TIME"],
                    comp_df["value"],
                    color=color,
                    linewidth=1.2,
                    linestyle=S13_COMPARTMENT_STYLES[comp_name],
                )
            ax_comp.set_ylabel("Compartment\nOccupancy")
            ax_comp.set_xlabel("Time")
            ax_comp.set_ylim(0.0, 0.3)

            for ax in axes:
                ax.set_xlim(left=0, right=xmax)

            sns.despine(fig=fig)
            fig.tight_layout()

            strain_legend = ax_prev.get_legend()
            strain_handles = strain_legend.legend_handles
            strain_labels = [t.get_text() for t in strain_legend.get_texts()]
            strain_legend.remove()
            strain_dummy = mlines.Line2D(
                [], [], linestyle="none", label="Strain"
            )
            fig.legend(
                handles=[strain_dummy, *strain_handles],
                labels=["Strain", *strain_labels],
                loc="upper center",
                bbox_to_anchor=(0.5, 1.1),
                ncol=len(strain_labels) + 1,
                frameon=False,
                handlelength=1.8,
                handletextpad=0.6,
                columnspacing=1.2,
            )

            comp_handles = [
                mlines.Line2D(
                    [],
                    [],
                    color=color,
                    linewidth=1.2,
                    linestyle=S13_COMPARTMENT_STYLES[comp_name],
                )
                for comp_name, color in S13_COMPARTMENT_COLORS.items()
            ]
            comp_labels = list(S13_COMPARTMENT_COLORS)
            comp_dummy = mlines.Line2D(
                [], [], linestyle="none", label="Host Compartment"
            )
            fig.legend(
                handles=[comp_dummy, *comp_handles],
                labels=["Host Compartment", *comp_labels],
                loc="upper center",
                bbox_to_anchor=(0.5, 1.055),
                ncol=len(comp_labels) + 1,
                frameon=False,
                handlelength=1.8,
                handletextpad=0.6,
                columnspacing=1.2,
            )
        return fig

    return (
        plot_figure,
        plot_figure_r0000,
        plot_figure_s13,
        prep_long_r0000,
        prep_long_s13,
    )


@app.cell
def plot_figs1a(fetch_osf, pathlib, plot_figure):
    _df = fetch_osf("shbde")
    _long_df, _has_susceptibility = prep_long(_df)
    plot_figure(
        _long_df,
        _has_susceptibility,
        "FigS1a",
        2000,
        pathlib.Path(__file__).stem,
    )
    return


@app.cell
def plot_figs1b(fetch_osf, pathlib, plot_figure):
    _df = fetch_osf("k5etm")
    _long_df, _has_susceptibility = prep_long(_df)
    plot_figure(
        _long_df,
        _has_susceptibility,
        "FigS1b",
        2000,
        pathlib.Path(__file__).stem,
    )
    return


@app.cell
def plot_figs1c(fetch_osf, pathlib, plot_figure):
    _df = fetch_osf("skwrg")
    _long_df, _has_susceptibility = prep_long(_df)
    plot_figure(
        _long_df,
        _has_susceptibility,
        "FigS1c",
        2000,
        pathlib.Path(__file__).stem,
    )
    return


@app.cell
def plot_figs1d(fetch_osf, pathlib, plot_figure):
    _df = fetch_osf("npkaj")
    _long_df, _has_susceptibility = prep_long(_df)
    plot_figure(
        _long_df,
        _has_susceptibility,
        "FigS1d",
        2000,
        pathlib.Path(__file__).stem,
    )
    return


@app.cell
def plot_figs2(fetch_osf, pathlib, plot_figure):
    _df = fetch_osf("4dv8c")
    _long_df, _has_susceptibility = prep_long(_df)
    plot_figure(
        _long_df,
        _has_susceptibility,
        "FigS2",
        3000,
        pathlib.Path(__file__).stem,
    )
    return


@app.cell
def plot_figs3(fetch_osf, pathlib, plot_figure):
    _df = fetch_osf("fs6ux")
    _long_df, _has_susceptibility = prep_long(_df)
    plot_figure(
        _long_df,
        _has_susceptibility,
        "FigS3",
        3000,
        pathlib.Path(__file__).stem,
    )
    return


@app.cell
def plot_figs4(fetch_osf, pathlib, plot_figure):
    _df = fetch_osf("j4e65")
    _long_df, _has_susceptibility = prep_long(_df)
    plot_figure(
        _long_df,
        _has_susceptibility,
        "FigS4",
        2000,
        pathlib.Path(__file__).stem,
    )
    return


@app.cell
def plot_figs5(fetch_osf, pathlib, plot_figure):
    _df = fetch_osf("f84ez")
    _long_df, _has_susceptibility = prep_long(_df)
    plot_figure(
        _long_df,
        _has_susceptibility,
        "FigS5",
        3000,
        pathlib.Path(__file__).stem,
    )
    return


@app.cell
def plot_figs6(fetch_osf, pathlib, plot_figure):
    _df = fetch_osf("95d8n")
    _long_df, _has_susceptibility = prep_long(_df)
    plot_figure(
        _long_df,
        _has_susceptibility,
        "FigS6",
        3000,
        pathlib.Path(__file__).stem,
    )
    return


@app.cell
def plot_figs7(fetch_osf, pathlib, plot_figure):
    _df = fetch_osf("ezumj")
    _long_df, _has_susceptibility = prep_long(_df)
    plot_figure(
        _long_df,
        _has_susceptibility,
        "FigS7",
        3000,
        pathlib.Path(__file__).stem,
    )
    return


@app.cell
def plot_figs8(fetch_osf, pathlib, plot_figure):
    _df = fetch_osf("ag2u3")
    _long_df, _has_susceptibility = prep_long(_df)
    plot_figure(
        _long_df,
        _has_susceptibility,
        "FigS8",
        3000,
        pathlib.Path(__file__).stem,
    )
    return


@app.cell
def plot_figs9a(fetch_osf, pathlib, plot_figure_r0000, prep_long_r0000):
    _df = fetch_osf("5qau7")
    _long_df, _r0000_df, _has_susceptibility = prep_long_r0000(_df)
    plot_figure_r0000(
        _long_df,
        _r0000_df,
        _has_susceptibility,
        "FigS9a",
        2000,
        pathlib.Path(__file__).stem,
    )
    return


@app.cell
def plot_figs9b(fetch_osf, pathlib, plot_figure_r0000, prep_long_r0000):
    _df = fetch_osf("zg3m2")
    _long_df, _r0000_df, _has_susceptibility = prep_long_r0000(_df)
    plot_figure_r0000(
        _long_df,
        _r0000_df,
        _has_susceptibility,
        "FigS9b",
        2000,
        pathlib.Path(__file__).stem,
    )
    return


@app.cell
def plot_figs9c(fetch_osf, pathlib, plot_figure_r0000, prep_long_r0000):
    _df = fetch_osf("m4pr3")
    _long_df, _r0000_df, _has_susceptibility = prep_long_r0000(_df)
    plot_figure_r0000(
        _long_df,
        _r0000_df,
        _has_susceptibility,
        "FigS9c",
        2000,
        pathlib.Path(__file__).stem,
    )
    return


@app.cell
def plot_figs10a(fetch_osf, pathlib, plot_figure):
    _df = fetch_osf("bjsw2")
    _long_df, _has_susceptibility = prep_long(_df)
    plot_figure(
        _long_df,
        _has_susceptibility,
        "FigS10a",
        3000,
        pathlib.Path(__file__).stem,
    )
    return


@app.cell
def plot_figs10b(fetch_osf, pathlib, plot_figure):
    _df = fetch_osf("pv6k8")
    _long_df, _has_susceptibility = prep_long(_df)
    plot_figure(
        _long_df,
        _has_susceptibility,
        "FigS10b",
        3000,
        pathlib.Path(__file__).stem,
    )
    return


@app.cell
def plot_figs11a(fetch_osf, pathlib, plot_figure):
    _df = fetch_osf("56nj4")
    _long_df, _has_susceptibility = prep_long(_df)
    plot_figure(
        _long_df,
        _has_susceptibility,
        "FigS11a",
        3000,
        pathlib.Path(__file__).stem,
    )
    return


@app.cell
def plot_figs11b(fetch_osf, pathlib, plot_figure):
    _df = fetch_osf("yrh2c")
    _long_df, _has_susceptibility = prep_long(_df)
    plot_figure(
        _long_df,
        _has_susceptibility,
        "FigS11b",
        3000,
        pathlib.Path(__file__).stem,
    )
    return


@app.cell
def plot_figs12a(fetch_osf, pathlib, plot_figure):
    _df = fetch_osf("xb5sh")
    _long_df, _has_susceptibility = prep_long(_df)
    plot_figure(
        _long_df,
        _has_susceptibility,
        "FigS12a",
        2000,
        pathlib.Path(__file__).stem,
    )
    return


@app.cell
def plot_figs12b(fetch_osf, pathlib, plot_figure):
    _df = fetch_osf("qzwcd")
    _long_df, _has_susceptibility = prep_long(_df)
    plot_figure(
        _long_df,
        _has_susceptibility,
        "FigS12b",
        2000,
        pathlib.Path(__file__).stem,
    )
    return


@app.cell
def plot_figs12c(fetch_osf, pathlib, plot_figure):
    _df = fetch_osf("vkqt5")
    _long_df, _has_susceptibility = prep_long(_df)
    plot_figure(
        _long_df,
        _has_susceptibility,
        "FigS12c",
        2000,
        pathlib.Path(__file__).stem,
    )
    return


@app.cell
def plot_figs13a(fetch_osf, pathlib, plot_figure_s13, prep_long_s13):
    _df = fetch_osf("n36yh")
    _long_df, _compartment_df, _has_susceptibility = prep_long_s13(_df)
    plot_figure_s13(
        _long_df,
        _compartment_df,
        _has_susceptibility,
        "FigS13a",
        2000,
        pathlib.Path(__file__).stem,
    )
    return


@app.cell
def plot_figs13b(fetch_osf, pathlib, plot_figure_s13, prep_long_s13):
    _df = fetch_osf("kvjrw")
    _long_df, _compartment_df, _has_susceptibility = prep_long_s13(_df)
    plot_figure_s13(
        _long_df,
        _compartment_df,
        _has_susceptibility,
        "FigS13b",
        2000,
        pathlib.Path(__file__).stem,
    )
    return


@app.cell
def plot_figs13c(fetch_osf, pathlib, plot_figure_s13, prep_long_s13):
    _df = fetch_osf("e6cv9")
    _long_df, _compartment_df, _has_susceptibility = prep_long_s13(_df)
    plot_figure_s13(
        _long_df,
        _compartment_df,
        _has_susceptibility,
        "FigS13c",
        2000,
        pathlib.Path(__file__).stem,
    )
    return


@app.cell
def plot_tables14(fetch_osf, pathlib, plot_figure):
    _df = fetch_osf("x5jze")
    _long_df, _has_susceptibility = prep_long(_df)
    plot_figure(
        _long_df,
        _has_susceptibility,
        "TableS14",
        3000,
        pathlib.Path(__file__).stem,
    )
    return


@app.cell
def plot_tables15a(fetch_osf, pathlib, plot_figure):
    _df = fetch_osf("g8rz9")
    _long_df, _has_susceptibility = prep_long(_df)
    plot_figure(
        _long_df,
        _has_susceptibility,
        "TableS15a",
        3000,
        pathlib.Path(__file__).stem,
    )
    return


@app.cell
def plot_tables15b(fetch_osf, pathlib, plot_figure):
    _df = fetch_osf("kepmh")
    _long_df, _has_susceptibility = prep_long(_df)
    plot_figure(
        _long_df,
        _has_susceptibility,
        "TableS15b",
        3000,
        pathlib.Path(__file__).stem,
    )
    return


@app.cell
def plot_tables16a(fetch_osf, pathlib, plot_figure):
    _df = fetch_osf("z4r2h")
    _long_df, _has_susceptibility = prep_long(_df)
    plot_figure(
        _long_df,
        _has_susceptibility,
        "TableS16a",
        3000,
        pathlib.Path(__file__).stem,
    )
    return


@app.cell
def plot_tables16aa(fetch_osf, pathlib, plot_figure):
    _df = fetch_osf("6afds")
    _long_df, _has_susceptibility = prep_long(_df)
    plot_figure(
        _long_df,
        _has_susceptibility,
        "TableS16aa",
        3000,
        pathlib.Path(__file__).stem,
    )
    return


@app.cell
def plot_tables16b(fetch_osf, pathlib, plot_figure):
    _df = fetch_osf("n93ae")
    _long_df, _has_susceptibility = prep_long(_df)
    plot_figure(
        _long_df,
        _has_susceptibility,
        "TableS16b",
        3000,
        pathlib.Path(__file__).stem,
    )
    return


@app.cell
def plot_tables17(fetch_osf, pathlib, plot_figure):
    _df = fetch_osf("zu3j6")
    _long_df, _has_susceptibility = prep_long(_df)
    plot_figure(
        _long_df,
        _has_susceptibility,
        "TableS17",
        3000,
        pathlib.Path(__file__).stem,
    )
    return


@app.cell
def plot_tables18a(fetch_osf, pathlib, plot_figure):
    # TableS18 holds two conditions in one file, distinguished by
    # ":1"/":2" column suffixes (fetch_osf strips only ":1"); split
    # into two standalone prevalence-only figures.
    _df = fetch_osf("ksva6")
    _cond_df = _df[["TIME", "J13", "J02", "J03", "J12"]]
    _long_df, _has_susceptibility = prep_long(_cond_df)
    plot_figure(
        _long_df,
        _has_susceptibility,
        "TableS18a",
        3000,
        pathlib.Path(__file__).stem,
    )
    return


@app.cell
def plot_tables18b(fetch_osf, pathlib, plot_figure):
    _df = fetch_osf("ksva6")
    _cond_df = _df[["TIME", "J13:2", "J02:2", "J03:2", "J12:2"]].rename(
        columns=lambda c: c.replace(":2", "")
    )
    _long_df, _has_susceptibility = prep_long(_cond_df)
    plot_figure(
        _long_df,
        _has_susceptibility,
        "TableS18b",
        3000,
        pathlib.Path(__file__).stem,
    )
    return


@app.cell
def plot_tables19(fetch_osf, pathlib, plot_figure):
    _df = fetch_osf("x45tc")
    _long_df, _has_susceptibility = prep_long(_df)
    plot_figure(
        _long_df,
        _has_susceptibility,
        "TableS19",
        3000,
        pathlib.Path(__file__).stem,
    )
    return


@app.cell
def plot_tables20(fetch_osf, pathlib, plot_figure):
    _df = fetch_osf("fya3n")
    _long_df, _has_susceptibility = prep_long(_df)
    plot_figure(
        _long_df,
        _has_susceptibility,
        "TableS20",
        3000,
        pathlib.Path(__file__).stem,
    )
    return


@app.cell
def plot_tables21a(fetch_osf, pathlib, plot_figure):
    _df = fetch_osf("6jctb")
    _long_df, _has_susceptibility = prep_long(_df)
    plot_figure(
        _long_df,
        _has_susceptibility,
        "TableS21a",
        3000,
        pathlib.Path(__file__).stem,
    )
    return


@app.cell
def plot_tables21b(fetch_osf, pathlib, plot_figure):
    _df = fetch_osf("aw9hn")
    _long_df, _has_susceptibility = prep_long(_df)
    plot_figure(
        _long_df,
        _has_susceptibility,
        "TableS21b",
        3000,
        pathlib.Path(__file__).stem,
    )
    return


@app.cell
def plot_tables21c(fetch_osf, pathlib, plot_figure):
    _df = fetch_osf("3uyn6")
    _long_df, _has_susceptibility = prep_long(_df)
    plot_figure(
        _long_df,
        _has_susceptibility,
        "TableS21c",
        3000,
        pathlib.Path(__file__).stem,
    )
    return


@app.cell
def plot_tables21d(fetch_osf, pathlib, plot_figure):
    _df = fetch_osf("kzuar")
    _long_df, _has_susceptibility = prep_long(_df)
    plot_figure(
        _long_df,
        _has_susceptibility,
        "TableS21d",
        3000,
        pathlib.Path(__file__).stem,
    )
    return


@app.cell
def plot_tables22(fetch_osf, pathlib, plot_figure):
    _df = fetch_osf("8f2hb")
    _long_df, _has_susceptibility = prep_long(_df)
    plot_figure(
        _long_df,
        _has_susceptibility,
        "TableS22",
        3000,
        pathlib.Path(__file__).stem,
    )
    return


@app.cell
def plot_tables23a(fetch_osf, pathlib, plot_figure):
    _df = fetch_osf("k54ud")
    _long_df, _has_susceptibility = prep_long(_df)
    plot_figure(
        _long_df,
        _has_susceptibility,
        "TableS23a",
        3000,
        pathlib.Path(__file__).stem,
    )
    return


@app.cell
def plot_tables23b(fetch_osf, pathlib, plot_figure):
    _df = fetch_osf("g4h8x")
    _long_df, _has_susceptibility = prep_long(_df)
    plot_figure(
        _long_df,
        _has_susceptibility,
        "TableS23b",
        3000,
        pathlib.Path(__file__).stem,
    )
    return


@app.cell
def plot_tables23c(fetch_osf, pathlib, plot_figure):
    _df = fetch_osf("evq5m")
    _long_df, _has_susceptibility = prep_long(_df)
    plot_figure(
        _long_df,
        _has_susceptibility,
        "TableS23c",
        3000,
        pathlib.Path(__file__).stem,
    )
    return


@app.cell
def plot_tables24a(fetch_osf, pathlib, plot_figure):
    _df = fetch_osf("pjnsc")
    _long_df, _has_susceptibility = prep_long(_df)
    plot_figure(
        _long_df,
        _has_susceptibility,
        "TableS24a",
        3000,
        pathlib.Path(__file__).stem,
    )
    return


@app.cell
def plot_tables24b(fetch_osf, pathlib, plot_figure):
    _df = fetch_osf("9rxpm")
    _long_df, _has_susceptibility = prep_long(_df)
    plot_figure(
        _long_df,
        _has_susceptibility,
        "TableS24b",
        3000,
        pathlib.Path(__file__).stem,
    )
    return


if __name__ == "__main__":
    app.run()
