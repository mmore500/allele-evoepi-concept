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
    import matplotlib.ticker as mticker
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
        mticker,
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

    Prototype redraws of a batch of figures from Simon, Koopman,
    Eisenberg, Zaman, Moreno & Polanco, "An allele-based model of
    coronavirus evolution under population immunity" (medRxiv,
    <https://doi.org/10.1101/2025.09.15.25335781>, v3), reusing the
    strain-dynamics rendering style already established for Figure 3
    in `2026-09-26-figrender.py`.

    Each underlying trajectory is hosted on OSF as a `FigN.csv` (or,
    for v3 `Fig11b`, tab-delimited `.txt`) file with the same column
    convention as `Fig3.csv`: `TIME` plus a `J**` prevalence column
    and (where present) a `SuscToI**` susceptibility column per
    2-allele strain (`02`, `03`, `12`, `13`). Datasets for Figures
    10, 11, and 12 omit the susceptibility columns and are rendered as a
    single prevalence panel; the rest render as a two-panel
    susceptibility+prevalence figure matching Figure 3's layout.

    Figure numbers below follow the current manuscript draft. OSF
    file names follow the v3 preprint, which had one more figure
    before this point: draft Figures 7-12 are v3 Figures 8-13, and v3
    Figure 7 (absent from the draft) is kept as `FigV3-7`. Figures 10
    and 11 are each a single multi-panel manuscript figure (10: left/
    center/right for joint-immunity `mij` = 0, 0.2, 0.4; 11: left/
    right for Case 1 allele-based vs. Case 2 strain-based immunity),
    split here into separate `10a`/`10b`/`10c` and `11a`/`11b` cells
    to match how the underlying data is hosted on OSF. Combined `Fig10`
    and `Fig11` cells also render each as a single horizontal figure
    with shared axes and one legend.
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
    shared `plot_manuscript_figure` helper below. The x-axis maximum
    for each is set to match the corresponding original figure.
    """)
    return


@app.cell
def def_plot(mcolors, mlegend_handler, mlines, mticker, plt, sns, tp):
    STRAIN_ORDER = ["02", "03", "12", "13"]
    # Same named-color palette as 2026-09-26-figrender.py, so strain
    # identity reads consistently across notebooks.
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

    def style_dotted_strain(ax):
        # Widen 12's dotted line beyond the shared base linewidth so its
        # sparser dots stay legible, and add a thin solid underlay so
        # the trajectory still reads as continuous between dots.
        for line in list(ax.lines):
            if mcolors.to_rgb(line.get_color()) == DOTTED_RGB:
                xdata, ydata = line.get_data()
                ax.plot(
                    xdata,
                    ydata,
                    color=line.get_color(),
                    linewidth=0.6,
                    zorder=line.get_zorder() - 0.1,
                    solid_capstyle="round",
                )
                line.set_linewidth(2.8)

    def add_strain_legend(fig, legend_ax, figheight, location="top"):
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
        if location != "top":
            # Titled "Strain", either a single column beside the axes or
            # an enlarged two-column key centered in a spare axes passed
            # as location.
            if location == "right":
                placement = {
                    "loc": "center left",
                    "bbox_to_anchor": (1.0, 0.5),
                    "ncol": 1,
                    "handlelength": 1.8,
                }
            else:
                placement = {
                    "loc": "center",
                    "bbox_to_anchor": location.get_position().bounds,
                    "bbox_transform": fig.transFigure,
                    "ncol": 2,
                    "fontsize": "x-large",
                    "title_fontsize": "x-large",
                    "handlelength": 2.6,
                    "columnspacing": 1.6,
                }
            fig.legend(
                handles=handles,
                labels=labels,
                title="Strain",
                **placement,
                frameon=False,
                handletextpad=0.6,
                handler_map={tuple: mlegend_handler.HandlerTuple(ndivide=1)},
            )
            return
        dummy = mlines.Line2D([], [], linestyle="none", label="Strain")
        fig.legend(
            handles=[dummy, *handles],
            labels=["Strain", *labels],
            loc="upper center",
            # Hold the legend a fixed physical distance above the axes,
            # regardless of figure height.
            bbox_to_anchor=(0.5, 1 + 0.23 / figheight),
            ncol=len(labels) + 1,
            frameon=False,
            handlelength=1.8,
            handletextpad=0.6,
            columnspacing=1.2,
            handler_map={tuple: mlegend_handler.HandlerTuple(ndivide=1)},
        )

    def plot_manuscript_figure(
        long_df,
        has_susceptibility,
        name,
        xmax,
        pathlib_stem,
        show_susceptibility=True,
        prevalence_ymax=None,
    ):
        prevalence_df = long_df[long_df["quantity"] == "prevalence"]
        has_susceptibility = has_susceptibility and show_susceptibility
        nrows = 2 if has_susceptibility else 1
        figheight = 3.9 if has_susceptibility else 2.2
        figsize = (4.6, figheight)

        outattrs = {"a": f"{name.lower()}-strain-dynamics"}
        if not show_susceptibility:
            outattrs["susceptibility"] = False
        if prevalence_ymax is not None:
            outattrs["prevalence-ymax"] = prevalence_ymax

        with tp.teed(
            plt.subplots,
            nrows=nrows,
            ncols=1,
            squeeze=False,
            figsize=figsize,
            teeplot_outattrs=outattrs,
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
            # Cap the prevalence y-axis, clipping tall early peaks so
            # later low-prevalence dynamics are legible.
            if prevalence_ymax is not None:
                # Match matplotlib's default 5% margin below zero.
                ax_prev.set_ylim(-0.05 * prevalence_ymax, prevalence_ymax)
                ax_prev.set_yticks([0.0, prevalence_ymax / 2, prevalence_ymax])
            if legend_ax is None:
                legend_ax = ax_prev

            for ax in axes:
                style_dotted_strain(ax)
                ax.set_xlim(left=0, right=xmax)

            sns.despine(fig=fig)
            fig.tight_layout()
            add_strain_legend(fig, legend_ax, figheight)
        return fig

    def plot_manuscript_panels(long_dfs, name, xmax, pathlib_stem, ncols=None):
        # Render several prevalence-only datasets as one figure, sharing
        # axes and a single legend, in a grid of ncols columns (default:
        # one row). Each panel is a quarter narrower than a standalone
        # prevalence figure. With spare grid cells, the legend fills the
        # first one; otherwise it sits to the right in one column.
        npanels = len(long_dfs)
        ncols = npanels if ncols is None else ncols
        nrows = -(-npanels // ncols)
        figheight = 2.2 * nrows
        figsize = (4.6 * 0.75 * ncols, figheight)

        with tp.teed(
            plt.subplots,
            nrows=nrows,
            ncols=ncols,
            sharex=True,
            sharey=True,
            squeeze=False,
            figsize=figsize,
            teeplot_outattrs={"a": f"{name.lower()}-strain-dynamics"},
            teeplot_show=True,
            teeplot_subdir=pathlib_stem,
        ) as (fig, axes_grid):
            axes = axes_grid.flatten()
            for i, (ax, long_df) in enumerate(zip(axes, long_dfs)):
                sns.lineplot(
                    data=long_df[long_df["quantity"] == "prevalence"],
                    x="TIME",
                    y="value",
                    hue="strain",
                    hue_order=STRAIN_ORDER,
                    style="strain",
                    style_order=STRAIN_ORDER,
                    dashes=DASHES,
                    palette=PALETTE,
                    linewidth=1.2,
                    legend=(i == 0),
                    ax=ax,
                )
                # Label only outer edges: y on the left column, x on
                # panels with no panel below them.
                if i % ncols == 0:
                    ax.set_ylabel("Strain\nPrevalence")
                else:
                    ax.set_ylabel("")
                    ax.tick_params(axis="y", labelleft=False)
                if i + ncols >= npanels:
                    ax.set_xlabel("Time")
                    # Shared-x subplots hide labels above the bottom row.
                    ax.xaxis.label.set_visible(True)
                    ax.tick_params(axis="x", labelbottom=True)
                else:
                    ax.set_xlabel("")
                    ax.tick_params(axis="x", labelbottom=False)
                style_dotted_strain(ax)
                ax.set_xlim(left=0, right=xmax)
                # Sparse ticks, dropping the right-edge tick so its label
                # doesn't collide with the next panel's zero.
                ax.xaxis.set_major_locator(
                    mticker.MaxNLocator(nbins=3, prune="upper")
                )
            spare_axes = axes[npanels:]
            for ax in spare_axes:
                ax.axis("off")

            sns.despine(fig=fig)
            # Snug inter-panel padding (in font-size units).
            fig.tight_layout(h_pad=0.2, w_pad=0.4)
            if len(spare_axes):
                # Close the row gap reserved for x labels of panels above
                # spare cells; those labels hang into the empty cell.
                fig.subplots_adjust(hspace=0.08)
            add_strain_legend(
                fig,
                axes[0],
                figheight,
                location=spare_axes[0] if len(spare_axes) else "right",
            )
        return fig

    return plot_manuscript_figure, plot_manuscript_panels


@app.cell
def plot_fig2(fetch_osf, pathlib, plot_manuscript_figure):
    _df = fetch_osf("sz4f9")
    _long_df, _has_susceptibility = prep_long(_df)
    plot_manuscript_figure(
        _long_df,
        _has_susceptibility,
        "Fig2",
        2000,
        pathlib.Path(__file__).stem,
    )
    return


@app.cell
def plot_fig4(fetch_osf, pathlib, plot_manuscript_figure):
    _df = fetch_osf("sd6ja")
    _long_df, _has_susceptibility = prep_long(_df)
    plot_manuscript_figure(
        _long_df,
        _has_susceptibility,
        "Fig4",
        3000,
        pathlib.Path(__file__).stem,
    )
    return


@app.cell
def plot_fig5(fetch_osf, pathlib, plot_manuscript_figure):
    _df = fetch_osf("wx3du")
    _long_df, _has_susceptibility = prep_long(_df)
    plot_manuscript_figure(
        _long_df,
        _has_susceptibility,
        "Fig5",
        3000,
        pathlib.Path(__file__).stem,
        show_susceptibility=False,
        prevalence_ymax=0.1,
    )
    return


@app.cell
def plot_fig6(fetch_osf, pathlib, plot_manuscript_figure):
    _df = fetch_osf("xjdnq")
    _long_df, _has_susceptibility = prep_long(_df)
    plot_manuscript_figure(
        _long_df,
        _has_susceptibility,
        "Fig6",
        3000,
        pathlib.Path(__file__).stem,
        prevalence_ymax=0.1,
    )
    return


@app.cell
def plot_fig_v3_7(fetch_osf, pathlib, plot_manuscript_figure):
    _df = fetch_osf("vmw46")
    _long_df, _has_susceptibility = prep_long(_df)
    plot_manuscript_figure(
        _long_df,
        _has_susceptibility,
        "FigV3-7",
        2000,
        pathlib.Path(__file__).stem,
    )
    return


@app.cell
def plot_fig7(fetch_osf, pathlib, plot_manuscript_figure):
    _df = fetch_osf("tzqxv")
    _long_df, _has_susceptibility = prep_long(_df)
    plot_manuscript_figure(
        _long_df,
        _has_susceptibility,
        "Fig7",
        3000,
        pathlib.Path(__file__).stem,
        prevalence_ymax=0.1,
    )
    return


@app.cell
def plot_fig8(fetch_osf, pathlib, plot_manuscript_figure):
    _df = fetch_osf("vbt2y")
    _long_df, _has_susceptibility = prep_long(_df)
    plot_manuscript_figure(
        _long_df,
        _has_susceptibility,
        "Fig8",
        2000,
        pathlib.Path(__file__).stem,
    )
    return


@app.cell
def plot_fig9(fetch_osf, pathlib, plot_manuscript_figure):
    _df = fetch_osf("e67fb")
    _long_df, _has_susceptibility = prep_long(_df)
    plot_manuscript_figure(
        _long_df,
        _has_susceptibility,
        "Fig9",
        3000,
        pathlib.Path(__file__).stem,
    )
    return


@app.cell
def plot_fig10a(fetch_osf, pathlib, plot_manuscript_figure):
    _df = fetch_osf("wkd37")
    _long_df, _has_susceptibility = prep_long(_df)
    fig10a = plot_manuscript_figure(
        _long_df,
        _has_susceptibility,
        "Fig10a",
        1200,
        pathlib.Path(__file__).stem,
    )
    return (fig10a,)


@app.cell
def plot_fig10b(fetch_osf, pathlib, plot_manuscript_figure):
    _df = fetch_osf("shm8d", sep="\t")
    _long_df, _has_susceptibility = prep_long(_df)
    fig10b = plot_manuscript_figure(
        _long_df,
        _has_susceptibility,
        "Fig10b",
        1200,
        pathlib.Path(__file__).stem,
    )
    return (fig10b,)


@app.cell
def plot_fig10c(fetch_osf, pathlib, plot_manuscript_figure):
    _df = fetch_osf("3uxqd")
    _long_df, _has_susceptibility = prep_long(_df)
    fig10c = plot_manuscript_figure(
        _long_df,
        _has_susceptibility,
        "Fig10c",
        1200,
        pathlib.Path(__file__).stem,
    )
    return (fig10c,)


@app.cell
def plot_fig11a(fetch_osf, pathlib, plot_manuscript_figure):
    _df = fetch_osf("zy5aq")
    _long_df, _has_susceptibility = prep_long(_df)
    fig11a = plot_manuscript_figure(
        _long_df,
        _has_susceptibility,
        "Fig11a",
        1200,
        pathlib.Path(__file__).stem,
    )
    return (fig11a,)


@app.cell
def plot_fig11b(fetch_osf, pathlib, plot_manuscript_figure):
    _df = fetch_osf("uk5sc")
    _long_df, _has_susceptibility = prep_long(_df)
    fig11b = plot_manuscript_figure(
        _long_df,
        _has_susceptibility,
        "Fig11b",
        1200,
        pathlib.Path(__file__).stem,
    )
    return (fig11b,)


@app.cell
def def_concat(pathlib):
    from PIL import Image

    def concat_panel_renders(names, name, pathlib_stem):
        # Paste standalone panel renders side by side as-is, each with
        # its own legend and axes, into one image named for the figure.
        outdir = pathlib.Path("teeplots") / pathlib_stem
        images = [
            Image.open(
                outdir / f"a={n.lower()}-strain-dynamics+viz=subplots+ext=.png"
            ).convert("RGB")
            for n in names
        ]
        width = sum(im.width for im in images)
        height = max(im.height for im in images)
        canvas = Image.new("RGB", (width, height), "white")
        x = 0
        for im in images:
            canvas.paste(im, (x, (height - im.height) // 2))
            x += im.width
        dpi = images[0].info.get("dpi", (300, 300))
        for ext in ".png", ".pdf":
            outpath = (
                outdir
                / f"a={name.lower()}-strain-dynamics+viz=concat+ext={ext}"
            )
            canvas.save(outpath, dpi=dpi)
            print(outpath)
        return canvas

    return (concat_panel_renders,)


@app.cell
def concat_fig10(concat_panel_renders, fig10a, fig10b, fig10c, pathlib):
    concat_panel_renders(
        ["Fig10a", "Fig10b", "Fig10c"], "Fig10", pathlib.Path(__file__).stem
    )
    return


@app.cell
def concat_fig11(concat_panel_renders, fig11a, fig11b, pathlib):
    concat_panel_renders(
        ["Fig11a", "Fig11b"], "Fig11", pathlib.Path(__file__).stem
    )
    return


@app.cell
def plot_fig10(fetch_osf, pathlib, plot_manuscript_panels):
    plot_manuscript_panels(
        [
            prep_long(fetch_osf("wkd37"))[0],
            prep_long(fetch_osf("shm8d", sep="\t"))[0],
            prep_long(fetch_osf("3uxqd"))[0],
        ],
        "Fig10",
        1200,
        pathlib.Path(__file__).stem,
        ncols=2,
    )
    return


@app.cell
def plot_fig11(fetch_osf, pathlib, plot_manuscript_panels):
    plot_manuscript_panels(
        [
            prep_long(fetch_osf("zy5aq"))[0],
            prep_long(fetch_osf("uk5sc"))[0],
        ],
        "Fig11",
        1200,
        pathlib.Path(__file__).stem,
    )
    return


@app.cell
def plot_fig12(fetch_osf, pathlib, plot_manuscript_figure):
    _df = fetch_osf("xsk4t")
    _long_df, _has_susceptibility = prep_long(_df)
    plot_manuscript_figure(
        _long_df,
        _has_susceptibility,
        "Fig12",
        2000,
        pathlib.Path(__file__).stem,
    )
    return


if __name__ == "__main__":
    app.run()
