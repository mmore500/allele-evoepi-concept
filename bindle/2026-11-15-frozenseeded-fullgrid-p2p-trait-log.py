import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell
def import_std():
    import pathlib

    return (pathlib,)


@app.cell
def import_pkg():
    import downstream
    import imageio
    import marimo as mo
    import numpy as np
    import polars as pl
    import seaborn as sns
    from teeplot import teeplot as tp
    from tqdm.auto import tqdm
    from watermark import watermark

    from pylib.munge import addend_groups, squeeze_int, union_upsample

    return (
        addend_groups,
        downstream,
        imageio,
        mo,
        np,
        pl,
        sns,
        squeeze_int,
        tp,
        tqdm,
        union_upsample,
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
def delimit_intro(mo):
    mo.md("""
    # 2026-11-15-frozenseeded-fullgrid-p2p-trait-log

    Spatial timeseries of logged trait prevalence for the 2026-11-15
    frozen-seeded full-grid run with point-to-point immigrant exchange
    (wse-async-ga trait logger data, OSF `jwuz7`).
    Adapted from `binder/wse-denovo-spatial2d-explicitsite-timeseries.ipynb`
    in mmore500/hypermutator-dynamics.
    """)
    return


@app.cell
def setup_outdir(pathlib):
    nbname = pathlib.Path(__file__).stem
    outdir = pathlib.Path("outdata") / nbname
    outdir.mkdir(parents=True, exist_ok=True)
    outdir
    return nbname, outdir


@app.cell(hide_code=True)
def delimit_data(mo):
    mo.md("""
    ## Prep Data
    """)
    return


@app.cell
def load_data(pl):
    dfxs = pl.read_parquet("https://osf.io/jwuz7/download").with_row_index(
        "dstream_data_id",
    )
    dfxs.describe()
    return (dfxs,)


@app.cell
def define_unpack(addend_groups, downstream, pl, squeeze_int, union_upsample):
    def unpack_logger_timeseries_data(df: pl.DataFrame) -> pl.DataFrame:

        # sample every 64th tile
        df = df.filter((pl.col("col") % 8 == 0) & (pl.col("row") % 8 == 0))

        # unpack dstream data items from logger hex strings
        dfu = downstream.dataframe.unpack_data_packed(df)
        df = downstream.dataframe.explode_lookup_unpacked(
            dfu.with_columns(
                # fast forward loggers to at least dstream_S entries
                # (i.e., buffer filled)
                # these will be semantically-correct zero values
                dstream_T=pl.max_horizontal(
                    pl.col("dstream_T"), pl.col("dstream_S")
                ),
            ),
            value_type="uint8",
        )

        # join metadata onto unpacked data
        df = df.join(
            dfu.select(
                "dstream_data_id",
                "CEREBRASLIB_TRAITLOGGER_DILATION",
                "col",
                "replicate",
                "row",
                "tile",
            ),
            on="dstream_data_id",
        )

        # apply dilation to dstream_T and dstream_Tbar
        df = df.with_columns(
            dstream_T=pl.col("dstream_T")
            * pl.col("CEREBRASLIB_TRAITLOGGER_DILATION"),
            dstream_Tbar=pl.col("dstream_Tbar")
            * pl.col("CEREBRASLIB_TRAITLOGGER_DILATION"),
        )

        # addend time time series groups with final zero value entry
        df = addend_groups(
            df,
            group_by=["replicate", "tile"],
            aggs=(
                pl.col("col").unique().pipe(squeeze_int),
                (pl.col("dstream_Tbar") + 1).max().alias("dstream_Tbar"),
                pl.lit(0).alias("dstream_value"),
                pl.col("row").unique().pipe(squeeze_int),
            ),
            inner_only=True,
        )

        # upsample to fill in dstream_Tbar dropped by some loggers
        # prevents sampling biases from longer- vs shorter-running loggers
        df = union_upsample(
            df,
            upsample="dstream_Tbar",
            group_by="tile",
            fill_null_ops=[
                (pl.all(), {"strategy": "forward"}),
                (pl.col("dstream_Tbar", "dstream_value"), {"strategy": "zero"}),
                (pl.all(), {"strategy": "backward"}),
            ],
        )
        assert len(df["dstream_Tbar"].value_counts()["count"].unique()) == 1

        return df

    return (unpack_logger_timeseries_data,)


@app.cell
def unpack_data(dfxs, pl, tqdm, unpack_logger_timeseries_data):
    df = pl.concat(
        [
            unpack_logger_timeseries_data(_df)
            for _, _df in tqdm(dfxs.group_by("replicate"))
        ],
    )
    df
    return (df,)


@app.cell(hide_code=True)
def delimit_plot(mo):
    mo.md("""
    ## Plot Data
    """)
    return


@app.cell
def summarize_data(df, pl):
    # take the mean of dstream_value at each dstream_Tbar across tiles
    df_mean = df.group_by("replicate", "dstream_Tbar").agg(
        pl.col("dstream_value").mean(),
    )
    return (df_mean,)


@app.cell
def plot_prevalence(df_mean, nbname, sns, tp):
    with tp.teed(
        sns.relplot,
        data=df_mean.sort("dstream_Tbar"),
        x="dstream_Tbar",
        y="dstream_value",
        hue="replicate",
        alpha=0.5,
        aspect=1.5,
        errorbar=None,
        height=2.5,
        linewidth=1,
        legend=False,
        kind="line",
        marker="d",
        markersize=4,
        teeplot_subdir=nbname,
        teeplot_outexclude=["marker"],
    ) as _g:
        _g.set(xlim=(10, 80000))
        _g.set(ylim=(0.00001, 2))
        _g.set_axis_labels(x_var="Generations", y_var="Trait\nPrevalence")
        for _ax in _g.axes.flat:
            _ax.axhline(1, ls="--", color="black", linewidth=1, zorder=-10)
            _ax.set_yscale("log")
            _ax.set_xscale("log")
    return


@app.cell(hide_code=True)
def delimit_animate(mo):
    mo.md("""
    ## Animate Data
    """)
    return


@app.cell
def animate_data(df, imageio, np, outdir, pl, tqdm):
    for (_replicate,), _dfr in tqdm(df.group_by("replicate")):
        _frames = []
        for (_dstream_Tbar,), _dff in _dfr.sort("dstream_Tbar").group_by(
            "dstream_Tbar", maintain_order=True
        ):
            _dff = _dff.with_columns(
                col_rank=pl.col("col").rank(method="dense").cast(pl.Int32) - 1,
                row_rank=pl.col("row").rank(method="dense").cast(pl.Int32) - 1,
            )

            _num_rows = _dff["row_rank"].max() + 1
            _num_cols = _dff["col_rank"].max() + 1

            _frame = np.empty((_num_rows, _num_cols), dtype=np.uint8)
            _frame[_dff["row_rank"], _dff["col_rank"]] = (
                _dff["dstream_value"] * 255
            )
            _frame = np.kron(_frame, np.ones((4, 4), dtype=_frame.dtype))
            _frames.append((_dstream_Tbar, _frame))

            _framedir = outdir / f"a=traitframes+rep={_replicate}"
            _framedir.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(
                _framedir / f"dstream_Tbar={_dstream_Tbar:06}+ext=.png",
                _frame.T,
            )

        print(len(_frames))
        imageio.mimsave(
            outdir / f"a=traits+rep={_replicate}+ext=.gif",
            [_f for _, _f in _frames],
            duration=0.2,
        )

        # sweep completes within first ~1000 generations; animate that window
        _sweep = [_f.T for _t, _f in _frames if _t <= 1000]
        print(len(_sweep))
        imageio.mimsave(
            outdir / f"a=traits-sweep+rep={_replicate}+tbar_max=1000+ext=.gif",
            _sweep,
            duration=0.1,
            loop=0,
        )
    return


if __name__ == "__main__":
    app.run()
