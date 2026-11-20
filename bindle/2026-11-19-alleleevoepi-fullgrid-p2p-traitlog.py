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
        squeeze_int,
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
    # 2026-11-19-alleleevoepi-fullgrid-p2p-traitlog

    Animates strains of each PE's sampled host (0th population member)
    over time, from the trait logger of wse-async-ga workflow
    `2026-11-19-alleleevoepi-fullgrid-p2p-traitlog` (genome
    `genome_alleleevoepi_2026_11_19`, 2 sites).

    The trait logger multiplexes three single-bit records per PE
    (infected, site 0 allele, site 1 allele); downstream demultiplexes
    them by the `dstream_storage_multiplex` column.

    Colors: black, uninfected; white, wildtype; blue, site 0 mutant; red,
    site 1 mutant; purple, double mutant.
    """)
    return


@app.cell
def setup_outdir(pathlib):
    nbname = pathlib.Path(__file__).stem
    outdir = pathlib.Path("outdata") / nbname
    outdir.mkdir(parents=True, exist_ok=True)
    outdir
    return (outdir,)


@app.cell(hide_code=True)
def delimit_data(mo):
    mo.md("""
    ## Get Data
    """)
    return


@app.cell
def configure_data():
    # trait logger records, one OSF file per run
    osf_ids = ["hu2w5", "cxkqu", "bezac", "dtc5w", "v5hjw"]

    config_columns = [
        "dispatchSlug",
        "globalSeed",
        "nRow",
        "nCol",
        "nRowSubgrid",
        "nColSubgrid",
        "tilePopSize",
        "tournSize",
        "nCycle",
        "evoepi_waning_rate",
        "evoepi_seed_pe_stride",
        "ASYNC_GA_P2P_SEND_PROBABILITY",
        "dstream_storage_multiplex",
    ]
    return config_columns, osf_ids


@app.cell(hide_code=True)
def delimit_unpack(mo):
    mo.md("""
    ## Unpack Data
    """)
    return


@app.cell
def define_unpack(addend_groups, downstream, pl, squeeze_int, union_upsample):
    def unpack_strain_timeseries(df: pl.DataFrame) -> pl.DataFrame:

        # sample every 64th tile
        df = df.filter((pl.col("col") % 8 == 0) & (pl.col("row") % 8 == 0))

        # unpack dstream data items from logger hex strings, demultiplexing
        # infected, site 0 allele, and site 1 allele records
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
            dfu.select("dstream_data_id", "col", "row", "tile").unique(
                "dstream_data_id",
            ),
            on="dstream_data_id",
        )

        # apply dilation to dstream_Tbar, then one column per record
        df = (
            df.with_columns(
                dstream_Tbar=pl.col("dstream_Tbar")
                * pl.col("dstream_T_dilation"),
            )
            .pivot(
                on="dstream_storage_multiplex_id",
                index=["tile", "row", "col", "dstream_Tbar"],
                values="dstream_value",
            )
            .rename({"0": "infected", "1": "site0", "2": "site1"})
        )

        # addend time series groups with final uninfected entry, after
        # logger's last seen time (i.e., last time any record was set)
        df = addend_groups(
            df,
            group_by="tile",
            aggs=(
                pl.col("col").unique().pipe(squeeze_int),
                pl.col("row").unique().pipe(squeeze_int),
                (pl.col("dstream_Tbar") + 1).max().alias("dstream_Tbar"),
                pl.lit(0).alias("infected"),
                pl.lit(0).alias("site0"),
                pl.lit(0).alias("site1"),
            ),
            inner_only=True,
        )

        # upsample to fill in dstream_Tbar dropped by some loggers
        df = union_upsample(
            df,
            upsample="dstream_Tbar",
            group_by="tile",
            fill_null_ops=[
                (pl.all(), {"strategy": "forward"}),
                (pl.all(), {"strategy": "backward"}),
            ],
        )
        assert len(df["dstream_Tbar"].value_counts()["count"].unique()) == 1

        # 0: uninfected, 1: wildtype, 2: site 0 mutant, 3: site 1 mutant,
        # 4: double mutant
        return df.with_columns(
            strain=pl.col("infected")
            * (1 + pl.col("site0") + 2 * pl.col("site1")),
        )

    return (unpack_strain_timeseries,)


@app.cell
def unpack_data(config_columns, osf_ids, pl, tqdm, unpack_strain_timeseries):
    # read and unpack one run at a time, keeping only unpacked strains
    _configs, dfs = [], {}
    for _osf_id in tqdm(osf_ids):
        _df = pl.read_parquet(
            f"https://osf.io/{_osf_id}/download",
        ).with_row_index("dstream_data_id")
        _configs.append(
            _df.select(config_columns)
            .unique()
            .with_columns(osf_id=pl.lit(_osf_id)),
        )
        dfs[_osf_id] = unpack_strain_timeseries(_df)
        del _df

    configs = pl.concat(_configs)
    assert len(configs) == len(osf_ids)
    configs
    return configs, dfs


@app.cell
def count_strains(dfs, pl):
    pl.concat(
        [
            _df.group_by("strain").len().with_columns(osf_id=pl.lit(_osf_id))
            for _osf_id, _df in dfs.items()
        ],
    ).pivot(on="strain", index="osf_id", values="len", sort_columns=True)
    return


@app.cell(hide_code=True)
def delimit_animate(mo):
    mo.md("""
    ## Animate Data
    """)
    return


@app.cell
def define_palette(np):
    palette = np.array(
        [
            [0, 0, 0],  # uninfected: black
            [255, 255, 255],  # wildtype: white
            [0, 0, 255],  # site 0 mutant: blue
            [255, 0, 0],  # site 1 mutant: red
            [160, 32, 240],  # double mutant: purple
        ],
        dtype=np.uint8,
    )
    return (palette,)


@app.cell
def animate_data(configs, dfs, imageio, np, outdir, palette, pl, tqdm):
    for _osf_id, _df in tqdm(dfs.items()):
        _config = configs.filter(pl.col("osf_id") == _osf_id).row(0, named=True)
        _label = (
            f"p2pmode={_config['dispatchSlug'].removeprefix('p2p-')}"
            f"+subgrid={_config['nRowSubgrid']}"
            f"+waning={_config['evoepi_waning_rate']:.2f}"
            f"+seed={_config['globalSeed']}"
        )
        print(_osf_id, _label)

        _df = _df.with_columns(
            row_rank=pl.col("row").rank(method="dense").cast(pl.Int32) - 1,
            col_rank=pl.col("col").rank(method="dense").cast(pl.Int32) - 1,
        )
        _num_rows = _df["row_rank"].max() + 1
        _num_cols = _df["col_rank"].max() + 1

        _frames = []
        for _, _dff in _df.sort("dstream_Tbar").group_by(
            "dstream_Tbar", maintain_order=True
        ):
            _frame = np.zeros((_num_rows, _num_cols), dtype=np.uint8)
            _frame[_dff["row_rank"], _dff["col_rank"]] = _dff["strain"]
            _frame = np.kron(
                palette[_frame], np.ones((4, 4, 1), dtype=np.uint8)
            )
            _frames.append(_frame)

        print(len(_frames))
        _outpath = outdir / f"a=strains+{_label}+osf={_osf_id}+ext="
        imageio.mimsave(f"{_outpath}.gif", _frames, duration=0.05, loop=0)
        # mp4 lasts 15 seconds, whatever its number of frames
        imageio.mimsave(
            f"{_outpath}.mp4",
            _frames,
            fps=len(_frames) / 15,
            macro_block_size=1,
        )
    return


if __name__ == "__main__":
    app.run()
