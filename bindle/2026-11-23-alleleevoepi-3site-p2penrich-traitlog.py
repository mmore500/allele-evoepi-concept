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
    from matplotlib import pyplot as plt
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import polars as pl
    import seaborn as sns
    from teeplot import teeplot as tp
    from tqdm.auto import tqdm
    from watermark import watermark

    from pylib import strain_palette
    from pylib.munge import addend_groups, squeeze_int

    return (
        Image,
        ImageDraw,
        ImageFont,
        addend_groups,
        downstream,
        imageio,
        mo,
        np,
        pl,
        plt,
        sns,
        squeeze_int,
        strain_palette,
        tp,
        tqdm,
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
    # 2026-11-23-alleleevoepi-3site-p2penrich-traitlog

    Strains of each PE's sampled host (0th population member) over time,
    from the trait logger of wse-async-ga workflow
    `2026-11-23-alleleevoepi-3site-p2penrich-traitlog`, runs
    `sites3-mutsweep` and `sites3-mutsweep2` (genome
    `genome_alleleevoepi_2026_11_23`, 3 sites, i.e., 8 strains).

    The 1152 x 576 PE grid is split into 2592 subgrids of 16 x 16 PEs, each
    its own replicate population of about 1 million hosts (4096 per PE).
    Mutation rate is set per subgrid, in diagonal stripes (wse-async-ga
    runtime config `calcSymbolValues`): in run `sites3-mutsweep`, 2^-26 to
    2^-10 by factors of 4 (288 replicate subgrids per rate); in run
    `sites3-mutsweep2`, 2^-32 to 2^-28 and 2^-8 to 2^-2 by factors of 4
    (about 370 per rate). Each PE's rate is read from its run's runtime
    config output (`a=rtc`).

    Genome `genome_alleleevoepi_2026_11_23` resembles allele-evoepi-concept's
    CPU ABM: infection is memoryless, recovering with probability
    `evoepi_recovery_rate` per generation; immunity to each allele is
    partial, a level set to 1 on recovery that decays by factor
    1 - `evoepi_waning_rate` per generation in expectation, to a floor of
    `evoepi_immunity_floor`, and multiplies susceptibility by
    1 - `evoepi_immune_strength` * level. Within-host evolution is neutral
    (`evoepi_withinhost_b` 0), so with `evoepi_withinhost_t` 1, each site of
    a new infection mutates with probability equal to the mutation rate.
    Infection is seeded on the corner PE of each subgrid. Within subgrids,
    PEs exchange hosts' pathogens with neighbors and, in batches of up to 8
    infected hosts' pathogens immigrated 8 times each, with a random p2p
    partner PE.

    For each run:
    1. maps mutation rate over the grid;
    2. animates the whole grid, sampling every `grid_stride`-th PE along
       each axis, transposed to landscape (PE rows as columns), keeping PE
       (0, 0) at top left;
    3. animates `subgrid_reps` replicate subgrids per mutation rate, every
       PE, with mutation rates in columns and replicates in rows;
    4. plots, in the same trellis, each subgrid's strain proportions over
       time as a stackplot, with and without uninfected hosts;
    5. plots strain proportions over time per mutation rate, pooling the
       whole-grid sample's PEs across all subgrids at that rate;

    Then, over both runs, plots mutant strains' share of infections against
    mutation rate.

    Colors: black, uninfected; strains colored by number of mutant sites
    (hue), with strains of the same number set apart by lightness (as
    `pylib.strain_palette`). Strain labels give alleles at sites 2, 1, 0.
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
    ## Get Data
    """)
    return


@app.cell
def configure_data():
    # per run, OSF files of trait logger records and of per-PE runtime
    # config: runs sites3-mutsweep, sites3-mutsweep2
    osf_ids = {"jvzmd": "mxacs", "be5cx": "svdyz"}

    n_sites = 3

    # sample every nth PE along each axis for whole-grid animation
    grid_stride = 4

    # replicate subgrids, per mutation rate, to animate and plot every PE of
    subgrid_reps = 6

    config_columns = [
        "dispatchSlug",
        "genomeFlavor",
        "globalSeed",
        "nRow",
        "nCol",
        "nRowSubgrid",
        "nColSubgrid",
        "tilePopSize",
        "tournSize",
        "nCycle",
        "p2pSendProbabilityMean",
        "evoepi_n_sites",
        "evoepi_contact_rate",
        "evoepi_immune_strength",
        "evoepi_recovery_rate",
        "evoepi_waning_rate",
        "evoepi_immunity_floor",
        "evoepi_withinhost_b",
        "evoepi_withinhost_t",
        "evoepi_seed_pe_stride",
        "ASYNC_GA_P2P_BATCH_SIZE",
        "ASYNC_GA_P2P_SEND_EXCLUDE_EMPTY",
        "ASYNC_GA_ENRICH_IMMIGRANTS_P2P",
        "ASYNC_GA_NEIGHBOR_SEND_NUM_GENOMES",
        "ASYNC_GA_NEIGHBOR_RECV_NUM_GENOMES",
        "dstream_storage_multiplex",
    ]
    return (
        config_columns,
        grid_stride,
        n_sites,
        osf_ids,
        subgrid_reps,
    )


@app.cell
def get_rtc(grid_stride, osf_ids, pl, subgrid_reps):
    # per run, each PE's subgrid and mutation rate, and whether it is sampled
    def _get_tiles(osf_id_rtc: str) -> pl.DataFrame:
        tiles = (
            pl.read_parquet(
                f"https://osf.io/{osf_id_rtc}/download",
                columns=[
                    "row",
                    "col",
                    "subgrid",
                    "subgrid_origin_row",
                    "subgrid_origin_col",
                    "subgrid_height",
                    "subgrid_width",
                    "rtc_evoepi_mutationRate_f32",
                ],
            )
            .with_columns(
                mutlog2=pl.col("rtc_evoepi_mutationRate_f32")
                .log(2)
                .round()
                .cast(pl.Int8),
            )
            .with_columns(
                # replicate subgrids of each mutation rate, in subgrid order
                subgrid_rep=(pl.col("subgrid").rank("dense") - 1).over(
                    "mutlog2"
                ),
                subgrid_i=pl.col("row") - pl.col("subgrid_origin_row"),
                subgrid_j=pl.col("col") - pl.col("subgrid_origin_col"),
            )
            .with_columns(
                is_grid_sample=(pl.col("row") % grid_stride == 0)
                & (pl.col("col") % grid_stride == 0),
                is_subgrid_sample=pl.col("subgrid_rep") < subgrid_reps,
            )
        )
        assert (
            tiles.group_by("subgrid").agg(pl.col("mutlog2").n_unique())[
                "mutlog2"
            ]
            == 1
        ).all()  # one rate per subgrid
        return tiles

    tiles_all = {
        _osf_id: _get_tiles(_osf_id_rtc)
        for _osf_id, _osf_id_rtc in osf_ids.items()
    }
    pl.concat(
        [
            _tiles.group_by("mutlog2")
            .agg(
                num_subgrids=pl.col("subgrid").n_unique(),
                num_pes=pl.len(),
                num_grid_sample=pl.col("is_grid_sample").sum(),
                num_subgrid_sample=pl.col("is_subgrid_sample").sum(),
            )
            .sort("mutlog2")
            .with_columns(osf_id=pl.lit(_osf_id))
            for _osf_id, _tiles in tiles_all.items()
        ]
    )
    return (tiles_all,)


@app.cell(hide_code=True)
def delimit_unpack(mo):
    mo.md("""
    ## Unpack Data
    """)
    return


@app.cell
def define_unpack(addend_groups, downstream, n_sites, np, pl, squeeze_int):
    def unpack_strains(df: pl.DataFrame) -> pl.DataFrame:
        """Strain time series of PEs in df, at their own logged times."""
        # unpack dstream data items from logger hex strings, demultiplexing
        # infected and each site's allele records
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
        sites = [f"site{i}" for i in range(n_sites)]
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
            .rename(
                {"0": "infected", **{str(1 + i): s for i, s in enumerate(sites)}}
            )
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
                *(pl.lit(0).alias(s) for s in sites),
            ),
            inner_only=True,
        )

        # 0: uninfected; else 1 + alleles, as bits (site i at bit i)
        return df.select(
            pl.col("tile").cast(pl.UInt32),
            pl.col("row").cast(pl.UInt16),
            pl.col("col").cast(pl.UInt16),
            pl.col("dstream_Tbar").cast(pl.UInt32),
            strain=(
                pl.col("infected")
                * (
                    1
                    + pl.sum_horizontal(
                        pl.col(s).cast(pl.UInt8) * (1 << i)
                        for i, s in enumerate(sites)
                    )
                )
            ).cast(pl.UInt8),
        )

    def unpack_strain_timeseries(
        df: pl.DataFrame, chunk_size: int = 2000
    ) -> tuple:
        """Strains of PEs in df, all at the union of their logged times.

        Returns times, PEs (tile, row, col), and strains, an array indexed
        by [time, PE]. Each PE's strain at a time is its strain at its last
        logged time at or before (or, if none, its first logged time).
        """
        # unpack in chunks of PEs, to save memory
        df = pl.concat(
            [
                unpack_strains(df.slice(start, chunk_size))
                for start in range(0, len(df), chunk_size)
            ],
        ).sort("tile", "dstream_Tbar")
        tiles = df.select("tile", "row", "col").unique("tile").sort("tile")
        times = np.sort(df["dstream_Tbar"].unique().to_numpy())

        # per PE, index of last logged time at or before each time (forward
        # fill), or first logged time if none (backward fill)
        tile_rank = np.searchsorted(tiles["tile"].to_numpy(), df["tile"].to_numpy())
        key = tile_rank.astype(np.int64) * (len(times) + 1) + np.searchsorted(
            times, df["dstream_Tbar"].to_numpy()
        )
        query = (
            np.arange(len(tiles), dtype=np.int64)[None, :] * (len(times) + 1)
            + np.arange(len(times), dtype=np.int64)[:, None]
        )
        first = np.searchsorted(tile_rank, np.arange(len(tiles)))
        index = np.maximum(np.searchsorted(key, query, side="right") - 1, first)
        strains = df["strain"].to_numpy()[index]
        assert strains.shape == (len(times), len(tiles))
        return times, tiles, strains

    return (unpack_strain_timeseries,)


@app.cell
def unpack_data(
    config_columns,
    n_sites,
    osf_ids,
    pl,
    tiles_all,
    tqdm,
    unpack_strain_timeseries,
):
    # read and unpack one run at a time, keeping only sampled PEs' strains
    _configs, runs = [], {}
    for _osf_id in tqdm(osf_ids):
        _samples = tiles_all[_osf_id].filter(
            pl.col("is_grid_sample") | pl.col("is_subgrid_sample")
        )
        _df = pl.read_parquet(
            f"https://osf.io/{_osf_id}/download",
            columns=[
                *config_columns,
                "tile",
                "row",
                "col",
                "data_hex",
                "dstream_algo",
                "dstream_storage_bitoffset",
                "dstream_storage_bitwidth",
                "dstream_S",
                "dstream_T_bitoffset",
                "dstream_T_bitwidth",
            ],
        ).with_row_index("dstream_data_id")
        _config = _df.select(config_columns).unique()
        assert len(_config) == 1
        assert _config["evoepi_n_sites"].item() == n_sites
        _configs.append(_config.with_columns(osf_id=pl.lit(_osf_id)))
        # keep only sampled PEs, and only columns needed to unpack
        _df = _df.join(
            _samples.select("row", "col"), on=["row", "col"]
        ).select(
            "dstream_data_id",
            "tile",
            "row",
            "col",
            pl.col("^dstream_.*$").exclude("dstream_data_id"),
            "data_hex",
        )
        _times, _tiles, _strains = unpack_strain_timeseries(_df)
        del _df
        runs[_osf_id] = {
            "times": _times,
            "tiles": _tiles.join(
                _samples.with_columns(pl.col("row", "col").cast(pl.UInt16)),
                on=["row", "col"],
                maintain_order="left",
            ),
            "strains": _strains,
            "mutlog2s": sorted(_samples["mutlog2"].unique()),
        }
        assert len(runs[_osf_id]["tiles"]) == _strains.shape[1]

    configs = pl.concat(_configs)
    assert len(configs) == len(osf_ids)
    configs
    return configs, runs


@app.cell
def count_strains(n_sites, np, pl, runs):
    pl.DataFrame(
        [
            {
                "osf_id": _osf_id,
                **{
                    str(_k): int(_n)
                    for _k, _n in enumerate(
                        np.bincount(
                            _run["strains"].ravel(),
                            minlength=1 + (1 << n_sites),
                        )
                    )
                },
            }
            for _osf_id, _run in runs.items()
        ],
    )
    return


@app.cell(hide_code=True)
def delimit_animate(mo):
    mo.md("""
    ## Animate Data
    """)
    return


@app.cell
def define_palette(n_sites, np, strain_palette):
    # uninfected black; strains by number of mutant sites (hue), then
    # lightness
    _strain_colors = strain_palette(n_sites)
    palette = np.array(
        [
            [0, 0, 0],
            *(
                [round(255 * _c) for _c in _strain_colors[_s]]
                for _s in range(1 << n_sites)
            ),
        ],
        dtype=np.uint8,
    )
    strain_names = [
        "uninfected",
        *(f"{_s:0{n_sites}b}" for _s in range(1 << n_sites)),
    ]
    strain_names
    return palette, strain_names


@app.cell
def define_helpers(configs, imageio, pl):
    def save_animation(outpath: str, frames, num_frames: int) -> None:
        """Write frames, an iterable, to gif and mp4 as they're made, so
        they needn't all fit in memory."""
        # mp4 lasts 15 seconds, whatever its number of frames
        with imageio.get_writer(
            f"{outpath}.gif", mode="I", duration=0.05, loop=0
        ) as gif, imageio.get_writer(
            f"{outpath}.mp4", fps=num_frames / 15, macro_block_size=1
        ) as mp4:
            for frame in frames:
                gif.append_data(frame)
                mp4.append_data(frame)

    def make_run_label(osf_id: str) -> str:
        row = configs.filter(pl.col("osf_id") == osf_id).row(0, named=True)
        return (
            f"run={row['dispatchSlug']}"
            f"+p2p={row['p2pSendProbabilityMean']:.3g}"
            f"+seed={row['globalSeed']}"
        )

    def subgrid_layout(tiles: pl.DataFrame, mutlog2s: list) -> list:
        """Sampled subgrids, with replicate (row) and mutation rate (col)
        trellis positions, and each one's PE columns in strain arrays,
        row-major."""
        tiles = tiles.with_row_index("pe").filter("is_subgrid_sample")
        layout = []
        for (subgrid,), group in tiles.group_by("subgrid", maintain_order=True):
            group = group.sort("subgrid_i", "subgrid_j")
            height = group["subgrid_height"][0]
            width = group["subgrid_width"][0]
            assert len(group) == height * width
            layout.append(
                {
                    "subgrid": subgrid,
                    "height": height,
                    "width": width,
                    "replicate": group["subgrid_rep"][0],
                    "mutlog2": group["mutlog2"][0],
                    "rate_index": mutlog2s.index(group["mutlog2"][0]),
                    "pes": group["pe"].to_numpy(),
                }
            )
        return layout

    return make_run_label, save_animation, subgrid_layout


@app.cell
def plot_rate_map(make_run_label, nbname, np, plt, tiles_all, tp):
    # each PE's mutation rate, in animation's orientation (PE rows as
    # columns, PE (0, 0) at top left)
    for _osf_id, _tiles in tiles_all.items():
        _mutlog2s = sorted(_tiles["mutlog2"].unique())
        _n_row, _n_col = _tiles["row"].max() + 1, _tiles["col"].max() + 1
        _grid = np.empty((_n_row, _n_col), dtype=int)
        _grid[_tiles["row"].to_numpy(), _tiles["col"].to_numpy()] = np.searchsorted(
            _mutlog2s, _tiles["mutlog2"].to_numpy()
        )
        with tp.teed(
            plt.subplots,
            figsize=(9, 4.2),
            constrained_layout=True,
            teeplot_outattrs={
                "a": "mutation-rate-map",
                "osf": _osf_id,
                **dict(
                    _kv.split("=") for _kv in make_run_label(_osf_id).split("+")
                ),
            },
            teeplot_subdir=nbname,
        ) as (_fig, _ax):
            _im = _ax.imshow(
                _grid.T,
                cmap=plt.get_cmap("viridis", len(_mutlog2s)),
                vmin=-0.5,
                vmax=len(_mutlog2s) - 0.5,
                interpolation="nearest",
            )
            _ax.set_xlabel("PE Row")
            _ax.set_ylabel("PE Col")
            _cbar = _fig.colorbar(
                _im,
                ax=_ax,
                ticks=range(len(_mutlog2s)),
                label="Mutation Rate (log2)",
                shrink=0.8,
            )
            _cbar.ax.set_yticklabels(_mutlog2s)
    return


@app.cell
def animate_grid(
    grid_stride,
    make_run_label,
    np,
    outdir,
    palette,
    runs,
    save_animation,
    tqdm,
):
    # whole grid, every grid_stride-th PE along each axis
    _scale = 4
    for _osf_id, _run in tqdm(runs.items()):
        _tiles = (
            _run["tiles"]
            .with_row_index("pe")
            .filter("is_grid_sample")
            .sort("row", "col")
        )
        _num_rows = _tiles["row"].n_unique()
        _num_cols = _tiles["col"].n_unique()
        assert len(_tiles) == _num_rows * _num_cols
        _strains = _run["strains"][:, _tiles["pe"].to_numpy()]

        def _grid_frames(strains, num_rows, num_cols):
            for _s in strains:
                _frame = np.kron(
                    palette[_s.reshape(num_rows, num_cols)],
                    np.ones((_scale, _scale, 1), dtype=np.uint8),
                )
                # landscape: transpose, keeping PE (0, 0) at top left
                yield np.ascontiguousarray(_frame.transpose(1, 0, 2))

        _label = make_run_label(_osf_id)
        print(_osf_id, _label, len(_strains), (_num_rows, _num_cols))
        save_animation(
            outdir
            / f"a=strains-grid+stride={grid_stride}+{_label}+osf={_osf_id}+ext=",
            _grid_frames(_strains, _num_rows, _num_cols),
            len(_strains),
        )
    return


@app.cell
def animate_subgrids(
    Image,
    ImageDraw,
    ImageFont,
    make_run_label,
    np,
    outdir,
    palette,
    runs,
    save_animation,
    subgrid_layout,
    subgrid_reps,
    tqdm,
):
    # sampled subgrids, every PE; mutation rates in cols, replicates in rows
    _scale, _gap, _label_height = 8, 6, 28
    _font = ImageFont.load_default(size=16)
    for _osf_id, _run in tqdm(runs.items()):
        _mutlog2s = _run["mutlog2s"]
        _layout = subgrid_layout(_run["tiles"], _mutlog2s)
        _panel_h = max(_sub["height"] for _sub in _layout) * _scale
        _panel_w = max(_sub["width"] for _sub in _layout) * _scale

        # layout: label strip, then subgrid panels, gray gaps between
        _height = _label_height + subgrid_reps * (_panel_h + _gap) + _gap
        _width = len(_mutlog2s) * (_panel_w + _gap) + _gap
        _height += _height % 2
        _width += _width % 2
        _background = Image.new("RGB", (_width, _height), (128, 128, 128))
        _draw = ImageDraw.Draw(_background)
        for _k, _mutlog2 in enumerate(_mutlog2s):  # column labels
            _text = f"2^{_mutlog2}"
            _tw = _draw.textlength(_text, font=_font)
            _draw.text(
                (_gap + _k * (_panel_w + _gap) + (_panel_w - _tw) / 2, 6),
                _text,
                font=_font,
                fill=(255, 255, 255),
            )
        _background = np.asarray(_background)

        def _subgrid_frames(strains, layout, background, panel_h, panel_w):
            for _s in strains:
                _frame = background.copy()
                for _sub in layout:
                    _y = (
                        _label_height
                        + _gap
                        + _sub["replicate"] * (panel_h + _gap)
                    )
                    _x = _gap + _sub["rate_index"] * (panel_w + _gap)
                    _frame[
                        _y : _y + _sub["height"] * _scale,
                        _x : _x + _sub["width"] * _scale,
                    ] = np.kron(
                        palette[
                            _s[_sub["pes"]].reshape(
                                _sub["height"], _sub["width"]
                            )
                        ],
                        np.ones((_scale, _scale, 1), dtype=np.uint8),
                    )
                yield _frame

        _label = make_run_label(_osf_id)
        print(_osf_id, _label, len(_run["strains"]), _background.shape)
        save_animation(
            outdir / f"a=strains-subgrids+{_label}+osf={_osf_id}+ext=",
            _subgrid_frames(
                _run["strains"], _layout, _background, _panel_h, _panel_w
            ),
            len(_run["strains"]),
        )
    return


@app.cell(hide_code=True)
def delimit_plot(mo):
    mo.md("""
    ## Plot Data
    """)
    return


@app.cell
def define_plot_stack(np, palette, strain_names):
    def plot_stack(ax, times, strains, with_uninfected: bool) -> None:
        """Stackplot of strain proportions over time among PEs (columns of
        strains)."""
        strain_order = [*range(1, len(palette))] + (
            [0] if with_uninfected else []
        )
        counts = np.stack(
            [(strains == k).sum(axis=1) for k in strain_order]
        ).astype(float)
        total = counts.sum(axis=0)
        props = np.divide(
            counts, total, out=np.zeros_like(counts), where=total > 0
        )
        ax.stackplot(
            times,
            props,
            colors=[palette[k] / 255 for k in strain_order],
            labels=[strain_names[k] for k in strain_order],
            edgecolor="none",
            linewidth=0.2,
        )
        ax.set_ylim(0, 1)
        ax.set_xlim(times.min(), times.max())
        ax.set_facecolor("lightgray")

    def finish_stacks(fig, ax, with_uninfected: bool) -> None:
        """Shared legend and axis labels."""
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=len(labels),
            frameon=False,
            title="Strain (alleles at sites 2, 1, 0)",
        )
        fig.supxlabel("Generations")
        fig.supylabel(
            "Proportion of Hosts"
            if with_uninfected
            else "Proportion of Infected Hosts"
        )

    return finish_stacks, plot_stack


@app.cell
def plot_subgrid_stacks(
    finish_stacks,
    make_run_label,
    nbname,
    plot_stack,
    plt,
    runs,
    subgrid_layout,
    subgrid_reps,
    tp,
):
    # each sampled subgrid's strain proportions over time, in same trellis
    # as subgrid animation; with uninfected hosts (black), and among
    # infected hosts only
    for _osf_id, _run in runs.items():
        _mutlog2s = _run["mutlog2s"]
        _layout = subgrid_layout(_run["tiles"], _mutlog2s)
        for _with_uninfected in (True, False):
            with tp.teed(
                plt.subplots,
                nrows=subgrid_reps,
                ncols=len(_mutlog2s),
                figsize=(1.7 * len(_mutlog2s), 1.3 * subgrid_reps),
                sharex=True,
                sharey=True,
                squeeze=False,
                constrained_layout=True,
                teeplot_outattrs={
                    "a": "strain-stacks-subgrids",
                    "osf": _osf_id,
                    "uninfected": _with_uninfected,
                    **dict(
                        _kv.split("=")
                        for _kv in make_run_label(_osf_id).split("+")
                    ),
                },
                teeplot_subdir=nbname,
            ) as (_fig, _axes):
                for _sub in _layout:
                    _ax = _axes[_sub["replicate"], _sub["rate_index"]]
                    plot_stack(
                        _ax,
                        _run["times"],
                        _run["strains"][:, _sub["pes"]],
                        _with_uninfected,
                    )
                    if _sub["replicate"] == 0:
                        _ax.set_title(f"2^{_sub['mutlog2']}")
                finish_stacks(_fig, _axes[0, 0], _with_uninfected)
    return


@app.cell
def plot_rate_stacks(
    finish_stacks,
    make_run_label,
    nbname,
    pl,
    plot_stack,
    plt,
    runs,
    tp,
):
    # strain proportions over time per mutation rate, pooling whole-grid
    # sample PEs over all subgrids at that rate
    for _osf_id, _run in runs.items():
        _mutlog2s = _run["mutlog2s"]
        _tiles = _run["tiles"].with_row_index("pe").filter("is_grid_sample")
        _ncols = 3
        _nrows = -(-len(_mutlog2s) // _ncols)
        for _with_uninfected in (True, False):
            with tp.teed(
                plt.subplots,
                nrows=_nrows,
                ncols=_ncols,
                figsize=(3 * _ncols, 2 * _nrows),
                sharex=True,
                sharey=True,
                squeeze=False,
                constrained_layout=True,
                teeplot_outattrs={
                    "a": "strain-stacks-rates",
                    "osf": _osf_id,
                    "uninfected": _with_uninfected,
                    **dict(
                        _kv.split("=")
                        for _kv in make_run_label(_osf_id).split("+")
                    ),
                },
                teeplot_subdir=nbname,
            ) as (_fig, _axes):
                for _ax in _axes.flat[len(_mutlog2s) :]:
                    _ax.set_visible(False)
                for _ax, _mutlog2 in zip(_axes.flat, _mutlog2s):
                    _pes = _tiles.filter(pl.col("mutlog2") == _mutlog2)["pe"]
                    plot_stack(
                        _ax,
                        _run["times"],
                        _run["strains"][:, _pes.to_numpy()],
                        _with_uninfected,
                    )
                    _ax.set_title(f"mutation rate 2^{_mutlog2}")
                finish_stacks(_fig, _axes[0, 0], _with_uninfected)
    return


@app.cell
def plot_mutant_share(nbname, np, pl, plt, runs, tp):
    # per subgrid, share of infected sampled hosts carrying any mutant
    # allele, over all logged times; whole-grid sample PEs (16 per subgrid),
    # both runs
    def _calc_shares(run: dict) -> pl.DataFrame:
        tiles = run["tiles"].with_row_index("pe").filter("is_grid_sample")
        pes = tiles["pe"].to_numpy()
        return (
            tiles.with_columns(
                infected=pl.Series((run["strains"] > 0).sum(axis=0)[pes]),
                mutant=pl.Series((run["strains"] > 1).sum(axis=0)[pes]),
            )
            .group_by("mutlog2", "subgrid")
            .agg(pl.col("infected", "mutant").sum())
            .with_columns(share=pl.col("mutant") / pl.col("infected"))
        )

    _df = pl.concat(
        [
            _calc_shares(_run).with_columns(osf_id=pl.lit(_osf_id))
            for _osf_id, _run in runs.items()
        ]
    ).sort("mutlog2")
    _mean = (
        _df.group_by("mutlog2")
        .agg(pl.col("share").mean(), num_subgrids=pl.len())
        .sort("mutlog2")
    )
    with tp.teed(
        plt.subplots,
        figsize=(7, 3.6),
        constrained_layout=True,
        teeplot_outattrs={"a": "mutant-share"},
        teeplot_subdir=nbname,
    ) as (_fig, _ax):
        _rng = np.random.default_rng(1)
        _ax.scatter(
            _df["mutlog2"].to_numpy() + _rng.uniform(-0.5, 0.5, len(_df)),
            _df["share"].to_numpy(),
            s=4,
            alpha=0.3,
            color="0.4",
            linewidths=0,
            label="subgrid",
        )
        _ax.plot(
            _mean["mutlog2"],
            _mean["share"],
            marker="o",
            color="C0",
            linewidth=2,
            label="mean over subgrids",
        )
        _ax.set_xticks(_mean["mutlog2"].to_list())
        _ax.set_xlabel("Mutation Rate (log2)")
        _ax.set_ylabel("Mutant Share of Infections")
        _ax.set_ylim(0, 1)
        _ax.legend(frameon=False)
    _mean
    return


if __name__ == "__main__":
    app.run()
