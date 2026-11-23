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
    # 2026-11-23-alleleevoepi-3site-nestedsplit-traitlog

    Strains of each PE's sampled host (0th population member) over time,
    from the trait logger of wse-async-ga workflow
    `2026-11-23-alleleevoepi-3site-nestedsplit-traitlog` (genome
    `genome_alleleevoepi_2026_11_23`, 3 sites, i.e., 8 strains), five runs
    sweeping mutation rate 0.000012056327 times 1/16, 1/4, 1, 4, and 16.

    The 1152 x 576 PE grid is split into nested subgrids, each its own
    replicate population (4096 hosts per PE): 9 x 9 PE subgrids at the
    corner PE (row 0, col 0), growing by factors of 2 outward, with each
    subgrid but the smallest split 2 x 2 (wse-async-ga runtime config
    `nestingFactorSubgrid` 2 and `nestingLevelSplitSubgrid` 1). So, there
    are 16 subgrids of 9 x 9 PEs, 12 each of 18, 36, 72, and 144 PEs on a
    side, and 4 of 288 PEs on a side (about 340 million hosts). Each PE's
    subgrid is read from its run's runtime config output (`a=rtc`).

    Genome `genome_alleleevoepi_2026_11_23` resembles allele-evoepi-concept's
    CPU ABM: infection is memoryless, recovering with probability
    `evoepi_recovery_rate` per generation; immunity to each allele is
    partial, a level set to 1 on recovery that decays by factor
    1 - `evoepi_waning_rate` per generation in expectation, to a floor of
    `evoepi_immunity_floor`, and multiplies susceptibility by
    1 - `evoepi_immune_strength` * level. Within-host evolution is neutral
    (`evoepi_withinhost_b` 0), so with `evoepi_withinhost_t` 1, each site of
    a new infection mutates with probability equal to the mutation rate.
    Infection is seeded on every PE whose row and col are multiples of 9.
    Within subgrids, PEs exchange hosts' pathogens with neighbors and, in
    batches of up to 8 infected hosts' pathogens immigrated 8 times each,
    with a random p2p partner PE.

    For each run:
    1. animates the whole grid, sampling every `grid_stride`-th PE along
       each axis, with subgrid boundaries overlaid as gray lines,
       transposed to landscape (PE rows as columns), keeping PE (0, 0) at
       top left;
    2. animates every subgrid, resized to the same panel size, with subgrid
       sizes in columns and replicate subgrids in rows; subgrids show every
       PE, up to `subgrid_max_res` PEs per side, with larger subgrids
       sampled evenly down to that resolution;
    3. plots, in the same trellis, each subgrid's strain proportions over
       time as a stackplot, with and without uninfected hosts.

    Then, over all runs, plots how many subgrids of each size end in founder
    complements, medial complements, or unconverged.

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
    # config: mutation rate 0.000012056327 times 1/16, 1/4, 1, 4, 16
    osf_ids = {
        "u7z9c": "stqg8",
        "j9yks": "u956y",
        "b9d64": "h6gb7",
        "tnws8": "4cf9k",
        "mju7p": "x9r6w",
    }

    n_sites = 3

    # sample every nth PE along each axis for whole-grid animation; the
    # smallest subgrid size (nRowSubgrid), so subgrid edges fall between
    # sampled cells, and each sampled cell lies within one subgrid
    grid_stride = 9

    # most PEs per side to sample within each subgrid
    subgrid_max_res = 36

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
        "evoepi_mutation_rate",
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
    return config_columns, grid_stride, n_sites, osf_ids, subgrid_max_res


@app.cell
def get_rtc(grid_stride, osf_ids, pl, subgrid_max_res):
    # per run, each PE's subgrid, and whether it is sampled
    def _get_tiles(osf_id_rtc: str) -> pl.DataFrame:
        tiles = pl.read_parquet(
            f"https://osf.io/{osf_id_rtc}/download",
            columns=[
                "row",
                "col",
                "subgrid",
                "subgrid_origin_row",
                "subgrid_origin_col",
                "subgrid_height",
                "subgrid_width",
            ],
        )
        assert (tiles["subgrid_height"] == tiles["subgrid_width"]).all()
        return (
            tiles.rename({"subgrid_height": "subgrid_size"})
            .drop("subgrid_width")
            .with_columns(
                # sample each subgrid evenly, at up to max resolution
                subgrid_stride=(
                    pl.col("subgrid_size") // subgrid_max_res
                ).clip(lower_bound=1),
            )
            .with_columns(
                subgrid_res=pl.col("subgrid_size") // pl.col("subgrid_stride"),
                subgrid_i=(pl.col("row") - pl.col("subgrid_origin_row"))
                // pl.col("subgrid_stride"),
                subgrid_j=(pl.col("col") - pl.col("subgrid_origin_col"))
                // pl.col("subgrid_stride"),
                # replicate subgrids of each size, in subgrid order
                subgrid_rep=(pl.col("subgrid").rank("dense") - 1).over(
                    "subgrid_size"
                ),
                is_grid_sample=(pl.col("row") % grid_stride == 0)
                & (pl.col("col") % grid_stride == 0),
                is_subgrid_sample=(
                    (pl.col("row") - pl.col("subgrid_origin_row"))
                    % pl.col("subgrid_stride")
                    == 0
                )
                & (
                    (pl.col("col") - pl.col("subgrid_origin_col"))
                    % pl.col("subgrid_stride")
                    == 0
                ),
            )
        )

    tiles_all = {
        _osf_id: _get_tiles(_osf_id_rtc)
        for _osf_id, _osf_id_rtc in osf_ids.items()
    }
    # same layout in every run
    next(iter(tiles_all.values())).group_by("subgrid_size").agg(
        num_subgrids=pl.col("subgrid").n_unique(),
        num_pes=pl.len(),
        num_grid_sample=pl.col("is_grid_sample").sum(),
        num_subgrid_sample=pl.col("is_subgrid_sample").sum(),
    ).sort("subgrid_size")
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
            f"+mut={row['evoepi_mutation_rate']:.3g}"
            f"+seed={row['globalSeed']}"
        )

    def subgrid_layout(tiles: pl.DataFrame) -> tuple:
        """Subgrids, with replicate (row) and size (col) trellis positions,
        and each one's PE columns in strain arrays, row-major."""
        tiles = tiles.with_row_index("pe").filter("is_subgrid_sample")
        sizes = sorted(tiles["subgrid_size"].unique())
        layout = []
        for (subgrid,), group in tiles.group_by("subgrid", maintain_order=True):
            group = group.sort("subgrid_i", "subgrid_j")
            res = group["subgrid_res"][0]
            assert len(group) == res**2
            layout.append(
                {
                    "subgrid": subgrid,
                    "subgrid_size": group["subgrid_size"][0],
                    "subgrid_res": res,
                    "replicate": group["subgrid_rep"][0],
                    "size_index": sizes.index(group["subgrid_size"][0]),
                    "pes": group["pe"].to_numpy(),
                }
            )
        num_reps = max(sub["replicate"] for sub in layout) + 1
        return layout, sizes, num_reps

    return make_run_label, save_animation, subgrid_layout


@app.cell
def animate_grid(
    grid_stride,
    make_run_label,
    np,
    outdir,
    palette,
    runs,
    save_animation,
    tiles_all,
    tqdm,
):
    # whole grid, every grid_stride-th PE along each axis, with subgrid
    # boundaries as gray lines
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

        # gray lines at each subgrid's top and left edges, at their exact PE
        # positions, scaled to pixels
        _edges = (
            tiles_all[_osf_id]
            .select("subgrid_origin_row", "subgrid_origin_col", "subgrid_size")
            .unique()
        )
        _boundary = np.zeros((_num_rows * _scale, _num_cols * _scale), dtype=bool)
        _px = lambda _pe: int(round(_pe / grid_stride * _scale))
        for _top, _left, _sz in _edges.iter_rows():
            _y0, _x0 = _px(_top), _px(_left)
            _y1 = min(_px(_top + _sz), _boundary.shape[0])
            _x1 = min(_px(_left + _sz), _boundary.shape[1])
            if _top > 0 and _y0 < _boundary.shape[0]:
                _boundary[_y0, _x0:_x1] = True
            if _left > 0 and _x0 < _boundary.shape[1]:
                _boundary[_y0:_y1, _x0] = True

        def _grid_frames(strains, num_rows, num_cols, boundary):
            for _s in strains:
                _frame = np.kron(
                    palette[_s.reshape(num_rows, num_cols)],
                    np.ones((_scale, _scale, 1), dtype=np.uint8),
                )
                _frame[boundary] = 128
                # landscape: transpose, keeping PE (0, 0) at top left
                yield np.ascontiguousarray(_frame.transpose(1, 0, 2))

        _label = make_run_label(_osf_id)
        print(_osf_id, _label, len(_strains), _boundary.shape[::-1])
        save_animation(
            outdir / f"a=strains-grid+{_label}+osf={_osf_id}+ext=",
            _grid_frames(_strains, _num_rows, _num_cols, _boundary),
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
    subgrid_max_res,
    tqdm,
):
    # every subgrid, resized to same panel size; sizes in cols, replicates
    # in rows
    _panel_px, _gap, _label_height = 2 * subgrid_max_res, 6, 28
    _font = ImageFont.load_default(size=14)
    for _osf_id, _run in tqdm(runs.items()):
        _layout, _sizes, _num_reps = subgrid_layout(_run["tiles"])

        # layout: label strip, then subgrid panels, gray gaps between
        _height = _label_height + _num_reps * (_panel_px + _gap) + _gap
        _width = len(_sizes) * (_panel_px + _gap) + _gap
        _height += _height % 2
        _width += _width % 2
        _background = Image.new("RGB", (_width, _height), (128, 128, 128))
        _draw = ImageDraw.Draw(_background)
        for _k, _size in enumerate(_sizes):  # column labels
            _text = f"{_size} x {_size}"
            _tw = _draw.textlength(_text, font=_font)
            _draw.text(
                (_gap + _k * (_panel_px + _gap) + (_panel_px - _tw) / 2, 6),
                _text,
                font=_font,
                fill=(255, 255, 255),
            )
        _background = np.asarray(_background)

        def _subgrid_frames(strains, layout, background):
            for _s in strains:
                _frame = background.copy()
                for _sub in layout:
                    _res = _sub["subgrid_res"]
                    _scale = _panel_px // _res
                    _y = (
                        _label_height
                        + _gap
                        + _sub["replicate"] * (_panel_px + _gap)
                    )
                    _x = _gap + _sub["size_index"] * (_panel_px + _gap)
                    _frame[
                        _y : _y + _res * _scale, _x : _x + _res * _scale
                    ] = np.kron(
                        palette[_s[_sub["pes"]].reshape(_res, _res)],
                        np.ones((_scale, _scale, 1), dtype=np.uint8),
                    )
                yield _frame

        _label = make_run_label(_osf_id)
        print(_osf_id, _label, len(_run["strains"]), _background.shape)
        save_animation(
            outdir / f"a=strains-subgrids+{_label}+osf={_osf_id}+ext=",
            _subgrid_frames(_run["strains"], _layout, _background),
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
    tp,
):
    # each subgrid's strain proportions over time, in same trellis as
    # subgrid animation; with uninfected hosts (black), and among infected
    # hosts only
    for _osf_id, _run in runs.items():
        _layout, _sizes, _num_reps = subgrid_layout(_run["tiles"])
        for _with_uninfected in (True, False):
            with tp.teed(
                plt.subplots,
                nrows=_num_reps,
                ncols=len(_sizes),
                figsize=(2 * len(_sizes), 1.1 * _num_reps),
                sharex=True,
                sharey=True,
                squeeze=False,
                constrained_layout=True,
                teeplot_outattrs={
                    "a": "strain-stacks",
                    "osf": _osf_id,
                    "uninfected": _with_uninfected,
                    **dict(
                        _kv.split("=")
                        for _kv in make_run_label(_osf_id).split("+")
                    ),
                },
                teeplot_subdir=nbname,
            ) as (_fig, _axes):
                for _ax in _axes.flat:
                    _ax.set_visible(False)
                for _sub in _layout:
                    _ax = _axes[_sub["replicate"], _sub["size_index"]]
                    _ax.set_visible(True)
                    plot_stack(
                        _ax,
                        _run["times"],
                        _run["strains"][:, _sub["pes"]],
                        _with_uninfected,
                    )
                    if _sub["replicate"] == 0:
                        _ax.set_title(
                            f"{_sub['subgrid_size']} x {_sub['subgrid_size']}"
                        )
                finish_stacks(_fig, _axes[0, 0], _with_uninfected)
    return


@app.cell(hide_code=True)
def delimit_outcome(mo):
    mo.md("""
    ## Outcome per Subgrid

    Classifies each subgrid's end state from the strains of its sampled
    PEs' sampled hosts (every PE, for subgrids up to `subgrid_max_res` PEs
    per side; else `subgrid_max_res`^2 PEs, sampled evenly), over the last
    `outcome_window` generations.

    A subgrid has converged on a complement pair if its two most common
    strains are bitwise complements, and together hold at least
    `outcome_pair_share` of its infections (3/10; with all 8 strains
    equally common, a pair holds 1/4).

    - **founder complements**: the pair is wildtype `000` and `111`;
    - **medial complements**: the pair is `001`/`110`, `010`/`101`, or
      `100`/`011`;
    - **unconverged**: neither.

    Plots, per run (mutation rate), how many subgrids of each size have each
    outcome.
    """)
    return


@app.cell
def compute_outcome(configs, n_sites, np, pl, runs, subgrid_layout):
    # generations, at end of run, to pool sampled hosts' strains over
    outcome_window = 5000
    # least share of infections for a subgrid's two most common strains
    outcome_pair_share = 3 / 10

    def _calc_outcomes(run: dict) -> pl.DataFrame:
        layout, _sizes, _num_reps = subgrid_layout(run["tiles"])
        strains = run["strains"][
            run["times"] >= run["times"].max() - outcome_window
        ]
        rows = []
        for sub in layout:
            counts = np.bincount(
                strains[:, sub["pes"]].ravel(), minlength=1 + (1 << n_sites)
            )[1:].astype(float)
            total = counts.sum()
            # two most common strains: complements, with enough share?
            top1, top2 = np.argsort(-counts, kind="stable")[:2]
            pair_share = (counts[top1] + counts[top2]) / max(total, 1)
            is_pair = (top1 ^ top2) == (1 << n_sites) - 1 and (
                pair_share >= outcome_pair_share
            )
            rows.append(
                {
                    "subgrid": sub["subgrid"],
                    "subgrid_size": sub["subgrid_size"],
                    "num_pes": len(sub["pes"]),
                    "num_infections": total,
                    "pair_share": pair_share,
                    "outcome": (
                        "unconverged"
                        if not is_pair
                        else "founder complements"
                        if 0 in (top1, top2)
                        else "medial complements"
                    ),
                }
            )
        return pl.DataFrame(rows)

    outcomes = pl.concat(
        [
            _calc_outcomes(_run).with_columns(osf_id=pl.lit(_osf_id))
            for _osf_id, _run in runs.items()
        ]
    ).join(
        configs.select("osf_id", "dispatchSlug", "evoepi_mutation_rate"),
        on="osf_id",
    )
    outcomes.group_by("subgrid_size").agg(
        num_subgrids=pl.len(),
        num_pes=pl.col("num_pes").first(),
        median_infections=pl.col("num_infections").median(),
    ).sort("subgrid_size")
    return (outcomes,)


@app.cell
def plot_outcomes(nbname, np, outcomes, pl, plt, sns, tp):
    # per run (mutation rate), subgrids of each size with each outcome
    _groups = ["founder complements", "medial complements", "unconverged"]
    _palette = dict(zip(_groups, sns.color_palette("colorblind", len(_groups))))
    _rates = sorted(outcomes["evoepi_mutation_rate"].unique())
    _sizes = sorted(outcomes["subgrid_size"].unique())
    _summary = (
        outcomes.group_by("evoepi_mutation_rate", "subgrid_size", "outcome")
        .agg(n=pl.len())
        .pivot(
            on="outcome",
            index=["evoepi_mutation_rate", "subgrid_size"],
            values="n",
            on_columns=_groups,
        )
        .fill_null(0)
        .sort("evoepi_mutation_rate", "subgrid_size")
    )
    with tp.teed(
        plt.subplots,
        ncols=len(_rates),
        figsize=(2.6 * len(_rates), 3.4),
        sharey=True,
        squeeze=False,
        constrained_layout=True,
        teeplot_outattrs={"a": "outcome-counts"},
        teeplot_subdir=nbname,
    ) as (_fig, _axes):
        for _ax, _rate in zip(_axes.flat, _rates):
            _sub = _summary.filter(pl.col("evoepi_mutation_rate") == _rate)
            assert _sub["subgrid_size"].to_list() == _sizes
            _x = np.arange(len(_sizes))
            _bottom = np.zeros(len(_sizes))
            for _g in _groups:
                _n = _sub[_g].to_numpy()
                _ax.bar(
                    _x,
                    _n,
                    bottom=_bottom,
                    width=0.8,
                    color=_palette[_g],
                    edgecolor="white",
                    linewidth=1,
                    label=_g,
                )
                _bottom += _n
            _ax.set_xticks(_x)
            _ax.set_xticklabels(_sizes, rotation=90)
            _ax.set_title(f"mutation rate\n{_rate:.3g}", fontsize=10)
        _axes[0, 0].set_ylabel("Number of Subgrids")
        _fig.supxlabel("Subgrid Size (PEs per Side)")
        _handles, _labels = _axes[0, 0].get_legend_handles_labels()
        _fig.legend(
            _handles,
            _labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=len(_groups),
            frameon=False,
        )
    _summary
    return


if __name__ == "__main__":
    app.run()
