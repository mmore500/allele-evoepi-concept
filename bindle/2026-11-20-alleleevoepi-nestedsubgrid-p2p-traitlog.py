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
    # 2026-11-20-alleleevoepi-nestedsubgrid-p2p-traitlog

    Strains of each PE's sampled host (0th population member) over time,
    from the trait logger of wse-async-ga workflow
    `2026-11-20-alleleevoepi-nestedsubgrid-p2p-traitlog` (genome
    `genome_alleleevoepi_2026_11_19`, 2 sites), nested subgrid runs
    sweeping mutation rate.

    Nested subgrids: `nRowSubgrid` x `nRowSubgrid` subgrids at the corner
    PE (row 0, col 0), then L-shaped shells of `factor`^2 - 1 subgrids,
    each `factor` times larger, with subgrids overhanging the grid edge
    clipped (wse-async-ga runtime config `nestedSubgridSizeFactor`).

    For each run:
    1. animates the whole grid, sampling every 8th PE along each axis, with
       nested subgrid boundaries overlaid as gray lines;
    2. animates every full (not clipped) subgrid, resized to the same panel
       size, with subgrid sizes in columns and replicate subgrids in rows;
       subgrids show every PE, up to `subgrid_max_res` PEs per side, with
       larger subgrids sampled evenly down to that resolution;
    3. plots, in the same trellis, each subgrid's strain proportions over
       time as a stackplot, with and without uninfected hosts.

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
    return nbname, outdir


@app.cell(hide_code=True)
def delimit_data(mo):
    mo.md("""
    ## Get Data
    """)
    return


@app.cell
def configure_data():
    # trait logger records, one OSF file per run (runtime config)
    osf_ids = ["b4cf5", "r9v8f"]

    # wse-async-ga runtime config nestedSubgridSizeFactor of these runs,
    # which trait logger output doesn't record
    nest_factor = 2

    # sample every nth PE along each axis for whole-grid animation
    grid_stride = 8

    # most PEs per side to sample within each subgrid
    subgrid_max_res = 72

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
        "p2pSendProbability",
        "evoepi_mutation_rate",
        "evoepi_waning_rate",
        "evoepi_seed_pe_stride",
        "dstream_storage_multiplex",
    ]
    return config_columns, grid_stride, nest_factor, osf_ids, subgrid_max_res


@app.cell(hide_code=True)
def delimit_unpack(mo):
    mo.md("""
    ## Unpack Data
    """)
    return


@app.cell
def define_subgrids(np, pl):
    def nested_subgrid_size(
        row: np.ndarray, col: np.ndarray, base: int, factor: int
    ) -> np.ndarray:
        """Size of each PE's nested subgrid."""
        shell = np.maximum(row, col) // base  # base-sized steps from corner
        size = np.full_like(shell, base)
        while (shell >= factor).any():
            grow = shell >= factor
            size[grow] *= factor
            shell[grow] //= factor
        return size

    def sample_tiles(
        n_row: int,
        n_col: int,
        base: int,
        factor: int,
        grid_stride: int,
        max_res: int,
    ) -> pl.DataFrame:
        """PEs to unpack: whole-grid sample, and each full subgrid sampled
        at up to max_res x max_res resolution (subgrid_i, subgrid_j)."""
        rows, cols = np.indices((n_row, n_col))
        size = nested_subgrid_size(rows, cols, base, factor)
        top, left = rows - rows % size, cols - cols % size
        is_full = (top + size <= n_row) & (left + size <= n_col)
        stride = np.maximum(size // max_res, 1)
        is_subgrid_sample = (
            is_full & ((rows - top) % stride == 0) & ((cols - left) % stride == 0)
        )
        is_grid_sample = (rows % grid_stride == 0) & (cols % grid_stride == 0)
        keep = is_subgrid_sample | is_grid_sample
        return pl.DataFrame(
            {
                "row": rows[keep],
                "col": cols[keep],
                "is_grid_sample": is_grid_sample[keep],
                "is_subgrid_sample": is_subgrid_sample[keep],
                "subgrid_size": size[keep],
                "subgrid_top": top[keep],
                "subgrid_left": left[keep],
                "subgrid_res": (size // stride)[keep],
                "subgrid_i": ((rows - top) // stride)[keep],
                "subgrid_j": ((cols - left) // stride)[keep],
            },
        ).cast({"row": pl.UInt16, "col": pl.UInt16})

    return nested_subgrid_size, sample_tiles


@app.cell
def define_unpack(addend_groups, downstream, np, pl, squeeze_int):
    def unpack_strains(df: pl.DataFrame) -> pl.DataFrame:
        """Strain time series of PEs in df, at their own logged times."""
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

        # 0: uninfected, 1: wildtype, 2: site 0 mutant, 3: site 1 mutant,
        # 4: double mutant
        return df.select(
            pl.col("tile").cast(pl.UInt32),
            pl.col("row").cast(pl.UInt16),
            pl.col("col").cast(pl.UInt16),
            pl.col("dstream_Tbar").cast(pl.UInt32),
            strain=(
                pl.col("infected") * (1 + pl.col("site0") + 2 * pl.col("site1"))
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
    grid_stride,
    nest_factor,
    osf_ids,
    pl,
    sample_tiles,
    subgrid_max_res,
    tqdm,
    unpack_strain_timeseries,
):
    # read and unpack one run at a time, keeping only sampled PEs' strains
    _configs, runs = [], {}
    for _osf_id in tqdm(osf_ids):
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
        _configs.append(_config.with_columns(osf_id=pl.lit(_osf_id)))
        _samples = sample_tiles(
            _config["nRow"].item(),
            _config["nCol"].item(),
            _config["nRowSubgrid"].item(),
            nest_factor,
            grid_stride,
            subgrid_max_res,
        )
        # keep only sampled PEs, and only columns needed to unpack
        _df = _df.join(_samples.select("row", "col"), on=["row", "col"]).select(
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
            "tiles": _tiles.join(_samples, on=["row", "col"], maintain_order="left"),
            "strains": _strains,
        }
        assert len(runs[_osf_id]["tiles"]) == _strains.shape[1]

    configs = pl.concat(_configs)
    assert len(configs) == len(osf_ids)
    configs
    return configs, runs


@app.cell
def count_strains(pl, runs):
    pl.DataFrame(
        [
            {
                "osf_id": _osf_id,
                **{
                    str(_k): int((_run["strains"] == _k).sum())
                    for _k in range(5)
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
    strain_names = [
        "uninfected",
        "wildtype",
        "site 0 mutant",
        "site 1 mutant",
        "double mutant",
    ]
    return palette, strain_names


@app.cell
def define_helpers(imageio, np, pl):
    def save_animation(outpath: str, frames: list) -> None:
        imageio.mimsave(f"{outpath}.gif", frames, duration=0.05, loop=0)
        # mp4 lasts 15 seconds, whatever its number of frames
        imageio.mimsave(
            f"{outpath}.mp4",
            frames,
            fps=len(frames) / 15,
            macro_block_size=1,
        )

    def make_run_label(config: dict) -> str:
        return (
            f"mutlog2={round(np.log2(config['evoepi_mutation_rate']))}"
            f"+p2p={config['p2pSendProbability']:.3g}"
            f"+waning={config['evoepi_waning_rate']:.3g}"
            f"+seed={config['globalSeed']}"
        )

    def subgrid_layout(tiles: pl.DataFrame) -> tuple:
        """Full subgrids, with replicate (row) and size (col) trellis
        positions, and each one's PE columns in strain arrays, row-major."""
        tiles = tiles.with_row_index("pe").filter("is_subgrid_sample")
        subgrids = (
            tiles.select(
                "subgrid_size", "subgrid_top", "subgrid_left", "subgrid_res"
            )
            .unique()
            .sort(pl.all())
            .with_columns(
                replicate=pl.int_range(pl.len()).over("subgrid_size"),
            )
        )
        sizes = sorted(subgrids["subgrid_size"].unique())
        num_reps = subgrids["replicate"].max() + 1
        layout = []
        for row in subgrids.iter_rows(named=True):
            pes = tiles.filter(
                (pl.col("subgrid_top") == row["subgrid_top"])
                & (pl.col("subgrid_left") == row["subgrid_left"])
            ).sort("subgrid_i", "subgrid_j")["pe"]
            assert len(pes) == row["subgrid_res"] ** 2
            layout.append(
                {
                    **row,
                    "size_index": sizes.index(row["subgrid_size"]),
                    "pes": pes.to_numpy(),
                }
            )
        return layout, sizes, num_reps

    return make_run_label, save_animation, subgrid_layout


@app.cell
def animate_grid(
    configs,
    make_run_label,
    nest_factor,
    nested_subgrid_size,
    np,
    outdir,
    palette,
    pl,
    runs,
    save_animation,
    tqdm,
):
    # whole grid, every grid_stride-th PE along each axis, with nested
    # subgrid boundaries as gray lines
    _scale = 4
    for _osf_id, _run in tqdm(runs.items()):
        _config = configs.filter(pl.col("osf_id") == _osf_id).row(0, named=True)
        _label = make_run_label(_config)
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

        # gray where pixel's subgrid differs from the pixel above or left
        _rows = _tiles["row"].to_numpy().astype(np.int64).reshape(_num_rows, _num_cols)
        _cols = _tiles["col"].to_numpy().astype(np.int64).reshape(_num_rows, _num_cols)
        _size = nested_subgrid_size(
            _rows, _cols, _config["nRowSubgrid"], nest_factor
        )
        _subgrid = np.kron(
            (_rows - _rows % _size) * _config["nCol"] + (_cols - _cols % _size),
            np.ones((_scale, _scale), dtype=np.int64),
        )
        _boundary = np.zeros_like(_subgrid, dtype=bool)
        _boundary[1:, :] |= _subgrid[1:, :] != _subgrid[:-1, :]
        _boundary[:, 1:] |= _subgrid[:, 1:] != _subgrid[:, :-1]

        _frames = []
        for _s in _strains:
            _frame = np.kron(
                palette[_s.reshape(_num_rows, _num_cols)],
                np.ones((_scale, _scale, 1), dtype=np.uint8),
            )
            _frame[_boundary] = 128
            _frames.append(_frame)
        print(_osf_id, _label, len(_frames), _frames[0].shape)
        save_animation(
            outdir / f"a=strains-grid+{_label}+osf={_osf_id}+ext=", _frames
        )
    return


@app.cell
def animate_subgrids(
    Image,
    ImageDraw,
    ImageFont,
    configs,
    make_run_label,
    np,
    outdir,
    palette,
    pl,
    runs,
    save_animation,
    subgrid_layout,
    subgrid_max_res,
    tqdm,
):
    # every full subgrid, resized to same panel size; sizes in cols,
    # replicates in rows
    _panel_px, _gap, _label_height = 2 * subgrid_max_res, 6, 28
    _font = ImageFont.load_default(size=16)
    for _osf_id, _run in tqdm(runs.items()):
        _config = configs.filter(pl.col("osf_id") == _osf_id).row(0, named=True)
        _label = make_run_label(_config)
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

        _frames = []
        for _s in _run["strains"]:
            _frame = _background.copy()
            for _sub in _layout:
                _res = _sub["subgrid_res"]
                _scale = _panel_px // _res
                _y = _label_height + _gap + _sub["replicate"] * (_panel_px + _gap)
                _x = _gap + _sub["size_index"] * (_panel_px + _gap)
                _frame[
                    _y : _y + _res * _scale, _x : _x + _res * _scale
                ] = np.kron(
                    palette[_s[_sub["pes"]].reshape(_res, _res)],
                    np.ones((_scale, _scale, 1), dtype=np.uint8),
                )
            _frames.append(_frame)
        print(_osf_id, _label, len(_frames), _frames[0].shape)
        save_animation(
            outdir / f"a=strains-subgrids+{_label}+osf={_osf_id}+ext=", _frames
        )
    return


@app.cell(hide_code=True)
def delimit_plot(mo):
    mo.md("""
    ## Plot Data
    """)
    return


@app.cell
def plot_subgrid_stacks(
    configs,
    make_run_label,
    nbname,
    np,
    palette,
    pl,
    plt,
    runs,
    sns,
    strain_names,
    subgrid_layout,
    tp,
):
    # each full subgrid's strain proportions over time, in same trellis as
    # subgrid animation; with uninfected hosts (black), and among infected
    # hosts only
    def _plot_stacks(axes, layout, times, strains, with_uninfected):
        strain_order = [1, 2, 3, 4] + ([0] if with_uninfected else [])
        for _ax in axes.flat:
            _ax.set_visible(False)
        for _sub in layout:
            _ax = axes[_sub["replicate"], _sub["size_index"]]
            _ax.set_visible(True)
            _counts = np.stack(
                [(strains[:, _sub["pes"]] == _k).sum(axis=1) for _k in strain_order]
            ).astype(float)
            _total = _counts.sum(axis=0)
            _props = np.divide(
                _counts, _total, out=np.zeros_like(_counts), where=_total > 0
            )
            _ax.stackplot(
                times,
                _props,
                colors=[palette[_k] / 255 for _k in strain_order],
                labels=[strain_names[_k] for _k in strain_order],
                edgecolor="none",
                linewidth=0.2,
            )
            _ax.set_ylim(0, 1)
            _ax.set_xlim(times.min(), times.max())
            _ax.set_facecolor("lightgray")
            if _sub["replicate"] == 0:
                _ax.set_title(f"{_sub['subgrid_size']} x {_sub['subgrid_size']}")
        _ax = axes[0, 0]
        _ax.figure.legend(*_ax.get_legend_handles_labels())
        sns.move_legend(
            _ax.figure,
            "lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=len(strain_order),
            frameon=False,
            title=None,
        )
        _ax.figure.supxlabel("Generations")
        _ax.figure.supylabel(
            "Proportion of Hosts" if with_uninfected
            else "Proportion of Infected Hosts"
        )

    for _osf_id, _run in runs.items():
        _config = configs.filter(pl.col("osf_id") == _osf_id).row(0, named=True)
        _layout, _sizes, _num_reps = subgrid_layout(_run["tiles"])
        for _with_uninfected in (True, False):
            with tp.teed(
                plt.subplots,
                nrows=_num_reps,
                ncols=len(_sizes),
                figsize=(2 * len(_sizes), 1.6 * _num_reps),
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
                        for _kv in make_run_label(_config).split("+")
                    ),
                },
                teeplot_subdir=nbname,
            ) as (_fig, _axes):
                _plot_stacks(
                    _axes,
                    _layout,
                    _run["times"],
                    _run["strains"],
                    _with_uninfected,
                )
    return


if __name__ == "__main__":
    app.run()
