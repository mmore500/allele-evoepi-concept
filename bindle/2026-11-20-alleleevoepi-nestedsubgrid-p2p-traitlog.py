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
    from PIL import Image, ImageDraw, ImageFont
    import polars as pl
    from tqdm.auto import tqdm
    from watermark import watermark

    from pylib.munge import addend_groups, squeeze_int, union_upsample

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
    # 2026-11-20-alleleevoepi-nestedsubgrid-p2p-traitlog

    Animates strains of each PE's sampled host (0th population member)
    over time, from the trait logger of wse-async-ga workflow
    `2026-11-20-alleleevoepi-nestedsubgrid-p2p-traitlog` (genome
    `genome_alleleevoepi_2026_11_19`, 2 sites), nested subgrid runs
    sweeping mutation rate.

    Nested subgrids: `nRowSubgrid` x `nRowSubgrid` subgrids at the corner
    PE (row 0, col 0), then L-shaped shells of `factor`^2 - 1 subgrids,
    each `factor` times larger, with subgrids overhanging the grid edge
    clipped (wse-async-ga runtime config `nestedSubgridSizeFactor`).

    For each run, animates (1) the whole grid, sampling every 8th PE along
    each axis, and (2) every full (not clipped) subgrid, resized to the same
    panel size, with subgrid sizes in columns and replicate subgrids in rows.
    Subgrids show every PE, up to `subgrid_max_res` PEs per side; larger
    subgrids are sampled evenly down to that resolution.

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

    return (sample_tiles,)


@app.cell
def define_unpack(addend_groups, downstream, pl, squeeze_int, union_upsample):
    def unpack_strain_timeseries(df: pl.DataFrame) -> pl.DataFrame:
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
    _configs, dfs = [], {}
    for _osf_id in tqdm(osf_ids):
        _df = pl.read_parquet(
            f"https://osf.io/{_osf_id}/download",
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
        _df = _df.join(_samples, on=["row", "col"])
        dfs[_osf_id] = unpack_strain_timeseries(_df).join(
            _samples, on=["row", "col"]
        )
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
def define_save(imageio):
    def save_animation(outpath: str, frames: list) -> None:
        imageio.mimsave(f"{outpath}.gif", frames, duration=0.05, loop=0)
        # mp4 lasts 15 seconds, whatever its number of frames
        imageio.mimsave(
            f"{outpath}.mp4",
            frames,
            fps=len(frames) / 15,
            macro_block_size=1,
        )

    return (save_animation,)


@app.cell
def define_label(np, pl):
    def make_run_label(config: dict) -> str:
        return (
            f"mutlog2={round(np.log2(config['evoepi_mutation_rate']))}"
            f"+p2p={config['p2pSendProbability']:.3g}"
            f"+waning={config['evoepi_waning_rate']:.3g}"
            f"+seed={config['globalSeed']}"
        )

    def strain_frames(df: pl.DataFrame, index: str) -> np.ndarray:
        """Strains, as array indexed by [time, index value]."""
        df = df.sort("dstream_Tbar", index)
        num_t = df["dstream_Tbar"].n_unique()
        return df["strain"].to_numpy().reshape(num_t, -1).astype(np.uint8)

    return make_run_label, strain_frames


@app.cell
def animate_grid(
    configs,
    dfs,
    make_run_label,
    np,
    outdir,
    palette,
    pl,
    save_animation,
    strain_frames,
    tqdm,
):
    # whole grid, every grid_stride-th PE along each axis
    for _osf_id, _df in tqdm(dfs.items()):
        _config = configs.filter(pl.col("osf_id") == _osf_id).row(0, named=True)
        _label = make_run_label(_config)
        _df = _df.filter("is_grid_sample").with_columns(
            row_rank=pl.col("row").rank(method="dense").cast(pl.Int32) - 1,
            col_rank=pl.col("col").rank(method="dense").cast(pl.Int32) - 1,
        )
        _num_rows = _df["row_rank"].max() + 1
        _num_cols = _df["col_rank"].max() + 1
        _df = _df.with_columns(cell=pl.col("row_rank") * _num_cols + pl.col("col_rank"))
        assert _df["cell"].n_unique() == _num_rows * _num_cols
        _strains = strain_frames(_df, "cell").reshape(-1, _num_rows, _num_cols)
        _frames = [
            np.kron(palette[_s], np.ones((4, 4, 1), dtype=np.uint8))
            for _s in _strains
        ]
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
    dfs,
    make_run_label,
    np,
    outdir,
    palette,
    pl,
    save_animation,
    strain_frames,
    subgrid_max_res,
    tqdm,
):
    # every full subgrid, resized to same panel size; sizes in cols,
    # replicates in rows
    _panel_px, _gap, _label_height = 2 * subgrid_max_res, 6, 28
    _font = ImageFont.load_default(size=16)
    for _osf_id, _df in tqdm(dfs.items()):
        _config = configs.filter(pl.col("osf_id") == _osf_id).row(0, named=True)
        _label = make_run_label(_config)
        _df = _df.filter("is_subgrid_sample")
        _subgrids = (
            _df.select("subgrid_size", "subgrid_top", "subgrid_left", "subgrid_res")
            .unique()
            .sort(pl.all())
            .with_columns(
                replicate=pl.int_range(pl.len()).over("subgrid_size"),
                subgrid_index=pl.int_range(pl.len()),
            )
        )
        _sizes = sorted(_subgrids["subgrid_size"].unique())
        _num_reps = _subgrids["replicate"].max() + 1
        _df = _df.join(
            _subgrids, on=["subgrid_size", "subgrid_top", "subgrid_left"]
        ).with_columns(  # sort key: subgrid, then row-major within it
            cell=pl.col("subgrid_index") * subgrid_max_res**2
            + pl.col("subgrid_i") * subgrid_max_res
            + pl.col("subgrid_j"),
        )
        assert _df["cell"].n_unique() == (_subgrids["subgrid_res"] ** 2).sum()
        _strains = strain_frames(_df, "cell")

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

        # where each subgrid's panel goes, and its columns in _strains
        _panels = []
        _start = 0
        for _row in _subgrids.iter_rows(named=True):
            _res = _row["subgrid_res"]
            _y = _label_height + _gap + _row["replicate"] * (_panel_px + _gap)
            _x = _gap + _sizes.index(_row["subgrid_size"]) * (_panel_px + _gap)
            _panels.append((_y, _x, _res, slice(_start, _start + _res * _res)))
            _start += _res * _res

        _frames = []
        for _s in _strains:
            _frame = _background.copy()
            for _y, _x, _res, _cols in _panels:
                _scale = _panel_px // _res
                _frame[_y : _y + _panel_px, _x : _x + _panel_px] = np.kron(
                    palette[_s[_cols].reshape(_res, _res)],
                    np.ones((_scale, _scale, 1), dtype=np.uint8),
                )
            _frames.append(_frame)
        print(_osf_id, _label, len(_frames), _frames[0].shape)
        save_animation(
            outdir / f"a=strains-subgrids+{_label}+osf={_osf_id}+ext=", _frames
        )
    return


if __name__ == "__main__":
    app.run()
