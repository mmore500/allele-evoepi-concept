import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell
def import_std():
    import gc
    import pathlib
    import random
    from typing import Dict, List, Sequence, Tuple, Union
    import uuid

    return Dict, List, Sequence, Tuple, Union, gc, pathlib, random, uuid


@app.cell
def import_pkg():
    try:
        import cupy as cp
    except ImportError:
        import warnings

        warnings.warn(
            "cupy import failed; falling back to numpy "
            "(GPU engine unavailable)",
            stacklevel=2,
        )
        import numpy as cp

    # workaround: iplotx 1.7.x uses importlib.metadata without importing it
    import importlib.metadata  # noqa: F401

    import downstream.dstream as dstream
    import hstrat
    import igraph as ig
    import iplotx
    import marimo as mo
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    import numpy as np
    import pandas as pd
    from phyloframe import legacy as pfl
    import polars as pl
    import seaborn as sns
    from teeplot import teeplot as tp
    from tqdm.auto import tqdm
    from watermark import watermark

    from pylib import draw_scatter_tree

    return (
        FuncFormatter,
        MaxNLocator,
        cp,
        draw_scatter_tree,
        dstream,
        hstrat,
        ig,
        iplotx,
        mo,
        np,
        pd,
        pfl,
        pl,
        plt,
        sns,
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


@app.cell
def configure_args(mo):
    # Marimo CLI args (set via `marimo edit notebook.py -- --pop-size N ...`
    # or `marimo export ipynb ... -- --pop-size N ...`). Defaults match the
    # current sweep settings: POP_SIZE=200_000 hosts and N_STEPS=100 steps.
    _args = mo.cli_args()
    POP_SIZE = int(_args.get("pop-size") or 200_000)
    POW = float(_args.get("pow") or 1.0)
    N_STEPS = int(_args.get("n-steps") or 100)
    N_REPLICATES = int(_args.get("n-replicates") or 1)
    ENGINE = str(_args.get("engine") or "numpy").lower()
    if ENGINE not in ("numpy", "cupy"):
        raise ValueError(
            f"engine must be 'numpy' or 'cupy', got {ENGINE!r}",
        )
    SKIP_PLOTTING = bool(_args.get("skip-plotting") or False)
    print(
        f"args: POP_SIZE={POP_SIZE} N_STEPS={N_STEPS} "
        f"N_REPLICATES={N_REPLICATES} ENGINE={ENGINE} "
        f"SKIP_PLOTTING={SKIP_PLOTTING}",
    )
    return ENGINE, N_REPLICATES, N_STEPS, POP_SIZE, POW, SKIP_PLOTTING


@app.cell
def configure_backend(ENGINE, cp, np):
    use_cupy = (
        ENGINE == "cupy"
    )  # use cupy backend (GPU), otherwise numpy (CPU)
    xp = [np, cp][use_cupy]
    return (xp,)


@app.cell(hide_code=True)
def delimit_simulation(mo):
    mo.md("""
    ## Simulation Implementation

    This notebook is the wide-genome variant of
    `2026-04-28-allele-abm-phylogeny-hstrat.py`: pathogen genomes are widened
    so up to `N_SITES=64` allele loci can be tracked (the genome dtype is
    chosen automatically — `uint16`/`uint32`/`uint64` for `N_SITES <=
    16`/`32`/`64`), and the phylogeny sweep loops over all three bit-widths
    so the resulting reconstructions can be compared side by side.
    Population size, simulation length, replicate count, and array engine
    (`numpy`/`cupy`) are all passed in as marimo CLI args (see the
    `configure_args` cell). Phylogeny estimation still uses
    *hereditary stratigraphic surface* annotations (see
    https://hstrat.rtfd.io). Each infected host carries a `dstream_S=64`-site,
    1-bit "hybrid" surface — all S=64 differentia bits are packed into a
    single `uint64` per host, so `pathogen_markers` is a flat
    `(POP_SIZE,)` array. Buffers are randomized at population init,
    representing the post-predeposit state after S implicit deposits, so
    real per-step deposits begin past `dstream_T = S`. On transmission
    the donor surface state is copied to the recipient (the ABM analogue
    of `CloneDescendant`). At the end of the run a single
    sample of `N_SAMPLE` currently-extant infections is taken; their surface
    buffers feed `surface_unpack_reconstruct` + `surface_postprocess_trie`
    to produce the estimated phylogeny. Per-step prevalence is logged by
    Hamming weight as a separate long-form dataframe (one row per
    `(Step, hw)` with a `count` column) rather than by individual
    strain, so the trajectory dataframe stays compact even at large
    `N_SITES`.

    The vectorized deposit pattern is adapted from
    https://github.com/mmore500/hstrat-synthesis (see `pylib/track_ca.py`).
    """)
    return


@app.cell
def def_simulate(
    Dict,
    List,
    Sequence,
    Tuple,
    Union,
    dstream,
    gc,
    np,
    pd,
    random,
    tqdm,
    xp,
):
    def simulate(
        N_SITES: int = 2,
        POP_SIZE: int = 1_000_000,
        CONTACT_RATE: float = 0.3,
        RECOVERY_RATE: float = 0.1,
        MUTATION_RATE: Union[float, Sequence[float]] = 1e-4,
        WANING_RATE: float = 0.02,
        IMMUNE_STRENGTH: float = 0.9,
        N_STEPS: int = 1_000,
        SEED_COUNT: int = 10,
        within_host_b: float = 0.2,
        within_host_t: float = 25.0,
        seed: int = 1,
        MUTATOR_HOSTS_N: int = 0,
        MUTATOR_HOSTS_MX: float = 1.0,
        MUTATION_THRESHOLD: float = 0.0,
        IMMUNITY_CEILING: float = 1.0,
        IMMUNITY_FLOOR: float = 0.0,
        track_phylogeny: bool = False,
        DSTREAM_S: int = 64,
        DSTREAM_ALGO=None,
        N_SAMPLE: int = 2000,
        pow: float = 1.0,
    ) -> Union[
        Tuple[pd.DataFrame, pd.DataFrame],
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
    ]:
        random.seed(seed)
        np.random.seed(seed)
        xp.random.seed(seed)

        if DSTREAM_ALGO is None:
            DSTREAM_ALGO = dstream.hybrid_0_steady_1_tilted_2_algo

        MUTATION_RATE = xp.asarray(MUTATION_RATE, dtype=xp.float32)

        # Pick the smallest unsigned int dtype that holds `N_SITES` bits.
        # 16/32/64-site sweeps round-trip through uint16/uint32/uint64
        # without truncation; >64 would need a custom multi-word genome.
        if N_SITES > 64:
            raise NotImplementedError(
                "current data types support only up to 64 sites",
            )
        elif N_SITES > 32:
            genome_dtype = xp.uint64
        elif N_SITES > 16:
            genome_dtype = xp.uint32
        else:
            genome_dtype = xp.uint16

        def initialize_pop() -> (
            Tuple[xp.ndarray, xp.ndarray, xp.ndarray, xp.ndarray]
        ):
            """Initialize population statuses, genomes, and immune history."""
            pathogen_genomes = xp.zeros(shape=POP_SIZE, dtype=genome_dtype)
            host_immunities = xp.full(
                shape=(POP_SIZE, 2 * N_SITES),
                fill_value=0.0,
                dtype=xp.float32,
            )
            host_statuses = xp.full(
                shape=POP_SIZE, fill_value=0, dtype=xp.uint8
            )
            pathogen_markers = xp.random.randint(
                low=0,
                high=2**63,
                size=POP_SIZE,
                dtype=xp.int64,
            ).astype(xp.uint64)
            pathogen_markers |= xp.random.randint(
                low=0,
                high=2,
                size=POP_SIZE,
                dtype=xp.uint64,
            ) << xp.uint64(63)
            return (
                host_statuses,
                pathogen_genomes,
                host_immunities,
                pathogen_markers,
            )

        def infect_initial(
            host_statuses: xp.ndarray,
            pathogen_genomes: xp.ndarray,
        ) -> Tuple[xp.ndarray, xp.ndarray]:
            """Seed the initial infection wave with the starting strain."""
            seeded_indices = xp.random.choice(
                POP_SIZE, size=SEED_COUNT, replace=False
            )
            host_statuses[seeded_indices] = 1
            pathogen_genomes[seeded_indices] = 0  # wildtype
            return host_statuses, pathogen_genomes

        def calc_infection_probabilities(
            host_immunities: xp.ndarray,
            pathogen_genomes: xp.ndarray,
        ) -> xp.ndarray:
            host_susceptibilities = xp.reshape(
                1.0 - (IMMUNE_STRENGTH * host_immunities),
                (POP_SIZE, 2 * N_SITES),
            )

            pathogen_bits = (
                pathogen_genomes[:, None] >> xp.arange(N_SITES, dtype=xp.uint8)
            ) & 1
            pathogen_alleles = (
                pathogen_bits[:, :, None] == xp.array([0, 1])
            ).reshape(-1, 2 * N_SITES)

            active_susc = xp.where(
                pathogen_alleles, host_susceptibilities, 1.0
            )
            res = xp.prod(active_susc, axis=1)
            assert res.shape == (POP_SIZE,)
            return xp.pow(res, pow)

        if MUTATION_RATE.size == 1:

            def calc_mutation_probabilities(
                host_immunities: xp.ndarray,
                pathogen_genomes: xp.ndarray,
            ) -> xp.ndarray:
                pathogen_bits = (
                    pathogen_genomes[:, None]
                    >> xp.arange(N_SITES, dtype=xp.uint8)
                ) & 1

                imm_reshaped = xp.reshape(host_immunities, (-1, N_SITES, 2))

                idx_curr = pathogen_bits[:, :, None]
                idx_opp = 1 - idx_curr

                imm_curr = xp.take_along_axis(
                    imm_reshaped, idx_curr, axis=2
                ).squeeze(axis=2)
                imm_opp = xp.take_along_axis(
                    imm_reshaped, idx_opp, axis=2
                ).squeeze(axis=2)

                host_immunity_deltas = imm_curr - imm_opp

                b_values = 1.0 + within_host_b * host_immunity_deltas
                b_values = xp.where(
                    xp.abs(b_values - 1.0) < 1e-7, 1.000001, b_values
                )

                return (MUTATION_RATE / (b_values - 1.0)) * (
                    xp.exp((b_values - 1.0) * within_host_t) - 1.0
                )

        else:

            def calc_mutation_probabilities(
                host_immunities: xp.ndarray,
                pathogen_genomes: xp.ndarray,
            ) -> xp.ndarray:
                n_infected = host_immunities.shape[0]
                return (
                    xp.ones((n_infected, 1), dtype=MUTATION_RATE.dtype)
                    * MUTATION_RATE
                )

        def update_waning(host_immunities: xp.ndarray) -> xp.ndarray:
            host_immunities *= 1.0 - WANING_RATE
            host_immunities[host_immunities > 0] = xp.clip(
                host_immunities[host_immunities > 0],
                IMMUNITY_FLOOR,
                IMMUNITY_CEILING,
            )
            return host_immunities

        def update_recoveries(
            host_statuses: xp.ndarray,
            host_immunities: xp.ndarray,
            pathogen_genomes: xp.ndarray,
        ) -> Tuple[xp.ndarray, xp.ndarray]:
            recovered_mask = (host_statuses >= 1) * xp.random.rand(
                POP_SIZE
            ) > 1 - RECOVERY_RATE

            pathogen_bits = (
                pathogen_genomes[:, None] >> xp.arange(N_SITES, dtype=xp.uint8)
            ) & 1
            pathogen_alleles = (
                pathogen_bits[:, :, None] == xp.array([0, 1])
            ).reshape(-1, 2 * N_SITES)

            assert np.all(
                pathogen_alleles[recovered_mask].sum(axis=1) == N_SITES
            )

            host_immunities[
                pathogen_alleles.astype(bool) & recovered_mask[:, None]
            ] = 1.0

            host_statuses += (host_statuses > 0).astype(xp.uint8)
            host_statuses[recovered_mask] = 0

            return host_statuses, host_immunities

        def transmit_infection(
            host_statuses: xp.ndarray,
            pathogen_genomes: xp.ndarray,
            host_immunities: xp.ndarray,
            pathogen_markers: xp.ndarray,
        ) -> Tuple[xp.ndarray, xp.ndarray, xp.ndarray]:
            contacts = xp.random.randint(
                low=0, high=POP_SIZE, size=POP_SIZE, dtype=xp.uint32
            )
            inf_probs = (
                calc_infection_probabilities(
                    host_immunities, pathogen_genomes[contacts]
                )
                * (host_statuses == 0)
                * (host_statuses[contacts] > 0)
                * CONTACT_RATE
            )

            new_infections = xp.random.rand(POP_SIZE) < inf_probs
            host_statuses[new_infections] = 1
            pathogen_genomes[new_infections] = pathogen_genomes[contacts][
                new_infections
            ]
            pathogen_markers[new_infections] = pathogen_markers[contacts][
                new_infections
            ]

            return host_statuses, pathogen_genomes, pathogen_markers

        def apply_mutations(
            pathogen_genomes: xp.ndarray,
            host_statuses: xp.ndarray,
        ) -> xp.ndarray:
            mutation_mask = (host_statuses == 1).astype(bool)

            mprobs = calc_mutation_probabilities(
                host_immunities[mutation_mask],
                pathogen_genomes[mutation_mask],
            )
            mutator_n = host_statuses[:MUTATOR_HOSTS_N].sum()
            mprobs[:mutator_n] = xp.minimum(
                mprobs[:mutator_n] * MUTATOR_HOSTS_MX, 1.0
            )

            mprobs[mprobs < MUTATION_THRESHOLD] = 0.0

            for s in range(N_SITES):
                mutation_occurs = (
                    xp.random.rand(mprobs.shape[0]) < mprobs[:, s]
                ).astype(genome_dtype)
                pathogen_genomes[mutation_mask] ^= (
                    mutation_occurs << s
                ).astype(genome_dtype)

            return pathogen_genomes

        def deposit_strata(
            pathogen_markers: xp.ndarray,
            t: int,
        ) -> xp.ndarray:
            site = DSTREAM_ALGO.assign_storage_site(DSTREAM_S, t + DSTREAM_S)
            if site is None:
                return pathogen_markers
            new_bits = xp.random.randint(
                0,
                2,
                size=POP_SIZE,
                dtype=xp.uint64,
            )
            pathogen_markers ^= new_bits << np.uint64(63 - site)
            return pathogen_markers

        def update_simulation(
            host_statuses: xp.ndarray,
            pathogen_genomes: xp.ndarray,
            host_immunities: xp.ndarray,
            pathogen_markers: xp.ndarray,
            t: int,
        ) -> Tuple[xp.ndarray, xp.ndarray, xp.ndarray, xp.ndarray]:
            (
                host_statuses,
                pathogen_genomes,
                pathogen_markers,
            ) = transmit_infection(
                host_statuses,
                pathogen_genomes,
                host_immunities,
                pathogen_markers,
            )
            pathogen_genomes = apply_mutations(pathogen_genomes, host_statuses)
            host_statuses, host_immunities = update_recoveries(
                host_statuses, host_immunities, pathogen_genomes
            )
            host_immunities = update_waning(host_immunities)
            pathogen_markers = deposit_strata(pathogen_markers, t)

            return (
                host_statuses,
                pathogen_genomes,
                host_immunities,
                pathogen_markers,
            )

        (
            host_statuses,
            pathogen_genomes,
            host_immunities,
            pathogen_markers,
        ) = initialize_pop()
        host_statuses, pathogen_genomes = infect_initial(
            host_statuses, pathogen_genomes
        )
        data_log: List[Dict[str, float]] = []
        hw_log: List[Dict[str, float]] = []

        for t in tqdm(range(N_STEPS), mininterval=20.0):
            (
                host_statuses,
                pathogen_genomes,
                host_immunities,
                pathogen_markers,
            ) = update_simulation(
                host_statuses,
                pathogen_genomes,
                host_immunities,
                pathogen_markers,
                t,
            )

            inf_mask = host_statuses > 0
            hw_prevalences: List[float] = [0.0] * (N_SITES + 1)

            if xp.any(inf_mask):
                infected_bits = (
                    pathogen_genomes[inf_mask, None]
                    >> xp.arange(N_SITES, dtype=xp.uint8)
                ) & 1
                hw_per_host = infected_bits.sum(axis=1).astype(xp.int64)
                hw_counts_arr = xp.bincount(hw_per_host, minlength=N_SITES + 1)
                hw_prevalences = [
                    float(c) / POP_SIZE for c in hw_counts_arr.tolist()
                ]

            for w, count in enumerate(hw_prevalences):
                hw_log.append(
                    {
                        "Step": t,
                        "Seed": seed,
                        "hw": w,
                        "count": count,
                    }
                )

            avg_susc = xp.mean(
                1.0 - (IMMUNE_STRENGTH * host_immunities), axis=0
            )

            immunity_dict = {}
            for i in range(2 * N_SITES):
                site = i // 2
                bit = i % 2
                immunity_dict[f"Susc_S{site}_B{bit}"] = float(avg_susc[i])

            log_entry = {
                "Step": t,
                "Seed": seed,
                "Total_Infected": float(xp.sum(inf_mask)) / POP_SIZE,
            }
            log_entry.update(immunity_dict)
            data_log.append(log_entry)

        df = pd.DataFrame(data_log).fillna(0).copy()
        hw_df = pd.DataFrame(hw_log)
        if not track_phylogeny:
            return df, hw_df

        algo_name = f"dstream.{DSTREAM_ALGO.__name__}"
        S = DSTREAM_S
        T_bitwidth = 32
        bitwidth = 1
        bytes_per_T = T_bitwidth // 8
        bytes_per_storage = S * bitwidth // 8
        bytes_per_row = bytes_per_T + bytes_per_storage
        empty_records = pd.DataFrame(
            {
                "data_hex": pd.Series([], dtype=str),
                "dstream_algo": pd.Series([], dtype=str),
                "dstream_storage_bitoffset": pd.Series([], dtype="int64"),
                "dstream_storage_bitwidth": pd.Series([], dtype="int64"),
                "dstream_T_bitoffset": pd.Series([], dtype="int64"),
                "dstream_T_bitwidth": pd.Series([], dtype="int64"),
                "dstream_S": pd.Series([], dtype="int64"),
                "extant": pd.Series([], dtype=bool),
                "snapshot_step": pd.Series([], dtype="int64"),
                "genome": pd.Series([], dtype="int64"),
                "taxon_id": pd.Series([], dtype="int64"),
            },
        )
        infected_idx = xp.where(host_statuses > 0)[0]
        n_inf = int(infected_idx.size)
        if n_inf == 0:
            return df, hw_df, empty_records

        n_sample = min(N_SAMPLE, n_inf)
        sampled = (
            xp.random.choice(infected_idx, size=n_sample, replace=False)
            if n_sample < n_inf
            else infected_idx
        )

        # CUPY SUPPORT: Transfer to host if using GPU
        _to_np = (lambda a: np.asarray(a)) if xp is np else (lambda a: a.get())
        sampled_markers = _to_np(pathogen_markers[sampled])
        sampled_genomes = _to_np(pathogen_genomes[sampled])
        # Rank includes implicit predeposits
        sampled_steps = np.full(n_sample, N_STEPS + DSTREAM_S, dtype=np.uint32)
        sampled_taxon_ids = np.arange(n_sample, dtype=np.int64)

        T_bytes_hex = sampled_steps.astype(">u4").tobytes().hex()
        marker_bytes_hex = sampled_markers.astype(">u8").tobytes().hex()
        data_hex = [
            (
                T_bytes_hex[i * bytes_per_T * 2 : (i + 1) * bytes_per_T * 2]
                + marker_bytes_hex[
                    i * bytes_per_storage * 2 : (i + 1) * bytes_per_storage * 2
                ]
            )
            for i in range(n_sample)
        ]

        records_df = pd.DataFrame(
            {
                "data_hex": data_hex,
                "dstream_algo": algo_name,
                "dstream_storage_bitoffset": bytes_per_T * 8,
                "dstream_storage_bitwidth": bytes_per_storage * 8,
                "dstream_T_bitoffset": 0,
                "dstream_T_bitwidth": T_bitwidth,
                "dstream_S": S,
                "extant": True,
                "snapshot_step": sampled_steps.astype(np.int64),
                "genome": sampled_genomes.astype(np.int64),
                "taxon_id": sampled_taxon_ids,
            }
        )
        assert all(len(h) == bytes_per_row * 2 for h in data_hex), (
            f"hex length mismatch (expected {bytes_per_row * 2}, "
            f"got {set(len(h) for h in data_hex)})"
        )
        del data_hex, T_bytes_hex, marker_bytes_hex
        gc.collect()
        return df, hw_df, records_df

    return (simulate,)


@app.cell(hide_code=True)
def delimit_reconstruct(mo):
    mo.md("""
    ## Surface-Annotation Reconstruction

    Given the snapshot records emitted by `simulate(..., track_phylogeny=
    True)`, run `hstrat.dataframe.surface_unpack_reconstruct` followed by
    `hstrat.dataframe.surface_postprocess_trie` to estimate the phylogenetic
    tree. We use `AssignOriginTimeNodeRankTriePostprocessor(t0="dstream_S")`
    so that origin times line up with simulation step numbers.
    """)
    return


@app.cell
def def_reconstruct_phylogeny(gc, hstrat, pd, pl):
    def reconstruct_phylogeny(
        records_df: pd.DataFrame,
    ) -> pd.DataFrame:
        in_df = pl.from_pandas(records_df)
        post = hstrat.dataframe.surface_build_tree(
            in_df,
            trie_postprocessor=(
                hstrat.phylogenetic_inference.AssignOriginTimeNodeRankTriePostprocessor(
                    t0="dstream_S",
                )
            ),
        ).to_pandas()
        gc.collect()
        for col in ("id", "ancestor_id"):
            if col in post.columns:
                post[col] = post[col].astype("int64")
        post["extant"] = post["extant"].fillna(False).astype(bool)
        for col in ("snapshot_step", "genome", "origin_time"):
            if col in post.columns:
                post[col] = pd.to_numeric(post[col], errors="coerce")
        return post

    return (reconstruct_phylogeny,)


@app.cell(hide_code=True)
def delimit_phylogeny(mo):
    mo.md("""
    ## Surface-Reconstructed Phylogeny

    Sweep `N_SITES` over a few values and render the surface-reconstructed
    phylogeny next to the absolute-prevalence and Hamming-weight stackplots.
    """)
    return


@app.cell
def def_make_phylogeny_plot(
    FuncFormatter,
    MaxNLocator,
    draw_scatter_tree,
    np,
    pathlib,
    pd,
    pfl,
    plt,
    sns,
    tp,
):
    def make_phylogeny_plot(
        N_SITES: int,
        phylo_df,
        hw_df,
        phylogeny_df,
        max_tips: int = 10_000,
        seed: int = 0,
        palette: str = "rocket_r",
        teeplot_outattrs: dict = {},
    ) -> None:
        pruned_df = (
            pfl.alifestd_downsample_tips_uniform_asexual(
                phylogeny_df,
                n_downsample=max_tips,
                seed=0,
            )
            .pipe(pfl.alifestd_add_global_root, root_attrs={"origin_time": 0})
            .pipe(pfl.alifestd_collapse_unifurcations)
            .pipe(pfl.alifestd_try_add_ancestor_list_col)
            .pipe(pfl.alifestd_ladderize_asexual)
            .pipe(pfl.alifestd_assign_contiguous_ids)
        )
        assert pfl.alifestd_validate(pruned_df)
        print(f"  downsampled tree: {len(pruned_df)} nodes")
        print(f"  leaf count: {pfl.alifestd_count_leaf_nodes(pruned_df)}")

        pruned_df = pruned_df.assign(
            hw=pruned_df["genome"].map(
                lambda g: None if pd.isna(g) else bin(int(g)).count("1"),
            ),
        )
        pruned_df = pruned_df.assign(
            hw_label=pruned_df["hw"].map(
                lambda w: None if pd.isna(w) else f"HW {int(w)}"
            ),
        )

        hw_values = list(range(N_SITES + 1))
        hw_palette = sns.color_palette(palette, len(hw_values))

        present_hw = sorted(int(w) for w in hw_df["hw"].unique())
        hw_labels = [f"HW {w}" for w in present_hw]
        plot_df = hw_df.assign(
            y=-hw_df["Step"],
            hw_label=hw_df["hw"].map(lambda w: f"HW {int(w)}"),
        )
        plot_df = plot_df[plot_df["count"] > 0]

        if len(present_hw) > 4:
            _idx = np.unique(
                np.linspace(0, len(present_hw) - 1, 4).round().astype(int)
            ).tolist()
        else:
            _idx = list(range(len(present_hw)))
        hw_legend_entries = [(present_hw[i], hw_labels[i]) for i in _idx]

        def _hw_handle(w, label):
            return plt.Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor=hw_palette[w],
                markersize=10,
                label=label,
            )

        hw_palette_map = {f"HW {w}": hw_palette[w] for w in present_hw}

        with tp.teed(
            plt.subplots,
            nrows=1,
            ncols=3,
            figsize=(8, 6),
            gridspec_kw={
                "width_ratios": [1.4, 1.0, 1.0],
                "wspace": 0.1,
            },
            sharey=True,
            teeplot_outattrs=teeplot_outattrs,
            teeplot_show=True,
            teeplot_subdir=pathlib.Path(__file__).stem,
        ) as (fig, axes):
            ax_tree, ax_strain, ax_hw = axes

            sns.histplot(
                data=plot_df,
                y="y",
                hue="hw_label",
                hue_order=hw_labels,
                weights="count",
                binwidth=phylo_df["Step"].diff().min(),
                multiple="stack",
                stat="count",
                element="poly",
                palette=hw_palette_map,
                ax=ax_strain,
                fill=True,
                linewidth=0,
                legend=False,
            )
            _band_xs = [
                c.get_paths()[0].vertices[:, 0].max()
                for c in ax_strain.collections
                if c.get_paths()
            ]
            if _band_xs:
                _peak = max(_band_xs)
                _lo, _ = ax_strain.get_xlim()
                ax_strain.set_xlim(_lo, _peak * 1.05)

            sns.kdeplot(
                data=plot_df,
                y="y",
                hue="hw_label",
                hue_order=hw_labels,
                weights="count",
                multiple="fill",
                common_norm=True,
                cut=0,
                palette=hw_palette_map,
                ax=ax_hw,
                fill=True,
                linewidth=0,
                legend=False,
                bw_adjust=0.5,
            )

            for ax in (ax_strain, ax_hw):
                ax.set_xlabel("")
            ax_hw.set_xlim(0, 1)

            ax_tree.set_ylabel("step")
            ax_tree.tick_params(left=True, labelleft=True)
            for ax in (ax_strain, ax_hw):
                ax.tick_params(labelleft=False, left=False)
                ax.xaxis.tick_top()
                ax.xaxis.set_label_position("top")

            ax_tree.tick_params(bottom=False, labelbottom=False)
            sns.despine(ax=ax_strain, left=True, bottom=True, top=False)
            sns.despine(ax=ax_hw, left=True, bottom=True, top=False)

            ax_hw.legend(
                handles=[
                    _hw_handle(w, label) for w, label in hw_legend_entries
                ],
                title="Hamming weight",
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=False,
                handletextpad=0.4,
            )

            draw_scatter_tree(
                pruned_df.reset_index(drop=True),
                ax=ax_tree,
                hue="hw_label",
                scatter_kws=dict(
                    legend=False,
                    palette=hw_palette_map,
                ),
                tree_kws=dict(
                    edge_color="gray",
                    edge_linewidth=0.7,
                    edge_zorder=1,
                    ladderize=True,
                ),
            )
            sns.despine(ax=ax_tree, top=True, right=True, bottom=True)
            ax_tree.set_ylim(ax_hw.get_ylim())
            ax_tree.set_aspect("auto")
            ax_tree.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax_tree.yaxis.set_major_formatter(
                FuncFormatter(lambda x, _: f"{abs(int(round(x)))}"),
            )

    return (make_phylogeny_plot,)


@app.cell
def def_make_strain_graph_plot(ig, iplotx, np, pathlib, plt, sns, tp):
    def make_strain_graph_plot(
        N_SITES: int,
        records_df,
        max_n: int = 4,
        palette: str = "rocket_r",
        teeplot_outattrs: dict = {},
    ) -> None:
        """Plot final-step strains as undirected graphs at thresholds n=1..max_n.

        Nodes are unique strains (sized by population copies, colored by
        Hamming weight, 80% alpha). Edges connect strains within Hamming
        distance ``n``. For ``n > 1`` only edges that are not implicitly
        present via a 2-hop path through previously-added edges are drawn,
        with line thickness encoding the threshold ``n`` at which the edge
        was introduced.
        """
        extant = records_df[records_df["extant"]].copy()
        if len(extant) == 0:
            print("  (no extant records --- skipping strain graph plot)")
            return

        genomes = extant["genome"].astype("int64").to_numpy()
        strains, counts = np.unique(genomes, return_counts=True)
        n_strains = int(strains.size)
        if n_strains < 2:
            print(
                f"  (only {n_strains} strain --- skipping strain graph plot)"
            )
            return

        hamming_weights = np.array(
            [bin(int(s)).count("1") for s in strains], dtype=int
        )

        xor = strains[:, None] ^ strains[None, :]
        dists = np.zeros_like(xor, dtype=np.int32)
        tmp = xor.copy()
        while tmp.any():
            dists += (tmp & 1).astype(np.int32)
            tmp >>= 1

        cumulative = {}  # (i, j) with i < j -> threshold n at which added
        adjacency = [set() for _ in range(n_strains)]
        snapshots = []
        for n in range(1, max_n + 1):
            iu, ju = np.where(np.triu(dists == n, k=1))
            for i, j in zip(iu.tolist(), ju.tolist()):
                if n > 1 and adjacency[i] & adjacency[j]:
                    continue
                cumulative[(i, j)] = n
                adjacency[i].add(j)
                adjacency[j].add(i)
            snapshots.append((n, dict(cumulative)))

        palette_colors = sns.color_palette(palette, N_SITES + 1)
        node_facecolors = [
            (*palette_colors[int(hw)], 0.8) for hw in hamming_weights
        ]

        max_count = int(counts.max())
        sizes = 4.0 + 18.0 * np.sqrt(counts / max_count)

        # Label only the dominant strains: those carrying more than 10% of
        # the sampled population, annotated with their Hamming weight.
        total = int(counts.sum())
        vertex_labels = [
            str(int(hamming_weights[i])) if counts[i] > 0.10 * total else ""
            for i in range(n_strains)
        ]

        # Layout: kamada_kawai on the densest graph (max_n) so vertex
        # positions are stable across subplots.
        ref_graph = ig.Graph(n=n_strains)
        ref_graph.add_edges(list(cumulative.keys()))
        if ref_graph.ecount() > 0:
            try:
                layout = ref_graph.layout_kamada_kawai()
            except Exception:
                layout = ref_graph.layout_fruchterman_reingold()
        else:
            layout = ref_graph.layout_circle()
        layout_coords = [tuple(c) for c in layout.coords]

        with tp.teed(
            plt.subplots,
            nrows=1,
            ncols=max_n,
            figsize=(4.5 * max_n, 4.5),
            teeplot_outattrs=teeplot_outattrs,
            teeplot_show=True,
            teeplot_subdir=pathlib.Path(__file__).stem,
        ) as (fig, axes):
            if max_n == 1:
                axes = [axes]

            for ax, (n, edges_dict) in zip(axes, snapshots):
                edges = list(edges_dict.keys())
                # 1-hop edges thickest, with progressively thinner widths
                # for higher-n edges; n>1 edges drawn dashed to distinguish
                # them from direct mutational neighbors.
                widths = [3.0 / edges_dict[e] for e in edges]
                linestyles = [
                    "-" if edges_dict[e] == 1 else "--" for e in edges
                ]

                g_plot = ig.Graph(n=n_strains)
                if edges:
                    g_plot.add_edges(edges)

                iplotx.network(
                    g_plot,
                    layout=layout_coords,
                    vertex_facecolor=node_facecolors,
                    vertex_edgecolor="black",
                    vertex_size=sizes.tolist(),
                    vertex_labels=vertex_labels,
                    vertex_label_color="black",
                    vertex_label_size=8,
                    edge_linewidth=widths if widths else 1.0,
                    edge_linestyle=linestyles if linestyles else "-",
                    edge_color="gray",
                    ax=ax,
                    show=False,
                )
                ax.set_title(f"n = {n}")
                ax.set_aspect("equal")

            # Match make_phylogeny_plot's legend: span the same HW limits
            # (min/max of present Hamming weights), with up to 4
            # evenly-spaced entries.
            present_hw = sorted(int(w) for w in set(hamming_weights.tolist()))
            if len(present_hw) > 4:
                idx = np.unique(
                    np.linspace(0, len(present_hw) - 1, 4).round().astype(int)
                ).tolist()
                legend_hw = [present_hw[i] for i in idx]
            else:
                legend_hw = present_hw

            handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=(*palette_colors[w], 0.8),
                    markeredgecolor="black",
                    markersize=10,
                    label=f"HW {w}",
                )
                for w in legend_hw
            ]
            axes[-1].legend(
                handles=handles,
                title="Hamming weight",
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=False,
                handletextpad=0.4,
            )

    return (make_strain_graph_plot,)


@app.cell
def run_phylogeny_sweep(
    ENGINE,
    N_REPLICATES,
    N_STEPS,
    POP_SIZE,
    POW,
    SKIP_PLOTTING,
    gc,
    make_phylogeny_plot,
    make_strain_graph_plot,
    pathlib,
    pd,
    reconstruct_phylogeny,
    simulate,
    uuid,
):
    PHYLO_MUTATION_RATE = 1e-5
    POW_ = POW

    nbname = pathlib.Path(__file__).stem
    out_dir = pathlib.Path("outdata") / nbname
    out_dir.mkdir(parents=True, exist_ok=True)

    def _save_replicate(kind: str, uid: str, df) -> None:
        rep_dir = out_dir / kind / uid
        rep_dir.mkdir(parents=True, exist_ok=True)
        rep_path = rep_dir / f"a={kind}+what={nbname}+ext=.pqt"
        df.to_parquet(rep_path, index=False)
        print(f"  wrote {kind} parquet ({len(df)} rows): {rep_path}")

    traj_chunks = []
    hw_chunks = []
    records_chunks = []
    phylo_chunks = []

    for _seed in range(1, N_REPLICATES + 1):
        for PHYLO_N_SITES in (2, 3, 4, 8, 16):
            replicate_uid = uuid.uuid4().hex
            print(
                f"=== seed={_seed} N_SITES={PHYLO_N_SITES} "
                f"uid={replicate_uid} ===",
            )
            _phylo_df, _hw_df, _records_df = simulate(
                MUTATION_RATE=PHYLO_MUTATION_RATE,
                N_SITES=PHYLO_N_SITES,
                N_STEPS=N_STEPS,
                POP_SIZE=POP_SIZE,
                CONTACT_RATE=0.35,
                RECOVERY_RATE=0.1,
                WANING_RATE=0.01,
                IMMUNE_STRENGTH=0.95,
                SEED_COUNT=2,
                IMMUNITY_FLOOR=0.05,
                IMMUNITY_CEILING=1.0,
                seed=_seed,
                track_phylogeny=True,
                N_SAMPLE=200,
                pow=POW_,
            )
            print(f"  snapshot rows: {len(_records_df)}")

            _params = {
                "replicate_uid": replicate_uid,
                "seed": _seed,
                "n_sites": PHYLO_N_SITES,
                "pop_size": POP_SIZE,
                "n_steps": N_STEPS,
                "engine": ENGINE,
                "pow": POW_,
            }
            _traj_stamped = _phylo_df.assign(**_params)
            _hw_stamped = _hw_df.assign(**_params)
            traj_chunks.append(_traj_stamped)
            hw_chunks.append(_hw_stamped)
            _save_replicate("traj", replicate_uid, _traj_stamped)
            _save_replicate("hw", replicate_uid, _hw_stamped)

            _records_stamped = _records_df.assign(**_params)
            records_chunks.append(_records_stamped)
            _save_replicate("records", replicate_uid, _records_stamped)

            if len(_records_df) == 0:
                print("  (no infected hosts --- skipping plot)")
                del _phylo_df, _hw_df, _records_df
                gc.collect()
                continue

            print(f"  extant rows: {int(_records_df['extant'].sum())}")
            for palette in "rocket_r", "tab20", "tab10":
                if SKIP_PLOTTING:
                    print("  (SKIP_PLOTTING=True — skipping strain graph)")
                else:
                    make_strain_graph_plot(
                        PHYLO_N_SITES,
                        _records_df,
                        max_n=4,
                        palette=palette,
                        teeplot_outattrs={
                            "a": "strain-graph",
                            "n_sites": PHYLO_N_SITES,
                            "n_steps": int(_phylo_df["Step"].max()) + 1,
                            "replicate": _seed,
                            "palette": palette,
                            "pow": POW_,
                        },
                    )
            _phylogeny_df = reconstruct_phylogeny(_records_df)
            del _records_df
            gc.collect()
            print(f"  reconstructed: {len(_phylogeny_df)} nodes")
            print(f"  extant tips: {int(_phylogeny_df['extant'].sum())}")

            _phylo_stamped = _phylogeny_df.assign(**_params)
            phylo_chunks.append(_phylo_stamped)
            _save_replicate("phylo", replicate_uid, _phylo_stamped)

            for palette in "rocket_r", "tab20", "tab10":
                if SKIP_PLOTTING:
                    print("  (SKIP_PLOTTING=True — skipping plot)")
                else:
                    make_phylogeny_plot(
                        PHYLO_N_SITES,
                        _phylo_df,
                        _hw_df,
                        _phylogeny_df,
                        seed=_seed,
                        palette=palette,
                        teeplot_outattrs={
                            "n_sites": PHYLO_N_SITES,
                            "n_steps": int(_phylo_df["Step"].max()) + 1,
                            "replicate": _seed,
                            "palette": palette,
                            "method": "hstrat-surface",
                            "pow": POW_,
                        },
                    )
            del _phylo_df, _hw_df, _phylogeny_df
            gc.collect()

    traj_df_all = (
        pd.concat(traj_chunks, ignore_index=True)
        if traj_chunks
        else pd.DataFrame()
    )
    hw_df_all = (
        pd.concat(hw_chunks, ignore_index=True)
        if hw_chunks
        else pd.DataFrame()
    )
    records_df_all = (
        pd.concat(records_chunks, ignore_index=True)
        if records_chunks
        else pd.DataFrame()
    )
    phylo_df_all = (
        pd.concat(phylo_chunks, ignore_index=True)
        if phylo_chunks
        else pd.DataFrame()
    )

    traj_path = out_dir / f"a=traj+what={nbname}+ext=.pqt"
    hw_path = out_dir / f"a=hw+what={nbname}+ext=.pqt"
    records_path = out_dir / f"a=records+what={nbname}+ext=.pqt"
    phylo_path = out_dir / f"a=phylo+what={nbname}+ext=.pqt"
    traj_df_all.to_parquet(traj_path, index=False)
    hw_df_all.to_parquet(hw_path, index=False)
    records_df_all.to_parquet(records_path, index=False)
    phylo_df_all.to_parquet(phylo_path, index=False)
    print(f"wrote trajectory parquet ({len(traj_df_all)} rows): {traj_path}")
    print(f"wrote hw parquet ({len(hw_df_all)} rows): {hw_path}")
    print(
        f"wrote records parquet ({len(records_df_all)} rows): {records_path}"
    )
    print(f"wrote phylogeny parquet ({len(phylo_df_all)} rows): {phylo_path}")
    return


if __name__ == "__main__":
    app.run()
