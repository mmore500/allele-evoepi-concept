import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell
def import_std():
    import gc
    import pathlib
    import random
    from typing import Dict, List, Sequence, Tuple, Union

    return Dict, List, Sequence, Tuple, Union, gc, pathlib, random


@app.cell
def import_pkg():
    try:
        import cupy as cp
    except ImportError:
        import numpy as cp

    # workaround: iplotx 1.7.x uses importlib.metadata without importing it
    import importlib.metadata  # noqa: F401

    import downstream.dstream as dstream
    import hstrat
    import iplotx as ipx
    import marimo as mo
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    import numpy as np
    import pandas as pd
    from phyloframe import legacy as pfl
    import polars as pl
    import seaborn as sns
    from teeplot import teeplot as tp
    from tqdm.auto import tqdm
    from watermark import watermark

    from pylib import rescale_stacked_kdeplot

    return (
        FuncFormatter,
        cp,
        dstream,
        hstrat,
        ipx,
        mcolors,
        mo,
        np,
        pd,
        pfl,
        pl,
        plt,
        rescale_stacked_kdeplot,
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
def configure_backend(cp, np):
    use_cupy = False  # use cupy backend (GPU), otherwise use numpy (CPU)
    xp = [np, cp][use_cupy]
    return (xp,)


@app.cell(hide_code=True)
def delimit_simulation(mo):
    mo.md("""
    ## Simulation Implementation

    This notebook is the 32-site variant of
    `2026-04-28-allele-abm-phylogeny-hstrat.py`: pathogen genomes are widened
    from 16-bit to 32-bit so up to `N_SITES=32` allele loci can be tracked,
    and the phylogeny sweep runs 1.5× longer (12,000 steps) to give the
    larger sequence space time to explore. Phylogeny estimation still uses
    *hereditary stratigraphic surface* annotations (see
    https://hstrat.rtfd.io). Each infected host carries a `dstream_S=64`-site,
    64-bit "hybrid" surface that ingests one differentia per simulation step;
    on transmission the donor surface state is copied to the recipient (the
    ABM analogue of `CloneDescendant`). At up to `MAX_SAMPLED_TAXA` infected
    hosts are snapshotted each step, so a single
    `surface_unpack_reconstruct` + `surface_postprocess_trie` pass at the end
    yields an estimated phylogeny that includes extinct lineages.

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
        MAX_SAMPLED_TAXA: int = 200,
        SNAPSHOT_INTERVAL: int = 1,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        random.seed(seed)
        np.random.seed(seed)
        xp.random.seed(seed)

        if DSTREAM_ALGO is None:
            DSTREAM_ALGO = dstream.hybrid_0_steady_1_tilted_2_algo

        MUTATION_RATE = xp.asarray(MUTATION_RATE, dtype=xp.float32)

        if N_SITES > 32:
            raise NotImplementedError(
                "current data types support only up to 32 sites",
            )

        # Vectorized hstrat surface state. Shape (POP_SIZE, S) uint64; each
        # row mirrors a `HereditaryStratigraphicSurface(stratum_differentia_
        # bit_width=64)` buffer. Initial random fill stands in for the S
        # "predeposit" strata that fill the buffer at construction time;
        # after `step_count` simulation steps the per-row dstream_T is
        # `step_count` (predeposit ranks live at `dstream_rank < dstream_S`
        # and are stripped by `surface_postprocess_trie(delete_trunk=True)`).
        # Snapshot rows accumulate across the run; at the end they are passed
        # through `hstrat.dataframe.surface_unpack_reconstruct` for tree
        # estimation.
        # Snapshots store (step, sampled_indices, sampled_markers); each
        # snapshot row contributes one extant tip in the reconstruction's
        # input dataframe (with `extant=False` except for the final step).
        snapshot_steps: List[int] = []
        snapshot_markers: List[xp.ndarray] = []
        snapshot_genomes: List[xp.ndarray] = []
        snapshot_taxon_ids: List[xp.ndarray] = []
        next_taxon_id = [0]

        def initialize_pop() -> (
            Tuple[xp.ndarray, xp.ndarray, xp.ndarray, xp.ndarray]
        ):
            """Initialize population statuses, genomes, and immune history."""
            # uint32 to support up to 32 sites (uint16 caps at 16).
            pathogen_genomes = xp.zeros(shape=POP_SIZE, dtype=xp.uint32)
            host_immunities = xp.full(
                shape=(POP_SIZE, 2 * N_SITES),
                fill_value=0.0,
                dtype=xp.float32,
            )
            host_statuses = xp.full(
                shape=POP_SIZE, fill_value=0, dtype=xp.uint8
            )
            # Random S-stratum fill mirrors the surface's predeposit phase;
            # uninfected slots are overwritten when a host gets infected.
            pathogen_markers = xp.random.randint(
                low=0,
                high=2**63,
                size=(POP_SIZE, DSTREAM_S),
                dtype=xp.int64,
            ).astype(xp.uint64)
            pathogen_markers |= (
                xp.random.randint(
                    low=0, high=2, size=(POP_SIZE, DSTREAM_S), dtype=xp.uint64
                )
                << 63
            )
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
                pathogen_genomes[:, None] >> xp.arange(N_SITES)
            ) & 1
            pathogen_alleles = (
                pathogen_bits[:, :, None] == xp.array([0, 1])
            ).reshape(-1, 2 * N_SITES)

            active_susc = xp.where(
                pathogen_alleles, host_susceptibilities, 1.0
            )
            res = xp.prod(active_susc, axis=1)
            assert res.shape == (POP_SIZE,)
            return res

        if MUTATION_RATE.size == 1:

            def calc_mutation_probabilities(
                host_immunities: xp.ndarray,
                pathogen_genomes: xp.ndarray,
            ) -> xp.ndarray:
                pathogen_bits = (
                    pathogen_genomes[:, None] >> xp.arange(N_SITES)
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

                # \\frac{y}{x} = \\frac{m}{b-1} \\left( e^{(b-1)t} - 1 \\right)
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
            """Decay immunity levels over time."""
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
            """Recover infected individuals with probability RECOVERY_RATE."""
            # Each infected host recovers independently at rate RECOVERY_RATE,
            # regardless of how long they have been infected. This matches the
            # ODE's per-person recovery rate.
            recovered_mask = (host_statuses >= 1) * xp.random.rand(
                POP_SIZE
            ) > 1 - RECOVERY_RATE

            pathogen_bits = (
                pathogen_genomes[:, None] >> xp.arange(N_SITES)
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
            """Vectorized transmission based on allele-specific susceptibility."""
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
            if track_phylogeny:
                # Recipient inherits donor's surface state — vectorized
                # analogue of `HereditaryStratigraphicSurface.CloneDescendant`
                # without the post-clone deposit (the deposit happens later
                # in the step for all infected hosts uniformly).
                pathogen_markers[new_infections] = pathogen_markers[
                    contacts[new_infections]
                ]

            return host_statuses, pathogen_genomes, pathogen_markers

        def apply_mutations(
            pathogen_genomes: xp.ndarray,
            host_statuses: xp.ndarray,
        ) -> xp.ndarray:
            """Apply mutations to newly infected individuals."""
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
                ).astype(xp.uint32)
                pathogen_genomes[mutation_mask] ^= (
                    mutation_occurs << s
                ).astype(xp.uint32)

            return pathogen_genomes

        def deposit_strata(
            host_statuses: xp.ndarray,
            pathogen_markers: xp.ndarray,
            t: int,
        ) -> xp.ndarray:
            """Deposit one stratum on every infected host's surface.

            The site is computed once per step from the global dstream_T
            (`t` here, where `t == 1` is the first real deposit on top of
            the random S-stratum init); a `None` site means the algorithm
            chose to retain the existing buffer this step.
            """
            site = DSTREAM_ALGO.assign_storage_site(DSTREAM_S, t)
            if site is None:
                return pathogen_markers
            infected_idx = xp.where(host_statuses > 0)[0]
            n_inf = int(infected_idx.size)
            if n_inf == 0:
                return pathogen_markers
            # uint64 random fill in two halves (numpy's randint maxes at int64)
            new_lo = xp.random.randint(
                0, 2**63, size=n_inf, dtype=xp.int64
            ).astype(xp.uint64)
            new_hi = xp.random.randint(0, 2, size=n_inf, dtype=xp.uint64) << 63
            pathogen_markers[infected_idx, site] = new_lo | new_hi
            return pathogen_markers

        def maybe_snapshot(
            host_statuses: xp.ndarray,
            pathogen_markers: xp.ndarray,
            pathogen_genomes: xp.ndarray,
            t: int,
        ) -> None:
            """Sample up to MAX_SAMPLED_TAXA infected hosts at step `t`."""
            if not track_phylogeny:
                return
            if SNAPSHOT_INTERVAL <= 0:
                return
            # `surface_unpack_reconstruct` requires `dstream_T >= dstream_S`;
            # since we encode dstream_T = step (one deposit per step in the
            # track_ca.py convention), skip snapshots before step `DSTREAM_S`.
            if t < DSTREAM_S:
                return
            if t % SNAPSHOT_INTERVAL != 0 and t != N_STEPS:
                return
            infected_idx = xp.where(host_statuses > 0)[0]
            n_inf = int(infected_idx.size)
            if n_inf == 0:
                return
            n_sample = min(MAX_SAMPLED_TAXA, n_inf)
            if n_sample < n_inf:
                sampled = xp.random.choice(
                    infected_idx, size=n_sample, replace=False
                )
            else:
                sampled = infected_idx
            taxon_ids = xp.arange(
                next_taxon_id[0], next_taxon_id[0] + n_sample, dtype=xp.int64
            )
            next_taxon_id[0] += n_sample
            snapshot_steps.append(t)
            snapshot_markers.append(pathogen_markers[sampled].copy())
            snapshot_genomes.append(pathogen_genomes[sampled].copy())
            snapshot_taxon_ids.append(taxon_ids)

        def update_simulation(
            host_statuses: xp.ndarray,
            pathogen_genomes: xp.ndarray,
            host_immunities: xp.ndarray,
            pathogen_markers: xp.ndarray,
            t: int,
        ) -> Tuple[xp.ndarray, xp.ndarray, xp.ndarray, xp.ndarray]:
            """Run one step of the simulation."""
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
            if track_phylogeny:
                # Stratum deposit comes after recoveries so just-recovered
                # hosts (whose `host_statuses` is now 0) don't get a fresh
                # stratum — their surface state stops evolving once the
                # lineage dies out, which is the correct semantics.
                pathogen_markers = deposit_strata(
                    host_statuses, pathogen_markers, t
                )
                maybe_snapshot(
                    host_statuses, pathogen_markers, pathogen_genomes, t
                )

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
                t + 1,
            )

            inf_mask = host_statuses > 0
            counts_dict: Dict[str, float] = {}

            if xp.any(inf_mask):
                unique_g, counts = xp.unique(
                    pathogen_genomes[inf_mask], return_counts=True
                )

                for g, count in zip(unique_g.tolist(), counts.tolist()):
                    fmt = f"0{N_SITES}b"
                    strain_name = format(int(g), fmt)[::-1]
                    counts_dict[f"Strain_{strain_name}"] = (
                        float(count) / POP_SIZE
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
            log_entry.update(counts_dict)
            log_entry.update(immunity_dict)
            data_log.append(log_entry)

        df = pd.DataFrame(data_log).fillna(0).copy()
        if not track_phylogeny:
            return df

        # Build the surface-records dataframe expected by
        # `hstrat.dataframe.surface_unpack_reconstruct`. Each row is one
        # snapshot tip (1 sampled host at 1 step); the dstream_T column lives
        # at the front of the data_hex bytes per the layout documented in
        # `hstrat.serialization.surf_to_hex`.
        algo_name = f"dstream.{DSTREAM_ALGO.__name__}"
        S = DSTREAM_S
        T_bitwidth = 32
        bitwidth = 64
        if not snapshot_steps:
            # No infected hosts ever made it past `DSTREAM_S` steps —
            # nothing to reconstruct. Return an empty records dataframe in
            # the expected schema so the caller can detect / skip cleanly.
            empty = pd.DataFrame(
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
            return df, empty
        # Concatenate snapshot arrays for vectorized hex conversion. Per-tip
        # `dstream_T` matches the simulation step at sample time, which
        # equals the number of "real" deposits (the implicit S predeposit
        # strata sit at negative ranks and are stripped by
        # `surface_postprocess_trie(delete_trunk=True)`).
        all_steps = np.concatenate(
            [
                np.full(arr.size, t, dtype=np.uint32)
                for t, arr in zip(
                    snapshot_steps,
                    snapshot_taxon_ids,
                )
            ]
        )
        all_markers = np.concatenate(
            [np.asarray(arr) for arr in snapshot_markers],
            axis=0,
        )
        all_genomes = np.concatenate(
            [np.asarray(arr) for arr in snapshot_genomes],
        )
        all_taxon_ids = np.concatenate(
            [np.asarray(arr) for arr in snapshot_taxon_ids],
        )

        # `np.uint32.byteswap` is the equivalent of dtype=">u4" for big-endian
        # encoding; pre-pend the dstream_T bytes to the dstream_storage hex.
        T_bytes_hex = all_steps.astype(">u4").tobytes()
        marker_bytes_hex = all_markers.astype(">u8").tobytes()
        bytes_per_T = T_bitwidth // 8  # 4
        bytes_per_storage = S * (bitwidth // 8)  # 64*8 = 512
        bytes_per_row = bytes_per_T + bytes_per_storage
        n_rows = len(all_steps)
        # Interleave T bytes + storage bytes per row.
        data_hex = [
            (
                T_bytes_hex[i * bytes_per_T : (i + 1) * bytes_per_T]
                + marker_bytes_hex[
                    i * bytes_per_storage : (i + 1) * bytes_per_storage
                ]
            ).hex()
            for i in range(n_rows)
        ]
        # Mark only final-step rows extant.
        is_extant = all_steps == np.uint32(N_STEPS)

        records_df = pd.DataFrame(
            {
                "data_hex": data_hex,
                "dstream_algo": algo_name,
                "dstream_storage_bitoffset": bytes_per_T * 8,
                "dstream_storage_bitwidth": bytes_per_storage * 8,
                "dstream_T_bitoffset": 0,
                "dstream_T_bitwidth": T_bitwidth,
                "dstream_S": S,
                "extant": is_extant,
                "snapshot_step": all_steps.astype(np.int64),
                "genome": all_genomes.astype(np.int64),
                "taxon_id": all_taxon_ids.astype(np.int64),
            }
        )
        # Sanity check: every hex string is the same length.
        assert all(len(h) == bytes_per_row * 2 for h in data_hex), (
            f"hex length mismatch (expected {bytes_per_row * 2}, "
            f"got {set(len(h) for h in data_hex)})"
        )
        # Release the large per-host simulation buffers and snapshot
        # accumulators; they're no longer needed once `records_df` is
        # materialized, and freeing them up-front keeps the
        # `surface_unpack_reconstruct` peak well below the runner's
        # 12 GB memory budget. Not deleting `host_immunities` /
        # `host_statuses` etc. by name since they're captured by the
        # nested closures (`apply_mutations`, `update_recoveries`, …)
        # and ruff flags those as `F821 undefined name` if the outer
        # binding gets `del`-ed; clearing the snapshot lists below is
        # what reclaims the bulk of the per-step accumulator memory.
        del (
            data_hex,
            T_bytes_hex,
            marker_bytes_hex,
            all_steps,
            all_markers,
            all_genomes,
            all_taxon_ids,
            is_extant,
        )
        snapshot_steps.clear()
        snapshot_markers.clear()
        snapshot_genomes.clear()
        snapshot_taxon_ids.clear()
        gc.collect()
        return df, records_df

    return (simulate,)


@app.cell(hide_code=True)
def delimit_reconstruct(mo):
    mo.md("""
    ## Surface-Annotation Reconstruction

    Given the snapshot records emitted by `simulate(..., track_phylogeny=
    True)`, run `hstrat.dataframe.surface_unpack_reconstruct` followed by
    `hstrat.dataframe.surface_postprocess_trie` to estimate the phylogenetic
    tree. We use `AssignOriginTimeNodeRankTriePostprocessor(t0="dstream_S")`
    so that origin times line up with simulation step numbers (the trunk
    deposits at `dstream_rank < dstream_S` are deleted by default).
    """)
    return


@app.cell
def def_reconstruct_phylogeny(gc, hstrat, pd, pl):
    def reconstruct_phylogeny(
        records_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Run surface_unpack_reconstruct + postprocess to get an alife df.

        The returned dataframe is in alife-standard format with extra
        ``snapshot_step`` / ``genome`` / ``extant`` columns forwarded from
        the input records. The `origin_time` column reports the simulation
        step at which each (estimated) lineage diverged.
        """
        in_df = pl.from_pandas(records_df)
        recon = hstrat.dataframe.surface_unpack_reconstruct(in_df)
        # The unpack frame is the largest intermediate (one row per
        # retained stratum across all sampled tips); free it before
        # `surface_postprocess_trie` allocates its own internal buffers.
        post = hstrat.dataframe.surface_postprocess_trie(
            recon,
            trie_postprocessor=(
                hstrat.phylogenetic_inference.AssignOriginTimeNodeRankTriePostprocessor(
                    t0="dstream_S",
                )
            ),
        ).to_pandas()
        del recon, in_df
        gc.collect()
        # phyloframe's CSR builder fast path uses
        # `np.full(n, -1, dtype=ancestor_ids.dtype)`, which OverflowErrors
        # when `ancestor_id` is `pl.UInt64`; cast id columns to int64 here
        # so all downstream phyloframe pipelines (e.g. `alifestd_to_iplotx`,
        # `alifestd_downsample_tips_uniform_asexual`) work.
        for col in ("id", "ancestor_id"):
            if col in post.columns:
                post[col] = post[col].astype("int64")
        # `extant` defaults to NaN for inner nodes; cast to bool with NaN→False
        # so phyloframe predicate filters work cleanly.
        post["extant"] = post["extant"].fillna(False).astype(bool)
        # Inner nodes don't carry a snapshot/genome — coerce to numeric (NaN
        # for inner) so leaf-only `.astype(int)` calls in the plotter work.
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
    phylogeny next to the absolute-prevalence and Hamming-weight stackplots,
    matching the layout of `2026-04-28-allele-abm-phylogeny.py`. Tips and the
    stackplots share a per-strain `husl` palette.
    """)
    return


@app.cell
def def_make_phylogeny_plot(
    FuncFormatter,
    ipx,
    mcolors,
    np,
    pathlib,
    pd,
    pfl,
    plt,
    rescale_stacked_kdeplot,
    sns,
    tp,
):
    def make_phylogeny_plot(
        N_SITES: int,
        phylo_df,
        phylogeny_df,
        max_tips: int = 10_000,
        height_scale: float = 1.0,
        seed: int = 0,
    ) -> None:
        # Mirror the exact-tracking notebook's pruning steps so the figures
        # are visually comparable: uniform tip downsampling + unifurcation
        # collapse + ladderize. Synthetic global root collapses multi-seed
        # origins to a single root (iplotx requirement).
        pruned_df = (
            pfl.alifestd_downsample_tips_uniform_asexual(
                phylogeny_df,
                n_downsample=max_tips,
                seed=0,
            )
            .pipe(pfl.alifestd_add_global_root)
            .pipe(pfl.alifestd_collapse_unifurcations)
            .pipe(pfl.alifestd_try_add_ancestor_list_col)
            .pipe(pfl.alifestd_ladderize_asexual)
            .pipe(pfl.alifestd_assign_contiguous_ids)
        )
        assert pfl.alifestd_validate(pruned_df)
        print(f"  downsampled tree: {len(pruned_df)} nodes")
        print(f"  leaf count: {pfl.alifestd_count_leaf_nodes(pruned_df)}")

        fmt = f"0{N_SITES}b"
        pruned_df = pruned_df.assign(
            strain=pruned_df["genome"].map(
                lambda g: None if pd.isna(g) else format(int(g), fmt)[::-1],
            ),
        )

        # Enumerating 2**N_SITES strains is infeasible for N_SITES=32 (4B
        # entries). Derive `all_strains` from observed strains: the per-step
        # `Strain_*` columns of `phylo_df` (every strain that ever reached
        # nonzero prevalence) plus any strains carried by reconstructed-tree
        # leaves (which may include rare strains never logged at the per-step
        # granularity). Sorting by Hamming weight then bit-string preserves
        # the original color/legend ordering.
        observed_strains = {
            c[len("Strain_") :]
            for c in phylo_df.columns
            if c.startswith("Strain_")
        }
        observed_strains.update(
            format(int(g), fmt)[::-1]
            for g in pruned_df["genome"].dropna().astype("int64").unique()
        )
        all_strains = sorted(
            observed_strains,
            key=lambda s: (s.count("1"), s),
        )
        hw_values = list(range(N_SITES + 1))
        hw_palette = sns.color_palette("rocket_r", len(hw_values))

        strain_palette = dict(
            zip(all_strains, sns.color_palette("rainbow", len(all_strains))),
        )
        vertex_colors = [
            "#cccccc" if s is None else mcolors.to_hex(strain_palette[s])
            for s in pruned_df["strain"]
        ]

        strain_cols = [
            f"Strain_{s}"
            for s in all_strains
            if f"Strain_{s}" in phylo_df.columns
        ]
        stack_strains = [c[len("Strain_") :] for c in strain_cols]
        steps = phylo_df["Step"].to_numpy()
        y_steps = -steps
        strain_layers = np.stack(
            [phylo_df[c].to_numpy() for c in strain_cols], axis=0
        )
        hw_layers = np.stack(
            [
                np.sum(
                    [
                        phylo_df[f"Strain_{s}"].to_numpy()
                        for s in stack_strains
                        if s.count("1") == w
                    ]
                    or [np.zeros_like(steps, dtype=float)],
                    axis=0,
                )
                for w in hw_values
            ],
            axis=0,
        )
        # "Top final" = strains carried by surface-reconstructed extant
        # leaves (each leaf's `genome` column is the last sampled strain
        # for that lineage); "top overall" stays as the integrated-
        # prevalence ranking from the per-step strain log.
        extant_strain_counts = (
            phylogeny_df.loc[phylogeny_df["extant"].astype(bool), "genome"]
            .dropna()
            .astype(int)
            .map(lambda g: format(int(g), fmt)[::-1])
            .value_counts()
        )
        top_final = extant_strain_counts.head(6).index.tolist()
        overall_totals = strain_layers.sum(axis=1)
        top_overall = [
            stack_strains[i] for i in np.argsort(overall_totals)[::-1][:6]
        ]

        if len(hw_values) > 4:
            _idx = np.unique(
                np.linspace(0, len(hw_values) - 1, 4).round().astype(int)
            ).tolist()
        else:
            _idx = list(range(len(hw_values)))
        hw_legend_entries = [(i, hw_values[i]) for i in _idx]

        def _wrap(s, width=8):
            return "\n".join(s[i : i + width] for i in range(0, len(s), width))

        def _strain_handle(s):
            return plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=mcolors.to_hex(strain_palette[s]),
                markersize=8,
                label=f"{_wrap(s)}\n(HW {s.count('1')})",
            )

        def _hw_handle(i, w):
            return plt.Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor=hw_palette[i],
                markersize=10,
                label=f"HW {w}",
            )

        with tp.teed(
            plt.subplots,
            nrows=2,
            ncols=3,
            figsize=(12, height_scale * max(7, 4.5 + 0.4 * N_SITES) + 2.5),
            gridspec_kw={
                "width_ratios": [1.4, 1.0, 1.0],
                "height_ratios": [10, 2.5],
                "wspace": 0.1,
                "hspace": 0.06,
            },
            sharey="row",
            teeplot_outattrs={
                "what": "phylogeny",
                "n_sites": N_SITES,
                "n_steps": int(phylo_df["Step"].max()) + 1,
                "seed": seed,
                "method": "hstrat-surface",
            },
            teeplot_show=True,
            teeplot_subdir=pathlib.Path(__file__).stem,
        ) as (fig, axes):
            ax_tree, ax_strain, ax_hw = axes[0]
            ax_leg_overall, ax_leg_final, ax_leg_hw = axes[1]

            ipx.tree(
                pfl.alifestd_to_iplotx_pandas(pruned_df),
                ax=ax_tree,
                layout="vertical",
                vertex_color=vertex_colors,
                vertex_alpha=0.5,
                vertex_size=5,
                vertex_zorder=3,
                edge_color="gray",
                edge_linewidth=0.7,
                edge_zorder=1,
                margins=0.05,
                strip_axes=False,
            )

            _strain_long = pd.DataFrame(
                {
                    "y": np.tile(y_steps, len(stack_strains)),
                    "strain": np.repeat(stack_strains, len(steps)),
                    "w": strain_layers.ravel(),
                }
            )
            _strain_long = _strain_long[_strain_long["w"] > 0]
            sns.kdeplot(
                data=_strain_long,
                y="y",
                hue="strain",
                hue_order=stack_strains,
                weights="w",
                multiple="stack",
                common_norm=True,
                cut=0,
                palette={s: strain_palette[s] for s in stack_strains},
                ax=ax_strain,
                fill=True,
                linewidth=0,
                legend=False,
                bw_adjust=0.5,
            )
            rescale_stacked_kdeplot(ax_strain, orient="y", scale="log")
            _band_xs = [
                c.get_paths()[0].vertices[:, 0].max()
                for c in ax_strain.collections
                if c.get_paths()
            ]
            if _band_xs:
                _peak = max(_band_xs)
                _lo, _ = ax_strain.get_xlim()
                ax_strain.set_xlim(_lo, _peak * 1.05)

            _hw_str = [f"HW {w}" for w in hw_values]
            _hw_long = pd.DataFrame(
                {
                    "y": np.tile(y_steps, len(hw_values)),
                    "hw": np.repeat(_hw_str, len(steps)),
                    "w": hw_layers.ravel(),
                }
            )
            _hw_long = _hw_long[_hw_long["w"] > 0]
            sns.kdeplot(
                data=_hw_long,
                y="y",
                hue="hw",
                hue_order=_hw_str,
                weights="w",
                multiple="fill",
                common_norm=True,
                cut=0,
                palette={
                    _hw_str[i]: hw_palette[i] for i in range(len(hw_values))
                },
                ax=ax_hw,
                fill=True,
                linewidth=0,
                legend=False,
                bw_adjust=0.5,
            )

            sns.kdeplot(
                data=_strain_long,
                y="y",
                hue="strain",
                hue_order=stack_strains,
                weights="w",
                multiple="fill",
                common_norm=True,
                cut=0,
                palette={s: "#555555" for s in stack_strains},
                ax=ax_hw,
                fill=False,
                linewidth=0.3,
                alpha=0.5,
                legend=False,
                bw_adjust=0.5,
                zorder=2,
            )
            sns.kdeplot(
                data=_hw_long,
                y="y",
                hue="hw",
                hue_order=_hw_str,
                weights="w",
                multiple="fill",
                common_norm=True,
                cut=0,
                palette={s: "white" for s in _hw_str},
                ax=ax_hw,
                fill=False,
                linewidth=1.2,
                legend=False,
                bw_adjust=0.5,
                zorder=3,
            )

            for ax in (ax_strain, ax_hw):
                ax.set_xlabel("")
            ax_hw.set_xlim(0, 1)

            ax_tree.set_ylabel("step")
            ax_tree.yaxis.set_major_formatter(
                FuncFormatter(lambda v, _pos: f"{abs(int(round(-v)))}"),
            )
            ax_tree.tick_params(left=True, labelleft=True)
            for ax in (ax_strain, ax_hw):
                ax.tick_params(labelleft=False, left=False)
                ax.xaxis.tick_top()
                ax.xaxis.set_label_position("top")

            sns.despine(ax=ax_tree, top=True, right=True, bottom=True)
            ax_tree.tick_params(bottom=False, labelbottom=False)
            sns.despine(ax=ax_strain, left=True, bottom=True, top=False)
            sns.despine(ax=ax_hw, left=True, bottom=True, top=False)

            for ax_leg, handles, title in (
                (
                    ax_leg_overall,
                    [_strain_handle(s) for s in top_overall],
                    "top 6 overall",
                ),
                (
                    ax_leg_final,
                    [_strain_handle(s) for s in top_final],
                    "top 6 extant",
                ),
                (
                    ax_leg_hw,
                    [_hw_handle(i, w) for i, w in hw_legend_entries],
                    "Hamming weight",
                ),
            ):
                ax_leg.set_axis_off()
                ax_leg.legend(
                    handles=handles,
                    title=title,
                    loc="center",
                    ncol=3,
                    frameon=False,
                    handletextpad=0.4,
                    columnspacing=1.0,
                )

    return (make_phylogeny_plot,)


@app.cell
def run_phylogeny_sweep(
    gc,
    make_phylogeny_plot,
    reconstruct_phylogeny,
    simulate,
):
    # Sweep at the widened 32-site genome only, for 1.5× the runtime of
    # the 16-site notebook (12,000 steps × 3 replicates) so the resulting
    # reconstructed phylogenies have time to coalesce into informative
    # topology over the larger sequence space. `MAX_SAMPLED_TAXA` caps
    # memory; it bounds the number of (snapshot × infected-host) sample
    # rows that flow into `surface_unpack_reconstruct`. With
    # `MAX_SAMPLED_TAXA=1` and `SNAPSHOT_INTERVAL=1`, a 12,000-step run
    # produces ~12k records (each ~1 KB of hex), keeping the records
    # dataframe + reconstruction intermediate buffers well under the
    # ~12 GB GitHub-runner memory budget. Use the canonical 64-bit hybrid
    # algo, `dstream.hybrid_0_steady_1_tilted_2_algo`.
    PHYLO_POP_SIZE = 200_000
    PHYLO_N_STEPS = 12_000
    PHYLO_MUTATION_RATE = 1e-5

    for _seed in (1, 2, 3):
        for PHYLO_N_SITES in (32,):
            print(f"=== seed={_seed} N_SITES={PHYLO_N_SITES} ===")
            _phylo_df, _records_df = simulate(
                MUTATION_RATE=PHYLO_MUTATION_RATE,
                N_SITES=PHYLO_N_SITES,
                N_STEPS=PHYLO_N_STEPS,
                POP_SIZE=PHYLO_POP_SIZE,
                CONTACT_RATE=0.35,
                RECOVERY_RATE=0.1,
                WANING_RATE=0.02,
                IMMUNE_STRENGTH=0.7,
                SEED_COUNT=2,
                IMMUNITY_FLOOR=0.05,
                IMMUNITY_CEILING=1.0,
                seed=_seed,
                track_phylogeny=True,
                MAX_SAMPLED_TAXA=1,
                SNAPSHOT_INTERVAL=1,
            )
            print(f"  snapshot rows: {len(_records_df)}")
            if len(_records_df) == 0:
                print("  (no infected hosts past S=64 — skipping plot)")
                del _phylo_df, _records_df
                gc.collect()
                continue
            print(f"  extant rows: {int(_records_df['extant'].sum())}")
            _phylogeny_df = reconstruct_phylogeny(_records_df)
            # Free the records dataframe right after reconstruct — the hex
            # column is the biggest single buffer in the pipeline and isn't
            # needed once the alife df is built.
            del _records_df
            gc.collect()
            print(f"  reconstructed: {len(_phylogeny_df)} nodes")
            print(f"  extant tips: {int(_phylogeny_df['extant'].sum())}")
            make_phylogeny_plot(
                PHYLO_N_SITES,
                _phylo_df,
                _phylogeny_df,
                seed=_seed,
            )
            del _phylo_df, _phylogeny_df
            gc.collect()
    return


if __name__ == "__main__":
    app.run()
