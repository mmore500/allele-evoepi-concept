import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell
def import_std():
    import pathlib
    import random
    from typing import Dict, List, Sequence, Tuple, Union

    return Dict, List, Sequence, Tuple, Union, pathlib, random


@app.cell
def import_pkg():
    try:
        import cupy as cp
    except ImportError:
        import numpy as cp

    # workaround: iplotx 1.7.x uses importlib.metadata without importing it
    import importlib.metadata  # noqa: F401

    import iplotx as ipx
    import marimo as mo
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    import numpy as np
    import pandas as pd
    from phyloframe import legacy as pfl
    import seaborn as sns
    from teeplot import teeplot as tp
    from tqdm.auto import tqdm
    from watermark import watermark

    return (
        FuncFormatter,
        cp,
        ipx,
        mcolors,
        mo,
        np,
        pd,
        pfl,
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
def configure_backend(cp, np):
    use_cupy = False  # use cupy backend (GPU), otherwise use numpy (CPU)
    xp = [np, cp][use_cupy]
    return (xp,)


@app.cell(hide_code=True)
def delimit_simulation(mo):
    mo.md(
        """
    ## Simulation Implementation
    """
    )
    return


@app.cell
def def_simulate(Dict, List, Sequence, Tuple, Union, np, pd, random, tqdm, xp):
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
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        random.seed(seed)
        np.random.seed(seed)
        xp.random.seed(seed)

        MUTATION_RATE = xp.asarray(MUTATION_RATE, dtype=xp.float32)

        if N_SITES > 16:
            raise NotImplementedError(
                "current data types support only up to 16 sites",
            )

        # Phylogeny tracking state. Each entry corresponds to one infection
        # event (initial seed or onward transmission); ancestor_id of a root
        # references itself, per the alife data standard.
        phylo_ids: List[int] = []
        phylo_ancestor_ids: List[int] = []
        phylo_origin_times: List[int] = []
        phylo_genomes: List[int] = []
        next_node_id = [0]

        def initialize_pop() -> (
            Tuple[xp.ndarray, xp.ndarray, xp.ndarray, xp.ndarray]
        ):
            """Initialize population statuses, genomes, and immune history."""
            # uint16 to support up to 16 sites (uint8 caps at 8).
            pathogen_genomes = xp.zeros(shape=POP_SIZE, dtype=xp.uint16)
            # host_immunities: Tracks each of the 2*N_SITES alleles
            host_immunities = xp.full(
                shape=(POP_SIZE, 2 * N_SITES),
                fill_value=0.0,
                dtype=xp.float32,
            )
            # host_statuses: days since infection (0 if not infected)
            host_statuses = xp.full(
                shape=POP_SIZE, fill_value=0, dtype=xp.uint8
            )
            # pathogen_node_ids: phylogeny node id of each host's pathogen
            # (only populated when track_phylogeny=True; -1 means uninfected)
            pathogen_node_ids = xp.full(
                shape=POP_SIZE, fill_value=-1, dtype=xp.int64
            )

            return (
                host_statuses,
                pathogen_genomes,
                host_immunities,
                pathogen_node_ids,
            )

        def infect_initial(
            host_statuses: xp.ndarray,
            pathogen_genomes: xp.ndarray,
            pathogen_node_ids: xp.ndarray,
        ) -> Tuple[xp.ndarray, xp.ndarray, xp.ndarray]:
            """Seed the initial infection wave with the starting strain."""
            seeded_indices = xp.random.choice(
                POP_SIZE, size=SEED_COUNT, replace=False
            )
            host_statuses[seeded_indices] = 1
            pathogen_genomes[seeded_indices] = 0  # wildtype
            if track_phylogeny:
                for idx in seeded_indices.tolist():
                    nid = next_node_id[0]
                    phylo_ids.append(nid)
                    phylo_ancestor_ids.append(nid)  # root self-reference
                    phylo_origin_times.append(0)
                    phylo_genomes.append(0)
                    pathogen_node_ids[idx] = nid
                    next_node_id[0] += 1
            return host_statuses, pathogen_genomes, pathogen_node_ids

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
            pathogen_node_ids: xp.ndarray,
            step: int,
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
                # Bulk-extend Python lists (per-event appends are ~10x slower
                # at million-event scale).
                new_idx = xp.where(new_infections)[0]
                n_new = int(new_idx.size)
                if n_new:
                    src_node_ids = pathogen_node_ids[contacts[new_infections]]
                    new_ids = xp.arange(
                        next_node_id[0],
                        next_node_id[0] + n_new,
                        dtype=xp.int64,
                    )
                    phylo_ids.extend(new_ids.tolist())
                    phylo_ancestor_ids.extend(src_node_ids.tolist())
                    phylo_origin_times.extend([int(step)] * n_new)
                    phylo_genomes.extend(
                        pathogen_genomes[new_infections].tolist(),
                    )
                    pathogen_node_ids[new_idx] = new_ids
                    next_node_id[0] += n_new

            return host_statuses, pathogen_genomes, pathogen_node_ids

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
                ).astype(xp.uint16)
                pathogen_genomes[mutation_mask] ^= (
                    mutation_occurs << s
                ).astype(xp.uint16)

            return pathogen_genomes

        def update_simulation(
            host_statuses: xp.ndarray,
            pathogen_genomes: xp.ndarray,
            host_immunities: xp.ndarray,
            pathogen_node_ids: xp.ndarray,
            step: int,
        ) -> Tuple[xp.ndarray, xp.ndarray, xp.ndarray, xp.ndarray]:
            """Run one step of the simulation."""
            (
                host_statuses,
                pathogen_genomes,
                pathogen_node_ids,
            ) = transmit_infection(
                host_statuses,
                pathogen_genomes,
                host_immunities,
                pathogen_node_ids,
                step,
            )
            pathogen_genomes = apply_mutations(pathogen_genomes, host_statuses)
            host_statuses, host_immunities = update_recoveries(
                host_statuses, host_immunities, pathogen_genomes
            )
            host_immunities = update_waning(host_immunities)

            return (
                host_statuses,
                pathogen_genomes,
                host_immunities,
                pathogen_node_ids,
            )

        (
            host_statuses,
            pathogen_genomes,
            host_immunities,
            pathogen_node_ids,
        ) = initialize_pop()
        host_statuses, pathogen_genomes, pathogen_node_ids = infect_initial(
            host_statuses, pathogen_genomes, pathogen_node_ids
        )
        data_log: List[Dict[str, float]] = []

        for t in tqdm(range(N_STEPS), mininterval=20.0):
            (
                host_statuses,
                pathogen_genomes,
                host_immunities,
                pathogen_node_ids,
            ) = update_simulation(
                host_statuses,
                pathogen_genomes,
                host_immunities,
                pathogen_node_ids,
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

        # Mark nodes whose lineage is still extant at end of simulation,
        # i.e. at least one currently-infected host carries that node's id.
        extant_node_ids = set(
            int(nid) for nid in pathogen_node_ids[host_statuses > 0].tolist()
        )
        phylogeny_df = pd.DataFrame(
            {
                "id": phylo_ids,
                "ancestor_id": phylo_ancestor_ids,
                "origin_time": phylo_origin_times,
                "genome": phylo_genomes,
                "extant": [nid in extant_node_ids for nid in phylo_ids],
            }
        )
        return df, phylogeny_df

    return (simulate,)


@app.cell(hide_code=True)
def delimit_phylogeny(mo):
    mo.md(
        """
    ## Exact Phylogeny Tracking

    Sweep `N_SITES` over a few values to compare phylogeny shapes; for each
    run, render the full phylogeny (including extinct lineages) via `iplotx`,
    uniformly downsampled to ≤10 000 tips with `phyloframe`. Tips and the
    absolute stackplot share a per-strain `husl` palette; the space-filling
    stackplot to the right groups strains by Hamming weight so bit-weight
    composition reads bottom-to-top.
    """
    )
    return


@app.cell
def def_make_phylogeny_plot(
    FuncFormatter,
    ipx,
    mcolors,
    np,
    pathlib,
    pfl,
    plt,
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
        # Keep extinct lineages — uniform tip downsampling + unifurcation
        # collapse is enough to keep the rendered tree legible. Synthetic
        # global root collapses multi-seed origins to a single root (iplotx
        # requirement). Ladderize last so the render reflects the final
        # topology, then reassign contiguous ids (iplotx wants id == row).
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
                lambda g: None if np.isnan(g) else format(int(g), fmt)[::-1],
            ),
        )

        all_strains = sorted(
            (format(g, fmt)[::-1] for g in range(2**N_SITES)),
            key=lambda s: (s.count("1"), s),
        )
        hw_values = list(range(N_SITES + 1))
        hw_palette = sns.color_palette("rocket_r", len(hw_values))

        # Tip dots and strain stackplot bands share this husl palette so
        # tip color matches the band color of its strain at any timestep.
        strain_palette = dict(
            zip(all_strains, sns.color_palette("husl", len(all_strains))),
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
        strain_colors = [strain_palette[s] for s in stack_strains]

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
        hw_totals = hw_layers.sum(axis=0)
        hw_layers_norm = np.where(hw_totals > 0, hw_layers / hw_totals, 0.0)

        def _fill_horiz(ax, layers, colors):
            prev = np.zeros_like(layers[0])
            for layer, color in zip(layers, colors):
                cur = prev + layer
                ax.fill_betweenx(
                    y_steps, prev, cur, color=color, edgecolor="none"
                )
                prev = cur

        # Top-5 "final" = extant viral population at end of simulation:
        # phylogeny tips marked extant (= currently-infected hosts) ranked
        # by their bit-string genome count. Top-5 "overall" stays as the
        # integrated-prevalence ranking over the whole run.
        extant_strain_counts = (
            phylogeny_df.loc[phylogeny_df["extant"].astype(bool), "genome"]
            .astype(int)
            .map(lambda g: format(int(g), fmt)[::-1])
            .value_counts()
        )
        top_final = extant_strain_counts.head(5).index.tolist()
        overall_totals = strain_layers.sum(axis=1)
        top_overall = [
            stack_strains[i] for i in np.argsort(overall_totals)[::-1][:5]
        ]

        # Cross-sample the HW legend so it never has more than 4 entries:
        # keep an evenly-spaced subset across [0, N_SITES].
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
            # Grow vertically with N_SITES; reserve a slim bottom row for
            # the three legends sitting in their own axes (overall on the
            # left, final in the centre, Hamming weights on the right).
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

            # sns.kdeplot smooths the per-strain / per-HW timeseries by
            # treating prevalence-at-step as weights on the step axis.
            # Center stackplot stacks per-strain bands (each layer keeps
            # its own normalization); the right-most stackplot fills to 1
            # so it reads as a HW composition.
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
                palette={s: strain_palette[s] for s in stack_strains},
                ax=ax_strain,
                fill=True,
                linewidth=0,
                legend=False,
                bw_adjust=0.5,
            )

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
                palette={
                    _hw_str[i]: hw_palette[i] for i in range(len(hw_values))
                },
                ax=ax_hw,
                fill=True,
                linewidth=0,
                legend=False,
                bw_adjust=0.5,
            )

            # Overplot strain bands as thin medium-dark-gray no-fill lines
            # at 50% alpha so individual strains are legible inside their
            # HW band without overpowering the fill; HW boundaries then
            # overplotted with a thicker white line to delimit HW regions.
            sns.kdeplot(
                data=_strain_long,
                y="y",
                hue="strain",
                hue_order=stack_strains,
                weights="w",
                multiple="fill",
                common_norm=True,
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
                palette={s: "white" for s in _hw_str},
                ax=ax_hw,
                fill=False,
                linewidth=1.2,
                legend=False,
                bw_adjust=0.5,
                zorder=3,
            )

            # Drop the kdeplot's "Density" x-label noise from the stackplot
            # panels; the bottom-row legends already convey what each axis
            # encodes.
            for ax in (ax_strain, ax_hw):
                ax.set_xlabel("")

            # Time axis lives on the leftmost panel; show positive step
            # numbers even though the underlying coordinate is negated to
            # align with the iplotx layout. Stackplot x-axes move to the
            # top so the time-flow direction reads consistently with the
            # tree.
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

            final_hi = int(steps[-1])
            for ax_leg, handles, title in (
                (
                    ax_leg_overall,
                    [_strain_handle(s) for s in top_overall],
                    "top 5 overall",
                ),
                (
                    ax_leg_final,
                    [_strain_handle(s) for s in top_final],
                    f"top 5 final (extant @ step {final_hi})",
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
                    ncol=len(handles),
                    frameon=False,
                    handletextpad=0.4,
                    columnspacing=1.0,
                )

    return (make_phylogeny_plot,)


@app.cell
def run_phylogeny_sweep(make_phylogeny_plot, simulate):
    # Baseline ABM/ODE-sync params; sweep N_SITES across a few genome sizes
    # to compare the resulting phylogenies. POP_SIZE / N_STEPS pulled in
    # from the sync notebook's regime; WANING_RATE bumped to 0.02 because
    # at the original 0.005 immunity saturates and larger genome runs go
    # extinct before the dynamics settle.
    PHYLO_POP_SIZE = 200_000
    PHYLO_N_STEPS = 1_200
    PHYLO_MUTATION_RATE = 5e-5

    # 3 replicates per N_SITES: outer seed, inner N_SITES so each plot
    # filename gets seed+n_sites in its teeplot outattrs and replicates
    # don't clobber each other.
    for _seed in (1, 2, 3):
        for PHYLO_N_SITES in (2, 4, 8, 16):
            print(f"=== seed={_seed} N_SITES={PHYLO_N_SITES} ===")
            _phylo_df, _phylogeny_df = simulate(
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
            )
            print(f"  phylogeny: {len(_phylogeny_df)} total nodes")
            print(f"  extant tips: {_phylogeny_df['extant'].sum()}")
            make_phylogeny_plot(
                PHYLO_N_SITES,
                _phylo_df,
                _phylogeny_df,
                seed=_seed,
            )
            # Free large per-iteration buffers before the next run.
            del _phylo_df, _phylogeny_df
    return


if __name__ == "__main__":
    app.run()
