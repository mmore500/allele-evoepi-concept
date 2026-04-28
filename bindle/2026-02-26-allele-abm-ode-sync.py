import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell
def import_std():
    import itertools as it
    import pathlib
    import random
    from typing import Dict, List, Sequence, Tuple, Union

    return Dict, List, Sequence, Tuple, Union, it, pathlib, random


@app.cell
def import_pkg():
    try:
        import cupy as cp
    except ImportError:
        import numpy as cp

    from IPython.display import display_html
    import marimo as mo
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from teeplot import teeplot as tp
    from tqdm.auto import tqdm
    from watermark import watermark

    return cp, display_html, mo, np, pd, sns, tp, tqdm, watermark


@app.cell(hide_code=True)
def do_watermark(mo, watermark):
    mo.md(f"""
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
    """)
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
    """)
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
    ) -> pd.DataFrame:
        random.seed(seed)
        np.random.seed(seed)
        xp.random.seed(seed)

        MUTATION_RATE = xp.asarray(MUTATION_RATE, dtype=xp.float32)

        if N_SITES > 8:
            raise NotImplementedError(
                "current data types support only up to 8 sites",
            )

        def initialize_pop() -> Tuple[xp.ndarray, xp.ndarray, xp.ndarray]:
            """Initialize population statuses, genomes, and immune history."""
            pathogen_genomes = xp.zeros(shape=POP_SIZE, dtype=xp.uint8)
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

            return host_statuses, pathogen_genomes, host_immunities

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
        ) -> Tuple[xp.ndarray, xp.ndarray]:
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

            return host_statuses, pathogen_genomes

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
                ).astype(xp.uint8)
                pathogen_genomes[mutation_mask] ^= (
                    mutation_occurs << s
                ).astype(xp.uint8)

            return pathogen_genomes

        def update_simulation(
            host_statuses: xp.ndarray,
            pathogen_genomes: xp.ndarray,
            host_immunities: xp.ndarray,
        ) -> Tuple[xp.ndarray, xp.ndarray, xp.ndarray]:
            """Run one step of the simulation."""
            host_statuses, pathogen_genomes = transmit_infection(
                host_statuses, pathogen_genomes, host_immunities
            )
            pathogen_genomes = apply_mutations(pathogen_genomes, host_statuses)
            host_statuses, host_immunities = update_recoveries(
                host_statuses, host_immunities, pathogen_genomes
            )
            host_immunities = update_waning(host_immunities)

            return host_statuses, pathogen_genomes, host_immunities

        host_statuses, pathogen_genomes, host_immunities = initialize_pop()
        host_statuses, pathogen_genomes = infect_initial(
            host_statuses, pathogen_genomes
        )
        data_log: List[Dict[str, float]] = []

        for t in tqdm(range(N_STEPS)):
            (
                host_statuses,
                pathogen_genomes,
                host_immunities,
            ) = update_simulation(
                host_statuses, pathogen_genomes, host_immunities
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

        return pd.DataFrame(data_log).fillna(0).copy()

    return (simulate,)


@app.cell(hide_code=True)
def delimit_plotting(mo):
    mo.md("""
    ## Plotting Implementation
    """)
    return


@app.cell
def def_plotter(it, np, pathlib, pd, sns, tp):
    def render_timeseries_plots(
        df: pd.DataFrame,
        suptitle: str,
        teeplot_outattrs: dict,
        POP_SIZE: int,
    ) -> None:
        for what, row in it.product(
            ["Susc", "Strain"],
            ["Seed", None],
        ):
            data = (
                df.filter(regex=f"Step|Seed|{what}", axis=1)
                .melt(
                    id_vars=["Step", "Seed"],
                    var_name="Class",
                    value_name="Prevalence",
                )
                .astype(
                    {
                        "Step": int,
                        "Class": str,
                        "Prevalence": float,
                    },
                )
            )
            data["Ham. Wt."] = data["Class"].str.count("1|3|5|7|9")
            palette = dict(
                zip(
                    data["Class"].unique(),
                    sns.color_palette(
                        "colorblind", len(data["Class"].unique())
                    ),
                )
            )
            with tp.teed(
                sns.relplot,
                data=data,
                x="Step",
                y="Prevalence",
                hue="Class",
                row=row,
                alpha=0.8,
                dashes=False,
                errorbar=("pi", 100),
                err_kws=dict(alpha=0.1),
                estimator=np.median,
                facet_kws=dict(margin_titles=True),
                kind="line",
                palette=palette,
                teeplot_outattrs={
                    **teeplot_outattrs,
                    "what": what,
                },
                teeplot_subdir=pathlib.Path(__file__).stem,
            ) as g:
                for ax in g.axes.flat:
                    ax.grid(True, alpha=0.3)
                g.figure.set_size_inches(w=4, h=1.5)
                sns.move_legend(
                    g,
                    "center left",
                    bbox_to_anchor=(0.9, 0.5),
                    frameon=False,
                )

            with tp.teed(
                sns.relplot,
                data=data,
                x="Step",
                y="Prevalence",
                hue="Class",
                col="Ham. Wt.",
                row=row,
                alpha=0.8,
                dashes=False,
                errorbar=("pi", 100),
                err_kws=dict(alpha=0.1),
                estimator=np.median,
                facet_kws=dict(margin_titles=True),
                kind="line",
                palette=palette,
                teeplot_outattrs={
                    **teeplot_outattrs,
                    "what": what,
                },
                teeplot_subdir=pathlib.Path(__file__).stem,
            ) as g:
                g.map_dataframe(
                    sns.lineplot,
                    x="Step",
                    y="Prevalence",
                    hue="Class",
                    style="Seed",
                    alpha=0.7,
                    dashes=False,
                    errorbar=None,
                    legend=False,
                    linestyle=":",
                    linewidth=0.6,
                    palette=palette,
                )
                for ax in g.axes.flat:
                    ax.grid(True, alpha=0.3)
                if what == "Strain":
                    g.set(yscale="log")

                g.set(ylim=(1 / POP_SIZE, 1.1))
                g.figure.suptitle(suptitle)
                if row is not None:
                    g.figure.subplots_adjust(hspace=0.16, top=0.9)
                    g.figure.set_size_inches(w=5, h=5)
                else:
                    g.figure.subplots_adjust(top=0.7)
                    g.figure.set_size_inches(w=5, h=2)

                sns.move_legend(
                    g,
                    "center left",
                    bbox_to_anchor=(0.9, 0.5),
                    frameon=False,
                )

    return (render_timeseries_plots,)


@app.cell(hide_code=True)
def delimit_run(mo):
    mo.md("""
    ## Run Simulation and Render Plots across Condition Matrix
    """)
    return


@app.cell
def run_simulation(
    display_html,
    it,
    pd,
    render_timeseries_plots,
    simulate,
    tqdm,
):
    N_REP = 1
    N_STEPS = 600
    condition_matrix = it.product(
        [5e-5],  # MUTATION_RATE
        [250_000],  # POP_SIZE
        [2],  # N_SITES
    )
    for MUTATION_RATE, POP_SIZE, N_SITES in tqdm([*condition_matrix]):
        suptitle = (
            f"Pop Size: {POP_SIZE / 1_000_000}M., "
            f"Mutation Rate: {MUTATION_RATE}, "
            f"Num Sites: {N_SITES}"
        )
        display_html(f"<h2>{suptitle}</h2>", raw=True)
        dfs = [
            simulate(
                MUTATION_RATE=MUTATION_RATE,
                N_SITES=N_SITES,
                N_STEPS=N_STEPS,
                seed=rep + 2,
                POP_SIZE=POP_SIZE,
                # 0.35 = e^0.3 - 1 corrects for lack of continuous
                # compounding in the ABM relative to the ODE's rate of 0.3.
                CONTACT_RATE=0.35,
                RECOVERY_RATE=0.1,
                WANING_RATE=0.005,
                IMMUNE_STRENGTH=0.7,
                SEED_COUNT=2,
                IMMUNITY_FLOOR=0.05,
                IMMUNITY_CEILING=1.0,
            )
            for rep in range(N_REP)
        ]
        df = pd.concat(dfs)
        render_timeseries_plots(
            df=df,
            suptitle=suptitle,
            teeplot_outattrs={
                "MUTATION_RATE".lower(): MUTATION_RATE,
                "N_SITES".lower(): N_SITES,
                "POP_SIZE".lower(): POP_SIZE,
            },
            POP_SIZE=POP_SIZE,
        )
    return


@app.cell(hide_code=True)
def delimit_ode(mo):
    mo.md("""
    ## ODE Reference Trajectories
    """)
    return


@app.cell
def plot_ode(pd, sns):
    from matplotlib import pyplot as plt

    try:
        ode_df = pd.read_csv("https://osf.io/pvfst/download")
    except Exception as exc:
        print(f"skipping ODE reference plot: {exc}")
    else:
        ode_long = ode_df.melt(
            id_vars="TIME",
            var_name="variable",
            value_name="value",
        )
        ode_long["type"] = ode_long["variable"].str.slice(0, 1)
        sns.relplot(
            ode_long,
            x="TIME",
            y="value",
            hue="variable",
            kind="line",
            col="type",
            facet_kws=dict(sharey=False),
        )
        plt.xlim(0, 1100)
        plt.gcf().set_size_inches(7, 2.2)
        plt.tight_layout()
        sns.move_legend(plt.gcf(), "upper left", bbox_to_anchor=(1, 1))
    return


if __name__ == "__main__":
    app.run()
