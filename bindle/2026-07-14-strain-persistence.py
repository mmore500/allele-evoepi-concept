import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell
def import_std():
    import pathlib
    import uuid

    return pathlib, uuid


@app.cell
def import_pkg():
    import marimo as mo
    import numpy as np
    import pandas as pd
    from watermark import watermark

    from pylib.abm_2026_03_16 import initialize_pop, update_simulation

    return initialize_pop, mo, np, pd, update_simulation, watermark


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
    # Command-line args for a single community-assembly replicate. Set
    # them after a `--` separator, e.g.
    #   marimo export ipynb 2026-07-14-strain-persistence.py -o out.ipynb \
    #       -- --array-id 42 --seed 1 --n-steps 25000 --pop-size 1000000
    # or `marimo edit 2026-07-14-strain-persistence.py -- --array-id 42`.
    # Every arg has a default, so the notebook also runs with no args.
    #
    # This is a **3-site** model, so the strain count is fixed at
    # 2 ** 3 == 8 (genomes 0..7). Which of those 8 strains is present in
    # the initial community is read straight off the binary code of
    # ARRAY_ID in [0, 255]: bit j of ARRAY_ID set  <=>  strain j (genome
    # integer j) is seeded. Thus
    #   ARRAY_ID = 0   (0b00000000) -> no strains,
    #   ARRAY_ID = 1   (0b00000001) -> just strain 000 (genome 0),
    #   ARRAY_ID = 2   (0b00000010) -> just strain 001 (genome 1),
    #   ARRAY_ID = 3   (0b00000011) -> strains 000 and 001,
    #   ...
    #   ARRAY_ID = 255 (0b11111111) -> all eight strains.
    # Enumerating ARRAY_ID over [0, 255] therefore walks every subset of
    # the eight strains exactly once (256 communities x 8 strains = 2048
    # output rows once concatenated across the array).
    # Defaults are deliberately light so an unconfigured run (e.g. the
    # bindle CI that executes every notebook with no args) finishes
    # quickly; the launch script overrides POP_SIZE / N_STEPS with the
    # production values. ARRAY_ID defaults to 3 (strains 000 and 001) so
    # the unconfigured run still exercises the seed / observe / save path
    # rather than the empty ARRAY_ID=0 community.
    _args = mo.cli_args()
    ARRAY_ID = int(_args.get("array-id") or 3)
    SEED = int(_args.get("seed") or 1)
    POP_SIZE = int(_args.get("pop-size") or 100_000)
    N_STEPS = int(_args.get("n-steps") or 1_200)
    SEED_COUNT_PER_STRAIN = int(_args.get("seed-count-per-strain") or 10)
    ENGINE = str(_args.get("engine") or "numpy").lower()
    if ENGINE != "numpy":
        raise ValueError(f"engine must be 'numpy', got {ENGINE!r}")
    # Accepted for launch-script compatibility; this notebook renders no
    # figures, so it is a no-op here.
    SKIP_PLOTTING = bool(_args.get("skip-plotting") or False)
    if not 0 <= ARRAY_ID <= 255:
        raise ValueError(f"array-id must be in [0, 255], got {ARRAY_ID}")
    print(
        f"args: ARRAY_ID={ARRAY_ID} SEED={SEED} POP_SIZE={POP_SIZE} "
        f"N_STEPS={N_STEPS} SEED_COUNT_PER_STRAIN={SEED_COUNT_PER_STRAIN} "
        f"ENGINE={ENGINE} SKIP_PLOTTING={SKIP_PLOTTING}",
    )
    return ARRAY_ID, N_STEPS, POP_SIZE, SEED, SEED_COUNT_PER_STRAIN, ENGINE


@app.cell(hide_code=True)
def delimit_simulation(mo):
    mo.md(
        """
    ## Simulation Implementation

    This notebook runs a *single replicate* of the allele-based
    agent-based model (ABM) for one initial community, using the shared
    `pylib.abm_2026_03_16` step primitives (`initialize_pop`,
    `update_simulation`). It is fixed to the **3-site** model, so there
    are `2 ** 3 == 8` possible strains (genome integers `0..7`).

    The initial community is decoded from the `--array-id` command-line
    argument's binary representation: bit `j` of `array_id` selects
    strain `j`. The seeded strains are **mixed equally** ---
    `--seed-count-per-strain` hosts are seeded for *each* strain in the
    community, at distinct host indices --- and the run proceeds forward
    **without mutation** (`MUTATION_RATE = 0`), so the only genomes that
    ever circulate are the seeded ones. A strain therefore either
    persists to the end of the run or goes extinct; it is never
    (re)created by mutation.

    The single per-replicate deliverable is, for each of the eight
    strains, **the last simulation update at which that strain was
    observed** (i.e. the largest step index at which at least one host
    carried that genome):

    * `-1` --- the strain was never added to this community (its bit is
      clear in `array_id`).
    * `N_STEPS` --- the strain was added and never went extinct (it was
      still circulating at the final update).
    * `0 <= u < N_STEPS` --- the strain was added but went extinct; `u`
      is the last update at which it was still observed.

    Update index `0` is the post-seeding snapshot (the community as
    seeded, before any dynamics), and updates `1..N_STEPS` follow each
    call to `update_simulation`. A seeded strain is thus always observed
    at update `0`, keeping the `-1` "never added" sentinel disjoint from
    every added strain's last-observed update.

    Every output row is stamped with the simulation parameters as
    constant-valued columns, the `array_id`, and a `replicate_uid`
    generated with the standard-library `uuid` module, so rows are
    self-describing and uniquely identifiable.
    """
    )
    return


@app.cell
def run_replicate(
    ARRAY_ID,
    ENGINE,
    N_STEPS,
    POP_SIZE,
    SEED,
    SEED_COUNT_PER_STRAIN,
    initialize_pop,
    np,
    pathlib,
    pd,
    update_simulation,
    uuid,
):
    # Fixed model / epidemiological parameters. N_SITES is fixed at 3
    # (the 3-site model, 8 strains). MUTATION_RATE is fixed at 0 --- the
    # community is run forward without mutation. The remaining
    # epidemiological settings mirror the founder notebook
    # 2026-05-20-founder.py.
    N_SITES = 3
    MUTATION_RATE = 0.0
    CONTACT_RATE = 0.35
    RECOVERY_RATE = 0.1
    WANING_RATE = 0.01
    IMMUNE_STRENGTH = 0.7
    IMMUNITY_FLOOR = 0.05
    IMMUNITY_CEILING = 1.0
    WITHIN_HOST_B = 0.2
    WITHIN_HOST_T = 25.0

    n_strains = 1 << N_SITES  # 8

    xp = np
    np.random.seed(SEED)
    xp.random.seed(SEED)

    # Decode the initial community from array_id's binary code: bit j set
    # <=> strain j (genome integer j) is seeded.
    seeded_strains = [g for g in range(n_strains) if (ARRAY_ID >> g) & 1]

    replicate_uid = uuid.uuid4().hex
    print(
        f"=== strain-persistence run: array_id={ARRAY_ID} "
        f"seeded_strains={seeded_strains} seed={SEED} pop_size={POP_SIZE} "
        f"n_steps={N_STEPS} uid={replicate_uid} ===",
    )

    host_statuses, pathogen_genomes, host_immunities = initialize_pop(
        POP_SIZE, N_SITES, xp=xp
    )

    # Seed the community, mixing the selected strains equally:
    # SEED_COUNT_PER_STRAIN hosts per strain, at distinct host indices.
    if seeded_strains:
        n_seed = SEED_COUNT_PER_STRAIN * len(seeded_strains)
        if n_seed > POP_SIZE:
            raise ValueError(
                f"cannot seed {n_seed} hosts into POP_SIZE={POP_SIZE}",
            )
        seed_idx = xp.random.choice(POP_SIZE, size=n_seed, replace=False)
        for k, g in enumerate(seeded_strains):
            block = seed_idx[
                k * SEED_COUNT_PER_STRAIN : (k + 1) * SEED_COUNT_PER_STRAIN
            ]
            host_statuses[block] = 1
            pathogen_genomes[block] = g

    # last_observed[g]: largest update index at which genome g was seen;
    # -1 means "never added" (updated only for genomes actually present).
    last_observed = {g: -1 for g in range(n_strains)}

    def observe(update: int) -> None:
        inf_mask = host_statuses > 0
        if xp.any(inf_mask):
            present = xp.unique(pathogen_genomes[inf_mask])
            for g in present.tolist():
                last_observed[int(g)] = update

    observe(0)  # post-seeding snapshot (community as seeded)
    for t in range(1, N_STEPS + 1):
        host_statuses, pathogen_genomes, host_immunities = update_simulation(
            host_statuses,
            pathogen_genomes,
            host_immunities,
            POP_SIZE,
            N_SITES,
            CONTACT_RATE,
            IMMUNE_STRENGTH,
            RECOVERY_RATE,
            WANING_RATE,
            MUTATION_RATE,
            WITHIN_HOST_B,
            WITHIN_HOST_T,
            0,  # MUTATOR_HOSTS_N
            1.0,  # MUTATOR_HOSTS_MX
            0.0,  # MUTATION_THRESHOLD
            IMMUNITY_CEILING,
            IMMUNITY_FLOOR,
            xp=xp,
        )
        observe(t)

    # One row per strain (genome 0..7). strain_bits is the plain
    # N_SITES-bit binary label of the genome integer (strain 0 -> "000",
    # strain 1 -> "001", ...), matching the array_id bit <-> strain
    # mapping documented above.
    rows = []
    for g in range(n_strains):
        rows.append(
            {
                "array_id": ARRAY_ID,
                "strain": g,
                "strain_bits": format(g, f"0{N_SITES}b"),
                "last_observed_update": last_observed[g],
            }
        )
    strainlast_df = pd.DataFrame(rows)

    # Simulation parameters recorded as constant-valued columns so each
    # output row is self-describing, plus the replicate UUID.
    params = {
        "replicate_uid": replicate_uid,
        "array_id": ARRAY_ID,
        "seed": SEED,
        "n_sites": N_SITES,
        "n_strains": n_strains,
        "pop_size": POP_SIZE,
        "n_steps": N_STEPS,
        "engine": ENGINE,
        "mutation_rate": MUTATION_RATE,
        "contact_rate": CONTACT_RATE,
        "recovery_rate": RECOVERY_RATE,
        "waning_rate": WANING_RATE,
        "immune_strength": IMMUNE_STRENGTH,
        "seed_count_per_strain": SEED_COUNT_PER_STRAIN,
        "immunity_floor": IMMUNITY_FLOOR,
        "immunity_ceiling": IMMUNITY_CEILING,
    }
    strainlast_df = strainlast_df.assign(**params)

    nbname = pathlib.Path(__file__).stem
    out_dir = pathlib.Path("outdata") / nbname

    def _save(kind, df):
        rep_dir = out_dir / kind / replicate_uid
        rep_dir.mkdir(parents=True, exist_ok=True)
        rep_path = rep_dir / f"a={kind}+what={nbname}+ext=.pqt"
        df.to_parquet(rep_path, index=False)
        print(f"  wrote {kind} parquet ({len(df)} rows): {rep_path}")

    _save("strainlast", strainlast_df)

    print(strainlast_df.to_string())
    return (strainlast_df,)


if __name__ == "__main__":
    app.run()
