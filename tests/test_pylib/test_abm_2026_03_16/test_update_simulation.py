import numpy as np

from pylib.abm_2026_03_16 import (
    infect_initial,
    initialize_pop,
    update_simulation,
)


def test_update_simulation_runs():
    POP_SIZE = 100
    N_SITES = 2

    host_statuses, pathogen_genomes, host_immunities = initialize_pop(
        POP_SIZE, N_SITES
    )
    host_statuses, pathogen_genomes = infect_initial(
        host_statuses, pathogen_genomes, POP_SIZE, SEED_COUNT=5
    )

    host_statuses, pathogen_genomes, host_immunities = update_simulation(
        host_statuses,
        pathogen_genomes,
        host_immunities,
        POP_SIZE=POP_SIZE,
        N_SITES=N_SITES,
        CONTACT_RATE=0.3,
        IMMUNE_STRENGTH=0.9,
        RECOVERY_RATE=0.1,
        WANING_RATE=0.02,
        MUTATION_RATE=1e-4,
        within_host_b=0.2,
        within_host_t=25.0,
    )

    assert host_statuses.shape == (POP_SIZE,)
    assert pathogen_genomes.shape == (POP_SIZE,)
    assert host_immunities.shape == (POP_SIZE, 2 * N_SITES)


def test_update_simulation_multiple_steps():
    np.random.seed(1)
    POP_SIZE = 200
    N_SITES = 1

    host_statuses, pathogen_genomes, host_immunities = initialize_pop(
        POP_SIZE, N_SITES
    )
    host_statuses, pathogen_genomes = infect_initial(
        host_statuses, pathogen_genomes, POP_SIZE, SEED_COUNT=10
    )

    for _ in range(10):
        host_statuses, pathogen_genomes, host_immunities = update_simulation(
            host_statuses,
            pathogen_genomes,
            host_immunities,
            POP_SIZE=POP_SIZE,
            N_SITES=N_SITES,
            CONTACT_RATE=0.3,
            IMMUNE_STRENGTH=0.9,
            RECOVERY_RATE=0.1,
            WANING_RATE=0.02,
            MUTATION_RATE=1e-4,
            within_host_b=0.2,
            within_host_t=25.0,
        )

    assert host_statuses.dtype == np.uint8
    assert pathogen_genomes.dtype == np.uint8
