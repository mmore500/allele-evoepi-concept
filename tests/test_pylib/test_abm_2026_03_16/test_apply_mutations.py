import numpy as np

from pylib.abm_2026_03_16 import apply_mutations


def test_no_mutations_when_no_newly_infected():
    POP_SIZE = 10
    N_SITES = 2
    host_statuses = np.zeros(POP_SIZE, dtype=np.uint8)
    host_statuses[0] = 5  # infected but not newly
    pathogen_genomes = np.zeros(POP_SIZE, dtype=np.uint8)
    host_immunities = np.zeros((POP_SIZE, 2 * N_SITES), dtype=np.float32)

    result = apply_mutations(
        pathogen_genomes.copy(),
        host_statuses,
        host_immunities,
        N_SITES=N_SITES,
        MUTATION_RATE=1.0,
        within_host_b=0.2,
        within_host_t=25.0,
    )
    np.testing.assert_array_equal(result, pathogen_genomes)


def test_mutations_can_change_genomes():
    np.random.seed(42)
    POP_SIZE = 100
    N_SITES = 2
    host_statuses = np.ones(POP_SIZE, dtype=np.uint8)  # all newly infected
    pathogen_genomes = np.zeros(POP_SIZE, dtype=np.uint8)
    host_immunities = np.zeros((POP_SIZE, 2 * N_SITES), dtype=np.float32)

    result = apply_mutations(
        pathogen_genomes.copy(),
        host_statuses,
        host_immunities,
        N_SITES=N_SITES,
        MUTATION_RATE=0.99,
        within_host_b=0.0,
        within_host_t=1.0,
    )
    # With high mutation rate and b=0, some mutations should occur
    assert np.any(result != 0)


def test_mutation_threshold():
    np.random.seed(42)
    POP_SIZE = 50
    N_SITES = 1
    host_statuses = np.ones(POP_SIZE, dtype=np.uint8)
    pathogen_genomes = np.zeros(POP_SIZE, dtype=np.uint8)
    host_immunities = np.zeros((POP_SIZE, 2 * N_SITES), dtype=np.float32)

    result = apply_mutations(
        pathogen_genomes.copy(),
        host_statuses,
        host_immunities,
        N_SITES=N_SITES,
        MUTATION_RATE=1e-10,
        within_host_b=0.2,
        within_host_t=25.0,
        MUTATION_THRESHOLD=1.0,
    )
    np.testing.assert_array_equal(result, pathogen_genomes)
