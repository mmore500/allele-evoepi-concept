import numpy as np

from pylib.abm_2026_03_16 import calc_mutation_probabilities


def test_scalar_mutation_rate_shape():
    n_infected = 5
    N_SITES = 2
    host_immunities = np.zeros((n_infected, 2 * N_SITES), dtype=np.float32)
    pathogen_genomes = np.zeros(n_infected, dtype=np.uint8)

    result = calc_mutation_probabilities(
        host_immunities,
        pathogen_genomes,
        N_SITES=N_SITES,
        MUTATION_RATE=1e-4,
        within_host_b=0.2,
        within_host_t=25.0,
    )
    assert result.shape == (n_infected, N_SITES)


def test_vector_mutation_rate_shape():
    n_infected = 5
    N_SITES = 3
    host_immunities = np.zeros((n_infected, 2 * N_SITES), dtype=np.float32)
    pathogen_genomes = np.zeros(n_infected, dtype=np.uint8)

    result = calc_mutation_probabilities(
        host_immunities,
        pathogen_genomes,
        N_SITES=N_SITES,
        MUTATION_RATE=[1e-4, 2e-4, 3e-4],
        within_host_b=0.2,
        within_host_t=25.0,
    )
    assert result.shape == (n_infected, N_SITES)


def test_scalar_zero_immunity_baseline():
    n_infected = 10
    N_SITES = 2
    host_immunities = np.zeros((n_infected, 2 * N_SITES), dtype=np.float32)
    pathogen_genomes = np.zeros(n_infected, dtype=np.uint8)

    result = calc_mutation_probabilities(
        host_immunities,
        pathogen_genomes,
        N_SITES=N_SITES,
        MUTATION_RATE=1e-4,
        within_host_b=0.2,
        within_host_t=25.0,
    )
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0)


def test_high_immunity_increases_mutation():
    N_SITES = 1
    # high immunity to current allele, none to opposite
    host_immunities_high = np.array([[1.0, 0.0]], dtype=np.float32)
    host_immunities_low = np.array([[0.0, 0.0]], dtype=np.float32)
    pathogen_genomes = np.array([0], dtype=np.uint8)

    result_high = calc_mutation_probabilities(
        host_immunities_high,
        pathogen_genomes,
        N_SITES=N_SITES,
        MUTATION_RATE=1e-4,
        within_host_b=0.2,
        within_host_t=25.0,
    )
    result_low = calc_mutation_probabilities(
        host_immunities_low,
        pathogen_genomes,
        N_SITES=N_SITES,
        MUTATION_RATE=1e-4,
        within_host_b=0.2,
        within_host_t=25.0,
    )
    assert result_high[0, 0] > result_low[0, 0]
