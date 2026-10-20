import numpy as np

from pylib.abm_2026_03_16 import calc_infection_probabilities


def test_no_immunity_full_susceptibility():
    POP_SIZE = 10
    N_SITES = 2
    host_immunities = np.zeros((POP_SIZE, 2 * N_SITES), dtype=np.float32)
    pathogen_genomes = np.zeros(POP_SIZE, dtype=np.uint8)

    probs = calc_infection_probabilities(
        host_immunities,
        pathogen_genomes,
        POP_SIZE=POP_SIZE,
        N_SITES=N_SITES,
        IMMUNE_STRENGTH=0.9,
    )
    assert probs.shape == (POP_SIZE,)
    np.testing.assert_allclose(probs, 1.0)


def test_full_immunity_reduces_susceptibility():
    POP_SIZE = 5
    N_SITES = 1
    host_immunities = np.ones((POP_SIZE, 2 * N_SITES), dtype=np.float32)
    pathogen_genomes = np.zeros(POP_SIZE, dtype=np.uint8)

    probs = calc_infection_probabilities(
        host_immunities,
        pathogen_genomes,
        POP_SIZE=POP_SIZE,
        N_SITES=N_SITES,
        IMMUNE_STRENGTH=0.9,
    )
    expected = 0.1  # 1 - 0.9 * 1.0
    np.testing.assert_allclose(probs, expected, atol=1e-6)


def test_output_shape():
    POP_SIZE = 20
    N_SITES = 3
    host_immunities = np.zeros((POP_SIZE, 2 * N_SITES), dtype=np.float32)
    pathogen_genomes = np.zeros(POP_SIZE, dtype=np.uint8)

    probs = calc_infection_probabilities(
        host_immunities,
        pathogen_genomes,
        POP_SIZE=POP_SIZE,
        N_SITES=N_SITES,
        IMMUNE_STRENGTH=0.5,
    )
    assert probs.shape == (POP_SIZE,)


def test_probabilities_bounded():
    POP_SIZE = 10
    N_SITES = 2
    host_immunities = np.random.rand(POP_SIZE, 2 * N_SITES).astype(np.float32)
    pathogen_genomes = np.random.randint(0, 4, size=POP_SIZE, dtype=np.uint8)

    probs = calc_infection_probabilities(
        host_immunities,
        pathogen_genomes,
        POP_SIZE=POP_SIZE,
        N_SITES=N_SITES,
        IMMUNE_STRENGTH=0.9,
    )
    assert np.all(probs >= 0)
    assert np.all(probs <= 1)
