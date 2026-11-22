import numpy as np

from pylib.abm_2026_03_16 import initialize_pop


def test_initialize_pop_shapes():
    host_statuses, pathogen_genomes, host_immunities = initialize_pop(
        POP_SIZE=100, N_SITES=2
    )
    assert host_statuses.shape == (100,)
    assert pathogen_genomes.shape == (100,)
    assert host_immunities.shape == (100, 4)


def test_initialize_pop_dtypes():
    host_statuses, pathogen_genomes, host_immunities = initialize_pop(
        POP_SIZE=50, N_SITES=3
    )
    assert host_statuses.dtype == np.uint8
    assert pathogen_genomes.dtype == np.uint8
    assert host_immunities.dtype == np.float32


def test_initialize_pop_zeros():
    host_statuses, pathogen_genomes, host_immunities = initialize_pop(
        POP_SIZE=10, N_SITES=1
    )
    assert np.all(host_statuses == 0)
    assert np.all(pathogen_genomes == 0)
    assert np.all(host_immunities == 0.0)


def test_initialize_pop_nsites():
    for n_sites in range(1, 9):
        _, _, host_immunities = initialize_pop(POP_SIZE=5, N_SITES=n_sites)
        assert host_immunities.shape == (5, 2 * n_sites)
