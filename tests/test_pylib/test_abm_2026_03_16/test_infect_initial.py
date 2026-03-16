import numpy as np

from pylib.abm_2026_03_16 import infect_initial, initialize_pop


def test_infect_initial_count():
    host_statuses, pathogen_genomes, _ = initialize_pop(
        POP_SIZE=100, N_SITES=2
    )
    host_statuses, pathogen_genomes = infect_initial(
        host_statuses, pathogen_genomes, POP_SIZE=100, SEED_COUNT=5
    )
    assert np.sum(host_statuses > 0) == 5


def test_infect_initial_wildtype():
    host_statuses, pathogen_genomes, _ = initialize_pop(
        POP_SIZE=100, N_SITES=2
    )
    host_statuses, pathogen_genomes = infect_initial(
        host_statuses, pathogen_genomes, POP_SIZE=100, SEED_COUNT=10
    )
    infected_mask = host_statuses > 0
    assert np.all(pathogen_genomes[infected_mask] == 0)


def test_infect_initial_status_value():
    host_statuses, pathogen_genomes, _ = initialize_pop(POP_SIZE=50, N_SITES=1)
    host_statuses, pathogen_genomes = infect_initial(
        host_statuses, pathogen_genomes, POP_SIZE=50, SEED_COUNT=3
    )
    assert np.all(host_statuses[host_statuses > 0] == 1)
