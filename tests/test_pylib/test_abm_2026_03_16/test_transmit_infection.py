import numpy as np

from pylib.abm_2026_03_16 import transmit_infection


def test_no_transmission_without_infected():
    POP_SIZE = 50
    N_SITES = 2
    host_statuses = np.zeros(POP_SIZE, dtype=np.uint8)
    pathogen_genomes = np.zeros(POP_SIZE, dtype=np.uint8)
    host_immunities = np.zeros((POP_SIZE, 2 * N_SITES), dtype=np.float32)

    host_statuses, pathogen_genomes = transmit_infection(
        host_statuses,
        pathogen_genomes,
        host_immunities,
        POP_SIZE=POP_SIZE,
        N_SITES=N_SITES,
        CONTACT_RATE=0.3,
        IMMUNE_STRENGTH=0.9,
    )
    assert np.sum(host_statuses > 0) == 0


def test_transmission_possible_with_infected():
    np.random.seed(42)
    POP_SIZE = 1000
    N_SITES = 1
    host_statuses = np.zeros(POP_SIZE, dtype=np.uint8)
    host_statuses[:100] = 5  # 100 infected
    pathogen_genomes = np.zeros(POP_SIZE, dtype=np.uint8)
    host_immunities = np.zeros((POP_SIZE, 2 * N_SITES), dtype=np.float32)

    host_statuses, pathogen_genomes = transmit_infection(
        host_statuses,
        pathogen_genomes,
        host_immunities,
        POP_SIZE=POP_SIZE,
        N_SITES=N_SITES,
        CONTACT_RATE=1.0,
        IMMUNE_STRENGTH=0.0,
    )
    # Some new infections should occur
    newly_infected = np.sum(host_statuses == 1)
    assert newly_infected > 0


def test_transmission_preserves_genomes():
    np.random.seed(42)
    POP_SIZE = 100
    N_SITES = 2
    host_statuses = np.zeros(POP_SIZE, dtype=np.uint8)
    host_statuses[:20] = 5
    pathogen_genomes = np.zeros(POP_SIZE, dtype=np.uint8)
    pathogen_genomes[:20] = 3  # genome 11
    host_immunities = np.zeros((POP_SIZE, 2 * N_SITES), dtype=np.float32)

    host_statuses, pathogen_genomes = transmit_infection(
        host_statuses,
        pathogen_genomes,
        host_immunities,
        POP_SIZE=POP_SIZE,
        N_SITES=N_SITES,
        CONTACT_RATE=1.0,
        IMMUNE_STRENGTH=0.0,
    )
    newly_infected = host_statuses == 1
    if np.any(newly_infected):
        assert np.all(pathogen_genomes[newly_infected] == 3)
