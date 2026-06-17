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


def test_scalar_transmissibility_zero_blocks_transmission():
    np.random.seed(42)
    POP_SIZE = 1000
    N_SITES = 1
    host_statuses = np.zeros(POP_SIZE, dtype=np.uint8)
    host_statuses[:100] = 5
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
        TRANSMISSIBILITY=0.0,
    )
    assert np.sum(host_statuses == 1) == 0


def test_scalar_transmissibility_default_matches_one():
    POP_SIZE = 1000
    N_SITES = 1
    args = dict(
        POP_SIZE=POP_SIZE,
        N_SITES=N_SITES,
        CONTACT_RATE=1.0,
        IMMUNE_STRENGTH=0.0,
    )

    def setup():
        host_statuses = np.zeros(POP_SIZE, dtype=np.uint8)
        host_statuses[:100] = 5
        pathogen_genomes = np.zeros(POP_SIZE, dtype=np.uint8)
        host_immunities = np.zeros((POP_SIZE, 2 * N_SITES), dtype=np.float32)
        return host_statuses, pathogen_genomes, host_immunities

    np.random.seed(42)
    default_statuses, _ = transmit_infection(*setup(), **args)
    np.random.seed(42)
    explicit_statuses, _ = transmit_infection(
        *setup(), TRANSMISSIBILITY=1.0, **args
    )
    assert np.array_equal(default_statuses, explicit_statuses)


def test_per_strain_transmissibility_vector_blocks_one_strain():
    np.random.seed(42)
    POP_SIZE = 2000
    N_SITES = 1
    host_statuses = np.zeros(POP_SIZE, dtype=np.uint8)
    # genome 0 infects first half, genome 1 infects second half
    host_statuses[:500] = 5
    host_statuses[1000:1500] = 5
    pathogen_genomes = np.zeros(POP_SIZE, dtype=np.uint8)
    pathogen_genomes[1000:1500] = 1
    host_immunities = np.zeros((POP_SIZE, 2 * N_SITES), dtype=np.float32)

    # strain 0 cannot transmit, strain 1 can
    host_statuses, pathogen_genomes = transmit_infection(
        host_statuses,
        pathogen_genomes,
        host_immunities,
        POP_SIZE=POP_SIZE,
        N_SITES=N_SITES,
        CONTACT_RATE=1.0,
        IMMUNE_STRENGTH=0.0,
        TRANSMISSIBILITY=(0.0, 1.0),
    )
    newly_infected = host_statuses == 1
    assert np.any(newly_infected)
    # all new infections must be the transmissible strain (genome 1)
    assert np.all(pathogen_genomes[newly_infected] == 1)
