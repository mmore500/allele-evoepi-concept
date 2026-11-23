import numpy as np

from pylib.abm_2026_03_16 import update_recoveries


def test_recovery_clears_status():
    POP_SIZE = 10
    N_SITES = 1
    RECOVERY_RATE = 0.1  # recover after 10 days

    host_statuses = np.zeros(POP_SIZE, dtype=np.uint8)
    host_statuses[0] = 15  # infected for 15 days, > 1/0.1
    host_immunities = np.zeros((POP_SIZE, 2 * N_SITES), dtype=np.float32)
    pathogen_genomes = np.zeros(POP_SIZE, dtype=np.uint8)

    host_statuses, host_immunities = update_recoveries(
        host_statuses,
        host_immunities,
        pathogen_genomes,
        N_SITES=N_SITES,
        RECOVERY_RATE=RECOVERY_RATE,
    )
    assert host_statuses[0] == 0  # recovered


def test_recovery_sets_immunity():
    POP_SIZE = 5
    N_SITES = 1
    RECOVERY_RATE = 0.1

    host_statuses = np.zeros(POP_SIZE, dtype=np.uint8)
    host_statuses[0] = 15
    host_immunities = np.zeros((POP_SIZE, 2 * N_SITES), dtype=np.float32)
    pathogen_genomes = np.zeros(POP_SIZE, dtype=np.uint8)
    pathogen_genomes[0] = 0  # wildtype (bit 0 = 0)

    host_statuses, host_immunities = update_recoveries(
        host_statuses,
        host_immunities,
        pathogen_genomes,
        N_SITES=N_SITES,
        RECOVERY_RATE=RECOVERY_RATE,
    )
    # Should have immunity to allele 0 of site 0
    assert host_immunities[0, 0] == 1.0


def test_active_infections_increment():
    POP_SIZE = 5
    N_SITES = 1
    RECOVERY_RATE = 0.1

    host_statuses = np.zeros(POP_SIZE, dtype=np.uint8)
    host_statuses[1] = 3  # actively infected, not recovered

    host_immunities = np.zeros((POP_SIZE, 2 * N_SITES), dtype=np.float32)
    pathogen_genomes = np.zeros(POP_SIZE, dtype=np.uint8)

    host_statuses, _ = update_recoveries(
        host_statuses,
        host_immunities,
        pathogen_genomes,
        N_SITES=N_SITES,
        RECOVERY_RATE=RECOVERY_RATE,
    )
    assert host_statuses[1] == 4  # incremented
