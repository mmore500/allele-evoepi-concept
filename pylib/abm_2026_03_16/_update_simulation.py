from typing import Tuple

from ._apply_mutations import apply_mutations
from ._auxlib import xp as _xp
from ._transmit_infection import transmit_infection
from ._update_recoveries import update_recoveries
from ._update_waning import update_waning


def update_simulation(
    host_statuses,
    pathogen_genomes,
    host_immunities,
    POP_SIZE: int,
    N_SITES: int,
    CONTACT_RATE: float,
    IMMUNE_STRENGTH: float,
    RECOVERY_RATE: float,
    WANING_RATE: float,
    MUTATION_RATE,
    within_host_b: float,
    within_host_t: float,
    MUTATOR_HOSTS_N: int = 0,
    MUTATOR_HOSTS_MX: float = 1.0,
    MUTATION_THRESHOLD: float = 0.0,
    IMMUNITY_CEILING: float = 1.0,
    IMMUNITY_FLOOR: float = 0.0,
    xp=None,
) -> Tuple:
    """Run one step of the simulation."""
    if xp is None:
        xp = _xp

    host_statuses, pathogen_genomes = transmit_infection(
        host_statuses,
        pathogen_genomes,
        host_immunities,
        POP_SIZE,
        N_SITES,
        CONTACT_RATE,
        IMMUNE_STRENGTH,
        xp=xp,
    )
    pathogen_genomes = apply_mutations(
        pathogen_genomes,
        host_statuses,
        host_immunities,
        N_SITES,
        MUTATION_RATE,
        within_host_b,
        within_host_t,
        MUTATOR_HOSTS_N,
        MUTATOR_HOSTS_MX,
        MUTATION_THRESHOLD,
        xp=xp,
    )
    host_statuses, host_immunities = update_recoveries(
        host_statuses,
        host_immunities,
        pathogen_genomes,
        N_SITES,
        RECOVERY_RATE,
        xp=xp,
    )
    host_immunities = update_waning(
        host_immunities,
        WANING_RATE,
        IMMUNITY_FLOOR,
        IMMUNITY_CEILING,
        xp=xp,
    )

    return host_statuses, pathogen_genomes, host_immunities
