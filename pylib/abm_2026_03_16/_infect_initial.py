from typing import Tuple

from ._auxlib import xp as _xp


def infect_initial(
    host_statuses,
    pathogen_genomes,
    POP_SIZE: int,
    SEED_COUNT: int,
    xp=None,
) -> Tuple:
    """Seed the initial infection wave with the starting strain."""
    if xp is None:
        xp = _xp

    seeded_indices = xp.random.choice(POP_SIZE, size=SEED_COUNT, replace=False)
    host_statuses[seeded_indices] = 1
    pathogen_genomes[seeded_indices] = 0  # wildtype
    return host_statuses, pathogen_genomes
