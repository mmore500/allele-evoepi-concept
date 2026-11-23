from typing import Tuple

from ._auxlib import xp as _xp


def initialize_pop(
    POP_SIZE: int,
    N_SITES: int,
    xp=None,
) -> Tuple:
    """Initialize population statuses, genomes, and immune history."""
    if xp is None:
        xp = _xp

    pathogen_genomes = xp.zeros(shape=POP_SIZE, dtype=xp.uint8)
    host_immunities = xp.full(
        shape=(POP_SIZE, 2 * N_SITES),
        fill_value=0.0,
        dtype=xp.float32,
    )
    host_statuses = xp.full(shape=POP_SIZE, fill_value=0, dtype=xp.uint8)

    return host_statuses, pathogen_genomes, host_immunities
