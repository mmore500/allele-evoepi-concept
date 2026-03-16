from typing import Tuple

import numpy as np

from ._auxlib import xp as _xp


def update_recoveries(
    host_statuses,
    host_immunities,
    pathogen_genomes,
    N_SITES: int,
    RECOVERY_RATE: float,
    xp=None,
) -> Tuple:
    """Recover infected individuals and update allele immunity."""
    if xp is None:
        xp = _xp

    recovered_mask = (host_statuses > 1 / RECOVERY_RATE).astype(bool)

    pathogen_bits = (pathogen_genomes[:, None] >> xp.arange(N_SITES)) & 1
    pathogen_alleles = (pathogen_bits[:, :, None] == xp.array([0, 1])).reshape(
        -1, 2 * N_SITES
    )

    assert np.all(pathogen_alleles[recovered_mask].sum(axis=1) == N_SITES)

    host_immunities[
        pathogen_alleles.astype(bool) & recovered_mask[:, None]
    ] = 1.0

    host_statuses += (host_statuses > 0).astype(xp.uint8)
    host_statuses[recovered_mask] = 0

    return host_statuses, host_immunities
