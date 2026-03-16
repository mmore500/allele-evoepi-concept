from ._auxlib import xp as _xp


def calc_infection_probabilities(
    host_immunities,
    pathogen_genomes,
    POP_SIZE: int,
    N_SITES: int,
    IMMUNE_STRENGTH: float,
    xp=None,
):
    """Calculate allele-specific susceptibility for each host."""
    if xp is None:
        xp = _xp

    host_susceptibilities = xp.reshape(
        1.0 - (IMMUNE_STRENGTH * host_immunities),
        (POP_SIZE, 2 * N_SITES),
    )

    pathogen_bits = (pathogen_genomes[:, None] >> xp.arange(N_SITES)) & 1
    pathogen_alleles = (pathogen_bits[:, :, None] == xp.array([0, 1])).reshape(
        -1, 2 * N_SITES
    )

    active_susc = xp.where(pathogen_alleles, host_susceptibilities, 1.0)
    res = xp.prod(active_susc, axis=1)
    assert res.shape == (POP_SIZE,)
    return res
