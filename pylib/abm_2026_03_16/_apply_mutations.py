from ._auxlib import xp as _xp
from ._calc_mutation_probabilities import calc_mutation_probabilities


def apply_mutations(
    pathogen_genomes,
    host_statuses,
    host_immunities,
    N_SITES: int,
    MUTATION_RATE,
    within_host_b: float,
    within_host_t: float,
    MUTATOR_HOSTS_N: int = 0,
    MUTATOR_HOSTS_MX: float = 1.0,
    MUTATION_THRESHOLD: float = 0.0,
    xp=None,
):
    """Apply mutations to newly infected individuals."""
    if xp is None:
        xp = _xp

    mutation_mask = (host_statuses == 1).astype(bool)

    mprobs = calc_mutation_probabilities(
        host_immunities[mutation_mask],
        pathogen_genomes[mutation_mask],
        N_SITES,
        MUTATION_RATE,
        within_host_b,
        within_host_t,
        xp=xp,
    )
    mutator_n = host_statuses[:MUTATOR_HOSTS_N].sum()
    mprobs[:mutator_n] = xp.minimum(mprobs[:mutator_n] * MUTATOR_HOSTS_MX, 1.0)

    mprobs[mprobs < MUTATION_THRESHOLD] = 0.0

    for s in range(N_SITES):
        mutation_occurs = (
            xp.random.rand(mprobs.shape[0]) < mprobs[:, s]
        ).astype(xp.uint8)
        pathogen_genomes[mutation_mask] ^= (mutation_occurs << s).astype(
            xp.uint8
        )

    return pathogen_genomes
