from ._auxlib import xp as _xp


def calc_mutation_probabilities(
    host_immunities,
    pathogen_genomes,
    N_SITES: int,
    MUTATION_RATE,
    within_host_b: float,
    within_host_t: float,
    xp=None,
):
    """Calculate per-site mutation probabilities for infected hosts.

    When MUTATION_RATE is scalar, uses within-host dynamics model.
    When MUTATION_RATE is a vector (per-site rates), broadcasts directly.
    """
    if xp is None:
        xp = _xp

    MUTATION_RATE = xp.asarray(MUTATION_RATE, dtype=xp.float32)

    if MUTATION_RATE.size == 1:
        return _calc_mutation_probabilities_scalar(
            host_immunities,
            pathogen_genomes,
            N_SITES,
            MUTATION_RATE,
            within_host_b,
            within_host_t,
            xp,
        )
    else:
        return _calc_mutation_probabilities_vector(
            host_immunities,
            N_SITES,
            MUTATION_RATE,
            xp,
        )


def _calc_mutation_probabilities_scalar(
    host_immunities,
    pathogen_genomes,
    N_SITES,
    MUTATION_RATE,
    within_host_b,
    within_host_t,
    xp,
):
    pathogen_bits = (pathogen_genomes[:, None] >> xp.arange(N_SITES)) & 1

    imm_reshaped = xp.reshape(host_immunities, (-1, N_SITES, 2))

    idx_curr = pathogen_bits[:, :, None]
    idx_opp = 1 - idx_curr

    imm_curr = xp.take_along_axis(imm_reshaped, idx_curr, axis=2).squeeze(
        axis=2
    )
    imm_opp = xp.take_along_axis(imm_reshaped, idx_opp, axis=2).squeeze(axis=2)

    host_immunity_deltas = imm_curr - imm_opp

    b_values = 1.0 + within_host_b * host_immunity_deltas
    b_values = xp.where(xp.abs(b_values - 1.0) < 1e-7, 1.000001, b_values)

    return (MUTATION_RATE / (b_values - 1.0)) * (
        xp.exp((b_values - 1.0) * within_host_t) - 1.0
    )


def _calc_mutation_probabilities_vector(
    host_immunities,
    N_SITES,
    MUTATION_RATE,
    xp,
):
    n_infected = host_immunities.shape[0]
    return xp.ones((n_infected, 1), dtype=MUTATION_RATE.dtype) * MUTATION_RATE
