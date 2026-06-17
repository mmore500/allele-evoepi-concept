from typing import Sequence, Tuple, Union

from ._auxlib import xp as _xp
from ._calc_infection_probabilities import calc_infection_probabilities


def transmit_infection(
    host_statuses,
    pathogen_genomes,
    host_immunities,
    POP_SIZE: int,
    N_SITES: int,
    CONTACT_RATE: float,
    IMMUNE_STRENGTH: float,
    TRANSMISSIBILITY: Union[float, Sequence[float]] = 1.0,
    xp=None,
) -> Tuple:
    """Vectorized transmission based on allele-specific susceptibility.

    ``TRANSMISSIBILITY`` is a multiplicative factor applied to the
    transmitting strain's infection probability. It may be a scalar (applied
    to all strains) or a per-strain vector of length ``2 ** N_SITES`` indexed
    by genome integer.
    """
    if xp is None:
        xp = _xp

    contacts = xp.random.randint(
        low=0, high=POP_SIZE, size=POP_SIZE, dtype=xp.uint32
    )

    TRANSMISSIBILITY = xp.asarray(TRANSMISSIBILITY, dtype=xp.float32)
    if TRANSMISSIBILITY.size == 1:
        strain_transmissibility = TRANSMISSIBILITY
    else:
        # index per-strain factor by the transmitting contact's genome
        strain_transmissibility = TRANSMISSIBILITY[pathogen_genomes[contacts]]

    inf_probs = (
        calc_infection_probabilities(
            host_immunities,
            pathogen_genomes[contacts],
            POP_SIZE,
            N_SITES,
            IMMUNE_STRENGTH,
            xp=xp,
        )
        * (host_statuses == 0)
        * (host_statuses[contacts] > 0)
        * CONTACT_RATE
        * strain_transmissibility
    )

    new_infections = xp.random.rand(POP_SIZE) < inf_probs
    host_statuses[new_infections] = 1
    pathogen_genomes[new_infections] = pathogen_genomes[contacts][
        new_infections
    ]

    return host_statuses, pathogen_genomes
