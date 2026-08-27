from typing import Tuple

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
    xp=None,
) -> Tuple:
    """Vectorized transmission based on allele-specific susceptibility."""
    if xp is None:
        xp = _xp

    contacts = xp.random.randint(
        low=0, high=POP_SIZE, size=POP_SIZE, dtype=xp.uint32
    )
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
    )

    new_infections = xp.random.rand(POP_SIZE) < inf_probs
    host_statuses[new_infections] = 1
    pathogen_genomes[new_infections] = pathogen_genomes[contacts][
        new_infections
    ]

    return host_statuses, pathogen_genomes
