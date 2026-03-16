import functools
import random
from typing import Dict, List, Sequence, Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ._auxlib import xp as _xp
from ._infect_initial import infect_initial
from ._initialize_pop import initialize_pop
from ._update_simulation import update_simulation


@functools.lru_cache(maxsize=None)
def simulate(
    N_SITES: int = 2,
    POP_SIZE: int = 1_000_000,
    CONTACT_RATE: float = 0.3,
    RECOVERY_RATE: float = 0.1,
    MUTATION_RATE: Union[float, Sequence[float]] = 1e-4,
    WANING_RATE: float = 0.02,
    IMMUNE_STRENGTH: float = 0.9,
    N_STEPS: int = 1_000,
    SEED_COUNT: int = 10,
    within_host_b: float = 0.2,
    within_host_t: float = 25.0,
    seed: int = 1,
    MUTATOR_HOSTS_N: int = 0,
    MUTATOR_HOSTS_MX: float = 1.0,
    MUTATION_THRESHOLD: float = 0.0,
    IMMUNITY_CEILING: float = 1.0,
    IMMUNITY_FLOOR: float = 0.0,
    xp=None,
) -> pd.DataFrame:
    if xp is None:
        xp = _xp

    random.seed(seed)
    np.random.seed(seed)
    xp.random.seed(seed)

    if N_SITES > 8:
        raise NotImplementedError(
            "current data types support only up to 8 sites",
        )

    host_statuses, pathogen_genomes, host_immunities = initialize_pop(
        POP_SIZE, N_SITES, xp=xp
    )
    host_statuses, pathogen_genomes = infect_initial(
        host_statuses, pathogen_genomes, POP_SIZE, SEED_COUNT, xp=xp
    )
    data_log: List[Dict[str, float]] = []

    for t in tqdm(range(N_STEPS)):
        host_statuses, pathogen_genomes, host_immunities = update_simulation(
            host_statuses,
            pathogen_genomes,
            host_immunities,
            POP_SIZE,
            N_SITES,
            CONTACT_RATE,
            IMMUNE_STRENGTH,
            RECOVERY_RATE,
            WANING_RATE,
            MUTATION_RATE,
            within_host_b,
            within_host_t,
            MUTATOR_HOSTS_N,
            MUTATOR_HOSTS_MX,
            MUTATION_THRESHOLD,
            IMMUNITY_CEILING,
            IMMUNITY_FLOOR,
            xp=xp,
        )

        # 1. Strain Prevalence
        inf_mask = host_statuses > 0
        counts_dict: Dict[str, float] = {}

        if xp.any(inf_mask):
            unique_g, counts = xp.unique(
                pathogen_genomes[inf_mask], return_counts=True
            )

            for g, count in zip(unique_g.tolist(), counts.tolist()):
                fmt = f"0{N_SITES}b"
                strain_name = format(int(g), fmt)[::-1]
                counts_dict[f"Strain_{strain_name}"] = float(count) / POP_SIZE

        # 2. Host Susceptibility per Allele
        avg_susc = xp.mean(1.0 - (IMMUNE_STRENGTH * host_immunities), axis=0)

        immunity_dict = {}
        for i in range(2 * N_SITES):
            site = i // 2
            bit = i % 2
            immunity_dict[f"Susc_S{site}_B{bit}"] = float(avg_susc[i])

        # 3. Aggregate Metrics
        log_entry = {
            "Step": t,
            "Seed": seed,
            "Total_Infected": float(xp.sum(inf_mask)) / POP_SIZE,
        }
        log_entry.update(counts_dict)
        log_entry.update(immunity_dict)
        data_log.append(log_entry)

    return pd.DataFrame(data_log).fillna(0).copy()
