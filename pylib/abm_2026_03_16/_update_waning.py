from ._auxlib import xp as _xp


def update_waning(
    host_immunities,
    WANING_RATE: float,
    IMMUNITY_FLOOR: float = 0.0,
    IMMUNITY_CEILING: float = 1.0,
    xp=None,
):
    """Decay immunity levels over time."""
    if xp is None:
        xp = _xp

    host_immunities *= 1.0 - WANING_RATE
    host_immunities[host_immunities > 0] = xp.clip(
        host_immunities[host_immunities > 0],
        IMMUNITY_FLOOR,
        IMMUNITY_CEILING,
    )
    return host_immunities
