import numpy as np

from pylib.abm_2026_03_16 import update_waning


def test_waning_decreases_immunity():
    host_immunities = np.full((5, 4), 1.0, dtype=np.float32)
    result = update_waning(host_immunities.copy(), WANING_RATE=0.02)
    np.testing.assert_allclose(result, 0.98, atol=1e-6)


def test_waning_zero_stays_zero():
    host_immunities = np.zeros((5, 4), dtype=np.float32)
    result = update_waning(host_immunities, WANING_RATE=0.02)
    np.testing.assert_allclose(result, 0.0)


def test_waning_respects_floor():
    host_immunities = np.full((3, 2), 0.01, dtype=np.float32)
    result = update_waning(
        host_immunities,
        WANING_RATE=0.5,
        IMMUNITY_FLOOR=0.1,
    )
    assert np.all(result[result > 0] >= 0.1)


def test_waning_respects_ceiling():
    host_immunities = np.full((3, 2), 0.9, dtype=np.float32)
    result = update_waning(
        host_immunities,
        WANING_RATE=0.01,
        IMMUNITY_CEILING=0.5,
    )
    assert np.all(result[result > 0] <= 0.5)
