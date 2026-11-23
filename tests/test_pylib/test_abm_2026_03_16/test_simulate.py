import pandas as pd

from pylib.abm_2026_03_16 import simulate


def test_simulate_returns_dataframe():
    result = simulate(
        N_SITES=1,
        POP_SIZE=100,
        N_STEPS=5,
        SEED_COUNT=5,
        seed=42,
    )
    assert isinstance(result, pd.DataFrame)


def test_simulate_has_expected_columns():
    result = simulate(
        N_SITES=1,
        POP_SIZE=100,
        N_STEPS=5,
        SEED_COUNT=5,
        seed=43,
    )
    assert "Step" in result.columns
    assert "Seed" in result.columns
    assert "Total_Infected" in result.columns


def test_simulate_step_count():
    n_steps = 10
    result = simulate(
        N_SITES=1,
        POP_SIZE=100,
        N_STEPS=n_steps,
        SEED_COUNT=5,
        seed=44,
    )
    assert len(result) == n_steps


def test_simulate_nsites_too_large():
    import pytest

    with pytest.raises(NotImplementedError):
        simulate(
            N_SITES=9,
            POP_SIZE=100,
            N_STEPS=1,
            seed=45,
        )


def test_simulate_deterministic():
    result1 = simulate(
        N_SITES=1,
        POP_SIZE=100,
        N_STEPS=5,
        SEED_COUNT=5,
        seed=46,
    )
    # lru_cache means same args return same object
    result2 = simulate(
        N_SITES=1,
        POP_SIZE=100,
        N_STEPS=5,
        SEED_COUNT=5,
        seed=46,
    )
    pd.testing.assert_frame_equal(result1, result2)


def test_simulate_susceptibility_columns():
    result = simulate(
        N_SITES=2,
        POP_SIZE=100,
        N_STEPS=3,
        SEED_COUNT=5,
        seed=47,
    )
    for site in range(2):
        for bit in range(2):
            col = f"Susc_S{site}_B{bit}"
            assert col in result.columns
