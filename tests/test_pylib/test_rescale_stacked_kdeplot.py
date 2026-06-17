"""Tests for pylib.rescale_stacked_kdeplot."""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import seaborn as sns  # noqa: E402

from pylib import rescale_stacked_kdeplot  # noqa: E402
from pylib._rescale_stacked_kdeplot import (  # noqa: E402
    _extract_band,
    _stacked_bands,
)


def _make_stacked_kde(orient="x"):
    tips = sns.load_dataset("tips")
    fig, ax = plt.subplots()
    if orient in ("x", "v"):
        sns.kdeplot(
            data=tips,
            x="total_bill",
            hue="time",
            multiple="stack",
            ax=ax,
        )
    else:
        sns.kdeplot(
            data=tips,
            y="total_bill",
            hue="time",
            multiple="stack",
            ax=ax,
        )
    return fig, ax


@pytest.mark.parametrize("orient", ["x", "v"])
def test_vertical_orientations(orient):
    fig, ax = _make_stacked_kde("x")
    out = rescale_stacked_kdeplot(ax, orient=orient)
    assert out is ax
    assert ax.get_yscale() == "log"
    assert ax.get_xscale() == "linear"
    plt.close(fig)


@pytest.mark.parametrize("orient", ["y", "h"])
def test_horizontal_orientations(orient):
    fig, ax = _make_stacked_kde("y")
    out = rescale_stacked_kdeplot(ax, orient=orient)
    assert out is ax
    assert ax.get_xscale() == "log"
    assert ax.get_yscale() == "linear"
    plt.close(fig)


def test_orient_positional():
    fig, ax = _make_stacked_kde("x")
    rescale_stacked_kdeplot(ax, "x")
    assert ax.get_yscale() == "log"
    plt.close(fig)


def test_invalid_orient_raises():
    fig, ax = _make_stacked_kde()
    with pytest.raises(ValueError):
        rescale_stacked_kdeplot(ax, orient="bogus")
    plt.close(fig)


def test_invalid_vmin_raises():
    fig, ax = _make_stacked_kde()
    with pytest.raises(ValueError):
        rescale_stacked_kdeplot(ax, orient="x", vmin=0)
    plt.close(fig)


def test_no_bands_raises():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    with pytest.raises(ValueError):
        rescale_stacked_kdeplot(ax, orient="x")
    plt.close(fig)


def test_unsupported_scale_raises():
    fig, ax = _make_stacked_kde()
    with pytest.raises(NotImplementedError):
        rescale_stacked_kdeplot(ax, orient="x", scale="symlog")
    plt.close(fig)


def test_default_scale_is_log():
    fig, ax = _make_stacked_kde()
    rescale_stacked_kdeplot(ax, "x")
    assert ax.get_yscale() == "log"
    plt.close(fig)


def test_vmin_overrides_default():
    fig, ax = _make_stacked_kde()
    rescale_stacked_kdeplot(ax, orient="x", vmin=1e-3)
    assert ax.get_ylim()[0] == pytest.approx(1e-3)
    plt.close(fig)


def test_band_count_unchanged():
    fig, ax = _make_stacked_kde("x")
    n_before = len(_stacked_bands(ax))
    rescale_stacked_kdeplot(ax, orient="x")
    n_after = len(_stacked_bands(ax))
    assert n_after == n_before
    plt.close(fig)


@pytest.mark.parametrize("orient", ["x", "y"])
def test_band_proportions_preserved_at_peak(orient):
    """Log-axis band shares should match the original linear shares."""
    data_axis_idx = 0 if orient == "x" else 1

    fig0, ax0 = _make_stacked_kde(orient)
    bands0 = _stacked_bands(ax0)
    curves0 = [_extract_band(b, data_axis_idx) for b in bands0]
    totals = curves0[-1][2]
    peak_i = int(np.argmax(totals))
    linear_props = [
        (hi[peak_i] - lo[peak_i]) / totals[peak_i] for (_, lo, hi) in curves0
    ]
    plt.close(fig0)

    fig, ax = _make_stacked_kde(orient)
    rescale_stacked_kdeplot(ax, orient=orient)
    bands = _stacked_bands(ax)
    curves = [_extract_band(b, data_axis_idx) for b in bands]
    totals_log = curves[-1][2]
    peak_log = int(np.argmax(totals_log))
    if orient == "x":
        bottom = ax.get_ylim()[0]
    else:
        bottom = ax.get_xlim()[0]
    span = np.log(totals_log[peak_log]) - np.log(bottom)
    log_props = []
    prev = np.log(bottom)
    for (_, _, hi) in curves:
        top = np.log(hi[peak_log])
        log_props.append((top - prev) / span)
        prev = top
    plt.close(fig)

    np.testing.assert_allclose(log_props, linear_props, atol=1e-6)


def test_height_axis_log_lower_bound_matches_vmin():
    fig, ax = _make_stacked_kde("x")
    rescale_stacked_kdeplot(ax, orient="x", vmin=5e-4)
    assert ax.get_yscale() == "log"
    assert ax.get_ylim()[0] == pytest.approx(5e-4)
    plt.close(fig)
