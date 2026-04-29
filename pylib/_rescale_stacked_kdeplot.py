"""Re-render a stacked seaborn KDE plot on a different height-axis scale.

Each band keeps its linear share of the visual height, so band proportions
look the same as on the original linear plot, but the overall envelope is
compressed onto the requested scale (currently only ``"log"``).
"""
from __future__ import annotations

from typing import List, Tuple

from matplotlib.axes import Axes
import numpy as np

try:
    from matplotlib.collections import FillBetweenPolyCollection

    _BAND_TYPES: Tuple[type, ...] = (FillBetweenPolyCollection,)
except ImportError:  # matplotlib < 3.10
    from matplotlib.collections import PolyCollection

    _BAND_TYPES = (PolyCollection,)


_VERTICAL = frozenset({"x", "v"})
_HORIZONTAL = frozenset({"y", "h"})


def _resolve_orient(orient: str) -> Tuple[int, int]:
    """Map a seaborn ``orient`` to ``(data_axis_idx, height_axis_idx)``.

    ``"x"`` / ``"v"`` mean the standard vertical KDE: data along x, density
    along y. ``"y"`` / ``"h"`` mean the rotated horizontal KDE.
    """
    if orient in _VERTICAL:
        return 0, 1
    if orient in _HORIZONTAL:
        return 1, 0
    valid = sorted(_VERTICAL | _HORIZONTAL)
    raise ValueError(f"orient must be one of {valid}, got {orient!r}")


def _stacked_bands(ax: Axes) -> List:
    bands = [c for c in ax.collections if isinstance(c, _BAND_TYPES)]
    if not bands:
        raise ValueError(
            "No stacked-KDE band polygons found on these axes.",
        )
    return bands


def _extract_band(
    band, data_axis_idx: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(data, height_lower, height_upper)`` on a common data grid.

    ``fill_between`` traces upper-left, lower-left, lower-edge along the
    data axis, then jumps up and traces back along the upper edge.
    Splitting at ``argmax`` of the data-axis coordinate cleanly separates
    the two edges.
    """
    verts = band.get_paths()[0].vertices
    height_idx = 1 - data_axis_idx
    k = int(np.argmax(verts[:, data_axis_idx]))
    lower = verts[1 : k + 1]
    upper = verts[k + 1 :][::-1]
    data = upper[:, data_axis_idx]
    h_upper = upper[:, height_idx]
    h_lower = np.interp(data, lower[:, data_axis_idx], lower[:, height_idx])
    return data, h_lower, h_upper


def rescale_stacked_kdeplot(
    ax: Axes,
    orient: str = "x",
    *,
    scale: str = "log",
    vmin: float | None = None,
) -> Axes:
    """Re-render a stacked-KDE axes on a non-linear height-axis scale.

    For ``scale="log"``, the top of the stacked envelope at column
    :math:`d` is placed at :math:`\\log T(d)`. Inside that envelope,
    band :math:`i`'s share of the visual (log-axis) height equals its
    linear share :math:`d_i / T`, so band proportions look the same as
    on a linear-scaled plot.

    Parameters
    ----------
    ax
        Axes containing a stacked seaborn ``kdeplot`` (``multiple="stack"``).
    orient
        Orientation in seaborn's convention. ``"x"`` / ``"v"`` for the
        standard ``sns.kdeplot(x=...)`` (densities on the y-axis);
        ``"y"`` / ``"h"`` for the rotated ``sns.kdeplot(y=...)``
        (densities on the x-axis).
    scale
        Height-axis scale to render onto. Only ``"log"`` is implemented;
        any other value raises ``NotImplementedError``.
    vmin
        Bottom of the log axis; must be > 0. Defaults to
        ``max(total) / 1000`` — roughly three decades of range.

    Returns
    -------
    Axes
        The same axes, modified in place.
    """
    if scale != "log":
        raise NotImplementedError(
            f"scale={scale!r} is not implemented; only 'log' is supported.",
        )
    data_axis_idx, height_axis_idx = _resolve_orient(orient)

    bands = _stacked_bands(ax)
    colors = [c.get_facecolor()[0] for c in bands]
    edgecolors = [c.get_edgecolor() for c in bands]
    linewidths = [c.get_linewidth() for c in bands]
    linestyles = [c.get_linestyle() for c in bands]

    band_curves = [_extract_band(b, data_axis_idx) for b in bands]
    data_grid = band_curves[0][0]
    totals = band_curves[-1][2]
    contribs = [hi - lo for (_, lo, hi) in band_curves]

    if vmin is None:
        vmin = max(float(totals.max()) / 1000.0, 1e-12)
    elif vmin <= 0:
        raise ValueError("vmin must be > 0")

    safe_totals = np.where(totals > 0, totals, vmin)

    new_edges = []
    cum_prop = np.zeros_like(totals)
    prev_edge = np.full_like(totals, vmin)
    for d_i in contribs:
        prop = np.where(totals > 0, d_i / safe_totals, 0.0)
        cum_prop = cum_prop + prop
        new_edge = vmin * (safe_totals / vmin) ** cum_prop
        new_edges.append((prev_edge.copy(), new_edge.copy()))
        prev_edge = new_edge

    for c in bands:
        c.remove()

    fill_fn = ax.fill_between if height_axis_idx == 1 else ax.fill_betweenx
    for color, ec, lw, ls, (lo, hi) in zip(
        colors, edgecolors, linewidths, linestyles, new_edges
    ):
        fill_fn(
            data_grid,
            lo,
            hi,
            facecolor=color,
            edgecolor=ec,
            linewidth=lw,
            linestyle=ls,
        )

    if height_axis_idx == 1:
        ax.set_yscale("log")
        ax.set_ylim(vmin, None)
    else:
        ax.set_xscale("log")
        ax.set_xlim(vmin, None)

    return ax
