import iplotx as ipx
import matplotlib.pyplot as plt
from phyloframe import legacy as pfl
import seaborn as sns


# adapted from https://iplotx.readthedocs.io/en/latest/gallery/tree/plot_phyloframe_tree.html#sphx-glr-gallery-tree-plot-phyloframe-tree-py
def draw_scatter_tree(
    phylogeny_df,
    *,
    hue=None,
    size=None,
    style=None,
    c=None,
    ax=None,
    layout="vertical",
    scatter_kws=None,
    tree_kws=None,
):
    """Draw a phylogenetic tree with a seaborn scatter overlay.

    Adapted from `hstrat-synthesis
    <https://github.com/mmore500/hstrat-synthesis/blob/main/pylib/tree/_draw_scatter_tree.py>`_.

    Parameters
    ----------
    phylogeny_df : polars.DataFrame
        Alife-standard phylogeny with optional metadata columns.
    hue, size, style : str, optional
        Column names forwarded to :func:`seaborn.scatterplot`.
    c : str, sequence, or None
        Colour values; a column name or explicit array.
    ax : matplotlib.axes.Axes, optional
        Target axes.  Created if *None*.
    layout : str
        Tree layout forwarded to :func:`iplotx.tree`.
    scatter_kws : dict, optional
        Extra keyword arguments for :func:`seaborn.scatterplot`.
    tree_kws : dict, optional
        Extra keyword arguments for :func:`iplotx.tree`.
    """
    if ax is None:
        ax = plt.gca()
    if scatter_kws is None:
        scatter_kws = {}
    if tree_kws is None:
        tree_kws = {}

    try:
        phylogeny_df = phylogeny_df.to_pandas()
    except AttributeError:
        pass

    tree_artist = ipx.tree(
        pfl.alifestd_to_iplotx_pandas(phylogeny_df),
        ax=ax,
        layout=layout,
        **{"margins": 0.0, "edge_linewidth": 1.5, **tree_kws},
    )

    # Extract node positions — radial layouts need Cartesian offsets
    ipx_layout = tree_artist.get_layout()
    if layout == "radial":
        xs, ys = tree_artist.get_nodes().get_offsets().T
    else:
        xs, ys = ipx_layout.T.to_numpy()

    pos = {
        node._id: (x, y) for node, (x, y) in zip(ipx_layout.index, zip(xs, ys))
    }

    # Map positions back onto the DataFrame
    plot_df = phylogeny_df.assign(
        __x__=phylogeny_df["id"]
        .map({k: v[0] for k, v in pos.items()})
        .astype(float),
        __y__=phylogeny_df["id"]
        .map({k: v[1] for k, v in pos.items()})
        .astype(float),
    )

    # Resolve colour argument
    if isinstance(c, str):
        c = plot_df[c].fillna("none").tolist()
    elif c is None:
        c = "none"

    sns.scatterplot(
        plot_df,
        x="__x__",
        y="__y__",
        hue=hue,
        size=size,
        style=style,
        c=c,
        ax=ax,
        **{"legend": False, "zorder": 5, **scatter_kws},
    )
    return ax
