import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell
def import_std():
    from collections import defaultdict, deque
    import itertools
    import pathlib

    return defaultdict, deque, itertools, pathlib


@app.cell
def import_pkg():
    # workaround: iplotx 1.7.x uses importlib.metadata without importing it
    import importlib.metadata  # noqa: F401

    import igraph as ig
    import iplotx
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import requests
    import seaborn as sns
    from teeplot import teeplot as tp
    from watermark import watermark

    return ig, iplotx, mo, np, pd, plt, requests, sns, tp, watermark


@app.cell(hide_code=True)
def do_watermark(mo, watermark):
    mo.md(
        f"""
    ```Text
    {watermark(
        current_date=True,
        iso8601=True,
        machine=True,
        updated=True,
        python=True,
        iversions=True,
        globals_=globals(),
    )}
    ```
    """
    )
    return


@app.cell(hide_code=True)
def delimit_intro(mo):
    mo.md(r"""
    # Estimated Community Assembly Graph from Invasibility

    This notebook builds an **estimated community assembly graph** for the
    3-site model (`N_SITES = 3`, 8 strains `000`..`111`) directly from the
    **susceptibility snapshots** produced by the strain-persistence array
    job (`2026-07-14-strain-persistence`). That job seeded every one of the
    256 possible strain communities (one per `array_id`, its 8-bit binary
    code selecting the seeded strains) and, every 1000 updates, recorded
    the population's mean susceptibility to each of the 8 strains.

    **Idea.** A community `C` (a set of resident strains) is *invasible* by
    a rare strain `x` when `x`'s reproduction number while rare exceeds 1.
    In this model that number is proportional to the mean population
    susceptibility to `x` measured in the run that seeded `C`, so we read
    invasibility straight off the susceptibility snapshot for
    `array_id = bitmask(C)`. Starting from the `000` founder we repeatedly
    let single-**mutation** neighbours of resident strains invade whenever
    they clear the threshold, growing an assembly graph whose nodes are
    communities and whose directed edges are invasions `C -> C + {x}`.

    **Arrival-order relabeling (the same trick as the mutation-sweep
    community-assembly notebook).** The three sites are exchangeable, so we
    canonicalize each community by relabeling sites in **order of arrival**:
    the site whose mutant allele arrives *first* becomes bit position 0, the
    second-to-arrive bit position 1, the third bit position 2. This is
    realized as the symmetry-minimal representative of the community, and it
    collapses the 6 site-permutations of every community onto one canonical
    node (e.g. all three single-mutant founders `001`/`010`/`100` become
    `001`).

    **Per-sample graphs.** Susceptibility drifts as herd immunity builds, so
    invasibility --- and thus the assembly graph --- can change over time. We
    loop over `teeplot` to draw one graph from the **first**, **second**, and
    **third** 1000-update susceptibility sample (`window_end_update` = 1000,
    2000, 3000).
    """)
    return


@app.cell
def configure_args(mo):
    # CLI args; defaults pull the susceptibility parquet that backs this
    # notebook (the strainlast parquet is optional annotation).
    _args = mo.cli_args()
    SUSC_SLUG = str(_args.get("susc-slug") or "jhnyk")
    STRAINLAST_SLUG = str(_args.get("strainlast-slug") or "9pebz")
    # Invasion threshold expressed as an R0-while-rare cutoff (a strain
    # invades when its estimated reproduction number while rare exceeds
    # this). 1.0 is the epidemic threshold.
    R_THRESHOLD = float(_args.get("r-threshold") or 1.0)
    # 1-based sample indices to draw (first, second, third 1000-update
    # susceptibility windows).
    SAMPLES = str(_args.get("samples") or "1,2,3")
    SAMPLE_INDICES = [int(s) for s in SAMPLES.split(",") if s.strip()]
    print(
        f"args: SUSC_SLUG={SUSC_SLUG} STRAINLAST_SLUG={STRAINLAST_SLUG} "
        f"R_THRESHOLD={R_THRESHOLD} SAMPLE_INDICES={SAMPLE_INDICES}",
    )
    return R_THRESHOLD, SAMPLE_INDICES, STRAINLAST_SLUG, SUSC_SLUG


@app.cell
def def_download(pathlib, pd, requests):
    def download_parquet(slug):
        cache_path = pathlib.Path("/tmp") / slug
        url = f"https://osf.io/{slug}/download"
        if not cache_path.exists():
            print(f"downloading {url} -> {cache_path}")
            resp = requests.get(url, allow_redirects=True, timeout=180)
            resp.raise_for_status()
            cache_path.write_bytes(resp.content)
        else:
            print(f"reusing cached {cache_path}")
        print(f"size: {cache_path.stat().st_size} bytes")
        return pd.read_parquet(cache_path)

    return (download_parquet,)


@app.cell
def download_data(STRAINLAST_SLUG, SUSC_SLUG, download_parquet):
    susc_df = download_parquet(SUSC_SLUG)
    strainlast_df = download_parquet(STRAINLAST_SLUG)

    N_SITES = int(susc_df["n_sites"].iloc[0])
    N_STRAINS = 1 << N_SITES
    CONTACT_RATE = float(susc_df["contact_rate"].iloc[0])
    RECOVERY_RATE = float(susc_df["recovery_rate"].iloc[0])
    WINDOW_ENDS = sorted(int(w) for w in susc_df["window_end_update"].unique())
    print(f"susc: {susc_df.shape}  strainlast: {strainlast_df.shape}")
    print(
        f"N_SITES={N_SITES} N_STRAINS={N_STRAINS} "
        f"CONTACT_RATE={CONTACT_RATE} RECOVERY_RATE={RECOVERY_RATE}",
    )
    print(
        f"n array_ids: {susc_df['array_id'].nunique()}  windows: {WINDOW_ENDS}"
    )
    return (
        CONTACT_RATE,
        N_SITES,
        N_STRAINS,
        RECOVERY_RATE,
        WINDOW_ENDS,
        susc_df,
    )


@app.cell
def build_susc_map(N_STRAINS, np, susc_df):
    # susc_map[(array_id, window_end_update)] -> length-N_STRAINS vector of
    # mean susceptibility to each strain (genome integer) in that community.
    def build_susc_map(df):
        out = {}
        for (aid, w), grp in df.groupby(["array_id", "window_end_update"]):
            vec = np.zeros(N_STRAINS)
            for strain, susc in zip(grp["strain"], grp["mean_susceptibility"]):
                vec[int(strain)] = float(susc)
            out[(int(aid), int(w))] = vec
        return out

    susc_map = build_susc_map(susc_df)
    print(f"susc_map entries: {len(susc_map)} (256 communities x windows)")
    return (susc_map,)


@app.cell
def def_susc_threshold(CONTACT_RATE, RECOVERY_RATE, R_THRESHOLD):
    # A rare strain x invading community C has, per infectious host,
    # ~CONTACT_RATE * susc new infections per update, and its deterministic
    # infectious period contributes an R0 multiplier (1 - RECOVERY_RATE) /
    # RECOVERY_RATE (the same multiplier the founder notebook uses: recovery
    # runs after transmission within an update, so a primary infected at t
    # transmits over sum_k (1-p)^k = (1-p)/p future updates). Hence
    #   R0(x | C) = CONTACT_RATE * (1 - RECOVERY_RATE) / RECOVERY_RATE * susc
    # and x invades when R0 > R_THRESHOLD, i.e. susc > SUSC_THRESHOLD.
    R0_PER_SUSC = CONTACT_RATE * (1.0 - RECOVERY_RATE) / RECOVERY_RATE
    SUSC_THRESHOLD = R_THRESHOLD / R0_PER_SUSC
    print(
        f"R0 per unit susceptibility = {R0_PER_SUSC:.4f}; "
        f"invade when susc > {SUSC_THRESHOLD:.4f} "
        f"(R0 > {R_THRESHOLD})",
    )
    return (SUSC_THRESHOLD,)


@app.cell(hide_code=True)
def delimit_build(mo):
    mo.md(r"""
    ## Build the Invasion Assembly Graph

    **Canonicalization.** `canon(community)` relabels sites by arrival order,
    implemented as the lexicographically-minimal genome set over the
    `N_SITES!` site permutations. Because every site has identical
    parameters, symmetric communities collapse to one canonical node and
    the founder `000` and all-ones `111` are fixed points.

    **Invasion step.** From canonical community `C` (with
    `array_id = bitmask(C)`) the candidate invaders are the single-mutation
    neighbours of resident strains that are not already present ---
    `x = r ^ (1 << s)` for a resident `r` and site `s`. Strain `x` invades
    when its measured susceptibility in `C`'s run exceeds `SUSC_THRESHOLD`;
    the resulting community `canon(C + {x})` is the edge target.

    **Assembly.** A breadth-first sweep from the `000` founder applies the
    invasion step until no new community appears, yielding the directed
    assembly graph for that susceptibility sample. Communities with no
    admissible invader are **terminal** (uninvadable) assembly endpoints.

    **Complement-pair end states.** A terminal community that harbours a
    strain and its bitwise complement (`a` and `a ^ 111`) is a *converged*
    end state --- the complement pair partitions the immune landscape and
    stably coexists. Each such terminal converges to a **complement-pair end
    node** (`000/111`, `001/110`, `010/101`, `011/100`); branches ending on
    the same pair share that node. Complementation commutes with the
    arrival-order relabeling, so the pairs are canonical too.
    """)
    return


@app.cell
def def_graph(N_SITES, deque, itertools, susc_map):
    _PERMS = list(itertools.permutations(range(N_SITES)))

    def _permute_genome(g, perm):
        return sum((1 << perm[i]) for i in range(N_SITES) if (g >> i) & 1)

    def canon(community):
        """Arrival-order canonical form: symmetry-minimal genome set.

        Relabels sites so the first-arriving site is bit 0, etc. --- the
        lexicographically smallest genome frozenset over all site
        permutations. Symmetric communities share one canonical node.
        """
        best_key = None
        best = None
        for perm in _PERMS:
            mapped = frozenset(_permute_genome(g, perm) for g in community)
            key = tuple(sorted(mapped))
            if best_key is None or key < best_key:
                best_key, best = key, mapped
        return best

    def bitmask(community):
        m = 0
        for g in community:
            m |= 1 << g
        return int(m)

    def invaders(community, window):
        """(strain, susceptibility) for admissible single-mutation invaders.

        Candidates are Hamming-1 neighbours of resident strains not already
        present; susceptibility is read from the community's own run
        (array_id = bitmask). Threshold comparison is left to the caller.
        """
        susc = susc_map[(bitmask(community), int(window))]
        cands = {
            r ^ (1 << s) for r in community for s in range(N_SITES)
        } - set(community)
        return sorted((x, float(susc[x])) for x in cands)

    def build_assembly_graph(window, susc_threshold):
        """BFS from the 000 founder; returns (nodes, edges).

        nodes: set of canonical community frozensets.
        edges: dict (src_tuple, dst_tuple) -> (invader_strain, susc).
        """
        root = canon(frozenset({0}))
        nodes = {root}
        edges = {}
        queue = deque([root])
        while queue:
            community = queue.popleft()
            for strain, susc in invaders(community, window):
                if susc <= susc_threshold:
                    continue
                child = canon(community | {strain})
                edges[(tuple(sorted(community)), tuple(sorted(child)))] = (
                    int(strain),
                    susc,
                )
                if child not in nodes:
                    nodes.add(child)
                    queue.append(child)
        return nodes, edges

    return (build_assembly_graph,)


@app.cell
def def_pairs(N_SITES):
    FULL = (1 << N_SITES) - 1  # all-ones strain (complement of the founder)

    def binstr(g):
        return format(g, f"0{N_SITES}b")

    def pair_key(a, b):
        """Canonical label for a complement pair, e.g. 000/111."""
        lo, hi = sorted((a, b))
        return f"{binstr(lo)}/{binstr(hi)}"

    def community_pairs(community):
        """Complement pairs {a, a ^ FULL} with both members present.

        A terminal community harbouring such a pair is a *converged*
        assembly end state: a strain and its bitwise complement stably
        coexisting (they partition the immune landscape). Complementation
        commutes with the arrival-order relabeling, so canonical communities
        keep canonical pairs.
        """
        return sorted(
            pair_key(a, a ^ FULL)
            for a in community
            if (a ^ FULL) in community and a <= (a ^ FULL)
        )

    # Outline colors for the complement-pair end states, mirroring the
    # mutation-sweep community-assembly notebook: 000/111 (founder + all-ones)
    # green, the remaining pairs each their own color.
    PAIR_KEYS = [
        pair_key(s, s ^ FULL) for s in range(FULL + 1) if s <= s ^ FULL
    ]
    _pair_palette = ["green", "#1f77b4", "#ff7f0e", "#9467bd", "#8c564b"]
    END_OUTLINE = {
        pk: _pair_palette[i % len(_pair_palette)]
        for i, pk in enumerate(PAIR_KEYS)
    }
    END_OUTLINE["000/111"] = "green"
    return END_OUTLINE, community_pairs


@app.cell
def def_flow(defaultdict):
    def assembly_flow(nodes, edges, pair_edges):
        """Even-split assembly flow (visitation probability) per node.

        A uniform assembly walk starts at the 000 founder with unit weight;
        at every juncture the node's weight is split *evenly* among its
        outgoing edges (invasions, or the convergence into a complement-pair
        end node). A node's flow is the total weight reaching it --- the
        probability the walk passes through it --- and the flow arriving at
        the terminal / pair end nodes partitions the unit weight across
        outcomes. Every invasion lengthens the community by one strain, so
        processing community nodes in order of size is a valid topological
        order (parents, being smaller, are always finished first).
        """
        founder = frozenset({0})
        succ = defaultdict(list)
        for a, b in edges:
            succ[frozenset(a)].append(frozenset(b))
        for a, pk in pair_edges:
            succ[frozenset(a)].append(pk)

        flow = {n: 0.0 for n in nodes}
        for _a, pk in pair_edges:
            flow[pk] = 0.0
        flow[founder] = 1.0
        for node in sorted(nodes, key=len):  # communities, small -> large
            outs = succ.get(node, [])
            if outs:
                share = flow[node] / len(outs)
                for target in outs:
                    flow[target] += share
        return flow

    return (assembly_flow,)


@app.cell
def build_graphs(
    SAMPLE_INDICES,
    SUSC_THRESHOLD,
    WINDOW_ENDS,
    assembly_flow,
    build_assembly_graph,
    community_pairs,
):
    # One assembly graph per requested susceptibility sample (1-based index
    # into WINDOW_ENDS -> the window's end update).
    graphs = {}
    for _i in SAMPLE_INDICES:
        if not 1 <= _i <= len(WINDOW_ENDS):
            print(f"skipping sample {_i}: out of range 1..{len(WINDOW_ENDS)}")
            continue
        _w = WINDOW_ENDS[_i - 1]
        _nodes, _edges = build_assembly_graph(_w, SUSC_THRESHOLD)
        _srcs = {e[0] for e in _edges}
        _terminal = {n for n in _nodes if tuple(sorted(n)) not in _srcs}
        # Complement-pair end states: each terminal community that harbours a
        # complement pair converges to a hollow, color-coded pair end node;
        # branches converging to the same pair share that node.
        _pair_nodes = set()
        _pair_edges = []
        for _n in _terminal:
            for _pk in community_pairs(_n):
                _pair_nodes.add(_pk)
                _pair_edges.append((tuple(sorted(_n)), _pk))
        _flow = assembly_flow(_nodes, _edges, _pair_edges)
        graphs[_i] = {
            "window_end": _w,
            "nodes": _nodes,
            "edges": _edges,
            "terminal": _terminal,
            "pair_nodes": _pair_nodes,
            "pair_edges": _pair_edges,
            "flow": _flow,
        }
        _pair_flow = {pk: _flow[pk] for pk in sorted(_pair_nodes)}
        print(
            f"sample {_i} (updates {_w - 1000 + 1}..{_w}): "
            f"{len(_nodes)} communities, {len(_edges)} invasions, "
            f"{len(_terminal)} terminal, "
            f"{len(_pair_nodes)} complement-pair end states; "
            f"end-state flow: "
            + ", ".join(f"{pk}={v:.2f}" for pk, v in _pair_flow.items()),
        )
    return (graphs,)


@app.cell
def peek_graph(N_SITES, graphs):
    # Text dump of the first requested sample's assembly graph.
    def _binset(t):
        return "{" + ",".join(format(g, f"0{N_SITES}b") for g in t) + "}"

    if graphs:
        _i = sorted(graphs)[0]
        _g = graphs[_i]
        print(f"sample {_i} (window end {_g['window_end']}):")
        for (a, b), (x, susc) in sorted(
            _g["edges"].items(), key=lambda kv: (len(kv[0][0]), kv[0])
        ):
            print(
                f"  {_binset(a)} --{format(x, f'0{N_SITES}b')}"
                f" (susc={susc:.3f})--> {_binset(b)}"
            )
    return


@app.cell(hide_code=True)
def delimit_plot(mo):
    mo.md("""
    ## Plot the Assembly Graphs

    One directed community assembly graph per susceptibility sample,
    rendered with `iplotx` (matplotlib backend) via a `teeplot` loop.
    Communities are laid out left-to-right by the **number of resident
    strains**, so assembly reads as growth from the `000` founder. Node
    fill encodes the community via the color key; the number on each node is
    the count of resident strains. The `000` founder is outlined **green**.

    **Node size** encodes the **even-split assembly flow**: a uniform
    assembly walk starts at the founder with unit weight and splits its
    weight *evenly* among the available invasions at each juncture, so a
    node's area is proportional to the probability the walk passes through
    it. The flow arriving at the terminal and complement-pair end nodes
    partitions the unit weight across outcomes (printed above).

    Each **complement-pair end state** is drawn as a **hollow circle with a
    color-coded outline** (`000/111` green, the other pairs their own colors)
    sharing the rightmost column; a terminal community converges into the
    hollow pair node it harbours. A terminal community that harbours *no*
    complement pair (an unresolved endpoint) is instead outlined **red**.

    Each invasion edge is optionally labelled with the arrival-ordered
    invading strain and its measured susceptibility (convergence edges into
    the pair nodes are unlabelled). Two versions are saved per sample (tagged
    `edge-labels=strain-susc` vs `edge-labels=none`).
    """)
    return


@app.cell
def def_layout(defaultdict):
    def layout_layered(nodes, np):
        """x = community size (# resident strains); y spreads within a layer."""
        cols = defaultdict(list)
        for n in nodes:
            cols[len(n)].append(n)
        coords = {}
        for x in sorted(cols):
            layer = sorted(cols[x], key=lambda k: tuple(sorted(k)))
            m = len(layer)
            ys = (
                np.linspace(-(m - 1) / 2.0, (m - 1) / 2.0, m)
                if m > 1
                else [0.0]
            )
            for k, y in zip(layer, ys):
                coords[k] = (float(x), float(y) * 1.6)
        return coords

    return (layout_layered,)


@app.cell
def def_plot(
    END_OUTLINE,
    N_SITES,
    defaultdict,
    ig,
    iplotx,
    layout_layered,
    np,
    plt,
    sns,
):
    def _binset(community):
        return ", ".join(format(g, f"0{N_SITES}b") for g in sorted(community))

    def _nid(k):
        # Node id: sorted-genome tuple for a community, the string for a
        # complement-pair end node.
        return tuple(sorted(k)) if isinstance(k, frozenset) else k

    def plot_graph(bundle, ax, legend_ax, palette="husl", edge_labels=True):
        comm_set = bundle["nodes"]
        edges = bundle["edges"]
        terminal = bundle["terminal"]
        pair_nodes = bundle["pair_nodes"]
        pair_edges = bundle["pair_edges"]
        flow = bundle["flow"]
        founder = frozenset({0})

        # Node size encodes the even-split assembly flow (visitation
        # probability): area proportional to flow, so radius ~ sqrt(flow).
        max_flow = max(flow.values()) if flow else 1.0

        def vsize(k):
            return 12.0 + 34.0 * np.sqrt(flow.get(k, 0.0) / max_flow)

        # Terminal communities that resolve to a complement pair (vs. an
        # unresolved "U"-like endpoint that harbours no pair). pair_edges
        # store sorted-tuple sources, so rebuild as frozensets to match the
        # community node keys.
        resolved = {frozenset(a) for a, _pk in pair_edges}

        communities = sorted(
            comm_set, key=lambda k: (len(k), tuple(sorted(k)))
        )

        # Layout: communities by size; complement-pair end nodes share the
        # rightmost column (like the reference notebook's end nodes), ordered
        # by the mean height of their source communities to limit crossings.
        coords = layout_layered(comm_set, np)
        maxsize = max((len(k) for k in comm_set), default=1)
        src_y = defaultdict(list)
        for a, pk in pair_edges:
            src_y[pk].append(coords[frozenset(a)][1])
        pairs = sorted(
            pair_nodes,
            key=lambda pk: sum(src_y[pk]) / max(len(src_y[pk]), 1),
        )
        m = len(pairs)
        ys = np.linspace(-(m - 1) / 2.0, (m - 1) / 2.0, m) if m > 1 else [0.0]
        for pk, y in zip(pairs, ys):
            coords[pk] = (float(maxsize + 1), float(y) * 1.8)

        nodes = communities + pairs  # community nodes first, pair nodes last
        idx = {_nid(k): i for i, k in enumerate(nodes)}

        colors = sns.color_palette(palette, n_colors=max(3, len(communities)))
        colormap = {k: colors[i] for i, k in enumerate(communities)}

        g = ig.Graph(directed=True)
        g.add_vertices(len(nodes))
        elist, ewidth, elabel = [], [], []
        max_susc = max((s for _x, s in edges.values()), default=1.0)
        for (a, b), (x, susc) in edges.items():
            elist.append((idx[a], idx[b]))
            ewidth.append(1.0 + 3.5 * (susc / max_susc))
            elabel.append(f"{format(x, f'0{N_SITES}b')}\n{susc:.2f}")
        for a, pk in pair_edges:  # convergence into the pair end node
            elist.append((idx[a], idx[pk]))
            ewidth.append(2.0)
            elabel.append("")
        g.add_edges(elist)

        vsizes, vface, vedge, vlw, vlabel = [], [], [], [], []
        for k in nodes:
            vsizes.append(vsize(k))
            if isinstance(k, str):  # complement-pair end node: hollow + coded
                vface.append("white")
                vedge.append(END_OUTLINE.get(k, "black"))
                vlw.append(3.0)
                vlabel.append("2")
            else:
                vface.append(colormap[k])
                if k == founder:
                    vedge.append("green")
                    vlw.append(3.0)
                elif k in terminal and k not in resolved:
                    vedge.append("red")  # unresolved endpoint (no pair)
                    vlw.append(3.0)
                else:
                    vedge.append("black")
                    vlw.append(1.0)
                vlabel.append(str(len(k)))

        iplotx.network(
            g,
            layout=[coords[k] for k in nodes],
            vertex_labels=vlabel,
            edge_labels=(elabel if edge_labels else None),
            ax=ax,
            vertex_facecolor=vface,
            vertex_edgecolor=vedge,
            vertex_size=vsizes,
            vertex_linewidth=vlw,
            edge_linewidth=ewidth,
            edge_color="0.55",
            vertex_label_color="black",
            vertex_label_size=9,
            edge_label_size=6,
            edge_label_rotate=False,
            show=False,
        )
        ax.margins(0.12)

        # color-coded key: communities (filled) then complement-pair end
        # states (hollow, color-coded outline).
        legend_ax.axis("off")
        handles, labels = [], []
        for k in communities:
            if k == founder:
                outline, mew, suffix = "green", 1.8, "  (founder)"
            elif k in terminal and k not in resolved:
                outline, mew, suffix = "red", 1.8, "  (terminal, no pair)"
            else:
                outline, mew, suffix = "black", 0.6, ""
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=colormap[k],
                    markeredgecolor=outline,
                    markeredgewidth=mew,
                    markersize=9,
                )
            )
            labels.append(f"{{{_binset(k)}}}{suffix}")
        for pk in pairs:
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="white",
                    markeredgecolor=END_OUTLINE.get(pk, "black"),
                    markeredgewidth=2.0,
                    markersize=9,
                )
            )
            labels.append(f"{pk}  (complement-pair end state)")
        legend_ax.legend(
            handles,
            labels,
            loc="center left",
            fontsize=6.5,
            title="community (resident strains)",
            title_fontsize=8,
            ncol=1 if len(handles) <= 16 else 2,
            frameon=False,
            handletextpad=0.4,
            labelspacing=0.3,
            borderaxespad=0.0,
        )

    return (plot_graph,)


@app.cell
def render_graphs(
    R_THRESHOLD,
    SUSC_THRESHOLD,
    graphs,
    pathlib,
    plot_graph,
    plt,
    tp,
):
    # Loop over teeplot: one assembly graph per susceptibility sample, in two
    # versions (edge labels vs bare arrows), tagged apart via teeplot_outattrs.
    for _i in sorted(graphs):
        _bundle = graphs[_i]
        _w = _bundle["window_end"]

        def _factory(**kwargs):
            fig, (ax, lax) = plt.subplots(
                1, 2, gridspec_kw={"width_ratios": [3, 1.2]}, **kwargs
            )
            return fig, (ax, lax)

        for _edge_labels, _tag in [(True, "strain-susc"), (False, "none")]:
            with tp.teed(
                _factory,
                figsize=(12, 7),
                teeplot_outattrs={
                    "a": "community-assembly-invasion-graph",
                    "sample": f"{_i}",
                    "window-end": f"{_w}",
                    "edge-labels": _tag,
                },
                teeplot_outexclude="viz",
                teeplot_show=True,
                teeplot_subdir=pathlib.Path(__file__).stem,
            ) as (fig, (ax, lax)):
                plot_graph(_bundle, ax, lax, edge_labels=_edge_labels)
                fig.suptitle(
                    f"estimated community assembly by invasion  ---  "
                    f"susceptibility sample {_i} (updates {_w - 1000 + 1}"
                    f"..{_w}),  R0>{R_THRESHOLD:g} "
                    f"(susc>{SUSC_THRESHOLD:.3f})",
                    fontsize=10,
                )
    return


if __name__ == "__main__":
    app.run()
