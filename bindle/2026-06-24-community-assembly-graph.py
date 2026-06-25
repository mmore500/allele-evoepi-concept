import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell
def import_std():
    import pathlib
    from collections import defaultdict

    return defaultdict, pathlib


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
    mo.md("""
    # Community Assembly Graphs

    This notebook builds **community assembly graphs** that summarize how
    the set of co-circulating strains turns over during each simulation
    replicate, then aggregates those per-replicate *transition sequences*
    into one directed graph **per mutation rate**.

    Each strain is a 3-bit genome (`N_SITES = 3`, so 8 possible strains
    `000`..`111`). A **community** (graph node) is the set of strains
    simultaneously *present*. As strains are introduced and go extinct,
    the community changes; each change is a directed **transition** (edge)
    from one community to the next. The founder/wildtype is the all-zero
    `000` strain and its bitwise complement is the all-ones `111` strain.

    The work is performed in five sections:

    1. **Setup / Download** --- pull the three OSF parquet caches.
    2. **Build Transition Sequence Graphs per Replicate** --- presence
       calling, per-replicate allele relabeling, end-state cutoff.
    3. **Plot the example transition sequence graph for each replicate
       uid** in the example parquet.
    4. **Combine Graphs per mutation rate and calculate weightings.**
    5. **Plot graphs for each mutation rate** with `iplotx`
       (matplotlib backend) via `teeplot`.
    """)
    return


@app.cell(hide_code=True)
def delimit_setup(mo):
    mo.md("""
    ## 1. Setup / Download

    Three OSF parquet slugs back this notebook, each downloaded with
    `requests` and cached under `/tmp/<slug>` so re-runs hit the local
    copy:

    - **main** (`wubva`) --- per-`(replicate, strain)` **presence-episode
      event log**. Each episode that a strain spends continuously present
      emits milestone rows at `continuous_steps` = 10, 30, 100, 300 (those
      it reaches) and a terminal `extinct=True` row when the episode ends.
      The persistence threshold is therefore already encoded in
      `continuous_steps`; we use the **30-step** threshold for calling
      introductions/extinctions (not 10, 100, or 300). An `extinct=True`
      row at a step *before* the final step is a genuine extinction; one
      at the final step is right-censoring (the strain is still present).
    - **dominant** (`nmzrj`) --- final-step (`Step == n_steps - 1`)
      per-strain case counts, used to identify each replicate's
      end-dominant strain and to test the convergence (>50% prevalence)
      criterion.
    - **examples** (`f4bzv`) --- a curated subset of `replicate_uid`s
      (a few per mutation rate) whose individual transition sequences are
      drawn in Section 3.
    """)
    return


@app.cell
def configure_args(mo):
    # CLI args; defaults pull the parquet caches that back this notebook.
    _args = mo.cli_args()
    MAIN_SLUG = str(_args.get("main-slug") or "wubva")
    DOM_SLUG = str(_args.get("dominant-slug") or "nmzrj")
    EX_SLUG = str(_args.get("examples-slug") or "f4bzv")
    # persistence threshold (days/steps) for introduction & extinction
    # calling; the main parquet encodes 10/30/100/300 --- we use 30.
    THRESHOLD = int(_args.get("threshold") or 30)
    print(
        f"args: MAIN_SLUG={MAIN_SLUG} DOM_SLUG={DOM_SLUG} "
        f"EX_SLUG={EX_SLUG} THRESHOLD={THRESHOLD}",
    )
    return DOM_SLUG, EX_SLUG, MAIN_SLUG, THRESHOLD


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
def download_data(DOM_SLUG, EX_SLUG, MAIN_SLUG, download_parquet):
    main_df = download_parquet(MAIN_SLUG)
    dom_df = download_parquet(DOM_SLUG)
    ex_df = download_parquet(EX_SLUG)

    N_SITES = int(main_df["n_sites"].iloc[0])
    FULL = (1 << N_SITES) - 1  # all-ones strain (complement of wildtype)
    N_STEPS = int(main_df["n_steps"].iloc[0])
    print(f"main: {main_df.shape}  dominant: {dom_df.shape}  ex: {ex_df.shape}")
    print(f"N_SITES={N_SITES} FULL={FULL:0{N_SITES}b} N_STEPS={N_STEPS}")
    print(
        "replicates per mutation_rate:\n"
        + str(main_df.groupby("mutation_rate")["replicate_uid"].nunique()),
    )
    return FULL, N_SITES, N_STEPS, dom_df, ex_df, main_df


@app.cell
def peek_main(main_df, pd):
    pd.concat(
        [
            main_df[
                [
                    "replicate_uid",
                    "Step",
                    "strain",
                    "n_cases",
                    "count",
                    "continuous_steps",
                    "extinct",
                    "mutation_rate",
                ]
            ].head(),
            main_df[
                [
                    "replicate_uid",
                    "Step",
                    "strain",
                    "n_cases",
                    "count",
                    "continuous_steps",
                    "extinct",
                    "mutation_rate",
                ]
            ].tail(),
        ],
    )
    return


@app.cell(hide_code=True)
def delimit_build(mo):
    mo.md(r"""
    ## 2. Build Transition Sequence Graphs per Replicate

    For each replicate we reconstruct the ordered sequence of community
    states (the *transition sequence*) from the presence-episode log.

    **Presence calling (30-step threshold).** Walking each strain's rows
    in time order, episodes are delimited by `extinct=True` rows. An
    episode counts as a real *introduction* only if it reaches
    `continuous_steps >= THRESHOLD` (briefer blips are dropped). The
    strain is then *present* from the step at which it crossed the
    threshold until its episode's terminal `extinct=True` row. A terminal
    row at the final step (`Step >= N_STEPS - 1`) is right-censored ---
    the strain is still present at the end --- whereas an earlier terminal
    row is a genuine extinction.

    **Per-replicate allele relabeling.** Bit positions (sites) are
    reordered so the site whose mutant allele *appears first* (earliest
    introduction of any strain carrying it) occupies position 0, the
    second-to-appear position 1, and the third position 2. Every strain
    label in the replicate is permuted accordingly. The wildtype `000`
    and its complement `111` are fixed points of any such permutation.

    **End-state cutoff.** Let `D` be the replicate's end-dominant strain
    (largest final-step count) and `comp = D ^ FULL` its complement. A
    replicate **converges** when `D` and `comp` both appear *and* together
    they exceed 50% of circulating cases. Because the main log records
    prevalence only at early episode milestones, the >50% test is applied
    to the accurate **final-step** relative prevalence
    `(count_D + count_comp) / total_cases` (from the dominant parquet);
    the sequence is then cut off at the first community in which both `D`
    and `comp` are present. Replicates that **fail to converge** terminate
    in a special end node labelled **`U`**.
    """)
    return


@app.cell
def def_conv_info(FULL, dom_df):
    # Per-replicate: end-dominant strain D, its complement, and whether
    # the run converged (final-step relative prevalence of D+comp > 0.5).
    def build_conv_info(dom_df):
        info = {}
        for uid, sub in dom_df.groupby("replicate_uid"):
            counts = sub.set_index("strain")["count"]
            total = float(counts.sum())
            D = int(counts.idxmax())
            comp = D ^ FULL
            rel = (
                (counts.get(D, 0.0) + counts.get(comp, 0.0)) / total
                if total > 0
                else 0.0
            )
            info[uid] = {"D": D, "comp": comp, "converged": bool(rel > 0.5)}
        return info

    conv_info = build_conv_info(dom_df)
    _n_conv = sum(v["converged"] for v in conv_info.values())
    print(
        f"convergence (final D+comp > 50% of cases): "
        f"{_n_conv} / {len(conv_info)} replicates",
    )
    return (conv_info,)


@app.cell
def def_intervals(N_SITES, N_STEPS, THRESHOLD, np):
    def build_intervals(rep_df):
        """Presence intervals for episodes reaching THRESHOLD.

        Returns list of (strain, intro_step, ext_step). ``ext_step ==
        N_STEPS`` marks a right-censored episode (present through the
        final step); otherwise it is the extinction step.
        """
        out = []
        for s, sdf in rep_df.groupby("strain"):
            sdf = sdf.sort_values("Step")
            episode = []
            for row in sdf.itertuples(index=False):
                episode.append(row)
                if row.extinct:
                    reached = max(x.continuous_steps for x in episode)
                    if reached >= THRESHOLD:
                        intro = next(
                            x
                            for x in episode
                            if x.continuous_steps >= THRESHOLD
                        )
                        term = episode[-1]
                        ext = (
                            N_STEPS
                            if term.Step >= N_STEPS - 1
                            else int(term.Step)
                        )
                        out.append((int(s), int(intro.Step), ext))
                    episode = []
        return out

    def relabel_fn(intervals):
        """Bit permutation ordering sites by first-appearance step."""
        first = {i: np.inf for i in range(N_SITES)}
        for strain, intro, _ext in intervals:
            for i in range(N_SITES):
                if (strain >> i) & 1:
                    first[i] = min(first[i], intro)
        order = sorted(range(N_SITES), key=lambda i: (first[i], i))
        perm = {order[k]: k for k in range(N_SITES)}
        return lambda x: sum(
            (1 << perm[i]) for i in range(N_SITES) if (x >> i) & 1
        )

    return build_intervals, relabel_fn


@app.cell
def def_assemble(
    N_SITES,
    N_STEPS,
    build_intervals,
    conv_info,
    defaultdict,
    relabel_fn,
):
    def assemble(uid, rep_df):
        """Build a replicate's transition sequence.

        Returns ``(path, converged)`` where ``path`` is a list of
        ``frozenset`` community states (relabeled strain ints). When the
        run fails to converge the caller appends the ``"U"`` end node.
        """
        intervals = build_intervals(rep_df)
        remap = relabel_fn(intervals)
        D = conv_info[uid]["D"]
        comp = conv_info[uid]["comp"]
        converged_run = conv_info[uid]["converged"]

        events = defaultdict(list)
        for strain, intro, ext in intervals:
            events[intro].append((1, strain))
            if ext < N_STEPS:
                events[ext].append((0, strain))

        present = set()
        seq = []
        for step in sorted(events):
            for kind, strain in events[step]:
                if kind:
                    present.add(strain)
                else:
                    present.discard(strain)
            ps = frozenset(present)
            if not ps and not seq:
                continue
            if seq and seq[-1] == ps:
                continue
            seq.append(ps)

        converged = False
        if converged_run:
            for idx, ps in enumerate(seq):
                if D in ps and comp in ps:
                    seq = seq[: idx + 1]
                    converged = True
                    break

        path = [frozenset(remap(s) for s in ps) for ps in seq]
        return path, converged

    def path_with_end(uid, rep_df):
        path, converged = assemble(uid, rep_df)
        keys = list(path)
        if not converged:
            keys = keys + ["U"]
        return keys, converged

    def binstr(strain):
        return format(strain, f"0{N_SITES}b")

    def node_text(key):
        if key == "U":
            return "U"
        return ", ".join(binstr(s) for s in sorted(key))

    return node_text, path_with_end


@app.cell
def peek_sequences(ex_df, main_df, node_text, path_with_end):
    # Sanity peek: a handful of example replicate sequences.
    _uids = ex_df["replicate_uid"].unique()[:6]
    for _uid in _uids:
        _keys, _conv = path_with_end(
            _uid, main_df[main_df["replicate_uid"] == _uid]
        )
        _mu = main_df.loc[
            main_df["replicate_uid"] == _uid, "mutation_rate"
        ].iloc[0]
        print(f"{_uid[:8]} mu={_mu:.0e} converged={_conv}")
        print("   " + " -> ".join("[" + node_text(k) + "]" for k in _keys))
    return


@app.cell(hide_code=True)
def delimit_examples(mo):
    mo.md("""
    ## 3. Example Transition Sequence Graph per Replicate

    For every `replicate_uid` in the examples parquet we draw its
    individual transition sequence as a left-to-right directed path, one
    figure per mutation rate (each row is one replicate). Nodes are
    color-coded by their community (present-strain set) via the key on
    the right; the **number on each node is the count of present
    strains**. A converged run's terminal node carries a **green outline**
    when it contains both the `000` wildtype and its `111` complement; a
    non-converging run terminates in the **red-outlined `U`** node.
    """)
    return


@app.cell
def def_plot_path(FULL, ig, iplotx, node_text, plt, sns):
    def node_colormap(allsets, palette="husl"):
        ordered = sorted(allsets, key=lambda s: (len(s), tuple(sorted(s))))
        colors = sns.color_palette(palette, n_colors=max(3, len(ordered)))
        return {k: colors[i] for i, k in enumerate(ordered)}

    def plot_path(keys, converged, ax, colormap):
        n = len(keys)
        g = ig.Graph(directed=True)
        g.add_vertices(n)
        g.add_edges([(i, i + 1) for i in range(n - 1)])
        coords = [(float(i), 0.0) for i in range(n)]
        vface, vedge, vlw, vlabel, vsize = [], [], [], [], []
        for j, k in enumerate(keys):
            last = j == n - 1
            if k == "U":
                vface.append("white")
                vedge.append("red")
                vlw.append(3.0)
                vlabel.append("U")
            else:
                vface.append(colormap[k])
                has_axis = (0 in k) and (FULL in k)
                if last and converged and has_axis:
                    vedge.append("green")
                    vlw.append(3.0)
                else:
                    vedge.append("black")
                    vlw.append(1.0)
                vlabel.append(str(len(k)))
            vsize.append(34.0 if last else 26.0)
        iplotx.network(
            g,
            layout=coords,
            vertex_labels=vlabel,
            ax=ax,
            vertex_facecolor=vface,
            vertex_edgecolor=vedge,
            vertex_size=vsize,
            vertex_linewidth=vlw,
            edge_linewidth=2.0,
            edge_color="0.55",
            vertex_label_color="black",
            vertex_label_size=9,
            show=False,
        )
        ax.margins(0.15)

    def path_legend(ax, colormap, allsets):
        ax.axis("off")
        ordered = sorted(allsets, key=lambda s: (len(s), tuple(sorted(s))))
        handles, labels = [], []
        for k in ordered:
            outline = "green" if (0 in k and FULL in k) else "black"
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=colormap[k],
                    markeredgecolor=outline,
                    markeredgewidth=1.6 if outline == "green" else 0.6,
                    markersize=9,
                )
            )
            labels.append(node_text(k))
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="white",
                markeredgecolor="red",
                markeredgewidth=2.0,
                markersize=9,
            )
        )
        labels.append("U  (failed to converge)")
        ax.legend(
            handles,
            labels,
            loc="center left",
            fontsize=6.5,
            title="community (present strains)",
            title_fontsize=8,
            ncol=1 if len(ordered) < 16 else 2,
            frameon=False,
            labelspacing=0.3,
            handletextpad=0.4,
            borderaxespad=0.0,
        )

    return node_colormap, path_legend, plot_path


@app.cell
def def_plot_examples(
    node_colormap,
    path_legend,
    path_with_end,
    pathlib,
    plot_path,
    plt,
    tp,
):
    def plot_examples_for_mu(mu, uids, main_df):
        paths = []
        for uid in uids:
            keys, conv = path_with_end(
                uid, main_df[main_df["replicate_uid"] == uid]
            )
            paths.append((uid, keys, conv))
        allsets = {k for _u, ks, _c in paths for k in ks if k != "U"}
        colormap = node_colormap(allsets)
        nrow = max(1, len(paths))

        def _factory(**kwargs):
            fig = plt.figure(**kwargs)
            gs = fig.add_gridspec(nrow, 2, width_ratios=[3, 1.4])
            axes = [fig.add_subplot(gs[i, 0]) for i in range(nrow)]
            lax = fig.add_subplot(gs[:, 1])
            return fig, (axes, lax)

        with tp.teed(
            _factory,
            figsize=(12, 1.5 * nrow + 1.2),
            teeplot_outattrs={
                "a": "example-transition-sequences",
                "mutation-rate": f"{mu:.0e}",
            },
            teeplot_outexclude="viz",
            teeplot_show=True,
            teeplot_subdir=pathlib.Path(__file__).stem,
        ) as (fig, (axes, lax)):
            for ax, (uid, keys, conv) in zip(axes, paths):
                plot_path(keys, conv, ax, colormap)
                ax.set_ylabel(
                    uid[:6],
                    rotation=0,
                    ha="right",
                    va="center",
                    fontsize=7,
                )
            path_legend(lax, colormap, allsets)
            fig.suptitle(
                f"example transition sequences  ---  mutation rate "
                f"= {mu:.0e}",
                fontsize=10,
            )

    return (plot_examples_for_mu,)


@app.cell
def run_examples(ex_df, main_df, plot_examples_for_mu):
    for _mu in sorted(ex_df["mutation_rate"].unique()):
        _uids = (
            ex_df.loc[ex_df["mutation_rate"] == _mu, "replicate_uid"]
            .unique()
            .tolist()
        )
        plot_examples_for_mu(_mu, _uids, main_df)
    return


@app.cell(hide_code=True)
def delimit_combine(mo):
    mo.md("""
    ## 4. Combine Graphs per Mutation Rate and Calculate Weightings

    For each mutation rate we overlay every replicate's transition
    sequence into one directed graph and tally:

    - **node occurrences** --- per replicate, each community is counted
      once as *transient* (appears mid-sequence) and/or once as
      *end-state* (the replicate's terminal node). The `U` node is always
      an end-state. A community can be transient in some replicates and an
      end-state in others; it is classed as an **end-state node** if it is
      terminal in at least one replicate.
    - **edge occurrences** --- how many replicate transitions follow each
      `source -> target`.

    Nodes are **sized by occurrence**, with end-states scaled *among
    end-states* and transient nodes scaled *among transient nodes* (so the
    two roles are sized on separate scales; sizing counts occurrences, not
    durations). Edges are sized by transition frequency, and each edge is
    labelled with the **percentage (2 digits) of its source node's
    outgoing transitions** that it accounts for. Sparse nodes/edges are
    pruned for legibility via per-mutation-rate minimum-occurrence
    thresholds (the `U` node and the `000` founder are always kept).
    """)
    return


@app.cell
def def_aggregate(defaultdict, path_with_end):
    def aggregate(reps_df):
        """Tally node/edge occurrences across a mutation rate's replicates."""
        uids = reps_df["replicate_uid"].unique()
        end_count = defaultdict(int)
        trans_count = defaultdict(int)
        edge_count = defaultdict(int)
        out_total = defaultdict(int)
        for uid in uids:
            keys, _conv = path_with_end(
                uid, reps_df[reps_df["replicate_uid"] == uid]
            )
            seen = set()
            for i, k in enumerate(keys):
                last = i == len(keys) - 1
                role = "end" if last else "tr"
                if (k, role) not in seen:
                    seen.add((k, role))
                    if last:
                        end_count[k] += 1
                    else:
                        trans_count[k] += 1
                if i > 0:
                    edge_count[(keys[i - 1], k)] += 1
                    out_total[keys[i - 1]] += 1
        return {
            "end_count": dict(end_count),
            "trans_count": dict(trans_count),
            "edge_count": dict(edge_count),
            "out_total": dict(out_total),
            "n_reps": len(uids),
        }

    def build_graph(agg, min_count, min_edge):
        occ = defaultdict(int)
        for k, v in agg["end_count"].items():
            occ[k] += v
        for k, v in agg["trans_count"].items():
            occ[k] += v
        keep = {k for k, v in occ.items() if v >= min_count}
        keep.add("U") if "U" in occ else None
        keep.add(frozenset({0}))  # always keep founder
        keep = {k for k in keep if k in occ}
        kept_edges = {
            e: c
            for e, c in agg["edge_count"].items()
            if c >= min_edge and e[0] in keep and e[1] in keep
        }
        return keep, kept_edges

    return aggregate, build_graph


@app.cell
def compute_aggregates(aggregate, build_graph, main_df):
    mutation_rates = sorted(main_df["mutation_rate"].unique())
    aggregates = {}
    for _mu in mutation_rates:
        _sub = main_df[main_df["mutation_rate"] == _mu]
        _agg = aggregate(_sub)
        _min_count = max(2, int(0.05 * _agg["n_reps"]))
        _min_edge = max(2, int(0.03 * _agg["n_reps"]))
        _keep, _edges = build_graph(_agg, _min_count, _min_edge)
        aggregates[_mu] = {
            "agg": _agg,
            "keep": _keep,
            "edges": _edges,
            "min_count": _min_count,
            "min_edge": _min_edge,
        }
        print(
            f"mu={_mu:.0e}: n_reps={_agg['n_reps']} "
            f"kept_nodes={len(_keep)} kept_edges={len(_edges)} "
            f"(min_count={_min_count}, min_edge={_min_edge})",
        )
    return aggregates, mutation_rates


@app.cell(hide_code=True)
def delimit_plot(mo):
    mo.md("""
    ## 5. Per-Mutation-Rate Community Assembly Graphs

    One directed community assembly graph per mutation rate, rendered with
    `iplotx` (matplotlib backend) via `teeplot`. Communities are laid out
    left-to-right by the **number of present strains**, so assembly reads
    as a progression from the `000` founder toward larger communities and
    on to the end-states (`U` sits in its own rightmost column).

    Node fill encodes the community via the color key; the **number on
    each node is the count of present strains**. End-state node sizes
    reflect end-state occurrence; transient node sizes reflect transient
    occurrence (separate scales). Edge width encodes transition frequency
    and each edge label is the 2-digit percentage of its source's outgoing
    transitions. The end-state containing both `000` and `111` is outlined
    **green**; the non-convergence `U` end-state is outlined **red**.
    """)
    return


@app.cell
def def_layout(defaultdict, np):
    def layout_layered(keep):
        """x = community size; U gets its own rightmost column."""
        cols = defaultdict(list)
        for k in keep:
            x = None if k == "U" else len(k)
            cols[x].append(k)
        coords = {}
        sizes = sorted(x for x in cols if x is not None)
        maxx = max(sizes) if sizes else 1
        for x in sizes:
            nodes = sorted(
                cols[x], key=lambda k: (-len(k), tuple(sorted(k)))
            )
            n = len(nodes)
            ys = (
                np.linspace(-(n - 1) / 2.0, (n - 1) / 2.0, n)
                if n > 1
                else [0.0]
            )
            for k, y in zip(nodes, ys):
                coords[k] = (float(x), float(y) * 1.6)
        if None in cols:
            coords["U"] = (float(maxx + 1), 0.0)
        return coords

    return (layout_layered,)


@app.cell
def def_plot_mu(FULL, ig, iplotx, layout_layered, node_text, np, plt, sns):
    def plot_mu(mu, bundle, ax, legend_ax, palette="husl"):
        agg = bundle["agg"]
        keep = bundle["keep"]
        kept_edges = bundle["edges"]
        ec = agg["end_count"]
        tc = agg["trans_count"]

        nodes = sorted(
            keep,
            key=lambda k: (
                k == "U",
                len(k) if k != "U" else 99,
                tuple(sorted(k)) if k != "U" else (),
            ),
        )
        idx = {k: i for i, k in enumerate(nodes)}
        coords = layout_layered(keep)

        # end-state nodes: terminal in >=1 replicate (U always end-state).
        end_nodes = {k for k in nodes if k == "U" or ec.get(k, 0) > 0}
        max_end = max([ec.get(k, 0) for k in end_nodes] + [1])
        max_tr = max(
            [tc.get(k, 0) for k in nodes if k not in end_nodes] + [1]
        )

        g = ig.Graph(directed=True)
        g.add_vertices(len(nodes))
        elist, ewidth, elabel = [], [], []
        max_e = max(kept_edges.values()) if kept_edges else 1
        for (a, b), c in kept_edges.items():
            elist.append((idx[a], idx[b]))
            ewidth.append(1.0 + 4.0 * (c / max_e))
            pct = 100.0 * c / max(agg["out_total"].get(a, c), 1)
            elabel.append(f"{min(pct, 99):02.0f}%")
        g.add_edges(elist)

        colors = sns.color_palette(
            palette, n_colors=max(3, len([k for k in nodes if k != "U"]))
        )
        vsize, vface, vedge, vlw, vlabel = [], [], [], [], []
        colormap = {}
        ci = 0
        for k in nodes:
            if k == "U":
                vsize.append(
                    16.0 + 34.0 * np.sqrt(ec.get("U", 0) / max_end)
                )
                vface.append("white")
                vedge.append("red")
                vlw.append(3.0)
                vlabel.append("U")
                continue
            colormap[k] = colors[ci]
            ci += 1
            if k in end_nodes:
                vsize.append(16.0 + 34.0 * np.sqrt(ec.get(k, 0) / max_end))
                has_axis = (0 in k) and (FULL in k)
                vedge.append("green" if has_axis else "black")
                vlw.append(3.5 if has_axis else 1.0)
            else:
                vsize.append(7.0 + 20.0 * np.sqrt(tc.get(k, 0) / max_tr))
                vedge.append("black")
                vlw.append(1.0)
            vface.append(colormap[k])
            vlabel.append(str(len(k)))

        iplotx.network(
            g,
            layout=[coords[k] for k in nodes],
            vertex_labels=vlabel,
            edge_labels=elabel,
            ax=ax,
            vertex_facecolor=vface,
            vertex_edgecolor=vedge,
            vertex_size=vsize,
            vertex_linewidth=vlw,
            edge_linewidth=ewidth,
            edge_color="0.55",
            vertex_label_color="black",
            vertex_label_size=8,
            edge_label_size=6,
            edge_label_rotate=False,
            show=False,
        )
        ax.set_title(
            f"mutation rate = {mu:.0e}   (n={agg['n_reps']} replicates)"
        )
        ax.margins(0.08)

        # ---- color-coded key ----
        legend_ax.axis("off")
        handles, labels = [], []
        for k in nodes:
            if k == "U":
                handles.append(
                    plt.Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        markerfacecolor="white",
                        markeredgecolor="red",
                        markeredgewidth=2.0,
                        markersize=9,
                    )
                )
                labels.append("U  (failed to converge)")
                continue
            outline = (
                "green"
                if (0 in k and FULL in k and k in end_nodes)
                else "black"
            )
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=colormap[k],
                    markeredgecolor=outline,
                    markeredgewidth=1.8 if outline == "green" else 0.6,
                    markersize=9,
                )
            )
            labels.append(node_text(k))
        legend_ax.legend(
            handles,
            labels,
            loc="center left",
            fontsize=6.5,
            title="community (present strains)",
            title_fontsize=8,
            ncol=1 if len(nodes) <= 18 else 2,
            frameon=False,
            handletextpad=0.4,
            labelspacing=0.3,
            borderaxespad=0.0,
        )

    return (plot_mu,)


@app.cell
def render_graphs(aggregates, mutation_rates, pathlib, plot_mu, plt, tp):
    for _mu in mutation_rates:
        _bundle = aggregates[_mu]
        _n_nodes = len(_bundle["keep"])
        _wide = _n_nodes > 18

        def _factory(**kwargs):
            fig, (ax, lax) = plt.subplots(
                1, 2, gridspec_kw={"width_ratios": [3, 1.3]}, **kwargs
            )
            return fig, (ax, lax)

        with tp.teed(
            _factory,
            figsize=(14 if _wide else 13, 8.5 if _wide else 7.5),
            teeplot_outattrs={
                "a": "community-assembly-graph",
                "mutation-rate": f"{_mu:.0e}",
            },
            teeplot_outexclude="viz",
            teeplot_show=True,
            teeplot_subdir=pathlib.Path(__file__).stem,
        ) as (fig, (ax, lax)):
            plot_mu(_mu, _bundle, ax, lax)
    return


@app.cell(hide_code=True)
def delimit_summary(mo):
    mo.md("""
    ## Convergence Summary

    Fraction of replicates that converge (`D` and complement present and
    >50% of cases) versus terminate in `U`, by mutation rate --- the
    transition from low-mutation convergence to high-mutation
    non-convergence.
    """)
    return


@app.cell
def convergence_table(aggregates, mo, mutation_rates, pd):
    _rows = []
    for _mu in mutation_rates:
        _agg = aggregates[_mu]["agg"]
        _n = _agg["n_reps"]
        _u = _agg["end_count"].get("U", 0)
        _rows.append(
            {
                "mutation_rate": f"{_mu:.0e}",
                "n_reps": _n,
                "converged": _n - _u,
                "U (failed)": _u,
                "conv_frac": round((_n - _u) / _n, 3) if _n else 0.0,
            }
        )
    _tbl = pd.DataFrame(_rows)
    print(_tbl.to_string(index=False))
    mo.ui.table(_tbl, selection=None)
    return


if __name__ == "__main__":
    app.run()
