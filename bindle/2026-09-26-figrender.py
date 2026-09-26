import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell
def import_std():
    import pathlib

    return (pathlib,)


@app.cell
def import_pkg():
    import marimo as mo
    import pandas as pd
    import requests
    import seaborn as sns
    from teeplot import teeplot as tp
    from watermark import watermark

    return mo, pd, requests, sns, tp, watermark


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
def delimit_data(mo):
    mo.md("""
    ## Data

    Render a reproduction of Figure 3 from Koopman & Eisenberg,
    "An allele-based evolution model of the population spread of
    SARS-CoV-2" (medRxiv, <https://doi.org/10.1101/2025.09.15.25335781>).
    That figure follows a fully susceptible population through an
    initial outbreak of a 2-allele strain, its immune-driven mutational
    replacement, and the ensuing damped oscillations in strain
    prevalence.
    The underlying Madonna ODE-model trajectory is hosted on OSF as
    `Fig3.csv` (<https://osf.io/jy59v>) and downloaded below with no
    caching-key CLI arguments (this is a visualization notebook).
    Columns are keyed `TIME` plus one variable per 2-allele strain
    (named by the pair of mutated sites, e.g. `02`, `03`, `12`, `13`):
    a `J**` column gives that strain's prevalence, and a `SuscToI**`
    column gives the population's fraction still susceptible to
    (re)infection by that strain.
    In the reference dynamic, strain `02` seeds the initial wave.
    Immunity to it wanes unevenly across its two alleles, permitting
    mutations to intermediate strains `03` and `12` at vanishingly low
    prevalence.
    Both intermediates mutate onward to strain `13`, which faces little
    standing immunity and comes to dominate later waves.
    """)
    return


@app.cell
def def_fetch(pathlib, pd, requests):
    def fetch_osf(slug):
        cache_path = pathlib.Path("/tmp") / slug
        url = f"https://osf.io/{slug}/download"
        if not cache_path.exists():
            print(f"downloading {url} -> {cache_path}")
            resp = requests.get(url, allow_redirects=True, timeout=120)
            resp.raise_for_status()
            cache_path.write_bytes(resp.content)
        else:
            print(f"reusing cached {cache_path}")
        print(f"size: {cache_path.stat().st_size} bytes")
        return pd.read_csv(cache_path)

    return (fetch_osf,)


@app.cell
def download_data(fetch_osf):
    fig3_df = fetch_osf("jy59v")
    fig3_df.columns = [col.replace(":1", "") for col in fig3_df.columns]
    print(f"loaded Fig3 dataframe: {fig3_df.shape}")
    print(fig3_df.head().to_string(index=False))
    return (fig3_df,)


@app.cell(hide_code=True)
def delimit_plot(mo):
    mo.md("""
    ## Strain Dynamics Render

    Two panels stacked vertically, sharing a strain color hue and a
    linear scale.
    Top: per-strain prevalence over time (zero-prevalence stretches,
    i.e. before a strain has yet mutated into existence, are left as
    gaps).
    Bottom: per-strain fraction of the population susceptible to
    reinfection.
    """)
    return


@app.cell
def prep_long(fig3_df):
    long_df = fig3_df.melt(
        id_vars="TIME",
        var_name="variable",
        value_name="value",
    )
    long_df["strain"] = long_df["variable"].str.slice(-2)
    is_susceptibility = long_df["variable"].str.startswith("Susc")
    long_df["quantity"] = is_susceptibility.map(
        {True: "susceptibility", False: "prevalence"},
    )
    zeroed_prevalence = (long_df["quantity"] == "prevalence") & (
        long_df["value"] == 0
    )
    long_df.loc[zeroed_prevalence, "value"] = float("nan")
    return (long_df,)


@app.cell
def plot_fig3(long_df, pathlib, sns, tp):
    _strain_order = ["02", "03", "12", "13"]
    # Okabe-Ito colorblind-safe hues: 02/13 get cool (blue/bluish-green),
    # 03/12 get warm (vermillion/orange).
    _palette = {
        "02": "#0072B2",
        "03": "#D55E00",
        "12": "#E69F00",
        "13": "#009E73",
    }
    _dashes = {
        "02": "",
        "03": "",
        "12": (1, 1),
        "13": (4, 1.5),
    }

    with tp.teed(
        sns.relplot,
        data=long_df,
        x="TIME",
        y="value",
        hue="strain",
        hue_order=_strain_order,
        style="strain",
        style_order=_strain_order,
        dashes=_dashes,
        palette=_palette,
        row="quantity",
        row_order=["prevalence", "susceptibility"],
        kind="line",
        facet_kws=dict(sharey=False),
        teeplot_outattrs={"a": "fig3-strain-dynamics"},
        teeplot_show=True,
        teeplot_subdir=pathlib.Path(__file__).stem,
    ) as g:
        g.axes_dict["prevalence"].set_ylabel("prevalence")
        g.axes_dict["susceptibility"].set_ylabel("susceptibility")
        g.axes_dict["susceptibility"].set_xlabel("time")
        g.set_titles("")
        g.figure.set_size_inches(4.6, 2.6)
        sns.move_legend(
            g,
            "upper center",
            bbox_to_anchor=(0.5, 1.2),
            ncol=len(_strain_order),
            frameon=False,
            title="strain",
        )
    return


if __name__ == "__main__":
    app.run()
