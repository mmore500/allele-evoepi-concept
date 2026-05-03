import colorsys

import seaborn as sns


def strain_palette(
    N_SITES: int,
    base_palette: str = "husl",
    lightness_spread: float = 0.45,
    lightness_floor: float = 0.25,
    lightness_ceiling: float = 0.85,
) -> dict:
    """Map each `N_SITES`-bit strain to a color.

    Strains are first grouped by Hamming weight; each weight gets a
    distinct base hue from `base_palette` (an evenly-spaced ggplot-style
    HCL palette by default). Strains that share a Hamming weight are
    then disambiguated by varying lightness symmetrically around the
    base hue's lightness.
    """
    n_strains = 1 << N_SITES
    hw_groups: dict = {}
    for s in range(n_strains):
        hw = bin(s).count("1")
        hw_groups.setdefault(hw, []).append(s)

    base_colors = sns.color_palette(base_palette, N_SITES + 1)
    color_map: dict = {}
    for hw, strains in hw_groups.items():
        h, lightness, saturation = colorsys.rgb_to_hls(*base_colors[hw])
        k = len(strains)
        for i, strain in enumerate(sorted(strains)):
            if k == 1:
                color_map[strain] = tuple(base_colors[hw])
            else:
                offset = (i / (k - 1) - 0.5) * lightness_spread
                new_l = max(
                    lightness_floor,
                    min(lightness_ceiling, lightness + offset),
                )
                color_map[strain] = colorsys.hls_to_rgb(h, new_l, saturation)
    return color_map
