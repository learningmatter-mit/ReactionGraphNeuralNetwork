import warnings
from csv import reader
from typing import List, Optional, Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import (
    ScalarMappable,
    cividis,
    inferno,
    magma,
    plasma,
    turbo,
    viridis,
)
from matplotlib.colors import LogNorm, Normalize, to_hex
from pandas import options
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm


PROPERTIES = {
    "center_diff": "B 3d $-$ O 2p difference",
    "op": "O 2p $-$ $E_v$",
    "form_e": "Formation energy",
    "e_hull": "Energy above hull",
    "tot_e": "Energy per atom",
    "time": "Runtime",
    "magmom": "Magnetic moment",
    "ads_e": "$E_{b}$",
    "acid_stab": "Electrochemical stability",
    "bandcenter": "DOS band center",
    "bandwidth": "DOS band width",
    "bandfilling": "Amount of Bandfilling",
    "phonon": "Atomic vibration frequency",
    "bader": "Bader charge",
    "freq": "ln($\\nu_{a}$ /THz)",
    "barrier": "$E_{b}$",
    "delta_e": "$\\Delta E$",
}

UNITS = {
    "center_diff": "eV",
    "op": "eV",
    "form_e": "eV",
    "e_hull": "eV/atom",
    "tot_e": "eV/atom",
    "time": "s",
    "magmom": "$\mu_B$",
    "ads_e": "eV",
    "acid_stab": "eV/atom",
    "bandcenter": "eV",
    "bandwidth": "eV",
    "phonon": "THz",
    "bandfilling": "$q_e$",
    "bader": "$q_e$",
    "freq": "",
    "barrier": "eV",
    "delta_e": "eV",
}


def plot_hexbin(
    targs,
    preds,
    prop_key: str,
    title="",
    num_col: int = 2,
    scale="linear",
    inc_factor=1.1,
    dec_factor=0.9,
    bins=None,
    plot_helper_lines=False,
    cmap="viridis",
    style="scifig",
):
    new_targ = targs
    new_pred = preds

    # Use style in matplotlib.style.available
    with plt.style.context(style):
        fig, ax = plt.subplots()

        mae = mean_absolute_error(new_targ, new_pred)
        r, _ = pearsonr(new_targ, new_pred)
        r_s = spearmanr(new_targ, new_pred).correlation

        if scale == "log":
            new_pred = np.abs(new_pred) + 1e-8
            new_targ = np.abs(new_targ) + 1e-8

        lim_min = min(np.min(new_pred), np.min(new_targ))
        if lim_min < 0:
            if lim_min > -0.1:
                lim_min = -0.1
            lim_min *= inc_factor
        else:
            if lim_min < 0.1:
                lim_min = -0.1
            lim_min *= dec_factor
        lim_max = max(np.max(new_pred), np.max(new_targ))
        if lim_max <= 0:
            if lim_max > -0.1:
                lim_max = 0.2
            lim_max *= dec_factor
        else:
            if lim_max < 0.1:
                lim_max = 0.25
            lim_max *= inc_factor

        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_xticks(ax.get_yticks())
        ax.set_yticks(ax.get_yticks())
        ax.set_aspect("equal")

        # ax.plot((lim_min, lim_max),
        #        (lim_min, lim_max),
        #        color='#000000',
        #        zorder=-1,
        #        linewidth=0.5)
        ax.axline((0, 0), (1, 1), color="#000000", zorder=-1, linewidth=0.5)

        hb = ax.hexbin(
            new_targ,
            new_pred,
            cmap=cmap,
            gridsize=60,
            bins=bins,
            mincnt=1,
            edgecolors=None,
            linewidths=(0.1,),
            xscale=scale,
            yscale=scale,
            extent=(lim_min, lim_max, lim_min, lim_max),
            norm=matplotlib.colors.LogNorm(),
        )

        cb = fig.colorbar(hb, shrink=0.822)
        cb.set_label("Count")

        if plot_helper_lines:
            if scale == "linear":
                x = np.linspace(lim_min, lim_max, 50)
                y_up = x + mae
                y_down = x - mae

            elif scale == "log":
                x = np.logspace(np.log10(lim_min), np.log10(lim_max), 50)

                # one order of magnitude
                y_up = np.maximum(x + 1e-2, x * 10)
                y_down = np.minimum(np.maximum(1e-8, x - 1e-2), x / 10)

                # one kcal/mol/Angs
                y_up = x + 1
                y_down = np.maximum(1e-8, x - 1)

            for y in [y_up, y_down]:
                ax.plot(x, y, color="#000000", zorder=2, linewidth=0.5, linestyle="--")

        ax.set_title(title, fontsize=8)
        if UNITS[prop_key] == "":
            ax.set_ylabel("Predicted %s" % (PROPERTIES[prop_key]), fontsize=8)
            ax.set_xlabel("Calculated %s" % (PROPERTIES[prop_key]), fontsize=8)

        else:
            ax.set_ylabel("Predicted %s [%s]" % (PROPERTIES[prop_key], UNITS[prop_key]), fontsize=8)
            ax.set_xlabel("Calculated %s [%s]" % (PROPERTIES[prop_key], UNITS[prop_key]), fontsize=8)
        # ax.set_yticks(ax.get_yticks())
        ax.set_aspect("equal")
        if num_col == 4:
            ax.annotate(
                "Pearson's r: %.3f \nSpearman's r: %.3f \nMAE: %.3f %s " % (r, r_s, mae, UNITS[prop_key]),
                (0.05, 0.75),
                xycoords="axes fraction",
                fontsize=6,
            )
        else:
            ax.annotate(
                "Pearson's r: %.3f \nSpearman's r: %.3f \nMAE: %.3f %s " % (r, r_s, mae, UNITS[prop_key]),
                (0.05, 0.85),
                xycoords="axes fraction",
                fontsize=6,
            )
        set_size(num_col=num_col, ax=ax)
    return fig, ax, r, mae, hb


def set_size(num_col, ax=None):
    """w, h: width, height in inches"""
    if num_col == 1:
        w = 5.6
        h = 5.6
    elif num_col == 2:
        w = 2.8
        h = 2.8
    elif num_col == 3:
        w = 1.867
        h = 1.867
    elif num_col == 4:
        w = 1.4
        h = 1.4
    if not ax:
        ax = plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w) / (r - l)
    figh = float(h) / (t - b)
    ax.figure.set_size_inches(figw, figh)
