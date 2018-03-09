#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 12:02:46 2018

@author: ycan

Code taken from
http://bkanuka.com/articles/native-latex-plots/
Native Looking matplotlib Plots in LaTeX

Changed use of mpl.use('pgf') based on
https://matplotlib.org/users/pgf.html
Special characters like mu are rendered correctly on saved
plots but not inline displayed ones, no idea why.

"""
import numpy as np
import matplotlib as mpl
import os

from matplotlib.backends.backend_pgf import FigureCanvasPgf
mpl.backend_bases.register_backend('pdf', FigureCanvasPgf)


def figsize(scale, aspect=None):
    fig_width_pt = 433.62                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    if aspect is None:
        aspect = golden_mean
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*aspect              # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size

pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 10,               # LaTeX default is 10pt font.
    "font.size": 10,
    "legend.fontsize": 8,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": figsize(0.9),     # default fig size of 0.9 textwidth
    "pgf.preamble": [
        r"\usepackage[utf8]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
        r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
        r"\usepackage{upgreek}",
        ]
    }
mpl.rcParams.update(pgf_with_latex)

import matplotlib.pyplot as plt

def texfig(width, aspect=None):
    plt.clf()
    fig = plt.figure(figsize=figsize(width, aspect))
    return fig

savepath = '/home/ycan/Documents/thesis/figures/'
maindir = '/home/ycan/Documents/thesis/'
# To be able to easily change all STA colormaps, in case of
# low distinguishability in printed form.
cmap = 'RdBu'

def savefig(name):
    """
    Plot name should be without extension, .pgf and .pdf versions will
    be saved as well as associated .pngs

    For pgf files, any images (like STA) are saved as .pngs and they
    have to be in the main directory for LaTeX to find them. This
    function moves any generated .pngs one directory up.
    See note at the beginning of results.tex for more details.
    """
    if name.endswith('.pdf'):
        raise ValueError('File name should be without extension.')
    # .pdf is saved for checking, the font quality is very low
    # only .pgf should be used in LaTeX document.
    plt.savefig(savepath+name+'.pdf', bbox_inches='tight')
    plt.savefig(savepath+name+'.pgf', bbox_inches='tight')
    # The generated png files are in the format fname-img<i>.png
    tomove = [i for i in os.listdir(savepath) if i.startswith(name)
                                                and i.endswith('.png')]
    [os.rename(savepath+i, maindir+i) for i in tomove]
    print('Moved following files to main directory: ')
    print(tomove)


def exclude_cells(cells):
    """
    Accepts list of tuples containing cells in the format (date, clusterid)
    Returns boolean array, that will exclude the cells indicated in the
    list when applied to the list of tuples.

    Usage with an already existing include array:
    include = np.logical_and(include, texplot.exclude_cells(cells))
    """
    exclude_file = os.path.join(maindir, 'analysis_auxillary_files',
                                'excluded_cells.txt')

    with open(exclude_file, 'r') as f:
        lines = [line.split(' #')[0] for line in f.readlines()
                            if not line.startswith('#')]
    lines = [tuple(line.split()) for line in lines if not len(line.split())==0]
    exclude = []
    for i, cell in enumerate(cells):
        for j, line in enumerate(lines):
            if np.all(lines[j] == cells[i]):
                exclude.append(i)

    indices_after_exclusion = [True if i not in exclude else False
                               for i in range(len(cells))]
    return indices_after_exclusion
