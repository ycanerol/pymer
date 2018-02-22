#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 10:23:51 2018

@author: ycan

http://bkanuka.com/articles/native-latex-plots/
Native Looking matplotlib Plots in LaTeX

Used for testing the font change etc.

"""

import numpy as np
import matplotlib as mpl
#mpl.use('pgf')

def figsize(scale):
    fig_width_pt = 433.62                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*golden_mean              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

pgf_with_latex = {                      # setup matplotlib to use latex for output
#    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
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
        r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
        r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
        r"\usepackage{upgreek}",
        ]
    }
mpl.rcParams.update(pgf_with_latex)

import matplotlib.pyplot as plt

# I make my own newfig and savefig functions
def newfig(width):
    plt.clf()
    fig = plt.figure(figsize=figsize(width ))
    ax = fig.add_subplot(111)
    return fig, ax

fig, ax  = newfig(0.85)

x = np.linspace(0, 200)
ax.plot(x, np.sin(x))
ax.set_title('Sinusoidal-ish wave')
ax.set_xlabel('Values for x')
ax.set_ylabel(r'Values for $\sin(x)$')
#plt.savefig('/home/ycan/Downloads/asd.pdf')
plt.show()
plt.close()

fig, ax  = newfig(.86)
import plotfuncs as plf
plf.stashow(sta, ax)
ax.set_xlabel('Time')
ax.set_ylabel('Displacement')
#plt.savefig('/home/ycan/Downloads/sta.pdf', bbox_inches='tight')
plt.show()
plt.close()
#%%