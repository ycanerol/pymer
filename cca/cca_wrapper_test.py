
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import rcca

from matplotlib.ticker import MaxNLocator

import analysis_scripts as asc
import model_fitting_tools as mft
import spikeshuffler
import nonlinearity as nlt
import plotfuncs as plf
from omb import OMB

from cca_wrapper import cca_omb_components

whiten = False



exp ='20180710_kilosorted'
stim_nr = 8
n_components = 6
regularization = 1000
filter_length = (20, 20)
maxframes = None
select_cells = None
shufflespikes = False
whiten = False
savedir = None
exclude_allzero_spike_rows = True

cca_solver = 'macke'

cca_omb_components(exp, stim_nr, n_components, regularization,
                    filter_length, cca_solver, whiten=whiten)
