import sys
import numpy as np
from pathlib import Path
sys.path.append( str(Path.home() / 'repos/pymer/'))
sys.path.append( str(Path.home() / 'repos/pymer/cca/'))
sys.path.append( str(Path.home() / 'repos/pymer/modules/'))
sys.path.append( str(Path.home() / 'repos/pymer/classes/'))
sys.path.append( str(Path.home() / 'repos/pymer/external_libs/'))
import analysis_scripts as asc
import cca_withpyrcca
import importlib
importlib.reload(cca_withpyrcca)

cca_omb_components = cca_withpyrcca.cca_omb_components

exps = ['20180712*kilosorted', '20180719_kilosorted', '20180815*_kilosorted']
exp_stimnr_pairs = []
# Find all the OMB stimuli in the experiment
for exp in exps:
    ombstimnrs = asc.stimulisorter(exp)['OMB']
    for ombstimnr in ombstimnrs:
        exp_stimnr_pairs.append((exp, ombstimnr))
# exp, stimnr = '20180815*_kilosorted', 10
#%%
for exp, stimnr in exp_stimnr_pairs:

    temp_save_folder = None
    maxframes = None

    cca_omb_components(exp, stimnr, n_components=6, regularization=100, maxframes=maxframes, savedir=temp_save_folder,)
    cca_omb_components(exp, stimnr, n_components=6, regularization=100, maxframes=maxframes, savedir=temp_save_folder, shufflespikes=True)

    cca_omb_components(exp, stimnr, n_components=6, filter_length=(20, 2), regularization=100, maxframes=maxframes, savedir=temp_save_folder,)
    cca_omb_components(exp, stimnr, n_components=6, filter_length=(20, 2), regularization=100, maxframes=maxframes, savedir=temp_save_folder,shufflespikes=True)

    cca_omb_components(exp, stimnr, n_components=6, filter_length=(20, 6), regularization=100, maxframes=maxframes, savedir=temp_save_folder,)
    cca_omb_components(exp, stimnr, n_components=6, filter_length=(20, 6), regularization=100, maxframes=maxframes, savedir=temp_save_folder,shufflespikes=True)

    cca_omb_components(exp, stimnr, n_components=6, filter_length=(6, 2), regularization=1000, maxframes=maxframes, savedir=temp_save_folder,)
    cca_omb_components(exp, stimnr, n_components=6, filter_length=(6, 2), regularization=1000, maxframes=maxframes, savedir=temp_save_folder,shufflespikes=True)
    #cell_pairs = [
    #            # [74, 76], # ON-ON, midget-midget
    #             # [34, 37], # OFF-OFF, midget-midge
    #            # [58, 60], # OFF-OFF parasol-parasol
    #            [1, 22] # ON-ON parasol-parasol
    #            ]


    #for selected in cell_pairs:
    #    cca_omb_components(exp, stimnr, n_components=6, filter_length=(6, 4), regularization=1000, maxframes=maxframes, savedir=temp_save_folder,select_cells=selected)
    #    cca_omb_components(exp, stimnr, n_components=6, filter_length=(6, 4), regularization=1000, maxframes=maxframes, savedir=temp_save_folder,select_cells=selected, shufflespikes=True,)
