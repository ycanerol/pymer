import sys
import numpy as np
from pathlib import Path
sys.path.append( str(Path.home() / 'repos/pymer/'))
sys.path.append( str(Path.home() / 'repos/pymer/cca/'))
sys.path.append( str(Path.home() / 'repos/pymer/modules/'))
sys.path.append( str(Path.home() / 'repos/pymer/classes/'))
sys.path.append( str(Path.home() / 'repos/pymer/external_libs/'))

import cca_withpyrcca
import importlib
importlib.reload(cca_withpyrcca)

cca_omb_components = cca_withpyrcca.cca_omb_components

exp, stimnr = '20180710_kilosorted', 8

# Cells have the most spikes
nsptop20 = np.array([31, 47, 14, 74, 16, 54, 64, 76, 46, 13, 35, 62, 72, 59,  2, 50, 27,
                   29, 78, 22])

nsp_wo_bottom30 = np.array([31, 47, 14, 74, 16, 54, 64, 76, 46, 13, 35, 62, 72, 59,  2, 50, 27,
       29, 78, 22, 15, 56, 26, 86, 69, 41, 48, 84, 57, 89,  5, 38, 44,  1,
       ])

temp_save_folder = '/Users/ycan/Documents/gollischlab/playground/'
maxframes = None

#%%
cca_omb_components(exp, stimnr, n_components=6, regularization=100, maxframes=maxframes, savedir=temp_save_folder,)
cca_omb_components(exp, stimnr, n_components=6, regularization=100, maxframes=maxframes, savedir=temp_save_folder, shufflespikes=True)
cca_omb_components(exp, stimnr, n_components=6, filter_length=(20, 2), regularization=100, maxframes=maxframes, savedir=temp_save_folder,)
cca_omb_components(exp, stimnr, n_components=6, filter_length=(20, 2), regularization=100, maxframes=maxframes, savedir=temp_save_folder,shufflespikes=True)

#%%
cca_omb_components(exp, stimnr, n_components=6, filter_length=(20, 6), regularization=100, maxframes=maxframes, savedir=temp_save_folder,)
cca_omb_components(exp, stimnr, n_components=6, filter_length=(20, 6), regularization=100, maxframes=maxframes, savedir=temp_save_folder,shufflespikes=True)
#%%
cca_omb_components(exp, stimnr, n_components=6, filter_length=(6, 2), regularization=1000, maxframes=maxframes, savedir=temp_save_folder,)
cca_omb_components(exp, stimnr, n_components=6, filter_length=(6, 2), regularization=1000, maxframes=maxframes, savedir=temp_save_folder,shufflespikes=True)
#%%
cell_pairs = [
            # [74, 76], # ON-ON, midget-midget
             # [34, 37], # OFF-OFF, midget-midge
            # [58, 60], # OFF-OFF parasol-parasol
            [1, 22] # ON-ON parasol-parasol
            ]


for selected in cell_pairs:
    cca_omb_components(exp, stimnr, n_components=6, filter_length=(6, 4), regularization=1000, maxframes=maxframes, savedir=temp_save_folder,select_cells=selected)
    cca_omb_components(exp, stimnr, n_components=6, filter_length=(6, 4), regularization=1000, maxframes=maxframes, savedir=temp_save_folder,select_cells=selected, shufflespikes=True,)

#%%
# without the cells with few spikes

cca_omb_components(exp, stimnr, n_components=6, filter_length=(6, 2), regularization=1000, maxframes=maxframes, savedir=temp_save_folder, select_cells=nsp_wo_bottom30)
cca_omb_components(exp, stimnr, n_components=6, filter_length=(6, 2), regularization=1000, maxframes=maxframes, savedir=temp_save_folder,shufflespikes=True, select_cells=nsp_wo_bottom30)

cca_omb_components(exp, stimnr, n_components=6, filter_length=(6, 4), regularization=1000, maxframes=maxframes, savedir=temp_save_folder, select_cells=nsp_wo_bottom30)
cca_omb_components(exp, stimnr, n_components=6, filter_length=(6, 4), regularization=1000, maxframes=maxframes, savedir=temp_save_folder,shufflespikes=True, select_cells=nsp_wo_bottom30)

