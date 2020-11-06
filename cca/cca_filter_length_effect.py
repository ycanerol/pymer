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

temp_save_folder = '/Users/ycan/Documents/gollischlab/playground/filter_lengthening'
maxframes = None

for i in range(1, 20):
    cca_omb_components(exp, stimnr, n_components=6, regularization=100, maxframes=maxframes, savedir=temp_save_folder,filter_length=(20, i))
