#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import matplotlib.pyplot as plt

import analysis_scripts as asc
import plotfuncs as plf

from omb import OMB

exp, ombstimnr = '20180710', 8

checkerstim = 6

st = OMB(exp, ombstimnr,
         maxframes=10000
         )

all_spikes = np.zeros((st.nclusters, st.ntotal))

for i in range(st.nclusters):
    all_spikes[i, :] = st.binnedspiketimes(i)[:-1]

filter_length = 20

contrast = st.generatecontrast(st.texpars.noiselim/2, 50, filter_length-1)
contrast_avg = contrast.mean(axis=-1)

rw = asc.rolling_window(contrast, filter_length, preserve_dims=False)
#rws = rw.transpose((2, 0, 1, 3))

stas = np.einsum('abcd,ec->eabd', rw, all_spikes)
stas = stas / all_spikes.sum(axis=1)[:, None, None, None]

from scratch_spikeshuffler import shufflebyrow

shuffled_spikes = shufflebyrow(all_spikes)

shuffled_stas = np.einsum('abcd,ec->eabd', rw, shuffled_spikes)
shuffled_stas = shuffled_stas / all_spikes.sum(axis=1)[:, None, None, None]
#shuffled_stas_normalized = asc.normalize(shuffled_stas, axis_inv=0)