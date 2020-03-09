#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read the number of sample points for each stimulus to use with
kilosort

Disused since bininfo.mat already includes this information
"""

import os
from fnmatch import fnmatch
import numpy as np
import neuroshare as ns

folder = '/media/ycan/Erol1/20180802_YE_252MEA_Marmoset_eye1_421/'

nsamples = []
fnames = []
stimnrs = []
for file in os.listdir(folder):
    if fnmatch(file, '*.mcd'):
        f = ns.File(os.path.join(folder, file))
        nsamples.append(int(round(f.time_span / f.time_stamp_resolution)))
        fnames.append(file)
        stimnrs.append(file.split('_')[0])

sortind = np.argsort(np.array(stimnrs, dtype=int))
nsamples = np.array(nsamples)[sortind]
fnames = np.array(fnames)[sortind]
cumsamples = np.cumsum(nsamples)

for i in range(len(stimnrs)):
    print(f'{cumsamples[i]:9.0f} {nsamples[i]:8.0f} {fnames[i]}')



