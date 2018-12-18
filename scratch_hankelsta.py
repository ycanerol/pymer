#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import numpy as np
from scipy.linalg import hankel

import analysis_scripts as asc
import genlinmod as glm

exp = '20180207'
stim = 4
x = glm.loadstim(exp, stim)
filter_length, frametimes =  asc.ft_nblinks(exp, stim)

xh = hankel(x)[:, :filter_length]

clusters = asc.read_spikesheet(exp)[0]

allsp = np.zeros((clusters.shape[0], frametimes.shape[0]))

for i, (ch, cl, _) in enumerate(clusters):
    sp = asc.read_raster(exp, stim, ch, cl)
    allsp[i, :] = asc.binspikes(sp, frametimes)

import matplotlib.pyplot as plt

imskw = {'vmin':x.min(), 'vmax':x.max()}

#%%
xh = xh[:-filter_length+1]
allsp = allsp[:, filter_length-1:]


stas = np.matmul(allsp, xh)
stas /= allsp.sum(axis=1)[:, np.newaxis]
plt.imshow(stas)
#%%
# Sort them
maxinds = np.argmax(np.abs(stas), axis=1)
sortedargs = np.argsort(stas[np.arange(stas.shape[0]), maxinds])

sortedstas = stas[sortedargs, :]

plt.imshow(sortedstas)

#%%

xstr = rolling_window(x, filter_length)

(xstr == xh).all()
