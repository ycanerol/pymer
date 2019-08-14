#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 14:24:09 2018

@author: ycan
"""
import warnings
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

import gaussfitter as gfit
import iofuncs as iof
import analysis_scripts as asc
import miscfuncs as msc

exp = '20180710'
sorted_stimuli = asc.stimulisorter(exp)
checker = sorted_stimuli['frozennoise'][0]
data = iof.load(exp, checker)
parameters = asc.read_parameters(exp, checker)

stas = data['stas']
max_inds = data['max_inds']

i = 0
sta = stas[i]
max_i = max_inds[i]
bound = 1.5

#%%
def fitgaussian(sta, f_size=10):
    max_i = np.unravel_index(np.argmax(np.abs(sta)), sta.shape)
    try:
        sta, max_i_cut = msc.cut_around_center(sta, max_i, f_size)
    except ValueError as e:
        if str(e).startswith('Frame is out'):
            raise ValueError('Fit failed.')
    fit_frame = sta[..., max_i_cut[-1]]
    # Parameters are in the format:
    # (height,amplitude,center_x,center_y,width_x,width_y,rota)
    pars = gfit.gaussfit(fit_frame)
#    f = gfit.twodgaussian(pars)
    pars_out = pars
    pars_out[2:4] = pars[2:4] - [f_size, f_size] + max_i[:2]
    return pars_out

def mahalonobis_convert(Z, pars):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', '.*divide by zero*.', RuntimeWarning)
        Zm = np.log((Z-pars[0])/pars[1])
        Zm[np.isinf(Zm)] = np.nan
        Zm = np.sqrt(Zm*-2)
    return Zm
#%%
stixelw, stixelh = parameters['stixelwidth'],parameters['stixelheight']
if stixelw != stixelh:
    ValueError('Stimulus is not checkerflicker.')

upscale_factor = stixelw
upscale_factor = 2
Y, X = np.meshgrid(np.arange(sta.shape[0]*upscale_factor),
                   np.arange(sta.shape[1]*upscale_factor))
X = X.T
Y = Y.T
plt.figure()
ax = plt.subplot(111)
all_pars = np.zeros((len(stas), 7))
masks = np.zeros((len(stas), *X.shape))


for i, _ in enumerate(data['clusters']):
    sta = stas[i]
    max_i = max_inds[i]
    try:
        pars = fitgaussian(sta)
    except ValueError as e:
        if str(e).startswith('Fit failed'):
            continue

    pars[2:6] *= upscale_factor


    all_pars[i, :] = pars
    f = gfit.twodgaussian(pars)
    Z = f(X, Y)
    Zm = mahalonobis_convert(Z, pars)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', '.*invalid value encountered in less*.', RuntimeWarning)
        masks[i] = np.where(Zm < bound, 1, 0)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        ax.contour(X, Y, Zm, [bound])
#image = mpimg.imread('/media/ycan/datadrive/data/Erol_20180207/microscope_images/afterexperiment_grid.tif')
#ax.imshow(image, zorder=-1, extent=(0, 800, 0, 600))
plt.axis('equal')
plt.show()

#%%
# Pad to 800 by 800 size
pad_height = int((masks.shape[2]-masks.shape[1])/2)
clusternr = masks.shape[0]
pad_array = np.zeros((clusternr, pad_height, masks.shape[2]))
masks = np.concatenate((pad_array, masks, pad_array), axis=1)

centers = all_pars[:, (2,3)]
widths = all_pars[:, (4,5)]

centers[:, 0] += pad_height

#%%
from matplotlib.patches import Ellipse

i = 23
ax = plt.subplot(111)
im = ax.imshow(masks[i, ...])
rf = Ellipse(tuple(centers[i, :]),
             width=widths[i][0], height=widths[i][1],
             angle=all_pars[i, -1])
ax.add_artist(rf)
plt.show()

#%%
import pandas as pd
import seaborn as sns

pars_filtered = all_pars[:, (4, 5)]
#pars_filtered = pars_filtered[pars_filtered[:, 0]<6]
#pars_filtered = pars_filtered[pars_filtered[:, 1]<6]

sizes = pd.DataFrame(data=pars_filtered, columns=['x', 'y'])

#sns.set(style='darkgrid')

g = sns.jointplot('x','y', sizes, 'scatter',
#                  shade_lowest = False,
#                  , xlim=[0, 6], ylim=[0, 6]
                  )
