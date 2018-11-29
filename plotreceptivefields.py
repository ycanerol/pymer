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

exp = '20171122'
sorted_stimuli = asc.stimulisorter(exp)
checker = sorted_stimuli['checkerflicker'][0]
data = iof.load(exp, checker)

stas = data['stas']
max_inds = data['max_inds']

i = 0
sta = stas[i]
max_i = max_inds[i]
bound = 1.5

#%%
def fitgaussian(sta, bound=2, f_size=10):
    max_i = np.unravel_index(np.argmax(np.abs(sta)), sta.shape)
    try:
        sta, max_i_cut = msc.cut_around_center(sta, max_i, f_size)
    except ValueError as e:
        if str(e).startswith('Frame is out'):
            raise ValueError('Fit failed.')
    fit_frame = sta[..., max_i_cut[-1]]
    pars = gfit.gaussfit(fit_frame)
#    f = gfit.twodgaussian(pars)
    pars_out = pars
    pars_out[2:4] = pars[2:4] - [f_size, f_size] + max_i[:2]
    return pars_out

def mahalonobis_convert(Z, pars):
    warnings.filterwarnings('ignore', '.*divide by zero*.', RuntimeWarning)
    with warnings.catch_warnings():
        Zm = np.log((Z-pars[0])/pars[1])
        Zm[np.isinf(Zm)] = np.nan
        Zm = np.sqrt(Zm*-2)
    return Zm
#%%
X, Y = np.meshgrid(np.arange(sta.shape[0]),
                   np.arange(sta.shape[1]))
ax = plt.subplot(111)
all_pars = np.zeros((len(stas), 7))

for i, _ in enumerate(data['clusters']):
    sta = stas[i]
    max_i = max_inds[i]
    try:
        pars = fitgaussian(sta, bound)
    except ValueError as e:
        if str(e).startswith('Fit failed'):
            continue
    all_pars[i, :] = pars
    f = gfit.twodgaussian(pars)
    Z = f(Y, X)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', '.*divide by zero*.', RuntimeWarning)
        Zm = np.log((Z-pars[0])/pars[1])
        Zm[np.isinf(Zm)] = np.nan
        Zm = np.sqrt(Zm*-2)
#    plf.stashow(sta[..., max_i[-1]], ax)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        ax.contour(Y, X, Zm, [bound])
plt.show()
#image = mpimg.imread('/media/ycan/datadrive/data/Erol_20180207/microscope_images/afterexperiment_grid.tif')
#ax.imshow(image)
#plt.show()

#%%
import pandas as pd
import seaborn as sns

pars_filtered = all_pars[:, (4, 5)]
pars_filtered = pars_filtered[pars_filtered[:, 0]<6]
pars_filtered = pars_filtered[pars_filtered[:, 1]<6]

sizes = pd.DataFrame(data=pars_filtered, columns=['x', 'y'])

sns.set(style='darkgrid')

g = sns.jointplot('x','y', sizes, 'kde', xlim=[0, 6], ylim=[0, 6])
