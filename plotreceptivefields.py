#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 14:24:09 2018

@author: ycan
"""
import warnings
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Ellipse

import numpy as np

import gaussfitter as gfit
import iofuncs as iof
import analysis_scripts as asc
import miscfuncs as msc

exp = '20171122'
sorted_stimuli = asc.stimulisorter(exp)
#checker = sorted_stimuli['frozennoise'][0]
checker = sorted_stimuli['checkerflicker'][0]
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
def drawellipse(pars, bound, ax=None):
    if ax is None:
        ax = plt.gca()
    center = pars[2:4][::-1]
    width_x = pars[5]*2 # HINT: magic number, possibly due to diameter vs radius
    width_y = pars[4]*2 # difference in the arguments accepted by Ellipse and gaussfit

    width_x *= bound
    width_y *= bound

    angle = pars[-1]
    rf = Ellipse(tuple(center), width_x, width_y, angle,
                 fill=False, edgecolor='green', lw=1)
    ax.add_artist(rf)
    return rf

#%%
stixelw, stixelh = parameters['stixelwidth'],parameters['stixelheight']
if stixelw != stixelh:
    ValueError('Stimulus is not checkerflicker.')

Y, X = np.meshgrid(np.arange(sta.shape[0]),
                   np.arange(sta.shape[1]), indexing='xy')
plt.figure()
ax = plt.subplot(111)
ax.axis('equal')
all_pars = np.zeros((len(stas), 7))

for i, _ in enumerate(data['clusters']):
    sta = stas[i]
    max_i = max_inds[i]
    try:
        pars = fitgaussian(sta)
    except ValueError as e:
        if str(e).startswith('Fit failed'):
            continue
    all_pars[i, :] = pars
    f = gfit.twodgaussian(pars)
    Z = f(X, Y)
    Zm = mahalonobis_convert(Z, pars)

    drawellipse(pars, bound, ax)

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
#        ax.contour(X, Y, Zm, [bound])
plt.axis([0, sta.shape[1], 0, sta.shape[0]])
	#image = mpimg.imread('/media/ycan/datadrive/data/Erol_20180207/microscope_images/afterexperiment_grid.tif')
#ax.imshow(image)
#plt.show()
#plt.savefig(f'/home/ycan/Downloads/TAC_outgoingfiles/RFs_{exp}.svg')

#%%
import plotfuncs as plf
stas = np.array(stas)
fig, sl = plf.multistabrowser(stas)
for i in range(stas.shape[0]):
    ax = fig.axes[i]
    drawellipse(all_pars[i], 1.5, ax)


#%%
import pandas as pd
import seaborn as sns

pars_filtered = all_pars[:, (4, 5)]
#pars_filtered = pars_filtered[pars_filtered[:, 0]<5]
#pars_filtered = pars_filtered[pars_filtered[:, 1]<5]

sizes = pd.DataFrame(data=pars_filtered, columns=['x', 'y'])

#sns.set(style='darkgrid')

g = sns.jointplot('x','y', sizes, 'scatter',
#                  shade_lowest = False,
#                  , xlim=[0, 6], ylim=[0, 6]
                  )


#%%
import plotfuncs as plf

plf.absmax() # stop the script

for i in range(len(stas)):
    sta = stas[i]
    plt.imshow(sta[..., max_inds[i][-1]], cmap='RdBu_r',
              vmax=asc.absmax(sta), vmin=asc.absmin(sta))
    drawellipse(all_pars[i])
    plt.title(f'{i}')
    plt.show()

#%%
fig, sl = plf.multistabrowser(stas)
for i in range(len(stas)):
    ax = fig.axes[i]

    drawellipse(all_pars[i], ax)
