#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 11:54:47 2018

@author: ycan
"""
import iofuncs as iof
import plotfuncs as plf
import numpy as np
import miscfuncs as msc
import matplotlib.pyplot as plt
import gaussfitter as gfit
#%%

def stashow(frame, ax):
    vmax = np.abs(frame).max()
    vmin = -vmax
    im = ax.imshow(frame, cmap='RdBu', vmin=vmin, vmax=vmax)
    plf.spineless(ax)
    plf.colorbar(im, size='2%', ticks=[vmin, vmax], format='%.2f')
    return im

def getfit(frame):
    if np.max(frame) != np.max(np.abs(frame)):
        onoroff = -1
    else:
        onoroff = 1

    pars = gfit.gaussfit(frame*onoroff)
    f = gfit.twodgaussian(pars)
    return f, pars, onoroff

def mahalonobis_convert(Z, pars):
    Zm = np.log((Z-pars[0])/pars[1])
    Zm[np.isinf(Zm)] = np.nan
    Zm = np.sqrt(Zm*-2)
    return Zm

#%%
exp_name = 'V'
stim_nr = 10

data = iof.load(exp_name, stim_nr)

clusters = data['clusters']
max_inds = data['max_inds']
stas = data['stas']
stx_h = data['stx_h']
exp_name = data['exp_name']
stimname = data['stimname']
max_inds = data['max_inds']
frame_duration = data['frame_duration']
filter_length = data['filter_length']
quals = data['quals'][-1, :]

spikenrs = np.array([a.sum() for a in data['all_spiketimes']])

clusterids = plf.clusters_to_ids(clusters)

# Determine frame size so that the total frame covers
# an area large enough i.e. 2*700um
px_size = 7.5
f_size = int(700/(stx_h*px_size))

i = 63
#%%
sta = stas[i]
max_i = max_inds[i]
fit_frame = sta[:, :, max_i[2]]

stac, max_i = msc.cut_around_center(sta, max_i, f_size+2)

sp1, sp2, t1, t2, _, _ = msc.svd(stac)

#%%
plt.figure(figsize=(12,10))
vmax = np.max(np.abs([sp1, sp2]))
vmin = -vmax
plt.subplot(131)
plt.imshow(sp1, cmap = 'RdBu', vmin=vmin, vmax=vmax)
plt.subplot(132)
plt.imshow(sp2, cmap = 'RdBu', vmin=vmin, vmax=vmax)
plt.subplot(133)
im = plt.imshow(fit_frame, cmap = 'RdBu', vmin=vmin, vmax=vmax)
plf.colorbar(im, size='2%', ticks=[vmin, vmax], format='%.2f')
plt.show()

sp1c, maxic = msc.cut_around_center(sp1, max_i, f_size)
sp2c, maxic = msc.cut_around_center(sp2, max_i, f_size)
fit_framec , maxic = msc.cut_around_center(fit_frame, max_i, f_size)

plt.figure(figsize=(12,10))
vmax = np.max(np.abs([sp1c, sp2c]))
vmin = -vmax
plt.subplot(131)
plt.imshow(sp1c, cmap = 'RdBu', vmin=vmin, vmax=vmax)
plt.subplot(132)
plt.imshow(sp2c, cmap = 'RdBu', vmin=vmin, vmax=vmax)
plt.subplot(133)
im = plt.imshow(fit_framec, cmap = 'RdBu', vmin=vmin, vmax=vmax)
plf.colorbar(im, size='2%', ticks=[vmin, vmax], format='%.2f')
plt.show()

#%%
sp = sp1c

rows = 2
columns = 2
plt.figure(figsize=(12,10))

f, pars0, pol0 = getfit(sp)

X, Y = np.meshgrid(np.arange(sp.shape[0]),
                   np.arange(sp.shape[1]))
Z0 = f(Y, X)
Z0m = mahalonobis_convert(Z0, pars0)

ax0 = plt.subplot(rows, columns, 1)
stashow(sp, ax0)
ax0.contour(X, Y, Z0m, [2])

d1 = sp - Z0*pol0
f1, pars1, pol1 = getfit(d1)
Z1 = f1(Y, X)
Z1m = mahalonobis_convert(Z1, pars1)

ax1 = plt.subplot(rows, columns, 2)
stashow(d1, ax1)
ax1.contour(X, Y, Z1m, [1.4])

d2 = sp - Z1*pol1
f2, pars2, pol2 = getfit(d2)
Z2 = f2(Y, X)
Z2m = mahalonobis_convert(Z2, pars2)
ax2 = plt.subplot(rows, columns, 3)
stashow(d2, ax2)
ax2.contour(X, Y, Z2m, [2])
plt.show()
#%%
plt.figure(figsize=(20,15))
rec = Z0*pol0+Z1*pol1*2

ax3 = plt.subplot(221)
stashow(rec, ax3)

ax4 = plt.subplot(222)
stashow(sp, ax4)

ax5 = plt.subplot(223)
stashow(sp-rec, ax5)
