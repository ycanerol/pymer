#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 16:00:51 2018

@author: ycan


This script was used to play around and optimize stripeflicker_SVD
"""
import matplotlib.pyplot as plt
from matplotlib import transforms
import numpy as np
import analysis_scripts as asc
import iofuncs as iof
import plotfuncs as plf

data = iof.load('20180207', 12)
stas = data['stas']
clusters = data['clusters']
clusterids = plf.clusters_to_ids(clusters)

choose = 4
clusterids=[clusterids[choose]]
stas = [stas[choose]]

def component(i, u, s, v):
#    v = np.dot(v, np.diag(s))
    c = np.outer(u[:, i], v[i, :])*s[i]
    return c

def sumcomponent(nr, u, s, v):
    cumulative = np.zeros((u.shape[0], v.shape[-1]))
    for comp in range(nr+1):
        cumulative += component(comp, u, s, v)
    return cumulative

for i, clusterid in enumerate(clusterids):
    sta = stas[i]
    rows = 2
    cols = 5

    plt.figure(figsize=(12, 12))
    ax1 = plt.subplot(rows, cols, 1)
    plf.stashow(sta, ax1)

    u, s, v = np.linalg.svd(sta)

    #componentnr = 1
    comp_range = 9
    sta_dn = np.zeros(sta.shape)
    for componentnr in range(comp_range):
    #u = u[:, :componentnr]
    #v = v[:componentnr, :]

    #
    #for i in range(componentnr):
    #    sta_dn += np.dot(u[:, i], v[i, :])

        sta_dn += component(componentnr, u, s, v)

        ax2 = plt.subplot(rows, cols, componentnr+2)
        plf.stashow(sta_dn, ax2)

    #plf.stashow(np.dot(u, v), plt.gca())
    ax1 = plt.subplot(rows, cols, 1)
    plf.stashow(sta, ax1)
    plt.suptitle(clusterid)
    plt.show()
    plt.close()

#%%
plt.figure(figsize=(5,5))
for i in range(40):
    ax = plt.subplot(5, 8, i+1)
    plt.plot(v[i, :])
    plf.spineless(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    vmax = np.abs(v).max()
    vmin = -vmax
    ax.set_ylim(vmin, vmax)
plt.suptitle('Temporal components')
plt.show()

plt.figure(figsize=(5,5))
for i in range(40):
    ax = plt.subplot(5, 8, i+1)
#    plt.plot(u[:, i], transform=ax.transData + transforms.Affine2D().rotate_deg(0.1))
    plt.plot(u[:, i])
    plf.spineless(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    vmax = np.abs(u).max()
    vmin = -vmax
    ax.set_ylim(vmin, vmax)
plt.suptitle('Spatial components')
plt.show()
