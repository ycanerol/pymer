#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

a = -7
c = 0.043
nbases = 15
bases_step = np.pi/2
phis = np.arange(nbases*bases_step, 0, -bases_step)
#t = np.arange(0, .6, .001)

def make_basis(t, a, c, phi):
    k = a*np.log((t+c))
    mask1 = np.where(k>(phi-np.pi))[0]
    mask2 = np.where(k<(phi+np.pi))[0]
    mask = [m for m in mask1 if m in mask2]
    basis = np.zeros(t.shape)
    basis[mask] = .5*np.cos(a*(np.log(t[mask]+c)-phi)) +.5

    return basis

def make_filter(t, *ws):
    w = np.array(ws).flatten()
    bases = np.zeros((phis.shape[0], t.shape[0]))
    for i, phi in enumerate(phis):
        bases[i] = make_basis(t, a, c, phi)
    res = bases * np.repeat(w[:, None], t.shape[0], axis=1)
    return res.sum(axis=0)

def show_bases(t=None):
    if t is None: t = np.arange(0, .6, .001)
    for phi in phis:
        plt.plot(t, make_basis(t, a, c, phi))
    plt.show()
#%%
def regularizeL2(t, sta, p0=None):
    if p0 is None:
        p0 = np.zeros(nbases)
    def cost(ws):
        model = make_filter(timev, *ws)
        reg = np.sum(np.ediff1d(ws)**2)
        reg = (np.sum(ws**2))
        regw = .1
        return np.sum((model - sta)**2) + reg*regw
    ret = minimize(cost, p0)
    return ret

#res = regularizeL2(timev, sta)
#fig = plt.figure(figsize=(8, 3.5))
#ax1=plt.subplot(1, 2, 1)
#ax1.bar(np.arange(nbases), res.x)
#ax1.set_title('Weights')
#ax2=plt.subplot(1, 2, 2)
#ax2.plot(make_filter(timev, res.x))
#ax2.plot(sta)
#plt.suptitle('Regularized with L2')
#ax2.legend(['Fitted filter', 'STA'])
#plt.savefig(saveloc+'regularizedL2.pdf')
#plt.savefig(saveloc+'regularizedL2.png')
#%%
def regularizeL1(t, sta, p0=None):
    if p0 is None:
        p0 = np.zeros(nbases)
    def cost(ws):
        model = make_filter(timev, *ws)
        reg = np.sum(np.ediff1d(ws)**2)
        reg = (np.sum(np.abs(ws)))
        regw = .1
        return np.sum((model - sta)**2) + reg*regw
    ret = minimize(cost, p0)
    return ret

#res = regularizeL1(timev, sta)
#fig = plt.figure(figsize=(8, 3.5))
#ax1=plt.subplot(1, 2, 1)
#ax1.bar(np.arange(nbases), res.x)
#ax1.set_title('Weights')
#ax2=plt.subplot(1, 2, 2)
#ax2.plot(make_filter(timev, res.x))
#ax2.plot(sta)
#plt.suptitle('Regularized with L1')
#ax2.legend(['Fitted filter', 'STA'])
#plt.savefig(saveloc+'regularizedL1.pdf')
#plt.savefig(saveloc+'regularizedL1.png')


#%%
fig = plt.figure()
ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)
t_1 = np.arange(0, .6, .001)
t_2 = np.arange(0, .6, .016)
for i, (ax, t) in enumerate(zip([ax1, ax2], [t_1, t_2])):
    for phi in phis:
        ax.plot(t, make_basis(t, a, c, phi))
        ax.set_yticks([0, 1])
        ax.set_ylabel(['dt=1 ms', 'dt=16 ms'][i])
#        ax2.plot(t_2, make_basis(t_2, a, c, phi))
ax2.set_xlabel('Time [s]')
fig.suptitle('Cosine bases for linear filters')
#plt.savefig(saveloc+'bases.pdf')
#plt.savefig(saveloc+'bases.png')



#%%

w = np.zeros(nbases)
w[-5] = -.3
w[-4] = .5

#filt = make_filter(t, w)

#plt.plot(t, filt)
#plt.show()

#%%
import iofuncs as iof
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

data = iof.load('20180710', 1)
stas = data['stas']
bars = np.arange(nbases)
barkw = {'width':0.2}


for i, sta in enumerate(stas):

    timev = np.arange(0, data['frame_duration']*data['filter_length'], data['frame_duration'])
    w0=np.zeros(nbases)

    popt, pcov = curve_fit(make_filter, timev, sta, p0=w0, bounds=[-2, 2])

    fig = plt.figure(figsize=(8, 3.5))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    ax1.bar(bars+1/4, popt, **barkw)
    ax1.set_title('Weights')
    ax2.plot(timev, make_filter(timev, *popt), label='no reg.')

    resL2 = regularizeL2(timev, sta)
    ax1.bar(bars+2/4, resL2.x, **barkw)
    ax2.plot(timev, make_filter(timev, resL2.x), label='L2')

    resL1 = regularizeL1(timev, sta)
    ax1.bar(bars+3/4, resL1.x, **barkw)
    ax2.plot(timev, make_filter(timev, resL1.x), label='L1')


    ax2.plot(timev, sta, label='STA')
    ax2.legend()

    plt.title(i)
    plt.show()
    break
#ax2.legend(['Fitted filter', 'STA'])
#plt.suptitle('No regularization')
#plt.savefig(saveloc+'notregularized.pdf')
#plt.savefig(saveloc+'notregularized.png')