#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import numpy as np
import matplotlib.pyplot as plt

def gauss2d(rx, ry, width):
    def func(X, Y):
        g = np.exp(-(((rx-X)/width)**2 + ((ry-Y)/width)**2)/2)
        return g
    return func

d = 40

np.random.seed(51)

x = y = np.arange(d)
X, Y = np.meshgrid(x, y, indexing='ij')

nbases = 10
steps = int(np.ceil(d/nbases))

A = np.zeros((d, d))
W = np.random.poisson(lam=.5, size=(nbases, nbases))
W = W / W.max()
inds = np.arange(0, d, steps)
for i, ii in enumerate(inds):
    for j, jj in enumerate(inds):
#        b = gauss2d(ii+steps/2, jj+steps/2, steps/2)(X, Y)
#        allbases += b
        A += W[i, j]*gauss2d(ii+steps/2, jj+steps/2, steps/2)(X, Y)

fig = plt.figure(figsize=(8, 3.5))
ax1 = plt.subplot(1, 2, 1)
ax1.imshow(A, cmap='cubehelix')
#ax1.axis('square')


ax2 = plt.subplot(1, 2, 2)
ax2.imshow(W, cmap='Greys_r')
#plt.axis('square')
plt.suptitle('Bases for Q')
#plt.savefig('/home/ycan/Documents/meeting_notes/2018-11-20/2dbases.pdf')
#plt.savefig('/home/ycan/Documents/meeting_notes/2018-11-20/2dbases.png')
#%%
#================
#================


myW = np.zeros((nbases, nbases))
myW[4:7, 2:4] = np.ones((3, 2))

def fit2dbases(nbases, d, Wtofit):
    x = y = np.arange(d)
    X, Y = np.meshgrid(x, y, indexing='ij')

    def applyW(W):

        steps = int(np.round(d/nbases))
        A = np.zeros((d, d))
        inds = np.arange(0, d, steps)
        for i, ii in enumerate(inds):
            for j, jj in enumerate(inds):
                A += W[i, j]*gauss2d(ii+steps/2, jj+steps/2, steps/2)(X, Y)
        return A
    A = applyW(Wtofit)
    return A

W = fit2dbases(nbases, d, myW)
plt.imshow(W)
