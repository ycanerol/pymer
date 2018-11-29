#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 09:50:39 2018

@author: ycan

Estimate linear filters from white noise stimuli without
using reverse correlation

Using generalized linear models

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize


def make_noise(n=5000):
    return np.random.rand(n)

#def basis(a, c, phi):
##   def k(t): a*np.log((t+c)-phi)
#    if k(t) > phi-np.pi and k(t) < phi+np.pi:
#        return lambda t:.5*np.cos(a*(np.log(t+c)-phi)) + .5
#    else:
#        return lambda t:0
#    return f

def fu(t, a, c, phi):
    k = a*np.log((t+c))
    mask1 = np.where(k>(phi-np.pi))[0]
    mask2 = np.where(k<(phi+np.pi))[0]
    mask = [m for m in mask1 if m in mask2]
#    print(mask)
    basis = np.zeros(t.shape)
#    print(t[mask].shape)
    basis[mask] = .5*np.cos(a*(np.log(t[mask]+c)-phi)) +.5
#    def f(t):
#        if t in mask:
#            return lambda t:0
#        else:
#            return lambda t:.5*np.cos(a*(np.log(t+c)-phi)) + .5
#    basis = f(t)
    return basis

def basis1(a, c, phi):
    f = lambda t:.5*np.cos(a*(np.log(t+c)-phi)) + .5
    return f

nbases = 15

# Somehow bases_step and a have to be in a certain relataionship
# (e.g. pi/2, -7) to get a sensible set of bases,
# otherwise the cosine functions are truncated at inappropriate
# locations
bases_step = np.pi/2
a = -7
c = .043


phis = np.arange(0, nbases*bases_step, bases_step)

t = np.arange(0, .6, .001)
weights = [.02, -.1, 0, 0, 0, -.2, -.1, 0, .5, .5, .1, .1, .1, .1]
for phi in phis:
#    f = basis1(a, c, phi)
    f1 = fu(t, a, c, phi)
#    plt.plot(t, f(t));
    plt.plot(t, f1)
plt.show()

def construct_filter(weights, phis):
    f = np.zeros(t.shape)
    for phi, w in zip(phis, weights):
        f += fu(t, a, c, phi)*w
    return f

plt.plot(t, construct_filter(weights, phis))
plt.ylim([-1, 1])
