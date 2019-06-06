#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import numpy as np
import analysis_scripts as asc
import iofuncs as iof


def calcstafast(stim, filter_length, spikes):
    sta = np.einsum('a,ab->b', spikes, asc.rolling_window(stim, filter_length))
    sta /= spikes.sum()
    return sta


def calcsta(stim, filter_length, spikes):
    """
    One cell, checkerflicker
    """
    sta = np.einsum('abcd,ec->eabd', asc.rolling_window(stim, filter_length),
                    spikes)
    sta /= spikes.sum()
    return sta


def calcallstas(stim, filter_length, allspikes):
    """
    Multiple cells, three stimulus dimensions (2 space, 1 time i.e. checkerflicker)
    """
    rw = asc.rolling_window(stim, filter_length)
    stas = np.einsum('abcd,ec->eabd', rw, allspikes)
    stas /= allspikes.sum(axis=(-1))[:, np.newaxis,
                                      np.newaxis, np.newaxis]
    return stas


def tensorsta(stim, filter_length, allspikes):
    stastensor = np.tensordot(
                     allspikes,
                     asc.rolling_window(stim, 20),
                     [(1), (2)])

#%%
if __name__ == '__main__':
    import genlinmod as glm

    exp, stimnr = '20180710', 11
    data = iof.load(exp, stimnr)
    raster = asc.read_raster(exp, stimnr, 1, 1)
    filter_length, ft = asc.ft_nblinks(exp, stimnr)
    lim = 10000
    spikes = asc.binspikes(raster, ft)[:lim]
    stim = glm.loadstim(exp, stimnr, maxframenr=lim)


#    sta = calcsta(stim, filter_length, spikes)

    stas = calcsta(stim, filter_length, np.repeat(spikes, 2).reshape(2, spikes.shape[0]))
