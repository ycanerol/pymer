#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions for implementing Generalized Linear Model
"""
import numpy as np
from scipy.optimize import minimize

import analysis_scripts as asc

from genlinmod import conv

stimdim = None
filter_length = None


def set_stimdim(stimdim_toset):
    global stimdim
    stimdim = stimdim_toset


def set_filter_length(filter_length_toset):
    global filter_length
    filter_length = filter_length_toset


def flattenpars(k, mu):
    """
    Flatten a set of parameters to be used with optimization
    functions.

    Inverse operation of splitpars.
    """
    kmu = np.concatenate((k.ravel(), [mu]))
    return kmu


def splitpars(kmu):
    global stimdim
    k = kmu[:-1]
    mu = kmu[-1]
    k = k.reshape((stimdim, -1))
    return k, mu


def glm_in(k, mu):
    """
    Given a set of filters, calculate the time series that goes into
    the exponential function.
    """
    def f(x):
        global stimdim
        if stimdim is None:
            if x.ndim > 1:
                stimdim = x.shape[0]
            else:
                stimdim = 1
        total = 0
        for j in range(stimdim):
            total += conv(k[j, :], x[j, :])
        return total + mu
    return f


def glm_neuron(k, mu, time_res):
    def firing_rate(x):
        return np.exp(glm_in(k, mu)(x))*time_res
    return firing_rate


def loglhd(kmu, x, spikes, time_res):
    global stimdim
    if stimdim is None:
        if x.ndim > 1:
            stimdim = x.shape[0]
        else:
            stimdim = 1
    k, mu = splitpars(kmu)
    nlt_in = glm_in(k, mu)(x)
    return -np.sum(spikes * nlt_in) + time_res*np.sum(np.exp(nlt_in))


def grad(kmu, x, spikes, time_res):
    k, mu = splitpars(kmu)
    nlt_in = glm_in(k, mu)(x)
    xr = asc.rolling_window(x, filter_length)[..., ::-1]
    dldk = spikes@xr - time_res*np.exp(nlt_in)@xr
#        dldk = (np.einsum('j,mjk->mk', spikes, xr) -
#                time_res * np.einsum('j,mjk->mk', nlt_in, xr))
#        dldk2 = np.zeros(l)
#        for i in range(len(spikes)):
#            dldk2 += spikes[i] * xr[i, :]
#            dldk2 -= time_res*np.exp(nlt_in[i])*xr[i, :]
#        assert np.isclose(dldk, dldk2).all()
#        import pdb; pdb.set_trace()
    dldm = spikes.sum() - time_res*np.exp(nlt_in).sum()
    dl = flattenpars(dldk, dldm)
    return -dl


def minimize_loglhd(k_initial, mu_initial, x, time_res, spikes, usegrad=True,
                    method='Newton-CG', **kwargs):
    """
    Calculate the filters that minimize the log likelihood function for a
    given set of spikes and stimulus.

    Parameters
    --------
    k_initial, mu_initial:
        Initial guesses for the parameters.
    x:
        The stimulus
    time_res:
        Length of each bin (referred also as Delta, frame_duration)
    spikes:
        Binned spikes, must have the same shape as the stimulus
    usegrad:
        Whether to use gradients for optimiziation. If set to False, only
        approximated gradients will be used with the appropriate optimization
        method.
    debug_grad:
        Whether to calculate and plot the gradients in the first iteration
        Setting it to True will change the returned values.
    method:
        Optimization method to use, see the Notes section in the  documentation of
        scipy.minimize for a full list.
    """
    minimizekwargs = {'method': method,
                      'tol': 1e-3,
                      }
    minimizekwargs.update(**kwargs)

    global filter_length, stimdim
    if filter_length is None:
        filter_length = k_initial.shape[-1]

    if stimdim is None:
        if x.ndim > 1:
            stimdim = x.shape[0]
        else:
            stimdim = 1

    if usegrad:
        minimizekwargs.update({'jac': grad})

    res = minimize(loglhd, flattenpars(k_initial, mu_initial), args=(x, spikes, time_res), **minimizekwargs)

    return res


def normalizestas(stas):
    #This function is obsolete, make sure it's not used
    raise ValueError('Normalization of STAs should not be used')
#    stas = np.array(stas)
#    b = .abs(stas).max(axis=1)
#    stas_normalized = stas / b.repeat(stas.shape[1]).reshape(stas.shape)
#    return stas_normalized

#%%
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    stimdim = 3
    filter_length = 20
    frame_rate = 60
    time_res = 1/frame_rate
    tstop = 100
    t = np.arange(0, tstop, time_res)

    np.random.seed(31)
    stim = np.random.normal(size=(stimdim, t.shape[0]))

    k = np.zeros((stimdim, filter_length))
    mu = np.random.random()

    k[0, 3:7] = [0.5, 1, 0, -1]
    k[1, 4:9] = [-0.5, -1, 0, .2, .8]

    myglmneuron = glm_neuron(k, mu, time_res)
    spikes = np.random.poisson(myglmneuron(stim))

    res = minimize_loglhd(np.zeros((stimdim, filter_length)), 0, stim, time_res,
                          spikes, method='BFGS', usegrad=True)
#%%
    k_res, mu_res = splitpars(res.x)
    fig, axes = plt.subplots(stimdim, 1, sharex=True, sharey=True)
    for i, ax in enumerate(axes):
        ax.plot(k[i, :])
        ax.plot(k_res[i, :])