#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions for implementing Generalized Linear Model
"""
import numpy as np
from scipy.optimize import minimize

from randpy import randpy
import analysis_scripts as asc
import iofuncs as iof

def loadstim(exp, stim_nr, maxframenr=10000):
    """
    Recreate the stimulus based on the seed for a given stimulus type.

    Each type of stimulus requires a different way of handling the
    random numbers from the PRNG.
    """
    sortedstim = asc.stimulisorter(exp)
    clusters, metadata = asc.read_spikesheet(exp)
    pars = asc.read_parameters(exp, stim_nr)

    for key, val in sortedstim.items():
        if stim_nr in val:
            stimtype = key
    if stimtype in ['fff', 'stripeflicker', 'checkerflicker']:
        seed = pars.get('seed', -10000)
        bw = pars.get('blackwhite', False)
        filter_length, frametimings = asc.ft_nblinks(exp, stim_nr)
        total_frames = frametimings.shape[0]

        if stimtype == 'fff':
            if bw:
                randnrs, seed = randpy.ranb(seed, total_frames)
                # Since ranb returns zeros and ones, we need to convert
                # the zeros into -1s.
                stimulus = np.array(randnrs) * 2 - 1
            else:
                randnrs, seed = randpy.gasdev(seed, total_frames)
                stimulus = np.array(randnrs)
        elif stimtype == 'checkerflicker':
            scr_width = metadata['screen_width']
            scr_height = metadata['screen_height']
            stx_h = pars['stixelheight']
            stx_w = pars['stixelwidth']
            # Check whether any parameters are given for margins, calculate
            # screen dimensions.
            marginkeys = ['tmargin', 'bmargin', 'rmargin', 'lmargin']
            margins = []
            for key in marginkeys:
                margins.append(pars.get(key, 0))
            # Subtract bottom and top from vertical dimension; left and right
            # from horizontal dimension
            scr_width = scr_width-sum(margins[2:])
            scr_height = scr_height-sum(margins[:2])
            sx, sy = scr_height/stx_h, scr_width/stx_w
            # Make sure that the number of stimulus pixels are integers
            # Rounding down is also possible but might require
            # other considerations.
            if sx % 1 == 0 and sy % 1 == 0:
                sx, sy = int(sx), int(sy)
            else:
                raise ValueError('sx and sy must be integers')

            # HINT: fixing stimulus length for now because of memory
            # capacity
            total_frames = maxframenr

            randnrs, seed = randpy.ranb(seed, sx*sy*total_frames)
            # Reshape and change 0's to -1's
            stimulus = np.reshape(randnrs, (sx, sy, total_frames), order='F')*2-1
        return stimulus
    if stimtype == 'OMB':
        stimframes = pars.get('stimFrames', 108000)
        preframes = pars.get('preFrames', 200)
        nblinks = pars.get('Nblinks', 2)

        seed = pars.get('seed', -10000)
        seed2 = pars.get('objseed', -1000)

        stepsize = pars.get('stepsize', 2)

        ntotal = int(stimframes / nblinks)

        clusters, metadata = asc.read_spikesheet(exp)

        refresh_rate = metadata['refresh_rate']
        filter_length, frametimings = asc.ft_nblinks(exp, stim_nr, nblinks,
                                                     refresh_rate)
        frame_duration = np.ediff1d(frametimings).mean()
        frametimings = frametimings[:-1]
        if ntotal != frametimings.shape[0]:
            print(f'For {exp}\nstimulus {iof.getstimname(exp, stim_nr)} :\n'
                  f'Number of frames specified in the parameters file ({ntotal}'
                  f' frames) and frametimings ({frametimings.shape[0]}) do not'
                  ' agree!'
                  ' The stimulus was possibly interrupted during recording.'
                  ' ntotal is changed to match actual frametimings.')
            ntotal = frametimings.shape[0]

        randnrs, seed = randpy.gasdev(seed, ntotal*2)
        randnrs = np.array(randnrs)*stepsize

        xsteps = randnrs[::2]
        ysteps = randnrs[1::2]

        return np.vstack((xsteps, ysteps))
    return None

def conv(k, x):
    """
    Define convolution in the required way.
    """
    # Using 'same' as the mode results in the estimated filter to be
    # half a filter length later in time. So we convolve in full mode
    # and truncate the end.
    k = np.array(k)
    return np.convolve(k, x, 'full')[:-k.shape[0]+1]


def glm_fr(k, mu):
    """
    Return a function for the firing rate of a GLM neuron,
    given a filter and baseline firing rate.
    """
    return lambda x:np.exp((conv(k, x) + mu)) # exponential
#    return lambda x:3/(1+np.exp(-(conv(k, x) + mu))) # logistic


def minimize_loglhd(k_initial, mu_initial, x, time_res, spikes, usegrad=True,
                    debug_grad=False, method='Newton-CG', **kwargs):

    minimizekwargs = {'method':method,
                      'tol':1e-1,
#                      'options':{'disp':True},
                     }
    minimizekwargs.update(**kwargs)

    def loglhd(kmu):
        k_ = kmu[:-1]
        mu_ = kmu[-1]
        nlt_in = (conv(k_, x)+mu_)
        return -np.sum(spikes * nlt_in) + time_res*np.sum(np.exp(nlt_in))

    def grad(kmu):
        k_ = np.array(kmu[:-1])
        mu_ = kmu[-1]
        nlt_in = (conv(k_, x)+mu_)
        xr = asc.rolling_window(x, k_.shape[0])[:, ::-1]
        dldk = spikes@xr - time_res*np.exp(nlt_in)@xr
#        dldk2 = np.zeros(l)
#        for i in range(len(spikes)):
#            dldk2 += spikes[i] * xr[i, :]
#            dldk2 -= time_res*np.exp(nlt_in[i])*xr[i, :]
#        assert np.isclose(dldk, dldk2).all()
#        import pdb; pdb.set_trace()
        dldm = spikes.sum() - time_res*nlt_in.sum()
        dl = -np.array([*dldk, dldm])
        return dl

    if usegrad:
        minimizekwargs.update({'jac':grad})
    if debug_grad:
        from scipy.optimize import check_grad, approx_fprime
        kmu = [*k_initial, mu_initial]
        auto = lambda a: approx_fprime(a, loglhd, 1e-2)
        manu = lambda a: grad(a)
        return auto, manu
    res = minimize(loglhd, [*k_initial, mu_initial], **minimizekwargs)

    return res


def normalizestas(stas):
    stas = np.array(stas)
    b = np.abs(stas).max(axis=1)
    stas_normalized = stas / b.repeat(stas.shape[1]).reshape(stas.shape)
    return stas_normalized
