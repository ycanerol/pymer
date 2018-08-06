#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 15:42:40 2018

@author: ycan
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

import iofuncs as iof
import plotfuncs as plf
import analysis_scripts as asc
import miscfuncs as msc


def onedgauss(x, *p):
    a, mu, sigma = p
    y = a*np.exp(-np.power((x-mu), 2)/(2 * sigma**2))
    return y


def centersurround_onedim(x, a1, mu1, sig1, a2, mu2, sig2):
    y1 = a1*np.exp(-np.power((x-mu1), 2)/(2 * sig1**2))
    y2 = a2*np.exp(-np.power((x-mu2), 2)/(2 * sig2**2))
    return y1-y2


def stripesurround(exp_name, stimnrs):
    exp_dir = iof.exp_dir_fixer(exp_name)

    if isinstance(stimnrs, int):
        stimnrs = [stimnrs]
    elif len(stimnrs) == 0:
        return

    for stimnr in stimnrs:
        data = iof.load(exp_name, stimnr)

        _, metadata = asc.read_ods(exp_dir)
        px_size = metadata['pixel_size(um)']

        clusters = data['clusters']
        stas = data['stas']
        max_inds = data['max_inds']
        filter_length = data['filter_length']
        stx_w = data['stx_w']
        exp_name = data['exp_name']
        stimname = data['stimname']
        frame_duration = data['frame_duration']
        quals = data['quals']

        # Record which clusters are ignored during analysis
        try:
            included = data['included']
        except KeyError:
            included = [True]*clusters.shape[0]

        # Average STA values 100 ms around the brightest frame to
        # minimize noise
        cut_time = int(100/(frame_duration*1000)/2)

        # Tolerance for distance between center and surround
        # distributions 60 μm
        dtol = int((60/px_size)/2)

        clusterids = plf.clusters_to_ids(clusters)

        fsize = int(700/(stx_w*px_size))
        t = np.arange(filter_length)*frame_duration*1000
        vscale = fsize * stx_w*px_size

        cs_inds = np.empty(clusters.shape[0])
        polarities = np.empty(clusters.shape[0])

        fits = []
        all_parameters = []

        savepath = os.path.join(exp_dir, 'data_analysis', stimname)

        for i in range(clusters.shape[0]):
            sta = stas[i]
            max_i = max_inds[i]

            try:
                sta, max_i = msc.cutstripe(sta, max_i, fsize*2)
            except ValueError as e:
                if str(e) == 'Cutting outside the STA range.':
                    included[i] = False
                    continue
                else:
                    print(f'Error while analyzing {stimname}\n'+
                          f'Index:{i}    Cluster:{clusterids[i]}')
                    raise

            # Isolate the time point from which the fit will
            # be obtained
            if max_i[1] < cut_time:
                max_i[1] = cut_time+1
            fitv = np.mean(sta[:, max_i[1]-cut_time:max_i[1]+cut_time+1],
                           axis=1)
            # Make a space vector
            s = np.arange(fitv.shape[0])

            if np.max(fitv) != np.max(np.abs(fitv)):
                onoroff = -1
            else:
                onoroff = 1
            polarities[i] = onoroff
            # Determine the peak values for center and surround
            # to give as initial parameters for curve fitting
            centerpeak = onoroff*np.max(fitv*onoroff)
            surroundpeak = onoroff*np.max(fitv*-onoroff)

            # Define initial guesses for the center and surround gaussians
            # First set of values are for center, second for surround.
            p_initial = [centerpeak, max_i[0], 2, surroundpeak, max_i[0], 8]
            if onoroff == 1:
                bounds = ([0, -np.inf, -np.inf,
                           0, max_i[0]-dtol, 4],
                          [np.inf, np.inf, np.inf,
                           np.inf, max_i[0]+dtol, 20])
            elif onoroff == -1:
                bounds = ([-np.inf, -np.inf, -np.inf,
                           -np.inf, max_i[0]-dtol, 4],
                          [0, np.inf, np.inf,
                           0, max_i[0]+dtol, 20])

            try:
                popt, _ = curve_fit(centersurround_onedim, s, fitv,
                                    p0=p_initial, bounds=bounds)
            except (ValueError, RuntimeError) as e:
                er = str(e)
                if (er == "`x0` is infeasible." or
                        er.startswith("Optimal parameters not found")):
                    popt, _ = curve_fit(onedgauss, s, fitv, p0=p_initial[:3])
                    popt = np.append(popt, [0, popt[1], popt[2]])
                elif er == "array must not contain infs or NaNs":
                    included[i] = False
                    continue
                else:
                    print(f'Error while analyzing {stimname}\n'+
                          f'Index:{i}    Cluster:{clusterids[i]}')
                    import pdb
                    pdb.set_trace()
                    raise

            fit = centersurround_onedim(s, *popt)
            fits.append(fit)
            popt[0] = popt[0]*onoroff
            popt[3] = popt[3]*onoroff
            all_parameters.append(popt)

            csi = popt[3]/popt[0]
            cs_inds[i] = csi

            plt.figure(figsize=(10, 9))
            ax = plt.subplot(121)
            plf.stashow(sta, ax, extent=[0, t[-1], -vscale, vscale])
            ax.set_xlabel('Time [ms]')
            ax.set_ylabel('Distance [µm]')

            ax = plt.subplot(122)
            plf.spineless(ax)
            ax.set_yticks([])
            # We need to flip the vertical axis to match
            # with the STA next to it
            plt.plot(onoroff*fitv, -s, label='Data')
            plt.plot(onoroff*fit, -s, label='Fit')
            plt.axvline(0, linestyle='dashed', alpha=.5)
            plt.title(f'Center: a: {popt[0]:4.2f}, μ: {popt[1]:4.2f},'+
                      f' σ: {popt[2]:4.2f}\n'+
                      f'Surround: a: {popt[3]:4.2f}, μ: {popt[4]:4.2f},'+
                      f' σ: {popt[5]:4.2f}'+
                      f'\n CS index: {csi:4.2f}')
            plt.subplots_adjust(top=.85)
            plt.suptitle(f'{exp_name}\n{stimname}\n{clusterids[i]} '+
                         f'Q: {quals[i]:4.2f}')
            os.makedirs(os.path.join(savepath, 'stripesurrounds'),
                        exist_ok=True)
            plt.savefig(os.path.join(savepath, 'stripesurrounds',
                                     clusterids[i]+'.svg'),
                        bbox_inches='tight')
            plt.close()

        data.update({'cs_inds':cs_inds, 'polarities':polarities,
                     'included':included, 'all_parameters':all_parameters,
                     'fits':fits})
        np.savez(os.path.join(savepath, f'{stimnr}_data.npz'), **data)
        print(f'Surround plotted and saved for {stimname}.')
