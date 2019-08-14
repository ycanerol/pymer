#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate the STA using the texture stimulus for OMB
"""
import os

import numpy as np
import matplotlib.pyplot as plt

import analysis_scripts as asc
import plotfuncs as plf

from omb import OMB


exp, ombstimnr = '20180710', 8
checkerstimnr = 6
maxframes = 20000
contrast_window=100


def ombtexturesta(exp, ombstimnr, maxframes=10000,
                  contrast_window=100, plot=False):
    """
    Calculates the spike-triggered average for the full texture for the OMB
    stimulus. Based on the maximum intensity pixel of the STAs, calculates
    the center of the receptive field and the contrast signal for this
    pixel throughout the stimulus; to be used as input for models.

    Parameters:
    --------
        exp:
            The experiment name
        ombstimulusnr:
            Number of the OMB stimulus in the experiment
        maxframes:
            Maximum number of frames that will be used, typically the
            array containing the contrast is very large and
            it is easy to fill the RAM. Refer to OMB.generatecontrast()
            documentation.
        contrast_window:
            Number of pixels to be used for the size of the texture.
            Measured in each direction starting from the center so
            a value of 100 will yield texture with size (201, 201, N)
            where N is the total number of frames.
        plot:
            If True draws an interactive plot for browsing all STAs,
            also marking the center pixels. Requires an interactive backend

    """
    st = OMB(exp, ombstimnr, maxframes=maxframes)
    st.clusterstats()

    contrast = st.generatecontrast(st.texpars.noiselim/2,
                                   window=contrast_window,
                                   pad_length=st.filter_length-1)

    contrast_avg = contrast.mean(axis=-1)

    RW = asc.rolling_window(contrast, st.filter_length, preserve_dim=False)

    all_spikes = np.zeros((st.nclusters, st.ntotal))
    for i in range(st.nclusters):
        all_spikes[i, :] = st.binnedspiketimes(i)


    texturestas = np.einsum('abcd,ec->eabd', RW, all_spikes)
    texturestas /= all_spikes.sum(axis=(-1))[:, np.newaxis,
                                             np.newaxis, np.newaxis]

    # Correct for the non-informative parts of the stimulus
    texturestas = texturestas - contrast_avg[None, ..., None]
    #%%
    if plot:
        fig_stas, _ = plf.multistabrowser(texturestas, cmap='Greys_r')

    texture_maxi = np.zeros((st.nclusters, 2), dtype=int)
    # Take the pixel with maximum intensity for contrast signal
    for i in range(st.nclusters):
        coords = np.unravel_index(np.argmax(np.abs(texturestas[i])),
                                  texturestas[i].shape)[:-1]
        texture_maxi[i, :] = coords
        if plot:
            ax = fig_stas.axes[i]
            # Coordinates need to be inverted for display
            ax.plot(*coords[::-1], 'r+', markersize=10, alpha=0.2)
    #%%
    contrast_signals = np.empty((st.nclusters, st.ntotal))
    # Calculate the time course of the center(maximal pixel of texture STAs
    stas_center = np.zeros((st.nclusters, st.filter_length))
    for i in range(st.nclusters):
        coords = texture_maxi[i, :]
        # Calculate the contrast signal that can be used for GQM
        # Cut the extra part at the beginning that was added by generatecontrast
        contrast_signals[i, :] = contrast[coords[0], coords[1],
                                          st.filter_length-1:]
        stas_center[i] = texturestas[i, coords[0], coords[1], :]

    stas_center_norm = asc.normalize(stas_center)

    fig_contrast, axes = plt.subplots(*plf.numsubplots(st.nclusters), sharey=True)
    for i, ax in enumerate(axes.ravel()):
        if i < st.nclusters:
            ax.plot(stas_center_norm[i, :])

    savepath = os.path.join(st.exp_dir, 'data_analysis', st.stimname)
    savefname = f'{st.stimnr}_texturesta'
    if not maxframes:
        maxframes = st.ntotal
    savefname += f'_{maxframes}fr'

    plt.ylim([np.nanmin(stas_center_norm), np.nanmax(stas_center_norm)])
    fig_contrast.suptitle('Time course of center pixel of texture STAs')
    fig_contrast.savefig(os.path.join(savepath, 'texturestas.svg'))

    # Do not save the contrast signal because it is ~6GB for 20000 frames of recording
    keystosave = ['texturestas', 'contrast_avg', 'stas_center',
                  'stas_center_norm', 'contrast_signals', 'texture_maxi',
                  'maxframes', 'contrast_window']
    datadict = {}
    for key in keystosave:
        datadict[key] = locals()[key]

    np.savez(os.path.join(savepath, savefname), **datadict)
    if plot:
        return fig_stas
