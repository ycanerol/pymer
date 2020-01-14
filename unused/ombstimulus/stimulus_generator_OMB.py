#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import numpy as np

import analysis_scripts as asc
import iofuncs as iof
from randpy import randpy
from scipy.ndimage import convolve


def mirrorexpand(x, times=1):
    """
    Rxpands a matrix by mirroring at all sides, to mimic OpenGL
    texture wrapping (GL_MIRORED_REPEAT) behaviour.

    """
    for i in range(times):
        x = np.vstack((np.flipud(x), x, np.flipud(x)))
        x = np.hstack((np.fliplr(x), x, np.fliplr(x)))
    return x


def stim_pars(exp, stimnr):
    pass

exp, stimnr = '20180710', 8

sortedstim = asc.stimulisorter(exp)
_, metadata = asc.read_spikesheet(exp)
pars = asc.read_parameters(exp, stimnr)

for key, val in sortedstim.items():
    if stimnr in val:
        stimtype = key

if stimtype != 'OMB':
    ValueError('The stimulus is not OMB.')


stimframes = pars.get('stimFrames', 108000)
preframes = pars.get('preFrames', 200)
nblinks = pars.get('Nblinks', 2)

seed = pars.get('seed', -10000)
seed2 = pars.get('objseed', -1000)

stepsize = pars.get('stepsize', 2)
gausssteps = pars.get('gaussteps', True)
smoothtrajectory = pars.get('smoothtrajectory', False)
eyemovements = pars.get('eyemovements', False)

if smoothtrajectory or eyemovements or not gausssteps:
    raise NotImplementedError('Analysis of only non-smoothed '
                              'gaussian steps are implemented.')

bgnoise = pars.get('bgnoise', 4)
if bgnoise != 1:
    bgstixel = pars.get('bgstixel', 5)
else:
    bgstixel = pars.get('bgstixel', 10)

if bgnoise != 4:
    raise NotImplementedError('Only gaussian correlated binary '
                              'noise is implemented.')

bgcontrast = pars.get('bgcontrast', 0.3)
bggenerationseed = -10000
filterstd = pars.get('filterstdv', bgstixel)
meanintensity = pars.get('meanintensity', 0.5)
contrast = pars.get('contrast', 1)
squareheight, squarewidth = (800, 800)

ntotal = int(stimframes / nblinks)
refresh_rate = metadata['refresh_rate']

_, frametimings = asc.ft_nblinks(exp, stimnr, nblinks,
                                             refresh_rate)
frame_duration = np.ediff1d(frametimings).mean()
frametimings = frametimings[:-1]
if ntotal != frametimings.shape[0]:
    print(f'For {exp}\nstimulus {iof.getstimname(exp, stimnr)} :\n'
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
