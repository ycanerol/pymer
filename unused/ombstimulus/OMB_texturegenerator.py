#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate the background texture for OMB stimulus

"""

import numpy as np
import matplotlib.pyplot as plt

import analysis_scripts as asc
import iofuncs as iof
from randpy import randpy
from scipy.ndimage.filters import gaussian_filter


def mirrorexpand(x, times=1):
    """
    Rxpands a matrix by mirroring at all sides, to mimic OpenGL
    texture wrapping (GL_MIRORED_REPEAT) behaviour.

    """
    for i in range(times):
        x = np.vstack((np.flipud(x), x, np.flipud(x)))
        x = np.hstack((np.fliplr(x), x, np.fliplr(x)))
    return x


#%%
exp, stim_nr = ('20180710', 8)

sortedstim = asc.stimulisorter(exp)
clusters, metadata = asc.read_spikesheet(exp)
pars = asc.read_parameters(exp, stim_nr)

for key, val in sortedstim.items():
    if stim_nr in val:
        stimtype = key

if stimtype != 'OMB':
    ValueError('The stimulus is not OMB.')

bgnoise = pars.get('bgnoise', 4)
if bgnoise != 1:
    bgstixel = pars.get('bgstixel', 5)
else:
    bgstixel = pars.get('bgstixel', 10)

bgcontrast = pars.get('bgcontrast', 0.3)
bggenerationseed = -10000
filterstd = pars.get('filterstdv', bgstixel)
meanintensity = pars.get('meanintensity', 0.5)
contrast = pars.get('contrast', 1)
squareheight, squarewidth = (800, 800)
#%%
filterwidth = filterstd/bgstixel*3
noiselim = (np.ceil(np.array([squareheight, squarewidth])/bgstixel)).astype(int)


# Gaussian filter is applied to the noise field by a for loop in the cpp code,
# and its norm is
xx, yy = np.meshgrid(np.arange(2*filterwidth), np.arange(2*filterwidth))
gfilter = np.exp2(-((xx-filterwidth)**2+(yy-filterwidth)**2)/(2*(filterstd/bgstixel)**2))

norm = gfilter.sum()
randnrs = np.reshape(randpy.gasdev(bggenerationseed,
                        noiselim[0]*noiselim[1])[0],
                     (noiselim[0], noiselim[1]))
noisefield = meanintensity + meanintensity*bgcontrast*randnrs
#noisefield = np.flipud(noisefield) # Flip the texture to match the stimulator
#%%
#filtered = gaussian_filter(noisefield, np.sqrt(filterstd))
#post_tiled = np.tile(filtered, [3,3]) # Filter first, then tile

# First tile the noise field, then filter.
tiled = gaussian_filter(np.tile(noisefield, [3, 3]), np.sqrt(filterstd))

import scipy.ndimage as snd

tiled2 = snd.convolve(np.tile(noisefield, [3, 3]), gfilter)
tiled2 = ((tiled2/norm-meanintensity)*filterstd/bgstixel + meanintensity)
tiled2 = np.clip(tiled2, 0, 1)*2-1

def texture_generator():
    return tiled2

if __name__ == '__main__':
#    plt.figure(figsize=(9,9))
#    plt.imshow(filtered, cmap='Greys_r',
#              vmin=0.3, vmax=.7
#               )
#
#    plt.figure(figsize=(9,9))
#    plt.imshow(mirrorexpand(filtered),
#               cmap='Greys_r',
#               vmin=0.3, vmax=.7
#               )
#    #%%
#    plt.figure(figsize=(9,9))
#    plt.imshow(post_tiled,
#               cmap='Greys_r',
#               vmin=0.3, vmax=.7
#               )
#
#    plt.figure(figsize=(9,9))
#    plt.imshow(gaussian_filter(post_tiled, np.sqrt(filterstd)),
#               cmap='Greys_r',
#               vmin=0.3, vmax=.7
#               )

    plt.figure(figsize=(9,9))
    plt.imshow(tiled2,
               cmap='Greys_r',
#               vmin=0, vmax=1
               )

    #%%

    tree = np.fromfile('/media/gruppenlaufwerk/Norma/OpenGL_images/TestImage0003.raw', dtype=np.uint8)
    tree = tree.reshape((1000, 900))
    tree = tree[:800, :800]
    tree = np.rot90(tree, 3)
    #plt.imshow(tree, cmap='Greys_r')
    plt.figure(figsize=(12, 12))
    plt.imshow(mirrorexpand(tree, times=2), cmap='Greys_r')
