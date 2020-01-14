#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import matplotlib.pyplot as plt
import genlinmod as glm
import OMB_texturegenerator as ombtg

exp = '20180710'
stimnr = 8

steps = glm.loadstim(exp, stimnr)
texture = ombtg.texture_generator()
noiselim = ombtg.noiselim
#texture = np.flipud(texture)
#%%
def clip_trajectory(stimulus, squaredim):
    pos_filter = np.where(stimulus>0)
    neg_filter = not pos_filter
    out = np.zeros(stimulus.shape)
    out = np.fmod(stimulus, squaredim[:, np.newaxis] * 1.5,
                 where=pos_filter)
    out = np.fmod(-stimulus, squaredim[:, np.newaxis] * 1.5,
                 where=neg_filter)
    return out
#clipped = clip_trajectory(stimulus, np.array([200, 200]))

def omb_trajectory(steps):
    """
    Calculate the displacement of the OMB stimulus background texture
    based on the steps by calculating the cumulative sum.

    The input is expected to be a 2,N array containing x and y steps.
    """
    return np.cumsum(steps, axis=1)

traj_raw = omb_trajectory(steps)/4

#traj *= 6

# Clip the trajectory so that it wraps around when it reaches the edges.
# Among the numpy remainder/modulo functions fmod is the one
# that suits this purpose.
traj = np.fmod(traj_raw, 1.5*noiselim[0])

traj *= -1

#traj[0, :] = 0


#plotlim = None
#plt.plot(traj_clip[0, :plotlim], traj_clip[1, :plotlim])
#plt.show()


#%%
#%matplotlib qt
plt.imshow(texture, cmap='Greys_r',
           vmin=0, vmax=1,
           )
import time; time.sleep(2)
for i in range(0,4):
    coord = traj[:, i]
    coord = coord + np.array([*noiselim])*1.5

    plt.xlim(coord[0]+[-noiselim[0]/2, noiselim[0]/2])
    # Order of ylim is reversed from defaults so y axis is flipped
    plt.ylim(coord[1]+[-noiselim[1]/2, noiselim[1]/2])
#    plt.title(i)
    plt.pause(1)

#    input('Press return.')

#plt.show()
