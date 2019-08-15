#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
#%%
import numpy as np
import matplotlib.pyplot as plt

import plotfuncs as plf

from omb import OMB

exp, ombstimnr = 'Kuehn', 13
checkerstimnr = 1


st = OMB(exp, ombstimnr,
         maxframes=500
         )

traj = st.bgtraj*3
traj = np.flipud(traj)
st.bgtraj = traj

#%matplotlib qt
st.bgtraj_clipped = np.fmod(st.bgtraj, 1.5*st.texpars.noiselim[0])

contrast = st.generatecontrast([0, 0], 100)

plt.imshow(contrast[..., 0], cmap='Greys_r')

fig, sl = plf.stabrowser(contrast, cmap='Greys_r')


#%%
# HINT: The loop and other versions are slightly different between

st._generatetexture_withloops()
tex_loop_sm = st.texturewithloops[::4, ::4]
tiled = st.texture[200:400, 200:400]

np.allclose(tex_loop_sm, st.texturebasic)
np.allclose(tiled, tex_loop_sm)
np.allclose(tiled, st.texturebasic)

dif = tex_loop_sm - st.texturebasic

kwargs = dict(vmin=-1, vmax=1)
#%%
fig, axes = plt.subplots(1, 3)
ax1, ax2, ax3 = axes.ravel()

ax1.imshow(tex_loop_sm, **kwargs)
ax2.imshow(st.texturebasic, **kwargs)
ax3.imshow(dif, **kwargs)

plt.figure()
plt.hist(dif.ravel(), bins=50)