#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 15:31:58 2018

@author: ycan
"""
#%%
import matplotlib.pyplot as plt
import miscfuncs as msc
import analysis_scripts as asc

path = ('/home/ycan/Documents/data/Erol_20171122_252MEA_fr_re_fp/data'
        '_analysis/7_checkerflicker5x5bw2blinks/7_data.h5')

locals().update(msc.loadh5(path))
index = 25
fit_frame = stas[index][:, :, max_inds[index][2]]

exp_dir = '/home/ycan/Documents/data/Erol_20171122_252MEA_fr_re_fp'
_, parameters = asc.read_spikesheet(exp_dir)
px_size = parameters['pixel_size(um)']

ax = plt.subplot(111)
ax.imshow(fit_frame, cmap = 'RdBu')


from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

scalebar = AnchoredSizeBar(ax.transData,
                           3, '{} µm'.format(3*stx_h*px_size), 'lower left',
                           pad=1,
                           color='k',
                           frameon=False,
                           size_vertical=.5)
ax.add_artist(scalebar)
plt.show()
