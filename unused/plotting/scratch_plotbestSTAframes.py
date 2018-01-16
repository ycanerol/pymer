#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 20:17:46 2017

@author: ycan
"""
import miscfuncs as msc
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


plt.figure(figsize=(8.27, 11.69*2)) # A4 size
#plotind=0
for i in range(clusters.shape[0]):
    a = stas[i]
    a = msc.cut_around_center(a, max_inds[i], 25)[0]
    try:
        sta_max = np.max(np.abs([np.max(a),np.min(a)]))
#        plotind+=1
    except ValueError:
        continue
    sta_min = -sta_max
    ax = plt.subplot(17, 6, i+1)
    ax.imshow(a[:, :, max_inds[i][2]], vmin=sta_min, vmax=sta_max, cmap='RdBu')
    ax.set_aspect('equal')
    plt.title('Cell {:0>3}{:0>2}'.format(clusters[i, 0], clusters[i, 1]),
             fontsize=9)
    ax.axis('off')

scalebar = AnchoredSizeBar(ax.transData,
                           5, '{} µm'.format(5*stx_h*px_size), 'lower left',
                           pad=1,
                           color='k',
                           frameon=False,
                           size_vertical=.5)
ax.add_artist(scalebar)
#plt.suptitle('{} {}'.format(metadata['experiment_date'], stimname))
plt.subplots_adjust(wspace=0.01, hspace=0.1)
plt.show()
