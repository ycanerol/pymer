#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Incorporated to plotfuncs
"""
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.widgets import Slider

import iofuncs as iof
import plotfuncs as plf


def multistabrowser(stas, frame_duration=None, cmap=None, centerzero=True):
    """
    Returns an interactive plot to browse multiple spatiotemporal
    STAs at the same time. Requires an interactive matplotlib backend.

    Parameters
    --------
    stas:
        Numpy array containing STAs. First dimension should index individual cells,
        last dimension should index time.
    frame_duration:
      Time between each frame. (optional)
    cmap:
      Colormap to use.
    centerzero:
      Whether to center the colormap around zero for diverging colormaps.

    Example
    ------
    >>> print(stas.shape) # (nrcells, xpixels, ypixels, time)
    (36, 75, 100, 40)
    >>> fig, slider = stabrowser(stas, frame_duration=1/60)

    Notes
    -----
    When calling the function, the slider is returned to prevent the reference
    to it getting destroyed and to keep it interactive.
    The dummy variable `_` can also be used.
    """
    interactive_backends = ['Qt', 'Tk']
    backend = mpl.get_backend()
    if not backend[:2] in interactive_backends:
        raise ValueError('Switch to an interactive backend (e.g. Qt) to see'
                         ' the animation.')

    if isinstance(stas, list):
        stas = np.array(stas)

    if cmap is None:
        cmap = iof.config('colormap')
    if centerzero:
        vmax = absmax(stas)
        vmin = absmin(stas)
    else:
        vmax, vmin = stas.max(), stas.min()

    imshowkwargs = dict(cmap=cmap, vmax=vmax, vmin=vmin)


    rows, cols = plf.numsubplots(stas.shape[0])
    fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True)

    initial_frame = 5

    axsl = fig.add_axes([0.25, 0.05, 0.65, 0.03])
    # For the slider to remain interactive, a reference to it should
    # be kept, so it is returned by the function
    slider_t = Slider(axsl, 'Frame before spike',
                      0, stas.shape[-1]-1,
                      valinit=initial_frame,
                      valstep=1,
                      valfmt='%2.0f')

    def update(frame):
        frame = int(frame)
        for i in range(rows):
            for j in range(cols):
                im = axes[i, j].get_images()[0]
                im.set_data(stas[i*rows+j, ..., frame])
        if frame_duration is not None:
            fig.suptitle(f'{frame*frame_duration*1000:4.0f} ms')
        fig.canvas.draw_idle()

    slider_t.on_changed(update)

    for i in range(rows):
        for j in range(cols):
            ax = axes[i, j]
            ax.imshow(stas[i*rows+j, ..., initial_frame], **imshowkwargs)
            ax.set_axis_off()
    plt.tight_layout()
    plt.subplots_adjust(wspace=.01, hspace=.01)
    return fig, slider_t


if __name__ == '__main__':
    data = iof.load('20180710', 6)
    stas = np.array(data['stas'])
    frame_duration = data['frame_duration']
    fig, _ = stabrowser(stas, frame_duration)
