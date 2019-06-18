#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 13:37:25 2017

@author: ycan
"""
import os

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.widgets import Slider

import iofuncs as iof
import analysis_scripts as asc

interactive_backends = ['Qt', 'Tk']


def spineless(axes, which='trlb'):
    """
    Set the spine visibility quickly in matplotlib.

    Parameters:
    --------
    ax:
        List of axis objects returned by e.g. plt.subplot()
    which:
        List of spines to turn off.

    Example usage:

    >>> ax=plt.subplot(111)
    >>> ax.plot(np.random.randint(5, 10, size=10))
    >>> spineless(ax, which='trlb')
    >>> plt.show()
    """
    # Check whether a single axes object is given
    if isinstance(axes, mpl.axes.Axes):
        axes = [axes]

    for ax in axes:
        if which.find('t') is not -1: ax.spines['top'].set_visible(False)
        if which.find('r') is not -1: ax.spines['right'].set_visible(False)
        if which.find('l') is not -1: ax.spines['left'].set_visible(False)
        if which.find('b') is not -1: ax.spines['bottom'].set_visible(False)


def savefigmkdir(path, **kwargs):
    try:
        plt.savefig(path, **kwargs)
    except FileNotFoundError:
        parent = os.path.split(path)[0]
        os.mkdir(parent)
        plt.savefig(path, **kwargs)


def RFcolormap(colors=None):
    """
    Return custom colormap for displaying surround in STA or SVD, based on a
    list of colors. Use hex colors when possible.

    Generated using http://hclwizard.org:64230/hclwizard/
    Default is based on RdBu.
    HCLwizard parameters:
        Hue1:       12
        Hue2:       265
        Chroma:     80
        Luminance1: 24
        Luminance2: 100
        Power:      0.35
        Number:     39
    """
    if colors is None:
        colors = ("#790102","#7C0B0C","#7F1314","#821B1C","#852122","#892828",
                  "#8C2E2E","#903434","#943A3B","#984141","#9C4848","#A14F4F",
                  "#A65757","#AB6060","#B16969","#B77474","#BF8181","#C89191",
                  "#D5A8A8","#FFFFFF","#AFB0D3","#9A9BC7","#8C8DBF","#8081B9",
                  "#7678B4","#6D6FAF","#6668AB","#5E61A8","#585AA5","#5254A3",
                  "#4C4FA1","#46499F","#41449E","#3B3F9D","#363A9D","#30359D",
                  "#2A309E","#232AA0","#1A24A2")
    cm = mpl.colors.ListedColormap(colors)
    return cm


def numsubplots(n, recursive=False):
    """
    Define the best arrangement of subplots for a
    given number of plots.

    Parameters:
        n:
            Number of total plots needed.
        recursive:
            Whether to return current number of subplots.
            Used for recursive calls.
    Returns:
        p:
            A list containing the ideal arrangement
            of subplots in the format of [nrows, ncolumns].
        n:
            Current number of subplots. Returned only
            when recursively calling the function.

    Ported to Python by Yunus Can Erol on Dec 2017
    from mathworks.com/matlabcentral/fileexchange/
    26310-numsubplots-neatly-arrange-subplots
    Original by Rob Campbell, Jan 2010

    Requires prime factorization and checking primality
    which is not provided by Python by default; therefore
    a custom package (primefac) is required.

    """
    import primefac

    while primefac.isprime(n) and n > 4:
        n += 1

    p = primefac.prime_factorize(n)
    if len(p) == 1:
        p = [1] + p
        if recursive:
            return p, n
        else:
            return p

    while len(p) > 2:
        if len(p) >= 4:
            p[0] = p[0]*p[-2]
            p[1] = p[1]*p[-1]
            del p[-2:]
        else:
            p[0] = p[0]*p[1]
            del p[1]
        p = sorted(p)

    while p[1]/p[0] > 2.5:
        N = n+1
        p, n = numsubplots(N, recursive=True)

    if recursive:
        return p, n
    else:
        return p


def clusters_to_ids(clusters):
    """
    Turns clusters array into a list containing formatted IDs for
    each cluster.

    e.g. channel 7, cluster 2 should have cluster id 00702.
    """
    # Store ID for clusters in a list for ease of use for plotting.
    clusterids = []

    for i in range(clusters.shape[0]):
        txt = '{:0>3}{:0>2}'.format(clusters[i, 0], clusters[i, 1])
        clusterids.append(txt)
    return clusterids


def colorbar(mappable, size='5%', **kwargs):
    """
    Make colorbars that scale properly.

    Size determines the proportion of the colorbar width
    with respect to image axis.

    kwargs will be passed to colorbar.

    Usage:
        im = ax.imshow(data)
        colorbar(im, size='5%)

    With STAs:
        im = ax.imshow(data)
        colorbar(im, ticks=[vmin, 0, vmax], format='%.2f')

    Taken from
    http://joseph-long.com/writing/colorbars/
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size=size, pad=0.05)
    cb = fig.colorbar(mappable, cax=cax, **kwargs)
    # Turn off the box around the colorbar.
    cb.outline.set_linewidth(0)
    # Turn off tick marks
    cb.ax.tick_params(length=0)
    return cb


def drawonoff(ax, preframedur, stimdur, h=1, contrast=1):
    """
    Draws rectangles on plot to represent different parts of
    the on off steps stimulus.

    Parameters:
        ax: matplotlib.Axes object
            The Axes object of the plot to draw rectangles on

        preframedur:
            Duration of the gray period in seconds

        stimdur:
            Duration of stimulus (step) in seconds

        h:
            Height of the rectangles to draw

        contrast:
            Stimulus contrast as given in parameter file

    Note:
        Rectangles are drawn starting from top left of the
        available area, therefore -h is used in the function
        as the height of the rectangles.
    """
    h = -h
    totaldur = 2*(preframedur+stimdur)

    rect1 = mpl.patches.Rectangle((0, 1),
                                  width=preframedur/totaldur,
                                  height=h,
                                  transform=ax.transAxes, facecolor='k',
                                  alpha=.5)
    rect2 = mpl.patches.Rectangle((preframedur/totaldur, 1),
                                  width=stimdur/totaldur,
                                  height=h,
                                  transform=ax.transAxes, facecolor='k',
                                  alpha=.5*(1-contrast))
    rect3 = mpl.patches.Rectangle(((preframedur +
                                    stimdur)/totaldur, 1),
                                  width=preframedur/totaldur,
                                  height=h,
                                  transform=ax.transAxes, facecolor='k',
                                  alpha=.5)
    rect4 = mpl.patches.Rectangle(((2*preframedur +
                                    stimdur)/totaldur, 1),
                                  width=stimdur/totaldur,
                                  height=h,
                                  transform=ax.transAxes, facecolor='k',
                                  alpha=.5*(1+contrast))
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    ax.add_patch(rect3)
    ax.add_patch(rect4)
    ax.set_xlim(0, totaldur)


def stashow(sta, ax=None, cbar=True, **kwargs):
    """
    Plot STA in a nice way with proper colormap and colorbar.

    STA can be single frame from checkerflicker or whole STA
    from stripeflicker.

    Following kwargs are available:
        imshow
            extent: Change the labels of the axes. [xmin, xmax, ymin, ymax]
            aspect: Aspect ratio of the image. 'auto', 'equal'
            cmap:  Colormap to be used. Default is set in config
        colorbar
            size: Width of the colorbar as percentage of image dimension
                  Default is 2%
            ticks: Where the ticks should be placed on the colorbar.
            format: Format for the tick labels. Default is '%.2f'
    Usage:
        ax = plt.subplot(111)
        stashow(sta, ax)
    """
    vmax = asc.absmax(sta)
    vmin = asc.absmin(sta)

    # Make a dictionary for imshow and colorbar kwargs
    imshowkw = {'cmap': iof.config('colormap'), 'vmin':vmin, 'vmax':vmax}
    cbarkw = {'size':'2%', 'ticks':[vmin, vmax], 'format':'%.2f'}
    for key in kwargs.keys():
        if key in ['extent', 'aspect', 'cmap']:
            imshowkw.update({key:kwargs[key]})
        elif key in ['size', 'ticks', 'format']:
            cbarkw.update({key:kwargs[key]})
        else:
            raise ValueError(f'Unknown kwarg: {key}')

    if ax is None:
        ax = plt.gca()

    im = ax.imshow(sta, **imshowkw)
    spineless(ax)
    if cbar:
        colorbar(im, **cbarkw)
    return im


def subplottext(text, axis, x=-.3, y=1.1, **kwargs):
    textkwargs = {'transform':axis.transAxes,
                  'fontsize':12,
                  'fontweight':'bold',
                  'va':'top',
                  'ha':'right'}
    textkwargs.update(kwargs)
    axis.text(x, y, text, **textkwargs)


def addarrowaxis(ax, x=0.5, y=0.5, dx=.1, dy=.2, xtext='',
                 ytext='', xtextoffset=0.02, ytextoffset=.032,
                 fontsize='xx-small'):
    """
    Add small arrows to indicate axes.

    ax:
        matplotlib.Axes instance
    x,y:
        Common origin of the arrows, units are in fraction of the axis
    dx, dy:
        Length of the arrows
    xtext, ytext:
        Text to label the arrows with
    xtextoffset, ytextoffset:
        Distance of the text from the arrows, requires manual
        adjustment
    fontsize:
        Font size of the text
    """
    arrowprops = dict(arrowstyle='<-', facecolor='black')
    ax.annotate('', xy=(x, y),  xycoords='axes fraction',
                xytext=(x+dx, y), textcoords='axes fraction',
                arrowprops=arrowprops)
    ax.annotate('', xy=(x, y),  xycoords='axes fraction',
                xytext=(x, y+dy), textcoords='axes fraction',
                arrowprops=arrowprops)
    ax.text(x, y - xtextoffset, xtext, transform=ax.transAxes,
        fontsize=fontsize, va='top')
    ax.text(x - ytextoffset, y, ytext, rotation=90, transform=ax.transAxes,
        fontsize=fontsize, ha='left', va='bottom')


def playsta(sta, frame_duration=None, cmap=None, centerzero=True, **kwargs):
    """
    Create a looped animation for a single STA with 3 dimensions.

    Parameters
    ---------
    cmap:
        Colormap to be used. Defaults to the specified colormap in the
        config file.
    centerzero:
        Center the colormap around zero if True.
    interval:
        Frame rate for the animation in ms.
    repeat_delay:
        Time to wait before the animation is repeated in ms.

    Note
    ----
    The returned animation can be saved like so:

    >>> ani = playsta(sta)
    >>> ani.save('wheretosave/sta.gif', writer='imagemagick', fps=10)
    """
    check_interactive_backend()

    if cmap is None:
        cmap = iof.config('colormap')
    if centerzero:
        vmax = asc.absmax(sta)
        vmin = asc.absmin(sta)
    else:
        vmax, vmin = sta.max(), sta.min()
    ims = []
    fig = plt.figure()
    ax = plt.gca()
    for i in range(sta.shape[-1]):
        im = ax.imshow(sta[:, :, i], animated=True,
                        cmap=cmap, vmin=vmin, vmax=vmax)

        ims.append([im]) # Needs to be a list of lists
    ani = animation.ArtistAnimation(fig, ims, **kwargs)

    return ani


def multistabrowser(stas, frame_duration=None, normalize=True, cmap=None,
                    centerzero=True, **kwargs):
    """
    Returns an interactive plot to browse multiple spatiotemporal
    STAs at the same time. Requires an interactive matplotlib backend.

    Parameters
    --------
    stas:
        Numpy array containing STAs. First dimension should index individual cells,
        last dimension should index time.
        Alternatively, this could be a list of numpy arrays.
    frame_duration:
      Time between each frame. (optional)
    normalize:
      Whether to normalize each STA
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
    check_interactive_backend()

    if isinstance(stas, list):
        stas = np.array(stas)

    if normalize:
        stas = asc.normalize(stas)

    if cmap is None:
        cmap = iof.config('colormap')
    if centerzero:
        vmax = asc.absmax(stas)
        vmin = asc.absmin(stas)
    else:
        vmax, vmin = stas.max(), stas.min()

    imshowkwargs = dict(cmap=cmap, vmax=vmax, vmin=vmin, **kwargs)

    rows, cols = numsubplots(stas.shape[0])
    fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True)

    initial_frame = 5

    axsl = fig.add_axes([0.25, 0.05, 0.65, 0.03])
    # For the slider to remain interactive, a reference to it should
    # be kept, so it set to a variable and is returned by the function
    slider_t = Slider(axsl, 'Frame before spike',
                      0, stas.shape[-1]-1,
                      valinit=initial_frame,
                      valstep=1,
                      valfmt='%2.0f')

    def update(frame):
        frame = int(frame)
        for i in range(rows):
            for j in range(cols):
                # Calculate the flattened index, equivalent to i*cols+j
                flat_idx = np.ravel_multi_index([i, j], (rows, cols))
                if flat_idx < stas.shape[0]:
                    im = axes[i, j].get_images()[0]
                    im.set_data(stas[flat_idx, ..., frame])
        if frame_duration is not None:
            fig.suptitle(f'{frame*frame_duration*1000:4.0f} ms')
        fig.canvas.draw_idle()

    slider_t.on_changed(update)

    for i in range(rows):
        for j in range(cols):
            flat_idx = np.ravel_multi_index([i, j], (rows, cols))
            ax = axes[i, j]
            if flat_idx < stas.shape[0]:
                ax.imshow(stas[i*cols+j, ..., initial_frame], **imshowkwargs)
            ax.set_axis_off()
    plt.tight_layout()
    plt.subplots_adjust(wspace=.01, hspace=.01)
    return fig, slider_t


def stabrowser(sta, frame_duration=None, cmap=None, centerzero=True, **kwargs):
    """
    Returns an interactive plot to browse an spatiotemporal
    STA. Requires an interactive matplotlib backend.

    Parameters
    --------
    sta:
        Numpy array containing the STA. Last dimension should index time.
    frame_duration:
      Time between each frame. (optional)
    cmap:
      Colormap to use.
    centerzero:
      Whether to center the colormap around zero for diverging colormaps.

    Example
    ------
    >>> print(sta.shape) # (xpixels, ypixels, time)
    (75, 100, 40)
    >>> fig, slider = stabrowser(sta, frame_duration=1/60)

    Notes
    -----
    When calling the function, the slider is returned to prevent the reference
    to it getting destroyed and to keep it interactive.
    The dummy variable `_` can also be used.
    """
    check_interactive_backend()

    if cmap is None:
        cmap = iof.config('colormap')
    if centerzero:
        vmax = asc.absmax(sta)
        vmin = asc.absmin(sta)
    else:
        vmax, vmin = sta.max(), sta.min()

    imshowkwargs = dict(cmap=cmap, vmax=vmax, vmin=vmin, **kwargs)

    fig = plt.figure()
    ax =  fig.add_axes([0.1, 0.1, 0.8, 0.8])


    initial_frame = 5

    axsl = fig.add_axes([0.25, 0.05, 0.65, 0.03])
    # For the slider to remain interactive, a reference to it should
    # be kept, so it set to a variable and is returned by the function
    slider_t = Slider(axsl, 'Frame before spike',
                      0, sta.shape[-1]-1,
                      valinit=initial_frame,
                      valstep=1,
                      valfmt='%2.0f')

    def update(frame):
        frame = int(frame)
        im = ax.get_images()[0]
        im.set_data(sta[..., frame])
        if frame_duration is not None:
            fig.suptitle(f'{frame*frame_duration*1000:4.0f} ms')
        fig.canvas.draw_idle()

    slider_t.on_changed(update)

    ax.imshow(sta[..., initial_frame], **imshowkwargs)
    ax.set_axis_off()
    plt.tight_layout()
    plt.subplots_adjust(wspace=.01, hspace=.01)
    return fig, slider_t


def check_interactive_backend():
    """
    Check whether the current backend is an interactive one for certain
    plots.
    """
    backend = mpl.get_backend()
    if not backend[:2] in interactive_backends:
        raise ValueError('Switch to an interactive backend (e.g. Qt) to see'
                         ' the animation.')