#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 13:37:25 2017

@author: ycan
"""
import matplotlib
import matplotlib.pyplot as plt
import os


def spineless(ax, which='trlb'):
    """
    Set the spine visibility quickly in matplotlib.

    Parameters:
        ax: The axis object returned by e.g. plt.subplot()
        which: List of spines to turn off.

    Example usage:
    ax=plt.subplot(111)
    ax.plot(np.random.randint(5, 10, size=10))
    spineless(ax, which='trlb')
    plt.show()
    """
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
    cm = matplotlib.colors.ListedColormap(colors)
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
    return cb
