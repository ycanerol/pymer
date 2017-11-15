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
