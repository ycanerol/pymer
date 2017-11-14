#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 13:37:25 2017

@author: ycan
"""
import matplotlib
import matplotlib.pyplot as plt
import os


def set_spines(ax, which='trlb'):
    """
    Set the spine visibility quickly in matplotlib.

    Parameters:
        ax: The axis object returned by e.g. plt.subplot()
        which: List of spines to turn off.

    Example usage:
    ax=plt.subplot(111)
    ax.plot(np.random.randint(5, 10, size=10))
    set_spines(ax, which='trlb')
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
