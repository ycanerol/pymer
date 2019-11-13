#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
#%%
import numpy as np

def gauss2d(rx, ry, width):
    def func(X, Y):
        g = np.exp(-(((rx-X)/width)**2 + ((ry-Y)/width)**2)/2)
        return g
    return func


def weightstofilter(W, d):
    """
    W:
        weights for the Q matrix
    d:
        Single dimension of the square Q matrix
    """
    nbases = np.sqrt(W.shape[0]).astype(int)
    stepsize = d/nbases
    W = W.reshape((nbases, nbases))
    D = np.zeros((d, d))
    for i in range(nbases):
        for j in range(nbases):
            D += W[i, j] * gauss2d(i*stepsize + stepsize*.5,
                                   j*stepsize + stepsize*.5,
                                   stepsize/2)(X, Y)
    return D


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    nbases = 10
    d = 20
    stepsize = d/nbases

    x = np.arange(d)
    y = np.arange(d)

    X, Y = np.meshgrid(x, y, indexing='ij')

    W = np.zeros(nbases**2)
    W[2] = 1
    W[0:6] = 1
    W[-1] = -1
    D = weightstofilter(W)
    plt.imshow(D)

    #%%
    import scipy.ndimage as snd
    from scipy import optimize

    if False:
        np.random.seed(12)
        q = np.random.normal(size=(d, d))
        q = snd.gaussian_filter(q, sigma=1)
    else:
        from omb import OMB
        st = OMB('20180710', 8)
        data = np.load(st.stim_dir
                       + f'/GQM_motioncontrast_val/{st.stimnr}_GQM_motioncontrast_val.npz')
        qall = data['Qall']
        q = qall[0, 0, ...]

    def objective_func(W):
        return np.sum((q - weightstofilter(W))**2) + .01*np.sum(np.abs(W))

    res = optimize.minimize(objective_func, np.zeros((nbases**2)))
    print(res.message)
    q_res = weightstofilter(res.x)

    fig, axes = plt.subplots(3, 1)
    vmax = np.max([q.max(), q_res.max()])
    vmin = np.min([q.min(), q_res.min()])
    imshowkwargs = dict(vmin=vmin, vmax=vmax)
    axes[0].imshow(q, **imshowkwargs)
    axes[1].imshow(q_res, **imshowkwargs)
    axes[2].imshow(q-q_res, **imshowkwargs)
    plt.show()
