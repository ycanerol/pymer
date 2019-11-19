#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

import pyglmnet

import analysis_scripts as asc
import nonlinearity as nlt
from omb import OMB
from train_test_split import train_test_split

st = OMB('20180710', 8)
use_motion = True
use_contrast = True
st.filter_length = 10
allspikes = st.allspikes()

stimdim = 0
if use_contrast:
    stimdim += 1
if use_motion:
    stimdim += 2

stimulus = np.zeros((stimdim, allspikes.shape[1]))

if use_motion:
    stimulus[:2, :] = st.bgsteps

i = 0
spikes = allspikes[i]
if use_contrast:
    stimulus[-1, :] = st.contrast_signal_cell(i)
sp_train, sp_test, st_train, st_test = train_test_split(spikes, stimulus)
#st_train_rw = asc.rolling_window(st_train, st.filter_length)
#st_test_rw = asc.rolling_window(st_test, st.filter_length)

reg_lambda = np.logspace(np.log(1e-7), np.log(1e-11), 8, base=np.exp(1))

#model = pyglmnet.GLMCV(reg_lambda=reg_lambda, eta=4.0,
#                       score_metric='pseudo_R2', verbose=True)
#
#model.fit(st_train_rw, sp_train)


#%%
def check_stim_arr_dimensions(stimulus):
    if not stimulus.ndim == 2:
        raise ValueError("Stimulus array should have exactly two dimensions.")


def flatten_time_series(stimulus, filter_length):
    check_stim_arr_dimensions(stimulus)
    rw = asc.rolling_window(stimulus, filter_length)
    flat_rw = rw.reshape((stimulus.shape[1], -1))
    return flat_rw, rw


def make_quadratic_time_series(stimulus, filter_length, rw=None):
    """
    Calculate the quadratic time series that will be used to optimize
    GQM parameters.

    Returns
    ----
    sTs:
        shape:(stimulus dimensions, N time bins, filter_length, filter_length )
    """
    check_stim_arr_dimensions(stimulus)
    if rw is None:
        rw = asc.rolling_window(stimulus, filter_length)
    sTs = np.zeros((*stimulus.shape[:2], filter_length, filter_length))
    for j in range(stimdim):  # stimdim
        for i in range(stimulus.shape[1]):
            x_temp = rw[j, i, :]
            sTs[j, i, ...] = np.outer(x_temp, x_temp)
    return sTs


def flatten_quadratic_time_series(sTs):
    """
    Reshape the quadratic time series so that the first dimension
    is time, and the rest of the dimensions are flat.

    sTs should have the shape (stimulus dimensions, N time bins, filter_length, filter_length )
    """
    arr = sTs.copy()
    arr = np.moveaxis(arr, 0, 1)
    shape = arr.shape
    return arr.reshape(shape[0], -1)


def design_matrix_quadratic(stimulus, filter_length):
    flat_rw, rw = flatten_time_series(stimulus, filter_length)
    sTs = flatten_quadratic_time_series(make_quadratic_time_series(stimulus, filter_length, rw=rw))
    design_matrix = np.hstack((flat_rw, sTs))
    return design_matrix


#sTs = np.zeros((st_train.shape[0], st.filter_length, st.filter_length))
#for i in range(st_train.shape[0]-st.filter_length):
#    x_temp = st_train[i, :]
#    sTs[i, ...] = np.outer(x_temp, x_temp)

#sTs_gqm = make_quadratic_time_series(st_train, st.filter_length)
desgn_mat_train = design_matrix_quadratic(st_train, st.filter_length)
desgn_mat_test = design_matrix_quadratic(st_test, st.filter_length)

modelgqm = pyglmnet.GLMCV(reg_lambda=reg_lambda, eta=4.0,
                          score_metric='pseudo_R2', cv=2)


start = datetime.now()
modelgqm.fit(desgn_mat_train, sp_train)
elapsed = datetime.now() - start
print(f'Took {elapsed.total_seconds()/60:4.2f} minutes')

k = modelgqm.beta_[:st.filter_length*stimdim].reshape((stimdim, st.filter_length))
Q = modelgqm.beta_[st.filter_length*stimdim:].reshape((stimdim,
                  st.filter_length, st.filter_length))
pred = modelgqm.predict(design_matrix_quadratic(st_test, st.filter_length))

#%%

plt.plot(k.T)
fig, axes = plt.subplots(stimdim, 5, figsize=(15, 5))
for j in range(stimdim):
    axk = axes[j, 0]
    axk.plot(k[j, :])
    axq = axes[j, 1]
    im = axq.imshow(Q[j, ...])
    plt.colorbar(im, ax=axq)

    w, v = np.linalg.eigh(Q[j, ...])
    axw, axv, axn = axes[j, 2:5]

    axw.plot(w, 'ko')
    eiginds = [0, 1, st.filter_length-2, st.filter_length-1]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for ind, (eigind, w) in enumerate(zip(eiginds, w[eiginds])):
        axw.plot(eigind, w, 'o', color=colors[ind])
        axv.plot(v[:, eigind], color=colors[ind])

        generator = np.convolve(v[:, eigind], stimulus[j, :],
                                mode='full')[:-st.filter_length+1]

        nonlinearity, bins = nlt.calc_nonlin(spikes, generator, nr_bins=40)
        axn.plot(bins, nonlinearity/st.frame_duration, color=colors[ind])


#%%
sTBs = np.zeros((st_train.shape[0], st.filter_length, st.filter_length))
for m in range(st.filter_length):
    for n in range(st.filter_length):
        B = np.zeros((st.filter_length, st.filter_length))
        B[m, n] = 1
        for i in range(st_train.shape[0]-st.filter_length):
            x_temp = st_train[i, :]
            sTBs[i, m, n] += x_temp.T @ B @ x_temp



