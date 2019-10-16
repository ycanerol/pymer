#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
#%%
import numpy as np
import matplotlib.pyplot as plt

import analysis_scripts as asc
from stimulus import Stimulus

from randpy import randpy
from train_test_split import train_test_split

exp, stimnr = '20180710', 1

ff = Stimulus(exp, stimnr)
#ff.frametimings = asc.ft_nblinks(exp, stimnr, nblinks=1, refresh_rate=ff.refresh_rate)

filter_length = 20
stim = np.array(randpy.gasdev(ff.param_file['seed'], ff.frametimings.shape[0])[0])
stimroll = asc.rolling_window(stim, filter_length, True)

allspikes = ff.allspikes()

stas = (allspikes @ stimroll) / allspikes.sum(axis=1)[:, None]
plt.plot(stas.T)
#%%

glmreg = np.linalg.inv(stimroll.T @ stimroll) @ (stimroll.T @ allspikes.T)
plt.plot(glmreg)
plt.show()
#%%
from pyglmnet import GLMCV

i = 0
spikes = allspikes[i]
sp_train, sp_test, st_train, st_test = train_test_split(spikes, stim)
st_train = asc.rolling_window(st_train, filter_length)
st_test = asc.rolling_window(st_test, filter_length)

reg_lambda = np.logspace(np.log(1e-6), np.log(1e-8), 100, base=np.exp(1))

# from https://github.com/pillowlab/GLMspiketraintutorial/blob/master/python/tutorial1_PoissonGLM.ipynb
glm = GLMCV(distr='poisson', verbose=False, alpha=0.05,
            max_iter=1000, learning_rate=2e-1, score_metric='pseudo_R2',
            reg_lambda=reg_lambda, eta=4.0, cv=10)

glm.fit(st_train, sp_train)

predspikes = glm.predict(st_test)
mse = np.mean((sp_test-predspikes)**2)
#%%
plt.plot(sp_test)
plt.plot(predspikes)


