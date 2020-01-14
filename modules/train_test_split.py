#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import numpy as np


def train_test_split(spikes, stimulus, test_size=0.2, split_pos=None):
    """
    split_pos:
        Where the split should start as a ratio.

    Returns:
        spikes_training, spikes_test, stimulus_training, stimulus_test
    """
    total_len = spikes.shape[0]
    if split_pos is None:
        split_pos = np.random.rand()*(1-test_size)
    split_ind = int(split_pos * total_len)
    split_end = split_ind + int(test_size * total_len)
    mask = np.array([True]*total_len)
    mask[split_ind:split_end] = False
    spikes_training = spikes[mask]
    spikes_test = spikes[~mask]
    stimulus_training = stimulus[..., mask]
    stimulus_test = stimulus[..., ~mask]
    return spikes_training, spikes_test, stimulus_training, stimulus_test


#%%
if __name__ == '__main__':

    from omb import OMB
    import gen_quad_model_multidimensional as gqm

    st = OMB('20180710', 8)
    i = 0
    spikes = st.allspikes()[i]

    stimulus = st.bgsteps
    stimdim = stimulus.shape[0]
    fl = st.filter_length

    sp_train, sp_test, stim_train, stim_test = train_test_split(spikes, stimulus,
                                                                0.1, split_pos=0.1)

    res = gqm.minimize_loglikelihood(np.zeros((stimdim, fl)),
                                     np.zeros((stimdim, fl, fl)), 0,
                                     stim_train,
                                     st.frame_duration,
                                     sp_train,
                                     minimize_disp=True,
                                     method='BFGS')

    k_out, Q_out, mu_out = gqm.splitpars(res.x)

    firing_rate = gqm.gqm_neuron(k_out, Q_out, mu_out, st.frame_duration)(stim_test)
    cross_corr = np.corrcoef(sp_test, firing_rate)[0, 1]

    train_cc = np.corrcoef(sp_train,
                           gqm.gqm_neuron(k_out,
                                          Q_out,
                                          mu_out,
                                          st.frame_duration)(stim_train))[0, 1]

    print(f'Cross-corr test: {cross_corr:4.2f} with training data: {train_cc:4.2f}')
