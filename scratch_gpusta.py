#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import cupy as cp
from tqdm import tqdm

import analysis_scripts as asc
import miscfuncs as msc

from datetime import datetime

from scratch_stimulusclass import OMB
from scratch_calcsta import calcallstas


def calcallstas_gpu(stim, filter_length, allspikes):
    """
    Multiple cells, three stimulus dimensions (2 space, 1 time i.e. checkerflicker)
    """
    total_length = stim.shape[-1]
    sx, sy = stim.shape[:-1]

    mempool = cp.get_default_memory_pool()
    # HINT: this should be based on the free memory at the time
    # mempool.free_bytes()
    desired_chunk_size = 3000000

    # Length of the chunks (specified in number of frames)
    chunklength = int(desired_chunk_size/(sx*sy))

#    chunksize = chunklength*sx*sy
    nrofchunks = int(np.ceil(total_length/chunklength))

    allspikes = cp.array(allspikes, dtype=cp.int8)
#    stim = cp.array(stim)
    stas = cp.empty((allspikes.shape[0], sx, sy, filter_length), dtype=cp.float32)
    for i in tqdm(range(nrofchunks)):
        if (i+1)*chunklength < total_length:
            chunkind = slice(i*chunklength, (i+1)*chunklength)
#            chunkend = chunklength
        else:
            chunkind = slice(i*chunklength, None)
#            chunkend = total_length - i*chunklength

        rw = asc.rolling_window(stim[..., chunkind], filter_length)
        stas += cp.einsum('abcd,ec->eabd', rw, allspikes[:, chunkind])

    stas /= allspikes.sum(axis=(-1))[:, cp.newaxis,
                                      cp.newaxis, cp.newaxis]
    stas = cp.asnumpy(stas)
    # Free the GPU memory after calculations are finished
    mempool.free_all_blocks()
    cp.get_default_pinned()
    return stas

if __name__ == '__main__':
#    data = np.load('/home/ycan/Downloads/sta_test.npz')
#    all_spikes = data['all_spikes']
#    stim = data['stim']
#    stas_np = data['stas_np']
#    filter_length = 20

    exp, ombstimnr = '20180710', 8
    checkerstimnr = 6

    st = OMB(exp, ombstimnr)
    filter_length = st.filter_length

    all_spikes = np.zeros((st.nclusters, st.ntotal))
    for i in range(st.nclusters):
        all_spikes[i, :] = st.binnedspiketimes(i)[:-1]
    stim = st.generatecontrast(st.texpars.noiselim/2, 100)

    start = datetime.now()
    _ = cp.array([1])
    print(f'{msc.timediff(start)} for overhead')

    start = datetime.now()
    stas_cp = calcallstas_gpu(stim, filter_length, all_spikes)
    elapsed_gpu = msc.timediff(start)
    print(f'{elapsed_gpu} for gpu')


    start = datetime.now()
    stas_np = calcallstas(stim, filter_length, all_spikes)
    elapsed_nogpu = msc.timediff(start)
    print(f'{elapsed_nogpu} for nogpu')

    print('Equal with einsum STA ? ', np.allclose(stas_np, stas_cp))
    print(f'Gpu speed up factor: {elapsed_nogpu/elapsed_gpu:4.2f}')

