#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generalized quadratic model without the mu parameter.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import hankel, eigh
from scipy.optimize import minimize
import analysis_scripts as asc

filter_length = None

#%%
def conv(k, x):
    return np.convolve(k, x, 'full')[:-k.shape[0]+1]


def conv2d_old(Q, x):
    l = Q.shape[0]
    out = np.zeros((x.shape[0]-l+1))
    for i in range(x.shape[0]-l+1):
        s = x[i:i+l]
        res = s[:, None].T @ Q @ s
        out[i] = res
    return out


def conv2d(Q, x, optimize='greedy'):
    """
    Calculate the quadratic form for each time bin for generalized quadratic
    model.

    Uses
    * rolling window to reduce used memory
    * np.broadcast_to for shaping the quadratic filter matrix in the required
      form without allocating memory
    """
    l = Q.shape[0]
    # Generate a rolling view of the stimulus wihtout allocating space in memory
    # Equivalent to "xr = hankel(x)[:, :l]" but much more memory efficient
    xr = asc.rolling_window(x, l)[:, ::-1]
    # Stack copies of Q along a new axis without copying in memory.
    Qb = np.broadcast_to(Q, (x.shape[0], *Q.shape))
    return np.einsum('ij,ijk,ki->i', xr, Qb, xr.T, optimize=optimize)


#%%

def flattenpars(k, Q):
    """
    Flatten a set of parameters to be used with optimization
    functions.

    Inverse operation of splitpars.
    """
    kQ = np.concatenate((k, Q.ravel()))
    return kQ


def splitpars(kQ):
    """
    Split the flattened array into original shape

    Inverse operation of flattenpars.
    """
    k, Q = np.split(kQ, [filter_length])
    Q = Q.reshape((filter_length, filter_length))
    return k, Q


def gqm_in(k, Q):
    """
    Given a set of parameters,
    calculates the time series that go into exponential function
    """
#    k, Q, mu = splitpars(kQmu)
    def f(x):
        return conv(k, x) + conv2d(Q, x)
    return f


def gqm_neuron(k, Q):
    def fr(x):
        return np.exp(gqm_in(k, Q)(x))
    return fr


def makeQ(t):
    x, y = np.meshgrid(t, t)
    Q = (-(x-0.18)**2/205) + (-(y-0.4)**2/415)
    return Q


#%%
def makeQ2(t):
    k1 = np.exp(-(t-0.12)**2/.0052)
    k2 = np.exp(-(t-.17)**2/.0023)-np.exp(-(t-.27)**2/.01)
    k3 = np.exp(-(t-0.32)**2/.004)
    ws = [.7, .67, -.8]
    ks = (k1, k2, k3)
    Q = np.zeros((t.shape[0], t.shape[0]))
    for k, w in zip(ks, ws):
        Q += w*np.outer(k, k)
    return Q, ks, ws


from scipy.optimize import check_grad, approx_fprime

def minimize_loglikelihood(k_initial, Q_initial,
                           x, time_res, spikes,
                           usegrad=True, method='CG', minimize_disp=False,
                           **kwargs):
    kQ_initial = flattenpars(k_initial, Q_initial)

    # Infer the filter length from the shape of the initial guesses and
    # set it globally so that other functions can also use it.
    global filter_length
    if filter_length is None:
        filter_length = k_initial.shape[0]
    # Trim away the first filter_length elements to align spikes array
    # with the output of the convolution operations
    if spikes.shape[0] == x.shape[0]:
#        print('spikes array reshaped while fitting GQM likelihood')
#        spikes = spikes[filter_length-1:]
        pass
    def loglikelihood(kQ):
        P = gqm_in(*splitpars(kQ))
        return -(np.sum(spikes*P(x) - time_res*np.sum(np.exp(P(x)))))
    # Instead of iterating over each time bin, generate a hankel matrix
    # from the stimulus vector and operate on that using matrix
    # multiplication like so: X @ xh , where X is a vector containing
    # some number for each time bin.

#    xh = hankel(x)[:, :filter_length]
    xr = asc.rolling_window(x, filter_length)[:, ::-1]
    sTs = np.zeros((spikes.shape[0], filter_length, filter_length))
    for i in range(spikes.shape[0]-filter_length):
#        x_temp = x[i:i+filter_length][np.newaxis,:]
        x_temp = xr[i, :]
        sTs[i, :, :] = np.outer(x_temp, x_temp)
    # Stimulus length in seconds, found this empirically.
    k_correction = x.shape[0]*time_res*xr.sum(axis=0)
    plt.plot(np.diag(sTs.sum(axis=0)));plt.title('diag(sTs.sum(axis=0))');plt.show()
#    import pdb; pdb.set_trace()
    q_correction = x.shape[0]*time_res*sTs.sum(axis=0) + np.eye(filter_length)*x.shape[0]
#    q_correction = x.shape[0]*time_res*sTs.sum(axis=0) + np.diag(sTs.sum(axis=0))
    def gradients(kQ):
        k, Q = splitpars(kQ)
        P = np.exp(gqm_in(k, Q)(x))
#        dLdk = np.zeros(k.shape)
#        dLdq = np.zeros(Q.shape)
#        dLdmu = 0
#        for i in range(filter_length, x_mini.shape[0]):
#            s = x[i:i+filter_length]
#            dLdk += (spikes[i] * s -
#                       time_res*P[i]*s)
#            dLdq += (spikes[i] * np.outer(s,s) - time_res*P[i] * np.outer(s, s))
#            dLdmu += spikes[i] - time_res * P[i]
        dLdk = spikes @ xr - time_res*(P @ xr)
        dLdk -= k_correction
        # Using einsum to multiply and sum along the desired axis.
        # more detailed explanation here:
        # https://stackoverflow.com/questions/26089893/understanding-numpys-einsum
        dLdq = (np.einsum('ijk,i->jk', sTs, spikes)
                - time_res*np.einsum('ijk,i->jk', sTs, P))
        dLdq -= q_correction
#        import pdb; pdb.set_trace()

        dL = flattenpars(dLdk, dLdq)
        return -dL

    minimizekwargs = {'options':{'disp':minimize_disp}}
    if usegrad:
        minimizekwargs.update({'jac':gradients})
    minimizekwargs.update(kwargs)

    res = minimize(loglikelihood, kQ_initial, tol=1e-1,
                   method=method, **minimizekwargs)
    return res


#%%
# If the script is being imported from elsewhere, do not run the simulation
if __name__ == '__main__':
    filter_length = 40
    frame_rate = 60
    time_res = (1/frame_rate)
    tstop = 100 # in seconds
    t = np.arange(0, tstop, time_res)
    np.random.seed(12221)
#    np.random.seed(45212) # sum is 0.01 for tstop=500

    stim = np.random.normal(size=t.shape)

    tmini = t[:filter_length]

    k_in = np.exp(-(tmini-0.12)**2/.002)*.2
    Q_in, Qks, Qws = makeQ2(tmini)
    Q_in *= .01

    #Q_in = np.zeros(Q_in.shape)


    f = gqm_neuron(k_in, Q_in)
    rate = f(stim)

    spikes = np.random.poisson(rate)
    plt.plot(spikes)
    plt.show()

    minimize_disp = True

    #%%
    import time
    start = time.time()
    #res = minimize_loglikelihood(k_in, Q_in, u_in, stim, time_res, spikes)
    res = minimize_loglikelihood(np.zeros(k_in.shape), np.zeros(Q_in.shape),
                                 stim, time_res, spikes,
                                 minimize_disp=minimize_disp)
    elapsed = time.time()-start
    print(f'Time elapsed: {elapsed/60:6.1f} mins')
    #%%
    k_out, Q_out = splitpars(res.x)

    axk = plt.subplot(211)
    axk.plot(k_in, label='k_in')
    axk.plot(k_out, label='k_out')
    axk.legend()

    axq1 = plt.subplot(223)
    axq2 = plt.subplot(224)
    axq1.imshow(Q_in)
    axq2.imshow(Q_out)
    savepath= '/home/ycan/Documents/meeting_notes/2019-01-16/'
    #plt.savefig(savepath+'simulatedsuccess.pdf')
    #plt.savefig(savepath+'simulatedsuccess.png')
    plt.show()


    w_in, v_in = eigh(Q_in)
    w_out, v_out = eigh(Q_out)

    [plt.plot(Qk*Qw, color='C1') for Qk, Qw in zip(Qks, Qws)]
    plt.plot(v_in[:, [0, -2, -1]], color='C0')
    plt.plot(v_out[:, [0, -2, -1]], color='C2')
    plt.show()
