#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
import analysis_scripts as asc

filter_length = None
stimdim = None

def conv(k, x):
    return np.convolve(k, x, 'full')[:-k.shape[0]+1]


def conv2d_old(Q, x):
    """
    Calculate the quadratic form. Equivalent to conv2d(), but slower.
    """
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


def flattenpars(k, Q, mu):
    """
    Flatten a set of parameters to be used with optimization
    functions.

    Inverse operation of splitpars.
    """
    kQmu = np.concatenate((k.ravel(), Q.ravel(), [mu]))
    return kQmu


def splitpars(kQmu):
    """
    Split the flattened array into original shape

    Inverse operation of flattenpars.
    """
    k, Q, mu = np.split(kQmu,
                        [filter_length*stimdim,
                        stimdim*(filter_length+filter_length**2)])
    k = k.reshape((stimdim, filter_length))
    Q = Q.reshape((stimdim, filter_length, filter_length))
    return k, Q, mu.squeeze()


def gqm_in(k, Q, mu):
    """
    Given a set of parameters,
    calculates the time series that go into exponential function
    """
    def f(x):
#        if len(x.shape)==2:
        total = 0
        for j in range(stimdim):
            total += conv(k[j, :], x[j, :]) + conv2d(Q[j, :, :], x[j, :])
        return total + mu
#        return conv(k, x) + conv2d(Q, x) + mu
    return f


def gqm_neuron(k, Q, mu, time_res):
    """
    Given a set of filters, return the firing rate of a neuron that would respond
    to a stimulus

    The output is scaled by the length of time bins
    """
    def fr(x):
        return np.exp(gqm_in(k, Q, mu)(x))*time_res
    return fr


def makeQ(t):
    """
    Quadratic filter example
    """
    x, y = np.meshgrid(t, t)
    Q = (-(x-0.18)**2/205) + (-(y-0.4)**2/415)
    return Q


#%%
def makeQ2(t):
    """
    Quadratic filter example 2
    """
    k1 = np.exp(-(t-0.12)**2/.0052)
    k2 = np.exp(-(t-.17)**2/.0023)-np.exp(-(t-.27)**2/.01)
    k3 = np.exp(-(t-0.32)**2/.004)
    ws = [.7, .67, -.8]
    ks = (k1, k2, k3)
    Q = np.zeros((t.shape[0], t.shape[0]))
    for k, w in zip(ks, ws):
        Q += w*np.outer(k, k)
    return Q, ks, ws


def minimize_loglikelihood(k_initial, Q_initial, mu_initial,
                           x, time_res, spikes, usegrad=True,
                           method='CG', minimize_disp=False,
                           **kwargs):
    """
    Calculate the filters that minimize the log likelihood function for a
    given set of spikes and stimulus.

    Parameters
    --------
    k_initial, Q_initial, mu_initial:
        Initial guesses for the parameters.
    x:
        The stimulus. Last axis should be temporal, and number of
        stimulus dimensions should match the initial guesses for parameters.
    time_res:
        Length of each bin (referred also as Delta, frame_duration)
    spikes:
        Binned spikes, must have the same shape as the stimulus
    usegrad:
        Whether to use gradients for optimiziation. If set to False, only
        approximated gradients will be used with the appropriate optimization
        method.
    method:
        Optimization method to use, see the Notes section in the  documentation of
        scipy.minimize for a full list.
    minimize_disp:
        Whether to print the convergence messages of the optimization function
    """
    kQmu_initial = flattenpars(k_initial, Q_initial, mu_initial)

    # Infer the filter length from the shape of the initial guesses and
    # set it globally so that other functions can also use it.
    global filter_length, stimdim
    if filter_length is None:
        filter_length = k_initial.shape[-1]
    if stimdim is None:
        if x.ndim>1:
            stimdim = x.shape[0]
        else:
            stimdim = 1

    def loglikelihood(kQmu):
        """
        Define the likelihood function for GQM

        :math: \\mathcal{L}(k, Q, mu) =
        """
        # Star before an argument expands (or unpacks) the values
        P = gqm_in(*splitpars(kQmu))
        return -np.sum(spikes*P(x)) + time_res*np.sum(np.exp(P(x)))

    # Initialize a N-D numpy array to keep outer products
    sTs = np.zeros((stimdim, spikes.shape[0], filter_length, filter_length))
    # Instead of iterating over each time bin, use the rolling window function
    # The expression in the brackets inverts the array.
    xr = asc.rolling_window(x, filter_length)[..., ::-1]
    # Add one extra dimension at the beginning in case the stimulus is
    # single dimensional
    xr = xr[None, ...] if stimdim==1 else xr
    for j in range(stimdim):
        for i in range(spikes.shape[0]-filter_length):
            x_temp = xr[j, i, :]
            sTs[j, i, :, :] = np.outer(x_temp, x_temp)
    def gradients(kQmu):
        """
        Calculate gradients for the log-likelihood function
        """
        k, Q, mu = splitpars(kQmu)
        P = np.exp(gqm_in(k, Q, mu)(x))
        # Fast way of calculating gradients using rolling window and einsum
#        dLdk = spikes @ xr - time_res*(P @ xr)
        dLdk = (np.einsum('j,mjk->mk', spikes, xr)
             - time_res*np.einsum('j,mjk->mk', P, xr))
        # Using einsum to multiply and sum along the desired axis.
        # more detailed explanation here:
        # https://stackoverflow.com/questions/26089893/understanding-numpys-einsum
        dLdq = (np.einsum('mijk,i->mjk', sTs, spikes)
                - time_res*np.einsum('mijk,i->mjk', sTs, P))
        dLdmu = spikes.sum() - time_res*np.sum(P)

        dL = flattenpars(dLdk, dLdq, dLdmu)
        return -dL

    minimizekwargs = {'options':{'disp':minimize_disp}}
    if usegrad:
        minimizekwargs.update({'jac':gradients})
    minimizekwargs.update(kwargs)

    res = minimize(loglikelihood, kQmu_initial, tol=1e-5,
                   method=method, **minimizekwargs)
    return res

#%%
# If the script is being imported from elsewhere to use the functions, do not run the simulation
if __name__ == '__main__':
    filter_length = 20
    stimdim = 3
    frame_rate = 60
    time_res = (1/frame_rate)
    tstop = 100 # simulation length in seconds
    t = np.arange(0, tstop, time_res)
    # Set the seed for PRNG for reproducibility
    np.random.seed(1221)

    stim = np.random.normal(size=(3, t.shape[0]))

    tmini = t[:filter_length]

    mu_in = .3
    k_in = np.exp(-(tmini-0.12)**2/.002)*.5
    k_in = np.stack((k_in, -k_in, -k_in/2))


    Q_in, Qks, Qws = makeQ2(tmini)
    Q_in *= .14
    Q_in = np.stack((Q_in, -Q_in, -Q_in/2))

    f = gqm_neuron(k_in, Q_in, mu_in, time_res)
    rate = f(stim)

    spikes = np.random.poisson(rate)
    plt.plot(spikes)
    plt.show()
    print(spikes.sum(), ' spikes generated')

    # Change the options here
    minimize_disp = True
    usegrad = True

    #%%
    import time
    start = time.time()
    #res = minimize_loglikelihood(k_in, Q_in, mu_in, stim, time_res, spikes)
    res = minimize_loglikelihood(np.zeros(k_in.shape), np.zeros(Q_in.shape), 0,
                                 stim, time_res, spikes,
                                 usegrad=usegrad,
                                 minimize_disp=minimize_disp)
    elapsed = time.time()-start
    print(f'Time elapsed: {elapsed/60:6.1f} mins')
    #%%
    k_out, Q_out, mu_out = splitpars(res.x)

    fig, axes = plt.subplots(3, stimdim)
    for j in range(stimdim):
        axk = axes[0, j]
        axk.plot(k_in[j, :], label='k_in')
        axk.plot(k_out[j, :], label='k_out')
        axk.set_title(f'x_{j}')
        axk.legend()

        axk.text(.8, .2,
                 'mu_in:  {:4.2f}\nmu_out: {:4.2f}'.format(mu_in, mu_out),
                 transform=axk.transAxes)

        axq1 = axes[1, j]
        axq2 = axes[2, j]
        imq1 = axq1.imshow(Q_in[j, ...])
        plt.colorbar(imq1, ax=axq1, format='%.0e')
        imq2 = axq2.imshow(Q_out[j, ...])
        plt.colorbar(imq2, ax=axq2, format='%.0e')
        savepath = '/home/ycan/Documents/meeting_notes/2018-12-05/'
        #plt.savefig(savepath+'simulatedsuccess.pdf')
        #plt.savefig(savepath+'simulatedsuccess.png')
    plt.show()
#%%
    w_in, v_in = eigh(Q_in)
    w_out, v_out = eigh(Q_out)

    [plt.plot(Qk*Qw, color='C1') for Qk, Qw in zip(Qks, Qws)]
    plt.plot(v_in[:, [0, -2, -1]], color='C0')
    plt.plot(v_out[:, [0, -2, -1]], color='C2')
    plt.show()
