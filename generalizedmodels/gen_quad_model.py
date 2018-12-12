#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

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

def flattenpars(k, Q, mu):
    """
    Flatten a set of parameters to be used with optimization
    functions.

    Inverse operation of splitpars.
    """
    kQmu = np.concatenate((k, Q.ravel(), [mu]))
    return kQmu


def splitpars(kQmu):
    """
    Split the flattened array into original shape

    Inverse operation of flattenpars.
    """
    k, Q, mu = np.split(kQmu, [filter_length, filter_length+filter_length**2])
    Q = Q.reshape((filter_length, filter_length))
    return k, Q, mu.squeeze()


def gqm_in(k, Q, mu):
    """
    Given a set of parameters,
    calculates the time series that go into exponential function
    """
#    k, Q, mu = splitpars(kQmu)
    def f(x):
        return conv(k, x) + conv2d(Q, x) + mu
    return f


def gqm_neuron(k, Q, mu):
    def fr(x):
        return np.exp(gqm_in(k, Q, mu)(x))
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

def minimize_loglikelihood(k_initial, Q_initial, mu_initial,
                           x, time_res, spikes, debug_grad=False,
                           usegrad=True, method='CG', minimize_disp=False,
                           **kwargs):
    kQmu_initial = flattenpars(k_initial, Q_initial, mu_initial)

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
    def loglikelihood(kQmu):
        P = gqm_in(*splitpars(kQmu))
        return -(np.sum(spikes*P(x) - time_res*np.sum(np.exp(P(x)))))
    # Instead of iterating over each time bin, generate a hankel matrix
    # from the stimulus vector and operate on that using matrix
    # multiplication like so: X @ xh , where X is a vector containing
    # some number for each time bin.

#    xh = hankel(x)[:, :filter_length]
    xr = asc.rolling_window(x, filter_length)[:, ::-1]
    sTs = np.zeros((spikes.shape[0], filter_length, filter_length))
    for i in range(spikes.shape[0]-filter_length):
        x_temp = x[i:i+filter_length][np.newaxis, ::-1]
        sTs[i, :, :] = np.dot(x_temp.T, x_temp)
#    import pdb; pdb.set_trace()

    def gradients(kQmu):
        k, Q, mu = splitpars(kQmu)
        P = np.exp(gqm_in(k, Q, mu)(x))
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

        # Using einsum to multiply and sum along the desired axis.
        # more detailed explanation here:
        # https://stackoverflow.com/questions/26089893/understanding-numpys-einsum
        dLdq = (np.einsum('ijk,i->jk', sTs, spikes)
                - time_res*np.einsum('ijk,i->jk', sTs, P))
        dLdmu = spikes.sum() - time_res*np.sum(P)
#        import pdb; pdb.set_trace()
        dL = flattenpars(dLdk, dLdq, dLdmu)
        return -dL
    if debug_grad:
        eps = 1e-10
        ap_grad = approx_fprime(kQmu_initial, loglikelihood, eps)
        man_grad = gradients(kQmu_initial)
        kda, qda, mda = splitpars(ap_grad)
        kdm, qdm, mdm = splitpars(man_grad)
        diff = ap_grad - man_grad
        k_diff, Q_diff, mu_diff = splitpars(diff)
        print('Gradient diff L2 norm', np.sum(diff**2))
        plt.figure(figsize=(7, 10))
        axk = plt.subplot(411)
        axk.plot(kda, label='Auto grad')
        axk.plot(kdm, label='Manual grad')
        axk.legend()
        axkdif = plt.subplot(412)
        axkdif.plot(k_diff, 'k', label='auto - manual gradient')
        axkdif.legend()
        axqa = plt.subplot(425)
        imqa = axqa.imshow(qda)
        axqa.set_title('Auto grad Q')
        plt.colorbar(imqa)
        axqm = plt.subplot(426)
        axqm.set_title('Manual grad Q')
        imqm = axqm.imshow(qdm)
        plt.colorbar(imqm)
        axqdif = plt.subplot(427)
        imqdif = axqdif.imshow(Q_diff)
        plt.colorbar(imqdif)
        plt.suptitle(f'Difference of numerical and explicit gradients, mu_diff: {mu_diff:11.2f}')
        plt.show()
#        import pdb; pdb.set_trace();
        return kda, qda, mda, kdm, qdm, mdm

    minimizekwargs = {'options':{'disp':minimize_disp}}
    if usegrad:
        minimizekwargs.update({'jac':gradients})
    minimizekwargs.update(kwargs)

    res = minimize(loglikelihood, kQmu_initial, tol=1e-1,
                   method=method, **minimizekwargs)
    return res


#%%
if __name__ == '__main__':
    filter_length = 40
    frame_rate = 60
    time_res = (1/frame_rate)
    tstop = 100 # in seconds
    t = np.arange(0, tstop, time_res)
    np.random.seed(1221)

    stim = np.random.normal(size=t.shape)*.2

    tmini = t[:filter_length]

    mu_in = .01
    k_in = np.exp(-(tmini-0.12)**2/.002)
    Q_in, Qks, Qws = makeQ2(tmini)

    #Q_in = np.zeros(Q_in.shape)


    f = gqm_neuron(k_in, Q_in, mu_in)
    rate = f(stim)

    spikes = np.random.poisson(rate)
    plt.plot(spikes)
    plt.show()

    debug_grad = True
    minimize_disp = True

    #%%
    import time
    start = time.time()
    #res = minimize_loglikelihood(k_in, Q_in, mu_in, stim, time_res, spikes)
    res = minimize_loglikelihood(np.zeros(k_in.shape), np.zeros(Q_in.shape), 0,
                                 stim, time_res, spikes,
                                 debug_grad=debug_grad, minimize_disp=minimize_disp)
    elapsed = time.time()-start
    print(f'Time elapsed: {elapsed/60:6.1f} mins')
#%%
    if not debug_grad:
        k_out, Q_out, mu_out = splitpars(res.x)

        axk = plt.subplot(211)
        axk.plot(k_in, label='k_in')
        axk.plot(k_out, label='k_out')
        axk.legend()

        axk.text(filter_length*.8, 0.5,
                 f'mu_in:  {mu_in:4.2f}\nmu_out: {mu_out:4.2f}')

        axq1 = plt.subplot(223)
        axq2 = plt.subplot(224)
        axq1.imshow(Q_in)
        axq2.imshow(Q_out)
        savepath= '/home/ycan/Documents/meeting_notes/2018-12-05/'
        #plt.savefig(savepath+'simulatedsuccess.pdf')
        #plt.savefig(savepath+'simulatedsuccess.png')
        plt.show()


        w_in, v_in = eigh(Q_in)
        w_out, v_out = eigh(Q_out)

        [plt.plot(Qk*Qw, color='C1') for Qk, Qw in zip(Qks, Qws)]
        plt.plot(v_in[:, [0, -2, -1]], color='C0')
        plt.plot(v_out[:, [0, -2, -1]], color='C2')
        plt.show()
    else:
        kda, qda, mda, kdm, qdm, mdm = res

        def remdiag(q): return q-np.diag(np.diag(q))

        plt.imshow(remdiag(qda))
        plt.title('Auto grad without diagonal')
        plt.colorbar()
        plt.show()

        plt.imshow(remdiag(qdm))
        plt.title('Manual grad without diagonal')
        plt.colorbar()
        plt.show()




