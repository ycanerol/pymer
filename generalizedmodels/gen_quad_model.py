#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import hankel, eigh
from scipy.optimize import minimize

#%%
def conv(k, x):
    return np.convolve(k, x, 'full')[k.shape[0]-1:-k.shape[0]+1]


def conv2d(Q, x):
    l = Q.shape[0]
    out = np.zeros((x.shape[0]-l+1))
    for i in range(x.shape[0]-l+1):
        s = x[i:i+l]
        res = s[:, None].T @ Q @ s
        out[i] = res
    return out


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

def minimize_loglikelihood(k_initial, Q_initial, mu_initial, x, time_res, spikes):
    kQmu_initial = flattenpars(k_initial, Q_initial, mu_initial)
    x_mini = x[filter_length-1:]
    def loglikelihood(kQmu):
        P = gqm_in(*splitpars(kQmu))
        return -(np.sum(spikes*P(x) - time_res*np.sum(np.exp(P(x)))))
    # Instead of iterating over each time bin, generate a hankel matrix
    # from the stimulus vector and operate on that using matrix
    # multiplication like so: X @ xh , where X is a vector containing
    # some number for each time bin.

    xh = hankel(x_mini)[:, :filter_length]
    sTs = np.zeros((spikes.shape[0], filter_length, filter_length))
    for i in range(spikes.shape[0]):
        x_temp = x[i:i+filter_length][np.newaxis, :]
        sTs[i, :, :] = np.dot(x_temp.T, x_temp)
#    import pdb; pdb.set_trace()

    def gradients(kQmu):
        k, Q, mu = splitpars(kQmu)
        P = np.exp(gqm_in(k, Q, mu)(x))
        dLdk = spikes @ xh - time_res*(P @ xh)

        # Using einsum to multiply and sum along the desired axis.
        # more detailed explanation here:
        # https://stackoverflow.com/questions/26089893/understanding-numpys-einsum
        dLdq = (np.einsum('ijk,i', sTs, spikes)
                - time_res*np.einsum('ijk,i', sTs, P))
        dLdmu = spikes.sum() - time_res*np.sum(P)
#        import pdb; pdb.set_trace()
        dL = flattenpars(dLdk, dLdq, dLdmu)
        return -dL

    print('Gradient diff', check_grad(loglikelihood, gradients, kQmu_initial))

#    print('approx. gradient', approx_fprime(kQmu_initial, loglikelihood, 1e-2))


    res = minimize(loglikelihood, kQmu_initial, tol=1e-2,
                   method='CG',
                   jac=gradients,
                   options={'disp':True})
    return res


#%%
filter_length = 40
frame_rate = 60
time_res = (1/frame_rate)
tstop = 20 # in seconds
t = np.arange(0, tstop, time_res)
np.random.seed(1221)

stim = np.random.normal(size=t.shape)*.2

tmini = t[:filter_length]

mu_in = .01
k_in = np.exp(-(tmini-0.12)**2/.002)
Q_in, Qks, Qws = makeQ2(tmini)

Q_in = np.zeros(Q_in.shape)


f = gqm_neuron(k_in, Q_in, mu_in)
rate = f(stim)

spikes = np.random.poisson(rate)
plt.plot(spikes)
plt.show()
#%%
import time
start = time.time()
res = minimize_loglikelihood(k_in, Q_in, mu_in, stim, time_res, spikes)
#res = minimize_loglikelihood(np.zeros(k_in.shape), np.zeros(Q_in.shape), 0, stim, time_res, spikes)
elapsed = time.time()-start
print(f'Time elapsed: {elapsed/60:6.1f} mins')

k_out, Q_out, mu_out = splitpars(res.x)

axk = plt.subplot(211)
axk.plot(k_in, label='k_in')
axk.plot(k_out, label='k_out')
axk.legend()

axk.text(filter_length*.8, 0.5, f'mu_in:  {mu_in:4.2f}\nmu_out: {mu_out:4.2f}')

axq1 = plt.subplot(223)
axq2 = plt.subplot(224)
axq1.imshow(Q_in)
axq2.imshow(Q_out)

plt.show()

#%%
w_in, v_in = eigh(Q_in)
w_out, v_out = eigh(Q_out)

[plt.plot(Qk*Qw, color='C1') for Qk, Qw in zip(Qks, Qws)]
plt.plot(v_in[:, [0, -2, -1]], color='C0')
plt.plot(v_out[:, [0, -2, -1]], color='C2')
plt.show()
