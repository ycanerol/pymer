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
    """
    Calculate the quadratic form. Equivalent to con2d(), but slower.
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
#    return 0


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
    def f(x):
        return conv(k, x) + conv2d(Q, x) + mu
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


from scipy.optimize import check_grad, approx_fprime

def minimize_loglikelihood(k_initial, Q_initial, mu_initial,
                           x, time_res, spikes, usegrad=True,
                           debug_grad=False, method='CG', minimize_disp=False,
                           **kwargs):
    """
    Calculate the filters that minimize the log likelihood function for a
    given set of spikes and stimulus.

    Parameters
    --------
    k_initial, Q_initial, mu_initial:
        Initial guesses for the parameters.
    x:
        The stimulus
    time_res:
        Length of each bin (referred also as Delta, frame_duration)
    spikes:
        Binned spikes, must have the same shape as the stimulus
    usegrad:
        Whether to use gradients for optimiziation. If set to False, only
        approximated gradients will be used with the appropriate optimization
        method.
    debug_grad:
        Whether to calculate and plot the gradients in the first iteration
        Setting it to True will change the returned values.
    method:
        Optimization method to use, see the Notes section in the  documentation of
        scipy.minimize for a full list.
    minimize_disp:
        Whether to print the convergence messages of the optimization function
    """
    kQmu_initial = flattenpars(k_initial, Q_initial, mu_initial)

    # Infer the filter length from the shape of the initial guesses and
    # set it globally so that other functions can also use it.
    global filter_length
    if filter_length is None:
        filter_length = k_initial.shape[0]

    def loglikelihood(kQmu):
        """
        Define the likelihood function for GQM
        """
        # Star before an argument expands (or unpacks) the values
        P = gqm_in(*splitpars(kQmu))
        return -(np.sum(spikes*P(x) - time_res*np.sum(np.exp(P(x)))))
    # Instead of iterating over each time bin, generate a hankel matrix
    # from the stimulus vector and operate on that using matrix
    # multiplication like so: X @ xh , where X is a vector containing
    # some number for each time bin.

#    xh = hankel(x)[:, :filter_length]
    # Instead of iterating over each time bin, use the rolling window function
    # The expression in the brackets inverts the array.
    xr = asc.rolling_window(x, filter_length)[:, ::-1]
    # Initialize a 3D numpy array to keep outer products
    sTs = np.zeros((spikes.shape[0], filter_length, filter_length))
    for i in range(spikes.shape[0]-filter_length):
#        x_temp = x[i:i+filter_length][np.newaxis,:]
        x_temp = xr[i, :]
        sTs[i, :, :] = np.outer(x_temp, x_temp)
    # Empirically found correction terms for the gradients.
    k_correction = x.shape[0]*time_res*xr.sum(axis=0)
    plt.plot(np.diag(sTs.sum(axis=0)));plt.title('diag(sTs.sum(axis=0))');plt.show()
#    import pdb; pdb.set_trace()
    q_correction = x.shape[0]*time_res*sTs.sum(axis=0) + np.eye(filter_length)*x.shape[0]
#    q_correction = x.shape[0]*time_res*sTs.sum(axis=0) + np.diag(sTs.sum(axis=0))
    mu_correction = (x.shape[0]-1) * x.shape[0]*time_res
    def gradients(kQmu):
        """
        Calculate gradients for the log-likelihood function
        """
        k, Q, mu = splitpars(kQmu)
        P = np.exp(gqm_in(k, Q, mu)(x))
#        Slow way of calculating the gradients
#        dLdk = np.zeros(k.shape)
#        dLdq = np.zeros(Q.shape)
#        dLdmu = 0
#        for i in range(filter_length, x_mini.shape[0]):
#            s = x[i:i+filter_length]
#            dLdk += (spikes[i] * s -
#                       time_res*P[i]*s)
#            dLdq += (spikes[i] * np.outer(s,s) - time_res*P[i] * np.outer(s, s))
#            dLdmu += spikes[i] - time_res * P[i]
        # Fast way of calculating gradients using rolling window and einsum
        dLdk = spikes @ xr - time_res*(P @ xr)
        dLdk -= k_correction
        # Using einsum to multiply and sum along the desired axis.
        # more detailed explanation here:
        # https://stackoverflow.com/questions/26089893/understanding-numpys-einsum
        dLdq = (np.einsum('ijk,i->jk', sTs, spikes)
                - time_res*np.einsum('ijk,i->jk', sTs, P))
        dLdq -= q_correction
        dLdmu = spikes.sum() - time_res*np.sum(P)
        dLdmu -= mu_correction
#        import pdb; pdb.set_trace()

        dL = flattenpars(dLdk, dLdq, dLdmu)
        return -dL
    if debug_grad:
        # Epsilon value to use when approximating the gradient
        eps = 1e-10
        ap_grad = approx_fprime(kQmu_initial, loglikelihood, eps)
        man_grad = gradients(kQmu_initial)
        # Split the auto and manual gradients into k, q and mu
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
#     If debug_grad is True, the function returns on the previous line, rest of the minimize_loglhd function
    # is not executed
    minimizekwargs = {'options':{'disp':minimize_disp}}
    if usegrad:
        minimizekwargs.update({'jac':gradients})
    minimizekwargs.update(kwargs)

    res = minimize(loglikelihood, kQmu_initial, tol=1e-1,
                   method=method, **minimizekwargs)
    return res


#%%
# If the script is being imported from elsewhere to use the functions, do not run the simulation
if __name__ == '__main__':
    filter_length = 20
    frame_rate = 60
    time_res = (1/frame_rate)
    tstop = 100 # simulation length in seconds
    t = np.arange(0, tstop, time_res)
    # Set the seed for PRNG for reproducibility
    np.random.seed(12221)
#    np.random.seed(45212) # sum is 0.01 for tstop=500

    stim = np.random.normal(size=t.shape)

    tmini = t[:filter_length]

    mu_in = .01
    k_in = np.exp(-(tmini-0.12)**2/.002)*.2
    Q_in, Qks, Qws = makeQ2(tmini)
    Q_in *= .01

    #Q_in = np.zeros(Q_in.shape)

    f = gqm_neuron(k_in, Q_in, mu_in, time_res)
    rate = f(stim)

    spikes = np.random.poisson(rate)
    plt.plot(spikes)
    plt.show()
    print(spikes.sum(), ' spikes generated')

    # Change the options here
    debug_grad = True
    minimize_disp = True
    usegrad = False

    #%%
    import time
    start = time.time()
    #res = minimize_loglikelihood(k_in, Q_in, mu_in, stim, time_res, spikes)
    res = minimize_loglikelihood(np.zeros(k_in.shape), np.zeros(Q_in.shape), 0,
                                 stim, time_res, spikes,
                                 usegrad=usegrad,
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

        axk.text(.8, .2,
                 'mu_in:  {:4.2f}\nmu_out: {:4.2f}'.format(mu_in, mu_out),
                 transform=axk.transAxes)

        axq1 = plt.subplot(223)
        axq2 = plt.subplot(224)
        imq1 = axq1.imshow(Q_in)
        plt.colorbar(imq1, ax=axq1, format='%.0e')
        imq2 = axq2.imshow(Q_out)
        plt.colorbar(imq2, ax=axq2, format='%.0e')
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

        def remdiag(q):
            """
            Remove the diagonal for a given matrix.
            """
            return q-np.diag(np.diag(q))

        qdad = np.diag(qda)
        qdmd = np.diag(qdm)
        plt.plot(qdad, label='diag(auto Qd)')
        plt.plot(qdmd, label='diag(manu Qd)')
        plt.legend(fontsize='x-small')
        plt.show()

        plt.plot(qdad-qdmd)
        plt.title('diag(auto Qd- manu Qd)')
        plt.show()

        plt.imshow(remdiag(qda))
        plt.title('Auto grad without diagonal')
        plt.colorbar()
        plt.show()

        plt.imshow(remdiag(qdm))
        plt.title('Manual grad without diagonal')
        plt.colorbar()
        plt.show()
