#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 15:39:11 2017

@author: ycan


Copied from https://github.com/keflavich/gaussfitter/blob/master/gaussfitter/gaussfitter.py

"""


import numpy as np
import sys

try:
    from mpfit import mpfit
except ImportError:
    sys.path.append('/home/ycan/Documents/scripts/external_libs')
    from mpfit import mpfit

def moments(data, circle, rotate, vheight, estimator=np.ma.median, angle_guess=45.0,
            **kwargs):
    """
    Returns (height, amplitude, x, y, width_x, width_y, rotation angle)
    the gaussian parameters of a 2D distribution by calculating its
    moments.  Depending on the input parameters, will only output
    a subset of the above.
    If using masked arrays, pass estimator=np.ma.median
    """
    total = np.abs(data).sum()
    Y, X = np.indices(data.shape)  # python convention: reverse x,y np.indices
    y = np.argmax((X*np.abs(data)).sum(axis=1)/total)
    x = np.argmax((Y*np.abs(data)).sum(axis=0)/total)
    col = data[int(y), :]
    # FIRST moment, not second!
    width_x = np.sqrt(np.abs((np.arange(col.size)-y)*col).sum() / np.abs(col).sum())
    row = data[:, int(x)]
    width_y = np.sqrt(np.abs((np.arange(row.size)-x)*row).sum() / np.abs(row).sum())
    width = (width_x + width_y) / 2.
    height = estimator(data.ravel())
    amplitude = data.max()-height
    mylist = [amplitude, x, y]
    if np.isnan((width_y, width_x, height, amplitude)).any():
        raise ValueError("something is nan")
    if vheight:
        mylist = [height] + mylist
    if not circle:
        mylist = mylist + [width_x, width_y]
        if rotate:
            # rotation "moment" is a little above zero to initiate the fitter
            # with something not locked at the edge of parameter space
            mylist = mylist + [angle_guess]
            # also, circles don't rotate.
    else:
        mylist = mylist + [width]
    return mylist


def twodgaussian(inpars, circle=False, rotate=True, vheight=True, shape=None):
    """
    Returns a 2d gaussian function of the form:
    x' = np.cos(rota) * x - np.sin(rota) * y
    y' = np.sin(rota) * x + np.cos(rota) * y
    (rota should be in degrees)
    g = b + a * np.exp ( - ( ((x-center_x)/width_x)**2 +
    ((y-center_y)/width_y)**2 ) / 2 )
    inpars = [b,a,center_x,center_y,width_x,width_y,rota]
             (b is background height, a is peak amplitude)
    where x and y are the input parameters of the returned function,
    and all other parameters are specified by this function
    However, the above values are passed by list.  The list should be:
    inpars = (height,amplitude,center_x,center_y,width_x,width_y,rota)
    You can choose to ignore / neglect some of the above input parameters using
    the following options:
    Parameters
    ----------
    circle : bool
        default is an elliptical gaussian (different x, y widths), but can
        reduce the input by one parameter if it's a circular gaussian
    rotate : bool
        default allows rotation of the gaussian ellipse.  Can
        remove last parameter by setting rotate=0
    vheight : bool
        default allows a variable height-above-zero, i.e. an
        additive constant for the Gaussian function.  Can remove first
        parameter by setting this to 0
    shape : tuple
        if shape is set (to a 2-parameter list) then returns an image with the
        gaussian defined by inpars
    """
    inpars_old = inpars
    inpars = list(inpars)
    if vheight:
        height = inpars.pop(0)
        height = float(height)
    else:
        height = float(0)
    amplitude, center_y, center_x = inpars.pop(0), inpars.pop(0), inpars.pop(0)
    amplitude = float(amplitude)
    center_x = float(center_x)
    center_y = float(center_y)
    if circle:
        width = inpars.pop(0)
        width_x = float(width)
        width_y = float(width)
        rotate = 0
    else:
        width_x, width_y = inpars.pop(0), inpars.pop(0)
        width_x = float(width_x)
        width_y = float(width_y)
    if rotate:
        rota = inpars.pop(0)
        rota = np.pi/180. * float(rota)
        rcen_x = center_x * np.cos(rota) - center_y * np.sin(rota)
        rcen_y = center_x * np.sin(rota) + center_y * np.cos(rota)
    else:
        rcen_x = center_x
        rcen_y = center_y
    if len(inpars) > 0:
        raise ValueError("There are still input parameters:" + str(inpars) +
                         " and you've input: " + str(inpars_old) +
                         " circle=%d, rotate=%d, vheight=%d" % (circle, rotate, vheight))

    def rotgauss(x, y):
        if rotate:
            xp = x * np.cos(rota) - y * np.sin(rota)
            yp = x * np.sin(rota) + y * np.cos(rota)
        else:
            xp = x
            yp = y
        g = height+amplitude*np.exp(-(((rcen_x-xp)/width_x)**2 +
                                      ((rcen_y-yp)/width_y)**2)/2.)
        return g
    if shape is not None:
        return rotgauss(*np.indices(shape))
    else:
        return rotgauss


def gaussfit(data, err=None, params=(), autoderiv=True, return_error=False,
             circle=False, fixed=np.repeat(False, 7),
             limitedmin=[False, False, False, False, True, True, True],
             limitedmax=[False, False, False, False, False, False, True],
             usemoment=np.array([], dtype='bool'), minpars=np.repeat(0, 7),
             maxpars=[0, 0, 0, 0, 0, 0, 180], rotate=True, vheight=True,
             quiet=True, returnmp=False, returnfitimage=False, **kwargs):
    """
    Gaussian fitter with the ability to fit a variety of different forms of
    2-dimensional gaussian.
    Parameters
    ----------
    data : `numpy.ndarray`
        2-dimensional data array
    err : `numpy.ndarray` or None
        error array with same size as data array.  Defaults to 1 everywhere.
    params : (height, amplitude, x, y, width_x, width_y, rota)
        Initial input parameters for Gaussian function.  If not input, these
        will be determined from the moments of the system, assuming no rotation
    autoderiv : bool
        Use the autoderiv provided in the lmder.f function (the alternative is
        to us an analytic derivative with lmdif.f: this method is less robust)
    return_error : bool
        Default is to return only the Gaussian parameters.
        If ``True``, return fit params & fit error
    returnfitimage : bool
        returns (best fit params,best fit image)
    returnmp : bool
        returns the full mpfit struct
    circle : bool
        The default is to fit an elliptical gaussian (different x, y widths),
        but the input is reduced by one parameter if it's a circular gaussian.
    rotate : bool
        Allow rotation of the gaussian ellipse.  Can remove
        last parameter of input & fit by setting rotate=False.
        Angle should be specified in degrees.
    vheight : bool
        Allows a variable height-above-zero, i.e. an additive constant
        background for the Gaussian function.  Can remove the first fitter
        parameter by setting this to ``False``
    usemoment : `numpy.ndarray`, dtype='bool'
        Array to choose which parameters to use a moment estimation for.  Other
        parameters will be taken from params.
    Returns
    -------
    (params, [parerr], [fitimage]) | (mpfit, [fitimage])
    parameters : list
        The default output is a set of Gaussian parameters with the same shape
        as the input parameters
    fitimage : `numpy.ndarray`
        If returnfitimage==True, the last return will be a 2D array holding the
        best-fit model
    mpfit : `mpfit` object
        If ``returnmp==True`` returns a `mpfit` object. This object contains a
        `covar` attribute which is the 7x7 covariance array generated by the
        mpfit class in the `mpfit_custom.py` module. It contains a `param`
        attribute that contains a list of the best fit parameters in the same
        order as the optional input parameter `params`.
    """
    data = data.view(np.ma.MaskedArray)
    usemoment = np.array(usemoment, dtype='bool')
    params = np.array(params, dtype='float')
    if usemoment.any() and len(params) == len(usemoment):
        moment = np.array(moments(data, circle, rotate, vheight, **kwargs), dtype='float')
        params[usemoment] = moment[usemoment]
    elif params == [] or len(params) == 0:
        params = (moments(data, circle, rotate, vheight, **kwargs))
    if not vheight:
        # If vheight is not set, we set it for sub-function calls but fix the
        # parameter at zero
        vheight = True
        params = np.concatenate([[0], params])
        fixed[0] = 1

    # mpfit will fail if it is given a start parameter outside the allowed range:
    for i in range(len(params)):
        if params[i] > maxpars[i] and limitedmax[i]: params[i] = maxpars[i]
        if params[i] < minpars[i] and limitedmin[i]: params[i] = minpars[i]

    # One time: check if error is set, otherwise fix it at 1.
    err = err if err is not None else 1.0

    def mpfitfun(data, err):
        def f(p, fjac):
            twodg = twodgaussian(p, circle, rotate, vheight)
            delta = (data - twodg(*np.indices(data.shape))) / err
            return [0, delta.compressed()]
        return f

    parinfo = [{'n': 1, 'value': params[1], 'limits': [minpars[1], maxpars[1]],
                'limited': [limitedmin[1], limitedmax[1]], 'fixed': fixed[1],
                'parname': "AMPLITUDE", 'error': 0},
               {'n': 2, 'value': params[2], 'limits': [minpars[2], maxpars[2]],
                'limited': [limitedmin[2], limitedmax[2]], 'fixed': fixed[2],
                'parname': "XSHIFT", 'error': 0},
               {'n': 3, 'value': params[3], 'limits': [minpars[3], maxpars[3]],
                'limited': [limitedmin[3], limitedmax[3]], 'fixed': fixed[3],
                'parname': "YSHIFT", 'error': 0},
               {'n': 4, 'value': params[4], 'limits': [minpars[4], maxpars[4]],
                'limited': [limitedmin[4], limitedmax[4]], 'fixed': fixed[4],
                'parname': "XWIDTH", 'error': 0}]
    if vheight:
        parinfo.insert(0, {'n': 0, 'value': params[0], 'limits': [minpars[0], maxpars[0]],
                           'limited': [limitedmin[0], limitedmax[0]], 'fixed': fixed[0],
                           'parname': "HEIGHT", 'error': 0})
    if not circle:
        parinfo.append({'n': 5, 'value': params[5], 'limits': [minpars[5], maxpars[5]],
                        'limited': [limitedmin[5], limitedmax[5]], 'fixed': fixed[5],
                        'parname': "YWIDTH", 'error': 0})
        if rotate:
            parinfo.append({'n': 6, 'value': params[6], 'limits': [minpars[6], maxpars[6]],
                            'limited': [limitedmin[6], limitedmax[6]], 'fixed': fixed[6],
                            'parname': "ROTATION", 'error': 0})

    if not autoderiv:
        # the analytic derivative, while not terribly difficult, is less
        # efficient and useful.  I only bothered putting it here because I was
        # instructed to do so for a class project - please ask if you would
        # like this feature implemented
        raise NotImplementedError("I'm sorry, I haven't implemented this feature yet.  "
                                  "Given that I wrote this message in 2008, "
                                  "it will probably never be implemented.")
    else:
        mp = mpfit(mpfitfun(data, err), parinfo=parinfo, quiet=quiet)

    if mp.errmsg:
        raise Exception("MPFIT error: {0}".format(mp.errmsg))

    if (not circle) and rotate:
        mp.params[-1] %= 180.0

    mp.chi2 = mp.fnorm
    try:
        mp.chi2n = mp.fnorm/mp.dof
    except ZeroDivisionError:
        mp.chi2n = np.nan

    if returnmp:
        returns = (mp)
    elif return_error:
        returns = mp.params, mp.perror
    else:
        returns = mp.params
    if returnfitimage:
        fitimage = twodgaussian(mp.params, circle, rotate, vheight)(*np.indices(data.shape))
        returns = (returns, fitimage)
    return returns


# %%
#plt.imshow(fit_frame)
p = gaussfit(fit_frame)
fit_func = gaussian(*p)

X, Y = np.meshgrid(np.arange(fit_frame.shape[0]), np.arange(fit_frame.shape[1]))

Z = fit_func(Y, X)

plt.imshow(Z)
#plt.contour(fit_func(*np.indices(fit_frame.shape)), cmap=plt.cm.copper)
plt.show()
plt.imshow(fit_frame)
plt.show()