#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import sys
import numpy as np
import matplotlib.pyplot as plt

#sys.path.append('/home/ycan/Documents/scripts/generalizedmodels/')


from scipy.optimize import minimize, check_grad, approx_fprime


def split(xy):
    return np.split(xy, [1])

def parabola(a, b, c):
    def f(xy):
        x, y = split(xy)
        return a*x**4 + y*b + c
    return f

def parabola_der(a, b, c):
    def f(xy):
        x, y = split(xy)
        dpdx = 4*a*x**3
        dpdy = b
        der = np.array([dpdx.squeeze(), dpdy])
        return der
    return f

x = np.linspace(-6, 6)

a = 5
b = 8
c = 3


p = parabola(a, b, c)
pd = parabola_der(a, b, c)
#plt.plot(x, p(x))
#plt.plot(x, pd(x))

initial = np.array([-22, 1])


res = minimize(p, initial,
               method='Newton-CG',
               jac=pd,
               tol=0.2,
               options={'disp':True})

print('Gradient diff', check_grad(p, pd, initial))

print('approx. gradient', approx_fprime(initial, p, 1e-2))


print(f'Result: {res.x}')

p(res.x)

