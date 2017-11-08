#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 16:05:21 2017

@author: ycan
"""

# Subtract the center, fit another gaussian for surround.
Zm_flat = Zm
Zm_flat[np.isnan(Zm_flat)] = np.nanmax(Zm_flat)
surround = fit_frame-Zm_flat
pars_sur = gfit.gaussfit(surround)

f_sur = gfit.twodgaussian(pars_sur)

Z_sur = f_sur(X, Y)

Zm_sur = np.log((Z_sur-pars_sur[0])/pars_sur[1])
Zm_sur[np.isinf(Zm_sur)] = np.nan
Zm_sur = np.sqrt(Zm_sur*-2)
plt.imshow(Zm_sur)