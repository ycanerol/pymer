#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 23:45:45 2018

@author: ycan
"""
import numpy as np
import matplotlib.pyplot as plt

s = 40
savedir = '/home/ycan/Downloads/'

np.random.seed(0)
plt.matshow(np.random.randint(0, 2, s*s).reshape(s,s), cmap='Greys');
plt.xticks([]);
plt.yticks([]);
plt.savefig(savedir+'checker.pdf', bbox_inches='tight')

np.random.seed(0)
a = np.repeat(np.random.randint(0, 2, s), s)
a = a.reshape(s,s)
plt.matshow(a, cmap='Greys')
plt.xticks([])
plt.yticks([])
plt.savefig(savedir+'stripe.pdf', bbox_inches='tight')