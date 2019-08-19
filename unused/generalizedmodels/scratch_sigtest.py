#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import matplotlib.pyplot as plt

from scipy import linalg

N = 1000
s = (40, 40)

ws = np.zeros((N, s[0]))

for i in range(N):
    a = np.random.random_sample(s)-.5
    w, v = linalg.eigh(a)
    ws[i] = w

plt.hist(ws.flatten())
plt.show()

plt.plot(ws[-1, :], 'o')
