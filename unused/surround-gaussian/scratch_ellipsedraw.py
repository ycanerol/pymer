#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 13:44:18 2018

@author: ycan
"""

import matplotlib.pyplot as plt
from  matplotlib.patches import Ellipse as elps
import numpy as np

fig, ax = plt.subplots()

#elps = matplotlib.patches.Ellipse

angles = np.arange(-20, 120, 10)

for i in range(len(angles)):
    rf = elps(((.1 *i) % 5, (.1 * i) % 5), .05, .12, angles[i])
    ax.add_artist(rf)
ax.set_aspect('equal')
plt.show()