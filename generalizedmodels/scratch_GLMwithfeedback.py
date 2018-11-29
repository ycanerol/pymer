#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import matplotlib.pyplot as plt

from genlinmod import conv

def glm_fr(k, h, mu):
    def fr(x, y):
        return np.exp(conv(k, x) + conv(h, y) + mu)
    return fr

