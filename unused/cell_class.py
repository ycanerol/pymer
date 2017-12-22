#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 10:45:19 2017

@author: ycan
"""
import numpy as np


class Experiment:
    import numpy as np
    import analysis_scripts as asc

    def __init__(self, exp_path):
        self.exp_path = exp_path
        self.ods_extracted = asc.read_ods(exp_path)

c = Cell('', '', '')