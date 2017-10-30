#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 10:45:19 2017

@author: ycan
"""
import numpy as np


class Cell:

    def __init__(self, exp_date, cluster, sta):
        self.exp_date = exp_date
        self.cluster = cluster
        self.sta = sta
        self.data = np.random.randint(-20000, high=20000, size=int(5e8))

    def save(self, path=''):
        path = str(self.exp_date + self.cluster+'.npz')

c = Cell('', '', '')