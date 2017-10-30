#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 16:44:26 2017

@author: ycan
"""



def readexps():

    main_dir = '/home/ycan/Documents/Yunus_rotation_2017_06/data/Experiments/Mouse/'

    file_paths = glob.glob(main_dir+'*/analyzed/{}*.npz'.format(3))
    exp_names = [i.split('/')[-3] for i in file_paths]
    clusters = [i.split('/')[-1].split('C')[-1].split('.')[0] for i in file_paths]

    files = [file_paths, exp_names, clusters]

    return files


files = readexps()
