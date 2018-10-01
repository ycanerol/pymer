#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 13:26:12 2018

@author: ycan
"""

import iofuncs as iof
import matplotlib.pyplot as plt
import plotfuncs as plf

exps = [('V', 10), ('Kara', 5), ('20171116', 6), ('20171122', 6),
        ('20171122', 7)]

all_quals = []

for i in range(len(exps)):
    exp = exps[i]
    data = iof.load(*exp)

    quals = data['quals'][-1, :]

    all_quals.append(quals)

ax = plt.subplot(111)
for j in range(len(all_quals)):
    plt.scatter(all_quals[j], [j]*len(all_quals[j]))
    plt.text(50, j, str(exps[j]), fontsize=8)
ax.set_yticks([])
plt.ylabel('Experiment')
plt.xlabel('Center px z-score')
plt.title('Distribution of STA qualities')
plf.spineless(ax)
plt.show()
