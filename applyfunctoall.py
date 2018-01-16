#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 11:47:09 2018

@author: ycan
"""

checkers = [('V', 10), ('20171116', 6), ('20171122', 6), ('20171122', 7),
            ('Kara', 5)]

stripes = [('20171116', 7), ('20171116', 8), ('20171116', 9),
           ('20171116', 10),
           ('20171122', 8), ('20171122', 9), ('20171122', 10),
           ('20171122', 11)]

exps = stripes

#f = stripeflickeranalysis
g = plotstripestas

for exp in exps:
#    f(*exp)
    g(*exp)
    print(exp, ' done')
