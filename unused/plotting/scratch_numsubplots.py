#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 18:13:38 2017

@author: ycan


"""


def numsubplots(n, recursive=False):
    """
    Define the best arrangement of subplots for a
    given number of plots.

    Parameters:
        n:
            Number of total plots needed.
        recursive:
            Whether to return current number of subplots.
            Used for recursive calls.
    Returns:
        p:
            A list containing the ideal arrangement
            of subplots.
        n:
            Current number of subplots. Returned only
            when recursively calling the function.

    Ported to Python by Yunus Can Erol on Dec 2017
    from mathworks.com/matlabcentral/fileexchange/
    26310-numsubplots-neatly-arrange-subplots
    Original by Rob Campbell, Jan 2010

    """
    import primefac

    while primefac.isprime(n) and n > 4:
        n += 1

    p = primefac.prime_factorize(n)
    if len(p) == 1:
        p = [1] + p
        if recursive:
            return p, n
        else:
            return p

    while len(p) > 2:
        if len(p) >= 4:
            p[0] = p[0]*p[-2]
            p[1] = p[1]*p[-1]
            del p[-2:]
        else:
            p[0] = p[0]*p[1]
            del p[1]
        p = sorted(p)

    while p[1]/p[0] > 2.5:
        N = n+1
        p, n = numsubplots(N, recursive=True)

    if recursive:
        return p, n
    else:
        return p

for i in range(2, 100):
    print(i, numsubplots(i))
