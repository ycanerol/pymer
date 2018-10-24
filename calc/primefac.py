#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 16:19:46 2017

@author: ycan

Ported to Python3 by Yunus Can Erol, Dec 2017
Original from https://gist.github.com/dzhou/2632362
Kefei Dan Zhou, 2011

prime_factorize was modified so that its output is
more similar to that of MATLAB factor().

"""
import math


def isprime(n):
    if not n % 1 == 0:
        raise ValueError('n must be an integer.')
    if n < 1:
        raise ValueError('n must be larger than 1.')
    nroot = int(math.sqrt(n))
    for i in range(2, nroot+1):
        if n % i == 0:
            return False
    return True


# return a dict or a list of primes up to N
# create full prime sieve for N=10^6 in 1 sec
def prime_sieve(n, output={}):
    nroot = int(math.sqrt(n))
    sieve = list(range(n+1))
    sieve[1] = 0

    for i in range(2, nroot+1):
        if sieve[i] != 0:
            m = int(n/i) - i
            sieve[i*i: n+1:i] = [0] * (m+1)

    if type(output) == dict:
        pmap = {}
        for x in sieve:
            if x != 0:
                pmap[x] = True
        return pmap
    elif type(output) == list:
        return [x for x in sieve if x != 0]
    else:
        return None


# get a list of all factors for N
# ex: get_factors(10) -> [1,2,5,10]
def factorize(n, primelist=None):
    if primelist is None:
        primelist = prime_sieve(n, output=[])

    fcount = {}
    for p in primelist:
        if p > n:
            break
        if n % p == 0:
            fcount[p] = 0

        while n % p == 0:
            n /= p
            fcount[p] += 1

    factors = [1]
    for i in fcount:
        level = []
        exp = [i**(x+1) for x in range(fcount[i])]
        for j in exp:
            level.extend([j*x for x in factors])
        factors.extend(level)

    return factors


# get a list of prime factors
# ex: get_prime_factors(140, returntuples=True) -> ((2,2), (5,1), (7,1))
#     140 = 2^2 * 5^1 * 7^1
def prime_factorize(n, primelist=None, returntuples=False):
    """
    Default behaviour is MATLAB-like; it returns a list of
    the prime factors that might contain multiples.
    Set returntuples to True in order to get a tuple containing
    each factor and how many times they occur.
    """
    if primelist is None:
        primelist = prime_sieve(n, output=[])

    fs = []
    for p in primelist:
        count = 0
        while n % p == 0:
            n /= p
            count += 1
        if count > 0:
            if returntuples:
                fs.append((p, count))
            else:
                [fs.append(p) for i in range(count)]

    return fs
