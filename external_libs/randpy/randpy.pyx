#
# To compile:
#	python setup.py build_ext --inplace
#
#
# To use:
#
# import randpy
#
# (random_number, seed)  = randpy.gasdev(seed)
# (random_numbers, seed) = randpy.gasdev(seed, n)
#
# (random_number, seed)  = randpy.ran1(seed)
# (random_numbers, seed) = randpy.ran1(seed, n)
#
#
#  Note: seed can be either a negative integer or a dictionary returned by the randpy functions
#
#  For Windows and Python 2.7, you need to install Visual Studio 2008 (or the Visual Studio C++ Tools for Python 2.7)
#
#  (Fernando Rozenblit, 2017)

cdef extern from "rng_gasdev_ran1.h":
	struct Seed:
		long idum
		long iy
		long iv[32] # const long NTAB = 32
		int iset
		double gset

cdef extern from "rng_gasdev_ran1.cpp":
	double c_ran1 "ran1" (Seed& seed)
	double c_gasdev "gasdev" (Seed& seed)

cpdef make_seed(seed):
	cdef Seed c_seed

	if isinstance(seed, dict):
		c_seed = seed
	else:
		c_seed.idum = seed

	return c_seed

def ran1(seed, n = 1):
	cdef Seed c_seed = make_seed(seed)

	if n > 1:
		res = [c_ran1(c_seed) for x in xrange(n)]
	else:
		res = c_ran1(c_seed)

	return (res, c_seed)

def gasdev(seed, n = 1):
	cdef Seed c_seed = make_seed(seed)

	if n > 1:
		res = [c_gasdev(c_seed) for x in xrange(n)]
	else:
		res = c_gasdev(c_seed)

	return (res, c_seed)
