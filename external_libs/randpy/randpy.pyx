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
# (random_number, seed)  = randpy.ranb(seed)
# (random_numbers, seed) = randpy.ranb(seed, n)
#
#
#
# Note: seed can be either a negative integer or a dictionary
# returned by the randpy functions
#
# For Windows and Python 2.7, you need to install Visual Studio 2008
# (or the Visual Studio C++ Tools for Python 2.7)
#
# (Fernando Rozenblit, 2017)
#
#
# Introduced ranb and improved performance.
# (Sören Zapp, 2018)

cdef extern from "rng_gasdev_ran1.h":
	struct Seed:
		long idum
		long iy
		long iv[32] # const long NTAB = 32
		int iset
		double gset

cdef extern from "rng_gasdev_ran1.cpp":
	list c_ran1_vec "ran1_vec" (Seed& seed, unsigned int num)
	list c_ranb_vec "ranb_vec" (Seed& seed, unsigned int num)
	double c_gasdev "gasdev" (Seed& seed)

cpdef make_seed(seed):
	cdef Seed c_seed

	if isinstance(seed, dict):
		c_seed = seed
	else:
		c_seed.idum = seed

	return c_seed

def ran1(seed, n = 1):
        """
        Generate a uniform random number in the range (0, 1).

        A negative number must be used as seed when initializing.
        """
	cdef Seed c_seed = make_seed(seed)
	return (c_ran1_vec(c_seed, n), c_seed)

def ranb(seed, n = 1):
	"""
        Generates 0 or 1 with equal probabilities. This is a wrapper for
        the ran1 function in C, it's around five times faster than the
        equivalent expression [0 if i<0.5 else 1 for i in ran1(seed, n)].

        A negative number must be used as seed when initializing.
        """cdef Seed c_seed = make_seed(seed)
	return (c_ranb_vec(c_seed, n), c_seed)

def gasdev(seed, n = 1):
        """
        Generate a random number from a Gaussian distribution with a
        mean of 0 and standard deviation of 1.

        A negative number must be used as seed when initializing.
        """
	cdef Seed c_seed = make_seed(seed)

	if n > 1:
		res = [c_gasdev(c_seed) for x in xrange(n)]
	else:
		res = c_gasdev(c_seed)

	return (res, c_seed)
