To compile:
```
  python setup.py build_ext --inplace
```
To use:
```
  import randpy

  rnd_number, seed  = randpy.gasdev(seed)
  rnd_numbers, seed = randpy.gasdev(seed, n)

  rnd_number, seed  = randpy.ran1(seed)
  rnd_numbers, seed = randpy.ran1(seed, n)
```
Note: seed can be either a negative integer or the seed returned by the gasdev/ran1 functions

If you are getting an error regarding 'cc1plus' not being found, install g++ by ```sudo apt-get install g++```
