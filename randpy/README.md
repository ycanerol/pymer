# randpy

## Installation

```bash
python setup.py build_ext --inplace
```

For Windows and Python 2.7, you need to install Visual Studio 2008 (or the Visual Studio C++ Tools for Python 2.7)

If you are getting an error regarding 'cc1plus' not being found, install g++ by ```sudo apt-get install g++```


## Usage

```python
import randpy

random_number,  seed = randpy.gasdev(seed)
random_numbers, seed = randpy.gasdev(seed, n)

random_number,  seed = randpy.ran1(seed)
random_numbers, seed = randpy.ran1(seed, n)

random_number,  seed = randpy.ranb(seed)
random_numbers, seed = randpy.ranb(seed, n)
```

**Note**: `seed` can be either a negative integer or the seed returned by the gasdev/ran1 functions
