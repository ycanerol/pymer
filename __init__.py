"""
pymer
=====

Collection of Python scripts for analyzing extracellular multielectrode
array recordings from the retina in the Gollisch lab, Göttingen.

"""
from __future__ import division, absolute_import, print_function

# Modules
from . import io
from . import misc
from . import frametimes
from . import plot
from . import randpy
from . import analyze


# Root
__all__ = [
    'analyze',
    'frametimes',
    'io',
    'misc',
    'plot',
    'randpy',
]

# Clean up relative imports and remove references
calc = None
del calc, division, absolute_import, print_function
