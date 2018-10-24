"""
pymer
=====

Collection of Python scripts for analyzing extracellular multielectrode
array recordings from the retina in the Gollisch lab, Göttingen.

"""
from __future__ import division, absolute_import, print_function

# Randpy
from .randpy import randpy

# Modules
from . import io
from . import misc

# Plotting module
from . import plot

# General functions
from .frametimesfrommat import frametimesfrommat
from .frametimestomat import savenpztomat

# Analyzers
from . import analyze

# Root
__all__ = [
    'frametimesfrommat',
    'savenpztomat',
    'io',
    'analyze',
    'misc',
    'plot',
    'randpy',
]

# Clean up relative imports
del division, absolute_import, print_function
