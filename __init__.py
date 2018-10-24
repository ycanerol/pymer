"""
pymer
=====

Collection of Python scripts for analyzing extracellular multielectrode
array recordings from the retina in the Gollisch lab, Göttingen.

"""
from __future__ import division, absolute_import, print_function

# Externals
from .external import randpy

# Modules
from . import modules

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
    'analyze',
    'plot',
    'randpy',
]

__all__.extend(modules.__all__)


# Clean up relative imports
del modules, division, absolute_import, print_function
