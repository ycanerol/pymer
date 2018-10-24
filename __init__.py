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

# General functions
from .csindexchange import csindexchange
from .frametimesfrommat import frametimesfrommat
from .plot_checker_stas import plotcheckerstas
from .plotcheckersurround import plotcheckersurround
from .plotcheckersvd import plotcheckersvd

# Analyzers
from . import analyze

# Root
__all__ = [
    'csindexchange',
    'frametimesfrommat',
    'plotcheckerstas',
    'plotcheckersurround',
    'plotcheckersvd',
    'analyze',
    'randpy',
]

__all__.extend(modules.__all__)


# Clean up relative imports
del modules, division, absolute_import, print_function
