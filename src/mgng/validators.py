# -*- coding: utf-8 -*-
r"""
Validators for numerical data
"""

__author__ = "Stefan Ulbrich"
__copyright__ = "Copyright 2020, All rights reserved."
# __credits__ = []
__license__ = "Confidential"
__version__ = "0.1"
__maintainer__ = "Stefan Ulbrich"
__email__ = "Stefan.Ulbrich@gmail.com"
__status__ = "alpha"
__date__ = "2020-01-27"

from typing import Any
import numpy as np
# from .mgng import MergeGNG


def is_weight_factor(instance: Any, attribute, value: float):
    r"""
    Validator for decay factors :math:`\gamma` : :math:`0 < \gamma < \leq 1`

    Parameters
    ----------
    instance : Any
    attribute : str
        The name of the attribute
    value : float
        The value of the attribute
    """
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Value out of bounds: {value}")


def is_greater_zero(instance: Any, attribute, value: float):
    """
    Validator to check for stricly positive values.
    """
    if not value > 0:
        raise ValueError(f"Value not strictly positive: {value}")


def repr_ndarray(array: np.ndarray):
    """
    Print array dimensions.

    Parameters
    ----------
    a : np.ndarray
    """
    return f"ndarray of size {array.shape}"
