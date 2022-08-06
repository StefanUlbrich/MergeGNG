# -*- coding: utf-8 -*-
r"""
Helper functions.
"""

__author__ = "Stefan Ulbrich"
__copyright__ = "Copyright 2020-2021, All rights reserved."
# __credits__ = []
__license__ = "Confidential"
__version__ = "0.1"
__maintainer__ = "Stefan Ulbrich"
__status__ = "alpha"
__date__ = "2020-01-27"

import logging

import numpy as np

logger = logging.getLogger(__name__)


def lemniscate(rotation: float = 0.0, alpha: float = 1.0, num: int = 1000) -> np.ndarray:
    """
    Sample a Bernoulli lemniscate.

    Code shamelessly copied from `here <https://stackoverflow.com/a/27803052>`_
    and `here <https://scipython.com/book/chapter-6-numpy/examples/creating-a-rotation-matrix-in-numpy/>`_

    Parameters
    ----------
    rotation : float, optional
        Radian by which the function is rotated, by default 0.0
    alpha : float, optional
        Scale factor of the lemniscate, by default 1.0
    num : int, optional
        Number of samples, by default 1000

    Returns
    -------
    np.ndarray
        2xnum numpy array
    """

    t = np.linspace(0.5 * np.pi, 2.5 * np.pi, num=num)

    c, s = np.cos(rotation), np.sin(rotation)
    R = np.array(((c, -s), (s, c)))

    values = np.array(
        (
            alpha * np.sqrt(2) * np.cos(t) / (np.sin(t) ** 2 + 1),
            alpha * np.sqrt(2) * np.cos(t) * np.sin(t) / (np.sin(t) ** 2 + 1),
        )
    )

    return R @ values


def get_dymmy_2D_data(n: int, std: float = 1.0, num: int = 1000) -> np.ndarray:
    """
    Create dummy 2-D data from overlapping lemniscates.
    """

    return np.hstack(
        [lemniscate(0, x, num) for x in np.random.randn(n) * std + 10]
        + [lemniscate(np.pi / 2, x, num) for x in np.random.randn(n) * std + 10]
    )
