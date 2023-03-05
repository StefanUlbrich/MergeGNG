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
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def lemniscate(rotation: float = 0.0, alpha: float = 1.0, num: int = 1000) -> NDArray[np.floating[Any]]:
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

    cos, sin = np.cos(rotation), np.sin(rotation)
    rotation_matrix = np.array(((cos, -sin), (sin, cos)))

    values = np.array(
        (
            alpha * np.sqrt(2) * np.cos(t) / (np.sin(t) ** 2 + 1),
            alpha * np.sqrt(2) * np.cos(t) * np.sin(t) / (np.sin(t) ** 2 + 1),
        )
    )

    return rotation_matrix @ values


def get_dymmy_2d_data(
    n_repeats: int, n_samples: int = 1000, std: float = 1.0, mean: float = 10.0
) -> NDArray[np.floating[Any]]:
    """
    _summary_

    _extended_summary_

    Parameters
    ----------
    n_repeats : int
        Number of different random :func:`lemniscate` to sample from.
    n_samples : int
        Number of samples from each :func:`lemniscate`, by default 1000
    std : float, optional
        Standard deviation of the :math:`alpha` parameter, by default 1.0
    mean : float, optional
        Mean of the :math:`alpha` parameter, by default 10.0

    Returns
    -------
    np.ndarray
        ``n_repeats`` * ``n_samples`` samples (in the colums).
    """
    # Create dummy 2-D data from overlapping lemniscates.
    return np.hstack(
        [lemniscate(0, x, n_samples) for x in np.random.randn(n_repeats) * std + mean]  # type: ignore
        + [lemniscate(np.pi / 2, x, n_samples) for x in np.random.randn(n_repeats) * std + mean]  # type: ignore
    )
