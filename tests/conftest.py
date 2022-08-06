# pylint: disable=R,C,W


import logging
import pytest
from numpy.random import dirichlet
import numpy as np

import mgng


logging.basicConfig(
    format="[%(levelname)s:%(name)s:%(funcName)s:%(lineno)d] %(message)s", level=logging.DEBUG
)


@pytest.fixture(scope="module")
def data():

    return np.zeros((42, 42))


@pytest.fixture
def logger():
    logger_ = logging.getLogger("UnitTest")
    return logger_
