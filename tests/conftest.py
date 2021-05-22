# pylint: disable=R,C,W


import logging
import pytest
from scipy.stats import multinomial, multivariate_normal
from numpy.random import dirichlet
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.datasets import make_spd_matrix
from sklearn.pipeline import make_pipeline, Pipeline


import mgng


logging.basicConfig(
    format="[%(levelname)s:%(name)s:%(funcName)s:%(lineno)d] %(message)s", level=logging.DEBUG
)


@pytest.fixture(scope="module")
def data():

    return np.zeros((42,42))

@pytest.fixture
def logger():
    logger_ = logging.getLogger("UnitTest")
    return logger_
