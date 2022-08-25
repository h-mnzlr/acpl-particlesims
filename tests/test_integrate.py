# Heiko Menzler
# heikogeorg.menzler@stud.uni-goettingen.de
#
# Date: 24.06.2022
import pytest
import numpy as np

import integrate

def symmetric_func(x):
    """Example integrable function."""
    return x

def test_integrate_1D():
    interval = np.array([-1, 1])
    samples = 10000
    res, _ = integrate.integrate_uniform(symmetric_func, samples, interval)
    assert pytest.approx(res, abs=0.1) == 0

def test_integrate_2D():
    interval = np.array([[-1, 1], [-1, 1]])
    samples = 10000
    res, _ = integrate.integrate_uniform(symmetric_func, samples, interval)
    assert pytest.approx(res, abs=0.1) == 0

def test_integrate_ND():
    dims = range(1, 10)
    for dim in dims:
        interval = np.array([-1, 1] * dim)
        samples = 10000
        res, _ = integrate.integrate_uniform(symmetric_func, samples, interval)
        assert pytest.approx(res, abs=0.1) == 0
