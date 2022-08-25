# Heiko Menzler
# heikogeorg.menzler@stud.uni-goettingen.de
#
# Date: 24.06.2022
"""Module that provides utility functions for MC integration."""

from typing import Callable

import numpy as np
from numpy.typing import NDArray, ArrayLike

IntegrableFunction = Callable[[NDArray[np.float64]], np.float64]
IntegrationInterval = NDArray[np.float64] | ArrayLike

def integrate_uniform(
    func: IntegrableFunction, samples: int, interval: IntegrationInterval
) -> tuple[NDArray[np.float64], np.float64]:
    """Integrate the function using MC methods on a uniform distribution."""
    interval = np.array(interval, ndmin=2)
    print(interval.shape)
    mc_numbers = np.random.uniform(
        interval[..., 0],
        interval[..., 1],
        size=(samples, *interval.shape[:-1])
    )

    lengths = interval[..., 1] - interval[..., 0]
    assert isinstance(lengths, np.ndarray)
    length = lengths.max()

    function_samples = func(mc_numbers)
    res = np.sum(function_samples / samples * length, axis=-1)
    error = length.max() * np.std(function_samples) / samples
    return np.squeeze(res), error
