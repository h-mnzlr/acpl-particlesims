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
IntegrationParameters = NDArray[np.int16] | ArrayLike

Sampler = Callable[[int], NDArray[np.float64]]

def uniform_sampler(samples: int, interval: IntegrationInterval):
    """Generate an array of uniformly distributed samples."""
    interval = np.array(interval, ndmin=2)
    mc_numbers = np.random.uniform(
        interval[..., 0],
        interval[..., 1],
        size=(samples, *interval.shape[:-1])
    )
    return mc_numbers

def flavor_sampler(samples: int):
    """Generate an array of flavor samples."""
    flavors = (1, 2, 3, 4, 5)
    return np.random.choice(flavors, size=(samples, ))

def quark_scattering_process(samples: int, interval: IntegrationInterval):
    """Generate a random event of a scattering process producing quarks."""
    flavor_samples = flavor_sampler(samples)
    theta_phi_array = uniform_sampler(samples, interval)
    return np.c_[theta_phi_array[:], flavor_samples.astype(np.float64)]

def integrate_sampler(
    func: IntegrableFunction,
    samples: int,
    sampler: Sampler,
    volume_element: float
) -> tuple[NDArray[np.float64], np.float64]:
    """Integrate the function using MC methods on a uniform distribution."""
    mc_numbers = sampler(samples)

    function_samples = func(mc_numbers)
    res = np.sum(function_samples / samples * volume_element, axis=-1)
    error = volume_element * np.std(function_samples) / np.sqrt(samples)
    return np.squeeze(res), error
