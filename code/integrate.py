# Heiko Menzler
# heikogeorg.menzler@stud.uni-goettingen.de
#
# Date: 24.06.2022
"""Module that provides utility functions for MC integration."""

from typing import Callable, Iterable

import numpy as np
from numpy.typing import NDArray, ArrayLike

import utils.vector
import utils.particle

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

def flavor_sampler(samples: int) -> NDArray[np.float64]:
    """Generate an array of flavor samples."""
    flavors = (1, 2, 3, 4, 5)
    return np.random.choice(flavors, size=(samples, )).astype(np.float64)

def breit_wiegner_transform(
    rho: NDArray[np.float64], mass: float, gamma: float
) -> NDArray[np.float64]:
    """Importance sampling for the Breit-Wiegner Propagator."""
    # interval = np.array(interval, ndmin=2)

    # rho_min = np.arctan((interval[0, 0] - mass**2) / mass / gamma)
    # rho_max = np.arctan((interval[0, 1] - mass**2) / mass / gamma)
    # rho = np.random.uniform(rho_min, rho_max, samples)
    s = mass * gamma * np.tan(rho) + mass * mass
    return s

def importance_quark_scattering(
    samples: int, interval: IntegrationInterval, **transform_params
) -> NDArray[np.float64]:
    rho_theta_phi_array = uniform_sampler(samples, interval)
    flavor_samples = flavor_sampler(samples)
    rho = rho_theta_phi_array[..., 0]
    s = breit_wiegner_transform(rho, **transform_params)
    return np.c_[s, rho_theta_phi_array[..., 1:], flavor_samples]

def combination_sampler(
    samples: int,
    *samplers: Sampler,
) -> NDArray[np.float64]:
    """Combines multiple samplers as a new sampler."""
    mc_nums = []
    for sampler in samplers:
        mc_nums.append(sampler(samples))

    return np.column_stack(mc_nums)

def quark_scattering_process(
    samples: int, interval: IntegrationInterval
) -> NDArray[np.float64]:
    """Generate a random event of a scattering process producing quarks."""
    theta_phi_array = uniform_sampler(samples, interval)
    flavor_samples = flavor_sampler(samples)
    return np.c_[theta_phi_array[:], flavor_samples]

def event_generator(
    samples: int,
    interval: IntegrationInterval,
    s: float,
) -> Iterable[list[utils.particle.Particle]]:
    theta_phi_flav_array = quark_scattering_process(samples, interval=interval)
    for theta, phi, flav in theta_phi_flav_array:
        four_momentum_e = {"E": 1, "px": 0, "py": 0, "pz": 1}
        four_momentum_anti_e = {"E": 1, "px": 0, "py": 0, "pz": -1}
        four_momentum_q = {
            "E": 1,
            "px": -np.cos(phi) * np.sin(theta),
            "py": -np.sin(phi) * np.sin(theta),
            "pz": -np.cos(theta)
        }
        four_momentum_anti_q = {
            "E": 1,
            "px": +np.cos(phi) * np.sin(theta),
            "py": +np.sin(phi) * np.sin(theta),
            "pz": +np.cos(theta)
        }
        q_momentum = utils.vector.Vec4(**four_momentum_q) * (np.sqrt(s) / 2)
        anti_q_momentum = utils.vector.Vec4(**four_momentum_anti_q
                                            ) * (np.sqrt(s) / 2)
        e_momentum = utils.vector.Vec4(**four_momentum_e) * (np.sqrt(s) / 2)
        anti_e_momentum = utils.vector.Vec4(**four_momentum_anti_e
                                            ) * (np.sqrt(s) / 2)

        elec = utils.particle.Particle(11, e_momentum)
        anti_elec = utils.particle.Particle(11, anti_e_momentum)
        quark = utils.particle.Particle(flav, q_momentum, color=[1, 0])
        anti_quark = utils.particle.Particle(
            flav, anti_q_momentum, color=[0, 1]
        )
        event = [elec, anti_elec, quark, anti_quark]
        yield event

def integrate_sampler(
    func: IntegrableFunction,
    samples: int,
    sampler: Sampler,
    volume_element: float,
    realizations: int = 1
) -> tuple[np.float64, np.float64]:
    """Integrate the function using MC methods on a uniform distribution."""

    res = []
    for _ in range(realizations):
        mc_numbers = sampler(samples)
        function_samples = func(mc_numbers)
        res.append(
            np.sum(function_samples / samples * volume_element, axis=-1)
        )

    return np.mean(res), np.std(res)
