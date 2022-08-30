# Heiko Menzler
# heikogeorg.menzler@stud.uni-goettingen.de
#
# Date: 25.08.2022

import scipy.constants as const

import numpy as np
from numpy.typing import NDArray, ArrayLike

QED_COUPLING = 1 / 129
QCD_COUPLING_Z_MASS = .118
Z_MASS = 91.2
Z_DECAY_WIDTH = 2.5
WEINBERG_ANGLE_SQ_SINE = .223
ELECTRON_CHARGE = -1
UP_QUARK_CHARGE = 2 / 3
DOWN_QUARK_CHARGE = -1 / 3
UP_QUARK_WEAK_ISOSPIN = .5
DOWN_QUARK_WEAK_ISOSPIN = -.5
NUM_QCD_COLORS = 3
CONVERSION_FACTOR = 3.89379656e8

QUARKS = {
    "u": (UP_QUARK_CHARGE, UP_QUARK_WEAK_ISOSPIN),
    "c": (UP_QUARK_CHARGE, UP_QUARK_WEAK_ISOSPIN),
    "d": (DOWN_QUARK_CHARGE, DOWN_QUARK_WEAK_ISOSPIN),
    "s": (DOWN_QUARK_CHARGE, DOWN_QUARK_WEAK_ISOSPIN),
    "b": (DOWN_QUARK_CHARGE, DOWN_QUARK_WEAK_ISOSPIN)
}
NUM_LIGHT_QUARK_FLAV = len(QUARKS)
_int_to_quark = {idx + 1: key for idx, key in enumerate(QUARKS.keys())}

def quark_info(flavors: NDArray[np.int16] | ArrayLike) -> NDArray[np.float64]:
    flavors = np.array(flavors, ndmin=1)
    return np.swapaxes(
        np.array([QUARKS[_int_to_quark[flav]] for flav in flavors]), 0, 1
    )

def __getattr__(name: str):
    return getattr(const, name)

def scattering_mat(flav, s, costheta, _):
    """Scattering matrix element for given flavor and particle outcome."""
    quark_charge, quark_iso_spin = quark_info(flav)
    prefactor = (4 * const.pi * QED_COUPLING)**2 * NUM_QCD_COLORS

    kappa = 1 / 4 / WEINBERG_ANGLE_SQ_SINE / (1 - WEINBERG_ANGLE_SQ_SINE)
    chi_denom = (s - Z_MASS**2)**2 + Z_MASS**2 * Z_DECAY_WIDTH**2
    chi1 = kappa * s * (s - Z_MASS**2) / chi_denom
    chi2 = kappa**2 * s**2 / chi_denom

    a_elec = DOWN_QUARK_WEAK_ISOSPIN
    a_quark = quark_iso_spin
    v_elec = DOWN_QUARK_WEAK_ISOSPIN - 2 * ELECTRON_CHARGE * WEINBERG_ANGLE_SQ_SINE
    v_quark = quark_iso_spin - 2 * quark_charge * WEINBERG_ANGLE_SQ_SINE
    cos_pre = 4 * ELECTRON_CHARGE * quark_charge * a_elec * a_quark * chi1 \
        + 8 * a_elec * v_elec * a_quark * v_quark * chi2

    cos_sq_pre = ELECTRON_CHARGE ** 2 * quark_charge ** 2 \
        + 2 * ELECTRON_CHARGE * quark_charge * v_elec * v_quark * chi1 \
        + (a_elec ** 2 + v_elec ** 2) * (a_quark ** 2 + v_quark ** 2) * chi2

    val = costheta * cos_pre + (1 + costheta * costheta) * cos_sq_pre

    return prefactor * val
