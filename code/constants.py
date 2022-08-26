# Heiko Menzler
# heikogeorg.menzler@stud.uni-goettingen.de
#
# Date: 25.08.2022

import scipy.constants

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
    return getattr(scipy.constants, name)
