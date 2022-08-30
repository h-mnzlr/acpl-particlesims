import itertools
import math as m

import constants as const

from utils.vector import Vec4
from utils.particle import Particle
from utils.histogram import Histo1D, Scatter2D

class Analysis:
    """Analyzer of 2->(n jet) scattering events, histogramming differential jet
    rate cross sections

      d\sigma / d\log_10(y_{n,n+1})

    for various n. The integrated n-jet rates are also calculated and stored as
    scatter data.
    """

    def __init__(self):

        self.num_events = 0.

        # Construct list of histograms for the differential jet rates up to
        # y_{n_max,n_max+1} and for the corresponding integrated jet rates.
        n_bins = 100
        self.left_edge = -4.3
        self.right_edge = -0.3
        n_max = 4
        self.y_n = [
            Histo1D(
                n_bins,
                self.left_edge,
                self.right_edge,
                '/LL_JetRates/log10_y_{0}{1}'.format(i + 2, i + 3)
            ) for i in range(n_max)
        ]
        self.y_n_integrated = [
            Scatter2D(
                n_bins,
                self.left_edge,
                self.right_edge,
                '/LL_JetRates/integ_log10_y_{0}'.format(i + 2)
            ) for i in range(n_max + 1)
        ]

    def analyze(self, event, weight):
        """Adds a single event (= list of Particle instances)
        with corresponding Monte-Carlo weight to the histograms."""

        self.num_events += 1.

        # Fill differential j -> (j+1) splitting scale distributions if there
        # have not been a sufficient number of to cluster, we add the event to
        # the underflow of the histogram.
        y_ij_list = self.cluster(event)
        for j in range(len(self.y_n)):
            log_y = self.left_edge - 1
            if len(y_ij_list) > j:
                log_y = m.log10(y_ij_list[-1 - j])
            self.y_n[j].fill(log_y, weight)

        # Fill integrated j-jet rates.
        previous_logy = 1e20
        for j in range(len(self.y_n_integrated) - 1):
            j_jet_rate = self.y_n_integrated[j]
            log_y = self.left_edge - 1
            if len(y_ij_list) > j:
                log_y = m.log10(y_ij_list[-1 - j])
            for p in j_jet_rate.points:
                if p.x > log_y and p.x < previous_logy:
                    p.y += weight
            previous_logy = log_y
        for p in self.y_n_integrated[-1].points:
            if p.x < previous_logy:
                p.y += weight

    def finalize(self, file_name):
        """Scales the histograms properly and writes them out as a YODA file
        with the given file_name."""

        # Divide out the number of events to get the correct cross section.
        for h in self.y_n:
            h.scale(1. / self.num_events)
        for s in self.y_n_integrated:
            s.scale(1. / self.num_events)

        # Write the histograms to a YODA file.
        file = open(file_name, "w")
        file.write("\n\n".join([str(h) for h in self.y_n]))
        file.write("\n\n")
        file.write("\n\n".join([str(s) for s in self.y_n_integrated]))
        file.close()

    def y_ij(self, p_i: Vec4, p_j: Vec4, q2: float) -> float:
        """Calculates the k_T-algorithm distance measure between four momenta
        p_i and p_j, and the reference scale Q^2"""
        pipj = p_i.px * p_j.px + p_i.py * p_j.py + p_i.pz * p_j.pz
        cos_theta = min(
            max(
                pipj /
                m.sqrt(p_i.length_3d_squared() * p_j.length_3d_squared()),
                -1.0
            ),
            1.0
        )
        return 2.0 * min(p_i.E**2, p_j.E**2) * (1.0 - cos_theta) / q2

    def cluster(self, event: list[Particle]):
        """Applies the k_T clustering algorithm to a an event (= list of
        Particle instances). A y_cut is not used, i.e. the clustering continues
        until only two jets are left.

        Returns a list of splitting scales y_ij, ordered from smallest y_ij to
        largest y_ij.
        """
        # TODO: Implement the k_T clustering algorithm described in
        # the lecture here, and return all found splitting scales y_ij as a
        # list, ordered from smallest y_ij to largest y_ij.

        # NOTE: There is no fixed y_cut, since we want to know at which point a
        # n-jet event starts looking like a (n+1)-jet event for a range of
        # different n (compare the kT jet fraction plot in the lecture, which
        # is a sort of integrated version of what we'd like to plot). Instead,
        # keep clustering until only two jets are left.

        # NOTE: The y_ij distance measure is already implemented, see above.
        # As a reference scale Q^2, use the invariant mass of two incoming (or
        # two outgoing) particles, which in our case will be equal the squared
        # Z mass.

        final_state_momenta = list(map(lambda part: part.mom, event[2:]))
        distances = map(
            lambda moms: self.y_ij(moms[0], moms[1], const.Z_MASS**2),
            itertools.combinations(final_state_momenta, 2)
        )

        # while len(final_state_momenta) >= 2:
        #     distances = []
        #     for (idx1, mom1), (idx2, mom2) in itertools.combinations(
        #         enumerate(final_state_momenta), 2
        #     ):
        #         y_ij = self.y_ij(mom1, mom2, const.Z_MASS**2)
        #         distances.append((y_ij, idx1, idx2))

        #     _, idx1, idx2 = min(distances, key=lambda x, *_: x)
        #     mom_combine = final_state_momenta.pop(max(idx1, idx2))
        #     final_state_momenta[min(idx1, idx2)] += mom_combine

        return list(sorted(distances))
