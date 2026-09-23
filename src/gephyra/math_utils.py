"""
Continuous scoring primitives for hydrogen-bond geometry.

Every numerical constant in this module is an empirical smoothing parameter
chosen for Gephyra. None of them is inherited from MDAnalysis, whose
WaterBridgeAnalysis applies hard geometric cutoffs (a single 3.0 A
donor-acceptor distance and a fixed donor-H-acceptor angle) rather than a
graded score. Specifically:

- The 6/12 exponents of `switching_function` come from the rational switching
  form used in coordination-number collective variables (as in PLUMED and
  similar codes). They control only how sharply the function falls between 1
  and 0 across the interval [0, threshold]; they carry no energetic meaning and
  are not a Lennard-Jones 6-12 potential.
- The 0.6 A threshold on the angular proxy in `calculate_hbond_probability` is
  applied to the triangle-inequality excess r(DH) + r(HA) - r(DA), which is zero
  for a perfectly linear bond and grows as the bond bends. 0.6 A is the excess
  at which the bond is taken to have vanished, and it stands in for a minimum
  donor-H-acceptor angle without ever computing an angle.
- The 2.40 A onset of the steric term, and its 12th power, are likewise chosen
  to suppress unphysically short contacts rather than to reproduce any
  particular force field.

Changing these values rescales the reported scores, so results obtained with
different values are not comparable.
"""

import numpy as np

def switching_function(distance, threshold, power_num=6, power_den=12):
    ratio = distance / threshold
    if ratio >= 1.0:
        return 0.0
    if ratio <= 0.0:
        return 1.0

    return (1.0 - (ratio ** power_num)) / (1.0 - (ratio ** power_den))

def calculate_hbond_probability(mod_rOO, mod_rOiH, mod_rOjH, r0_oo=2.80, r0_threshold=0.45, ignore_angle=False):
    """
    Evaluates the continuous hydrogen bond weight between two molecules.
    """
    # Angle/Hydrogen placement component
    if ignore_angle:
        p_angle = 1.0
    else:
        h_dist_factor = mod_rOiH + mod_rOjH - mod_rOO
        p_angle = switching_function(h_dist_factor, threshold=0.6)

    # Heavy atom distance component
    oo_dist_factor = mod_rOO - r0_oo
    p_dist = switching_function(oo_dist_factor, threshold=r0_threshold)

    # Asymmetric 12th-power repulsive potential for steric clashes
    if mod_rOO < 2.40:
        p_steric = (mod_rOO / 2.40)**12
    else:
        p_steric = 1.0

    # Total fractional probability
    return p_angle * p_dist * p_steric
