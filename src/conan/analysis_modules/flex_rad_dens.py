# import matplotlib.pyplot as plt
# import networkx as nx
import numpy as np

import conan.analysis_modules.cnt_fill as cf

# import pandas as pd

# import conan.analysis_modules.traj_an as traj_an
# import conan.analysis_modules.traj_info as traj_info
# import conan.defdict as ddict


def flex_rad_dens(traj_file, molecules, an):
    flex_rd = FlexRadDens(traj_file, molecules, an)
    flex_rd.flex_rad_dens_prep()


def points_in_cylinder(pt1, pt2, r, atom_positions):
    pt1 = np.asarray(pt1, dtype=np.float64)
    pt2 = np.asarray(pt2, dtype=np.float64)
    atom_positions = np.asarray(atom_positions, dtype=np.float64)

    vec = pt2 - pt1
    vec /= np.linalg.norm(vec)  # Normalize axis vector
    proj = np.dot(atom_positions - pt1, vec)  # Projection along CNT axis

    radial_dist = np.linalg.norm((atom_positions - pt1) - np.outer(proj, vec), axis=1)

    within_cylinder = np.logical_and.reduce((proj >= 0, proj <= np.linalg.norm(pt2 - pt1), radial_dist <= r))

    return within_cylinder


class FlexRadDens:
    """
    Calculate the radial density of moleculas confined within a flexible CNT
    """

    def __init__(self, traj_file, molecules, an):
        self.traj_file = traj_file
        self.molecules = molecules
        self.an = an
        self.shortening_q = "n"
        self.shortening = 0.0

    def flex_rad_dens_prep(self):
        """
        Prepare the flexible radial density analysis.
        For this we need to do:
        - Ask the user how many increments the CNT should be radially divided into.
        - Let the user decide if the full length of the CNT should be subject to this analysis
        (to avoid opening effects if wanted).
          - For this one can use use the setup of the loading mass module.

        """

        # run the cnt_loading_mass_prep function in the CNTload class in cf
        cnt_load = cf.CNTload(self.traj_file, self.molecules, self.an)
        cnt_load.cnt_loading_mass_prep()
