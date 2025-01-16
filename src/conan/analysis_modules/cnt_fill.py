import networkx as nx
import numpy as np

import conan.analysis_modules.traj_an as traj_an
import conan.analysis_modules.traj_info as traj_info
import conan.defdict as ddict


# check if a given point is within a cylinder
def points_in_cylinder(pt1, pt2, r, q):
    vec = pt2 - pt1
    const = r * np.linalg.norm(vec)
    return np.where(
        (np.dot(q - pt1, vec) >= 0)
        & (np.dot(q - pt2, vec) <= 0)
        & (np.linalg.norm(np.cross(q - pt1, vec), axis=1) <= const)
    )


def cnt_loading_mass(traj_file, molecules, an):
    clm = CNTload(traj_file, molecules, an)
    clm.cnt_loading_mass_prep()
    traj_an.process_trajectory(traj_file, molecules, an, clm)


class CNTload:
    """
    Class to calculate the loading mass of the liquid within the carbon nanotube.
    """

    def __init__(self, traj_file, molecules, an):
        self.traj_file = traj_file
        self.molecules = molecules
        self.an = an
        self.liquid_mass = 0
        self.total_mass = 0

    def cnt_loading_mass_prep(self):
        """
        initializes the variables, the number of CNTs and identifies the most outwards atoms of the CNTs.
        """

        # get number of CNTs in the trajectory file. (for now we consider all pores to be CNTs, as the CNT definition
        # describes just frozen CNT along the z-axis).
        # extract all entries from the frame0 dataframe where the Struc column contains "Pore" and store it
        # in a new dataframe called pore_atoms.
        pore_atoms = self.traj_file.frame0[self.traj_file.frame0["Struc"].str.contains("Pore")].copy()
        # drop the "Pore" string from the Struc column and just keep the number
        pore_atoms.loc[:, "Struc"] = pore_atoms["Struc"].str.replace("Pore", "")
        pore_atoms.loc[:, "Struc"] = pore_atoms["Struc"].astype(int)
        # print(pore_atoms)
        pore_No = pore_atoms["Struc"].nunique()

        # add a new column to the pore_atoms dataframe with the number of bonds the atom has
        pore_atoms["bonds"] = 0

        # print(self.molecules.molecule_bonds)

        # identify an atom from the CNT_atoms list which belongs to a given CNT (from 1 to the maximum number of CNTs)
        for i in range(1, pore_No + 1):
            atoms = pore_atoms[pore_atoms["Struc"] == i]
            for index, row in atoms.iterrows():
                count = 0
                for j in range(len(self.molecules.molecule_bonds)):
                    for k in range(len(self.molecules.molecule_bonds[j])):
                        if index in self.molecules.molecule_bonds[j][k]:
                            count += 1
                pore_atoms.loc[index, "bonds"] = count

        """
        Check which atoms belong to which side of the CNT.
        """
        Catom_positions = np.array(
            [
                [atom["x"], atom["y"], atom["z"], int(atom.name)]
                for index, atom in pore_atoms.iterrows()
                if atom["bonds"] == 2
            ]
        )

        G = nx.Graph()
        for i in range(len(Catom_positions)):
            for j in range(i + 1, len(Catom_positions)):
                # calculate the distance between the two atoms (x,y,z are the first three entries in the array)
                dist = np.linalg.norm(
                    traj_info.minimum_image_distance(
                        Catom_positions[i][:3], Catom_positions[j][:3], self.traj_file.box_size
                    )
                )
                # if the distance is smaller than 3 angstroms, add the edge to the graph
                if dist < 3:
                    G.add_edge(i, j)

        # rename the nodes of the graph with the atom indices
        mapping = {i: index for i, index in enumerate(Catom_positions[:, 3])}
        G = nx.relabel_nodes(G, mapping)

        # each connected component of the graph is a molecule
        molecules = list(nx.connected_components(G))

        # add a new column to the pore_atoms dataframe with the ring number using .loc
        pore_atoms["ring"] = 0

        for i in range(len(molecules)):
            for j in molecules[i]:
                pore_atoms.loc[j, "ring"] = i + 1

        ddict.printLog(pore_atoms[pore_atoms["bonds"] == 2])

        # save the index of ring1 and ring2 atoms to seperate numpy arrays
        self.ring1 = np.array([atom for atom in pore_atoms[pore_atoms["ring"] == 1].index])
        self.ring2 = np.array([atom for atom in pore_atoms[pore_atoms["ring"] == 2].index])

        ddict.printLog(
            "Please use global, unwrapped coordinates for this analysis.",
            " Otherwise the results are potentially erroneous.",
        )

    def analyze_frame(self, split_frame, frame_counter):
        """
        Calculate the loading mass of the liquid within the CNTs.
        For this we first need to calculate the center of geometry
        for each open end of the CNT. (ring1, ring2)

        """
        print(split_frame)
        # turn the X, Y and Z values into floats
        split_frame["X"] = split_frame["X"].astype(float)
        split_frame["Y"] = split_frame["Y"].astype(float)
        split_frame["Z"] = split_frame["Z"].astype(float)

        # calculate the center of geometry for the ring1 atoms
        ring1 = split_frame.loc[self.ring1]
        ring1_ref = ring1.iloc[0][["X", "Y", "Z"]].values
        ring1_x = ring1["X"].mean()
        ring1_y = ring1["Y"].mean()
        ring1_z = ring1["Z"].mean()

        ring2 = split_frame.loc[self.ring2]
        ring2_ref = ring2.iloc[0][["X", "Y", "Z"]].values
        ring2_x = ring2["X"].mean()
        ring2_y = ring2["Y"].mean()
        ring2_z = ring2["Z"].mean()

        ring1_array = np.array([ring1_x, ring1_y, ring1_z])
        ring2_array = np.array([ring2_x, ring2_y, ring2_z])

        print(ring1_array, ring2_array)

        # calculate the distance between the mean posistion and a reference atom of the same ring
        dist_ring1 = np.linalg.norm(traj_info.minimum_image_distance(ring1_array, ring1_ref, self.traj_file.box_size))
        dist_ring2 = np.linalg.norm(traj_info.minimum_image_distance(ring2_array, ring2_ref, self.traj_file.box_size))

        print(dist_ring1, dist_ring2)

        dist = np.linalg.norm(traj_info.minimum_image_distance(ring1_array, ring2_array, self.traj_file.box_size))

        print(dist)

        """ Now we calculate the mass of the liquid within the CNTs.
        First identify which species are within the CNTs"""
