import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

import conan.analysis_modules.traj_an as traj_an
import conan.analysis_modules.traj_info as traj_info
import conan.defdict as ddict


# check if a given point is within a cylinder
def points_in_cylinder(pt1, pt2, r, atom_to_check):
    pt1 = np.array(pt1, dtype=np.float64)
    pt2 = np.array(pt2, dtype=np.float64)
    atom_to_check = np.array(atom_to_check, dtype=np.float64)

    vec = pt2 - pt1
    const = r * np.linalg.norm(vec)
    return (
        (np.dot(atom_to_check - pt1, vec) >= 0)
        and (np.dot(atom_to_check - pt2, vec) <= 0)
        and (np.linalg.norm(np.cross(atom_to_check - pt1, vec)) <= const)
    )


def cnt_loading_mass(traj_file, molecules, an):
    clm = CNTload(traj_file, molecules, an)
    clm.cnt_loading_mass_prep()
    traj_an.process_trajectory(traj_file, molecules, an, clm)
    clm.cnt_loading_mass_processing()


class CNTload:
    """
    Class to calculate the loading mass of the liquid within the carbon nanotube.
    """

    def __init__(self, traj_file, molecules, an):
        self.traj_file = traj_file
        self.molecules = molecules
        self.an = an
        self.proc_frame_counter = 0
        self.shortening_q = "n"
        self.shortening = 0

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
        pore_No = pore_atoms["Struc"].nunique()

        pore_atoms["bonds"] = 0

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

        molecules = list(nx.connected_components(G))

        pore_atoms["ring"] = 0

        for i in range(len(molecules)):
            for j in molecules[i]:
                pore_atoms.loc[j, "ring"] = i + 1

        # ddict.printLog(pore_atoms[pore_atoms["bonds"] == 2])

        # save the index of ring1 and ring2 atoms to seperate numpy arrays
        self.ring1 = np.array([atom for atom in pore_atoms[pore_atoms["ring"] == 1].index])
        self.ring2 = np.array([atom for atom in pore_atoms[pore_atoms["ring"] == 2].index])

        ddict.printLog(
            "\nTo ensure that the program delivers correct results,"
            " it must be ensured that all atoms are wrapped in the simulation box"
            " and that the CNT does not cross the PBC boundaries."
            " For pre-processing, please use e.g. Travis.\n",
            color="red",
        )

        self.traj_file.frame0["x"] = self.traj_file.frame0["x"].astype(float)
        self.traj_file.frame0["y"] = self.traj_file.frame0["y"].astype(float)
        self.traj_file.frame0["z"] = self.traj_file.frame0["z"].astype(float)

        # calculate the center of geometry for the ring1 atoms
        ring1 = self.traj_file.frame0.loc[self.ring1]
        ring1_ref = ring1.iloc[0][["x", "y", "z"]].values
        ring1_x = ring1["x"].mean()
        ring1_y = ring1["y"].mean()
        ring1_z = ring1["z"].mean()

        ring2 = self.traj_file.frame0.loc[self.ring2]
        ring2_x = ring2["x"].mean()
        ring2_y = ring2["y"].mean()
        ring2_z = ring2["z"].mean()

        ring1_array = np.array([ring1_x, ring1_y, ring1_z])
        ring2_array = np.array([ring2_x, ring2_y, ring2_z])
        ring1_ref = np.array([ring1_ref[0], ring1_ref[1], ring1_ref[2]])

        # self.dist_ring = np.linalg.norm(traj_info.minimum_image_distance(
        # ring1_array, ring1_ref, self.traj_file.box_size))
        self.dist_ring = np.linalg.norm(ring1_array - ring1_ref)
        ddict.printLog(f"Radius of the CNT: {np.round(self.dist_ring, 3)}")

        # self.dist = np.linalg.norm(traj_info.minimum_image_distance(
        # ring1_array, ring2_array, self.traj_file.box_size))
        self.dist = np.linalg.norm(ring1_array - ring2_array)

        ddict.printLog(f"Distance between the rings: {np.round(self.dist, 3)}")

        """The user shall have the option to select an region within the CNTs to calculate the loading mass.
        This is preferred, as the liquid might behave somewhat differently at the openeing of the CNTs.
        In Order to do this, the user defines a distane,
        which is used to diplace the center of geometry of the rings along the CNT axis.
        The distance is defined in angstroms."""

        # first we need to define the normal vector of the CNT axis. It is the vector between the two rings.
        # The vector is normalized.
        cnt_axis = (ring2_array - ring1_array) / np.linalg.norm(ring2_array - ring1_array)
        ddict.printLog(f"CNT axis: {cnt_axis}")

        self.shortening_q = ddict.get_input(
            "Do you want to shorten the CNT axis (space to analyze within the CNT)? (y/n): ",
            self.traj_file.args,
            "string",
        )
        if self.shortening_q == "y":
            self.shortening = float(
                ddict.get_input(
                    "Please enter the amount you want the axis to be shortened"
                    " (amount is subtracted from both sides): ",
                    self.traj_file.args,
                    "float",
                )
            )

            ring1_array = ring1_array + self.shortening * cnt_axis
            ring2_array = ring2_array - self.shortening * cnt_axis
            # self.dist = np.linalg.norm(traj_info.minimum_image_distance(
            # ring1_array, ring2_array, self.traj_file.box_size))
            self.dist = np.linalg.norm(ring1_array - ring2_array)
        print("Length of CNT axis considered:", self.dist.round(3))

        # finally set up an array to store the liquid mass each time frame which is processed.
        self.liquid_mass = 0
        self.frame_masses = np.array([])

    def analyze_frame(self, split_frame, frame_counter):
        """
        Calculate the loading mass of the liquid within the CNTs.
        For this we first need to calculate the center of geometry
        for each open end of the CNT. (ring1, ring2)

        """

        frame_mass = 0
        # turn the X, Y and Z values into floats
        split_frame["X"] = split_frame["X"].astype(float)
        split_frame["Y"] = split_frame["Y"].astype(float)
        split_frame["Z"] = split_frame["Z"].astype(float)

        # calculate the center of geometry of the rings
        ring1 = split_frame.loc[self.ring1]
        ring1_x = ring1["X"].mean()
        ring1_y = ring1["Y"].mean()
        ring1_z = ring1["Z"].mean()

        ring2 = split_frame.loc[self.ring2]
        ring2_x = ring2["X"].mean()
        ring2_y = ring2["Y"].mean()
        ring2_z = ring2["Z"].mean()

        ring1_array = np.array([ring1_x, ring1_y, ring1_z])
        ring2_array = np.array([ring2_x, ring2_y, ring2_z])

        """
        This is the shortening part regarding the cnt space to observe
        """
        # calculate the normalized vector between the two rings
        cnt_axis = (ring2_array - ring1_array) / np.linalg.norm(ring2_array - ring1_array)

        ring1_array = ring1_array + self.shortening * cnt_axis
        ring2_array = ring2_array - self.shortening * cnt_axis
        # self.dist = np.linalg.norm(
        #    traj_info.minimum_image_distance(ring1_array, ring2_array, self.traj_file.box_size))
        self.dist = np.linalg.norm(ring1_array - ring2_array)

        """ Calculate the mass of the liquid within the CNTs.
        First identify which species are within the CNTs.
        For this use the points in cylider function."""

        liquid_atoms = split_frame[split_frame["Struc"].str.contains("Liquid")].copy()

        # get the atom positions of the current frame
        atom_positions = np.array(
            [[atom["X"], atom["Y"], atom["Z"], atom["Mass"], atom.index] for index, atom in liquid_atoms.iterrows()],
            dtype=object,
        )

        # check if a liquid atom is within the CNT
        for atom_position in atom_positions:
            if points_in_cylinder(ring1_array, ring2_array, self.dist_ring, atom_position[:3]):
                self.liquid_mass += atom_position[3]
                frame_mass += atom_position[3]

        self.frame_masses = np.append(self.frame_masses, frame_mass)

        self.proc_frame_counter += 1

    def cnt_loading_mass_processing(self):
        """
        Calculate the loading mass of the liquid within the CNTs.
        """

        self.liq_mass_per_frame = self.liquid_mass / self.proc_frame_counter
        self.liq_mass_per_angstrom = self.liq_mass_per_frame / self.dist

        ddict.printLog(f"Average confined mass: {np.round(self.liq_mass_per_frame, 5)}")
        ddict.printLog(f"Average mass per \u00c5: {np.round(self.liq_mass_per_angstrom, 5)}")
        self.liquid_mass = 0

        pd_frame_masses = pd.DataFrame(self.frame_masses)
        pd_frame_masses.columns = ["Frame_masses"]

        # check if the CNT is empty, all entries are zero
        if np.all(pd_frame_masses["Frame_masses"] == 0):
            ddict.printLog("CNT is empty, no liquid is present.", color="red")
            return

        # add a new column with the average of 5, 10 and 50 frames
        pd_frame_masses["5_frame_average"] = pd_frame_masses["Frame_masses"].rolling(window=5).mean().shift(-2)
        pd_frame_masses["10_frame_average"] = pd_frame_masses["Frame_masses"].rolling(window=10).mean().shift(-5)
        pd_frame_masses["50_frame_average"] = pd_frame_masses["Frame_masses"].rolling(window=50).mean().shift(-25)

        # plot the results
        fig, ax = plt.subplots()
        ax.plot(pd_frame_masses["Frame_masses"], label="Frame", color="#440154")
        ax.plot(pd_frame_masses["5_frame_average"], label="5 frame average", color="#21908C")
        ax.plot(pd_frame_masses["10_frame_average"], label="10 frame average", color="#5ec962")
        ax.plot(pd_frame_masses["50_frame_average"], label="50 frame average", color="#fde725")
        ax.axhline(y=self.liq_mass_per_frame, color="darkgray", linestyle="--", label="Average mass")

        ax.set_xlabel("Frame number")
        ax.set_ylabel("Confined mass / u")
        ax.legend()
        ax.grid()

        # save the plot
        fig.savefig("conf_mass_temp.png", dpi=300)

        pd_mass_per_angstrom = pd_frame_masses.copy().div(self.dist)
        pd_mass_per_angstrom.rename(columns={"Frame_masses": "Mass_per_angstrom"}, inplace=True)

        # plot the results per angstrom
        fig, ax = plt.subplots()
        ax.plot(pd_mass_per_angstrom["Mass_per_angstrom"], label="Frame", color="#440154")
        ax.plot(pd_mass_per_angstrom["5_frame_average"], label="5 frame average", color="#21908C")
        ax.plot(pd_mass_per_angstrom["10_frame_average"], label="10 frame average", color="#5ec962")
        ax.plot(pd_mass_per_angstrom["50_frame_average"], label="50 frame average", color="#fde725")
        ax.axhline(y=self.liq_mass_per_angstrom, color="darkgray", linestyle="--", label="Average mass")

        ax.set_xlabel("Frame Number")
        ax.set_ylabel("Confined mass / u")
        ax.legend()
        ax.grid()

        # save the data
        fig.savefig("conf_mass_t_ang.png", dpi=300)
        pd_frame_masses.to_csv("frame_masses.csv")
        pd_mass_per_angstrom.to_csv("mass_per_angstrom.csv")
