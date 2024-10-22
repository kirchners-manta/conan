import math
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

import conan.analysis_modules.traj_an as traj_an
import conan.defdict as ddict
from conan.analysis_modules import utils as ut


def accessible_volume_analysis(traj_file, molecules, an):
    accvol = AccessibleVolumeAnalysis(traj_file, molecules)
    accvol.accessible_volume_prep()
    traj_an.process_trajectory(traj_file, molecules, an, accvol)
    accvol.accessible_volume_processing()


class AccessibleVolumeAnalysis:
    def __init__(self, traj_file, molecules):
        self.traj_file = traj_file
        self.molecules = molecules
        self.element_radii = {}
        self.maxdisp_atom_dist = 0
        self.maxdisp_atom_row = None
        self.CNT_length = molecules.length_pore[0]
        self.CNT_atoms = molecules.CNT_atoms

    def accessible_volume_prep(self):
        ddict.printLog("")
        if len(self.molecules.structure_data["CNT_centers"]) > 1:
            ddict.printLog("-> Multiple CNTs detected. The analysis will be conducted on the first CNT.\n", color="red")
        if len(self.molecules.structure_data["CNT_centers"]) == 0:
            ddict.printLog("-> No CNTs detected. Aborting...\n", color="red")
            sys.exit(1)

        which_element_radii = ddict.get_input(
            "Do you want to use the van der Waals radii (1) or the covalent radii (2) of the elements? [1/2] ",
            self.traj_file.args,
            "int",
        )
        if which_element_radii == 1:
            ddict.printLog("-> Using van der Waals radii.")
            self.element_radii = ddict.dict_vdW()
        elif which_element_radii == 2:
            ddict.printLog("-> Using covalent radii.")
            self.element_radii = ddict.dict_covalent()

    def analyze_frame(self, split_frame, frame_counter):
        CNT_centers = self.molecules.CNT_centers
        max_z_pore = self.molecules.max_z_pore
        min_z_pore = self.molecules.min_z_pore

        # Apply periodic boundary conditions
        split_frame["X"] = split_frame["X"].astype(float) % self.traj_file.box_size[0]
        split_frame["Y"] = split_frame["Y"].astype(float) % self.traj_file.box_size[1]
        split_frame["Z"] = split_frame["Z"].astype(float) % self.traj_file.box_size[2]

        max_z_pore[0] = max_z_pore[0] % self.traj_file.box_size[2]
        min_z_pore[0] = min_z_pore[0] % self.traj_file.box_size[2]
        CNT_centers[0][0] = CNT_centers[0][0] % self.traj_file.box_size[0]
        CNT_centers[0][1] = CNT_centers[0][1] % self.traj_file.box_size[1]

        # Handle cases where the pore is split over the periodic boundary
        if min_z_pore[0] > max_z_pore[0]:
            part1 = split_frame[split_frame["Z"].astype(float) >= min_z_pore[0]].copy()
            part2 = split_frame[split_frame["Z"].astype(float) <= max_z_pore[0]].copy()
            split_frame = pd.concat([part1, part2])
        else:
            split_frame = split_frame[split_frame["Z"].astype(float) <= max_z_pore[0]].copy()
            split_frame = split_frame[split_frame["Z"].astype(float) >= min_z_pore[0]].copy()

        # Adjust the coordinates relative to the CNT center
        split_frame.loc[:, "X_adjust"] = split_frame["X"].astype(float) - CNT_centers[0][0]
        split_frame.loc[:, "Y_adjust"] = split_frame["Y"].astype(float) - CNT_centers[0][1]

        # Calculate the radial distance and add the atomic radius
        split_frame.loc[:, "Distance"] = np.sqrt(
            split_frame["X_adjust"] ** 2 + split_frame["Y_adjust"] ** 2
        ) + split_frame["Atom"].map(self.element_radii)

        # Update the most displaced atom if necessary
        if split_frame["Distance"].max() > self.maxdisp_atom_dist:
            self.maxdisp_atom_dist = split_frame["Distance"].max()
            self.maxdisp_atom_row = split_frame.loc[split_frame["Distance"].idxmax()]

    def accessible_volume_processing(self):
        if self.maxdisp_atom_row is None:
            ddict.printLog("No displaced atom found.")
            return

        accessible_radius = self.maxdisp_atom_row["Distance"]
        ddict.printLog("Maximal displacement: %0.3f" % accessible_radius)
        ddict.printLog("Accessible volume: %0.3f" % (math.pi * accessible_radius**2 * self.CNT_length))

        # Optional: save the pore with the most displaced atom
        pore_disp_atom = ddict.get_input(
            "Do you want to produce a xyz file with the pore including the most displaced atom? [y/n] ",
            self.traj_file.args,
            "string",
        )
        if pore_disp_atom == "y":
            self.save_pore_with_displaced_atom()

    def save_pore_with_displaced_atom(self):
        with open("pore_disp_atom.xyz", "w") as f:
            f.write("%d\n#Pore with most displaced atom\n" % (len(self.CNT_atoms) + 1))
            for index, row in self.CNT_atoms.iterrows():
                f.write("%s\t%0.3f\t%0.3f\t%0.3f\n" % (row["Element"], row["x"], row["y"], row["z"]))
            f.write(
                "%s\t%0.3f\t%0.3f\t%0.3f\n"
                % (
                    self.maxdisp_atom_row["Atom"],
                    float(self.maxdisp_atom_row["X"]),
                    float(self.maxdisp_atom_row["Y"]),
                    float(self.maxdisp_atom_row["Z"]),
                )
            )
        ddict.printLog("Pore with most displaced atom saved as pore_disp_atom.xyz")


def distance_search_analysis(traj_file, molecules, an):
    dist_search = DistanceSearchAnalysis(traj_file, molecules)
    dist_search.distance_search_prep()
    traj_an.process_trajectory(traj_file, molecules, an, dist_search)
    dist_search.distance_search_processing()


class DistanceSearchAnalysis:
    def __init__(self, traj_file, molecules):
        self.traj_file = traj_file
        self.molecules = molecules
        self.structure_atoms_tree = None
        self.minimal_distance = 1000
        self.maximal_distance = 0
        self.minimal_distance_row = None
        self.maximal_distance_row = None

    def distance_search_prep(self):
        # Drop all rows labeled 'Liquid' in the 'Struc' column
        structure_atoms = self.traj_file.frame0[self.traj_file.frame0["Struc"] != "Liquid"]

        # Transform the structure atoms into a KD-tree
        self.structure_atoms_tree = scipy.spatial.KDTree(structure_atoms[["x", "y", "z"]].values)

    def analyze_frame(self, split_frame, frame_counter):
        # Ensure coordinates are within the simulation box using PBC (wrapping coordinates)
        split_frame = ut.wrapping_coordinates(self.traj_file.box_size, split_frame)

        # Extract the coordinates of the split_frame
        split_frame_coords = split_frame[["X", "Y", "Z"]].values

        # Query the KD-tree for the closest atom to each atom in split_frame
        closest_atom_dist, closest_atom_idx = self.structure_atoms_tree.query(split_frame_coords)

        # Add the distances to the DataFrame
        split_frame["Distance"] = closest_atom_dist

        # Find the minimum and maximum distances
        min_dist = split_frame["Distance"].min()
        max_dist = split_frame["Distance"].max()

        # Update minimal and maximal distances and rows
        if min_dist < self.minimal_distance:
            self.minimal_distance = min_dist
            self.minimal_distance_row = split_frame.loc[split_frame["Distance"].idxmin()]

        if max_dist > self.maximal_distance:
            self.maximal_distance = max_dist
            self.maximal_distance_row = split_frame.loc[split_frame["Distance"].idxmax()]

    def distance_search_processing(self):
        if self.minimal_distance_row is not None and self.maximal_distance_row is not None:
            ddict.printLog(
                "The closest atom is: ",
                self.minimal_distance_row["Atom"],
                " with a distance of: ",
                round(self.minimal_distance, 2),
                " \u00C5",
            )
            ddict.printLog(
                "The furthest atom is: ",
                self.maximal_distance_row["Atom"],
                " with a distance of: ",
                round(self.maximal_distance, 2),
                " \u00C5",
            )
        else:
            ddict.printLog("No atoms found in the specified distance search.")


def axial_density_analysis(traj_file, molecules, an):
    axial_density = AxialDensityAnalysis(traj_file, molecules)
    axial_density.axial_density_prep()
    traj_an.process_trajectory(traj_file, molecules, an, axial_density)
    axial_density.axial_density_processing()


class AxialDensityAnalysis:
    def __init__(self, traj_file, molecules):
        self.traj_file = traj_file
        self.molecules = molecules
        self.element_radii = {}
        self.maxdisp_atom_dist = 0
        self.z_incr_CNT = []
        self.z_incr_bulk1 = []
        self.z_incr_bulk2 = []
        self.z_bin_edges_pore = []
        self.z_bin_edges_bulk1 = []
        self.z_bin_edges_bulk2 = []
        self.z_bin_edges = []
        self.z_bin_labels = []
        self.zdens_df = None
        self.num_increments = 0
        self.maxdisp_atom_row = None
        self.proc_frames = 0

    def axial_density_prep(self):
        args = self.traj_file.args

        # Number of increments for each section
        self.num_increments = int(
            ddict.get_input(
                "How many increments per section do you want to use to calculate the density profile? ", args, "int"
            )
        )
        ddict.printLog(
            "\nThe simulation box is subdivided into two bulk phases and the pore. "
            "The number of increments set here is the number of increments for each section.\n",
            color="red",
        )

        CNT_atoms = self.molecules.CNT_atoms
        max_z_pore = self.molecules.max_z_pore
        min_z_pore = self.molecules.min_z_pore
        CNT_centers = self.molecules.CNT_centers

        # Wrap periodic boundary conditions for CNT centers and CNT_atoms
        self._wrap_pbc_values(CNT_centers, CNT_atoms, self.traj_file)

        # Initialize increment distances and bin edges for each section
        for i in range(len(CNT_centers)):
            self.z_incr_CNT.append(self.molecules.length_pore[i] / self.num_increments)
            self.z_incr_bulk1.append(min_z_pore[i] / self.num_increments)
            self.z_incr_bulk2.append((self.traj_file.box_size[2] - max_z_pore[i]) / self.num_increments)

            ddict.printLog(f"Increment distance CNT: {self.z_incr_CNT[i]:.3f} Å")
            ddict.printLog(f"Increment distance bulk1: {self.z_incr_bulk1[i]:.3f} Å")
            ddict.printLog(f"Increment distance bulk2: {self.z_incr_bulk2[i]:.3f} Å")

            self._calculate_bin_edges(i, CNT_atoms, max_z_pore, min_z_pore)

        # Select element radii for displacement calculation
        self._select_element_radii()

        # Prepare DataFrame for storing axial density information
        self.zdens_df = self._initialize_zdens_df()

    def _wrap_pbc_values(self, CNT_centers, CNT_atoms, traj_file):
        CNT_centers[0][0] = CNT_centers[0][0] % traj_file.box_size[0]
        CNT_centers[0][1] = CNT_centers[0][1] % traj_file.box_size[1]

        CNT_atoms = CNT_atoms.copy()
        CNT_atoms.loc[:, "x"] = CNT_atoms["x"] % traj_file.box_size[0]
        CNT_atoms.loc[:, "y"] = CNT_atoms["y"] % traj_file.box_size[1]
        CNT_atoms.loc[:, "z"] = CNT_atoms["z"] % traj_file.box_size[2]

    def _calculate_bin_edges(self, i, CNT_atoms, max_z_pore, min_z_pore):
        self.z_bin_edges_pore.append(np.linspace(CNT_atoms["z"].min(), CNT_atoms["z"].max(), self.num_increments + 1))
        self.z_bin_edges_bulk1.append(np.linspace(0, min_z_pore[i], self.num_increments + 1))
        self.z_bin_edges_bulk2.append(np.linspace(max_z_pore[i], self.traj_file.box_size[2], self.num_increments + 1))

        self.z_bin_edges.append(
            np.concatenate((self.z_bin_edges_bulk1[i], self.z_bin_edges_pore[i], self.z_bin_edges_bulk2[i]))
        )
        self.z_bin_edges[i] = np.unique(self.z_bin_edges[i])
        self.num_increments = len(self.z_bin_edges[i]) - 1

        ddict.printLog(f"\nTotal number of increments: {self.num_increments}")
        ddict.printLog(f"Number of edges: {len(self.z_bin_edges[i])}")
        self.z_bin_labels.append(np.arange(1, len(self.z_bin_edges[i]), 1))

    def _select_element_radii(self):
        args = self.traj_file.args
        which_element_radii = ddict.get_input(
            "Do you want to use the van der Waals radii (1) or the covalent radii (2) of the elements? [1/2] ",
            args,
            "int",
        )
        if which_element_radii == 1:
            ddict.printLog("-> Using van der Waals radii.")
            self.element_radii = ddict.dict_vdW()
        elif which_element_radii == 2:
            ddict.printLog("-> Using covalent radii.")
            self.element_radii = ddict.dict_covalent()

    def _initialize_zdens_df(self):
        # Create a dictionary to store the data for each column
        data = {"Frame": np.arange(1, self.traj_file.number_of_frames + 1)}

        # Add columns for each bin at once
        for i in range(self.num_increments):
            data[f"Bin {i + 1}"] = np.zeros(self.traj_file.number_of_frames)  # Initialize with zeros

        # Convert the dictionary into a DataFrame
        zdens_df = pd.DataFrame(data)

        # Return the DataFrame
        return zdens_df.copy()

    def analyze_frame(self, split_frame, frame_counter):
        # Wrap coordinates into the simulation box using PBC
        split_frame = ut.wrapping_coordinates(self.traj_file.box_size, split_frame)

        # Create bins and assign atoms to z bins
        z_bin_edges = np.ravel(self.z_bin_edges)
        z_bin_labels = np.ravel(self.z_bin_labels)

        # Ensure we work on a copy to avoid the warning
        split_frame["Z_bin"] = pd.cut(
            split_frame["Z"].astype(float).values, bins=z_bin_edges, labels=z_bin_labels
        ).copy()

        # Aggregate atom masses into bins
        zdens_df_temp = (
            split_frame.groupby(pd.cut(split_frame["Z"].astype(float), z_bin_edges))["Mass"]
            .sum()
            .reset_index(name="Weighted_counts")
            .copy()
        )
        zdens_df_temp["Counts"] = (
            split_frame.groupby(pd.cut(split_frame["Z"].astype(float), z_bin_edges))["Mass"]
            .count()
            .reset_index(name="Counts")["Counts"]
            .copy()
        )
        zdens_df_temp.insert(0, "Bin", zdens_df_temp.index + 1)

        # Write results to the z-density dataframe for the current frame
        for i in range(self.num_increments):
            self.zdens_df.loc[frame_counter - 1, f"Bin {i + 1}"] = zdens_df_temp.loc[i, "Weighted_counts"]

        # Calculate the most displaced atom
        self._calculate_displaced_atom(split_frame)

    def _calculate_displaced_atom(self, split_frame):

        if self.molecules.min_z_pore[0] > self.molecules.max_z_pore[0]:
            # Split the selection into two parts
            part1 = split_frame[split_frame["Z"].astype(float) >= self.molecules.min_z_pore[0]].copy()
            part2 = split_frame[split_frame["Z"].astype(float) <= self.molecules.max_z_pore[0]].copy()
            split_frame = pd.concat([part1, part2])
        else:
            split_frame = split_frame[split_frame["Z"].astype(float) <= self.molecules.max_z_pore[0]].copy()
            split_frame = split_frame[split_frame["Z"].astype(float) >= self.molecules.min_z_pore[0]].copy()

        # Update coordinates using PBC
        split_frame["X"] = split_frame["X"].astype(float) % self.traj_file.box_size[0]
        split_frame["Y"] = split_frame["Y"].astype(float) % self.traj_file.box_size[1]
        split_frame["Z"] = split_frame["Z"].astype(float) % self.traj_file.box_size[2]

        # Adjust coordinates relative to CNT center and calculate distance
        split_frame["X_adjust"] = split_frame["X"].astype(float) - self.molecules.CNT_centers[0][0]
        split_frame["Y_adjust"] = split_frame["Y"].astype(float) - self.molecules.CNT_centers[0][1]

        # Add element radius and calculate distance from CNT center
        split_frame["Distance"] = np.sqrt(split_frame["X_adjust"] ** 2 + split_frame["Y_adjust"] ** 2) + split_frame[
            "Atom"
        ].map(self.element_radii)

        if split_frame["Distance"].max() > self.maxdisp_atom_dist:
            self.maxdisp_atom_dist = split_frame["Distance"].max()
            self.maxdisp_atom_row = split_frame.loc[split_frame["Distance"].idxmax()]

    def axial_density_processing(self):
        args = self.traj_file.args
        zdens_df = self.zdens_df
        bin_edges = np.ravel(self.z_bin_edges)
        num_increments = self.num_increments

        # Compute averaged density and volumes
        results_zd_df = pd.DataFrame(zdens_df.iloc[:, 1:].sum(axis=0) / self.proc_frames, columns=["Mass"]).copy()
        results_zd_df["Bin_lowedge"] = bin_edges[:-1]
        results_zd_df["Bin_highedge"] = bin_edges[1:]
        results_zd_df["Bin_center"] = (bin_edges[1:] + bin_edges[:-1]) / 2

        # write the index to a new column named "Bin" (as column 0)
        results_zd_df["Bin"] = results_zd_df.index

        # Choose radius for volume calculation
        used_radius = self._choose_radius_for_volume(args)

        # Calculate volume and density for each bin
        vol_increment = self._calculate_bin_volumes(bin_edges, num_increments, used_radius)

        # Finalize results and compute density
        results_zd_df["Volume"] = vol_increment
        results_zd_df["Density [u/Ang^3]"] = results_zd_df["Mass"] / results_zd_df["Volume"]

        # Convert density to g/cm^3
        results_zd_df["Density [g/cm^3]"] = results_zd_df["Density [u/Ang^3]"] * 1.66053907

        # Option to shift the center of the box to zero
        if (
            ddict.get_input("Do you want to set the center of the simulation box to zero? (y/n) ", args, "string")
            == "y"
        ):
            results_zd_df["Bin_center"] -= self.molecules.CNT_centers[0][2]

        # Plot and save data
        self._plot_and_save_results(results_zd_df)

    def _choose_radius_for_volume(self, args):
        which_radius = ddict.get_input(
            "Do you want to use the accessible radius (1) or the CNT radius (2)",
            " to compute the increments' volume? [1/2] ",
            args,
            "string",
        )
        if self.maxdisp_atom_row is None:
            ddict.printLog("None of the analyzed atoms were found in the tube")
            return 0
        else:
            accessible_radius = self.maxdisp_atom_row["Distance"]
            ddict.printLog(f"Maximal displacement: {accessible_radius:.3f} Å")
            ddict.printLog(
                f"Accessible volume: {math.pi * accessible_radius**2 * self.molecules.length_pore[0]:.3f} Å³"
            )

            if which_radius == "1":
                return accessible_radius
            elif which_radius == "2":
                return self.molecules.tuberadii[0]

    def _calculate_bin_volumes(self, bin_edges, num_increments, used_radius):
        vol_increment = []
        for i in range(num_increments):
            if bin_edges[i] < self.molecules.min_z_pore or bin_edges[i + 1] > self.molecules.max_z_pore:
                vol_increment.append(
                    self.traj_file.box_size[0] * self.traj_file.box_size[1] * (bin_edges[i + 1] - bin_edges[i])
                )
            else:
                vol_increment.append(math.pi * (used_radius**2) * (bin_edges[i + 1] - bin_edges[i]))
        return vol_increment

    def _plot_and_save_results(self, results_zd_df):
        if ddict.get_input("Do you want to plot the data? (y/n) ", self.traj_file.args, "string") == "y":
            fig, ax = plt.subplots()
            ax.plot(
                results_zd_df["Bin_center"],
                results_zd_df["Density [g/cm^3]"],
                "-",
                label="Axial density function",
                color="black",
            )
            ax.set(xlabel="Distance from tube center [Å]", ylabel="Density [g/cm³]", title="Axial density function")
            ax.grid()
            fig.savefig("Axial_density.pdf")
            ddict.printLog("\nAxial density function saved as Axial_density.pdf")

        results_zd_df.to_csv("Axial_density.csv", sep=";", index=True, header=True, float_format="%.5f")
        self.zdens_df.to_csv("Axial_mass_dist_raw.csv", sep=";", index=False, header=True, float_format="%.5f")
        ddict.printLog("\nRaw data saved as Axial_mass_dist_raw.csv")


# 3D density analysis
""" What this function is about:

    This function is used to calculate the 3D density of the system.
    First it creates a 3D grid with the dimensions of the simulation box.
    The incrementation is set by the user.
    Then all atoms of interest are assigned to the grid for every frame.
    In the end we have a 3D grid with the density information for each grid point for each individual atom/molecule.
    This data can then be used to calculate the density profile of the system.
    Visualization is done with a cube file.
    """


def density_analysis_3D(traj_file, molecules, an):
    dens_analyzer = DensityAnalysis(traj_file, molecules)
    dens_analyzer.density_analysis_prep()
    traj_an.process_trajectory(traj_file, molecules, an, dens_analyzer)
    dens_analyzer.density_analysis_processing()


class DensityAnalysis:
    def __init__(self, traj_file, molecules):
        self.traj_file = traj_file
        self.molecules = molecules
        self.cube_array = None
        self.grid_points_tree = None
        self.grid_point_atom_labels = None
        self.analysis_counter = 0
        self.grid_point_chunk_atom_labels = None

    def density_analysis_prep(self):
        inputdict = ut.grid_generator(
            {"box_size": self.traj_file.box_size, "traj_file": self.traj_file, "args": self.traj_file.args}
        )

        self.maindict = inputdict

        # Retrieve mesh dimensions and increments from inputdict
        self.x_incr = inputdict["x_incr"]
        self.y_incr = inputdict["y_incr"]
        self.z_incr = inputdict["z_incr"]
        self.x_mesh = inputdict["x_mesh"]
        self.y_mesh = inputdict["y_mesh"]
        self.z_mesh = inputdict["z_mesh"]
        self.x_grid = inputdict["x_grid"]
        self.y_grid = inputdict["y_grid"]
        self.z_grid = inputdict["z_grid"]

        self.x_incr_dist = inputdict["x_incr_dist"]
        self.y_incr_dist = inputdict["y_incr_dist"]
        self.z_incr_dist = inputdict["z_incr_dist"]

        # Initialize the cube array for density data
        self.cube_array = np.zeros((self.x_incr * self.y_incr * self.z_incr))

        # Generate KD-tree from grid points
        grid_points = np.vstack((self.x_mesh.flatten(), self.y_mesh.flatten(), self.z_mesh.flatten())).T
        # the index of the grid point is the same as the index in the cube array
        self.grid_points_tree = scipy.spatial.cKDTree(grid_points)

        # Initialize lists to hold atom labels for each grid point
        self.grid_point_atom_labels = [[] for _ in range(len(grid_points))]
        self.grid_point_chunk_atom_labels = [Counter() for _ in range(len(grid_points))]

    def analyze_frame(self, split_frame, frame_counter):
        box_size = self.traj_file.box_size

        # Wrap coordinates into the simulation box using PBC
        split_frame = ut.wrapping_coordinates(box_size, split_frame)

        # Get coordinates of atoms in the current frame
        split_frame_coords = np.array(split_frame[["X", "Y", "Z"]]).astype(float)

        # Find the closest grid point for each atom
        closest_grid_point_dist, closest_grid_point_idx = self.grid_points_tree.query(split_frame_coords)

        # Update cube array and grid_point_atom_labels
        self.cube_array[closest_grid_point_idx] += split_frame["Mass"].values
        for i in range(len(split_frame)):
            self.grid_point_atom_labels[closest_grid_point_idx[i]].append(split_frame["Label"].values[i])

        self.analysis_counter += 1
        # Perform chunk processing every 500 frames
        if self.analysis_counter == 500:
            self.chunk_processing()
            self.analysis_counter = 0

    def chunk_processing(self):
        for i, atom_labels in enumerate(self.grid_point_atom_labels):
            self.grid_point_chunk_atom_labels[i].update(atom_labels)
        self.grid_point_atom_labels = [[] for _ in range(len(self.grid_point_atom_labels))]

    def density_analysis_processing(self):
        # Perform final chunk processing
        self.chunk_processing()

        # Extract density information and calculate densities
        self.calculate_grid_point_densities()

    def calculate_grid_point_densities(self):

        grid_volume = self.x_incr_dist * self.y_incr_dist * self.z_incr_dist
        print("Volume of each grid point: %0.3f \u00C5\u00B3" % grid_volume)

        # Initialize list to store densities of grid points
        grid_point_densities = []

        # Calculate total mass and density for each grid point
        list_of_masses = self.get_masses_per_molecule()

        for grid_point in self.grid_point_chunk_atom_labels:
            total_mass = 0
            for label_count_pair in grid_point.items():
                label = label_count_pair[0]
                count = label_count_pair[1]
                for molecule_mass_dict in list_of_masses:
                    if label in molecule_mass_dict:
                        total_mass += molecule_mass_dict[label] * count
                        break
            grid_point_density = total_mass / grid_volume / self.proc_frames
            grid_point_densities.append(grid_point_density)

        self.grid_point_densities = grid_point_densities
        ut.write_cube_file(self, filename="density.cube")

        # Extract density profiles along x, y, and z axes
        self.extract_density_profiles()

    def get_masses_per_molecule(self):
        unique_molecule_frame = self.molecules.unique_molecule_frame
        unique_molecule_frame["Masses"] = unique_molecule_frame["Atoms_sym"].apply(ut.symbols_to_masses)

        list_of_masses = []
        for _, row in unique_molecule_frame.iterrows():
            list_of_masses.append(dict(zip(row["Labels"], row["Masses"])))
        return list_of_masses

    def extract_density_profiles(self):
        grid_point_densities = np.array(self.grid_point_densities).reshape(self.x_incr, self.y_incr, self.z_incr)

        # Extract density profiles along x, y, z axes
        x_dens_profile = np.sum(grid_point_densities, axis=(1, 2))
        y_dens_profile = np.sum(grid_point_densities, axis=(0, 2))
        z_dens_profile = np.sum(grid_point_densities, axis=(0, 1))

        # Sum grid points along each axis
        sum_gp_x = int(self.y_incr * self.z_incr)
        sum_gp_y = int(self.x_incr * self.z_incr)
        sum_gp_z = int(self.x_incr * self.y_incr)

        # Normalize density profiles by number of grid points
        x_dens_profile /= sum_gp_x
        y_dens_profile /= sum_gp_y
        z_dens_profile /= sum_gp_z

        # Save and plot density profiles
        self.save_density_profiles(x_dens_profile, y_dens_profile, z_dens_profile)

    def save_density_profiles(self, x_dens_profile, y_dens_profile, z_dens_profile):
        # Save profiles to CSV and plot
        self.save_profile("x", x_dens_profile)
        self.save_profile("y", y_dens_profile)
        self.save_profile("z", z_dens_profile)

    def save_profile(self, axis, profile):
        df = pd.DataFrame(
            {
                axis: getattr(self, f"{axis}_grid"),
                "Density [u/Ang^3]": profile,
                "Density [g/cm^3]": profile * 1.66053907,
            }
        )
        df.to_csv(f"{axis}_dens_profile.csv", sep=";", index=False, header=True, float_format="%.5f")

        # Plot profile
        fig, ax = plt.subplots()
        ax.plot(df[axis], df["Density [g/cm^3]"], "-", label="Density profile", color="black")
        ax.set(xlabel=f"{axis} [\u00C5]", ylabel="Density [g/cm\u00B3]", title="Density profile")
        ax.grid()
        fig.savefig(f"{axis}_density_profile.pdf")
