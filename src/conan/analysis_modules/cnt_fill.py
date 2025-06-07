import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

import conan.analysis_modules.traj_an as traj_an
import conan.analysis_modules.traj_info as traj_info
import conan.defdict as ddict


def cnt_loading_mass(traj_file, molecules, an):
    clm = CNTload(traj_file, molecules, an)
    clm.cnt_loading_mass_prep()
    traj_an.process_trajectory(traj_file, molecules, an, clm)
    clm.cnt_loading_mass_processing()


def points_in_cylinder(pt1, pt2, r, atom_positions):
    pt1 = np.asarray(pt1, dtype=np.float64)
    pt2 = np.asarray(pt2, dtype=np.float64)
    atom_positions = np.asarray(atom_positions, dtype=np.float64)

    # set up vector
    vec = pt2 - pt1
    # Normalize axis vector
    vec /= np.linalg.norm(vec)
    # Projection along CNT axis
    proj = np.dot(atom_positions - pt1, vec)

    radial_dist = np.linalg.norm((atom_positions - pt1) - np.outer(proj, vec), axis=1)

    within_cylinder = np.logical_and.reduce((proj >= 0, proj <= np.linalg.norm(pt2 - pt1), radial_dist <= r))

    return within_cylinder


def plot_and_save_results(masses_df, avg_mass, name):
    """
    Plot the results of the confined mass calculation
    """
    fig, ax = plt.subplots()

    ax.plot(masses_df["Frame_masses"], label="Frame", color="#440154")
    ax.plot(masses_df["5_frame_average"], label="5 frame average", color="#21908C")
    ax.plot(masses_df["10_frame_average"], label="10 frame average", color="#5ec962")
    ax.plot(masses_df["50_frame_average"], label="50 frame average", color="#fde725")
    ax.axhline(y=avg_mass, color="darkgray", linestyle="--", label="Average mass")

    ax.set_xlabel("Frame")
    ax.set_ylabel("Mass / u")
    ax.legend()
    ax.grid()

    # save the plot
    fig.savefig(f"{name}.png", dpi=300)
    masses_df.to_csv(f"{name}.csv")


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
        self.shortening = 0.0

    def ring_mean(self, ring):
        ring_x = ring["x"].mean()
        ring_y = ring["y"].mean()
        ring_z = ring["z"].mean()
        ring_array = np.array([ring_x, ring_y, ring_z])

        return ring_array

    def adjust_ring_pbc(self, ring_df, first_atom_coords, box_size, ddict):
        """
        Adjust atoms in a ring that are too far away due to periodic boundary conditions.

        Parameters:
        -----------
        ring_df : pandas.DataFrame
            DataFrame containing atom coordinates
        first_atom_coords : numpy.ndarray
            Coordinates of the reference atom
        box_size : numpy.ndarray
            Box dimensions [x, y, z]
        ddict : module
            Module containing printLog function

        Returns:
        --------
        pandas.DataFrame
            DataFrame with adjusted coordinates
        """
        # Make a copy to avoid modifying the original
        ring = ring_df.copy()

        # Calculate distances
        ring["dist_x"] = np.abs(ring[["x"]].values - first_atom_coords[0])
        ring["dist_y"] = np.abs(ring[["y"]].values - first_atom_coords[1])
        ring["dist_z"] = np.abs(ring[["z"]].values - first_atom_coords[2])
        ring["dist"] = np.sqrt(ring["dist_x"] ** 2 + ring["dist_y"] ** 2 + ring["dist_z"] ** 2)

        iterations = 0
        max_iterations = 10

        while iterations < max_iterations:
            # Find atoms that are too far away in any dimension
            far_atoms = ring[
                (ring["dist_x"] > box_size[0] / 2)
                | (ring["dist_y"] > box_size[1] / 2)
                | (ring["dist_z"] > box_size[2] / 2)
            ]

            if far_atoms.empty:
                break

            for i, atom in far_atoms.iterrows():
                if atom["dist_x"] > box_size[0] / 2:
                    if atom["x"] > first_atom_coords[0]:
                        ring.at[i, "x"] -= box_size[0]
                    else:
                        ring.at[i, "x"] += box_size[0]

                if atom["dist_y"] > box_size[1] / 2:
                    if atom["y"] > first_atom_coords[1]:
                        ring.at[i, "y"] -= box_size[1]
                    else:
                        ring.at[i, "y"] += box_size[1]

                if atom["dist_z"] > box_size[2] / 2:
                    if atom["z"] > first_atom_coords[2]:
                        ring.at[i, "z"] -= box_size[2]
                    else:
                        ring.at[i, "z"] += box_size[2]

            # Recalculate distances after adjustments
            ring["dist_x"] = np.abs(ring[["x"]].values - first_atom_coords[0])
            ring["dist_y"] = np.abs(ring[["y"]].values - first_atom_coords[1])
            ring["dist_z"] = np.abs(ring[["z"]].values - first_atom_coords[2])
            ring["dist"] = np.sqrt(ring["dist_x"] ** 2 + ring["dist_y"] ** 2 + ring["dist_z"] ** 2)

            iterations += 1

        if iterations == max_iterations:
            ddict.printLog("Warning: Maximum iterations reached, some atoms may still be misplaced", color="red")

        return ring

    def cnt_loading_mass_prep(self):
        """
        initializes the variables, the number of CNTs and identifies the most outwards atoms of the CNTs.
        """
        ddict.printLog(
            "\nTo ensure that the program delivers correct results,"
            " it must be ensured that all atoms are wrapped molecule wise in the simulation box"
            " and that the CNT does not cross the PBC boundaries."
            " For pre-processing, please use e.g. Travis.\n",
            color="red",
        )

        pore_atoms = self.traj_file.frame0[self.traj_file.frame0["Struc"].str.contains("Pore")].copy()
        pore_atoms.loc[:, "Struc"] = pore_atoms["Struc"].str.replace("Pore", "").astype(int)
        self.pore_No = pore_atoms["Struc"].nunique()

        box_size = self.traj_file.box_size
        pore_atoms["Bondcount"] = 0

        # identify an atom from the CNT_atoms list which belongs to a given CNT (from 1 to the maximum number of CNTs)
        for i in range(1, self.pore_No + 1):
            atoms = pore_atoms[pore_atoms["Struc"] == i]
            for index, row in atoms.iterrows():
                count = 0
                for j in range(len(self.molecules.molecule_bonds)):
                    for k in range(len(self.molecules.molecule_bonds[j])):
                        if index in self.molecules.molecule_bonds[j][k]:
                            count += 1
                pore_atoms.loc[index, "Bondcount"] = count

        """
        Check which atoms belong to which side of the CNT.
        """
        Catom_positions = np.array(
            [
                [atom["x"], atom["y"], atom["z"], int(atom.name)]
                for index, atom in pore_atoms.iterrows()
                if atom["Bondcount"] == 2
            ]
        )
        # check if there are any atoms with Bondcount == 2
        # if not, the pore has to be infinite through PBC.
        # Ask the user in which direction the CNT is infinite
        if len(Catom_positions) == 0:
            infinite_direction = (
                ddict.get_input(
                    "No C atoms with two carbon bonds found. The CNT is likely infinite through PBC. "
                    "Please specify the direction in which the CNT is infinite (x, y, or z): ",
                    self.traj_file.args,
                    "string",
                )
                .strip()
                .lower()
            )
            while infinite_direction not in ["x", "y", "z"]:
                ddict.printLog(
                    "Invalid direction. Enter the direction (x, y, or z): ", self.traj_file.args, "string"
                ).strip().lower()

            Catom_positions = np.array(
                [[atom["x"], atom["y"], atom["z"], int(atom.name)] for index, atom in pore_atoms.iterrows()]
            )

            G = nx.Graph()
            for i in range(len(pore_atoms)):
                for j in range(i + 1, len(pore_atoms)):
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
            mapping = {i: index for i, index in enumerate(pore_atoms.index)}
            G = nx.relabel_nodes(G, mapping)
            molecules = list(nx.connected_components(G))
            # create a new column in pore_atoms to store the ring number
            pore_atoms["ring"] = 0
            for i in range(len(molecules)):
                for j in molecules[i]:
                    pore_atoms.loc[j, "ring"] = i + 1

            # now just one ring per cnt is needed, not all atoms.
            # By symmetry, looking at just the first atoms of each ring
            # and all other atoms sharing the same coordinate in the infinite_direction
            # should result in one ring of atoms per CNT.
            mol_rings = []
            self.cnt_rings = {}
            self.cnt_data = {}

            for i in range(1, self.pore_No + 1):
                raw_ring = pore_atoms[pore_atoms["Struc"] == i][["x", "y", "z", "ring", "Struc"]].sort_values(by="ring")
                full_ring_adjusted = self.adjust_ring_pbc(
                    raw_ring, raw_ring.iloc[0][["x", "y", "z"]].values, box_size, ddict
                )

                # filter the atoms in the ring by the infinite direction with a threshold of 0.2 angstrom
                if infinite_direction == "x":
                    ring_atoms = full_ring_adjusted[
                        np.abs(full_ring_adjusted["x"] - full_ring_adjusted.iloc[0]["x"]) < 0.2
                    ]
                elif infinite_direction == "y":
                    ring_atoms = full_ring_adjusted[
                        np.abs(full_ring_adjusted["y"] - full_ring_adjusted.iloc[0]["y"]) < 0.2
                    ]
                elif infinite_direction == "z":
                    ring_atoms = full_ring_adjusted[
                        np.abs(full_ring_adjusted["z"] - full_ring_adjusted.iloc[0]["z"]) < 0.2
                    ]

                # now with respect to the first atom,
                # we need to make sure that the mirror image of a given atom is at a minimum distance
                # check if a given distance between the first atom and the other atoms
                # in the ring is larger than half the box size.

                mol_rings.append(ring_atoms.loc[:, ["x", "y", "z", "ring", "Struc"]])

                # For periodic CNTs, we need to create a second virtual ring that's a periodic image
                # Store the ring atoms for further processing
                ring1 = ring_atoms.copy()
                ring2 = ring1.copy()

                # Add the box size in the infinite direction to create a virtual second ring
                if infinite_direction == "x":
                    ring2["x"] = ring2["x"] + box_size[0]
                elif infinite_direction == "y":
                    ring2["y"] = ring2["y"] + box_size[1]
                elif infinite_direction == "z":
                    ring2["z"] = ring2["z"] + box_size[2]

                # Calculate centers-of-mass for each ring and the ring radius
                ring1_array = self.ring_mean(ring1)
                ring2_array = self.ring_mean(ring2)

                ring1_ref = ring1.iloc[0][["x", "y", "z"]].values.astype(float)
                dist_ring = np.linalg.norm(ring1_array - ring1_ref)

                # Calculate the distance between both rings
                pair_dist = np.linalg.norm(ring2_array - ring1_array)

                # Calculate the CNT axis vector
                cnt_axis = (ring2_array - ring1_array) / np.linalg.norm(ring2_array - ring1_array)

                # Store in data structures similar to non-periodic case
                rings_dict = {1: ring1.index.values}
                self.cnt_rings[i] = rings_dict

                # Create and store the pair data
                cnt_pair_data = [
                    {
                        "r1_key": 1,
                        "r2_key": 1,
                        "center_ring1": ring1_array,
                        "center_ring2": ring2_array,
                        "ring_radius": dist_ring,
                        "pair_distance": pair_dist,
                        "cnt_axis": cnt_axis,
                        "is_periodic": True,
                        "periodic_direction": infinite_direction,
                    }
                ]

                self.cnt_data[i] = cnt_pair_data

                ddict.printLog(
                    f"Periodic CNT {i}, Ring radius: {np.round(dist_ring,3)},",
                    " Virtual distance: {np.round(pair_dist,3)}, number of atoms of: {len(ring1)}",
                )

            print(f"Number of CNTs: {len(mol_rings)}")

            self.periodic_cnt_processed = True

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

        if hasattr(self, "periodic_cnt_processed") and self.periodic_cnt_processed:
            # Then skip
            pass
        else:
            mapping = {i: index for i, index in enumerate(Catom_positions[:, 3])}
            G = nx.relabel_nodes(G, mapping)

            molecules = list(nx.connected_components(G))

            pore_atoms["ring"] = 0

            for i in range(len(molecules)):
                for j in molecules[i]:
                    pore_atoms.loc[j, "ring"] = i + 1

            # create a loop over all molecules in pore_atoms and add the ring numbers

            mol_rings = []
            for i in range(1, self.pore_No + 1):
                mol_rings.append(
                    pore_atoms[pore_atoms["Struc"] == i][["x", "y", "z", "ring", "Struc"]]
                    .sort_values(by="ring")
                    .reset_index(drop=True)
                )
            print(f"Number of CNTs: {len(mol_rings)}")

            self.cnt_rings = {}
            for cid in pore_atoms["Struc"].unique():
                rings_in_cnt = pore_atoms[pore_atoms["Struc"] == cid].copy()
                # make sure to only keep the rows with ring numbers > 0
                rings_in_cnt = rings_in_cnt[rings_in_cnt["ring"] > 0]
                unique_rings = sorted(rings_in_cnt["ring"].unique())
                rings_dict = {}
                for r in unique_rings:
                    rings_dict[r] = rings_in_cnt[rings_in_cnt["ring"] == r].index.values
                self.cnt_rings[cid] = rings_dict

            # Process non-periodic CNT data
            self.cnt_data = {}  # Store calculated data per CNT and ring pair
            for cid, rings_dict in self.cnt_rings.items():
                ring_keys = sorted(rings_dict.keys())
                cnt_pair_data = []
                # Loop over adjacent ring pairs
                for i in range(len(ring_keys) - 1):

                    r1_key = ring_keys[i]
                    r2_key = ring_keys[i + 1]
                    # Retrieve atoms for the two rings from frame0
                    ring1 = self.traj_file.frame0.loc[rings_dict[r1_key]]
                    ring2 = self.traj_file.frame0.loc[rings_dict[r2_key]]
                    # Get the first atom's coordinates for each ring
                    first_atom_ring1 = ring1.iloc[0][["x", "y", "z"]].values
                    first_atom_ring2 = ring2.iloc[0][["x", "y", "z"]].values

                    # Ensure the coordinates are numpy arrays for vectorized operations
                    first_atom_ring1 = np.array(first_atom_ring1, dtype=float)
                    first_atom_ring2 = np.array(first_atom_ring2, dtype=float)

                    # Adjust rings to account for periodic boundary conditions
                    ring1 = self.adjust_ring_pbc(ring1, first_atom_ring1, box_size, ddict)
                    ring2 = self.adjust_ring_pbc(ring2, first_atom_ring2, box_size, ddict)

                    ring1_array = self.ring_mean(ring1)
                    ring2_array = self.ring_mean(ring2)

                    # Calculate ring radius as the norm of the deviation of ring1 atoms from its center
                    ring1_ref = ring1.iloc[0][["x", "y", "z"]].values.astype(float)
                    dist_ring = np.linalg.norm(ring1_array - ring1_ref)

                    # Calculate the distance between both rings
                    pair_dist = np.linalg.norm(ring2_array - ring1_array)

                    # Calculate the CNT axis vector
                    cnt_axis = (ring2_array - ring1_array) / np.linalg.norm(ring2_array - ring1_array)
                    cnt_pair_data.append(
                        {
                            "r1_key": r1_key,
                            "r2_key": r2_key,
                            "center_ring1": ring1_array,
                            "center_ring2": ring2_array,
                            "ring_radius": dist_ring,
                            "pair_distance": pair_dist,
                            "cnt_axis": cnt_axis,
                        }
                    )
                    ddict.printLog(
                        f"CNT {cid}, Ring pair ({r1_key}:{r2_key}) => radius: {np.round(dist_ring,3)}",
                        ", distance: {np.round(pair_dist,3)}",
                    )
                self.cnt_data[cid] = cnt_pair_data

        self.shortening_q = ddict.get_input(
            "Do you want to shorten the CNT axis (space to analyze within the CNT)? (y/n): ",
            self.traj_file.args,
            "string",
        )
        if self.shortening_q == "y":
            self.shortening = float(
                ddict.get_input(
                    "Please enter the amount you want the axis to be shortened (from both sides): ",
                    self.traj_file.args,
                    "float",
                )
            )
            # Update each CNT pair center based on the shortening input
            for cid, pair_list in self.cnt_data.items():
                print(f"Shortening the CNT axis for CNT {cid} by {self.shortening} angstroms on each side.")
                for pair in pair_list:
                    pair["center_ring1"] = pair["center_ring1"] + self.shortening * pair["cnt_axis"]
                    pair["center_ring2"] = pair["center_ring2"] - self.shortening * pair["cnt_axis"]
                    pair["pair_distance"] = np.linalg.norm(pair["center_ring2"] - pair["center_ring1"])
                    pair["cnt_axis"] = (pair["center_ring2"] - pair["center_ring1"]) / np.linalg.norm(
                        pair["center_ring2"] - pair["center_ring1"]
                    )

        # initialize the lists to store the results for each CNT
        self.liquid_mass = 0
        self.frame_masses = np.array([])
        self.ring_ring_distances = np.array([])
        self.ring_radii = np.array([])

    def analyze_frame(self, split_frame, frame_counter):
        """
        Calculate the loading mass of the liquid within the CNTs for all CNTs in the system.
        Uses the CNT data prepared in cnt_loading_mass_prep, accounting for PBC.
        """
        # Convert coordinate columns to float
        split_frame["X"] = split_frame["X"].astype(float)
        split_frame["Y"] = split_frame["Y"].astype(float)
        split_frame["Z"] = split_frame["Z"].astype(float)

        # Get box size for PBC handling
        box_size = self.traj_file.box_size

        # Identify liquid atoms once for all CNTs
        liquid_atoms = split_frame[split_frame["Struc"].str.contains("Liquid")]

        # Get the atom positions of the current frame
        atom_positions = np.array(
            [[atom["X"], atom["Y"], atom["Z"], atom["Mass"], atom.index] for index, atom in liquid_atoms.iterrows()],
            dtype=object,
        )

        total_frame_mass = 0

        # Process each CNT
        for cnt_id, pair_list in self.cnt_data.items():
            # Process each ring pair in this CNT
            for pair_idx, pair_data in enumerate(pair_list):

                r1_key = pair_data["r1_key"]
                r2_key = pair_data["r2_key"]

                is_periodic = pair_data.get("is_periodic", False)

                # Get coordinates from this frame
                ring1 = split_frame.loc[self.cnt_rings[cnt_id][r1_key]].copy()

                # Handle periodic CNTs differently
                if is_periodic:
                    # For periodic CNTs, create a virtual second ring that's a periodic image
                    ring2 = ring1.copy()
                    periodic_direction = pair_data.get("periodic_direction", "z")

                    # Add the box size in the infinite direction to create a virtual second ring
                    if periodic_direction == "x":
                        ring2["X"] += ring2["X"]
                    elif periodic_direction == "y":
                        ring2["Y"] += ring2["Y"]
                    elif periodic_direction == "z":
                        ring2["Z"] += box_size[2]
                else:
                    ring2 = split_frame.loc[self.cnt_rings[cnt_id][r2_key]].copy()

                # Convert column names from X,Y,Z to x,y,z for consistency with adjust_ring_pbc
                ring1 = ring1.rename(columns={"X": "x", "Y": "y", "Z": "z"})
                ring2 = ring2.rename(columns={"X": "x", "Y": "y", "Z": "z"})

                # Get the first atom's coordinates for each ring to use as reference
                first_atom_ring1 = ring1.iloc[0][["x", "y", "z"]].values.astype(float)
                first_atom_ring2 = ring2.iloc[0][["x", "y", "z"]].values.astype(float)

                # Adjust rings to account for periodic boundary conditions
                ring1_adjusted = self.adjust_ring_pbc(ring1, first_atom_ring1, box_size, ddict)
                ring2_adjusted = self.adjust_ring_pbc(ring2, first_atom_ring2, box_size, ddict)

                # Calculate centers of rings using the adjusted coordinates
                ring1_array = self.ring_mean(ring1_adjusted)
                ring2_array = self.ring_mean(ring2_adjusted)

                # Calculate radius using the first ring's adjusted coordinates
                ring_radii = []
                for index, row in ring1_adjusted.iterrows():
                    ring1_ref = row[["x", "y", "z"]].values.astype(float)
                    dist_ring = np.linalg.norm(ring1_array - ring1_ref)
                    ring_radii.append(dist_ring)

                dist_ring = np.mean(ring_radii)

                # Store ring radii per CNT and ring pair
                if not hasattr(self, "cnt_ring_radii"):
                    self.cnt_ring_radii = {}

                if cnt_id not in self.cnt_ring_radii:
                    self.cnt_ring_radii[cnt_id] = {}

                radius_key = f"{r1_key}"
                if radius_key not in self.cnt_ring_radii[cnt_id]:
                    self.cnt_ring_radii[cnt_id][radius_key] = np.array([])

                self.cnt_ring_radii[cnt_id][radius_key] = np.append(self.cnt_ring_radii[cnt_id][radius_key], dist_ring)

                self.ring_radii = np.append(self.ring_radii, dist_ring)

                # Calculate CNT axis
                cnt_axis = (ring2_array - ring1_array) / np.linalg.norm(ring2_array - ring1_array)

                # Apply shortening
                if hasattr(self, "shortening") and self.shortening > 0:
                    ring1_array = ring1_array + self.shortening * cnt_axis
                    ring2_array = ring2_array - self.shortening * cnt_axis

                # Calculate distance between rings
                ring_distance = np.linalg.norm(ring2_array - ring1_array)

                # Store ring distances per CNT and ring pair
                if not hasattr(self, "cnt_ring_distances"):
                    self.cnt_ring_distances = {}

                if cnt_id not in self.cnt_ring_distances:
                    self.cnt_ring_distances[cnt_id] = {}

                distance_key = f"{r1_key}_{r2_key}"
                if distance_key not in self.cnt_ring_distances[cnt_id]:
                    self.cnt_ring_distances[cnt_id][distance_key] = np.array([])

                self.cnt_ring_distances[cnt_id][distance_key] = np.append(
                    self.cnt_ring_distances[cnt_id][distance_key], ring_distance
                )

                # store the ring distance
                self.ring_ring_distances = np.append(self.ring_ring_distances, ring_distance)

                atom_xyz = atom_positions[:, :3].astype(float)

                M = 0.5 * (ring1_array + ring2_array)
                half = 0.5 * np.linalg.norm(ring2_array - ring1_array)

                delta = atom_xyz - M

                delta[:, 0] -= box_size[0] * np.round(delta[:, 0] / box_size[0])
                delta[:, 1] -= box_size[1] * np.round(delta[:, 1] / box_size[1])
                delta[:, 2] -= box_size[2] * np.round(delta[:, 2] / box_size[2])

                proj = np.dot(delta, cnt_axis)

                # radial distance
                perp = delta - np.outer(proj, cnt_axis)
                radial = np.linalg.norm(perp, axis=1)

                # inside‐cylinder mask: |proj| <= half-length, radial <= radius
                inside_cylinder = (np.abs(proj) <= half) & (radial <= dist_ring)

                # Calculate total mass of atoms inside the cylinder
                section_mass = atom_positions[inside_cylinder, 3].sum()
                total_frame_mass += section_mass

                # Store data per CNT if needed
                if not hasattr(self, "cnt_frame_masses"):
                    self.cnt_frame_masses = {}

                if cnt_id not in self.cnt_frame_masses:
                    self.cnt_frame_masses[cnt_id] = {}

                pair_key = f"{r1_key}_{r2_key}"
                if pair_key not in self.cnt_frame_masses[cnt_id]:
                    self.cnt_frame_masses[cnt_id][pair_key] = np.array([])

                self.cnt_frame_masses[cnt_id][pair_key] = np.append(
                    self.cnt_frame_masses[cnt_id][pair_key], section_mass
                )

        # Update total mass
        self.liquid_mass += total_frame_mass

        # Store total mass for this frame
        self.frame_masses = np.append(self.frame_masses, total_frame_mass)

        # Increment frame counter
        self.proc_frame_counter += 1

    def cnt_loading_mass_processing(self):
        """
        Calculate and process the loading mass of the liquid within the CNTs for all CNTs in the system.
        Generates statistics and visualizations for each individual CNT as well as aggregate data.
        """
        # Calculate total system statistics
        self.avg_mass_per_frame = self.liquid_mass / self.proc_frame_counter

        # Calculate total distance across all CNTs for system average
        total_distance = 0
        for cnt_id, cnt_dist_data in self.cnt_ring_distances.items():
            cnt_distance = 0
            for key, distances in cnt_dist_data.items():
                cnt_distance += np.mean(distances)
            total_distance += cnt_distance

        self.dist = total_distance
        self.avg_mass_per_angstrom = self.avg_mass_per_frame / self.dist if self.dist > 0 else 0

        ddict.printLog("Overall system summary:")
        ddict.printLog(f"Total average confined mass: {np.round(self.avg_mass_per_frame, 3)} u")
        ddict.printLog(f"Total average mass per \u00c5: {np.round(self.avg_mass_per_angstrom, 3)} u/\u00c5")

        # Process aggregate data
        pd_frame_masses = pd.DataFrame(self.frame_masses)
        pd_frame_masses.columns = ["Frame_masses"]

        # Check if all CNTs are empty
        if np.all(pd_frame_masses["Frame_masses"] == 0):
            ddict.printLog("All CNTs are empty, no liquid is present.", color="red")
            return

        # Average of 5, 10 and 50 frames for aggregate data
        pd_frame_masses["5_frame_average"] = pd_frame_masses["Frame_masses"].rolling(window=5).mean().shift(-2)
        pd_frame_masses["10_frame_average"] = pd_frame_masses["Frame_masses"].rolling(window=10).mean().shift(-5)
        pd_frame_masses["50_frame_average"] = pd_frame_masses["Frame_masses"].rolling(window=50).mean().shift(-25)

        pd_mass_per_angstrom = pd_frame_masses.div(self.dist, axis=0).copy()

        # Save aggregate results
        plot_and_save_results(pd_frame_masses, self.avg_mass_per_frame, "frame_masses_total")
        plot_and_save_results(pd_mass_per_angstrom, self.avg_mass_per_angstrom, "mass_per_angstrom_total")
        pd_frame_masses.to_csv("frame_masses_total.csv")
        pd_mass_per_angstrom.to_csv("mass_per_angstrom_total.csv")

        # Process each individual CNT
        for cnt_id in self.cnt_frame_masses.keys():
            ddict.printLog(f"\nProcessing CNT {cnt_id}:", color="green")

            # Calculate total mass and distance for this CNT
            cnt_masses = []
            cnt_distance = 0

            # Get all masses for this CNT across all frame pairs
            for pair_key, masses in self.cnt_frame_masses[cnt_id].items():
                cnt_masses.extend(masses)

            # Calculate total CNT length from ring distances
            for distance_key, distances in self.cnt_ring_distances[cnt_id].items():
                cnt_distance += np.mean(distances)

            # Calculate average radius for this CNT
            cnt_radii = []
            for radius_key, radii in self.cnt_ring_radii[cnt_id].items():
                cnt_radii.extend(radii)
            mean_cnt_radius = np.mean(cnt_radii)

            # Create per-frame masses for this CNT
            frames_per_cnt = len(self.frame_masses)
            cnt_frame_masses = np.zeros(frames_per_cnt)

            # Aggregate all section masses for this CNT per frame
            section_count = len(next(iter(self.cnt_frame_masses[cnt_id].values())))
            for frame_idx in range(section_count):
                frame_mass = 0
                for pair_key, masses in self.cnt_frame_masses[cnt_id].items():
                    if frame_idx < len(masses):
                        frame_mass += masses[frame_idx]
                if frame_idx < len(cnt_frame_masses):
                    cnt_frame_masses[frame_idx] = frame_mass

            # Calculate statistics for this CNT
            avg_cnt_mass = np.mean(cnt_frame_masses)
            avg_cnt_mass_per_angstrom = avg_cnt_mass / cnt_distance if cnt_distance > 0 else 0

            ddict.printLog(f"CNT {cnt_id} - Average mass: {np.round(avg_cnt_mass, 3)} u")
            ddict.printLog(f"CNT {cnt_id} - Average mass per \u00c5: {np.round(avg_cnt_mass_per_angstrom, 3)} u/\u00c5")
            ddict.printLog(f"CNT {cnt_id} - Mean radius: {np.round(mean_cnt_radius, 3)} \u00c5")
            ddict.printLog(f"CNT {cnt_id} - Total length: {np.round(cnt_distance, 3)} \u00c5")

            # Create DataFrame for this CNT
            pd_cnt_masses = pd.DataFrame(cnt_frame_masses)
            pd_cnt_masses.columns = ["Frame_masses"]

            # Check if this specific CNT is empty
            if np.all(pd_cnt_masses["Frame_masses"] == 0):
                ddict.printLog(f"CNT {cnt_id} is empty, no liquid is present.", color="yellow")
                continue

            # Calculate rolling averages
            pd_cnt_masses["5_frame_average"] = pd_cnt_masses["Frame_masses"].rolling(window=5).mean().shift(-2)
            pd_cnt_masses["10_frame_average"] = pd_cnt_masses["Frame_masses"].rolling(window=10).mean().shift(-5)
            pd_cnt_masses["50_frame_average"] = pd_cnt_masses["Frame_masses"].rolling(window=50).mean().shift(-25)

            # Calculate per angstrom values
            pd_cnt_mass_per_angstrom = pd_cnt_masses.div(cnt_distance, axis=0).copy()

            # Save results for this CNT
            plot_and_save_results(pd_cnt_masses, avg_cnt_mass, f"frame_masses_cnt_{cnt_id}")
            plot_and_save_results(
                pd_cnt_mass_per_angstrom, avg_cnt_mass_per_angstrom, f"mass_per_angstrom_cnt_{cnt_id}"
            )
            pd_cnt_masses.to_csv(f"frame_masses_cnt_{cnt_id}.csv")
            pd_cnt_mass_per_angstrom.to_csv(f"mass_per_angstrom_cnt_{cnt_id}.csv")

            # Save ring distances and radii for this CNT
            cnt_ring_distances = []
            for key, distances in self.cnt_ring_distances[cnt_id].items():
                cnt_ring_distances.extend(distances)

            pd_cnt_ring_distances = pd.DataFrame(cnt_ring_distances)
            pd_cnt_ring_distances.columns = ["Ring_ring_distances"]
            pd_cnt_ring_distances.to_csv(f"ring_ring_distances_cnt_{cnt_id}.csv")

            pd_cnt_ring_radii = pd.DataFrame(cnt_radii)
            pd_cnt_ring_radii.columns = ["Ring_radii"]
            pd_cnt_ring_radii.to_csv(f"ring_radii_cnt_{cnt_id}.csv")

        # Still save the original aggregate data for backward compatibility
        pd_ring_ring_distances = pd.DataFrame(self.ring_ring_distances)
        pd_ring_ring_distances.columns = ["Ring_ring_distances"]
        mean_ring_ring_distance = pd_ring_ring_distances["Ring_ring_distances"].mean()
        ddict.printLog(f"Mean lengths of the CNTs: {np.round(mean_ring_ring_distance), 3} \u00c5")
        pd_ring_ring_distances.to_csv("ring_ring_distances.csv")

        pd_ring_radii = pd.DataFrame(self.ring_radii)
        pd_ring_radii.columns = ["Ring_radii"]
        mean_ring_radii = pd_ring_radii["Ring_radii"].mean()
        ddict.printLog(f"Mean radius of the CNTs: {np.round(mean_ring_radii), 3} \u00c5")
        pd_ring_radii.to_csv("ring_radii.csv")
