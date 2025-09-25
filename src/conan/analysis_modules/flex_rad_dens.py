import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import conan.analysis_modules.cnt_fill as cf
import conan.analysis_modules.traj_an as traj_an
import conan.defdict as ddict


def flex_rad_dens(traj_file, molecules, an):
    flex_rd = FlexRadDens(traj_file, molecules, an)
    flex_rd.flex_rad_dens_prep()
    traj_an.process_trajectory(traj_file, molecules, an, flex_rd)
    flex_rd.flex_rad_dens_processing()


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
        self.proc_frame_counter = 0

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

    def flex_rad_dens_prep(self):
        """
        Prepare the flexible radial density analysis.
        For this, things are needed:
        - Ask the user how many increments the CNT should be radially divided into.
        - Let the user decide if the full length of the CNT should be subject to this analysis
        (to avoid opening effects if wanted).
          - For this one can use use the setup of the loading mass module.

        """

        # run the cnt_loading_mass_prep function in the CNTload class in cf
        cnt_load = cf.CNTload(self.traj_file, self.molecules, self.an)
        cnt_load.cnt_loading_mass_prep()

        # store the attributes of the cnt_load object in the current object
        self.cnt_data = cnt_load.cnt_data
        self.cnt_rings = cnt_load.cnt_rings

        # Iterate over the CNTs to set up incrementation / radial density bins
        cnts_bin_edges = {}
        cnts_bin_labels = {}
        cnts_dataframes = {}
        self.num_increments = {}
        same_max_distance_q = ddict.get_input(
            "Do you want to use the same maximum displacement for all CNTs? (y/n) ",
            self.traj_file.args,
            "string",
        )

        if same_max_distance_q.lower() == "y":
            max_distance = ddict.get_input("Maximum displacement [Å]: ", self.traj_file.args, "float")

        for cnt_id, pair_list in cnt_load.cnt_data.items():
            if same_max_distance_q.lower() == "y":
                tuberadius = max_distance
            else:
                tuberadius = pair_list[0]["ring_radius"]
            ddict.printLog(f"\n-> CNT {cnt_id} with radius {tuberadius:.2f} Å")
            self.num_increments[cnt_id] = int(
                ddict.get_input(
                    f"({cnt_id}) How many increments do you want to use? ",
                    self.traj_file.args,
                    "int",
                )
            )
            rad_increment = tuberadius / self.num_increments[cnt_id]

            cnts_bin_edges[cnt_id] = np.linspace(0, tuberadius, self.num_increments[cnt_id] + 1)
            cnts_bin_labels[cnt_id] = np.arange(1, len(cnts_bin_edges[cnt_id]), 1)
            ddict.printLog("Increment distance: %0.3f  \u00c5" % (rad_increment))

            data = {"Frame": np.arange(1, self.traj_file.number_of_frames + 1)}
            # Add columns for each bin
            for i in range(self.num_increments[cnt_id]):
                data["Bin %d" % (i + 1)] = np.zeros(self.traj_file.number_of_frames, dtype=float)
            cnts_dataframes[cnt_id] = pd.DataFrame(data)

        self.cnts_bin_edges = cnts_bin_edges
        self.cnts_bin_labels = cnts_bin_labels
        self.cnts_dataframes = cnts_dataframes

    def analyze_frame(self, split_frame, frame_counter):
        """
        Calculate the radial density distribution of molecules inside CNTs.
        Based on cnt_fill.py's analyze_frame but adds radial binning.
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

        # Process each CNT
        for cnt_id, pair_list in self.cnt_data.items():
            # Get the bins for this CNT
            bin_edges = self.cnts_bin_edges[cnt_id]

            # Lists to store atoms that fall within this CNT and their radial distances
            inside_atoms_masses = []
            radial_distances = []

            # Process each ring pair in this CNT
            for pair_idx, pair_data in enumerate(pair_list):
                # Get ring identifiers
                r1_key = pair_data["r1_key"]
                r2_key = pair_data["r2_key"]

                is_periodic = pair_data.get("is_periodic", False)

                ring1 = split_frame.loc[self.cnt_rings[cnt_id][r1_key]].copy()

                # periodic CNTs
                if is_periodic:
                    # create a virtual second ring that's a periodic image
                    ring2 = ring1.copy()
                    periodic_direction = pair_data.get("periodic_direction", "z")

                    # Add the box size in the infinite direction to create a virtual second ring
                    if periodic_direction == "x":
                        ring2["X"] = ring2["X"] + box_size[0]
                    elif periodic_direction == "y":
                        ring2["Y"] = ring2["Y"] + box_size[1]
                    elif periodic_direction == "z":
                        ring2["Z"] = ring2["Z"] + box_size[2]
                else:
                    ring2 = split_frame.loc[self.cnt_rings[cnt_id][r2_key]].copy()

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

                # Calculate CNT axis using adjusted ring centers
                cnt_axis = (ring2_array - ring1_array) / np.linalg.norm(ring2_array - ring1_array)

                # Apply shortening if requested
                if hasattr(self, "shortening") and self.shortening > 0:
                    ring1_array = ring1_array + self.shortening * cnt_axis
                    ring2_array = ring2_array - self.shortening * cnt_axis

                atom_xyz = atom_positions[:, :3].astype(float)

                # Compute midpoint M and half‐length
                M = 0.5 * (ring1_array + ring2_array)
                half = 0.5 * np.linalg.norm(ring2_array - ring1_array)

                delta = atom_xyz - M

                delta[:, 0] -= box_size[0] * np.round(delta[:, 0] / box_size[0])
                delta[:, 1] -= box_size[1] * np.round(delta[:, 1] / box_size[1])
                delta[:, 2] -= box_size[2] * np.round(delta[:, 2] / box_size[2])

                # projec back onto the axis
                proj = np.dot(delta, cnt_axis)

                # compute the radial distances
                radial_vecs = delta - np.outer(proj, cnt_axis)
                radial = np.linalg.norm(radial_vecs, axis=1)

                inside_cylinder = (np.abs(proj) <= half) & (radial <= dist_ring)

                # finally get the masses & radial distances
                inside_atoms_masses = atom_positions[inside_cylinder, 3].astype(float)
                radial_distances = radial[inside_cylinder]

                if radial_distances.size > 0:
                    counts, _ = np.histogram(radial_distances, bins=bin_edges, weights=inside_atoms_masses)
                    for b in range(self.num_increments[cnt_id]):
                        self.cnts_dataframes[cnt_id].iat[frame_counter - 1, b + 1] = float(counts[b])

        self.proc_frame_counter += 1

    def flex_rad_dens_processing(self):
        """
        Process the radial density data for each CNT after the frame analysis is complete.
        Calculates densities, statistics, and generates visualizations.
        """
        ddict.printLog("\nPost-processing radial density data...")
        ddict.printLog(f"Processed {self.proc_frame_counter} frames in total.")
        # Extract relevant attributes from self
        cnts_dataframes = self.cnts_dataframes
        cnts_bin_edges = self.cnts_bin_edges

        # Process each CNT separately
        for cnt_id, dataframe in cnts_dataframes.items():
            ddict.printLog(f"\nProcessing CNT {cnt_id}...")

            # Get bin edges for this CNT
            bin_edges = cnts_bin_edges[cnt_id]

            # Get CNT properties
            tube_radius = self.cnt_data[cnt_id][0]["ring_radius"]

            # Calculate total CNT length for volume calculation
            cnt_length = 0
            cnt_length = self.cnt_data[cnt_id][0]["pair_distance"]
            print(f"Total CNT length for {cnt_id}: {cnt_length:.2f} Å")

            # Create a results dataframe with average masses per bin
            results_df = pd.DataFrame(dataframe.iloc[:, 1:].sum(axis=0) / self.proc_frame_counter)
            results_df.columns = ["Mass"]

            # Add bin information
            results_df["Bin_lowedge"] = bin_edges[:-1]
            results_df["Bin_highedge"] = bin_edges[1:]
            results_df["Bin_center"] = (bin_edges[1:] + bin_edges[:-1]) / 2

            # Calculate cylindrical shell volumes for each bin
            vol_increment = np.pi * (bin_edges[1:] ** 2 - bin_edges[:-1] ** 2) * cnt_length
            results_df["Volume"] = vol_increment

            # Calculate densities
            results_df["Density [u/Ang^3]"] = results_df["Mass"] / results_df["Volume"]
            results_df["Density [g/cm^3]"] = results_df["Density [u/Ang^3]"] * 1.66053907

            # Reset index and add bin numbers
            results_df.reset_index(drop=True, inplace=True)
            results_df.insert(0, "Bin", results_df.index + 1)

            # Calculate per-frame density statistics
            density_df = pd.DataFrame()
            for i in range(len(results_df)):
                density_df[i] = dataframe.iloc[:, i + 1] / results_df.loc[i, "Volume"]

            # Add statistics
            results_df["Variance"] = density_df.var(axis=0).values
            results_df["Standard dev."] = density_df.std(axis=0).values
            results_df["Standard error"] = density_df.sem(axis=0).values

            # Add frame numbers to density dataframe
            density_df.columns = dataframe.columns[1:]
            density_df.insert(0, "Frame", dataframe["Frame"])

            # Save raw data files
            dataframe.to_csv(f"CNT_{cnt_id}_radial_mass_dist_raw.csv", sep=";", index=False, float_format="%.5f")
            density_df.to_csv(f"CNT_{cnt_id}_radial_density_raw.csv", sep=";", index=False, float_format="%.5f")

            # plot the data
            plot_data = ddict.get_input(
                f"Do you want to plot the data for CNT {cnt_id}? (y/n) ", self.traj_file.args, "string"
            )

            if plot_data.lower() == "y":
                results_df_copy = results_df.copy()

                normalize = ddict.get_input(
                    "Do you want to normalize the increments with respect to the CNT's radius? (y/n) ",
                    self.traj_file.args,
                    "string",
                )

                if normalize.lower() == "y":
                    results_df["Bin_center"] = results_df["Bin_center"] / tube_radius

                mirror = ddict.get_input("Do you want to mirror the plot? (y/n) ", self.traj_file.args, "string")

                if mirror.lower() == "y":
                    results_df_mirror = results_df.copy()
                    results_df_mirror["Bin_center"] = results_df["Bin_center"] * (-1)
                    results_df_mirror.sort_values(by=["Bin_center"], inplace=True)
                    results_df = pd.concat([results_df_mirror, results_df], ignore_index=True)

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(
                    results_df["Bin_center"],
                    results_df["Density [g/cm^3]"],
                    "-",
                    label="Radial density function",
                    color="black",
                )

                if mirror.lower() != "y":
                    upper_bound = results_df["Density [g/cm^3]"] + results_df["Standard dev."]
                    lower_bound = results_df["Density [g/cm^3]"] - results_df["Standard dev."]
                    ax.fill_between(
                        results_df["Bin_center"],
                        lower_bound,
                        upper_bound,
                        alpha=0.2,
                        color="gray",
                        label="Standard deviation",
                    )

                # Set axis labels and title
                x_label = (
                    "Distance from tube center [normalized]"
                    if normalize.lower() == "y"
                    else "Distance from tube center [Å]"
                )
                ax.set(xlabel=x_label, ylabel="Density [g/cm³]", title=f"Radial Density Profile - CNT {cnt_id}")

                ax.grid(True, linestyle="--", alpha=0.7)
                ax.legend()

                # Save figure
                filename = f"CNT_{cnt_id}_radial_density_function.png"
                fig.savefig(filename, dpi=300, bbox_inches="tight")
                ddict.printLog(f"-> Radial density function saved as {filename}")

                results_df.to_csv(
                    f"CNT_{cnt_id}_radial_density_function.csv", sep=";", index=False, float_format="%.5f"
                )

                # Create radial contour plot option
                results_df = results_df_copy.copy()
                radial_plot = ddict.get_input(
                    "Do you also want to create a radial contour plot? (y/n) ", self.traj_file.args, "string"
                )

                if radial_plot.lower() == "y":
                    theta = np.linspace(0, 2 * np.pi, 500)

                    if normalize.lower() == "y":
                        r = np.linspace(0, 1, len(results_df["Bin_center"]))
                    else:
                        r = np.linspace(0, tube_radius, len(results_df["Bin_center"]))

                    Theta, R = np.meshgrid(theta, r)
                    values = np.tile(results_df["Density [g/cm^3]"].values, (len(theta), 1)).T

                    fig = plt.figure(figsize=(10, 8))
                    ax = fig.add_subplot(projection="polar")
                    c = ax.contourf(Theta, R, values, cmap="viridis")

                    ax.spines["polar"].set_visible(False)
                    ax.set_xticklabels([])

                    # Set radial gridlines and labels
                    ax.set_yticks(np.linspace(0, tube_radius if normalize.lower() == "n" else 1, 5))
                    ax.set_yticklabels(
                        ["{:.2f}".format(x) for x in np.linspace(0, tube_radius if normalize.lower() == "n" else 1, 5)]
                    )

                    ax.set_rlabel_position(22.5)
                    ax.grid(color="black", linestyle="--", alpha=0.5)

                    plt.title(f"Radial Density Contour Plot - CNT {cnt_id}", fontsize=16, pad=20)
                    cbar = fig.colorbar(c, ax=ax, pad=0.10, fraction=0.046, orientation="horizontal")
                    cbar.set_ticklabels(["{:.2f}".format(x) for x in cbar.get_ticks()])
                    cbar.set_label(r"Mass density $[g/cm^{3}]$", fontsize=12)

                    if normalize.lower() == "n":
                        ax.set_ylabel(r"$d_{rad}$ [Å]", labelpad=10, fontsize=12)
                    else:
                        ax.set_ylabel(r"$d_{rad}/r_{CNT}$", labelpad=10, fontsize=12)

                    filename = f"CNT_{cnt_id}_radial_density_contour.png"
                    fig.savefig(filename, dpi=300, bbox_inches="tight")
                    ddict.printLog(f"-> Radial density contour plot saved as {filename}")

        # Create a combined plot for all CNTs if there are multiple
        if len(cnts_dataframes) > 1:
            combined_plot = ddict.get_input(
                "Do you want to create a combined plot comparing all CNTs? (y/n) ", self.traj_file.args, "string"
            )

            if combined_plot.lower() == "y":
                fig, ax = plt.subplots(figsize=(12, 7))

                # Plot each CNT with a different color/style
                colors = plt.cm.viridis(np.linspace(0, 0.9, len(cnts_dataframes)))

                for idx, (cnt_id, dataframe) in enumerate(cnts_dataframes.items()):
                    # Get bin edges and tube radius for this CNT
                    bin_edges = cnts_bin_edges[cnt_id]
                    tube_radius = self.cnt_data[cnt_id][0]["ring_radius"]

                    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

                    cnt_length = 0
                    for pair_data in self.cnt_data[cnt_id]:
                        cnt_length += pair_data["pair_distance"]

                    # Calculate density
                    avg_masses = dataframe.iloc[:, 1:].sum(axis=0) / self.proc_frame_counter
                    vol_increments = np.pi * (bin_edges[1:] ** 2 - bin_edges[:-1] ** 2) * cnt_length
                    densities = avg_masses / vol_increments * 1.66053907  # Convert to g/cm³

                    # Plot this CNT's density profile
                    ax.plot(
                        bin_centers,
                        densities,
                        "-",
                        color=colors[idx],
                        linewidth=2,
                        label=f"CNT {cnt_id} (r={tube_radius:.2f}Å)",
                    )

                x_label = "Distance from tube center [Å]"
                ax.set(
                    xlabel=x_label, ylabel="Density [g/cm³]", title="Comparison of Radial Density Profiles Across CNTs"
                )

                ax.grid(True, linestyle="--", alpha=0.7)
                ax.legend()

                fig.savefig("CNT_comparison_radial_density.png", dpi=300, bbox_inches="tight")
                ddict.printLog("-> Comparison plot saved as CNT_comparison_radial_density.png")
