import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import conan.analysis_modules.cnt_fill as cf
import conan.analysis_modules.traj_an as traj_an
import conan.defdict as ddict


def resid_time_analysis(traj_file, molecules, an):
    resid_t = ResidTime(traj_file, molecules, an)
    resid_t.resid_time_prep()
    traj_an.process_trajectory(traj_file, molecules, an, resid_t)
    resid_t.resid_time_processing()


class ResidTime:
    """
    Residence time analysis for molecules in CNT layers.
    Calculate the residence time of a given molecule in a radial layer within a CNT.
    The radial layers are set by the user.
    """

    def __init__(self, traj_file, molecules, an):
        self.traj_file = traj_file
        self.molecules = molecules
        self.an = an
        self.proc_frame_counter = 0
        # Maximum transient departure time
        self.t_star = 2.0
        # Maximum correlation time
        self.max_correlation_time = 50.0
        # Dictionary to store frame-by-frame molecule layer data
        self.molecule_frame_data = {}
        # Dictionary to store molecule count per layer per frame
        self.layer_population_data = {}
        # Dict to store correlation function results
        self.correlation_functions = {}

    def resid_time_prep(self):
        """
        Preparation of the residence time analysis.
        """
        cnt_load = cf.CNTload(self.traj_file, self.molecules, self.an)
        cnt_load.cnt_loading_mass_prep()

        # store the attributes of the cnt_load object in the current object
        self.cnt_data = cnt_load.cnt_data
        self.cnt_rings = cnt_load.cnt_rings

        # Set up radial layers for each CNT
        self.cnts_layer_edges = {}
        self.cnts_layer_labels = {}

        ddict.printLog("\n=== Residence Time Analysis Setup ===")

        # Ask if user wants same layer setup for all CNTs
        same_layers_q = ddict.get_input(
            "Do you want to use the same radial layers for all CNTs? (y/n) ",
            self.traj_file.args,
            "string",
        )

        if same_layers_q.lower() == "y":
            # Get layer setup once for all CNTs
            layer_edges = self._get_layer_setup("all CNTs")
            for cnt_id in self.cnt_data.keys():
                self.cnts_layer_edges[cnt_id] = layer_edges
                self.cnts_layer_labels[cnt_id] = [f"{i + 1}" for i in range(len(layer_edges) - 1)]
                self.correlation_functions[cnt_id] = {label: {} for label in self.cnts_layer_labels[cnt_id]}
                self.molecule_frame_data[cnt_id] = {}
                self.layer_population_data[cnt_id] = {}
        else:
            # Get layer setup for each CNT individually
            for cnt_id, pair_list in self.cnt_data.items():
                tube_radius = pair_list[0]["ring_radius"]
                ddict.printLog(f"\n-> CNT {cnt_id} with radius {tube_radius:.2f} Å")
                layer_edges = self._get_layer_setup(f"CNT {cnt_id}")
                self.cnts_layer_edges[cnt_id] = layer_edges
                self.cnts_layer_labels[cnt_id] = [f"{i + 1}" for i in range(len(layer_edges) - 1)]
                self.correlation_functions[cnt_id] = {label: {} for label in self.cnts_layer_labels[cnt_id]}
                self.molecule_frame_data[cnt_id] = {}
                self.layer_population_data[cnt_id] = {}

        # Get time step between frames
        self.time_step = ddict.get_input(
            "Time step between consecutive frames [ps]: ",
            self.traj_file.args,
            "float",
        )
        if self.time_step is None or self.time_step <= 0:
            self.time_step = 1.0
            ddict.printLog("Warning: Invalid time step, using default 1.0 ps", color="orange")

        # Get parameters for correlation function residence time calculation
        self.t_star = ddict.get_input(
            "Maximum transient departure time t* [ps]: ",
            self.traj_file.args,
            "float",
        )
        if self.t_star is None or self.t_star < 0:
            self.t_star = 2.0

        self.max_correlation_time = ddict.get_input(
            "Maximum correlation time for residence analysis [ps]: ",
            self.traj_file.args,
            "float",
        )
        if self.max_correlation_time is None or self.max_correlation_time <= 0:
            self.max_correlation_time = 50.0

        ddict.printLog("\nLayer setup complete. Starting trajectory analysis...")

    def _get_layer_setup(self, cnt_description):
        """
        Get the radial layer boundaries from user input.

        Parameters:
        -----------
        cnt_description : str
            Description of the CNT(s) for user prompts

        Returns:
        --------
        list
            List of layer boundary distances
        """
        ddict.printLog(f"\n--- Setting up radial layers for {cnt_description} ---")

        # Always start from center
        layer_edges = [0.0]
        layer_num = 1

        while True:
            distance = ddict.get_input(
                f"Distance for outer edge of layer {layer_num} [Å] (or 'done' to finish): ",
                self.traj_file.args,
                "string",
            )

            if distance.lower() == "done":
                if len(layer_edges) < 2:
                    ddict.printLog("You need at least one layer! Please specify a distance.", color="red")
                    continue
                break

            try:
                dist_float = float(distance)
                if dist_float <= layer_edges[-1]:
                    ddict.printLog(f"Distance must be greater than {layer_edges[-1]:.2f} Å", color="red")
                    continue
                layer_edges.append(dist_float)
                layer_num += 1
            except ValueError:
                ddict.printLog("Invalid input. Please enter a number or 'done'.", color="red")

        ddict.printLog(f"Layer setup: {len(layer_edges) - 1} layers with boundaries at {layer_edges} Å")
        return layer_edges

    def ring_mean(self, ring):
        """Calculate the mean position of a ring (copied from flex_rad_dens)"""
        ring_x = ring["x"].mean()
        ring_y = ring["y"].mean()
        ring_z = ring["z"].mean()
        ring_array = np.array([ring_x, ring_y, ring_z])
        return ring_array

    def adjust_ring_pbc(self, ring_df, first_atom_coords, box_size, ddict):
        """
        Adjust atoms in a ring that are too far away due to periodic boundary conditions.
        (Copied from flex_rad_dens)
        """
        # Make a copy
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

            ring["dist_x"] = np.abs(ring[["x"]].values - first_atom_coords[0])
            ring["dist_y"] = np.abs(ring[["y"]].values - first_atom_coords[1])
            ring["dist_z"] = np.abs(ring[["z"]].values - first_atom_coords[2])
            ring["dist"] = np.sqrt(ring["dist_x"] ** 2 + ring["dist_y"] ** 2 + ring["dist_z"] ** 2)

            iterations += 1

        if iterations == max_iterations:
            ddict.printLog("Warning: Maximum iterations reached, some atoms may still be misplaced", color="red")

        return ring

    def analyze_frame(self, split_frame, frame_counter):
        """
        Analyze each frame to track molecule positions in radial layers.
        Uses center of mass for each molecule.
        """
        # Convert coordinate columns to float
        split_frame["X"] = split_frame["X"].astype(float)
        split_frame["Y"] = split_frame["Y"].astype(float)
        split_frame["Z"] = split_frame["Z"].astype(float)

        # Get box size for PBC handling
        box_size = self.traj_file.box_size

        liquid_atoms = split_frame[split_frame["Struc"].str.contains("Liquid")]

        molecules_data = {}

        molecule_groups = liquid_atoms.groupby("Molecule")

        for mol_id, molecule_atoms in molecule_groups:
            if molecule_atoms.empty:
                continue

            # Get molecule coordinates and masses
            mol_coords = molecule_atoms[["X", "Y", "Z"]].values.astype(float)
            atom_masses = molecule_atoms["Mass"].values.astype(float)

            # Get box size for PBC handling
            box_size = self.traj_file.box_size

            # Use the first atom as reference for PBC correction
            ref_coord = mol_coords[0]
            corrected_coords = np.zeros_like(mol_coords)
            corrected_coords[0] = ref_coord  # First atom stays as reference

            # Apply PBC correction to other atoms relative to the first atom
            for i in range(1, len(mol_coords)):
                coord = mol_coords[i]
                delta = coord - ref_coord
                # Apply minimum image convention for each component
                delta[0] -= box_size[0] * np.round(delta[0] / box_size[0])
                delta[1] -= box_size[1] * np.round(delta[1] / box_size[1])
                delta[2] -= box_size[2] * np.round(delta[2] / box_size[2])
                corrected_coords[i] = ref_coord + delta

            # Calculate center of mass using corrected coordinates
            total_mass = np.sum(atom_masses)
            molecule_com = np.sum(corrected_coords * atom_masses[:, np.newaxis], axis=0) / total_mass

            molecules_data[mol_id] = {
                "x": molecule_com[0],
                "y": molecule_com[1],
                "z": molecule_com[2],
                "mass": total_mass,
            }

        # Process each CNT
        for cnt_id, pair_list in self.cnt_data.items():
            layer_edges = self.cnts_layer_edges[cnt_id]
            layer_labels = self.cnts_layer_labels[cnt_id]

            # Initialize frame data for this CNT if not exists
            if cnt_id not in self.molecule_frame_data:
                self.molecule_frame_data[cnt_id] = {}

            # Initialize layer population count for this frame
            if frame_counter not in self.layer_population_data[cnt_id]:
                self.layer_population_data[cnt_id][frame_counter] = {label: 0 for label in layer_labels}

            molecules_in_cnt = {}

            if len(pair_list) > 0:
                # Use first pair as representative
                pair_data = pair_list[0]
                # Get ring identifiers and setup
                r1_key = pair_data["r1_key"]
                r2_key = pair_data["r2_key"]
                is_periodic = pair_data.get("is_periodic", False)

                # Get coordinates from the current frame
                ring1 = split_frame.loc[self.cnt_rings[cnt_id][r1_key]].copy()

                if is_periodic:
                    ring2 = ring1.copy()
                    periodic_direction = pair_data.get("periodic_direction", "z")
                    if periodic_direction == "x":
                        ring2["X"] = ring2["X"] + box_size[0]
                    elif periodic_direction == "y":
                        ring2["Y"] = ring2["Y"] + box_size[1]
                    elif periodic_direction == "z":
                        ring2["Z"] = ring2["Z"] + box_size[2]
                else:
                    ring2 = split_frame.loc[self.cnt_rings[cnt_id][r2_key]].copy()

                # Convert column names for consistency
                ring1 = ring1.rename(columns={"X": "x", "Y": "y", "Z": "z"})
                ring2 = ring2.rename(columns={"X": "x", "Y": "y", "Z": "z"})

                # Get reference coordinates and adjust for PBC
                first_atom_ring1 = ring1.iloc[0][["x", "y", "z"]].values.astype(float)
                first_atom_ring2 = ring2.iloc[0][["x", "y", "z"]].values.astype(float)

                ring1_adjusted = self.adjust_ring_pbc(ring1, first_atom_ring1, box_size, ddict)
                ring2_adjusted = self.adjust_ring_pbc(ring2, first_atom_ring2, box_size, ddict)

                # Calculate ring centers and CNT properties
                ring1_array = self.ring_mean(ring1_adjusted)
                ring2_array = self.ring_mean(ring2_adjusted)

                # Calculate tube radius
                ring_radii = []
                for index, row in ring1_adjusted.iterrows():
                    ring1_ref = row[["x", "y", "z"]].values.astype(float)
                    dist_ring = np.linalg.norm(ring1_array - ring1_ref)
                    ring_radii.append(dist_ring)
                dist_ring = np.mean(ring_radii)

                # Calculate CNT axis
                cnt_axis = (ring2_array - ring1_array) / np.linalg.norm(ring2_array - ring1_array)

                # Process each molecule to determine its layer using center of mass
                for mol_id, mol_data in molecules_data.items():
                    mol_pos = np.array([mol_data["x"], mol_data["y"], mol_data["z"]])

                    # Calculate position relative to CNT
                    M = 0.5 * (ring1_array + ring2_array)
                    half = 0.5 * np.linalg.norm(ring2_array - ring1_array)

                    delta = mol_pos - M

                    delta[0] -= box_size[0] * np.round(delta[0] / box_size[0])
                    delta[1] -= box_size[1] * np.round(delta[1] / box_size[1])
                    delta[2] -= box_size[2] * np.round(delta[2] / box_size[2])

                    proj = np.dot(delta, cnt_axis)
                    radial_vec = delta - proj * cnt_axis
                    radial_dist = np.linalg.norm(radial_vec)

                    # Check if molecule center of mass is inside the CNT
                    inside_cylinder = (np.abs(proj) <= half) and (radial_dist <= dist_ring)

                    if inside_cylinder:
                        # Determine which layer the molecule center of mass is in
                        current_layer = None
                        for i in range(len(layer_edges) - 1):
                            if layer_edges[i] <= radial_dist < layer_edges[i + 1]:
                                current_layer = layer_labels[i]
                                break

                        # If radial distance is exactly at the outer edge, assign to the outermost layer
                        if current_layer is None and radial_dist <= layer_edges[-1]:
                            current_layer = layer_labels[-1]

                        if current_layer is not None:
                            # Mark this molecule as inside this CNT with the determined layer
                            molecules_in_cnt[mol_id] = current_layer
                        else:
                            # Inside CNT but outside defined layers
                            molecules_in_cnt[mol_id] = "outside"

            # Now process all molecules for this CNT (only once each)
            for mol_id, mol_data in molecules_data.items():
                # Initialize molecule frame data if not exists
                if mol_id not in self.molecule_frame_data[cnt_id]:
                    self.molecule_frame_data[cnt_id][mol_id] = []

                if mol_id in molecules_in_cnt:
                    # Molecule is inside this CNT
                    current_layer = molecules_in_cnt[mol_id]

                    if current_layer != "outside":
                        # Store frame data
                        self.molecule_frame_data[cnt_id][mol_id].append(current_layer)

                        # Increment population count for this layer
                        self.layer_population_data[cnt_id][frame_counter][current_layer] += 1
                    else:
                        # Inside CNT but outside defined layers
                        self.molecule_frame_data[cnt_id][mol_id].append("outside")
                else:
                    # Molecule center of mass is outside CNT
                    self.molecule_frame_data[cnt_id][mol_id].append("Outside")

        self.proc_frame_counter += 1

    def resid_time_processing(self):
        """
        Process the molecule trajectories to calculate residence times using correlation function approach.
        """
        ddict.printLog("\nProcessing residence times...")
        ddict.printLog("Parameters:")
        ddict.printLog(f"- Frames analyzed: {self.proc_frame_counter}")
        ddict.printLog(f"- Time step: {self.time_step} ps")
        ddict.printLog(f"- Transient departure threshold (t*): {self.t_star} ps")
        ddict.printLog(f"- Maximum correlation time: {self.max_correlation_time} ps")

        # Calculate correlation functions for each CNT and layer
        for cnt_id in self.molecule_frame_data:
            ddict.printLog(f"\nProcessing CNT {cnt_id}...")
            layer_labels = self.cnts_layer_labels[cnt_id]

            for layer in layer_labels:
                ddict.printLog(f"  Calculating correlation function for layer {layer}...")

                # Calculate time correlation function for this layer
                correlation_times, correlation_values = self._calculate_layer_correlation_function(cnt_id, layer)

                # Fit exponential decay to determine residence time
                if len(correlation_times) > 3 and len(correlation_values) > 3:
                    residence_time, fit_success = self._perform_exponential_fitting(
                        correlation_times, correlation_values
                    )

                    # Store results
                    self.correlation_functions[cnt_id][layer] = {
                        "time": correlation_times,
                        "correlation": correlation_values,
                        "residence_time": residence_time,
                        "fit_success": fit_success,
                    }

                    if residence_time is not None and fit_success:
                        ddict.printLog(f"    -> Residence time τ = {residence_time:.3f} ps (fit successful)")
                    else:
                        ddict.printLog(f"    -> Residence time τ = {residence_time:.3f} ps (fit failed)")
                else:
                    ddict.printLog("    -> Insufficient data for correlation analysis")
                    self.correlation_functions[cnt_id][layer] = {
                        "time": correlation_times,
                        "correlation": correlation_values,
                        "residence_time": np.nan,
                        "fit_success": False,
                    }

        # Generate statistics
        self._generate_residence_statistics()
        self._generate_layer_population_statistics()

        # Plot correlation functions and results
        plot_correlation = ddict.get_input(
            "Do you want to plot the correlation functions? (y/n) ",
            self.traj_file.args,
            "string",
        )

        if plot_correlation.lower() == "y":
            self._plot_correlation_functions()

        plot_population = ddict.get_input(
            "Do you want to plot the layer population data? (y/n) ",
            self.traj_file.args,
            "string",
        )

        if plot_population.lower() == "y":
            self._plot_layer_populations()

        # Generate combined analysis report
        self._generate_combined_analysis_report()

    def _validate_segments(self, segments):
        """
        Validate that filtered segments meet the minimum consecutive frames requirement.
        This is a debugging/validation method.

        Parameters:
        -----------
        segments : list
            List of filtered segments

        Returns:
        --------
        bool
            True if all segments are valid, False otherwise
        """
        for seg in segments:
            seg_length = seg["end"] - seg["start"] + 1
            if seg_length < self.min_consecutive_frames:
                ddict.printLog(
                    f"Warning: Invalid segment found with length {seg_length} < {self.min_consecutive_frames}",
                    color="red",
                )
                return False
        return True

    def _generate_residence_statistics(self):
        """
        Generate and save residence time statistics based on correlation functions.
        """
        ddict.printLog("\n=== Residence Time Statistics ===")

        for cnt_id, layer_data in self.correlation_functions.items():
            ddict.printLog(f"\nCNT {cnt_id}:")

            # Create results dataframe
            results_data = []

            for layer, correlation_data in layer_data.items():
                if correlation_data and "correlation" in correlation_data:
                    time_points = correlation_data["time"]
                    correlation = correlation_data["correlation"]
                    residence_time = correlation_data.get("residence_time", np.nan)
                    fit_success = correlation_data.get("fit_success", False)

                    # Calculate basic statistics from correlation function
                    initial_population = correlation[0] if len(correlation) > 0 else 0
                    final_population = correlation[-1] if len(correlation) > 0 else 0

                    # Find time when correlation drops to 1/e
                    tau_1_e_index = None
                    for i, c_val in enumerate(correlation):
                        if c_val <= np.exp(-1) * initial_population:
                            tau_1_e_index = i
                            break

                    tau_1_e_time = time_points[tau_1_e_index] if tau_1_e_index is not None else np.nan

                    stats = {
                        "CNT_ID": cnt_id,
                        "Layer": layer,
                        "Initial_Population": initial_population,
                        "Final_Population": final_population,
                        "Residence_Time_ps": residence_time,
                        "Tau_1_e_Time_ps": tau_1_e_time,
                        "Fit_Success": fit_success,
                        "Correlation_Points": len(correlation),
                    }

                    results_data.append(stats)

                    if fit_success:
                        ddict.printLog(f"  {layer}: Residence time = {residence_time:.2f} ps (fit successful)")
                    else:
                        ddict.printLog(f"  {layer}: Residence time = {residence_time:.2f} ps (fit failed)")
                        if not np.isnan(tau_1_e_time):
                            ddict.printLog(f"    Alternative τ(1/e) = {tau_1_e_time:.2f} ps")

            # Save results
            if results_data:
                results_df = pd.DataFrame(results_data)
                results_df.to_csv(
                    f"CNT_{cnt_id}_residence_time_statistics.csv", sep=";", index=False, float_format="%.3f"
                )
                ddict.printLog(f"-> Statistics saved as CNT_{cnt_id}_residence_time_statistics.csv")

                # Save raw correlation function data
                correlation_raw_data = []
                for layer, correlation_data in layer_data.items():
                    if correlation_data and "correlation" in correlation_data:
                        for i, (time, corr_val) in enumerate(
                            zip(correlation_data["time"], correlation_data["correlation"])
                        ):
                            correlation_raw_data.append(
                                {
                                    "CNT_ID": cnt_id,
                                    "Layer": layer,
                                    "Time_ps": time,
                                    "Correlation": corr_val,
                                    "Time_Index": i,
                                }
                            )

                if correlation_raw_data:
                    correlation_df = pd.DataFrame(correlation_raw_data)
                    correlation_df.to_csv(
                        f"CNT_{cnt_id}_correlation_functions_raw.csv", sep=";", index=False, float_format="%.6f"
                    )
                    ddict.printLog(f"-> Raw correlation data saved as CNT_{cnt_id}_correlation_functions_raw.csv")

            # Save frame-by-frame molecule layer data (one row per molecule, one column per frame)
            if cnt_id in self.molecule_frame_data and self.molecule_frame_data[cnt_id]:
                frame_columns = [f"Frame_{i + 1}" for i in range(self.proc_frame_counter)]
                molecule_frame_df_data = []

                for mol_id, layer_sequence in self.molecule_frame_data[cnt_id].items():
                    # Pad sequence to match total frames if necessary
                    padded_sequence = layer_sequence + ["Outside"] * (self.proc_frame_counter - len(layer_sequence))

                    row_data = {"Molecule_ID": mol_id}
                    for i, layer in enumerate(padded_sequence):
                        row_data[frame_columns[i]] = layer

                    molecule_frame_df_data.append(row_data)

                if molecule_frame_df_data:
                    molecule_frame_df = pd.DataFrame(molecule_frame_df_data)
                    molecule_frame_df.to_csv(f"CNT_{cnt_id}_molecule_layer_trajectory.csv", sep=";", index=False)
                    ddict.printLog(
                        f"-> Frame-by-frame molecule layer data saved as CNT_{cnt_id}_molecule_layer_trajectory.csv"
                    )

    def _generate_layer_population_statistics(self):
        """
        Generate and save layer population statistics (molecules per layer per frame).
        """
        ddict.printLog("\n=== Layer Population Statistics ===")

        for cnt_id, frame_data in self.layer_population_data.items():
            if not frame_data:
                continue

            ddict.printLog(f"\nCNT {cnt_id}:")

            # Create population dataframe (frames as rows, layers as columns)
            sorted_frames = sorted(frame_data.keys())
            layer_labels = self.cnts_layer_labels[cnt_id]

            population_data = []
            for frame in sorted_frames:
                row_data = {"Frame": frame, "Time_ps": frame * self.time_step}
                for layer in layer_labels:
                    row_data[f"Layer_{layer}"] = frame_data[frame].get(layer, 0)
                population_data.append(row_data)

            population_df = pd.DataFrame(population_data)
            population_df.to_csv(f"CNT_{cnt_id}_layer_population.csv", sep=";", index=False)
            ddict.printLog(f"-> Layer population data saved as CNT_{cnt_id}_layer_population.csv")

            # Calculate and display summary statistics
            for layer in layer_labels:
                layer_counts = [frame_data[frame].get(layer, 0) for frame in sorted_frames]
                mean_count = np.mean(layer_counts)
                std_count = np.std(layer_counts)
                max_count = np.max(layer_counts)
                min_count = np.min(layer_counts)

                ddict.printLog(
                    f"  {layer}: mean = {mean_count:.2f}, "
                    f"std = {std_count:.2f}, "
                    f"range = {min_count}-{max_count} molecules"
                )

    def _plot_layer_populations(self):
        """
        Create plots for layer population analysis.
        """
        for cnt_id, frame_data in self.layer_population_data.items():
            if not frame_data:
                continue

            sorted_frames = sorted(frame_data.keys())
            layer_labels = self.cnts_layer_labels[cnt_id]
            times = [frame * self.time_step for frame in sorted_frames]

            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

            # Plot 1: Time series of molecule counts per layer
            colors = plt.cm.viridis(np.linspace(0, 0.9, len(layer_labels)))

            for idx, layer in enumerate(layer_labels):
                layer_counts = [frame_data[frame].get(layer, 0) for frame in sorted_frames]
                ax1.plot(
                    times, layer_counts, label=f"{layer}", color=colors[idx], linewidth=1.5, marker="o", markersize=2
                )

            ax1.set_xlabel("Time [ps]")
            ax1.set_ylabel("Number of Molecules")
            ax1.set_title(f"CNT {cnt_id} - Molecule Count per Layer vs Time")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot 2: Stacked area plot
            layer_counts_matrix = []
            for layer in layer_labels:
                layer_counts = [frame_data[frame].get(layer, 0) for frame in sorted_frames]
                layer_counts_matrix.append(layer_counts)

            ax2.stackplot(times, *layer_counts_matrix, labels=layer_labels, colors=colors, alpha=0.7)
            ax2.set_xlabel("Time [ps]")
            ax2.set_ylabel("Number of Molecules")
            ax2.set_title(f"CNT {cnt_id} - Stacked Molecule Count per Layer")
            ax2.legend(loc="upper right")
            ax2.grid(True, alpha=0.3)

            # Plot 3: Box plot of population distributions
            box_data = []
            box_labels = []
            for layer in layer_labels:
                layer_counts = [frame_data[frame].get(layer, 0) for frame in sorted_frames]
                box_data.append(layer_counts)
                box_labels.append(layer)

            ax3.boxplot(box_data, labels=box_labels)
            ax3.set_xlabel("Layer")
            ax3.set_ylabel("Number of Molecules")
            ax3.set_title(f"CNT {cnt_id} - Population Distribution per Layer")
            ax3.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f"CNT_{cnt_id}_layer_population_analysis.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
            ddict.printLog(f"-> Layer population plot saved as CNT_{cnt_id}_layer_population_analysis.png")

    def _generate_combined_analysis_report(self):
        """
        Generate a combined report showing both residence times and population statistics.
        """
        ddict.printLog("\n=== Combined Residence Time and Population Analysis ===")

        for cnt_id in self.correlation_functions.keys():
            combined_data = []

            # Get population statistics
            if cnt_id in self.layer_population_data:
                frame_data = self.layer_population_data[cnt_id]
                sorted_frames = sorted(frame_data.keys())
                layer_labels = self.cnts_layer_labels[cnt_id]

                for layer in layer_labels:
                    # Population statistics
                    layer_counts = [frame_data[frame].get(layer, 0) for frame in sorted_frames]
                    mean_population = np.mean(layer_counts)
                    std_population = np.std(layer_counts)
                    max_population = np.max(layer_counts)

                    # Residence time from correlation function
                    correlation_data = self.correlation_functions[cnt_id].get(layer, {})
                    if correlation_data and "residence_time" in correlation_data:
                        residence_time = correlation_data["residence_time"]
                        fit_success = correlation_data.get("fit_success", False)
                        initial_pop = (
                            correlation_data.get("correlation", [0])[0] if "correlation" in correlation_data else 0
                        )
                    else:
                        residence_time = 0
                        fit_success = False
                        initial_pop = 0

                    combined_data.append(
                        {
                            "CNT_ID": cnt_id,
                            "Layer": layer,
                            "Mean_Population": mean_population,
                            "Std_Population": std_population,
                            "Max_Population": max_population,
                            "Residence_Time_ps": residence_time,
                            "Fit_Success": fit_success,
                            "Initial_Population": initial_pop,
                            "Correlation_Available": bool(correlation_data),
                        }
                    )

            if combined_data:
                combined_df = pd.DataFrame(combined_data)
                combined_df.to_csv(f"CNT_{cnt_id}_combined_analysis.csv", sep=";", index=False, float_format="%.4f")
                ddict.printLog(f"-> Combined analysis saved as CNT_{cnt_id}_combined_analysis.csv")

                # Print summary
                ddict.printLog(f"\nCNT {cnt_id} Combined Analysis Summary:")
                for _, row in combined_df.iterrows():
                    fit_status = "fit_ok" if row["Fit_Success"] else "fit_failed"
                    ddict.printLog(
                        f"  {row['Layer']}: "
                        f"avg pop = {row['Mean_Population']:.1f}, "
                        f"residence time = {row['Residence_Time_ps']:.1f} ps ({fit_status})"
                    )

    def _plot_residence_times(self):
        """
        Create plots for residence time analysis using correlation functions.
        This method now calls the correlation function plotting method.
        """
        ddict.printLog("Plotting correlation functions for residence time analysis...")
        self._plot_correlation_functions()

    def _plot_cnt_comparison(self):
        """
        Create comparison plot for multiple CNTs based on residence times from correlation functions.
        """
        if len(self.correlation_functions) <= 1:
            return

        fig, ax = plt.subplots(figsize=(12, 8))

        x_offset = 0
        x_positions = []
        x_labels = []

        for cnt_id, layer_data in self.correlation_functions.items():
            valid_layers = {}

            # Extract residence times from correlation function results
            for layer, correlation_data in layer_data.items():
                if correlation_data and "residence_time" in correlation_data:
                    residence_time = correlation_data["residence_time"]
                    if not np.isnan(residence_time):
                        valid_layers[layer] = residence_time

            if not valid_layers:
                continue

            layer_times = list(valid_layers.values())
            layer_names = list(valid_layers.keys())

            x_pos = np.arange(len(layer_names)) + x_offset

            ax.bar(x_pos, layer_times, label=f"CNT {cnt_id}", alpha=0.7)

            x_positions.extend(x_pos)
            x_labels.extend([f"{cnt_id}_{layer}" for layer in layer_names])

            x_offset += len(layer_names) + 1

        ax.set_xlabel("CNT and Layer")
        ax.set_ylabel("Residence Time [ps]")
        ax.set_title("Comparison of Residence Times Across CNTs and Layers")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.savefig("CNT_residence_time_comparison.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        ddict.printLog("-> CNT comparison plot saved as CNT_residence_time_comparison.png")

    def _calculate_layer_correlation_function(self, cnt_id, layer):
        """
        Calculate the time correlation function for a specific layer using a fast
        segment/survival algorithm (O(N_frames × N_molecules) pre-processing + O(max_lag)).

        Parameters:
        -----------
        cnt_id : str
            CNT identifier
        layer : str
            Layer identifier

        Returns:
        --------
        tuple
            (correlation_times, correlation_values) lists
        """
        max_time_frames = min(int(self.max_correlation_time / self.time_step), self.proc_frame_counter - 1)
        t_star_frames = int(self.t_star / self.time_step)

        # Get all molecules that were ever in this CNT
        cnt_molecules = list(self.molecule_frame_data[cnt_id].keys())
        if len(cnt_molecules) == 0:
            return [], []

        ddict.printLog(f"    -> Processing {len(cnt_molecules)} molecules using fast survival correlation...")

        correlation_values = self._fast_survival_correlation(cnt_id, layer, max_time_frames, t_star_frames)
        if len(correlation_values) == 0:
            return [], []

        # Prepend zero-lag point C(0)=1 (survival probability at zero lag)
        correlation_times = [0.0] + [(i + 1) * self.time_step for i in range(len(correlation_values))]
        correlation_values = [1.0] + correlation_values
        return correlation_times, correlation_values

    def _fast_survival_correlation(self, cnt_id, layer, max_lag, t_star_frames):

        T = self.proc_frame_counter
        if T < 2:
            return []
        max_lag = min(max_lag, T - 1)
        occupancy_counts = np.zeros(T, dtype=np.int32)
        N_diff = np.zeros(max_lag + 2, dtype=np.int64)

        for mol_id, seq in self.molecule_frame_data[cnt_id].items():
            if len(seq) < T:
                seq = seq + ["Outside"] * (T - len(seq))
            segments = []
            in_seg = False
            gap = 0
            seg_start = None
            last_layer_frame = None
            for f in range(T):
                is_layer = seq[f] == layer
                if is_layer:
                    if not in_seg:
                        in_seg = True
                        seg_start = f
                    last_layer_frame = f
                    gap = 0
                else:
                    if in_seg:
                        gap += 1
                        if gap > t_star_frames:
                            if last_layer_frame is not None and seg_start is not None:
                                segments.append((seg_start, last_layer_frame))
                            in_seg = False
                            gap = 0
                            seg_start = None
                            last_layer_frame = None
            if in_seg and last_layer_frame is not None:
                segments.append((seg_start, last_layer_frame))

            if not segments:
                continue

            for seg_start, seg_end in segments:
                for f in range(seg_start, seg_end + 1):
                    if seq[f] != layer:
                        continue
                    occupancy_counts[f] += 1
                    L_i = seg_end - f
                    if L_i < 1:
                        continue
                    L_lim = min(L_i, T - 1 - f, max_lag)
                    if L_lim >= 1:
                        N_diff[1] += 1
                        N_diff[L_lim + 1] -= 1

        occ_prefix = np.cumsum(occupancy_counts)
        N = np.cumsum(N_diff)

        correlation = []
        for d in range(1, max_lag + 1):
            last_time_origin = T - 1 - d
            if last_time_origin < 0:
                break
            D_d = occ_prefix[last_time_origin]
            if D_d > 0:
                correlation.append(N[d] / D_d)
            else:
                correlation.append(0.0)
        return correlation

    def _perform_exponential_fitting(self, correlation_times, correlation_values):
        """
        Fit exponential decay to correlation function and extract residence time.

        Parameters:
        -----------
        correlation_times : list
            Time lag values
        correlation_values : list
            Correlation function values

        Returns:
        --------
        tuple
            (residence_time, fit_success) where residence_time is tau at 1/e decay
        """
        if len(correlation_times) < 3 or len(correlation_values) < 3:
            return 0.0, False

        try:
            # Convert to numpy arrays and filter positive values
            times = np.array(correlation_times)
            values = np.array(correlation_values)

            valid_mask = (values > 0) & np.isfinite(values) & np.isfinite(times)
            if np.sum(valid_mask) < 3:
                return 0.0, False

            times = times[valid_mask]
            values = values[valid_mask]

            # Model with fixed amplitude: C(t) = exp(-t / tau)
            def exp_decay(t, tau):
                return np.exp(-t / tau)

            # Initial guess for tau: time where C(t) ~ 0.5 (half-life)
            half_idx = np.argmin(np.abs(values - 0.5))
            tau_init = times[half_idx] if half_idx > 0 else max(times[-1] / 3, 1e-6)

            # Exclude t = 0 from fitting to avoid overweighting the exact 1.0 point
            nonzero_mask = times > 0
            if np.sum(nonzero_mask) >= 3:
                fit_times = times[nonzero_mask]
                fit_values = values[nonzero_mask]
            else:
                fit_times = times
                fit_values = values

            popt, pcov = curve_fit(
                exp_decay,
                fit_times,
                fit_values,
                p0=[tau_init],
                bounds=([0.0], [10 * fit_times[-1]]),
                maxfev=5000,
            )

            tau_fit = popt[0]

            y_pred = exp_decay(fit_times, tau_fit)
            denom = np.sum((fit_values - np.mean(fit_values)) ** 2)
            if denom <= 0:
                r_squared = 1.0 if np.allclose(fit_values, y_pred) else 0.0
            else:
                r_squared = 1 - np.sum((fit_values - y_pred) ** 2) / denom

            if r_squared > 0.7 and tau_fit > 0:
                return tau_fit, True
            else:
                return tau_fit, False

        except Exception as e:
            ddict.printLog(f"      Warning: Exponential fit failed: {e}")
            return 0.0, False

    def _plot_correlation_functions(self):
        """
        Create plots for correlation functions and residence time fits.
        """
        for cnt_id, layer_data in self.correlation_functions.items():
            non_empty_layers = {
                layer: data
                for layer, data in layer_data.items()
                if data and len(data.get("time", [])) > 0 and len(data.get("correlation", [])) > 0
            }

            if not non_empty_layers:
                continue

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            # Plot 1: Correlation functions
            colors = plt.cm.viridis(np.linspace(0, 0.9, len(non_empty_layers)))

            for idx, (layer, data) in enumerate(non_empty_layers.items()):
                times = data["time"]
                correlations = data["correlation"]
                residence_time = data.get("residence_time", np.nan)
                fit_success = data.get("fit_success", False)

                label = f"{layer}"
                if not np.isnan(residence_time):
                    fit_mark = "fit_ok" if fit_success else "fit_failed"
                    label += f" (τ={residence_time:.2f} ps, {fit_mark})"

                ax1.plot(times, correlations, "o-", label=label, color=colors[idx], linewidth=1.5, markersize=3)

                # Mark 1/e point
                if not np.isnan(residence_time) and residence_time <= max(times):
                    ax1.axhline(y=1 / np.e, color=colors[idx], linestyle="--", alpha=0.5)
                    ax1.axvline(x=residence_time, color=colors[idx], linestyle="--", alpha=0.5)

            ax1.set_xlabel("Time [ps]")
            ax1.set_ylabel("Correlation Function C(t)")
            ax1.set_title(f"CNT {cnt_id} - Residence Time Correlation Functions")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1.1)

            # Plot 2: Semi-log plot for better visualization of exponential decay
            for idx, (layer, data) in enumerate(non_empty_layers.items()):
                times = data["time"]
                correlations = data["correlation"]

                # Filter positive correlations for log plot
                pos_corr = [(t, c) for t, c in zip(times, correlations) if c > 1e-6]
                if len(pos_corr) > 0:
                    t_pos, c_pos = zip(*pos_corr)
                    ax2.semilogy(t_pos, c_pos, "o-", label=f"{layer}", color=colors[idx], linewidth=1.5, markersize=3)

            ax2.axhline(y=1 / np.e, color="black", linestyle=":", alpha=0.7, label="1/e")
            ax2.set_xlabel("Time [ps]")
            ax2.set_ylabel("Correlation Function C(t) [log]")
            ax2.set_title(f"CNT {cnt_id} - Correlation Functions (Semi-log)")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f"CNT_{cnt_id}_correlation_functions.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
            ddict.printLog(f"-> Correlation functions plot saved as CNT_{cnt_id}_correlation_functions.png")
