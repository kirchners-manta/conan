# import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import conan.analysis_modules.flex_rad_dens as flexrd
import conan.analysis_modules.traj_an as traj_an

# import conan.analysis_modules.traj_info as traj_info
import conan.defdict as ddict


def flex_angle(traj_file, molecules, an):
    """
    Main function.
    """
    flex_angle_analyzer = FlexAngle(traj_file, molecules, an)
    flex_angle_analyzer.flex_rad_dens_prep()
    traj_an.process_trajectory(traj_file, molecules, an, flex_angle_analyzer)
    flex_angle_analyzer.flex_angle_processing()

    return flex_angle_analyzer


class FlexAngle(flexrd.FlexRadDens):
    """
    Calculate the angle distribution of molecules confined within a flexible CNT.
    Inherits from FlexRadDens to reuse functionality.
    """

    def __init__(self, traj_file, molecules, an):

        # Initialize parent class
        super().__init__(traj_file, molecules, an)

        # Initialize additional data structures
        self.angle_data = {}
        self.angle_bins = {}

        # Check if dipole vector information is given in the trajectory
        if hasattr(self.traj_file, "dipole_info") and self.traj_file.dipole_info:
            self.dipole_info = True
            ddict.printLog("=> Dipole vector information found in trajectory data. <=")
        else:
            self.dipole_info = False
            ddict.printLog("No dipole vector information found in trajectory data.")

    def flex_rad_dens_prep(self):
        """
        Prepare the radial density angle calculation.
        Uses the parent class preparation method and adds angle-specific setup.
        """
        # Call the parent class preparation method
        super().flex_rad_dens_prep()

        ddict.printLog("\n" + "=" * 50)
        ddict.printLog("ANGLE ANALYSIS SETUP")
        ddict.printLog("=" * 50)
        ddict.printLog("The radial bin position is set through the center of mass (COM)")
        ddict.printLog("-" * 40)

        # 1. Set up the reference molecule
        ddict.printLog("\n1. REFERENCE MOLECULE SETUP")
        ddict.printLog("-" * 40)
        self.setup_target_molecule()

        # 2. Set up the vectors
        ddict.printLog("\n2. VECTOR SETUP")
        ddict.printLog("-" * 40)
        self.vector_setup()

        # 3. Set up the angle incrementation
        ddict.printLog("\n3. ANGLE INCREMENTATION SETUP")
        ddict.printLog("-" * 40)
        self.setup_angle_bins()

        ddict.printLog("\n" + "=" * 50)
        ddict.printLog("ANGLE ANALYSIS SETUP COMPLETE")
        ddict.printLog("=" * 50)

    def setup_angle_bins(self):
        """
        Set up angle-specific analysis parameters.
        """
        # Set up angle bins (0 to 180 degrees, for example)
        num_angle_bins = ddict.get_input(
            "How many angle bins do you want to use? (default: 60 for 3° increments): ", self.traj_file.args, "int"
        )

        # Set default value if no input
        if num_angle_bins is None or num_angle_bins == 0:
            num_angle_bins = 60
            ddict.printLog(f"Using default: {num_angle_bins} angle bins (3° increments)")
        else:
            ddict.printLog(f"Using {num_angle_bins} angle bins ({180 / num_angle_bins:.1f}° increments)")

        self.angle_bins = np.linspace(0, 180, num_angle_bins + 1)

        # Initialize angle data storage for each CNT
        for cnt_id in self.cnts_bin_edges.keys():
            self.angle_data[cnt_id] = {"angles": [], "radial_positions": [], "frame_data": pd.DataFrame()}

    def vector_setup(self):
        """
        Set up vectors for angle calculations.
        Available vector types:
        1. CNT center axis direction
        2. Connection from CNT center to atom/COM
        3. Connection between two atoms
        4. Mean of bond vectors (e.g., for H2O)
        5. Dipole vector (if available)
        """
        vector_choices = [1, 2, 3, 4, 5] if self.dipole_info else [1, 2, 3, 4]
        vectors = {}
        vector_configs = {}  # Store additional configuration for each vector

        ddict.printLog("Available vector types:")
        ddict.printLog("  1. CNT center axis direction")
        ddict.printLog("  2. Connection from CNT center to atom")
        ddict.printLog("  3. Connection between two atoms")
        ddict.printLog("  4. Mean of bond vectors (e.g., for H2O)")
        if self.dipole_info:
            ddict.printLog("  5. Dipole vector")

        for i in range(1, 3):
            ddict.printLog(f"\nSetting up Vector {i}:")
            while True:
                if self.dipole_info:
                    vector_choice = ddict.get_input(f"Choose setup for vector {i} (1-5): ", self.traj_file.args, "int")
                else:
                    vector_choice = ddict.get_input(f"Choose setup for vector {i} (1-4): ", self.traj_file.args, "int")
                if vector_choice in vector_choices:
                    vectors[i] = vector_choice
                    vector_configs[i] = {}

                    # Additional configuration based on vector type
                    if vector_choice == 1:
                        # CNT center axis direction - ask for direction preference
                        ddict.printLog("  CNT center axis vector:")
                        ddict.printLog("  This vector represents the direction along the CNT axis.")
                        direction_choice = ddict.get_input(
                            "  Direction preference (1: ring1->ring2, 2: ring2->ring1): ", self.traj_file.args, "int"
                        )
                        if direction_choice == 1:
                            vector_configs[i]["direction"] = "forward"
                            ddict.printLog("  Vector will point from ring1 to ring2")
                        else:
                            vector_configs[i]["direction"] = "reverse"
                            ddict.printLog("  Vector will point from ring2 to ring1")

                    elif vector_choice == 2:
                        # Connection from CNT center to atom - clarify direction
                        ddict.printLog("  Radial vector from CNT center:")
                        direction_choice = ddict.get_input(
                            "  Vector direction (1: center -> atom, 2: atom -> center): ", self.traj_file.args, "int"
                        )
                        if direction_choice == 1:
                            vector_configs[i]["direction"] = "outward"
                            ddict.printLog("  Vector points from CNT center toward the atom (outward)")
                        else:
                            vector_configs[i]["direction"] = "inward"
                            ddict.printLog("  Vector points from atom toward CNT center (inward)")

                    elif vector_choice == 3:
                        # Connection between two atoms - let user choose atoms
                        ddict.printLog("  Bond vector between two atoms:")
                        ddict.printLog("  Specific atoms will be selected after molecule setup.")
                        vector_configs[i]["atom_selection"] = "manual"

                    elif vector_choice == 4:
                        # Mean of bond vectors - let user choose central atom and bonded atoms
                        ddict.printLog("  Mean of bond vectors:")
                        ddict.printLog("  Central and bonded atoms will be selected after molecule setup.")
                        vector_configs[i]["atom_selection"] = "manual"

                    elif vector_choice == 5 and self.dipole_info:
                        # Dipole vector - no additional configuration needed
                        ddict.printLog("  Reading the dipole vector from trajectory data.")

                    break
                else:
                    ddict.printLog(f"  Invalid choice: {vector_choice}. Please choose from {vector_choices}.")

        # Store the vectors and configurations for use in analysis
        self.vectors = vectors
        self.vector_configs = vector_configs
        ddict.printLog(f"\nVector setup complete: 1.Vector = type {vectors[1]}, 2.Vector = type {vectors[2]}")

        # Setup atom selection for reference points
        self.setup_atom_selection()

    def setup_target_molecule(self):
        """
        Set up the target species for angle analysis.
        """
        # Get available species from the molecules object
        if hasattr(self.molecules, "unique_molecule_frame") and not self.molecules.unique_molecule_frame.empty:
            available_molecules = self.molecules.unique_molecule_frame["Species"].tolist()

            # Ensure all species are integers
            available_molecules = [int(species) for species in available_molecules]

            # Check which species are structures and remove them from the list
            # in the traj_info.frame0 dataset, the column "Struc" not being "Liquid" indicates a structure
            struc_species = (
                self.traj_file.frame0[self.traj_file.frame0["Struc"] != "Liquid"]["Species"].unique().tolist()
            )

            available_molecules = [species for species in available_molecules if species not in struc_species]
            ddict.printLog(f"Available liquid species: {available_molecules}")

            # check if there are more than one available species
            if len(available_molecules) == 0:
                ddict.printLog("No valid species found for angle analysis. Please check your trajectory data.")
                self.target_molecule = None
            elif len(available_molecules) == 1:
                self.target_molecule = available_molecules[0]
                ddict.printLog(f"Species automatically selected: {self.target_molecule}")
            else:
                while True:
                    target_molecule = ddict.get_input(
                        f"Which molecule type do you want to analyze? {available_molecules}: ",
                        self.traj_file.args,
                        "int",
                    )

                    target_molecule = int(target_molecule)

                    if target_molecule in available_molecules:
                        self.target_molecule = target_molecule
                        ddict.printLog(f"Target molecule selected: {target_molecule}")
                        break
                    else:
                        ddict.printLog(f"Invalid molecule type. Please choose from: {available_molecules}")

    def setup_atom_selection(self):
        """
        Set up atom selection for vector calculations.
        """
        # Get atom types for the target molecule from the molecules object
        if hasattr(self.molecules, "unique_molecule_frame") and not self.molecules.unique_molecule_frame.empty:
            target_mol_data = self.molecules.unique_molecule_frame[
                self.molecules.unique_molecule_frame["Species"] == self.target_molecule
            ]

            if not target_mol_data.empty:
                # Extract atom labels - the 'Labels' column contains a list
                labels_entry = target_mol_data["Labels"].iloc[0]
                if isinstance(labels_entry, list):
                    atom_labels = labels_entry
                else:
                    # Try to parse if it's a string representation of a list
                    try:
                        atom_labels = eval(labels_entry) if isinstance(labels_entry, str) else [labels_entry]
                    except (ValueError, SyntaxError, NameError, TypeError):
                        atom_labels = [str(labels_entry)]

                ddict.printLog(f"\nAtom types in molecule {self.target_molecule}:")
                for i, atom in enumerate(atom_labels):
                    ddict.printLog(f"  {i + 1}: {atom}")

                # Handle reference point selection for vector type 2
                needs_reference = any(self.vectors[i] == 2 for i in [1, 2])
                if needs_reference:
                    ddict.printLog("\nSetting up reference point for radial vectors:")
                    # Ask user for reference point choice
                    use_com = ddict.get_input(
                        "Use center of mass (COM) or specific atom as reference? (com/atom): ",
                        self.traj_file.args,
                        "string",
                    ).lower()

                    if use_com == "com":
                        self.reference_method = "com"
                        ddict.printLog("  Using center of mass as reference point")
                    elif use_com == "atom":
                        self.reference_method = "atom"
                        while True:
                            try:
                                atom_idx = ddict.get_input(
                                    f"  Which atom to use as reference? (1-{len(atom_labels)}): ",
                                    self.traj_file.args,
                                    "int",
                                )
                                if 1 <= atom_idx <= len(atom_labels):
                                    self.reference_atom_idx = atom_idx - 1  # Convert to 0-based index
                                    self.reference_atom_label = atom_labels[self.reference_atom_idx]
                                    ddict.printLog(
                                        f"  Using atom {atom_idx} ({self.reference_atom_label}) as reference point"
                                    )
                                    break
                                else:
                                    ddict.printLog(f"  Invalid choice. Please choose between 1 and {len(atom_labels)}")
                            except (ValueError, IndexError):
                                ddict.printLog(f"  Invalid input. Please choose between 1 and {len(atom_labels)}")
                    else:
                        ddict.printLog("  Invalid choice, defaulting to center of mass")
                        self.reference_method = "com"
                else:
                    self.reference_method = "com"  # Default for non-type-2 vectors

                # Handle manual atom selection for vectors 3 and 4
                for vector_num in [1, 2]:
                    if self.vectors[vector_num] == 3:
                        # Manual selection for bond vector (two atoms)
                        ddict.printLog(f"\nManual atom selection for Vector {vector_num} (Bond Vector):")
                        ddict.printLog("  Select two atoms to define the bond vector:")

                        # Select first atom
                        while True:
                            try:
                                atom1_idx = ddict.get_input(
                                    f"  Select first atom (base of vector) (1-{len(atom_labels)}): ",
                                    self.traj_file.args,
                                    "int",
                                )
                                if 1 <= atom1_idx <= len(atom_labels):
                                    break
                                else:
                                    ddict.printLog(f"  Invalid choice. Please choose between 1 and {len(atom_labels)}")
                            except (ValueError, IndexError):
                                ddict.printLog(f"  Invalid input. Please choose between 1 and {len(atom_labels)}")

                        # Select second atom
                        while True:
                            try:
                                atom2_idx = ddict.get_input(
                                    f"  Select second atom (tip of vector) (1-{len(atom_labels)}): ",
                                    self.traj_file.args,
                                    "int",
                                )
                                if 1 <= atom2_idx <= len(atom_labels):
                                    if atom2_idx != atom1_idx:
                                        break
                                    else:
                                        ddict.printLog("  Please select a different atom for the second position")
                                else:
                                    ddict.printLog(f"  Invalid choice. Please choose between 1 and {len(atom_labels)}")
                            except (ValueError, IndexError):
                                ddict.printLog(f"  Invalid input. Please choose between 1 and {len(atom_labels)}")

                        # Store the atom selection
                        self.vector_configs[vector_num]["atom1_idx"] = atom1_idx - 1  # Convert to 0-based
                        self.vector_configs[vector_num]["atom2_idx"] = atom2_idx - 1  # Convert to 0-based
                        ddict.printLog(
                            f"  Vector {vector_num}: {atom_labels[atom1_idx - 1]} -> {atom_labels[atom2_idx - 1]}"
                        )

                    elif self.vectors[vector_num] == 4:
                        # Manual selection for mean bond vectors (central atom + bonded atoms)
                        ddict.printLog(f"\nManual atom selection for vector {vector_num} (mean bond vector):")

                        # Select central atom
                        while True:
                            try:
                                central_idx = ddict.get_input(
                                    f"  Select central atom (base of all bond vectors) (1-{len(atom_labels)}): ",
                                    self.traj_file.args,
                                    "int",
                                )
                                if 1 <= central_idx <= len(atom_labels):
                                    break
                                else:
                                    ddict.printLog(f"  Invalid choice. Please choose between 1 and {len(atom_labels)}")
                            except (ValueError, IndexError):
                                ddict.printLog(f"  Invalid input. Please choose between 1 and {len(atom_labels)}")

                        # Select bonded atoms
                        bonded_indices = []
                        ddict.printLog("  Select atoms bonded to the central atom (enter 0 to finish):")

                        while True:
                            try:
                                prompt = (
                                    f"  Select bonded atom {len(bonded_indices) + 1} "
                                    f"(1-{len(atom_labels)}, 0 to finish): "
                                )
                                bonded_idx = ddict.get_input(
                                    prompt,
                                    self.traj_file.args,
                                    "int",
                                )
                                if bonded_idx == 0:
                                    if len(bonded_indices) >= 1:
                                        break
                                    else:
                                        ddict.printLog("  Please select at least one bonded atom")
                                elif 1 <= bonded_idx <= len(atom_labels):
                                    if bonded_idx != central_idx and (bonded_idx - 1) not in bonded_indices:
                                        bonded_indices.append(bonded_idx - 1)  # Convert to 0-based
                                        ddict.printLog(f"    Added: {atom_labels[bonded_idx - 1]}")
                                    elif bonded_idx == central_idx:
                                        ddict.printLog("  Cannot select the central atom as a bonded atom")
                                    else:
                                        ddict.printLog("  This atom is already selected")
                                else:
                                    ddict.printLog(f"  Invalid choice. Please choose between 1 and {len(atom_labels)}")
                            except (ValueError, IndexError):
                                ddict.printLog(f"  Invalid input. Please choose between 1 and {len(atom_labels)}")

                        # Store the atom selection
                        self.vector_configs[vector_num]["central_idx"] = central_idx - 1  # Convert to 0-based
                        self.vector_configs[vector_num]["bonded_indices"] = bonded_indices

                        bonded_labels = [atom_labels[idx] for idx in bonded_indices]
                        ddict.printLog(
                            f"  Vector {vector_num}: Mean of bonds from "
                            f"{atom_labels[central_idx - 1]} "
                            f"to {bonded_labels}"
                        )

            else:
                ddict.printLog("Could not find atom information for target molecule, using center of mass")
                self.reference_method = "com"
        else:
            ddict.printLog("Could not access molecule information, using center of mass")
            self.reference_method = "com"

    def analyze_frame(self, split_frame, frame_counter):
        """
        Analyze angles in the current frame.
        Based on flex_rad_dens.py's analyze_frame but adds angle calculations.
        """
        # Convert coordinate columns to float
        split_frame["X"] = split_frame["X"].astype(float)
        split_frame["Y"] = split_frame["Y"].astype(float)
        split_frame["Z"] = split_frame["Z"].astype(float)

        # Get box size for PBC handling
        box_size = self.traj_file.box_size

        """
        # Check frame structure for first frame
        if frame_counter == 1:
            ddict.printLog(f"Frame structure - Index range: {split_frame.index.min()} to {split_frame.index.max()}")
            if hasattr(self, 'cnt_rings'):
                for cnt_id in list(self.cnt_rings.keys())[:1]:  # Check first CNT only
                    first_ring_key = list(self.cnt_rings[cnt_id].keys())[0]
                    ring_indices = self.cnt_rings[cnt_id][first_ring_key]
                    # Check if these indices exist in the frame
                    missing_indices = [idx for idx in ring_indices[:5] if idx not in split_frame.index]
                    if missing_indices:
                        ddict.printLog(f"Warning: Some CNT ring indices missing from frame")
                    else:
                        ddict.printLog("CNT structure successfully detected in frame")
        """

        # Identify liquid atoms for angle analysis (AFTER we confirm CNT structure is accessible)
        liquid_atoms = split_frame[split_frame["Struc"].str.contains("Liquid")]

        # Filter for target molecule type if specified
        if hasattr(self, "target_molecule") and self.target_molecule:
            liquid_atoms = liquid_atoms[liquid_atoms["Species"] == self.target_molecule]

        if liquid_atoms.empty:
            ddict.printLog(
                f"No {getattr(self, 'target_molecule', 'target')}", " molecules found in frame {frame_counter}"
            )
            return

        # Process each CNT - use FULL split_frame for CNT ring access
        for cnt_id, pair_list in self.cnt_data.items():
            # Get the bins for this CNT
            bin_edges = self.cnts_bin_edges[cnt_id]

            # Lists to store angles and radial positions for this CNT
            frame_angles = []
            frame_radial_positions = []

            # Process each ring pair in this CNT
            for pair_idx, pair_data in enumerate(pair_list):
                # Get ring identifiers
                r1_key = pair_data["r1_key"]
                r2_key = pair_data["r2_key"]

                # Check if this is a periodic CNT ring pair
                is_periodic = pair_data.get("is_periodic", False)

                # Get coordinates from the current frame
                ring1 = split_frame.loc[self.cnt_rings[cnt_id][r1_key]].copy()

                # periodic CNTs
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

                ring1 = ring1.rename(columns={"X": "x", "Y": "y", "Z": "z"})
                ring2 = ring2.rename(columns={"X": "x", "Y": "y", "Z": "z"})

                # Get reference coordinates and adjust for PBC
                first_atom_ring1 = ring1.iloc[0][["x", "y", "z"]].values.astype(float)
                first_atom_ring2 = ring2.iloc[0][["x", "y", "z"]].values.astype(float)

                ring1_adjusted = self.adjust_ring_pbc(ring1, first_atom_ring1, box_size, ddict)
                ring2_adjusted = self.adjust_ring_pbc(ring2, first_atom_ring2, box_size, ddict)

                # Calculate centers and CNT axis
                ring1_array = self.ring_mean(ring1_adjusted)
                ring2_array = self.ring_mean(ring2_adjusted)
                cnt_axis = (ring2_array - ring1_array) / np.linalg.norm(ring2_array - ring1_array)

                # Calculate ring radius
                ring_radii = []
                for index, row in ring1_adjusted.iterrows():
                    ring1_ref = row[["x", "y", "z"]].values.astype(float)
                    dist_ring = np.linalg.norm(ring1_array - ring1_ref)
                    ring_radii.append(dist_ring)
                dist_ring = np.mean(ring_radii)

                # Apply shortening if requested
                if hasattr(self, "shortening") and self.shortening > 0:
                    ring1_array = ring1_array + self.shortening * cnt_axis
                    ring2_array = ring2_array - self.shortening * cnt_axis

                # Calculate molecules within CNT and their angles
                molecule_angles, molecule_radial_positions = self.calculate_molecular_angles(
                    liquid_atoms, ring1_array, ring2_array, cnt_axis, dist_ring, box_size
                )

                frame_angles.extend(molecule_angles)
                frame_radial_positions.extend(molecule_radial_positions)

            # Store angle data for this CNT and frame
            if frame_angles:
                # Bin the angles by radial position
                for angle, radial_pos in zip(frame_angles, frame_radial_positions):
                    # Find which radial bin this molecule belongs to
                    bin_idx = np.digitize(radial_pos, bin_edges) - 1
                    if 0 <= bin_idx < len(bin_edges) - 1:
                        # Store angle in the appropriate bin
                        self.angle_data[cnt_id]["angles"].append(
                            {
                                "frame": frame_counter,
                                "angle": angle,
                                "radial_bin": bin_idx,
                                "radial_position": radial_pos,
                            }
                        )

        self.proc_frame_counter += 1

    def calculate_molecular_angles(self, liquid_atoms, ring1_center, ring2_center, cnt_axis, cnt_radius, box_size):
        """
        Calculate angles for molecules within the CNT.
        Returns lists of angles and corresponding radial positions.
        """
        angles = []
        radial_positions = []

        # Calculate CNT center line parameters
        M = 0.5 * (ring1_center + ring2_center)
        half_length = 0.5 * np.linalg.norm(ring2_center - ring1_center)

        # Group liquid atoms by individual molecules
        molecule_groups = liquid_atoms.groupby("Molecule")

        for mol_id, molecule_atoms in molecule_groups:
            if molecule_atoms.empty:
                continue
            # molecule coordinates
            mol_coords = molecule_atoms[["X", "Y", "Z"]].values.astype(float)

            # minimum image convention
            delta = mol_coords - M
            delta[:, 0] -= box_size[0] * np.round(delta[:, 0] / box_size[0])
            delta[:, 1] -= box_size[1] * np.round(delta[:, 1] / box_size[1])
            delta[:, 2] -= box_size[2] * np.round(delta[:, 2] / box_size[2])

            # Project onto CNT axis
            proj = np.dot(delta, cnt_axis)

            # Calculate radial distances
            radial_vecs = delta - np.outer(proj, cnt_axis)
            radial_distances = np.linalg.norm(radial_vecs, axis=1)

            # Check if any atom of this molecule is inside the CNT
            inside_cylinder = (np.abs(proj) <= half_length) & (radial_distances <= cnt_radius)

            if not np.any(inside_cylinder):
                continue

            # Calculate center of mass of the molecule (weighted by atomic masses)
            atom_masses = molecule_atoms["Mass"].values.astype(float)
            total_mass = np.sum(atom_masses)
            molecule_com = np.sum(mol_coords * atom_masses[:, np.newaxis], axis=0) / total_mass

            # Apply PBC correction to COM
            com_delta = molecule_com - M
            com_delta[0] -= box_size[0] * np.round(com_delta[0] / box_size[0])
            com_delta[1] -= box_size[1] * np.round(com_delta[1] / box_size[1])
            com_delta[2] -= box_size[2] * np.round(com_delta[2] / box_size[2])

            com_proj = np.dot(com_delta, cnt_axis)
            com_radial_vec = com_delta - com_proj * cnt_axis
            com_radial_distance = np.linalg.norm(com_radial_vec)

            # Calculate vectors based on the chosen setup
            vectors = self.get_molecule_vectors(molecule_atoms, M, cnt_axis, box_size)

            if vectors is not None and len(vectors) == 2:
                vector1, vector2 = vectors

                # Check for zero-length vectors which cause issues
                v1_magnitude = np.linalg.norm(vector1)
                v2_magnitude = np.linalg.norm(vector2)

                # Skip molecules with very small vectors (likely numerical artifacts)
                min_vector_threshold = 1e-6  # Minimum vector length threshold
                if v1_magnitude < min_vector_threshold or v2_magnitude < min_vector_threshold:
                    continue

                # Calculate angle between the two vectors
                cos_angle = np.dot(vector1, vector2) / (v1_magnitude * v2_magnitude)
                # Clamp to avoid numerical errors
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle_rad = np.arccos(cos_angle)
                angle_deg = np.degrees(angle_rad)

                """
                # Debug: Log suspicious angles for investigation (use a counter to limit output)
                if angle_deg < 1.0 or angle_deg > 179.0:
                    if not hasattr(self, '_debug_angle_count'):
                        self._debug_angle_count = 0
                    if self._debug_angle_count < 20:  # Only log first 20 suspicious angles
                        ddict.printLog(f"Debug: Suspicious angle {angle_deg:.2f}° found for molecule {mol_id}")
                        ddict.printLog(f"  Vector 1: {vector1} (magnitude: {v1_magnitude:.6f})")
                        ddict.printLog(f"  Vector 2: {vector2} (magnitude: {v2_magnitude:.6f})")
                        ddict.printLog(f"  cos_angle: {cos_angle:.6f}")
                        self._debug_angle_count += 1
                """
                angles.append(angle_deg)
                radial_positions.append(com_radial_distance)

        return angles, radial_positions

    def get_molecule_vectors(self, molecule_atoms, cnt_center, cnt_axis, box_size):
        """
        Calculate the two vectors for angle analysis based on the chosen setup.
        """
        if not hasattr(self, "vectors") or not self.vectors:
            return None

        vectors = []
        mol_coords = molecule_atoms[["X", "Y", "Z"]].values.astype(float)

        # Use the first atom as reference for the molecule
        ref_coord = mol_coords[0]
        corrected_coords = np.zeros_like(mol_coords)
        corrected_coords[0] = ref_coord  # First atom stays as reference

        for i in range(1, len(mol_coords)):
            coord = mol_coords[i]
            delta = coord - ref_coord
            # Apply minimum image convention for each component
            delta[0] -= box_size[0] * np.round(delta[0] / box_size[0])
            delta[1] -= box_size[1] * np.round(delta[1] / box_size[1])
            delta[2] -= box_size[2] * np.round(delta[2] / box_size[2])
            corrected_coords[i] = ref_coord + delta

        # Apply PBC correction only to the reference atom relative to CNT center
        ref_to_center = ref_coord - cnt_center
        ref_to_center[0] -= box_size[0] * np.round(ref_to_center[0] / box_size[0])
        ref_to_center[1] -= box_size[1] * np.round(ref_to_center[1] / box_size[1])
        ref_to_center[2] -= box_size[2] * np.round(ref_to_center[2] / box_size[2])

        # Calculate the corrected reference position
        corrected_ref_pos = cnt_center + ref_to_center

        # Translate all atoms by the same offset to preserve internal structure
        translation_offset = corrected_ref_pos - ref_coord
        corrected_coords += translation_offset

        for i in range(1, 3):  # Vector 1 and Vector 2
            vector_setup = self.vectors.get(i)
            vector_config = self.vector_configs.get(i, {})

            if vector_setup == 1:
                # CNT center axis direction
                direction = vector_config.get("direction", "forward")
                if direction == "forward":
                    vector = cnt_axis  # ring1 -> ring2
                else:  # reverse
                    vector = -cnt_axis  # ring2 -> ring1

            elif vector_setup == 2:
                # Connection from CNT center to atom/COM
                direction = vector_config.get("direction", "outward")

                if self.reference_method == "com":
                    # Use center of mass of molecule (weighted by atomic masses)
                    atom_masses = molecule_atoms["Mass"].values.astype(float)
                    total_mass = np.sum(atom_masses)
                    molecule_com = np.sum(corrected_coords * atom_masses[:, np.newaxis], axis=0) / total_mass
                    # Project COM onto plane perpendicular to CNT axis
                    com_radial = molecule_com - cnt_center
                    com_radial_proj = com_radial - np.dot(com_radial, cnt_axis) * cnt_axis

                    if direction == "outward":
                        vector = com_radial_proj  # center -> atom
                    else:  # inward
                        vector = -com_radial_proj  # atom -> center
                else:
                    # Use specific atom as reference
                    if len(corrected_coords) > self.reference_atom_idx:
                        ref_atom = corrected_coords[self.reference_atom_idx]
                        # Project atom onto plane perpendicular to CNT axis
                        atom_radial = ref_atom - cnt_center
                        atom_radial_proj = atom_radial - np.dot(atom_radial, cnt_axis) * cnt_axis

                        if direction == "outward":
                            vector = atom_radial_proj  # center -> atom
                        else:  # inward
                            vector = -atom_radial_proj  # atom -> center
                    else:
                        return None

            elif vector_setup == 3:
                # Connection between two atoms - use manual selection
                atom1_idx = vector_config.get("atom1_idx", 0)
                atom2_idx = vector_config.get("atom2_idx", 1)

                if len(corrected_coords) > max(atom1_idx, atom2_idx):
                    vector = corrected_coords[atom2_idx] - corrected_coords[atom1_idx]

                    # Debug: Check bond length for sanity
                    bond_length = np.linalg.norm(vector)
                    if bond_length > 5.0:  # Suspiciously long bond (likely PBC issue)
                        if not hasattr(self, "_debug_bond_count"):
                            self._debug_bond_count = 0
                        if self._debug_bond_count < 10:  # Limit debug output
                            ddict.printLog(f"Debug: Suspicious bond length {bond_length:.2f} Å detected")
                            ddict.printLog(f"  Atom {atom1_idx} -> Atom {atom2_idx}")
                            ddict.printLog(
                                f"  Corrected coords: {corrected_coords[atom1_idx]} -> {corrected_coords[atom2_idx]}"
                            )
                            self._debug_bond_count += 1
                        return None  # Skip this molecule if bond is too long
                else:
                    return None

            elif vector_setup == 4:
                # Mean of bond vectors - use manual selection
                central_idx = vector_config.get("central_idx", 0)
                bonded_indices = vector_config.get("bonded_indices", list(range(1, len(corrected_coords))))

                if len(corrected_coords) > central_idx and all(idx < len(corrected_coords) for idx in bonded_indices):
                    central_atom = corrected_coords[central_idx]

                    # Calculate bond vectors from central atom to bonded atoms
                    bond_vectors = corrected_coords[bonded_indices] - central_atom

                    # Debug: Check bond lengths for sanity
                    bond_lengths = np.linalg.norm(bond_vectors, axis=1)
                    max_bond_length = np.max(bond_lengths)
                    # Suspiciously long bond if length > 5.0 Å
                    if max_bond_length > 5.0:
                        if not hasattr(self, "_debug_bond_count"):
                            self._debug_bond_count = 0
                        if self._debug_bond_count < 10:
                            ddict.printLog(f"Debug: Suspicious bond length {max_bond_length:.2f} Å in mean vector")
                            ddict.printLog(f"  Bond lengths: {bond_lengths}")
                            ddict.printLog(f"  Central atom: {central_atom}")
                            ddict.printLog(f"  Bonded atoms: {corrected_coords[bonded_indices]}")
                            self._debug_bond_count += 1
                        # skip this molecule if any bond is too long
                        return None

                    # Check for zero-length bond vectors before normalization,
                    # Skip if any bond is too short
                    if np.any(bond_lengths < 1e-6):
                        return None

                    # Calculate the bisector vector (mean of normalized bond vectors)
                    # Normalize each bond vector
                    unit_bond_vectors = bond_vectors / bond_lengths[:, np.newaxis]
                    # Average the unit vectors
                    bisector_vector = np.mean(unit_bond_vectors, axis=0)

                    # Check if the unit vectors cancel each other out (opposing directions)
                    bisector_magnitude = np.linalg.norm(bisector_vector)

                    # Skip if bond vectors cancel out (e.g., linear molecule)
                    if bisector_magnitude < 1e-6:
                        return None

                    # The bisector is already close to unit length, but normalize for consistency
                    vector = bisector_vector / bisector_magnitude
                else:
                    return None
            elif vector_setup == 5:
                # In the molecule_atoms df, the atom information for the given molecule is stored.
                # the x,y and z components of the dipole vector are are stored in the columns "Dipole_X",
                # "Dipole_Y" and "Dipole_Z".
                # The molecule dipole vector is stored in a line of one atom. The entries for all other atoms are 0.
                if (
                    "Dipole_X" in molecule_atoms.columns
                    and "Dipole_Y" in molecule_atoms.columns
                    and "Dipole_Z" in molecule_atoms.columns
                ):
                    # drop all rows where the dipole vector is 0
                    dipole_row = molecule_atoms[
                        (molecule_atoms["Dipole_X"] != 0)
                        | (molecule_atoms["Dipole_Y"] != 0)
                        | (molecule_atoms["Dipole_Z"] != 0)
                    ]
                    dipole_vector = dipole_row[["Dipole_X", "Dipole_Y", "Dipole_Z"]].values.astype(float)
                    if len(dipole_vector) == 1:
                        vector = dipole_vector[0]
                    else:
                        ddict.printLog("Error: More than one dipole vector found for the molecule, skipping.")
                        return None
            else:
                return None

            vectors.append(vector)

        return vectors if len(vectors) == 2 else None

    def flex_angle_processing(self):
        """
        Process the angle data after all frames have been analyzed.
        Calculate angle distributions and generate visualizations.
        """
        ddict.printLog("\nPost-processing angle data...")
        ddict.printLog(f"Processed {self.proc_frame_counter} frames in total.")

        # Process each CNT separately
        for cnt_id in self.angle_data.keys():
            ddict.printLog(f"\nProcessing angle data for CNT {cnt_id}...")

            angle_records = self.angle_data[cnt_id]["angles"]

            if not angle_records:
                ddict.printLog(f"No angle data found for CNT {cnt_id}")
                continue

            # Convert to DataFrame for easier processing
            df = pd.DataFrame(angle_records)

            ddict.printLog(f"  Total angles: {len(df)}")

            # near_zero_angles = df[df['angle'] < 5.0]
            # near_180_angles = df[df['angle'] > 175.0]
            # ddict.printLog(f"  Angles < 5°: {len(near_zero_angles)} ({100*len(near_zero_angles)/len(df):.1f}%)")
            # ddict.printLog(f"  Angles > 175°: {len(near_180_angles)} ({100*len(near_180_angles)/len(df):.1f}%)")

            ddict.printLog(f"  Min angle: {df['angle'].min():.2f}°")
            ddict.printLog(f"  Max angle: {df['angle'].max():.2f}°")

            # Create angle distribution for each radial bin
            num_radial_bins = len(self.cnts_bin_edges[cnt_id]) - 1
            bin_edges = self.cnts_bin_edges[cnt_id]

            # Create results dataframe
            results_data = []

            for radial_bin in range(num_radial_bins):
                bin_data = df[df["radial_bin"] == radial_bin]

                if len(bin_data) > 0:
                    # Calculate angle histogram
                    hist, _ = np.histogram(bin_data["angle"], bins=self.angle_bins)

                    # Normalize by number of frames and bin volume
                    bin_volume = np.pi * (bin_edges[radial_bin + 1] ** 2 - bin_edges[radial_bin] ** 2)
                    normalized_hist = hist / (self.proc_frame_counter * bin_volume)

                    # Store results
                    for angle_bin_idx, (angle_start, angle_end, count) in enumerate(
                        zip(self.angle_bins[:-1], self.angle_bins[1:], normalized_hist)
                    ):
                        results_data.append(
                            {
                                "CNT_ID": cnt_id,
                                "Radial_Bin": radial_bin,
                                "Radial_Range": f"{bin_edges[radial_bin]:.2f}-{bin_edges[radial_bin + 1]:.2f}",
                                "Angle_Bin": angle_bin_idx,
                                "Angle_Range": f"{angle_start:.1f}-{angle_end:.1f}",
                                "Angle_Center": (angle_start + angle_end) / 2,
                                "Count_Density": count,
                                "Raw_Count": hist[angle_bin_idx],
                            }
                        )

            # Create results DataFrame
            results_df = pd.DataFrame(results_data)

            # Save results
            results_df.to_csv(f"CNT_{cnt_id}_angle_distribution.csv", index=False)

            # Save raw angle data
            df.to_csv(f"CNT_{cnt_id}_raw_angles.csv", index=False)

            # Generate plots if requested
            plot_data = ddict.get_input(
                f"Do you want to plot the angle distribution for CNT {cnt_id}? (y/n) ", self.traj_file.args, "string"
            )

            if plot_data.lower() == "y":
                self.plot_angle_distribution(cnt_id, results_df, df)

    def plot_angle_distribution(self, cnt_id, results_df, raw_df):
        """
        Create visualizations for the angle distribution data with heatmap and marginal distributions.
        """
        bin_edges = self.cnts_bin_edges[cnt_id]

        # Create figure with custom layout for marginal plots
        fig = plt.figure(figsize=(11, 12))
        fig.suptitle(f"Angle Distribution in CNT {cnt_id}", fontsize=24, y=0.98)

        # Define the layout: left margin plot, main plots, and space for colorbar
        gs = fig.add_gridspec(
            3,
            3,
            width_ratios=[1, 4, 0.2],
            height_ratios=[1, 4, 1.5],
            hspace=0.08,
            wspace=0.1,
            left=0.12,
            bottom=0.08,
            right=0.88,
            top=0.90,
        )

        ax_bottom = fig.add_subplot(gs[2, 1])  # Bottom: average angle vs radial position
        # Main heatmap (center)
        ax_main = fig.add_subplot(gs[1, 1], sharex=ax_bottom)

        # Marginal distributions
        ax_top = fig.add_subplot(gs[0, 1], sharex=ax_bottom)  # Top: radial distribution
        ax_left = fig.add_subplot(gs[1, 0], sharey=ax_main)  # Left: angle distribution

        # Create 2D histogram for the main heatmap
        hist2d, xedges, yedges = np.histogram2d(
            raw_df["radial_position"], raw_df["angle"], bins=[bin_edges, self.angle_bins]
        )

        # Calculate radial bin centers for consistent x-axis
        radial_bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Create a masked array to show white background for zero values
        hist2d_masked = np.ma.masked_where(hist2d == 0, hist2d)

        # Main heatmap
        im = ax_main.imshow(
            hist2d_masked.T,
            origin="lower",
            aspect="auto",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap="viridis",
            interpolation="nearest",
        )

        ax_main.tick_params(labelbottom=False, labelleft=False, labelsize=16)  # Show y-axis labels, hide x-axis labels
        ax_main.grid(True, alpha=0.9, color="gray", linewidth=0.5)

        # Set x-axis limits to match the data range properly
        x_min, x_max = radial_bin_centers.min(), radial_bin_centers.max()
        ax_main.set_xlim(x_min, x_max)

        # Top plot: Radial position distribution (histogram along x-axis)
        radial_hist, _ = np.histogram(raw_df["radial_position"], bins=bin_edges)
        ax_top.bar(
            radial_bin_centers, radial_hist, width=np.diff(bin_edges), alpha=0.7, edgecolor="black", color="skyblue"
        )
        ax_top.set_ylabel("Count", fontsize=19)
        ax_top.set_xlabel(r"$r_{rad}$ / Å", fontsize=19)
        ax_top.tick_params(
            labelbottom=False, labelsize=16, top=True, labeltop=True
        )  # Show both bottom and top x-labels
        ax_top.xaxis.set_label_position("top")  # Put x-label at top
        ax_top.grid(True, alpha=0.9, color="gray", linewidth=0.5)

        # Left plot: Angle distribution (histogram along y-axis, positive values from right to left)
        angle_hist, _ = np.histogram(raw_df["angle"], bins=self.angle_bins)
        angle_bin_centers = (self.angle_bins[:-1] + self.angle_bins[1:]) / 2
        ax_left.barh(
            angle_bin_centers,
            angle_hist,
            height=np.diff(self.angle_bins),
            alpha=0.7,
            edgecolor="black",
            color="lightcoral",
        )
        ax_left.set_xlabel("Count", fontsize=19)
        ax_left.set_ylabel(r"$\theta$ / °", fontsize=19)
        ax_left.tick_params(labelsize=16)
        ax_left.grid(True, alpha=0.9, color="gray", linewidth=0.5)
        ax_left.invert_xaxis()  # Invert to show values from right to left

        # Bottom plot: Average angle by radial bin
        num_radial_bins = len(bin_edges) - 1
        avg_angles = []
        std_angles = []

        for radial_bin in range(num_radial_bins):
            bin_data = raw_df[raw_df["radial_bin"] == radial_bin]
            if len(bin_data) > 0:
                avg_angles.append(bin_data["angle"].mean())
                std_angles.append(bin_data["angle"].std())
            else:
                avg_angles.append(np.nan)
                std_angles.append(np.nan)

        ax_bottom.errorbar(
            radial_bin_centers,
            avg_angles,
            yerr=std_angles,
            fmt="o-",
            capsize=2,
            capthick=1,
            color="darkblue",
            markersize=5,
            linewidth=2,
        )
        ax_bottom.set_xlabel(r"$r_{rad}$ / Å", fontsize=19)
        ax_bottom.set_ylabel(r"$\langle\theta\rangle$ / °", fontsize=19)
        ax_bottom.tick_params(labelsize=16)
        ax_bottom.grid(True, alpha=0.9, color="gray", linewidth=0.5)

        # Ensure all plots have the same x-axis limits
        ax_bottom.set_xlim(x_min, x_max)
        ax_top.set_xlim(x_min, x_max)

        # Add colorbar in the third column
        cax = fig.add_subplot(gs[1, 2])  # Use the middle row of the third column for colorbar
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label("Count", fontsize=19)
        cbar.ax.tick_params(labelsize=16)

        # Save the figure (layout is already handled by gridspec parameters)
        plt.savefig(f"CNT_{cnt_id}_angle_analysis.png", dpi=300, bbox_inches="tight")
        ddict.printLog(f"Angle analysis plots saved as CNT_{cnt_id}_angle_analysis.png")

    def verify_molecular_integrity(self, corrected_coords, mol_coords, molecule_id=None):
        """
        Verify that the PBC-corrected coordinates maintain reasonable molecular geometry.
        This is a debug/verification method.
        """
        if len(corrected_coords) < 2:
            return True

        # Check all pairwise distances in the molecule
        max_reasonable_bond = 3.0  # Angstroms - adjust based on your system

        for i in range(len(corrected_coords)):
            for j in range(i + 1, len(corrected_coords)):
                distance = np.linalg.norm(corrected_coords[i] - corrected_coords[j])
                if distance > max_reasonable_bond:
                    if not hasattr(self, "_integrity_check_count"):
                        self._integrity_check_count = 0
                    if self._integrity_check_count < 5:  # Limit output
                        ddict.printLog(f"Warning: Large intramolecular distance {distance:.2f} Å")
                        ddict.printLog(f"  Molecule {molecule_id}, atoms {i}-{j}")
                        ddict.printLog(f"  Original: {mol_coords[i]} - {mol_coords[j]}")
                        ddict.printLog(f"  Corrected: {corrected_coords[i]} - {corrected_coords[j]}")
                        self._integrity_check_count += 1
                    return False
        return True
