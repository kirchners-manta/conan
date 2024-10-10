import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import conan.analysis_modules.traj_an as traj_an
import conan.analysis_modules.utils as ut
import conan.defdict as ddict


def msd_analysis(traj_file, molecules, an):
    msd_analyzer = MSDAnalysis(traj_file, molecules)
    msd_analyzer.msd_prep()
    traj_an.process_trajectory(traj_file, molecules, an, msd_analyzer)
    msd_analyzer.msd_processing()


class MSDAnalysis:
    def __init__(self, traj_file, molecules):
        self.traj_file = traj_file
        self.molecules = molecules
        self.number_of_frames = traj_file.number_of_frames
        self.box_size = np.array(traj_file.box_size, dtype=float)
        self.first_frame = traj_file.frame0.copy()
        self.displacements = {}
        self.unwrapped_positions = {}
        self.unwrapped_positions_current = {}
        self.initial_positions = {}
        self.previous_positions = {}
        self.molecule_indices = {}
        self.num_liq_species_dict = {}
        self.species_mol_numbers = {}
        self.counter = 0
        self.args = traj_file.args
        self.dt = None  # Time step between frames
        self.analyzed_frame_indices = []  # List to keep track of analyzed frames

    def msd_prep(self):
        # Rename columns for consistency
        self.first_frame.rename(columns={"x": "X", "y": "Y", "z": "Z"}, inplace=True)

        # Map masses to atoms
        self.first_frame["Mass"] = self.first_frame["Element"].map(ddict.dict_mass())

        self.dt = ddict.get_input("What is the time step in the trajectory? [fs]  ", self.args, "float")

        # Identify liquid species
        num_liq_species = self.first_frame[self.first_frame["Struc"] == "Liquid"]["Species"].unique()

        for species in num_liq_species:
            # Get unique molecule numbers for the species
            molecule_numbers = self.first_frame[
                (self.first_frame["Struc"] == "Liquid") & (self.first_frame["Species"] == species)
            ]["Molecule"].unique()

            num_molecules = len(molecule_numbers)
            self.num_liq_species_dict[species] = num_molecules
            self.species_mol_numbers[species] = molecule_numbers

            # Map molecule indices for efficient lookup
            self.molecule_indices[species] = {molecule: i for i, molecule in enumerate(molecule_numbers)}

            # Initialize displacement arrays
            displacements = np.zeros((num_molecules, self.number_of_frames, 3))
            self.displacements[species] = displacements

            # Initialize unwrapped positions and previous positions
            self.unwrapped_positions[species] = []
            self.unwrapped_positions_current[species] = np.zeros((num_molecules, 3))
            self.initial_positions[species] = np.zeros((num_molecules, 3))
            self.previous_positions[species] = np.zeros((num_molecules, 3))

            ddict.printLog(f"Initialized displacement array for species '{species}' with shape {displacements.shape}")

        # Calculate initial COM for each molecule
        COM_frame_initial = self.calculate_COM_frame(self.first_frame)

        for species in self.num_liq_species_dict:
            molecule_indices = self.molecule_indices[species]
            for molecule, idx in molecule_indices.items():
                com_initial = COM_frame_initial[
                    (COM_frame_initial["Species"] == species) & (COM_frame_initial["Molecule"] == molecule)
                ][["X", "Y", "Z"]].values[0]
                self.initial_positions[species][idx, :] = com_initial
                self.unwrapped_positions_current[species][idx, :] = com_initial
                self.previous_positions[species][idx, :] = com_initial

            # Append initial unwrapped positions to the list
            # self.unwrapped_positions[species].append(self.unwrapped_positions_current[species].copy())

    def analyze_frame(self, split_frame, frame_counter):
        self.counter = frame_counter
        # Ensure data types are correct
        split_frame["X"] = split_frame["X"].astype(float)
        split_frame["Y"] = split_frame["Y"].astype(float)
        split_frame["Z"] = split_frame["Z"].astype(float)
        split_frame["Mass"] = split_frame["Mass"].astype(float)

        # Calculate current COM for each molecule
        COM_frame_current = self.calculate_COM_frame(split_frame)

        # Calculate displacements
        self.calculate_displacements(COM_frame_current)

        # Keep track of analyzed frames
        self.analyzed_frame_indices.append(frame_counter - 1)

        for species in self.num_liq_species_dict:
            self.unwrapped_positions[species].append(self.unwrapped_positions_current[species].copy())

    def calculate_COM_frame(self, frame):
        # Group by Species and Molecule to calculate COM
        grouped = frame.groupby(["Species", "Molecule"])
        com_list = []
        for (species, molecule), group in grouped:
            com = ut.calculate_com(group, self.box_size)
            com_list.append({"Species": species, "Molecule": molecule, "X": com[0], "Y": com[1], "Z": com[2]})
        COM_frame = pd.DataFrame(com_list)
        return COM_frame

    def calculate_displacements(self, COM_frame_current):
        for species in self.num_liq_species_dict:
            # Create mappings for efficient access
            curr_com_dict = COM_frame_current[COM_frame_current["Species"] == species].set_index("Molecule")

            molecule_indices = self.molecule_indices[species]
            for molecule, idx in molecule_indices.items():
                if molecule in curr_com_dict.index:
                    com_current = curr_com_dict.loc[molecule, ["X", "Y", "Z"]].values

                    # Get previous wrapped position
                    com_prev = self.previous_positions[species][idx, :]

                    # Compute displacement between current and previous positions
                    delta = com_current - com_prev

                    # Unwrap displacement
                    delta -= self.box_size * np.round(delta / self.box_size)

                    # Update unwrapped positions
                    self.unwrapped_positions_current[species][idx, :] += delta

                    # Update previous positions
                    self.previous_positions[species][idx, :] = com_current

                else:
                    ddict.printLog(f"Warning: Molecule {molecule} of species {species} not found in current COM frame.")

    def msd_processing(self):
        # Ask user whether to calculate MSD or RMSD
        msd_or_rmsd = int(ddict.get_input("Do you want to calculate the [1] MSD or the [2] RMSD?  ", self.args, "int"))

        if msd_or_rmsd == 1:
            self.calculate_msd()
        elif msd_or_rmsd == 2:
            self.calculate_rmsd()
        else:
            ddict.printLog("Invalid input. Please enter 1 for MSD or 2 for RMSD.")
            return

    def calculate_msd(self):
        ddict.printLog("Calculating MSD values for each species...")
        for species in self.num_liq_species_dict:
            # Convert list of unwrapped positions to numpy array
            unwrapped_positions_list = self.unwrapped_positions[species]
            unwrapped_positions_array = np.array(unwrapped_positions_list)
            # Shape: (N_frames, num_molecules, 3)
            # Transpose to shape (num_molecules, N_frames, 3)
            unwrapped_positions_array = np.transpose(unwrapped_positions_array, (1, 0, 2))
            num_analyzed_frames = unwrapped_positions_array.shape[1]
            dt = float(self.dt)

            max_tau = num_analyzed_frames

            msd_tau = []
            msd_x_tau = []
            msd_y_tau = []
            msd_z_tau = []
            tau_times = []

            for tau_idx in range(0, max_tau):  # Start from tau_idx = 1 to exclude tau = 0
                tau = tau_idx * dt
                tau_times.append(tau)

                # Initialize accumulators for squared displacements
                sq_disp_accum = []
                sq_disp_x_accum = []
                sq_disp_y_accum = []
                sq_disp_z_accum = []

                # Loop over all possible time origins
                num_time_origins = num_analyzed_frames - tau_idx
                for start_idx in range(num_time_origins):
                    pos_t0 = unwrapped_positions_array[:, start_idx, :]
                    pos_t1 = unwrapped_positions_array[:, start_idx + tau_idx, :]
                    disp = pos_t1 - pos_t0
                    sq_disp = np.square(disp)

                    # Accumulate squared displacements
                    sq_disp_accum.append(np.sum(sq_disp, axis=1))  # Shape: (num_molecules,)
                    sq_disp_x_accum.append(sq_disp[:, 0])
                    sq_disp_y_accum.append(sq_disp[:, 1])
                    sq_disp_z_accum.append(sq_disp[:, 2])

                # Concatenate and average over all molecules and time origins
                sq_disp_accum = np.concatenate(sq_disp_accum)
                sq_disp_x_accum = np.concatenate(sq_disp_x_accum)
                sq_disp_y_accum = np.concatenate(sq_disp_y_accum)
                sq_disp_z_accum = np.concatenate(sq_disp_z_accum)

                msd_tau.append(np.mean(sq_disp_accum))
                msd_x_tau.append(np.mean(sq_disp_x_accum))
                msd_y_tau.append(np.mean(sq_disp_y_accum))
                msd_z_tau.append(np.mean(sq_disp_z_accum))

            # Now we have msd_tau, msd_x_tau, msd_y_tau, msd_z_tau, and tau_times
            self.plot_msd(msd_tau, msd_x_tau, msd_y_tau, msd_z_tau, tau_times, species, "MSD")

    def calculate_rmsd(self):
        ddict.printLog("Calculating RMSD values for each species...")
        num_analyzed_frames = len(self.analyzed_frame_indices)
        for species in self.num_liq_species_dict:
            displacements = self.displacements[species]

            rmsd = np.zeros(num_analyzed_frames)
            rmsd_x = np.zeros(num_analyzed_frames)
            rmsd_y = np.zeros(num_analyzed_frames)
            rmsd_z = np.zeros(num_analyzed_frames)

            time = np.array(self.analyzed_frame_indices) * float(self.dt)

            for idx, dt in enumerate(self.analyzed_frame_indices):
                disp = displacements[:, dt, :]
                sq_disp = np.square(disp)
                rms_disp = np.sqrt(np.sum(sq_disp, axis=1))
                rmsd[idx] = np.mean(rms_disp)
                rmsd_x[idx] = np.mean(np.sqrt(sq_disp[:, 0]))
                rmsd_y[idx] = np.mean(np.sqrt(sq_disp[:, 1]))
                rmsd_z[idx] = np.mean(np.sqrt(sq_disp[:, 2]))

            self.plot_msd(rmsd, rmsd_x, rmsd_y, rmsd_z, time, species, "RMSD")

    def plot_msd(self, msd, msd_x, msd_y, msd_z, time_lags, species, label):
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        axs[0, 0].plot(time_lags, msd, label=f"{species}")
        axs[0, 1].plot(time_lags, msd_x, label=f"{species} X")
        axs[1, 0].plot(time_lags, msd_y, label=f"{species} Y")
        axs[1, 1].plot(time_lags, msd_z, label=f"{species} Z")

        axs[0, 0].set(xlabel="Time lag [fs]", ylabel=label, title=f"Overall {label} for {species}")
        axs[0, 1].set(xlabel="Time lag [fs]", ylabel=label, title=f"{label} in X for {species}")
        axs[1, 0].set(xlabel="Time lag [fs]", ylabel=label, title=f"{label} in Y for {species}")
        axs[1, 1].set(xlabel="Time lag [fs]", ylabel=label, title=f"{label} in Z for {species}")

        for ax in axs.flat:
            ax.legend()
            ax.grid()

        plt.tight_layout()
        plt.savefig(f"{label.lower()}_{species}.png")
        plt.close(fig)

        # Save data to CSV
        data = {
            "Time lag [fs]": time_lags,
            label: msd,
            f"{label} X": msd_x,
            f"{label} Y": msd_y,
            f"{label} Z": msd_z,
        }
        df = pd.DataFrame(data)
        df.to_csv(f"{label.lower()}_{species}.csv", index=False)

        ddict.printLog(f"{label} data and plot for species '{species}' saved.")
