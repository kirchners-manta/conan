import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import conan.analysis_modules.traj_an as traj_an
import conan.analysis_modules.utils as ut
import conan.defdict as ddict


def radial_velocity_analysis(traj_file, molecules, an):
    va = VelocityAnalysis(traj_file, molecules)
    va.velocity_prep()
    traj_an.process_trajectory(traj_file, molecules, an, va)
    va.post_processing()
    sys.exit(0)


class VelocityAnalysis:
    def __init__(self, traj_file, molecules):
        self.traj_file = traj_file
        self.molecules = molecules
        self.args = traj_file.args
        self.old_frame = None  # Initialize as None
        self.rad_increment = None
        self.velocity_bin_edges = None
        self.velocity_bin_labels = None
        self.velocity_df = None
        self.num_increments = None
        self.dt = None
        self.velocity_bin_counter = None
        self.proc_frames = 0
        self.radius_tube = self.molecules.tuberadii[0]

    def velocity_prep(self):
        ddict.printLog("")
        if len(self.molecules.CNT_centers) > 1:
            ddict.printLog("-> Multiple CNTs detected. The analysis will be conducted on the first CNT.\n", color="red")
        if len(self.molecules.CNT_centers) == 0:
            ddict.printLog("-> No CNTs detected. Aborting...\n", color="red")
            sys.exit(1)

        for i in range(len(self.molecules.CNT_centers)):
            ddict.printLog(f"\n-> CNT{i + 1}")
            self.num_increments = int(
                ddict.get_input(
                    "How many increments do you want to use to calculate the velocity profile? ", self.args, "int"
                )
            )
            self.rad_increment = self.molecules.tuberadii[i] / self.num_increments
            self.velocity_bin_edges = np.linspace(0, self.molecules.tuberadii[0], self.num_increments + 1)
            self.velocity_bin_labels = np.arange(self.num_increments)  # Zero-based indexing
            ddict.printLog("Increment distance: %0.3f angstrom" % (self.rad_increment))

        self.dt = ddict.get_input("What is the time step in the trajectory? [fs]  ", self.args, "float")
        self.initialize_data_frames()

    def initialize_data_frames(self):
        # Initialize the velocity_df and velocity_bin_counter
        self.velocity_df = pd.DataFrame({"Frame": np.arange(1, self.traj_file.number_of_frames + 1)})
        for i in range(self.num_increments):
            self.velocity_df["Bin %d" % (i + 1)] = 0.0  # Initialize with float zeros
        # Set 'Frame' as the index
        self.velocity_df.set_index("Frame", inplace=True)
        self.velocity_bin_counter = pd.DataFrame({"Bin": self.velocity_bin_labels, "Total count": 0})

    def calculate_velocity(self, split_frame):
        # Ensure 'X', 'Y', 'Z' columns are present
        new_frame = ut.wrapping_coordinates(self.traj_file.box_size, split_frame)
        new_frame.rename(columns={"x": "X", "y": "Y", "z": "Z"}, inplace=True)
        new_frame = new_frame.reset_index(drop=True)

        if self.old_frame is None:
            # First frame, velocities are zero
            velocity = np.zeros(len(new_frame))
            # Set old_frame to new_frame for next step
            self.old_frame = new_frame.copy()
        else:
            # Ensure old_frame has matching indices
            self.old_frame = self.old_frame.reset_index(drop=True)

            # Check if number of atoms is the same
            if len(self.old_frame) != len(new_frame):
                raise ValueError("Number of atoms has changed between frames.")

            # Compute displacements with periodic boundary conditions
            displacements = {}
            for i, axis in enumerate(["X", "Y", "Z"]):
                delta = new_frame[axis].astype(float).values - self.old_frame[axis].astype(float).values
                box_size = self.traj_file.box_size[i]
                delta -= box_size * np.round(delta / box_size)
                displacements[axis] = delta

            # Calculate distance and velocity
            distance = np.sqrt(displacements["X"] ** 2 + displacements["Y"] ** 2 + displacements["Z"] ** 2)
            velocity = distance / self.dt

            # Update self.old_frame with the current frame for the next step in trajectory
            self.old_frame = new_frame.copy()

        # Assign velocities back to split_frame
        split_frame = split_frame.reset_index(drop=True)
        split_frame["velocity"] = velocity

        return split_frame  # Return updated split_frame

    def analyze_frame(self, split_frame, frame_counter):
        # Call the calculate_velocity function
        split_frame = self.calculate_velocity(split_frame)
        # Call the radial_velocity_analysis for each frame
        self.radial_velocity_analysis(split_frame, frame_counter)

    def radial_velocity_analysis(self, split_frame, frame_counter):
        split_frame["X"] = split_frame["X"].astype(float) % self.traj_file.box_size[0]
        split_frame["Y"] = split_frame["Y"].astype(float) % self.traj_file.box_size[1]
        split_frame["Z"] = split_frame["Z"].astype(float) % self.traj_file.box_size[2]

        CNT_centers = self.molecules.CNT_centers
        max_z_pore = self.molecules.max_z_pore[0] % self.traj_file.box_size[2]
        min_z_pore = self.molecules.min_z_pore[0] % self.traj_file.box_size[2]

        CNT_centers[0][0] = CNT_centers[0][0] % self.traj_file.box_size[0]
        CNT_centers[0][1] = CNT_centers[0][1] % self.traj_file.box_size[1]

        if min_z_pore > max_z_pore:
            part1 = split_frame[split_frame["Z"].astype(float) >= min_z_pore].copy()
            part2 = split_frame[split_frame["Z"].astype(float) <= max_z_pore].copy()
            split_frame = pd.concat([part1, part2], ignore_index=True)
        else:
            split_frame = split_frame[
                (split_frame["Z"].astype(float) >= min_z_pore) & (split_frame["Z"].astype(float) <= max_z_pore)
            ].copy()

        # Calculate distance from CNT center
        split_frame["X_adjust"] = split_frame["X"].astype(float) - CNT_centers[0][0]
        split_frame["Y_adjust"] = split_frame["Y"].astype(float) - CNT_centers[0][1]
        split_frame["Distance"] = np.sqrt(split_frame["X_adjust"] ** 2 + split_frame["Y_adjust"] ** 2)

        # Use numeric bin indices starting from 0
        split_frame["Distance_bin"] = pd.cut(
            split_frame["Distance"], bins=self.velocity_bin_edges, labels=False, include_lowest=True
        )

        # Group by bins and calculate counts and sums
        velocity_bin_counter_temp = split_frame.groupby("Distance_bin")["velocity"].count().reset_index()
        velocity_df_temp = split_frame.groupby("Distance_bin")["velocity"].sum().reset_index()

        # Update total counts
        for idx, row in velocity_bin_counter_temp.iterrows():
            bin_idx = int(row["Distance_bin"])
            count = row["velocity"]
            self.velocity_bin_counter.loc[self.velocity_bin_counter["Bin"] == bin_idx, "Total count"] += count

        for idx, row in velocity_df_temp.iterrows():
            bin_idx = int(row["Distance_bin"])
            bin_label = "Bin %d" % (bin_idx + 1)  # Adjust for one-based indexing in column names
            bin_sum = row["velocity"]
            self.velocity_df.loc[frame_counter, bin_label] += bin_sum

    def post_processing(self):
        results_vd_df = pd.DataFrame()
        for i in range(self.num_increments):
            bin_label = "Bin %d" % (i + 1)
            bin_sum = self.velocity_df[bin_label].sum()
            bin_count = self.velocity_bin_counter.loc[self.velocity_bin_counter["Bin"] == i, "Total count"].values[0]

            average_velocity = bin_sum / bin_count if bin_count != 0 else 0
            results_vd_df.loc[i, "Average Velocity"] = average_velocity

        results_vd_df["Bin_lowedge"] = self.velocity_bin_edges[:-1]
        results_vd_df["Bin_highedge"] = self.velocity_bin_edges[1:]
        results_vd_df["Bin_center"] = (self.velocity_bin_edges[1:] + self.velocity_bin_edges[:-1]) / 2
        results_vd_df.reset_index(drop=True, inplace=True)
        results_vd_df.insert(0, "Bin", results_vd_df.index + 1)

        self.plot_results(results_vd_df)

    def plot_results(self, results_vd_df):
        plot_data = ddict.get_input("Do you want to plot the data? (y/n) ", self.args, "string")
        if plot_data.lower() == "y":
            # Normalization and mirroring options
            normalize = ddict.get_input(
                "Do you want to normalize the increments with respect to the CNT's radius? (y/n) ", self.args, "string"
            )
            mirror = ddict.get_input("Do you want to mirror the plot? (y/n) ", self.args, "string")

            if normalize.lower() == "y":
                results_vd_df["Bin_center"] = results_vd_df["Bin_center"] / self.radius_tube

            if mirror.lower() == "y":
                results_vd_dummy = results_vd_df.copy()
                results_vd_dummy["Bin_center"] *= -1
                results_vd_dummy.sort_values(by=["Bin_center"], inplace=True)
                results_vd_df = pd.concat([results_vd_dummy, results_vd_df], ignore_index=True)

            # Adjust the x-label based on normalization
            xlabel = r"Distance from tube center / $\mathrm{\AA}$"
            if normalize.lower() == "y":
                xlabel = r"Normalized distance from tube center ($d_{rad}/r_{CNT}$)"

            # Plotting
            fig, ax = plt.subplots()
            ax.plot(
                results_vd_df["Bin_center"],
                results_vd_df["Average Velocity"],
                "-",
                label="Radial velocity profile",
                color="black",
            )
            ax.set(
                xlabel=xlabel,
                ylabel=r"Velocity / $\mathrm{\AA/fs}$",
                title="Radial Velocity Profile",
            )
            ax.grid()
            fig.savefig("Radial_velocity_profile.pdf")
            ddict.printLog("-> Radial velocity profile saved as Radial_velocity_profile.pdf\n")

            # Save the processed data
            results_vd_df.to_csv("Radial_velocity_profile.csv", sep=";", index=False, header=True, float_format="%.5f")

        # Save raw data
        save_raw = ddict.get_input("Do you want to save the raw data? (y/n) ", self.args, "string")
        if save_raw.lower() == "y":
            self.velocity_df.to_csv("Radial_velocity_raw.csv", sep=";", index=False, header=True, float_format="%.5f")
            ddict.printLog("-> Raw velocity data saved as Radial_velocity_raw.csv\n")
