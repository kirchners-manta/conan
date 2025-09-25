import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import conan.analysis_modules.traj_an as traj_an
import conan.defdict as ddict


def radial_density_analysis(traj_file, molecules, an):
    rda = RadialDensityAnalysis(traj_file, molecules, an)
    rda.raddens_prep()
    traj_an.process_trajectory(traj_file, molecules, an, rda)
    rda.raddens_post_processing()

    sys.exit(0)


class RadialDensityAnalysis:
    def __init__(self, traj_file, molecules, an):
        self.traj_file = traj_file
        self.molecules = molecules
        self.args = traj_file.args
        self.frame0 = traj_file.frame0
        self.analysis_choice2 = an.choice2
        self.number_of_frames = traj_file.number_of_frames
        self.tuberadii = molecules.tuberadii
        self.length_pore = molecules.length_pore
        self.CNT_centers = molecules.CNT_centers
        self.analysis_function = self.get_analysis_function()

        self.rad_increment = None
        self.raddens_bin_edges = None
        self.raddens_bin_labels = None
        self.raddens_df = None
        self.num_increments = None
        self.atom_charges = None
        self.proc_frames = 0

    def raddens_prep(self):
        ddict.printLog("")
        if len(self.CNT_centers) > 1:
            ddict.printLog("-> Multiple CNTs detected. The analysis will be conducted on the first CNT.\n", color="red")
        if len(self.molecules.structure_data["CNT_centers"]) == 0:
            ddict.printLog("-> No CNTs detected. Aborting...\n", color="red")
            sys.exit(1)

        for i in range(len(self.CNT_centers)):
            ddict.printLog(f"\n-> CNT{i + 1}")
            self.num_increments = int(
                ddict.get_input(
                    "How many increments do you want to use to calculate the density profile? ", self.args, "int"
                )
            )
            self.rad_increment = self.tuberadii[i] / self.num_increments
            self.raddens_bin_edges = np.linspace(0, self.tuberadii[i], self.num_increments + 1)
            self.raddens_bin_labels = np.arange(1, len(self.raddens_bin_edges), 1)
            ddict.printLog("Increment distance: %0.3f angstrom" % (self.rad_increment))

        data = {"Frame": np.arange(1, self.number_of_frames + 1)}
        # Add columns for each bin
        for i in range(self.num_increments):
            data["Bin %d" % (i + 1)] = np.zeros(self.number_of_frames)

        # Create the DataFrame from the dictionary
        self.raddens_df = pd.DataFrame(data)

    def analyze_frame(self, split_frame, frame_counter):
        self.analysis_function(split_frame, frame_counter)

    def get_analysis_function(self):
        if self.analysis_choice2 == 1:
            return self.radial_density_analysis
        elif self.analysis_choice2 == 2:
            return self.radial_charge_density_analysis
        else:
            raise ValueError("Invalid analysis choice. Choose between 1 and 2.")

    def radial_analysis(self, property_name, split_frame, counter):

        raddens_df = self.raddens_df
        raddens_bin_edges = self.raddens_bin_edges
        raddens_bin_labels = self.raddens_bin_labels
        num_increments = self.num_increments
        CNT_centers = self.molecules.CNT_centers
        max_z_pore = self.molecules.max_z_pore
        min_z_pore = self.molecules.min_z_pore

        # Modulo operation for periodic boundary conditions
        split_frame["X"] = split_frame["X"].astype(float) % self.traj_file.box_size[0]
        split_frame["Y"] = split_frame["Y"].astype(float) % self.traj_file.box_size[1]
        split_frame["Z"] = split_frame["Z"].astype(float) % self.traj_file.box_size[2]

        # Handling z_pore values
        max_z_pore[0] = max_z_pore[0] % self.traj_file.box_size[2]
        min_z_pore[0] = min_z_pore[0] % self.traj_file.box_size[2]

        # Adjust CNT centers
        CNT_centers[0][0] = CNT_centers[0][0] % self.traj_file.box_size[0]
        CNT_centers[0][1] = CNT_centers[0][1] % self.traj_file.box_size[1]

        # Handle splitting over the boundary
        # Handle cases where the pore is split over the periodic boundary
        if min_z_pore[0] > max_z_pore[0]:
            part1 = split_frame[split_frame["Z"].astype(float) >= min_z_pore[0]].copy()
            part2 = split_frame[split_frame["Z"].astype(float) <= max_z_pore[0]].copy()
            split_frame = pd.concat([part1, part2])
        else:
            split_frame = split_frame[split_frame["Z"].astype(float) <= max_z_pore[0]].copy()
            split_frame = split_frame[split_frame["Z"].astype(float) >= min_z_pore[0]].copy()

        # Calculate distance from CNT center
        split_frame["X_adjust"] = split_frame["X"].astype(float) - CNT_centers[0][0]
        split_frame["Y_adjust"] = split_frame["Y"].astype(float) - CNT_centers[0][1]
        split_frame["Distance"] = np.sqrt(split_frame["X_adjust"] ** 2 + split_frame["Y_adjust"] ** 2)

        # Bin the distances
        split_frame["Distance_bin"] = pd.cut(split_frame["Distance"], bins=raddens_bin_edges, labels=raddens_bin_labels)

        # Group by bins and sum the property
        raddens_df_temp = (
            split_frame.groupby(pd.cut(split_frame["Distance"], raddens_bin_edges), observed=False)[property_name]
            .sum()
            .reset_index(name="Weighted_counts")
        )

        # Add the results to the dataframe
        for i in range(num_increments):
            raddens_df.loc[counter, "Bin %d" % (i + 1)] = raddens_df_temp.loc[i, "Weighted_counts"]

        self.raddens_df = raddens_df

    def radial_density_analysis(self, split_frame, frame_counter):
        self.radial_analysis("Mass", split_frame, frame_counter)

    def radial_charge_density_analysis(self, split_frame, frame_counter):
        self.radial_analysis("Charge", split_frame, frame_counter)

    def raddens_post_processing(self):

        raddens_df = self.raddens_df
        raddens_bin_edges = self.raddens_bin_edges
        number_of_frames = self.proc_frames
        analysis_choice2 = self.analysis_choice2
        args = self.args
        tuberadii = self.tuberadii
        length_pore = self.length_pore

        CNT_length = length_pore[0]
        radius_tube = tuberadii[0]

        # Check the analysis choice. -> If mass or charge density is plotted.
        if analysis_choice2 == 1:
            choice = "Mass"
        elif analysis_choice2 == 2:
            choice = "Charge"
        ddict.printLog("")
        results_rd_df = pd.DataFrame(raddens_df.iloc[:, 1:].sum(axis=0) / number_of_frames)
        results_rd_df.columns = [choice]

        # Add the bin edges to the results_rd_df dataframe.
        results_rd_df["Bin_lowedge"] = raddens_bin_edges[:-1]
        results_rd_df["Bin_highedge"] = raddens_bin_edges[1:]

        # The center of the bin is the average of the bin edges.
        results_rd_df["Bin_center"] = (raddens_bin_edges[1:] + raddens_bin_edges[:-1]) / 2

        # Calculate the Volume of each bin. By setting the length of the CNT as length of a cylinder,
        # and the radius of the bin as the radius of the cylinder.
        # Subtract the volume of the smaller cylinder from the volume of the larger cylinder (next larger bin).
        #  The volume of a cylinder is pi*r^2*h.
        vol_increment = math.pi * (raddens_bin_edges[1:] ** 2 - raddens_bin_edges[:-1] ** 2) * CNT_length
        results_rd_df["Volume"] = vol_increment

        if choice == "Mass":
            # Calculate the density of each bin by dividing the average mass by the volume.
            results_rd_df["Density [u/Ang^3]"] = results_rd_df[choice] / results_rd_df["Volume"]
            # Calculate the density in g/cm^3.
            results_rd_df["Density [g/cm^3]"] = results_rd_df["Density [u/Ang^3]"] * 1.66053907

        if choice == "Charge":
            # Calculate the charge density in e/Ang^3.
            results_rd_df["Charge density [e/Ang^3]"] = results_rd_df[choice] / results_rd_df["Volume"]

        # Reset the index of the dataframe.
        results_rd_df.reset_index(drop=True, inplace=True)
        # Add a new first column with the index+1 of the bin.
        results_rd_df.insert(0, "Bin", results_rd_df.index + 1)

        # Now for the initial raw_data frame raddens_df -> Make new dataframe with the densities for each bin.
        # To do this we divide the mass/charge of each bin (column in raddens_df) by the volume of the bin it is in
        # (row in results_rd_df).
        raddens_df_density = pd.DataFrame()
        for i in range(len(results_rd_df)):
            raddens_df_density[i] = raddens_df.iloc[:, i + 1] / results_rd_df.loc[i, "Volume"]
        raddens_df_density = raddens_df_density.copy()

        # Calculate the variance for all columns in raddens_df_density.
        results_rd_df["Variance"] = pd.DataFrame(raddens_df_density.var(axis=0))
        # Calculate the standard deviation for all columns in raddens_df_density.
        results_rd_df["Standard dev."] = pd.DataFrame(raddens_df_density.std(axis=0))
        # Calculate the standard error for all columns in raddens_df_density.
        results_rd_df["Standard error"] = pd.DataFrame(raddens_df_density.sem(axis=0))

        # Change the column names to the according bins, as in raddens_df and add the frame number.
        raddens_df_density.columns = raddens_df.columns[1:]
        raddens_df_density.insert(0, "Frame", raddens_df["Frame"])

        # Plot the data.
        raddens_data_preparation = ddict.get_input("Do you want to plot the data? (y/n) ", args, "string")
        # Adjusting the resulting dataframe for plotting, as the user set it.
        if raddens_data_preparation == "y":
            results_rd_df_copy = results_rd_df.copy()

            # Normalization of the data.
            normalize = ddict.get_input(
                "Do you want to normalize the increments with respect to the CNTs' radius? (y/n) ", args, "string"
            )
            if normalize == "y":
                results_rd_df["Bin_center"] = results_rd_df["Bin_center"] / radius_tube

            # Mirroring the data.
            mirror = ddict.get_input("Do you want to mirror the plot? (y/n) ", args, "string")
            if mirror == "y":
                # Mirror the data by multiplying the bin center by -1.
                # Then sort the dataframe by the bin center values and
                # combine the dataframes.
                results_rd_dummy = results_rd_df.copy()
                results_rd_dummy["Bin_center"] = results_rd_df["Bin_center"] * (-1)
                results_rd_dummy.sort_values(by=["Bin_center"], inplace=True)
                results_rd_df = pd.concat([results_rd_dummy, results_rd_df], ignore_index=True)

                # Generate the plot.
            fig, ax = plt.subplots()
            if choice == "Mass":
                ax.plot(
                    results_rd_df["Bin_center"],
                    results_rd_df["Density [g/cm^3]"],
                    "-",
                    label="Radial density function",
                    color="black",
                )
                ax.set(
                    xlabel="Distance from tube center [Ang]", ylabel="Density [g/cm^3]", title="Radial density function"
                )

            if choice == "Charge":
                ax.plot(
                    results_rd_df["Bin_center"],
                    results_rd_df["Charge density [e/Ang^3]"],
                    "-",
                    label="Radial density function",
                    color="black",
                )
                ax.set(
                    xlabel="Distance from tube center [Ang]",
                    ylabel="Charge density [e/Ang^3]",
                    title="Radial density function",
                )

            ax.grid()
            fig.savefig("radial_density_function.pdf")
            ddict.printLog("-> Radial density function saved as Radial_density_function.pdf\n")

            # Save the data.
            results_rd_df.to_csv("Radial_density_function.csv", sep=";", index=False, header=True, float_format="%.5f")

            # Radial contour plot.
            results_rd_df = results_rd_df_copy.copy()
            radial_plot = ddict.get_input("Do you also want to create a radial countour plot? (y/n) ", args, "string")
            if radial_plot == "y":
                theta = np.linspace(0, 2 * np.pi, 500)
                if normalize == "n":
                    r = np.linspace(0, radius_tube, len(results_rd_df["Bin_center"]))
                else:
                    r = np.linspace(0, 1, len(results_rd_df["Bin_center"]))

                Theta, R = np.meshgrid(theta, r)

                if choice == "Mass":
                    values = np.tile(results_rd_df["Density [g/cm^3]"].values, (len(theta), 1)).T
                elif choice == "Charge":
                    values = np.tile(results_rd_df["Charge density [e/Ang^3]"].values, (len(theta), 1)).T

                fig = plt.figure()
                ax = fig.add_subplot(projection="polar")
                c = ax.contourf(Theta, R, values, cmap="Reds")

                ax.spines["polar"].set_visible(False)

                # Remove angle labels.
                ax.set_xticklabels([])

                # Set radial gridlines and their labels.
                ax.set_yticks(np.linspace(0, radius_tube if normalize == "n" else 1, 5))
                ax.set_yticklabels(
                    ["{:.2f}".format(x) for x in np.linspace(0, radius_tube if normalize == "n" else 1, 5)]
                )

                # Set the position of radial labels.
                ax.set_rlabel_position(22.5)
                ax.grid(color="black", linestyle="--", alpha=0.5)

                # Set title and add a colorbar.
                plt.title("Radial Density Contour Plot", fontsize=20, pad=20)
                cbar = fig.colorbar(c, ax=ax, pad=0.10, fraction=0.046, orientation="horizontal")

                if choice == "Mass":
                    cbar.set_ticklabels(["{:.2f}".format(x) for x in cbar.get_ticks()])
                    cbar.set_label(r"Mass density $[g/cm^{3}]$", fontsize=15)

                elif choice == "Charge":
                    cbar.set_ticklabels(["{:.3f}".format(x) for x in cbar.get_ticks()])
                    cbar.set_label(r"Charge density $[e/Ang^{3}]$", fontsize=15)

                # Set the y label.
                if normalize == "n":
                    ax.set_ylabel(r"$d_{rad}$", labelpad=10, fontsize=20)
                else:
                    ax.set_ylabel(r"$d_{rad}$/$r_{CNT}$", labelpad=10, fontsize=20)

                # Save the data.
                fig.savefig("Radial_density_function_polar.pdf")
                ddict.printLog("-> Radial density function countour plot saved as Radial_density_function_polar.pdf\n")

        # Add a new column to the raddens_df dataframe with the total mass/charge in each frame.
        # (excluding the frame column)
        raddens_df["Total"] = raddens_df.iloc[:, 1:].sum(axis=1)

        raddens_df.to_csv("Radial_mass_dist_raw.csv", sep=";", index=False, header=True, float_format="%.5f")
        ddict.printLog("Raw mass distribution data saved as Radial_mass_dist_raw.csv")
        raddens_df_density.to_csv("Radial_density_raw.csv", sep=";", index=False, header=True, float_format="%.5f")
        ddict.printLog("Raw density data saved as Radial_density_raw.csv")
