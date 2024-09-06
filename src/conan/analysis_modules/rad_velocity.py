import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import conan.analysis_modules.utils as ut
import conan.defdict as ddict

"""
With this tool the velocity of every atom within the pore at every time step is calculated.
The atom is velocity is then assigned to a radial increment of the CNT.
"""


def velocity_calc_atom(inputdict):

    box_size = inputdict["box_size"]

    old_frame = inputdict["old_frame"]
    new_frame = inputdict["split_frame"]
    new_frame = ut.wrapping_coordinates(box_size, new_frame)

    # combine the dataframes
    all_coords = pd.concat(
        [old_frame[["X", "Y", "Z"]].add_prefix("old_"), new_frame[["X", "Y", "Z"]].add_prefix("new_")], axis=1
    )
    all_coords.dropna(inplace=True)

    # Vectorized distance calculations
    for i, axis in enumerate(["X", "Y", "Z"]):
        distance = np.abs(all_coords[f"new_{axis}"] - all_coords[f"old_{axis}"])
        boundary_adjusted = np.where(distance > box_size[i] / 2, distance - box_size[i], distance)
        all_coords[f"distance_{axis.lower()}"] = boundary_adjusted

    all_coords["distance"] = np.linalg.norm(all_coords[["distance_x", "distance_y", "distance_z"]], axis=1)
    all_coords["velocity"] = all_coords["distance"] / inputdict["dt"]

    # Update new_frame with velocity
    new_frame["velocity"] = all_coords["velocity"]
    inputdict["old_frame"] = new_frame

    return inputdict


def rad_velocity_prep(inputdict, traj_file, molecules):

    args = inputdict["args"]

    first_frame = traj_file.frame0.copy()
    first_frame.rename(columns={"x": "X", "y": "Y", "z": "Z"}, inplace=True)

    ddict.printLog("")
    if len(molecules.CNT_centers) > 1:
        ddict.printLog("-> Multiple CNTs detected. The analysis will be conducted on the first CNT.\n", color="red")
    if len(molecules.CNT_centers) == 0:
        ddict.printLog("-> No CNTs detected. Aborting...\n", color="red")
        sys.exit(1)
    for i in range(len(molecules.CNT_centers)):
        ddict.printLog(f"\n-> CNT{i + 1}")
        num_increments = int(
            ddict.get_input("How many increments do you want to use to calculate the density profile? ", args, "int")
        )
        rad_increment = molecules.tuberadii[i] / num_increments
        # Make an array which start at 0 and end at tuberadius with num_increments + 1 steps.
        velocity_bin_edges = np.linspace(0, molecules.tuberadii[0], num_increments + 1)
        # Define velocity_bin_labels, they are a counter for the bin edges.
        velocity_bin_labels = np.arange(1, len(velocity_bin_edges), 1)
        ddict.printLog("Increment distance: %0.3f angstrom" % (rad_increment))
    velocity_bin_labels = np.arange(0, num_increments, 1)
    # Make new dataframe with the number of frames of the trajectory.
    velocity_df_dummy = pd.DataFrame(np.arange(1, traj_file.number_of_frames + 1), columns=["Frame"])
    # Add a column to the dataframe for each increment.
    for i in range(num_increments):
        velocity_df_dummy["Bin %d" % (i + 1)] = 0
        velocity_df_dummy = velocity_df_dummy.copy()
    velocity_df = velocity_df_dummy.copy()

    # first ask the user how large the time step is in the trajectory
    dt = ddict.get_input("What is the time step in the trajectory? [fs]  ", args, "float")

    # make a new dataframe with two columns, the first is the Bin number and the second is the total count of velocities
    # in that bin
    velocity_bin_counter = pd.DataFrame({"Bin": velocity_bin_labels, "Total count": 0})
    # Prepare output dict
    outputdict = {
        "rad_increment": rad_increment,
        "velocity_bin_edges": velocity_bin_edges,
        "velocity_bin_labels": velocity_bin_labels,
        "velocity_df": velocity_df,
        "num_increments": num_increments,
        "dt": dt,
        "velocity_bin_counter": velocity_bin_counter,
        "old_frame": first_frame,
    }

    outputdict.update(**inputdict)

    return outputdict


def rad_velocity_analysis(inputdict, traj_file, molecules, analysis):

    max_z_pore = inputdict["max_z_pore"]
    min_z_pore = inputdict["min_z_pore"]

    inputdict["split_frame"] = inputdict["split_frame"][inputdict["split_frame"]["Z"].astype(float) <= max_z_pore[0]]
    inputdict["split_frame"] = inputdict["split_frame"][inputdict["split_frame"]["Z"].astype(float) >= min_z_pore[0]]
    # Calculate velocity using velocity_calc_atom function
    inputdict = velocity_calc_atom(inputdict)

    # Extract the frame with velocity information
    velocity_frame = inputdict["old_frame"]
    CNT_centers = inputdict["CNT_centers"]
    velocity_bin_edges = inputdict["velocity_bin_edges"]
    velocity_bin_labels = inputdict["velocity_bin_labels"]
    num_increments = inputdict["num_increments"]
    counter = inputdict["counter"]
    velocity_df = inputdict["velocity_df"]
    velocity_bin_counter = inputdict["velocity_bin_counter"]

    # Calculate the radial distance of each atom from the center of the CNT
    velocity_frame["X_adjust"] = velocity_frame["X"].astype(float) - CNT_centers[0][0]
    velocity_frame["Y_adjust"] = velocity_frame["Y"].astype(float) - CNT_centers[0][1]
    velocity_frame["Distance"] = np.sqrt(velocity_frame["X_adjust"] ** 2 + velocity_frame["Y_adjust"] ** 2)

    # Group the velocities based on the radial distance
    velocity_frame["Distance_bin"] = pd.cut(
        velocity_frame["Distance"], bins=velocity_bin_edges, labels=velocity_bin_labels
    )

    # We need to store two values for each bin: the total number of atoms counted for each bin and the sum of the
    # velocities for each bin
    # We will use these values to calculate the average velocity for each bin, after the loop
    velocity_bin_counter_temp = velocity_frame.groupby("Distance_bin")["velocity"].count().reset_index()

    # Update the velocity_bin_counter DataFrame with the new data
    for i in range(num_increments):
        if i in velocity_bin_counter_temp["Distance_bin"].values:
            velocity_bin_counter.loc[i, "Total count"] += velocity_bin_counter_temp.loc[
                velocity_bin_counter_temp["Distance_bin"] == i, "velocity"
            ].values[0]
        else:
            velocity_bin_counter.loc[i, "Total count"] += 0

    velocity_df_temp = velocity_frame.groupby("Distance_bin")["velocity"].sum().reset_index()
    velocity_df_temp["Bin"] = velocity_df_temp.index

    # Update the velocity_df DataFrame with the new data
    for i in range(num_increments):
        if i in velocity_df_temp["Bin"].values:
            velocity_df.loc[counter, "Bin %d" % (i + 1)] = velocity_df_temp.loc[
                velocity_df_temp["Bin"] == i, "velocity"
            ].values[0]
        else:
            velocity_df.loc[counter, "Bin %d" % (i + 1)] = 0

    # Prepare output dict
    outputdict = inputdict
    outputdict["velocity_df"] = velocity_df
    outputdict["velocity_bin_counter"] = velocity_bin_counter

    return outputdict


def rad_velocity_processing(inputdict):

    velocity_df = inputdict["velocity_df"]
    velocity_bin_edges = inputdict["velocity_bin_edges"]
    velocity_bin_counter = inputdict["velocity_bin_counter"]
    args = inputdict["args"]
    tuberadii = inputdict["tuberadii"]
    radius_tube = tuberadii[0]

    # first add +1 to the velocity_bin_counter column 'Bin' to get the correct bin number
    velocity_bin_counter["Bin"] = velocity_bin_counter["Bin"] + 1

    # Initialize a DataFrame to store the results
    results_vd_df = pd.DataFrame()

    for i in range(1, len(velocity_bin_edges)):
        bin_label = "Bin %d" % i
        # Sum velocities and count entries for each bin
        bin_sum = velocity_df[bin_label].sum()
        # the count is the number in the velocity_bin_counter dataframe 'Total count' column
        bin_count = velocity_bin_counter[velocity_bin_counter["Bin"] == i]["Total count"].values[0]

        # Calculate the average velocity for the bin
        average_velocity = bin_sum / bin_count if bin_count != 0 else 0
        results_vd_df.loc[i, "Average Velocity"] = average_velocity

    # Add bin edge information to the results DataFrame
    results_vd_df["Bin_lowedge"] = velocity_bin_edges[:-1]
    results_vd_df["Bin_highedge"] = velocity_bin_edges[1:]
    results_vd_df["Bin_center"] = (velocity_bin_edges[1:] + velocity_bin_edges[:-1]) / 2

    # Reset the index of the DataFrame
    results_vd_df.reset_index(drop=True, inplace=True)
    results_vd_df.insert(0, "Bin", results_vd_df.index + 1)

    # Plotting
    plot_data = ddict.get_input("Do you want to plot the data? (y/n) ", args, "string")
    if plot_data == "y":
        # Normalization and mirroring options
        normalize = ddict.get_input(
            "Do you want to normalize the increments with respect to the CNTs' radius? (y/n) ", args, "string"
        )
        mirror = ddict.get_input("Do you want to mirror the plot? (y/n) ", args, "string")

        if normalize == "y":
            results_vd_df["Bin_center"] = results_vd_df["Bin_center"] / radius_tube

        if mirror == "y":
            results_vd_dummy = results_vd_df.copy()
            results_vd_dummy["Bin_center"] *= -1
            results_vd_dummy.sort_values(by=["Bin_center"], inplace=True)
            results_vd_df = pd.concat([results_vd_dummy, results_vd_df], ignore_index=True)

        # Plot
        fig, ax = plt.subplots()
        ax.plot(
            results_vd_df["Bin_center"],
            results_vd_df["Average Velocity"],
            "-",
            label="Radial velocity profile",
            color="black",
        )
        ax.set(
            xlabel=r"Distance from tube center / $\mathrm{\AA}$",
            ylabel=r"Velocity / $\mathrm{\AA/fs}$",
            title="Radial Velocity Profile",
        )
        ax.grid()
        fig.savefig("Radial_velocity_profile.pdf")

        # Save the data
        results_vd_df.to_csv("Radial_velocity_profile.csv", sep=";", index=False, header=True, float_format="%.5f")

    # Save raw data
    save_raw = ddict.get_input("Do you want to save the raw data? (y/n) ", args, "string")
    if save_raw == "y":
        velocity_df.to_csv("Radial_velocity_raw.csv", sep=";", index=False, header=True, float_format="%.5f")
