import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import conan.analysis_modules.axial_dens as axdens
import conan.analysis_modules.rad_velocity as rad_vel
import conan.defdict as ddict


def COM_calculation(frame):

    # We now calculate the center of mass (COM) for each molecule
    # Convert all values to float (this is needed so that the agg-function works)
    frame["X"] = frame["X"].astype(float)
    frame["Y"] = frame["Y"].astype(float)
    frame["Z"] = frame["Z"].astype(float)
    frame["Mass"] = frame["Mass"].astype(float)

    # Precompute total mass for each molecule
    total_mass_per_molecule = frame.groupby("Molecule")["Mass"].transform("sum")

    # Calculate mass weighted coordinates
    frame["X_COM"] = (frame["X"] * frame["Mass"]) / total_mass_per_molecule
    frame["Y_COM"] = (frame["Y"] * frame["Mass"]) / total_mass_per_molecule
    frame["Z_COM"] = (frame["Z"] * frame["Mass"]) / total_mass_per_molecule

    # Calculate the center of mass for each molecule
    mol_com = (
        frame.groupby("Molecule")
        .agg(Species=("Species", "first"), X_COM=("X_COM", "sum"), Y_COM=("Y_COM", "sum"), Z_COM=("Z_COM", "sum"))
        .reset_index()
    )

    return mol_com


def velocity_prep(inputdict):

    args = inputdict["args"]
    box_size = inputdict["box_size"]
    # first ask the user how large the time step is in the trajectory
    inputdict["dt"] = ddict.get_input("What is the time step in the trajectory? [fs]  ", args, "float")

    # set up the grid by using the prep function of the axial_dens module
    inputdict = axdens.density_analysis_prep(inputdict)

    # get first frame
    first_frame = inputdict["id_frame"].copy()
    # rename columns from x to X, y to Y and z to Z
    first_frame.rename(columns={"x": "X", "y": "Y", "z": "Z"}, inplace=True)

    # add a new mass column to the first frame
    element_masses = ddict.dict_mass()
    first_frame["Mass"] = first_frame["Element"].map(element_masses)

    velocity_choice = ddict.get_input(
        "Do you want to calculate the velocity of each [1]molecule or each [2]atom? ", args, "string"
    )
    if velocity_choice == "1":
        inputdict["velocity_choice"] = "molecule"
        ddict.printLog(
            "Warning: This analysis will yield erroneous results, if the trajectory is wrapped atomwise!\n", color="red"
        )
        COM_frame = COM_calculation(first_frame)
        inputdict["old_frame"] = COM_frame
    elif velocity_choice == "2":
        inputdict["velocity_choice"] = "atom"
        first_frame = axdens.wrapping_coordinates(box_size, first_frame)
        inputdict["old_frame"] = first_frame

    inputdict["grid_points_velocities"] = inputdict["grid_point_atom_labels"]
    del inputdict["grid_point_atom_labels"]

    occurrences = [0 for i in range(len(inputdict["grid_points_velocities"]))]
    inputdict["grid_point_occurrences"] = occurrences

    return inputdict


def velocity_calc_molecule(
    inputdict,
):  # -> something is still wrong here, the velocity is not calculated correctly probably due too the COM calculation
    # and the wrapping afterwards

    box_size = inputdict["box_size"]

    old_frame = inputdict["old_frame"]
    new_frame = inputdict["split_frame"]
    # calculate the COM for the new frame
    new_frame = COM_calculation(new_frame)

    # rename the columns of both frames from X_COM to X, Y_COM to Y and Z_COM to Z
    old_frame.rename(columns={"X_COM": "X", "Y_COM": "Y", "Z_COM": "Z"}, inplace=True)
    new_frame.rename(columns={"X_COM": "X", "Y_COM": "Y", "Z_COM": "Z"}, inplace=True)

    # wrap the coordinates of the frames
    old_frame = axdens.wrapping_coordinates(box_size, old_frame)
    new_frame = axdens.wrapping_coordinates(box_size, new_frame)

    # combine the dataframes
    all_coords = pd.concat(
        [old_frame[["X", "Y", "Z"]].add_prefix("old_"), new_frame[["X", "Y", "Z"]].add_prefix("new_")], axis=1
    )

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


def velocity_analysis_chunk_processing(inputdict):

    grid_points_velocities = inputdict["grid_points_velocities"]

    chunk_occurances = [len(grid_points_velocities[i]) for i in range(len(grid_points_velocities))]

    # now calculate the mean velocity for each grid point for the chunk, if there is no velocity, set the mean velocity
    # to zero
    chunk_mean_velocities = [
        np.mean(grid_points_velocities[i]) if len(grid_points_velocities[i]) != 0 else 0
        for i in range(len(grid_points_velocities))
    ]

    # first check if the grid_points_average_velocities and grid_points_occurrances are already in the inputdict
    try:
        grid_points_average_velocities = inputdict["grid_point_average_velocities"]
        grid_points_occurrances = inputdict["grid_point_occurrences"]
    except KeyError:
        grid_points_average_velocities = chunk_mean_velocities
        grid_points_occurrances = chunk_occurances

    # now we can calculate the new average velocities
    for i in range(len(grid_points_average_velocities)):
        total_occurrances = grid_points_occurrances[i] + chunk_occurances[i]
        if total_occurrances == 0:
            grid_points_average_velocities[i] = 0
        else:
            grid_points_average_velocities[i] = (
                grid_points_average_velocities[i] * grid_points_occurrances[i]
                + chunk_mean_velocities[i] * chunk_occurances[i]
            ) / (grid_points_occurrances[i] + chunk_occurances[i])
            grid_points_occurrances[i] += chunk_occurances[i]

    # reset the grid_points_velocities
    inputdict["grid_points_velocities"] = [[] for _ in range(len(grid_points_velocities))]

    # now update the inputdict
    inputdict["grid_point_average_velocities"] = grid_points_average_velocities
    inputdict["grid_point_occurrences"] = grid_points_occurrances

    return inputdict


def velocity_analysis(inputdict):

    analysis_methods = {"molecule": velocity_calc_molecule, "atom": rad_vel.velocity_calc_atom}
    analysis_choice = inputdict["velocity_choice"]
    analysis_function = analysis_methods[analysis_choice]
    inputdict = analysis_function(inputdict)

    cube_array = inputdict["cube_array"]
    grid_points_tree = inputdict["grid_points_tree"]
    analysis_counter = inputdict["analysis_counter"]
    old_frame = inputdict["old_frame"]
    grid_points_velocities = inputdict["grid_points_velocities"]

    # now get the coordinates of the split_frame
    old_frame_coords = np.array(old_frame[["X", "Y", "Z"]])
    old_frame_coords = old_frame_coords.astype(float)

    # now find the corresponding grid point for each atom
    closest_grid_point_dist, closest_grid_point_idx = grid_points_tree.query(old_frame_coords)

    velocities = old_frame[["velocity"]]
    velocities = velocities.astype(float)

    # now we can append the velocities to the grid points
    for i in range(len(old_frame)):
        grid_points_velocities[closest_grid_point_idx[i]].append(old_frame["velocity"].values[i])

    analysis_counter += 1
    # after 1000 frames, we need to do some chunk processing
    if analysis_counter == 100:
        inputdict = velocity_analysis_chunk_processing(inputdict)
        analysis_counter = 0
    else:
        inputdict["grid_points_velocities"] = grid_points_velocities

    outputdict = inputdict
    outputdict["cube_array"] = cube_array
    outputdict["analysis_counter"] = analysis_counter

    return outputdict


def velocity_processing(inputdict):

    inputdict = velocity_analysis_chunk_processing(inputdict)

    grid_points_average_velocities = inputdict["grid_point_average_velocities"]

    # currently the velocity is calulated in Angstrom/fs. We want to convert it to m/s
    # 1 Angstrom = 1e-10 m
    # 1 fs = 1e-15 s
    # 1 Angstrom/fs = 1e-10 m / 1e-15 s = 1e5 m/s

    for i in range(len(grid_points_average_velocities)):
        grid_points_average_velocities[i] = float(grid_points_average_velocities[i]) * 1e5

    # print the maximum value of the velocities
    ddict.printLog(f"The maximum velocity is: {max(grid_points_average_velocities)} ")
    # write a cube file from the grid_points_velocities
    inputdict["grid_point_densities"] = grid_points_average_velocities
    axdens.write_cube_file(inputdict, filename="velocity.cube")

    # Reshape the velocity data
    velocities = np.array(grid_points_average_velocities).reshape(
        inputdict["x_incr"], inputdict["y_incr"], inputdict["z_incr"]
    )

    # Initialize velocity profiles and count non-zero entries
    x_vel_profile = np.zeros(inputdict["x_incr"])
    y_vel_profile = np.zeros(inputdict["y_incr"])
    z_vel_profile = np.zeros(inputdict["z_incr"])
    x_count_nonzero = np.zeros(inputdict["x_incr"])
    y_count_nonzero = np.zeros(inputdict["y_incr"])
    z_count_nonzero = np.zeros(inputdict["z_incr"])

    # Calculate velocity profiles and count non-zero velocities
    for i in range(inputdict["x_incr"]):
        for j in range(inputdict["y_incr"]):
            for k in range(inputdict["z_incr"]):
                if velocities[i, j, k] != 0:
                    x_vel_profile[i] += velocities[i, j, k]
                    y_vel_profile[j] += velocities[i, j, k]
                    z_vel_profile[k] += velocities[i, j, k]
                    x_count_nonzero[i] += 1
                    y_count_nonzero[j] += 1
                    z_count_nonzero[k] += 1

    # Normalize the profiles, avoid division by zero
    x_vel_profile = np.divide(
        x_vel_profile, x_count_nonzero, out=np.zeros_like(x_vel_profile), where=x_count_nonzero != 0
    )
    y_vel_profile = np.divide(
        y_vel_profile, y_count_nonzero, out=np.zeros_like(y_vel_profile), where=y_count_nonzero != 0
    )
    z_vel_profile = np.divide(
        z_vel_profile, z_count_nonzero, out=np.zeros_like(z_vel_profile), where=z_count_nonzero != 0
    )

    # Replace NaNs resulting from division by zero with zeros
    x_vel_profile = np.nan_to_num(x_vel_profile)
    y_vel_profile = np.nan_to_num(y_vel_profile)
    z_vel_profile = np.nan_to_num(z_vel_profile)

    # Create DataFrames for the velocity profiles
    x_vel_df = pd.DataFrame({"x": inputdict["x_grid"], "Velocity": x_vel_profile})
    y_vel_df = pd.DataFrame({"y": inputdict["y_grid"], "Velocity": y_vel_profile})
    z_vel_df = pd.DataFrame({"z": inputdict["z_grid"], "Velocity": z_vel_profile})

    # Save to CSV
    x_vel_df.to_csv("x_velocity_profile.csv", sep=";", index=False, header=True, float_format="%.5f")
    y_vel_df.to_csv("y_velocity_profile.csv", sep=";", index=False, header=True, float_format="%.5f")
    z_vel_df.to_csv("z_velocity_profile.csv", sep=";", index=False, header=True, float_format="%.5f")

    # Plot the velocity profiles
    for direction, df in zip(["x", "y", "z"], [x_vel_df, y_vel_df, z_vel_df]):
        fig, ax = plt.subplots()
        ax.plot(df.iloc[:, 0], df["Velocity"], "-", label="Velocity profile", color="black")
        ax.set(xlabel=f"{direction} [Ang]", ylabel="Velocity", title=f"Velocity Profile along {direction.upper()}")
        ax.grid()
        fig.savefig(f"{direction}_velocity_profile.pdf")

    return inputdict
