import numpy as np

import conan.defdict as ddict

"""
calculates the minimum image distance between two sets of coordinates"""


def minimum_image_distance(box_dimension, coordinates_reference, coordinates_observed):

    dx = coordinates_reference["X_COM"].values[:, np.newaxis] - coordinates_observed["X_COM"].values
    dy = coordinates_reference["Y_COM"].values[:, np.newaxis] - coordinates_observed["Y_COM"].values
    dz = coordinates_reference["Z_COM"].values[:, np.newaxis] - coordinates_observed["Z_COM"].values
    dx = dx - box_dimension[0] * np.round(dx / box_dimension[0])
    dy = dy - box_dimension[1] * np.round(dy / box_dimension[1])
    dz = dz - box_dimension[2] * np.round(dz / box_dimension[2])
    distances = np.sqrt(dx**2 + dy**2 + dz**2)
    return distances


def calculate_com_molecule(molecule_df, box_size):
    total_mass = molecule_df["Mass"].sum()
    positions = molecule_df[["X", "Y", "Z"]].values
    # Make masses a column vector
    masses = molecule_df["Mass"].values[:, np.newaxis]
    box_size_array = np.array(box_size, dtype=float)

    # Initialize center of mass
    com = np.zeros(3)

    for i in range(len(molecule_df)):
        vector = positions[i] - com / total_mass
        vector_divided = vector / box_size_array
        vector_rounded = np.around(vector_divided.astype(np.double))
        # Minimum image convention
        vector -= vector_rounded * box_size_array
        com += vector * masses[i]

    com /= total_mass

    return com


def calculate_com_box(frame, box_size):

    # First wrap the coordinates
    frame_wrapped = wrapping_coordinates(box_size, frame)

    total_mass = frame_wrapped["Mass"].sum()
    positions = frame_wrapped[["X", "Y", "Z"]].values
    # Make masses a column vector
    masses = frame_wrapped["Mass"].values[:, np.newaxis]

    # Initialize center of mass
    com = np.zeros(3)

    # add all the positions multiplied by the mass to the center of mass
    for i in range(len(frame_wrapped)):
        com += positions[i] * masses[i]

    com /= total_mass

    return com


def symbols_to_masses(symbols):
    mass_dict = ddict.dict_mass()
    return [mass_dict[sym] if sym in mass_dict else print(f"Warning: {sym} not found in mass_dict") for sym in symbols]


def grid_generator(inputdict):

    # Inputdict
    box_size = inputdict["box_size"]
    args = inputdict["args"]

    # Get the input from the user
    x_incr = ddict.get_input("How many increments do you want to use in the x direction? ", args, "int")
    y_incr = ddict.get_input("How many increments do you want to use in the y direction? ", args, "int")
    z_incr = ddict.get_input("How many increments do you want to use in the z direction? ", args, "int")

    # Calculate the incrementation distance
    x_incr_dist = box_size[0] / x_incr
    y_incr_dist = box_size[1] / y_incr
    z_incr_dist = box_size[2] / z_incr

    # Create a grid with the dimensions of the simulation box
    x_grid = np.linspace(0, box_size[0], x_incr, endpoint=False) + (x_incr_dist / 2)
    y_grid = np.linspace(0, box_size[1], y_incr, endpoint=False) + (y_incr_dist / 2)
    z_grid = np.linspace(0, box_size[2], z_incr, endpoint=False) + (z_incr_dist / 2)

    # Create a meshgrid
    x_mesh, y_mesh, z_mesh = np.meshgrid(x_grid, y_grid, z_grid, indexing="ij")

    # Print the grid information
    ddict.printLog("Incrementation distance in x direction: %0.3f Ang" % (x_incr_dist))
    ddict.printLog("Incrementation distance in y direction: %0.3f Ang" % (y_incr_dist))
    ddict.printLog("Incrementation distance in z direction: %0.3f Ang" % (z_incr_dist))

    # Number of grid points
    number_grid_points = x_incr * y_incr * z_incr
    ddict.printLog("Total number of grid points: %d" % (number_grid_points))

    # Return the inputdict
    outputdict = inputdict
    outputdict["x_incr"] = x_incr
    outputdict["y_incr"] = y_incr
    outputdict["z_incr"] = z_incr
    outputdict["x_incr_dist"] = x_incr_dist
    outputdict["y_incr_dist"] = y_incr_dist
    outputdict["z_incr_dist"] = z_incr_dist
    outputdict["x_grid"] = x_grid
    outputdict["y_grid"] = y_grid
    outputdict["z_grid"] = z_grid
    outputdict["x_mesh"] = x_mesh
    outputdict["y_mesh"] = y_mesh
    outputdict["z_mesh"] = z_mesh
    outputdict["number_grid_points"] = number_grid_points

    return outputdict


def write_cube_file(analysis, filename):

    box_size = analysis.traj_file.box_size

    # Extract necessary data from outputdict
    xbin_edges = box_size[0] / analysis.x_incr * np.arange(analysis.x_incr)
    ybin_edges = box_size[1] / analysis.y_incr * np.arange(analysis.y_incr)
    zbin_edges = box_size[2] / analysis.z_incr * np.arange(analysis.z_incr)

    grid_point_densities = analysis.grid_point_densities
    id_frame = analysis.traj_file.frame0

    # Drop all lines in the id_frame which are labeled 'Liquid' in the 'Struc' column
    id_frame = id_frame[id_frame["Struc"] != "Liquid"]

    number_of_atoms = len(id_frame)

    # Write to file
    with open(filename, "w") as file:
        # Header
        file.write("Cube file generated by density_analysis\n")
        file.write("OUTER loop: X, MIDDLE loop: Y, INNER loop: Z\n")

        # Write the number of atoms and origin
        file.write(f"{number_of_atoms} 0.00 0.00 0.00\n")

        # Write the number of atoms and origin
        file.write(f"{len(xbin_edges)} {xbin_edges[1] - xbin_edges[0]:.5f} 0.00 0.00\n")
        file.write(f"{len(ybin_edges)} 0.00 {ybin_edges[1] - ybin_edges[0]:.5f} 0.00\n")
        file.write(f"{len(zbin_edges)} 0.00 0.00 {zbin_edges[1] - zbin_edges[0]:.5f}\n")

        # Write atom information of the atoms in the id_frame
        for index, row in id_frame.iterrows():
            file.write(f"12 {row['Charge']} {row['x']} {row['y']} {row['z']}\n")

        # Volumetric data
        for i, density in enumerate(grid_point_densities):
            file.write(f"{density:.5f} ")
            if (i + 1) % 6 == 0:  # 6 densities per line
                file.write("\n")


def wrapping_coordinates(box_size, frame):
    # in this function the coordinates of the atoms in the split_frame are wrapped.
    # Check if there are atoms outside the simulation box and wrap them.
    # Then check again if it worked
    # If not: Wrap them again. We do this until all atoms are inside the simulation box.

    # Get the coordinates of the split_frame
    split_frame_coords = frame[["X", "Y", "Z"]].astype(float).values

    # Check if there are atoms outside the simulation box
    while (split_frame_coords > box_size).any() or (split_frame_coords < 0).any():
        # now wrap the coordinates
        split_frame_coords = np.where(split_frame_coords > box_size, split_frame_coords - box_size, split_frame_coords)
        split_frame_coords = np.where(split_frame_coords < 0, split_frame_coords + box_size, split_frame_coords)

    # Print the wrapped coordinates to the split_frame
    frame[["X", "Y", "Z"]] = split_frame_coords

    return frame
