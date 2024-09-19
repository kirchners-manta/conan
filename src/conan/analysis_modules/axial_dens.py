import math
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

import conan.defdict as ddict
from conan.analysis_modules import utils as ut


# Accessible volume
def accessible_volume_prep(inputdict, traj_file, molecules):

    args = inputdict["args"]

    element_radii = dict()
    ddict.printLog("")

    if len(molecules.structure_data["CNT_centers"]) > 1:
        ddict.printLog("-> Multiple CNTs detected. The analysis will be conducted on the first CNT.\n", color="red")
    if len(molecules.structure_data["CNT_centers"]) == 0:
        ddict.printLog("-> No CNTs detected. Aborting...\n", color="red")
        sys.exit(1)

    which_element_radii = ddict.get_input(
        "Do you want to use the van der Waals radii (1) or the covalent radii (2) of the elements? [1/2] ", args, "int"
    )
    if which_element_radii == 1:
        ddict.printLog("-> Using van der Waals radii.")
        element_radii = ddict.dict_vdW()
    elif which_element_radii == 2:
        ddict.printLog("-> Using covalent radii.")
        element_radii = ddict.dict_covalent()

    outputdict = inputdict
    outputdict["element_radii"] = element_radii
    outputdict["maxdisp_atom_dist"] = 0

    return outputdict


def accessible_volume_analysis(inputdict, traj_file, molecules):

    split_frame = inputdict["split_frame"]
    CNT_centers = molecules.CNT_centers
    max_z_pore = molecules.max_z_pore
    min_z_pore = molecules.min_z_pore

    element_radii = inputdict["element_radii"]
    maxdisp_atom_dist = inputdict["maxdisp_atom_dist"]
    maxdisp_atom_row = inputdict["maxdisp_atom_row"]

    # instead of checking the global coordinates, we consider the PBC and check the modulus of the box_size
    split_frame["X"] = split_frame["X"].astype(float) % traj_file.box_size[0]
    split_frame["Y"] = split_frame["Y"].astype(float) % traj_file.box_size[1]
    split_frame["Z"] = split_frame["Z"].astype(float) % traj_file.box_size[2]

    # we need to also account this for the max and min z_pore values
    max_z_pore[0] = max_z_pore[0] % traj_file.box_size[2]
    min_z_pore[0] = min_z_pore[0] % traj_file.box_size[2]

    # we also need to do this for the CNT_centers
    CNT_centers[0][0] = CNT_centers[0][0] % traj_file.box_size[0]
    CNT_centers[0][1] = CNT_centers[0][1] % traj_file.box_size[1]

    # if the pore is split over the periodic boundary (pot. because of modulo operation)
    if min_z_pore[0] > max_z_pore[0]:
        # Split the selection into two parts
        part1 = split_frame[split_frame["Z"].astype(float) >= min_z_pore[0]]
        part2 = split_frame[split_frame["Z"].astype(float) <= max_z_pore[0]]
        split_frame = pd.concat([part1, part2])
    else:
        split_frame = split_frame[split_frame["Z"].astype(float) <= max_z_pore[0]]
        split_frame = split_frame[split_frame["Z"].astype(float) >= min_z_pore[0]]

    # Calculate the radial density function with the remaining atoms.
    split_frame["X_adjust"] = split_frame["X"].astype(float) - CNT_centers[0][0]
    split_frame["Y_adjust"] = split_frame["Y"].astype(float) - CNT_centers[0][1]

    # While calculating the (most displaced) distance, add the elements radius.
    split_frame["Distance"] = np.sqrt(split_frame["X_adjust"] ** 2 + split_frame["Y_adjust"] ** 2) + split_frame[
        "Atom"
    ].map(element_radii)
    if split_frame["Distance"].max() > maxdisp_atom_dist:
        maxdisp_atom_dist = split_frame["Distance"].max()

        # Save the complete info of the most displaced atom.
        maxdisp_atom_row = split_frame.loc[split_frame["Distance"].idxmax()]

    inputdict["maxdisp_atom_row"] = maxdisp_atom_row
    inputdict["maxdisp_atom_dist"] = maxdisp_atom_dist

    return inputdict


def accessible_volume_processing(inputdict):
    args = inputdict["args"]
    maxdisp_atom_row = inputdict["maxdisp_atom_row"]
    length_pore = inputdict["length_pore"]
    CNT_length = length_pore[0]
    CNT_atoms = inputdict["CNT_atoms"]

    # Write the maximal displaced atom to the log file and the terminal.
    ddict.printLog(
        "\nMaximal displaced atom: %s (%0.3f|%0.3f|%0.3f)"
        % (
            maxdisp_atom_row["Atom"],
            float(maxdisp_atom_row["X"]),
            float(maxdisp_atom_row["Y"]),
            float(maxdisp_atom_row["Z"]),
        )
    )
    accessible_radius = maxdisp_atom_row["Distance"]
    ddict.printLog("Maximal displacement: %0.3f" % accessible_radius)
    ddict.printLog("Accessible volume: %0.3f" % (math.pi * accessible_radius**2 * CNT_length))
    ddict.printLog("")

    # Print the carbon structure with the most displaced atom, if the user wants to.
    pore_disp_atom = ddict.get_input(
        "Do you want to produce a xyz file with the pore including the most displaced atom? [y/n] ", args, "string"
    )
    if pore_disp_atom == "y":
        f = open("pore_disp_atom.xyz", "w")
        f.write("%d\n#Pore with most displaced atom\n" % (len(CNT_atoms) + 1))
        for index, row in CNT_atoms.iterrows():
            f.write("%s\t%0.3f\t%0.3f\t%0.3f\n" % (row["Element"], row["x"], row["y"], row["z"]))
        f.write(
            "%s\t%0.3f\t%0.3f\t%0.3f\n"
            % (
                maxdisp_atom_row["Atom"],
                float(maxdisp_atom_row["X"]),
                float(maxdisp_atom_row["Y"]),
                float(maxdisp_atom_row["Z"]),
            )
        )
        f.close()
        ddict.printLog("Pore with most displaced atom saved as pore_disp_atom.xyz")


# Distance search


def distance_search_prep(inputdict, traj_file, molecules):

    # drop all rows, which are labeled 'Liquid' in the 'Struc' column
    structure_atoms = traj_file.frame0[traj_file.frame0["Struc"] != "Liquid"]

    # now transform the structure atoms into a kd-tree
    structure_atoms_tree = scipy.spatial.KDTree(structure_atoms[["x", "y", "z"]].values)

    # now add the structure_atoms_tree to the inputdict
    inputdict["structure_atoms_tree"] = structure_atoms_tree

    inputdict["minimal_distance"] = 1000
    inputdict["minimal_distance_row"] = None
    inputdict["maximal_distance"] = 0
    inputdict["maximal_distance_row"] = None

    inputdict["minimal_distance_vdW"] = 1000
    inputdict["minimal_distance_row_vdW"] = None
    inputdict["maximal_distance_vdW"] = 0
    inputdict["maximal_distance_row_vdW"] = None

    # now return the inputdict
    return inputdict


def distance_search_analysis(inputdict, traj_file, molecules):

    split_frame = inputdict["split_frame"]
    # get the structure atoms from the inputdict
    structure_atoms_tree = inputdict["structure_atoms_tree"]

    # check if there are any atoms outside the simulation box
    split_frame = ut.wrapping_coordinates(traj_file.box_size, split_frame)

    # now get the coordinates of the split_frame
    split_frame_coords = split_frame[["X", "Y", "Z"]].values

    # now query the structure_atoms_tree for the closest atom to each atom in the split_frame
    closest_atom_dist, closest_atom_idx = structure_atoms_tree.query(split_frame_coords)

    # print the distance to an extra column in the split_frame
    split_frame["Distance"] = closest_atom_dist

    min_dist = split_frame["Distance"].min()
    max_dist = split_frame["Distance"].max()

    # now check if the min_dist is smaller than the min_dist in the inputdict
    if min_dist < inputdict["minimal_distance"]:
        inputdict["minimal_distance"] = min_dist
        # add the row to the inputdict
        inputdict["minimal_distance_row"] = split_frame.loc[split_frame["Distance"].idxmin()]

    # now check if the max_dist is larger than the max_dist in the inputdict
    if max_dist > inputdict["maximal_distance"]:
        inputdict["maximal_distance"] = max_dist
        # add the row to the inputdict
        inputdict["maximal_distance_row"] = split_frame.loc[split_frame["Distance"].idxmax()]

    # now return the inputdict
    return inputdict


def distance_search_processing(inputdict):
    minimal_distance = inputdict["minimal_distance"]
    maximal_distance = inputdict["maximal_distance"]
    minimal_distance_row = inputdict["minimal_distance_row"]
    maximal_distance_row = inputdict["maximal_distance_row"]

    ddict.printLog(
        "The closest atom is: ",
        minimal_distance_row["Atom"],
        " with a distance of: ",
        round(minimal_distance, 2),
        " \u00C5",
    )
    ddict.printLog(
        "The furthest atom is: ",
        maximal_distance_row["Atom"],
        " with a distance of: ",
        round(maximal_distance, 2),
        " \u00C5",
    )


# Axial density profile


def axial_density_prep(inputdict, traj_file, molecules):

    args = inputdict["args"]
    CNT_atoms = inputdict["CNT_atoms"]

    num_increments = int(
        ddict.get_input(
            "How many increments per section do you want to use to calculate the density profile? ", args, "int"
        )
    )
    ddict.printLog(
        "\nNote here, that the simulation box is subdivided between two bulk phases and the pore. \nThe number of "
        "increments set here is the number of increments for each section.\n",
        color="red",
    )

    max_z_pore = molecules.max_z_pore
    min_z_pore = molecules.min_z_pore
    CNT_centers = molecules.CNT_centers

    # we need to also account this for the max and min z_pore values
    max_z_pore[0] = max_z_pore[0] % traj_file.box_size[2]
    min_z_pore[0] = min_z_pore[0] % traj_file.box_size[2]

    # we also need to do this for the CNT_centers
    CNT_centers[0][0] = CNT_centers[0][0] % traj_file.box_size[0]
    CNT_centers[0][1] = CNT_centers[0][1] % traj_file.box_size[1]

    # do the same for the CNT_atoms
    CNT_atoms = CNT_atoms.copy()
    CNT_atoms.loc[:, "x"] = CNT_atoms["x"] % traj_file.box_size[0]
    CNT_atoms.loc[:, "y"] = CNT_atoms["y"] % traj_file.box_size[1]
    CNT_atoms.loc[:, "z"] = CNT_atoms["z"] % traj_file.box_size[2]

    # Initialize arrays
    z_incr_CNT = [0] * len(CNT_centers)
    z_incr_bulk1 = [0] * len(CNT_centers)
    z_incr_bulk2 = [0] * len(CNT_centers)
    z_bin_edges_pore = [0] * len(CNT_centers)
    z_bin_edges_bulk1 = [0] * len(CNT_centers)
    z_bin_edges_bulk2 = [0] * len(CNT_centers)
    z_bin_edges = [0] * len(CNT_centers)
    z_bin_labels = [0] * len(CNT_centers)

    # Calculate the increment distance for each section.
    for i in range(len(CNT_centers)):
        z_incr_CNT[i] = molecules.length_pore[i] / num_increments
        z_incr_bulk1[i] = min_z_pore[i] / num_increments
        z_incr_bulk2[i] = (traj_file.box_size[2] - max_z_pore[i]) / num_increments

        ddict.printLog("Increment distance CNT: %0.3f \u00C5" % (z_incr_CNT[i]))
        ddict.printLog("Increment distance bulk1: %0.3f \u00C5" % (z_incr_bulk1[i]))
        ddict.printLog("Increment distance bulk2: %0.3f \u00C5" % (z_incr_bulk2[i]))

        z_bin_edges_pore[i] = np.linspace(CNT_atoms["z"].min(), CNT_atoms["z"].max(), num_increments + 1)
        z_bin_edges_bulk1[i] = np.linspace(0, min_z_pore[i], num_increments + 1)
        z_bin_edges_bulk2[i] = np.linspace(max_z_pore[i], traj_file.box_size[2], num_increments + 1)
        z_bin_edges[i] = np.concatenate((z_bin_edges_bulk1[i], z_bin_edges_pore[i], z_bin_edges_bulk2[i]))
        z_bin_edges[i] = np.unique(z_bin_edges[i])
        num_increments = len(z_bin_edges[i]) - 1

        ddict.printLog("\nTotal number of increments: %d" % (num_increments))
        ddict.printLog("Number of edges: %s" % len((z_bin_edges[i])))
        z_bin_labels[i] = np.arange(1, len(z_bin_edges[i]), 1)

    maxdisp_atom_dist = 0
    which_element_radii = ddict.get_input(
        "Do you want to use the van der Waals radii (1) or the covalent radii (2) "
        "of the elements to calculate the acccessible volume? [1/2] ",
        args,
        "int",
    )
    if which_element_radii == 1:
        ddict.printLog("-> Using van der Waals radii.")
        element_radii = ddict.dict_vdW()
    elif which_element_radii == 2:
        ddict.printLog("-> Using covalent radii.")
        element_radii = ddict.dict_covalent()

    # Make new dataframe with the number of frames of the trajectory.
    zdens_df_dummy = pd.DataFrame(np.arange(1, traj_file.number_of_frames + 1), columns=["Frame"])
    for i in range(num_increments):
        zdens_df_dummy["Bin %d" % (i + 1)] = 0
        zdens_df_dummy = zdens_df_dummy.copy()
    zdens_df = zdens_df_dummy.copy()

    # Prepare output dict
    outputdict = inputdict
    outputdict["element_radii"] = element_radii
    outputdict["maxdisp_atom_dist"] = maxdisp_atom_dist
    outputdict["z_incr_CNT"] = z_incr_CNT
    outputdict["z_incr_bulk1"] = z_incr_bulk1
    outputdict["z_incr_bulk2"] = z_incr_bulk2
    outputdict["z_bin_edges_pore"] = z_bin_edges_pore
    outputdict["z_bin_edges_bulk1"] = z_bin_edges_bulk1
    outputdict["z_bin_edges_bulk2"] = z_bin_edges_bulk2
    outputdict["z_bin_edges"] = z_bin_edges
    outputdict["z_bin_labels"] = z_bin_labels
    outputdict["zdens_df"] = zdens_df
    outputdict["num_increments"] = num_increments

    return outputdict


def axial_density_analysis(inputdict, traj_file, molecules):

    num_increments = inputdict["num_increments"]
    zdens_df = inputdict["zdens_df"]
    z_bin_edges = inputdict["z_bin_edges"]
    z_bin_labels = inputdict["z_bin_labels"]
    element_radii = inputdict["element_radii"]
    maxdisp_atom_dist = inputdict["maxdisp_atom_dist"]
    maxdisp_atom_row = inputdict["maxdisp_atom_row"]

    counter = inputdict["counter"]
    split_frame = inputdict["split_frame"]
    CNT_centers = molecules.CNT_centers
    max_z_pore = molecules.max_z_pore
    min_z_pore = molecules.min_z_pore

    split_frame = ut.wrapping_coordinates(traj_file.box_size, split_frame)

    # For now we concentate the edges/bins (This has to be changed in the future for the analysis on multiple CNTs).
    z_bin_edges = np.ravel(z_bin_edges)
    z_bin_labels = np.ravel(z_bin_labels)

    # Make an array from split_frame['Z'].
    split_frame["Z_bin"] = pd.cut(split_frame["Z"].astype(float).values, bins=z_bin_edges, labels=z_bin_labels)

    # Add all masses of the atoms in each bin to the corresponding bin. Then add a new first column with the index+1 of
    # the bin.
    zdens_df_temp = (
        split_frame.groupby(pd.cut(split_frame["Z"].astype(float), z_bin_edges))["Mass"]
        .sum()
        .reset_index(name="Weighted_counts")
    )
    # Calculate the number of atoms in each bin (no weighting).
    zdens_df_temp["Counts"] = (
        split_frame.groupby(pd.cut(split_frame["Z"].astype(float), z_bin_edges))["Mass"]
        .count()
        .reset_index(name="Counts")["Counts"]
    )
    zdens_df_temp = pd.DataFrame(zdens_df_temp)
    zdens_df_temp.insert(0, "Bin", zdens_df_temp.index + 1)

    # Write the results into the zdens_df dataframe. The row is defined by the frame number.
    for i in range(num_increments):
        zdens_df.loc[counter, "Bin %d" % (i + 1)] = zdens_df_temp.loc[i, "Weighted_counts"]

    # TUBE SECTION -> displacement for accessible volume.
    # All atoms are dropped which have a z coordinate larger than the maximum z coordinate of the CNT or smaller than
    # the minimum z coordinate of the CNT.
    # instead of checking the global coordinates, we consider the PBC and check the modulus of the box_size
    split_frame["X"] = split_frame["X"].astype(float) % traj_file.box_size[0]
    split_frame["Y"] = split_frame["Y"].astype(float) % traj_file.box_size[1]
    split_frame["Z"] = split_frame["Z"].astype(float) % traj_file.box_size[2]

    # we need to also account this for the max and min z_pore values
    max_z_pore[0] = max_z_pore[0] % traj_file.box_size[2]
    min_z_pore[0] = min_z_pore[0] % traj_file.box_size[2]

    # we also need to do this for the CNT_centers
    CNT_centers[0][0] = CNT_centers[0][0] % traj_file.box_size[0]
    CNT_centers[0][1] = CNT_centers[0][1] % traj_file.box_size[1]

    # if the pore is split over the periodic boundary (pot. because of modulo operation)
    if min_z_pore[0] > max_z_pore[0]:
        # Split the selection into two parts
        part1 = split_frame[split_frame["Z"].astype(float) >= min_z_pore[0]]
        part2 = split_frame[split_frame["Z"].astype(float) <= max_z_pore[0]]
        split_frame = pd.concat([part1, part2])
    else:
        split_frame = split_frame[split_frame["Z"].astype(float) <= max_z_pore[0]]
        split_frame = split_frame[split_frame["Z"].astype(float) >= min_z_pore[0]]

    # Calculate the radial density function with the remaining atoms.
    split_frame["X_adjust"] = split_frame["X"].astype(float) - CNT_centers[0][0]
    split_frame["Y_adjust"] = split_frame["Y"].astype(float) - CNT_centers[0][1]

    # While caluclating the (most displaced) distance, add the elements radius.
    split_frame["Distance"] = np.sqrt(split_frame["X_adjust"] ** 2 + split_frame["Y_adjust"] ** 2) + split_frame[
        "Atom"
    ].map(element_radii)
    if split_frame["Distance"].max() > maxdisp_atom_dist:
        maxdisp_atom_dist = split_frame["Distance"].max()
        # Save the complete info of the most displaced atom.
        maxdisp_atom_row = split_frame.loc[split_frame["Distance"].idxmax()]

        # Prepare output dict
    outputdict = inputdict
    outputdict["maxdisp_atom_dist"] = maxdisp_atom_dist
    outputdict["zdens_df"] = zdens_df
    outputdict["maxdisp_atom_row"] = maxdisp_atom_row
    outputdict["z_bin_edges"] = z_bin_edges
    outputdict["z_bin_labels"] = z_bin_labels

    return outputdict


def axial_density_processing(inputdict):
    tuberadii = inputdict["tuberadii"]
    CNT_centers = inputdict["CNT_centers"]
    length_pore = inputdict["length_pore"]
    CNT_length = length_pore[0]
    maxdisp_atom_row = inputdict["maxdisp_atom_row"]
    zdens_df = inputdict["zdens_df"]
    number_of_frames = inputdict["number_of_frames"]
    bin_edges = inputdict["z_bin_edges"]
    num_increments = inputdict["num_increments"]
    args = inputdict["args"]
    min_z_pore = inputdict["min_z_pore"]
    max_z_pore = inputdict["max_z_pore"]
    box_size = inputdict["box_size"]

    results_zd_df = pd.DataFrame(zdens_df.iloc[:, 1:].sum(axis=0) / number_of_frames)
    results_zd_df.columns = ["Mass"]
    results_zd_df["Bin_lowedge"] = bin_edges[:-1]
    results_zd_df["Bin_highedge"] = bin_edges[1:]

    # The center of the bin is the average of its bin edges.
    results_zd_df["Bin_center"] = (bin_edges[1:] + bin_edges[:-1]) / 2
    which_radius = ddict.get_input(
        "Do you want to use the accessible radius (1) or the CNT radius (2) to compute the increments' volume? "
        "[1/2] ",
        args,
        "string",
    )
    if maxdisp_atom_row is None:
        ddict.printLog("None of the analyzed atoms were found in the tube")
        used_radius = 0
    else:
        ddict.printLog(
            "\nMaximal displaced atom: %s (%0.3f|%0.3f|%0.3f)"
            % (
                maxdisp_atom_row["Atom"],
                float(maxdisp_atom_row["X"]),
                float(maxdisp_atom_row["Y"]),
                float(maxdisp_atom_row["Z"]),
            )
        )
        accessible_radius = maxdisp_atom_row["Distance"]
        ddict.printLog("Maximal displacement: %0.3f \u00C5" % accessible_radius)
        ddict.printLog("Accessible volume: %0.3f \u00C5\u00B3" % (math.pi * accessible_radius**2 * CNT_length))
        if which_radius == "1":
            used_radius = accessible_radius
        if which_radius == "2":
            used_radius = tuberadii[0]

    # Calculate the volume of each bin. For the bulk sections, the volume is defined by the x and y values of the
    # simbox, multiplied by the bin width. For the tube sections, the volume is defined by the pi*r^2*bin width.
    vol_increment = []
    for i in range(num_increments):
        if bin_edges[i] < min_z_pore or bin_edges[i + 1] > max_z_pore:
            vol_increment.append(box_size[0] * box_size[1] * (bin_edges[i + 1] - bin_edges[i]))
        else:
            vol_increment.append(math.pi * (used_radius**2) * (bin_edges[i + 1] - bin_edges[i]))

    # Add a new first column with the index+1 of the bin. Add the volume, calculate the density.
    results_zd_df.reset_index(drop=True, inplace=True)
    results_zd_df.insert(0, "Bin", results_zd_df.index + 1)
    results_zd_df["Volume"] = vol_increment
    results_zd_df["Density [u/Ang^3]"] = results_zd_df["Mass"] / results_zd_df["Volume"]

    # Mirror the data.
    mirror_dens_CNT = ddict.get_input(
        "Do you want to mirror the results inside the CNT to increase the binning? (y/n) ", args, "string"
    )
    if mirror_dens_CNT == "y":

        # First identify all the bins that are inside the CNT.
        inside_CNT = []
        for i in range(num_increments):
            if bin_edges[i] >= min_z_pore and bin_edges[i + 1] <= max_z_pore:
                inside_CNT.append(i)
        # Make dataframe with one column named 'Density [u/Ang^3]' and the same number of rows as the number of bins
        # inside the CNT.
        inside_CNT_df = pd.DataFrame()
        inside_CNT_df["Density [u/Ang^3]"] = results_zd_df["Density [u/Ang^3]"][inside_CNT]

        # Take the first and last bin that are inside the CNT and average the density of the two bins. Then do this for
        # the second and second last bin, and so on.
        for i in range(int(len(inside_CNT))):
            inside_CNT_df["Density [u/Ang^3]"][inside_CNT[i]] = (
                (
                    results_zd_df["Density [u/Ang^3]"][inside_CNT[i]]
                    + results_zd_df["Density [u/Ang^3]"][inside_CNT[-i - 1]]
                )
                / 2
            ).copy()
            results_zd_df.at[inside_CNT[i], "Density [u/Ang^3]"] = inside_CNT_df["Density [u/Ang^3]"][inside_CNT[i]]
            results_zd_df.at[inside_CNT[-i - 1], "Density [u/Ang^3]"] = inside_CNT_df["Density [u/Ang^3]"][
                inside_CNT[i]
            ]
        ddict.printLog("")
        ddict.printLog(results_zd_df)
        ddict.printLog("")

    # Calculate the density in g/cm^3.
    results_zd_df["Density [g/cm^3]"] = results_zd_df["Density [u/Ang^3]"] * 1.66053907

    center_to_zero = ddict.get_input(
        "Do you want to set the center of the simulation box to zero? (y/n) ", args, "string"
    )
    if center_to_zero == "y":
        results_zd_df["Bin_center"] = results_zd_df["Bin_center"] - CNT_centers[0][2]

    # Plot the data.
    zdens_data_plot = ddict.get_input("Do you want to plot the data? (y/n) ", args, "string")
    if zdens_data_plot == "y":
        fig, ax = plt.subplots()
        ax.plot(
            results_zd_df["Bin_center"],
            results_zd_df["Density [g/cm^3]"],
            "-",
            label="Axial density function",
            color="black",
        )
        ax.set(
            xlabel="Distance from tube center [\u00C5]", ylabel="Density [g/cm\u00B3]", title="Axial density function"
        )
        ax.grid()
        fig.savefig("Axial_density.pdf")
        ddict.printLog("\nAxial density function saved as Axial_density.pdf")

    # Save the data.
    results_zd_df.to_csv("Axial_density.csv", sep=";", index=True, header=True, float_format="%.5f")
    ddict.printLog("Axial density data saved as Axial_density.csv")

    zdens_df.to_csv("Axial_mass_dist_raw.csv", sep=";", index=False, header=True, float_format="%.5f")
    ddict.printLog("\nRaw data saved as Axial_mass_dist_raw.csv")


# 3D density analysis
""" What this function is about:

    This function is used to calculate the 3D density of the system.
    First it creates a 3D grid with the dimensions of the simulation box.
    The incrementation is set by the user.
    Then the atoms are assigned to the grid.
    In the end we have a 3D grid with the density information for each grid point for each individual atom/molecule.
    This data can then be used to calculate the density profile of the system.
    Visualisation is done with a cube file.
    """


def density_analysis_prep(inputdict, traj_file, molecules):
    # call the grid generator function
    inputdict = ut.grid_generator(inputdict)

    # first get the inputdict
    x_incr = inputdict["x_incr"]
    y_incr = inputdict["y_incr"]
    z_incr = inputdict["z_incr"]
    x_mesh = inputdict["x_mesh"]
    y_mesh = inputdict["y_mesh"]
    z_mesh = inputdict["z_mesh"]

    # array which we will later use to set up the gauss cube file.
    cube_array = np.zeros((x_incr * y_incr * z_incr))

    # kdtree from all grid points
    # for this we need to get the coordinates of all grid points, we simply use the x_mesh, y_mesh and z_mesh arrays
    # for this
    grid_points = np.vstack((x_mesh.flatten(), y_mesh.flatten(), z_mesh.flatten())).T
    grid_points_tree = scipy.spatial.KDTree(grid_points)

    # We also store a list with as many entries as there are grid points. Each entry is a list with the atom labels of
    # the atoms which are in the grid point.
    # We need this to calculate the density of each grid point.
    grid_points_atom_labels = [[] for i in range(len(grid_points))]

    # Also set up a list for each bin. It contains the information about how often each atom label is present in the
    # bin.
    grid_point_chunk_atom_labels = [[] for i in range(len(grid_points))]

    outputdict = inputdict
    outputdict["cube_array"] = cube_array
    outputdict["grid_points_tree"] = grid_points_tree
    outputdict["grid_point_atom_labels"] = grid_points_atom_labels
    outputdict["analysis_counter"] = 0
    outputdict["grid_point_chunk_atom_labels"] = grid_point_chunk_atom_labels

    return outputdict


def chunk_processing(inputdict):
    grid_point_atom_labels = inputdict["grid_point_atom_labels"]

    # Initialize 'grid_point_chunk_atom_labels' as a list of Counter objects if not already done
    if "grid_point_chunk_atom_labels" not in inputdict or not all(
        isinstance(c, Counter) for c in inputdict["grid_point_chunk_atom_labels"]
    ):
        inputdict["grid_point_chunk_atom_labels"] = [Counter() for _ in range(len(grid_point_atom_labels))]

    # Loop over all grid points and update the counts
    for i, atom_labels in enumerate(grid_point_atom_labels):
        inputdict["grid_point_chunk_atom_labels"][i].update(atom_labels)

    # Convert the Counter objects to a string format for easier viewing
    grid_point_chunk_atom_labels_str = []
    for counter in inputdict["grid_point_chunk_atom_labels"]:
        label_counts_str = ", ".join(f"{count}*{label}" for label, count in counter.items())
        grid_point_chunk_atom_labels_str.append(label_counts_str)

    # Update the inputdict with the string formatted list
    inputdict["grid_point_chunk_atom_labels_str"] = grid_point_chunk_atom_labels_str

    # Now reset the grid_point_atom_labels
    inputdict["grid_point_atom_labels"] = [[] for i in range(len(grid_point_atom_labels))]

    return inputdict


def density_analysis_analysis(inputdict, traj_file, molecules):
    split_frame = inputdict["split_frame"]
    box_size = traj_file.box_size

    cube_array = inputdict["cube_array"]
    grid_points_tree = inputdict["grid_points_tree"]
    grid_point_atom_labels = inputdict["grid_point_atom_labels"]
    analysis_counter = inputdict["analysis_counter"]

    # first wrap the coordinates
    split_frame = ut.wrapping_coordinates(box_size, split_frame)

    # get the coordinates of the split_frame
    split_frame_coords = np.array(split_frame[["X", "Y", "Z"]])
    split_frame_coords = split_frame_coords.astype(float)

    # find the corresponding grid point for each atom
    closest_grid_point_dist, closest_grid_point_idx = grid_points_tree.query(split_frame_coords)

    # print the corresponding atom label to the cube_array. If there is already an atom label, add the new atom
    # label to the existing one.
    cube_array[closest_grid_point_idx] = split_frame["Mass"].values

    # add the atom label to the grid_point_atom_labels list
    for i in range(len(split_frame)):
        grid_point_atom_labels[closest_grid_point_idx[i]].append(split_frame["Label"].values[i])

    # add the grid_point_atom_labels to the inputdict
    inputdict["grid_point_atom_labels"] = grid_point_atom_labels

    analysis_counter += 1
    outputdict = inputdict
    # for every 500 steps do the chunk processing
    if analysis_counter == 500:
        inputdict = chunk_processing(inputdict)
        analysis_counter = 0
    else:
        outputdict["grid_point_atom_labels"] = grid_point_atom_labels

    outputdict["cube_array"] = cube_array
    outputdict["analysis_counter"] = analysis_counter

    return outputdict


def density_analysis_processing(inputdict):
    # do the last chunk processing
    inputdict = chunk_processing(inputdict)

    # first get the inputdict
    unique_molecule_frame = inputdict["unique_molecule_frame"]
    grid_point_chunk_atom_labels_str = inputdict["grid_point_chunk_atom_labels_str"]
    number_of_frames = inputdict["number_of_frames"]
    # add a new column with the atom masses. The Elements are stored in lists in the Atoms_sym column.
    unique_molecule_frame["Masses"] = unique_molecule_frame["Atoms_sym"].apply(ut.symbols_to_masses)

    # now write a new file with the masses of each atom grouped with the corresponding label, like this:
    # example = ['H1':1.008, 'H2':1.008,...]
    list_of_masses = []
    for index, row in unique_molecule_frame.iterrows():
        list_of_masses.append(dict(zip(row["Labels"], row["Masses"])))

    # so we want to change the grid_point_chunk_atom_labels_str list to a list of lists
    # so we split the strings in the list and then convert the strings to lists
    grid_point_chunk_atom_labels_str = [x.split(", ") for x in grid_point_chunk_atom_labels_str]
    grid_point_chunk_atom_labels_str = [list(map(lambda x: x.split("*"), y)) for y in grid_point_chunk_atom_labels_str]
    # print(grid_point_chunk_atom_labels_str)

    # Calculate the volume of each grid point
    grid_volume = inputdict["x_incr_dist"] * inputdict["y_incr_dist"] * inputdict["z_incr_dist"]
    print("Volume of each grid point: %0.3f \u00C5\u00B3" % grid_volume)

    # Initialize a list to store the density of each grid point
    grid_point_densities = []

    # Iterate over each grid point's atom label counts
    for grid_point in grid_point_chunk_atom_labels_str:
        total_mass = 0
        grid_point_density = 0

        # Iterate over each atom label and its count in the grid point
        for label_count_pair in grid_point:
            # Ensure that the label_count_pair has at least two elements
            if len(label_count_pair) >= 2:
                label = label_count_pair[1]
                count = int(label_count_pair[0])

                # Retrieve the mass of the atom label from each molecule's dictionary and add it to the total mass
                for molecule_mass_dict in list_of_masses:
                    if label in molecule_mass_dict:
                        total_mass += molecule_mass_dict[label] * count
                        break

        grid_point_density = total_mass / grid_volume
        # divide by the number of frames
        grid_point_density = grid_point_density / number_of_frames
        grid_point_densities.append(grid_point_density)

    inputdict["grid_point_densities"] = grid_point_densities
    ut.write_cube_file(inputdict, filename="density.cube")

    # now extract the density profiles along x,y and z axis using the grid_point_densities list
    # first reshape the list to a 3D array
    grid_point_densities = np.array(grid_point_densities)
    grid_point_densities = grid_point_densities.reshape(inputdict["x_incr"], inputdict["y_incr"], inputdict["z_incr"])

    # now extract the density profiles.
    x_dens_profile = np.sum(grid_point_densities, axis=(1, 2))
    y_dens_profile = np.sum(grid_point_densities, axis=(0, 2))
    z_dens_profile = np.sum(grid_point_densities, axis=(0, 1))

    sum_gp_x = int(inputdict["y_incr"] * inputdict["z_incr"])
    sum_gp_y = int(inputdict["x_incr"] * inputdict["z_incr"])
    sum_gp_z = int(inputdict["x_incr"] * inputdict["y_incr"])

    # divide each value in the density profiles by the number of grid points in the corresponding direction
    x_dens_profile = x_dens_profile / sum_gp_x
    y_dens_profile = y_dens_profile / sum_gp_y
    z_dens_profile = z_dens_profile / sum_gp_z

    # Calculate total mass per slice
    total_mass_x = x_dens_profile * sum_gp_x * grid_volume
    total_mass_y = y_dens_profile * sum_gp_y * grid_volume
    total_mass_z = z_dens_profile * sum_gp_z * grid_volume

    # Calculate volume of each slice
    volume_x = sum_gp_x * grid_volume
    volume_y = sum_gp_y * grid_volume
    volume_z = sum_gp_z * grid_volume

    # now we print the density profiles to dataframes. The first column is the x (y,z) coordinate of the grid point,
    # the second column is the density. Also calculate the density in g/cm^3.
    # the last columns print the total mass per slice and the volume of the slice.
    x_dens_profile_df = pd.DataFrame()
    x_dens_profile_df["x"] = inputdict["x_grid"]
    x_dens_profile_df["Density [u/Ang^3]"] = x_dens_profile
    x_dens_profile_df["Density [g/cm^3]"] = x_dens_profile_df["Density [u/Ang^3]"] * 1.66053907
    x_dens_profile_df["Total Mass [u]"] = total_mass_x
    x_dens_profile_df["Volume [Ang^3]"] = volume_x

    y_dens_profile_df = pd.DataFrame()
    y_dens_profile_df["y"] = inputdict["y_grid"]
    y_dens_profile_df["Density [u/Ang^3]"] = y_dens_profile
    y_dens_profile_df["Density [g/cm^3]"] = y_dens_profile_df["Density [u/Ang^3]"] * 1.66053907
    y_dens_profile_df["Total Mass [u]"] = total_mass_y
    y_dens_profile_df["Volume [Ang^3]"] = volume_y

    z_dens_profile_df = pd.DataFrame()
    z_dens_profile_df["z"] = inputdict["z_grid"]
    z_dens_profile_df["Density [u/Ang^3]"] = z_dens_profile
    z_dens_profile_df["Density [g/cm^3]"] = z_dens_profile_df["Density [u/Ang^3]"] * 1.66053907
    z_dens_profile_df["Total Mass [u]"] = total_mass_z
    z_dens_profile_df["Volume [Ang^3]"] = volume_z
    # now we save the dataframes to csv files
    x_dens_profile_df.to_csv("x_dens_profile.csv", sep=";", index=False, header=True, float_format="%.5f")
    y_dens_profile_df.to_csv("y_dens_profile.csv", sep=";", index=False, header=True, float_format="%.5f")
    z_dens_profile_df.to_csv("z_dens_profile.csv", sep=";", index=False, header=True, float_format="%.5f")

    # now we plot the density profiles
    # first the x direction
    fig, ax = plt.subplots()
    ax.plot(x_dens_profile_df["x"], x_dens_profile_df["Density [g/cm^3]"], "-", label="Density profile", color="black")
    ax.set(xlabel="x [\u00C5]", ylabel="Density [g/cm\u00B3]", title="Density profile")
    ax.grid()
    fig.savefig("x_density_profile.pdf")

    # now the y direction
    fig, ax = plt.subplots()
    ax.plot(y_dens_profile_df["y"], y_dens_profile_df["Density [g/cm^3]"], "-", label="Density profile", color="black")
    ax.set(xlabel="y [\u00C5]", ylabel="Density [g/cm\u00B3]", title="Density profile")
    ax.grid()
    fig.savefig("y_density_profile.pdf")

    # now the z direction
    fig, ax = plt.subplots()
    ax.plot(z_dens_profile_df["z"], z_dens_profile_df["Density [g/cm^3]"], "-", label="Density profile", color="black")
    ax.set(xlabel="z [\u00C5]", ylabel="Density [g/cm\u00B3]", title="Density profile")
    ax.grid()
    fig.savefig("z_density_profile.pdf")
