# The program is written by Leonard Dick, 2023

import math
import os
import sys
import time

import numpy as np
import pandas as pd
from prettytable import PrettyTable

import conan.defdict as ddict
from conan.analysis_modules import traj_info


# Information on the trajectory / cutting the frames into chunks
def traj_chunk_info(id_frame, args):
    # GENERAL INFORMATION ON CHUNKS
    ddict.printLog("-> Reading the trajectory.\n")
    trajectory_file_size = os.path.getsize(args["trajectoryfile"])
    # Calculate how many atoms each frame has.
    number_of_atoms = len(id_frame)
    # Calculate how many lines the trajectory file has.
    with open(args["trajectoryfile"]) as f:
        number_of_lines = sum(1 for i in f)

    lines_per_frame = 0
    # Calculate how many frames the trajectory file has.
    if args["trajectoryfile"].endswith(".xyz") or args["trajectoryfile"].endswith(".pdb"):
        lines_per_frame = number_of_atoms + 2
    elif args["trajectoryfile"].endswith(".lammpstrj") or args["trajectoryfile"].endswith(".lmp"):
        lines_per_frame = number_of_atoms + 9

    number_of_frames = int(number_of_lines / lines_per_frame)

    # Calculate how many bytes each line of the trajectory file has.
    bytes_per_line = trajectory_file_size / (number_of_lines)
    # The number of lines in a chunk. Each chunk is roughly 50 MB large.
    chunk_size = int(100000000 / ((lines_per_frame) * bytes_per_line))
    # The number of chunks (always round up).
    number_of_chunks = math.ceil(number_of_frames / chunk_size)
    # The number of frames in the last chunk.
    last_chunk_size = number_of_frames - (number_of_chunks - 1) * chunk_size
    number_of_bytes_per_chunk = chunk_size * (lines_per_frame) * bytes_per_line
    number_of_lines_per_chunk = chunk_size * (lines_per_frame)
    number_of_lines_last_chunk = last_chunk_size * (lines_per_frame)
    # Table with the information on the trajectory file.
    table = PrettyTable(["", "Trajectory", "Chunk(%d)" % (number_of_chunks)])
    table.add_row(
        [
            "Size in MB",
            "%0.1f" % (trajectory_file_size / 1000000),
            "%0.1f (%0.1f)"
            % (number_of_bytes_per_chunk / 1000000, last_chunk_size * (lines_per_frame) * bytes_per_line / 1000000),
        ]
    )
    table.add_row(["Frames", number_of_frames, "%d(%d)" % (chunk_size, last_chunk_size)])
    table.add_row(["Lines", number_of_lines, number_of_lines_per_chunk])
    ddict.printLog(table)
    ddict.printLog("")

    return (
        number_of_frames,
        number_of_lines_per_chunk,
        number_of_lines_last_chunk,
        number_of_chunks,
        chunk_size,
        last_chunk_size,
    )


# MAIN
def analysis_opt(maindict) -> None:

    # id_frame, CNT_centers, box_size, tuberadii, min_z_pore, max_z_pore, length_pore, Walls_positions, args
    CNT_centers = maindict["CNT_centers"]
    args = maindict["args"]

    # General analysis options (Is the whole trajectory necessary or just the first frame?).
    ddict.printLog("(1) Produce xyz files of the simulation box or pore structure.")
    ddict.printLog("(2) Analyze the trajectory.")
    choice = int(ddict.get_input("Picture or analysis mode?: ", args, "int"))
    ddict.printLog("")
    if choice == 1:
        ddict.printLog("PICTURE mode.\n", color="red")
        generating_pictures(maindict)
    elif choice == 2:
        ddict.printLog("ANALYSIS mode.\n", color="red")
        if len(CNT_centers) >= 0:
            trajectory_analysis(maindict)
        else:
            ddict.printLog("-> No CNTs detected.", color="red")
    else:
        ddict.printLog("-> The choice is not known.")
    ddict.printLog("")


# Generating pictures.
def generating_pictures(maindict) -> None:

    id_frame = maindict["id_frame"]
    CNT_centers = maindict["CNT_centers"]
    box_size = maindict["box_size"]
    args = maindict["args"]
    ddict.printLog("(1) Produce xyz file of the whole simulation box.")
    ddict.printLog("(2) Produce xyz file of empty pore structure.")
    ddict.printLog("(3) Produce xyz file of the pore structures' tube.")
    analysis1_choice = int(ddict.get_input("What do you want to do?: ", args, "int"))

    if analysis1_choice == 1:
        ddict.printLog("\n-> Pics of box.")
        # Write the xyz file. The first line has the number of atoms (column in the first_drame), the second line is
        # empty.
        id_frame_print = open("simbox_frame.xyz", "w")
        id_frame_print.write("%d\n#Made with CONAN\n" % len(id_frame))
        for index, row in id_frame.iterrows():
            id_frame_print.write("%s\t%0.3f\t%0.3f\t%0.3f\n" % (row["Element"], row["x"], row["y"], row["z"]))
        id_frame_print.close()
        ddict.printLog("-> Saved simulation box as simbox_frame.xyz.")

    elif analysis1_choice == 2:
        ddict.printLog("\n-> Pics of pore structure(s).")
        # Loop over the number of entries in the tuberadius numpy array.
        for i in range(len(CNT_centers)):
            # Create a dataframe with the just atoms of the respective pore structure. Extract the atoms from the
            # id_frame. They are labeled Pore1, Pore2... in the Struc column.
            CNT_atoms_pic = id_frame.loc[id_frame["Struc"] == "Pore%d" % (i + 1)]
            # Remove all columns except the Element, x, y, and z columns.
            ddict.printLog(CNT_atoms_pic)
            CNT_atoms_pic = CNT_atoms_pic.drop(["Charge", "Struc", "CNT", "Molecule"], axis=1)
            add_centerpoint = ddict.get_input("Add the center point of the CNT to the file? [y/n] ", args, "string")
            if add_centerpoint == "y":
                # Add the center point of the CNT to the dataframe, labeled as X in a new row.
                CNT_atoms_pic.loc[len(CNT_atoms_pic.index)] = [
                    "X",
                    CNT_centers[0][0],
                    CNT_centers[0][1],
                    CNT_centers[0][2],
                ]
            CNT_atoms_print = open(f"pore{i + 1}.xyz", "w")
            CNT_atoms_print.write("%d\n#Made with CONAN\n" % len(CNT_atoms_pic))
            # Print the CNT_atoms dataframe to a xyz file. Just the atoms, x, y, and z column.
            for index, row in CNT_atoms_pic.iterrows():
                CNT_atoms_print.write("%s\t%0.3f\t%0.3f\t%0.3f\n" % (row["Element"], row["x"], row["y"], row["z"]))
            CNT_atoms_print.close()
            ddict.printLog(f"-> saved as pore{i + 1}.xyz")

    elif analysis1_choice == 3:
        ddict.printLog("\n-> Tube pictures.")

        for i in range(len(CNT_centers)):
            # Create a dataframe with the just atoms of the respective pore structure. Extract the atoms from the
            # id_frame.
            CNT_atoms_pic = pd.DataFrame(id_frame.loc[id_frame["CNT"] == i + 1])
            # Remove all columns except the Element, x, y, and z columns.
            CNT_atoms_pic = CNT_atoms_pic.drop(["Charge", "Struc", "CNT", "Molecule"], axis=1)
            add_liquid = ddict.get_input(f"Add liquid which is inside the CNT{i + 1}? [y/n] ", args, "string")

            if add_liquid == "y":
                add_liquid2 = ddict.get_input(
                    "Add all confined atoms (1), or entire molecules (2) ? [1/2] ", args, "int"
                )

                if add_liquid2 == 1:
                    # Scan the id_frame and add all atoms which are inside the tube to the tube_atoms dataframe.
                    for index, row in id_frame.iterrows():
                        if (
                            row["Struc"] == "Liquid"
                            and row["z"] <= CNT_atoms_pic["z"].max()
                            and row["z"] >= CNT_atoms_pic["z"].min()
                        ):
                            # Add the row to the tube_atoms dataframe.
                            CNT_atoms_pic.loc[index] = [row["Element"], row["x"], row["y"], row["z"]]

                elif add_liquid2 == 2:
                    # Do the molecule recognition.
                    ddict.printLog("-> Molecule recognition.")
                    id_frame, unique_molecule_frame = traj_info.molecule_recognition(id_frame, box_size)
                    id_frame = id_frame.drop(["Charge", "Label", "CNT"], axis=1)
                    # Add the Molecule column to the CNT_atoms_pic dataframe.
                    CNT_atoms_pic["Molecule"] = np.nan
                    # Scan the id_frame and add all atoms which are inside the tube to the tube_atoms dataframe.
                    for index, row in id_frame.iterrows():
                        if (
                            row["Struc"] == "Liquid"
                            and row["z"] <= CNT_atoms_pic["z"].max()
                            and row["z"] >= CNT_atoms_pic["z"].min()
                        ):
                            # Add the row to the tube_atoms dataframe.
                            CNT_atoms_pic.loc[index] = [row["Element"], row["x"], row["y"], row["z"], row["Molecule"]]

                    # List the molecules which are inside the tube.
                    mol_list = []
                    mol_list.append(CNT_atoms_pic["Molecule"].unique())
                    tube_atoms_mol = pd.DataFrame(columns=["Element", "x", "y", "z", "Molecule"])
                    mol_list = mol_list[0]
                    # Scan the id_frame and add all atoms which are in the mol_list to the tube_atoms_mol dataframe.
                    for index, row in id_frame.iterrows():
                        if row["Molecule"] in mol_list:
                            # Add the row to the tube_atoms dataframe.
                            tube_atoms_mol.loc[index] = [row["Element"], row["x"], row["y"], row["z"], row["Molecule"]]
                    # Append the tube_atoms_mol dataframe to the tube_atoms_pic dataframe.
                    CNT_atoms_pic = pd.concat([CNT_atoms_pic, tube_atoms_mol], ignore_index=True)

                    # Finally remove all duplicates from the tube_atoms_pic dataframe.
                    CNT_atoms_pic = CNT_atoms_pic.drop_duplicates(
                        subset=["Element", "x", "y", "z", "Molecule"], keep="first"
                    )

            else:
                add_centerpoint = ddict.get_input(
                    f"Add the center point of the CNT{i + 1} to the file? [y/n] ", args, "string"
                )
                if add_centerpoint == "y":
                    CNT_atoms_pic.loc[len(CNT_atoms_pic.index)] = [
                        "X",
                        CNT_centers[0][0],
                        CNT_centers[0][1],
                        CNT_centers[0][2],
                    ]

            tube_atoms_print = open(f"CNT{i + 1}.xyz", "w")
            tube_atoms_print.write("%d\n#Made with CONAN\n" % len(CNT_atoms_pic))

            for index, row in CNT_atoms_pic.iterrows():
                tube_atoms_print.write("%s\t%0.3f\t%0.3f\t%0.3f\n" % (row["Element"], row["x"], row["y"], row["z"]))
            tube_atoms_print.close()
            ddict.printLog(f"-> Saved as CNT{i + 1}.xyz")
    else:
        ddict.printLog("\nThe analysis you entered is not known.")
    ddict.printLog("")


# Analysis of the trajectory.
def trajectory_analysis(inputdict) -> None:

    id_frame = inputdict["id_frame"]
    box_size = inputdict["box_size"]
    args = inputdict["args"]

    # Analysis choice.
    ddict.printLog("(1) Calculate the radial density inside the CNT")
    ddict.printLog("(2) Calculate the radial charge density inside the CNT (if charges are provided)")
    ddict.printLog("(3) Calculate the radial velocity of the liquid in the CNT.")
    ddict.printLog("(4) Calculate the accessibe volume of the CNT")
    ddict.printLog("(5) Calculate the average density along the z axis of the simulation box")
    ddict.printLog("(6) Calculate the coordination number")
    ddict.printLog("(7) Calculate the distance between liquid and pore atoms")
    ddict.printLog("(8) Calculate the density of the liquid in the simulation box.")

    # ddict.printLog('(10) Calculate the occurrence of a specific atom in the simulation box.')
    analysis_choice2 = int(ddict.get_input("Which analysis should be conducted?:  ", args, "int"))
    analysis_choice2_options = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    if analysis_choice2 not in analysis_choice2_options:
        ddict.printLog("-> The analysis you entered is not known.")
        sys.exit(1)
    ddict.printLog("")

    # MOLECULAR RECOGNITION
    # Perform the molecule recognition by loading the module molidentifier.
    id_frame, unique_molecule_frame = traj_info.molecule_recognition(id_frame, box_size, args)
    species_max = id_frame["Species"].max()
    spec_molecule = 0
    spec_atom = []
    ddict.printLog("")
    analysis_spec_molecule = ddict.get_input(
        "Do you want to perform the analysis for a specific molecule kind? (y/n) ", args, "string"
    )
    if analysis_spec_molecule == "y":
        spec_molecule = int(ddict.get_input(f"Which species to analyze? (1-{species_max}) ", args, "int"))
        # Ask user for the atom type to analyze. Multiple options are possible, default is 'all'.
        spec_atom = ddict.get_input("Which atoms to analyze? [default:all] ", args, "string")

        if spec_atom == "" or spec_atom == "[default:all]":
            spec_atom = "all"

        # Get the atom type into a list.
        spec_atom = spec_atom.replace(", ", ",").split(",")
        ddict.printLog(f"\n-> Species {spec_molecule} and atom type {spec_atom} will be analyzed.\n")

    (
        number_of_frames,
        number_of_lines_per_chunk,
        number_of_lines_last_chunk,
        number_of_chunks,
        chunk_size,
        last_chunk_size,
    ) = traj_chunk_info(id_frame, args)

    # PREPERATION
    # Main loop preperation.
    Main_time = time.time()
    counter = 0

    # Get atomic masses.
    element_masses = ddict.dict_mass()

    # First it is necessary to get the molecule number of each atom to attach it to every frame later in the loop.
    molecule_id = id_frame["Molecule"].values
    molecule_species = id_frame["Species"].values
    molecule_structure = id_frame["Struc"].values
    molecule_label = id_frame["Label"].values

    CNT_atoms = id_frame[id_frame["CNT"].notnull()]

    # Analysis preperation.
    if analysis_choice2 == 1 or analysis_choice2 == 2:
        from conan.analysis_modules.rad_dens import raddens_prep as main_loop_preparation

        if analysis_choice2 == 1:
            from conan.analysis_modules.rad_dens import radial_density_analysis as analysis
        if analysis_choice2 == 2:
            from conan.analysis_modules.rad_dens import radial_charge_density_analysis as analysis
        from conan.analysis_modules.rad_dens import raddens_post_processing as post_processing

    if analysis_choice2 == 3:
        from conan.analysis_modules.rad_velocity import rad_velocity_analysis as analysis
        from conan.analysis_modules.rad_velocity import rad_velocity_prep as main_loop_preparation
        from conan.analysis_modules.rad_velocity import rad_velocity_processing as post_processing

    if analysis_choice2 == 4:
        from conan.analysis_modules.axial_dens import accessible_volume_analysis as analysis
        from conan.analysis_modules.axial_dens import accessible_volume_prep as main_loop_preparation
        from conan.analysis_modules.axial_dens import accessible_volume_processing as post_processing

    if analysis_choice2 == 5:
        from conan.analysis_modules.axial_dens import axial_density_analysis as analysis
        from conan.analysis_modules.axial_dens import axial_density_prep as main_loop_preparation
        from conan.analysis_modules.axial_dens import axial_density_processing as post_processing

    if analysis_choice2 == 6:
        from conan.analysis_modules.coordination_number import Coord_chunk_processing as chunk_processing
        from conan.analysis_modules.coordination_number import Coord_number_analysis as analysis
        from conan.analysis_modules.coordination_number import Coord_number_prep as main_loop_preparation
        from conan.analysis_modules.coordination_number import Coord_post_processing as post_processing

    if analysis_choice2 == 7:
        from axial_dens import distance_search_analysis as analysis
        from axial_dens import distance_search_prep as main_loop_preparation
        from axial_dens import distance_search_processing as post_processing

    if analysis_choice2 == 8:
        from axial_dens import density_analysis_analysis as analysis
        from axial_dens import density_analysis_prep as main_loop_preparation
        from axial_dens import density_analysis_processing as post_processing

    maindict = inputdict
    maindict["counter"] = counter
    maindict["unique_molecule_frame"] = unique_molecule_frame
    maindict["CNT_atoms"] = CNT_atoms
    maindict["maxdisp_atom_row"] = None
    maindict["maxdisp_atom_dist"] = 0
    maindict["number_of_frames"] = number_of_frames
    maindict["analysis_choice2"] = analysis_choice2
    maindict["do_xyz_analysis"] = "n"

    maindict = main_loop_preparation(maindict)

    if analysis_choice2 == 5:
        if maindict["do_xyz_analysis"] == "y":
            from coordination_number import Coord_number_xyz_analysis as analysis
            from coordination_number import Coord_xyz_chunk_processing as chunk_processing
            from coordination_number import Coord_xyz_post_processing as post_processing

    maindict["box_dimension"] = np.array(
        maindict["box_size"]
    )  # needed for fast calculation of minimal image convention

    # Ask if the analysis should be performed in a specific region
    maindict["regional"] = ddict.get_input(
        "Do you want the calculation to be performed in a specific region? [y/n] ", args, "string"
    )
    regions = [0] * 6
    if maindict["regional"] == "y":
        regions[0] = float(ddict.get_input("Enter minimum x-value ", args, "float"))
        regions[1] = float(ddict.get_input("Enter maximum x-value ", args, "float"))
        regions[2] = float(ddict.get_input("Enter minimum y-value ", args, "float"))
        regions[3] = float(ddict.get_input("Enter maximum y-value ", args, "float"))
        regions[4] = float(ddict.get_input("Enter minimum z-value ", args, "float"))
        regions[5] = float(ddict.get_input("Enter maximum z-value ", args, "float"))
    maindict["regions"] = regions

    # MAIN LOOP
    # Define which function to use reading the trajectory file. Pull definition from traj_info.py.
    if args["trajectoryfile"].endswith(".xyz"):
        from traj_info import xyz as run
    elif args["trajectoryfile"].endswith(".pdb"):
        from traj_info import pdb as run
    elif args["trajectoryfile"].endswith(".lmp") or args["trajectoryfile"].endswith(".lammpstrj"):
        from traj_info import lammpstrj as run

    # The trajectory xyz file is read in chunks of size chunk_size. The last chunk is smaller than the other chunks.
    trajectory = pd.read_csv(args["trajectoryfile"], chunksize=number_of_lines_per_chunk, header=None)
    chunk_number = 0
    # Loop over chunks.
    for chunk in trajectory:
        chunk_number = chunk_number + 1
        maindict["chunk_number"] = chunk_number
        print("")
        print("Chunk %d of %d" % (chunk_number, number_of_chunks))
        # Divide the chunk into individual frames. If the chunk is the last chunk, the number of frames is different.
        if chunk.shape[0] == number_of_lines_last_chunk:
            frames = np.split(chunk, last_chunk_size)
        else:
            frames = np.split(chunk, chunk_size)

        for frame in frames:

            # First load the frame into the function run() to get a dataframe. Then reset the index.
            split_frame = run(frame, element_masses, id_frame)
            split_frame.reset_index(drop=True, inplace=True)

            # Add the necessary columns to the dataframe.
            split_frame["Struc"] = molecule_structure
            split_frame["Molecule"] = molecule_id
            split_frame["Species"] = molecule_species
            split_frame["Label"] = molecule_label

            # Drop all CNT and carbon_wall atoms, just the Liquid atoms are needed for the analysis.
            split_frame = split_frame[split_frame["Struc"] == "Liquid"]
            split_frame = split_frame.drop(["Struc"], axis=1)

            # Drop the other atoms which are not needed for the analysis.
            if analysis_spec_molecule == "y":
                split_frame = split_frame[split_frame["Species"].astype(int) == int(spec_molecule)]
                # If the spec_atom list does not contain "all" then only the atoms in the list are kept.
                if spec_atom[0] != "all":
                    # If specific atoms are requested, only these atoms are kept.
                    split_frame = split_frame[split_frame["Label"].isin(spec_atom)]

            # Save the split_frame in maindict
            maindict["split_frame"] = split_frame
            maindict["counter"] = counter

            maindict = analysis(maindict)

            counter += 1
            print("Frame %d of %d" % (counter, number_of_frames), end="\r")

        # For memory intensive analyses (e.g. CN) we need to do the processing after every chunk
        if analysis_choice2 == 5:
            maindict = chunk_processing(maindict)

    ddict.printLog("")
    ddict.printLog("Finished processing the trajectory. %d frames were processed." % (counter))
    ddict.printLog("")

    # DATA PROCESSING
    post_processing(maindict)

    ddict.printLog("The main loop took %0.3f seconds to run." % (time.time() - Main_time))


if __name__ == "__main__":
    # ARGUMENTS
    args = ddict.read_commandline()
