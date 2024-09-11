# The program is written by Leonard Dick, 2023

import sys
import time

import numpy as np
import pandas as pd

import conan.defdict as ddict
from conan.analysis_modules import traj_info


# MAIN
def analysis_opt(traj_file, molecules, maindict) -> None:

    # General analysis options (Is the whole trajectory necessary or just the first frame?).
    ddict.printLog("(1) Produce xyz files of the simulation box or pore structure.")
    ddict.printLog("(2) Analyze the trajectory.")
    choice = int(ddict.get_input("Picture or analysis mode?: ", traj_file.args, "int"))
    ddict.printLog("")
    if choice == 1:
        ddict.printLog("PICTURE mode.\n", color="red")
        generating_pictures(traj_file, molecules)
    elif choice == 2:
        ddict.printLog("ANALYSIS mode.\n", color="red")
        if len(molecules.structure_data["CNT_centers"]) >= 0:
            trajectory_analysis(traj_file, molecules, maindict)
        else:
            ddict.printLog("-> No CNTs detected.", color="red")
    else:
        ddict.printLog("-> The choice is not known.")
    ddict.printLog("")


# Generating pictures.
def generating_pictures(traj_file, molecules) -> None:

    ddict.printLog("(1) Produce xyz file of the whole simulation box.")
    ddict.printLog("(2) Produce xyz file of empty pore structure.")
    ddict.printLog("(3) Produce xyz file of the pore structures' tube.")
    analysis1_choice = int(ddict.get_input("What do you want to do?: ", traj_file.args, "int"))

    if analysis1_choice == 1:
        ddict.printLog("\n-> xyz file of simulation box.")
        # Write the xyz file. The first line has the number of atoms (column in the first_frame).
        sort_species = ddict.get_input(
            "Do you want to sort the rows in a certain species order? [y/n]: ", traj_file.args, "string"
        )
        if sort_species == "y":
            species_order = ddict.get_input(
                "Enter the species in the order you want them to be sorted: ", traj_file.args, "string"
            )
            # They will be given as 3,2,1,4,5,6 for example
            species_order = species_order.split(",")
            # The entries in the species_order list are strings, so they need to be converted to integers
            species_order = [int(i) for i in species_order]
            # Convert the 'Species' column to a categorical type with the specified order
            traj_file.frame0["Species"] = pd.Categorical(
                traj_file.frame0["Species"], categories=species_order, ordered=True
            )
            # Sort the frame0 dataframe by species and molecule and then by the row index.
            # store the original index
            traj_file.frame0["index"] = traj_file.frame0.index
            traj_file.frame0 = traj_file.frame0.sort_values(by=["Species", "Molecule", "index"])
        else:
            species_order = None
            # Sort the frame0 dataframe by species and molecule
            traj_file.frame0 = traj_file.frame0.sort_values(by=["Species", "Molecule"])

        traj_file.frame0 = traj_file.frame0.drop("index", axis=1)
        print(traj_file.frame0)
        frame_print = open("simbox_frame.xyz", "w")
        frame_print.write("%d\n#Made with CONAN\n" % len(traj_file.frame0))
        for index, row in traj_file.frame0.iterrows():
            frame_print.write("%s\t%0.3f\t%0.3f\t%0.3f\n" % (row["Element"], row["x"], row["y"], row["z"]))
        frame_print.close()
        ddict.printLog("-> Saved simulation box as simbox_frame.xyz.")

    elif analysis1_choice == 2:
        ddict.printLog("\n-> Pics of pore structure(s).")
        # Loop over the number of entries in the tuberadius numpy array.
        for i in range(len(molecules.structure_data["CNT_centers"])):
            # Create a dataframe with the just atoms of the respective pore structure. Extract the atoms from the
            # traj_file.frame0. They are labeled Pore1, Pore2... in the Struc column.
            CNT_atoms_pic = traj_file.frame0.loc[traj_file.frame0["Struc"] == "Pore%d" % (i + 1)]
            # Remove all columns except the Element, x, y, and z columns.
            ddict.printLog(CNT_atoms_pic)
            CNT_atoms_pic = CNT_atoms_pic.drop(["Charge", "Struc", "CNT", "Molecule", "Label", "Species"], axis=1)
            add_centerpoint = ddict.get_input(
                "Add the center point of the CNT to the file? [y/n] ", traj_file.args, "string"
            )
            if add_centerpoint == "y":
                # Add the center point of the CNT to the dataframe, labeled as X in a new row.
                CNT_atoms_pic.loc[len(CNT_atoms_pic.index)] = [
                    "X",
                    molecules.structure_data["CNT_centers"][0][0],
                    molecules.structure_data["CNT_centers"][0][1],
                    molecules.structure_data["CNT_centers"][0][2],
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

        for i in range(len(molecules.structure_data["CNT_centers"])):
            # Dataframe with the atoms of the pore structure from traj_file.frame0.
            CNT_atoms_pic = pd.DataFrame(traj_file.frame0.loc[traj_file.frame0["CNT"] == i + 1])
            # Remove all unneeded columns except the Element, x, y, and z.
            CNT_atoms_pic = CNT_atoms_pic.drop(["Charge", "Struc", "CNT", "Molecule"], axis=1)
            add_liquid = ddict.get_input(f"Add liquid which is inside the CNT{i + 1}? [y/n] ", traj_file.args, "string")

            if add_liquid == "y":
                add_liquid2 = ddict.get_input(
                    "Add all confined atoms (1), or entire molecules (2) ? [1/2] ", traj_file.args, "int"
                )

                # if add_liquid2 == 1:
                # Scan the traj_file.frame0 and add all atoms which are inside the tube to the tube_atoms dataframe.
                CNT_atoms_pic["Molecule"] = np.nan
                for index, row in traj_file.frame0.iterrows():
                    if (
                        row["Struc"] == "Liquid"
                        and row["z"] <= CNT_atoms_pic["z"].max()
                        and row["z"] >= CNT_atoms_pic["z"].min()
                    ):
                        # Add the row to the tube_atoms dataframe.
                        CNT_atoms_pic.loc[index] = [
                            row["Element"],
                            row["x"],
                            row["y"],
                            row["z"],
                            row["Label"],
                            row["Species"],
                            row["Molecule"],
                        ]

                if add_liquid2 == 2:
                    # traj_file.frame0 = traj_file.frame0.drop(["Charge", "CNT"], axis=1)

                    # List the molecules which are inside the tube.
                    mol_list = []
                    mol_list.append(CNT_atoms_pic["Molecule"].unique())
                    tube_atoms_mol = pd.DataFrame(columns=["Element", "x", "y", "z", "Label", "Species", "Molecule"])
                    mol_list = mol_list[0]
                    # Scan the traj_file.frame0 and add all atoms to the tube_atoms_mol dataframe.
                    for index, row in traj_file.frame0.iterrows():
                        if row["Molecule"] in mol_list:
                            # Add the row to the tube_atoms dataframe.
                            tube_atoms_mol.loc[index] = [
                                row["Element"],
                                row["x"],
                                row["y"],
                                row["z"],
                                row["Label"],
                                row["Species"],
                                row["Molecule"],
                            ]
                    # Append the tube_atoms_mol dataframe to the tube_atoms_pic dataframe.
                    CNT_atoms_pic = pd.concat([CNT_atoms_pic, tube_atoms_mol], ignore_index=True)

                    # Finally remove all duplicates from the tube_atoms_pic dataframe.
                    CNT_atoms_pic = CNT_atoms_pic.drop_duplicates(
                        subset=["Element", "x", "y", "z", "Label", "Species", "Molecule"], keep="first"
                    )

            else:
                add_centerpoint = ddict.get_input(
                    f"Add the center point of the CNT{i + 1} to the file? [y/n] ", traj_file.args, "string"
                )
                if add_centerpoint == "y":
                    CNT_atoms_pic.loc[len(CNT_atoms_pic.index)] = [
                        "X",
                        molecules.structure_data["CNT_centers"][0][0],
                        molecules.structure_data["CNT_centers"][0][1],
                        molecules.structure_data["CNT_centers"][0][2],
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


def region_question(maindict, traj_file) -> dict:

    maindict["box_dimension"] = np.array(traj_file.box_size)

    # Ask if the analysis should be performed in a specific region
    regional_q = ddict.get_input(
        "Do you want the calculation to be performed in a specific region? [y/n] ", traj_file.args, "string"
    )
    regions = [0] * 6
    if regional_q == "y":
        regions[0] = float(ddict.get_input("Enter minimum x-value ", traj_file.args, "float"))
        regions[1] = float(ddict.get_input("Enter maximum x-value ", traj_file.args, "float"))
        regions[2] = float(ddict.get_input("Enter minimum y-value ", traj_file.args, "float"))
        regions[3] = float(ddict.get_input("Enter maximum y-value ", traj_file.args, "float"))
        regions[4] = float(ddict.get_input("Enter minimum z-value ", traj_file.args, "float"))
        regions[5] = float(ddict.get_input("Enter maximum z-value ", traj_file.args, "float"))

    return regional_q, regions


def frame_question(traj_file) -> int:

    start_frame = int(ddict.get_input("Start analysis at which frame?: ", traj_file.args, "int"))
    frame_interval = int(ddict.get_input("Analyse every nth step: ", traj_file.args, "int"))

    return start_frame, frame_interval


# Analysis of the trajectory.
def trajectory_analysis(traj_file, molecules, inputdict) -> None:

    analysis_opt = Analysis(traj_file)

    traj_analysis(analysis_opt, traj_file, molecules, inputdict)

    return analysis_opt


class Analysis:
    def __init__(self, traj_file):

        self.choice2 = self.analysis_choice(traj_file)

    def analysis_choice(self, traj_file) -> None:
        # Analysis choice.
        ddict.printLog("These functions are limited to rigid/frozen pores containing undistorted CNTs.", color="red")
        ddict.printLog("(1) Calculate the radial density inside the CNT")
        ddict.printLog("(2) Calculate the radial charge density inside the CNT (if charges are provided)")
        ddict.printLog("(3) Calculate the radial velocity of the liquid in the CNT.")
        ddict.printLog("(4) Calculate the accessibe volume of the CNT")
        ddict.printLog(
            "(5) Calculate the axial density along the z axis of the simulation box,",
            " with the accessible volume of the CNT considered.",
        )
        ddict.printLog("(6) Calculate the maximal/minimal distance between the liquid and pore atoms")
        ddict.printLog("\nThese functions are generally applicable.", color="red")
        ddict.printLog("(7) Calculate the coordination number")
        ddict.printLog("(8) Calculate the density along the axes.")

        analysis_choice2 = int(ddict.get_input("What analysis should be performed?:  ", traj_file.args, "int"))
        analysis_choice2_options = [1, 2, 3, 4, 5, 6, 7, 8]
        if analysis_choice2 not in analysis_choice2_options:
            ddict.printLog("-> The analysis you entered is not known.")
            sys.exit(1)
        ddict.printLog("")

        return analysis_choice2


def prepare_analysis_dict(maindict, traj_file, molecules, choice2) -> dict:

    maindict["maxdisp_atom_row"] = None
    maindict["maxdisp_atom_dist"] = 0

    maindict["analysis_choice2"] = choice2
    maindict["unique_molecule_frame"] = molecules.unique_molecule_frame
    maindict["CNT_atoms"] = traj_file.frame0[traj_file.frame0["CNT"].notnull()]
    maindict["number_of_frames"] = traj_file.number_of_frames

    return maindict


def get_preperation(choice2) -> callable:
    if choice2 in [1, 2]:
        from conan.analysis_modules.rad_dens import raddens_prep as main_loop_preparation
    elif choice2 == 3:
        from conan.analysis_modules.rad_velocity import rad_velocity_prep as main_loop_preparation
    elif choice2 == 4:
        from conan.analysis_modules.axial_dens import accessible_volume_prep as main_loop_preparation
    elif choice2 == 5:
        from conan.analysis_modules.axial_dens import axial_density_prep as main_loop_preparation
    elif choice2 == 6:
        from conan.analysis_modules.axial_dens import distance_search_prep as main_loop_preparation
    elif choice2 == 7:
        from conan.analysis_modules.coordination_number import Coord_number_prep as main_loop_preparation
    elif choice2 == 8:
        from conan.analysis_modules.axial_dens import density_analysis_prep as main_loop_preparation
    else:
        raise ValueError("Invalid choice")
    return main_loop_preparation


def get_analysis_and_processing(analysis_opt, maindict):
    if analysis_opt.choice2 in [1, 2]:
        if analysis_opt.choice2 == 1:
            from conan.analysis_modules.rad_dens import radial_density_analysis as analysis
        elif analysis_opt.choice2 == 2:
            from conan.analysis_modules.rad_dens import radial_charge_density_analysis as analysis
        from conan.analysis_modules.rad_dens import raddens_post_processing as post_processing
    elif analysis_opt.choice2 == 3:
        from conan.analysis_modules.rad_velocity import rad_velocity_analysis as analysis
        from conan.analysis_modules.rad_velocity import rad_velocity_processing as post_processing
    elif analysis_opt.choice2 == 4:
        from conan.analysis_modules.axial_dens import accessible_volume_analysis as analysis
        from conan.analysis_modules.axial_dens import accessible_volume_processing as post_processing
    elif analysis_opt.choice2 == 5:
        from conan.analysis_modules.axial_dens import axial_density_analysis as analysis
        from conan.analysis_modules.axial_dens import axial_density_processing as post_processing
    elif analysis_opt.choice2 == 6:
        from conan.analysis_modules.axial_dens import distance_search_analysis as analysis
        from conan.analysis_modules.axial_dens import distance_search_processing as post_processing
    elif analysis_opt.choice2 == 7:
        if maindict["do_xyz_analysis"] == "y":
            from conan.analysis_modules.coordination_number import Coord_number_xyz_analysis as analysis
            from conan.analysis_modules.coordination_number import Coord_xyz_post_processing as post_processing
        else:
            from conan.analysis_modules.coordination_number import Coord_number_analysis as analysis
            from conan.analysis_modules.coordination_number import Coord_post_processing as post_processing
    elif analysis_opt.choice2 == 8:
        from conan.analysis_modules.axial_dens import density_analysis_analysis as analysis
        from conan.analysis_modules.axial_dens import density_analysis_processing as post_processing
    else:
        raise ValueError("Invalid choice")

    return analysis, post_processing


def get_chunk_processing(maindict) -> callable:

    if maindict["do_xyz_analysis"] == "y":
        from conan.analysis_modules.coordination_number import Coord_xyz_chunk_processing as chunk_processing
    else:
        from conan.analysis_modules.coordination_number import Coord_chunk_processing as chunk_processing
    return chunk_processing


def traj_analysis(analysis_opt, traj_file, molecules, maindict) -> None:
    maindict = prepare_analysis_dict(maindict, traj_file, molecules, analysis_opt.choice2)

    # Get the preparation function
    main_loop_preparation = get_preperation(analysis_opt.choice2)

    maindict = main_loop_preparation(maindict, traj_file, molecules)

    # Get the analysis and post-processing functions
    analysis, post_processing = get_analysis_and_processing(analysis_opt, maindict)

    if analysis_opt.choice2 == 7:
        chunk_processing = get_chunk_processing(maindict)

    spec_molecule, spec_atom, analysis_spec_molecule = traj_info.molecule_choice(traj_file.args, traj_file.frame0, 1)

    Main_time = time.time()

    counter = 0

    regional_q, regions = region_question(maindict, traj_file)
    start_frame, frame_interval = frame_question(traj_file)

    # MAIN LOOP
    # Define which function to use reading the trajectory file.
    if traj_file.args["trajectoryfile"].endswith(".xyz"):
        from conan.analysis_modules.traj_info import xyz as run
    elif traj_file.args["trajectoryfile"].endswith(".pdb"):
        from conan.analysis_modules.traj_info import pdb as run
    elif traj_file.args["trajectoryfile"].endswith(".lmp") or traj_file.args["trajectoryfile"].endswith(".lammpstrj"):
        from conan.analysis_modules.traj_info import lammpstrj as run

    # Atomic masses.
    element_masses = ddict.dict_mass()
    trajectory = pd.read_csv(traj_file.args["trajectoryfile"], chunksize=traj_file.lines_chunk, header=None)
    chunk_number = 0
    frame_counter = 0

    # Loop over chunks.
    for chunk in trajectory:
        chunk_number += 1
        maindict["chunk_number"] = chunk_number
        ddict.printLog("\nChunk %d of %d" % (chunk_number, traj_file.num_chunks))

        # Divide chunk into individual frames. If it is the last chunk, the number of frames is smaller.
        if chunk.shape[0] == traj_file.lines_last_chunk:
            frames = np.split(chunk, traj_file.last_chunk_size)
        else:
            frames = np.split(chunk, traj_file.chunk_size)

        for frame in frames:
            frame_counter += 1

            # Skip frames based on start_frame and frame_interval
            if frame_counter < start_frame or (frame_counter - start_frame) % frame_interval != 0:
                continue

            # First load the frame into the function run() to get a dataframe. Then reset the index.
            split_frame = run(frame, element_masses, traj_file.frame0)
            split_frame.reset_index(drop=True, inplace=True)

            # Add the necessary columns to the dataframe.
            split_frame["Struc"] = traj_file.frame0["Struc"].values
            split_frame["Molecule"] = traj_file.frame0["Molecule"].values
            split_frame["Species"] = traj_file.frame0["Species"].values
            split_frame["Label"] = traj_file.frame0["Label"].values

            # Drop all CNT and carbon_wall atoms, just the Liquid atoms are needed for the analysis.
            split_frame = split_frame[split_frame["Struc"] == "Liquid"]
            split_frame = split_frame.drop(["Struc"], axis=1)

            # Drop the other atoms which are not needed for the analysis.
            if analysis_spec_molecule == "y":
                split_frame = split_frame[split_frame["Species"].isin(spec_molecule)]
                # If the spec_atom list does not contain "all" then only the atoms in the list are kept.
                if spec_atom[0] != "all":
                    # If specific atoms are requested, only these atoms are kept.
                    split_frame = split_frame[split_frame["Label"].isin(spec_atom)]

            if regional_q == "y":
                split_frame = split_frame[split_frame["X"].astype(float) >= regions[0]]
                split_frame = split_frame[split_frame["X"].astype(float) <= regions[1]]
                split_frame = split_frame[split_frame["Y"].astype(float) >= regions[2]]
                split_frame = split_frame[split_frame["Y"].astype(float) <= regions[3]]
                split_frame = split_frame[split_frame["Z"].astype(float) >= regions[4]]
                split_frame = split_frame[split_frame["Z"].astype(float) <= regions[5]]

            # Save the split_frame in maindict
            maindict["split_frame"] = split_frame
            maindict["counter"] = counter
            maindict["regions"] = regions
            maindict["regional_q"] = regional_q

            maindict = analysis(maindict, traj_file, molecules, analysis_opt)

            counter += 1
            print(
                "Processed frame %d (frame %d of %d)" % (counter, frame_counter, traj_file.number_of_frames),
                end="\r",
            )

        # For memory intensive analyses (e.g. CN) we need to do the processing after every chunk
        if analysis_opt.choice2 == 7:
            maindict = chunk_processing(maindict)

    ddict.printLog("")
    ddict.printLog("Finished processing the trajectory. %d frames were processed." % (counter))
    ddict.printLog("")

    # DATA PROCESSING
    post_processing(maindict)

    ddict.printLog("\nThe main loop took %0.3f seconds to run." % (time.time() - Main_time))


if __name__ == "__main__":
    # ARGUMENTS
    args = ddict.read_commandline()
