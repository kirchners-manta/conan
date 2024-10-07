# The program is written by Leonard Dick, 2023

import sys
import time

import numpy as np
import pandas as pd

import conan.analysis_modules.rad_dens2 as raddens2
import conan.analysis_modules.rad_velocity2 as radvel2
import conan.defdict as ddict
from conan.analysis_modules import traj_info
from conan.analysis_modules import xyz_output as xyz


def analysis_opt(traj_file, molecules, maindict):
    ddict.printLog("(1) Produce xyz files of the simulation box or pore structure.")
    ddict.printLog("(2) Analyze the trajectory.")
    choice = int(ddict.get_input("Picture or analysis mode?: ", traj_file.args, "int"))
    ddict.printLog("")
    if choice == 1:
        ddict.printLog("PICTURE mode.\n", color="red")
        xyz.xyz_generator(traj_file, molecules)
    elif choice == 2:
        ddict.printLog("ANALYSIS mode.\n", color="red")
        if len(molecules.structure_data["CNT_centers"]) >= 0:
            run_analysis(traj_file, molecules, maindict)
        else:
            ddict.printLog("-> The choice is not known.")
        ddict.printLog("")


def run_analysis(traj_file, molecules, maindict):
    an = Analysis(traj_file, molecules, maindict)

    # now run the analysis
    if an.choice2 == 1 or an.choice2 == 2:
        raddens2.radial_density_analysis(traj_file, molecules, an)
    elif an.choice2 == 3:
        radvel2.radial_velocity_analysis(traj_file, molecules, an)

    # traj_analysis(traj_file, molecules, maindict, an)


class Analysis:
    def __init__(self, traj_file, molecules, maindict) -> None:

        self.choice2 = self.analysis_choice(traj_file)
        self.run = self.get_traj_module(traj_file)

    def analysis_choice(self, traj_file) -> int:
        # Analysis choice.
        ddict.printLog("These functions are limited to rigid/frozen pores containing undistorted CNTs.", color="red")
        ddict.printLog("(1) Calculate the radial density inside the CNT")
        ddict.printLog("(2) Calculate the radial charge density inside the CNT (if charges are provided)")
        ddict.printLog("(3) Calculate the radial atomic velocity of the liquid in the CNT.")
        ddict.printLog("(4) Calculate the accessible volume of the CNT")
        ddict.printLog(
            "(5) Calculate the axial density along the z axis of the simulation box,",
            " with the accessible volume of the CNT considered.",
        )
        ddict.printLog("(6) Calculate the maximal/minimal distance between the liquid and pore atoms")
        ddict.printLog("\nThese functions are generally applicable.", color="red")
        ddict.printLog("(7) Calculate the coordination number")
        ddict.printLog("(8) Calculate the density along the axes.")
        ddict.printLog("(9) Calculate the velocity of the liquid in the CNT.")

        analysis_choice2 = int(ddict.get_input("What analysis should be performed?:  ", traj_file.args, "int"))
        analysis_choice2_options = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        if analysis_choice2 not in analysis_choice2_options:
            ddict.printLog("-> The analysis you entered is not known.")
            sys.exit(1)
        ddict.printLog("")

        return analysis_choice2

    def get_traj_module(self, traj_file) -> callable:
        if traj_file.args["trajectoryfile"].endswith(".xyz"):
            from conan.analysis_modules.traj_info import xyz as run
        elif traj_file.args["trajectoryfile"].endswith(".pdb"):
            from conan.analysis_modules.traj_info import pdb as run
        elif traj_file.args["trajectoryfile"].endswith(".lmp") or traj_file.args["trajectoryfile"].endswith(
            ".lammpstrj"
        ):
            from conan.analysis_modules.traj_info import lammpstrj as run
        else:
            raise ValueError("Unsupported trajectory file format")
        return run


def region_question(traj_file) -> tuple:
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


def frame_question(traj_file) -> tuple:
    start_frame = int(ddict.get_input("Start analysis at which frame?: ", traj_file.args, "int"))
    frame_interval = int(ddict.get_input("Analyse every nth step: ", traj_file.args, "int"))
    return start_frame, frame_interval


def process_trajectory(traj_file, molecules, an, analysis_option):

    element_masses = ddict.dict_mass()  # Erstellen eines Massen-Dictionarys
    trajectory = pd.read_csv(traj_file.args["trajectoryfile"], chunksize=traj_file.lines_chunk, header=None)

    chunk_number = 0
    frame_counter = 0
    proc_frames = 0

    spec_molecule, spec_atom, analysis_spec_molecule = traj_info.molecule_choice(traj_file.args, traj_file.frame0, 1)
    regional_q, regions = region_question(traj_file)
    start_frame, frame_interval = frame_question(traj_file)

    Main_time = time.time()
    for chunk in trajectory:
        chunk_number += 1
        ddict.printLog(f"\nChunk {chunk_number} of {traj_file.num_chunks}")

        if chunk.shape[0] == traj_file.lines_last_chunk:
            frames = np.split(chunk, traj_file.last_chunk_size)
        else:
            frames = np.split(chunk, traj_file.chunk_size)

        for frame in frames:
            frame_counter += 1

            # Skip frames based on start_frame and frame_interval
            if frame_counter < start_frame or (frame_counter - start_frame) % frame_interval != 0:
                continue

            split_frame = prepare_frame(
                an,
                frame,
                element_masses,
                traj_file,
                regional_q,
                regions,
                spec_molecule,
                spec_atom,
                analysis_spec_molecule,
            )
            if split_frame is None:
                continue
            analysis_option.analyze_frame(split_frame, frame_counter)
            proc_frames += 1
            print(
                "Processed frame %d (frame %d of %d)" % (frame_counter, frame_counter, traj_file.number_of_frames),
                end="\r",
            )

    analysis_option.proc_frames = proc_frames
    ddict.printLog(f"\nFinished processing the trajectory in {time.time() - Main_time:.2f} seconds.")


def prepare_frame(
    an, frame, element_masses, traj_file, regional_q, regions, spec_molecule, spec_atom, analysis_spec_molecule
):
    """Vorbereiten des Frames für die Analyse."""
    split_frame = run_trajectory_module(an, frame, element_masses, traj_file)
    if split_frame is None:
        return None

    split_frame.reset_index(drop=True, inplace=True)
    split_frame["Struc"] = traj_file.frame0["Struc"]
    split_frame["Molecule"] = traj_file.frame0["Molecule"]
    split_frame["Species"] = traj_file.frame0["Species"]
    split_frame["Label"] = traj_file.frame0["Label"]

    split_frame = split_frame[split_frame["Struc"] == "Liquid"].drop(["Struc"], axis=1)

    if regional_q == "y":
        split_frame = split_frame[split_frame["X"].astype(float) >= regions[0]]
        split_frame = split_frame[split_frame["X"].astype(float) <= regions[1]]
        split_frame = split_frame[split_frame["Y"].astype(float) >= regions[2]]
        split_frame = split_frame[split_frame["Y"].astype(float) <= regions[3]]
        split_frame = split_frame[split_frame["Z"].astype(float) >= regions[4]]
        split_frame = split_frame[split_frame["Z"].astype(float) <= regions[5]]

    if analysis_spec_molecule == "y":
        split_frame = split_frame[split_frame["Species"].isin(spec_molecule)]
        # If the spec_atom list does not contain "all" then only the atoms in the list are kept.
        if spec_atom[0] != "all":
            # If specific atoms are requested, only these atoms are kept.
            split_frame = split_frame[split_frame["Label"].isin(spec_atom)]

    return split_frame


def run_trajectory_module(an, frame, element_masses, traj_file):
    """Rufe das spezifische Modul auf, um das Frame in die richtige Struktur zu bringen."""
    try:
        split_frame = an.run(frame, element_masses, traj_file.frame0)
    except Exception as e:
        ddict.printLog(f"Error processing frame: {e}")
        return None
    return split_frame


if __name__ == "__main__":
    # ARGUMENTS
    args = ddict.read_commandline()


"""
def prepare_analysis_dict(maindict, traj_file, molecules, choice2, an) -> dict:
    maindict["maxdisp_atom_row"] = None
    maindict["maxdisp_atom_dist"] = 0
    maindict["analysis_choice2"] = an.choice2
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
    elif choice2 == 9:
        from conan.analysis_modules.velocity import velocity_prep as main_loop_preparation
    else:
        raise ValueError("Invalid choice")
    return main_loop_preparation


def get_analysis_and_processing(choice2, maindict) -> tuple:
    if choice2 in [1, 2]:
        if choice2 == 1:
            from conan.analysis_modules.rad_dens import radial_density_analysis as analysis
        elif choice2 == 2:
            from conan.analysis_modules.rad_dens import radial_charge_density_analysis as analysis
        from conan.analysis_modules.rad_dens import raddens_post_processing as post_processing
    elif choice2 == 3:
        from conan.analysis_modules.rad_velocity import rad_velocity_analysis as analysis
        from conan.analysis_modules.rad_velocity import rad_velocity_processing as post_processing
    elif choice2 == 4:
        from conan.analysis_modules.axial_dens import accessible_volume_analysis as analysis
        from conan.analysis_modules.axial_dens import accessible_volume_processing as post_processing
    elif choice2 == 5:
        from conan.analysis_modules.axial_dens import axial_density_analysis as analysis
        from conan.analysis_modules.axial_dens import axial_density_processing as post_processing
    elif choice2 == 6:
        from conan.analysis_modules.axial_dens import distance_search_analysis as analysis
        from conan.analysis_modules.axial_dens import distance_search_processing as post_processing
    elif choice2 == 7:
        if maindict["do_xyz_analysis"] == "y":
            from conan.analysis_modules.coordination_number import Coord_number_xyz_analysis as analysis
            from conan.analysis_modules.coordination_number import Coord_xyz_post_processing as post_processing
        else:
            from conan.analysis_modules.coordination_number import Coord_number_analysis as analysis
            from conan.analysis_modules.coordination_number import Coord_post_processing as post_processing
    elif choice2 == 8:
        from conan.analysis_modules.axial_dens import density_analysis_analysis as analysis
        from conan.analysis_modules.axial_dens import density_analysis_processing as post_processing
    elif choice2 == 9:
        from conan.analysis_modules.velocity import velocity_analysis as analysis
        from conan.analysis_modules.velocity import velocity_processing as post_processing
    else:
        raise ValueError("Invalid choice")
    return analysis, post_processing


def get_chunk_processing(maindict) -> callable:
    if maindict["do_xyz_analysis"] == "y":
        from conan.analysis_modules.coordination_number import Coord_xyz_chunk_processing as chunk_processing
    else:
        from conan.analysis_modules.coordination_number import Coord_chunk_processing as chunk_processing
    return chunk_processing


def traj_analysis(traj_file, molecules, maindict, an) -> None:
    choice2 = an.choice2
    # choice2 = analysis_choice(traj_file)
    maindict = prepare_analysis_dict(maindict, traj_file, molecules, choice2, an)

    # Get the preparation function
    main_loop_preparation = get_preperation(choice2)
    maindict = main_loop_preparation(maindict, traj_file, molecules)

    # Get the analysis and post-processing functions
    analysis, post_processing = get_analysis_and_processing(choice2, maindict)

    if choice2 == 7:
        chunk_processing = get_chunk_processing(maindict)

    spec_molecule, spec_atom, analysis_spec_molecule = traj_info.molecule_choice(traj_file.args, traj_file.frame0, 1)

    Main_time = time.time()
    counter = 0

    regions = an.regions
    regional_q = an.reg_q
    start_frame = an.start_frame
    frame_interval = an.frame_interval

    element_masses = ddict.dict_mass()
    trajectory = pd.read_csv(traj_file.args["trajectoryfile"], chunksize=traj_file.lines_chunk, header=None)
    chunk_number = 0
    frame_counter = 0

    # MAIN LOOP
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
            split_frame = an.run(frame, element_masses, traj_file.frame0)
            split_frame.reset_index(drop=True, inplace=True)

            # Add the necessary columns to the dataframe.
            split_frame["Struc"] = traj_file.frame0["Struc"]
            split_frame["Molecule"] = traj_file.frame0["Molecule"]
            split_frame["Species"] = traj_file.frame0["Species"]
            split_frame["Label"] = traj_file.frame0["Label"]

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

            maindict = analysis(maindict, traj_file, molecules)

            counter += 1
            print(
                "Processed frame %d (frame %d of %d)" % (counter, frame_counter, traj_file.number_of_frames),
                end="\r",
            )

        # For memory intensive analyses (e.g. CN) we need to do the processing after every chunk
        if choice2 == 7:
            maindict = chunk_processing(maindict)

    ddict.printLog("")
    ddict.printLog("Finished processing the trajectory. %d frames were processed." % (counter))
    ddict.printLog("")


    # DATA PROCESSING
    post_processing(maindict)

    ddict.printLog("\nThe main loop took %0.3f seconds to run." % (time.time() - Main_time))

"""
