import sys
import time

import pandas as pd

import conan.analysis_modules.axial_dens as axdens
import conan.analysis_modules.cnt_fill as cnt_fill
import conan.analysis_modules.coordination_number as cn
import conan.analysis_modules.flex_rad_dens as flex_raddens
import conan.analysis_modules.msd as msd
import conan.analysis_modules.rad_dens as raddens
import conan.analysis_modules.rad_velocity as radvel
import conan.analysis_modules.traj_info as traj_info
import conan.analysis_modules.utils as ut
import conan.analysis_modules.velocity as vel
import conan.analysis_modules.xyz_output as xyz
import conan.defdict as ddict

# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning, module="numpy.core.fromnumeric")


def analysis_opt(traj_file, molecules, maindict):
    """Choice between picture or analysis mode."""

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
    """Run the analysis based on the user's choice."""

    an = Analysis(traj_file, molecules, maindict)
    if an.choice2 == 1 or an.choice2 == 2:
        raddens.radial_density_analysis(traj_file, molecules, an)
    elif an.choice2 == 3:
        radvel.radial_velocity_analysis(traj_file, molecules, an)
    elif an.choice2 == 4:
        axdens.accessible_volume_analysis(traj_file, molecules, an)
    elif an.choice2 == 5:
        axdens.axial_density_analysis(traj_file, molecules, an)
    elif an.choice2 == 6:
        axdens.distance_search_analysis(traj_file, molecules, an)
    elif an.choice2 == 7:
        cn.coordination_number_analysis(traj_file, molecules, an)
    elif an.choice2 == 8:
        axdens.density_analysis_3D(traj_file, molecules, an)
    elif an.choice2 == 9:
        vel.mol_velocity_analysis(traj_file, molecules, an)
    elif an.choice2 == 10:
        msd.msd_analysis(traj_file, molecules, an)
    elif an.choice2 == 11:
        cnt_fill.cnt_loading_mass(traj_file, molecules, an)
    elif an.choice2 == 12:
        flex_raddens.flex_rad_dens(traj_file, molecules, an)


class Analysis:

    def __init__(self, traj_file, molecules, maindict) -> None:
        self.choice2 = self.analysis_choice(traj_file)
        self.run = self.get_traj_module(traj_file)

    def analysis_choice(self, traj_file) -> int:
        """Choose the analysis to be performed."""

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
        ddict.printLog("(9) Calculate the velocity along the axes.")
        ddict.printLog("(10) Calculate the mean square displacement of the liquid in the CNT.")
        ddict.printLog("(11) Calculate the mass of the liquid inside a CNT.")
        ddict.printLog("(12) Calculate the radial mass density of the liquid inside a flexible CNT.")
        analysis_choice2 = int(ddict.get_input("What analysis should be performed?:  ", traj_file.args, "int"))
        analysis_choice2_options = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        if analysis_choice2 not in analysis_choice2_options:
            ddict.printLog("-> The analysis you entered is not known.")
            sys.exit(1)
        ddict.printLog("")

        return analysis_choice2

    def get_traj_module(self, traj_file) -> callable:
        """Get the trajectory module based on the file format.
        Based on the file format, each frame needs to be processed differently."""

        if traj_file.args["trajectoryfile"].endswith(".xyz"):
            from traj_info import xyz as run
        elif traj_file.args["trajectoryfile"].endswith(".pdb"):
            from traj_info import pdb as run
        elif traj_file.args["trajectoryfile"].endswith(".lmp") or traj_file.args["trajectoryfile"].endswith(
            ".lammpstrj"
        ):
            from traj_info import lammpstrj as run
        else:
            raise ValueError("Unsupported trajectory file format")
        return run

    def region_question(self, traj_file) -> tuple:
        """Question for the user if the calculation should be performed in a specific region."""
        self.regional_q = ddict.get_input(
            "Do you want the calculation to be performed in a specific region? [y/n] ", traj_file.args, "string"
        )
        self.regions = [0] * 6
        if self.regional_q == "y":
            self.regions[0] = float(ddict.get_input("Enter minimum x-value ", traj_file.args, "float"))
            self.regions[1] = float(ddict.get_input("Enter maximum x-value ", traj_file.args, "float"))
            self.regions[2] = float(ddict.get_input("Enter minimum y-value ", traj_file.args, "float"))
            self.regions[3] = float(ddict.get_input("Enter maximum y-value ", traj_file.args, "float"))
            self.regions[4] = float(ddict.get_input("Enter minimum z-value ", traj_file.args, "float"))
            self.regions[5] = float(ddict.get_input("Enter maximum z-value ", traj_file.args, "float"))
        return self.regional_q, self.regions

    def frame_question(self, traj_file) -> tuple:
        """Question for the user to specify the start frame and the frame interval."""

        self.start_frame = int(ddict.get_input("Start analysis at which frame?: ", traj_file.args, "int"))
        self.frame_interval = int(ddict.get_input("Analyse every nth step: ", traj_file.args, "int"))
        return self.start_frame, self.frame_interval


def process_trajectory(traj_file, molecules, an, analysis_option):
    """Process the trajectory file. This is the main loop."""

    element_masses = ddict.dict_mass()
    trajectory = pd.read_csv(traj_file.args["trajectoryfile"], chunksize=traj_file.lines_chunk, header=None)

    chunk_number = 0
    frame_counter = 0
    proc_frames = 0

    spec_molecule, spec_atom, analysis_spec_molecule = traj_info.molecule_choice(traj_file.args, traj_file.frame0, 1)
    analysis_option.analysis_spec_molecule = analysis_spec_molecule

    regional_q = "n"
    regions = [0] * 6
    if an.choice2 in [1, 2, 3, 4, 5, 6, 7, 8]:
        regional_q, regions = an.region_question(traj_file)
        analysis_option.regional_q = regional_q
        analysis_option.regions = regions
    start_frame, frame_interval = an.frame_question(traj_file)

    Main_time = time.time()
    for chunk in trajectory:
        chunk_number += 1
        print(f"\nChunk {chunk_number} of {traj_file.num_chunks}")

        frames = []
        num_frames = chunk.shape[0] // traj_file.lines_per_frame
        for i in range(num_frames):
            start_idx = i * traj_file.lines_per_frame
            end_idx = (i + 1) * traj_file.lines_per_frame
            frames.append(chunk.iloc[start_idx:end_idx])

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
                "Processed frame %d (frame %d of %d)" % (proc_frames, frame_counter, traj_file.number_of_frames),
                end="\r",
            )

        # Run chunk processing for certain analysis options
        if isinstance(analysis_option, cn.CoordinationNumberAnalysis):
            analysis_option.proc_chunk()

        # Avoid division by 0 if analysis is started on a frame that lies outside of the first chunk
        if proc_frames != 0:
            time_per_frame = (time.time() - Main_time) / proc_frames
            remaining_frames = (traj_file.number_of_frames - frame_counter) / frame_interval
            remaining_time = time_per_frame * remaining_frames
            print(
                f"\nTime per frame: {time_per_frame:.2f} s,",
                f" Remaining time: {remaining_time:.2f} s ({remaining_time / 60:.2f} min)",
            )

    analysis_option.proc_frames = proc_frames
    ddict.printLog(f"\n\nFinished processing the trajectory in {time.time() - Main_time:.2f} seconds.\n")


def prepare_frame(
    an, frame, element_masses, traj_file, regional_q, regions, spec_molecule, spec_atom, analysis_spec_molecule
):
    """Preparing the frame for the analysis module by removing atoms, which are not needed."""
    split_frame = run_trajectory_module(an, frame, element_masses, traj_file)

    if split_frame is None:
        return None
    split_frame.reset_index(drop=True, inplace=True)
    split_frame["Struc"] = traj_file.frame0["Struc"]
    split_frame["Molecule"] = traj_file.frame0["Molecule"]
    split_frame["Species"] = traj_file.frame0["Species"]
    split_frame["Label"] = traj_file.frame0["Label"]

    # analysis which need the structure positions and all atoms:
    set_struc_analysis = [10, 11, 12]

    if an.choice2 not in set_struc_analysis:
        split_frame = split_frame[split_frame["Struc"] == "Liquid"]

    if regional_q == "y":
        # Wrap coordinates into the simulation box using PBC
        split_frame = ut.wrapping_coordinates(traj_file.box_size, split_frame)
        split_frame = split_frame[split_frame["X"].astype(float) >= regions[0]]
        split_frame = split_frame[split_frame["X"].astype(float) <= regions[1]]
        split_frame = split_frame[split_frame["Y"].astype(float) >= regions[2]]
        split_frame = split_frame[split_frame["Y"].astype(float) <= regions[3]]
        split_frame = split_frame[split_frame["Z"].astype(float) >= regions[4]]
        split_frame = split_frame[split_frame["Z"].astype(float) <= regions[5]]

    if analysis_spec_molecule == "y":
        # filter liquid molecules
        liquid_mask = split_frame["Struc"] == "Liquid"
        # Create a boolean filter mask that keeps rows of structure atoms
        # and filters the liquid atoms which are to be analyzed
        species_mask = ~liquid_mask | (liquid_mask & split_frame["Species"].isin(spec_molecule))
        split_frame = split_frame[species_mask]
        if spec_atom[0] != "all":
            # Create a filter mask that keeps non-liquid rows and filters liquid rows by atom label
            atom_mask = ~(split_frame["Struc"] == "Liquid") | split_frame["Label"].isin(spec_atom)
            split_frame = split_frame[atom_mask]

    return split_frame


def run_trajectory_module(an, frame, element_masses, traj_file):
    """Run the trajectory module based on the file format."""

    try:
        split_frame = an.run(frame, element_masses, traj_file.frame0)
    except Exception as e:
        ddict.printLog(f"Error processing frame: {e}")
        return None
    return split_frame


if __name__ == "__main__":
    # ARGUMENTS
    args = ddict.read_commandline()
