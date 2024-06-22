# The program is written by Leonard Dick, 2023

"""
This module was set up to create simulation boxes from separate bulk liquid and structure files.
The liquid bulk file must be named bulk.xyz.
The wall file must be named wall.xyz.
The pore file must be named pore.xyz.
The pore_left file must be named pore_left.xyz.
The pore_right file must be named pore_right.xyz.
All files are either located in the current directory or in the cbuild directory.
If the file is not found in the current directory, the program searches in the structures directory.
If the file is not found in the cbuild directory, the program exits.
"""

import os
import sys

import pandas as pd

import conan.defdict as ddict


def simbox_mode(args) -> None:
    ddict.printLog("")
    ddict.printLog("SIMBOX mode", color="red")
    ddict.printLog("")
    ddict.printLog("This program adds solid structures and liquid bulk xyz files to one simulation box.")
    ddict.printLog("P stands for pore, B for liquid bulk, and W for wall.")
    ddict.printLog(
        "If pore_left and pore_right from the cbuild section are used, enter L for the left pore and R for the right"
        " pore."
    )
    combination = ddict.get_input(
        "Please enter the wanted combination for the simulation box [eg.: BPBW]: ", args, "string"
    )
    ddict.printLog("")

    combination_list = list(combination)
    ddict.printLog(f"The combination list is: {combination_list}")

    combination_list_unique = list(set(combination_list))

    possible_letters = ["B", "P", "W", "R", "L"]
    for i in combination_list:
        if i not in possible_letters:
            ddict.printLog("The combination list contains an undefined letter. Exiting...")
            sys.exit(1)

    file_name_list = {"B": "bulk", "P": "pore", "W": "wall", "R": "pore_right", "L": "pore_left"}
    ddict.printLog(f"The file name list is: {file_name_list}")

    structure_data = {}
    for i in possible_letters:
        if i in combination_list_unique:
            file_name = f"{file_name_list[i]}.xyz"
            structure_data[file_name_list[i]] = read_file(file_name)
            structure_data[file_name_list[i]] = adjust_dataframe(structure_data[file_name_list[i]])
            ddict.printLog(f"The {file_name} file was found.")

    simbox = create_simulation_box(combination_list, file_name_list, structure_data)
    ddict.printLog("The simbox dataframe was created.")

    save_simbox_file(simbox)


def read_file(file_name: str) -> pd.DataFrame:
    """
    Reads a file into a DataFrame, searching in current and structures directories.
    """

    def read_and_process(file_path):
        data = []
        try:
            with open(file_path, "r") as file:
                lines = file.readlines()[2:]  # Skip the first two lines
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 4:
                        data.append(parts[:4])
            df = pd.DataFrame(data, columns=["atom", "x", "y", "z"])
            df["x"] = pd.to_numeric(df["x"], errors="coerce")
            df["y"] = pd.to_numeric(df["y"], errors="coerce")
            df["z"] = pd.to_numeric(df["z"], errors="coerce")
            return df
        except Exception as e:
            ddict.printLog(f"Error reading {file_path}: {e}. Exiting...")
            sys.exit(1)

    # Read file from current directory or structures
    if os.path.exists(file_name):
        return read_and_process(file_name)
    elif os.path.exists(f"structures/{file_name}"):
        return read_and_process(f"structures/{file_name}")
    else:
        ddict.printLog(f"The {file_name} file could not be found. Exiting...")
        sys.exit(1)


def adjust_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adjusts the z-coordinate to start at 0 for proper alignment of structures.
    """
    min_z = df["z"].min()
    if min_z != 0:
        df["z"] -= min_z
    return df


def create_simulation_box(combination_list: list, file_name_list: dict, structure_data: dict) -> pd.DataFrame:
    """
    Creates the simulation box DataFrame based on the provided combination list.
    """
    simbox = pd.DataFrame(columns=["atom", "x", "y", "z"])
    simbox_max_z = 0

    for i in combination_list:
        df_copy = structure_data[file_name_list[i]].copy()
        df_copy["z"] += simbox_max_z
        simbox = pd.concat([simbox, df_copy], ignore_index=True)
        simbox_max_z = simbox["z"].max() + 3

    return simbox


def save_simbox_file(simbox: pd.DataFrame) -> None:
    """
    Saves the simbox DataFrame to a .xyz file, renaming existing files if necessary.
    """
    id_num = 1
    if os.path.exists("simbox.xyz"):
        while os.path.exists(f"simbox-{id_num}.xyz"):
            id_num += 1
        os.rename("simbox.xyz", f"simbox-{id_num}.xyz")
        ddict.printLog(f"\nThe existing simbox.xyz file was renamed to simbox-{id_num}.xyz.\n")

    with open("simbox.xyz", "w") as f:
        f.write(f"{len(simbox)}\n\n")
        simbox.to_csv(f, sep="\t", header=False, index=False, mode="a")
