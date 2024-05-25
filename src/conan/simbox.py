# The program is written by Leonard Dick, 2023

"""This module was set up to create simulation boxes from seperate bulk liquid and structure files
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
        "If pore_left and pore_right from the cbuild section are used, enter L for the left pore and R for the right pore."
    )
    combination = ddict.get_input(
        "Please enter the wanted combination for the simulation box [eg.: BPBW]: ", args, "string"
    )
    ddict.printLog("")

    # Split the string into a list.
    combination_list = list(combination)
    ddict.printLog("The combination list is: %s" % (combination_list))

    # Make another list with just the unique entries.
    combination_list_unique = list(set(combination_list))

    # Write the possible combination list
    possible_letters = ["B", "P", "W", "R", "L"]

    # If a different letter than those is given, the program exits.
    for i in combination_list:
        if i not in possible_letters:
            ddict.printLog("The combination list contains a wrong letter. Exiting...")
            sys.exit(1)

    # Find the necessary files.
    # Create a list with the possible file names.
    file_name_list = dict([("B", "bulk"), ("P", "pore"), ("W", "wall"), ("R", "pore_right"), ("L", "pore_left")])
    ddict.printLog("The file name list is: %s" % (file_name_list))

    for i in possible_letters:
        if i in combination_list_unique:
            # Create the file name.
            file_name = "%s.xyz" % (file_name_list[i])
            try:
                # Read file to dataframe.
                df = pd.read_csv(file_name, sep="\s+", header=None, skiprows=2, names=["atom", "x", "y", "z"])
            except:
                try:
                    df = pd.read_csv(
                        "structures/%s" % (file_name), sep="\s+", header=None, skiprows=2, names=["atom", "x", "y", "z"]
                    )
                except:
                    ddict.printLog("The %s file could not be found. Exiting..." % (i))
                    sys.exit(1)
            # Rename the dataframe variable to the file name.
            exec("%s = df" % (file_name_list[i]))
            # Print the dataframe.
            ddict.printLog("The %s file was found." % (file_name))
    ddict.printLog("")

    # Find the minimal z values in all dataframes. If it is not zero, shift the respective dataframe to 0.
    for i in possible_letters:
        if i in combination_list_unique:
            # Create the file name.
            file_name = "%s.xyz" % (file_name_list[i])
            # Find the minimal z value
            exec("%s_min_z = %s['z'].min()" % (file_name_list[i], file_name_list[i]))
            # Shift the dataframe to 0.
            if eval("%s_min_z" % (file_name_list[i])) != 0:
                exec("%s['z'] = %s['z'] - %s_min_z" % (file_name_list[i], file_name_list[i], file_name_list[i]))
            # Find the maximal z value.
            exec("%s_max_z = %s['z'].max()" % (file_name_list[i], file_name_list[i]))

    # Now start building the simulation box by setting up an empty dataframe with the correct column names.
    simbox = pd.DataFrame(columns=["atom", "x", "y", "z"])
    tmp = []
    simbox_max_z = 0
    for i in combination_list:

        # Make a dummy dataframe.
        exec("%s_dummy = %s.copy()" % (file_name_list[i], file_name_list[i]))
        # Shift the dataframe in z direction by simbox_max_z.
        exec("%s_dummy['z'] = %s_dummy['z'] + simbox_max_z" % (file_name_list[i], file_name_list[i]))
        # Append the dataframe to the tmp list.
        exec("tmp.append(%s_dummy)" % (file_name_list[i]))
        # Find the maximum z value in the tmp list.
        simbox_max_z = max([df["z"].max() for df in tmp]) + 3

    # Concatenate the tmp list to the simbox dataframe.
    simbox = pd.concat(tmp, ignore_index=True)
    ddict.printLog("The simbox dataframe was created.")
    ddict.printLog(simbox)

    # File creation.
    id_num = 1
    if os.path.exists("simbox.xyz"):
        while os.path.exists("simbox-%s.xyz" % id_num):
            id_num += 1
        # Rename the existing xyz file to simbox-(id_num).xyz.
        os.rename("simbox.xyz", "simbox-%s.xyz" % id_num)
        ddict.printLog("\nThe existing simbox.xyz file was renamed to simbox-%s.xyz.\n" % id_num)

    # Write the simbox.xyz file.
    with open("simbox.xyz", "w") as f:
        f.write("%d\n" % (len(simbox)))
        f.write("\n")
    simbox.to_csv("simbox.xyz", sep="\t", header=False, index=False, mode="a")
