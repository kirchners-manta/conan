# The program is written by Leonard Dick, 2023

"""This file contains all globally needed functions and dictionaries."""

# MODULES
import argparse
import sys


# DEFINITIONS
# Read command line arguments.
def read_commandline() -> dict:

    # ARGUMENT READER
    parser = argparse.ArgumentParser(description="CONAN - CONfinement ANalysis")
    parser.add_argument("-f", "--trajectoryfile", help="The xyz file containing the trajectory.")
    parser.add_argument("-c", "--cbuild", action="store_true", help="Generate carbon structures.")
    parser.add_argument("-b", "--box", action="store_true", help="Build a simulation box.")
    parser.add_argument("-i", "--input", help="Use an input file to run the program.")
    parser.add_argument("-m", "--manual", action="store_true", help="Manual mode to define carbon_structures")
    parser.parse_args()
    args = vars(parser.parse_args())

    # If no arguments are given in the command line, print help and exit.
    if len(sys.argv) == 1:
        parser.print_help()
        printLog("")
        printLog("No arguments given. Exiting...")
        sys.exit(1)
    return args


# Write to log file. With color if set.
def printLog(*args, color=None, **kwargs) -> None:
    message = " ".join(map(str, args))
    if color == "red":
        print("\033[91m" + message + "\033[0m", **kwargs)
    elif color == "yellow":
        print("\033[93m" + message + "\033[0m", **kwargs)
    else:
        print(message, **kwargs)
    with open("conan.log", "a") as file:
        print(message, file=file)


# Write input to log file.
def get_input(question, args, input_type) -> str:
    var = None
    # If an input file is given, use that instead of asking for input.
    if args["input"] is not None:
        with open(args["input"], "r") as file:
            # Search for the question given to the user in the input file.
            for line in file:
                if line.startswith(question):
                    # Return the answer, it is everthing after the question which is in the same line.
                    if input_type == "float":
                        var = float(line[len(question) :].strip())
                    elif input_type == "int":
                        var = int(line[len(question) :].strip())
                    else:
                        var = line[len(question) :].strip()
                    print(question, var)
                    break

    # If no input file is given or the question is not found, ask for input
    if var is None:
        while True:
            try:
                if input_type == "float":
                    var = float(input(question))
                elif input_type == "int":
                    var = int(input(question))
                else:
                    var = input(question)
                break
            except ValueError:
                printLog("Please enter a valid answer.", color="red")
                continue

    # Write the input to the log file.
    with open("conan.log", "a") as file:
        print(question, var, file=file)

    return var


# DICTIONARIES
# Atomic masses.
def dict_mass() -> dict:
    elem_masses = {
        "H": 1.008,
        "Li": 6.941,
        "Na": 22.990,
        "K": 39.098,
        "Mg": 24.305,
        "Ca": 40.078,
        "Zn": 65.38,
        "C": 12.011,
        "N": 14.007,
        "O": 15.999,
        "B": 10.811,
        "F": 18.998,
        "P": 30.974,
        "S": 32.065,
        "Cl": 35.453,
        "Br": 79.904,
        "I": 126.904,
        "Ag": 107.868,
        "Au": 196.967,
        "D": 0.00,
        "X": 1.00,
    }
    return elem_masses


# Atomic van der Waals radii.
def dict_vdW() -> dict:
    elem_vdW = {
        "H": 1.20,
        "Li": 1.81,
        "Na": 2.27,
        "K": 2.75,
        "Mg": 1.73,
        "Ca": 2.31,
        "Zn": 1.39,
        "C": 1.70,
        "N": 1.55,
        "O": 1.52,
        "B": 1.65,
        "F": 1.47,
        "P": 1.80,
        "S": 1.80,
        "Cl": 1.75,
        "Br": 1.85,
        "I": 1.98,
        "Ag": 1.72,
        "Au": 1.66,
        "D": 0.00,
        "X": 0.00,
    }
    return elem_vdW


# Atomic covalent radii.
def dict_covalent() -> dict:
    elem_covalent = {
        "H": 0.31,
        "Li": 1.28,
        "Na": 1.66,
        "K": 2.03,
        "Mg": 1.41,
        "Ca": 1.71,
        "Zn": 1.18,
        "C": 0.76,
        "N": 0.71,
        "O": 0.66,
        "B": 0.84,
        "F": 0.57,
        "P": 1.07,
        "S": 1.05,
        "Cl": 1.02,
        "Br": 1.20,
        "I": 1.39,
        "Ag": 1.45,
        "Au": 1.36,
        "D": 0.00,
        "X": 0.00,
    }
    return elem_covalent


# Cutoff distances for bond identification.
def dict_cutoff() -> dict:
    # Set up a dictionary with cutoff distances for the molecule identification by combining the covalent radii of all
    # element combinations.
    comb_cutoff = dict()

    # Get the covalent radii
    elem_covalent = dict_covalent()

    # Define a list of all elements in elem_covalent. Then a list of all possible combinations.
    elem_list = list(elem_covalent.keys())
    elem_comb = [(elem_list[i], elem_list[j]) for i in range(len(elem_list)) for j in range(i, len(elem_list))]

    # Define the cutoff distances.
    dist_comb = [elem_covalent[elem_comb[i][0]] + elem_covalent[elem_comb[i][1]] for i in range(len(elem_comb))]
    comb_cutoff = dict(zip(dist_comb, elem_comb))

    # Add a tolerance of 0.6 to the cutoff distances. This needs to be done to identify molecules, where some bonds are
    # stretched.
    comb_cutoff = {k + 0.6: v for k, v in comb_cutoff.items()}

    return comb_cutoff
