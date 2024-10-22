import numpy as np
import pandas as pd

import conan.defdict as ddict


def sort_species_order(traj_file):
    sort_species = ddict.get_input(
        "Do you want to sort the rows in a certain species order? [y/n]: ", traj_file.args, "string"
    )
    if sort_species == "y":
        species_order = ddict.get_input(
            "Enter the species in the order you want them to be sorted: ", traj_file.args, "string"
        )
        species_order = [int(i) for i in species_order.split(",")]
        traj_file.frame0["Species"] = pd.Categorical(
            traj_file.frame0["Species"], categories=species_order, ordered=True
        )
        traj_file.frame0["index"] = traj_file.frame0.index
        traj_file.frame0 = traj_file.frame0.sort_values(by=["Species", "Molecule", "index"])
    else:
        traj_file.frame0 = traj_file.frame0.sort_values(by=["Species", "Molecule"])
    traj_file.frame0 = traj_file.frame0.drop("index", axis=1, errors="ignore")


def save_simulation_box(traj_file):
    ddict.printLog("\n-> xyz file of simulation box.")
    sort_species_order(traj_file)
    with open("simbox_frame.xyz", "w") as frame_print:
        frame_print.write("%d\n#Made with CONAN\n" % len(traj_file.frame0))
        for _, row in traj_file.frame0.iterrows():
            frame_print.write("%s\t%0.3f\t%0.3f\t%0.3f\n" % (row["Element"], row["x"], row["y"], row["z"]))
    ddict.printLog("-> Saved simulation box as simbox_frame.xyz.")


def save_pore_structure(traj_file, molecules):
    ddict.printLog("\n-> Pics of pore structure(s).")
    for i in range(len(molecules.structure_data["CNT_centers"])):
        CNT_atoms_pic = traj_file.frame0.loc[traj_file.frame0["Struc"] == "Pore%d" % (i + 1)]
        ddict.printLog(CNT_atoms_pic)
        CNT_atoms_pic = CNT_atoms_pic.drop(["Charge", "Struc", "CNT", "Molecule", "Label", "Species"], axis=1)
        add_centerpoint = ddict.get_input(
            "Add the center point of the CNT to the file? [y/n] ", traj_file.args, "string"
        )
        if add_centerpoint == "y":
            CNT_atoms_pic.loc[len(CNT_atoms_pic.index)] = [
                "X",
                molecules.structure_data["CNT_centers"][0][0],
                molecules.structure_data["CNT_centers"][0][1],
                molecules.structure_data["CNT_centers"][0][2],
            ]
        with open(f"pore{i + 1}.xyz", "w") as CNT_atoms_print:
            CNT_atoms_print.write("%d\n#Made with CONAN\n" % len(CNT_atoms_pic))
            for _, row in CNT_atoms_pic.iterrows():
                CNT_atoms_print.write("%s\t%0.3f\t%0.3f\t%0.3f\n" % (row["Element"], row["x"], row["y"], row["z"]))
        ddict.printLog(f"-> saved as pore{i + 1}.xyz")


def save_tube_pictures(traj_file, molecules):
    ddict.printLog("\n-> Tube pictures.")
    for i in range(len(molecules.structure_data["CNT_centers"])):
        CNT_atoms_pic = pd.DataFrame(traj_file.frame0.loc[traj_file.frame0["CNT"] == i + 1])
        CNT_atoms_pic = CNT_atoms_pic.drop(["Charge", "Struc", "CNT", "Molecule"], axis=1)
        add_liquid = ddict.get_input(f"Add liquid which is inside the CNT{i + 1}? [y/n] ", traj_file.args, "string")

        if add_liquid == "y":
            add_liquid2 = ddict.get_input(
                "Add all confined atoms (1), or entire molecules (2) ? [1/2] ", traj_file.args, "int"
            )
            CNT_atoms_pic["Molecule"] = np.nan
            for index, row in traj_file.frame0.iterrows():
                if (
                    row["Struc"] == "Liquid"
                    and row["z"] <= CNT_atoms_pic["z"].max()
                    and row["z"] >= CNT_atoms_pic["z"].min()
                ):
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
                mol_list = CNT_atoms_pic["Molecule"].unique()
                tube_atoms_mol = pd.DataFrame(columns=["Element", "x", "y", "z", "Label", "Species", "Molecule"])
                for index, row in traj_file.frame0.iterrows():
                    if row["Molecule"] in mol_list:
                        tube_atoms_mol.loc[index] = [
                            row["Element"],
                            row["x"],
                            row["y"],
                            row["z"],
                            row["Label"],
                            row["Species"],
                            row["Molecule"],
                        ]
                CNT_atoms_pic = pd.concat([CNT_atoms_pic, tube_atoms_mol], ignore_index=True)
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

        with open(f"CNT{i + 1}.xyz", "w") as tube_atoms_print:
            tube_atoms_print.write("%d\n#Made with CONAN\n" % len(CNT_atoms_pic))
            for _, row in CNT_atoms_pic.iterrows():
                tube_atoms_print.write("%s\t%0.3f\t%0.3f\t%0.3f\n" % (row["Element"], row["x"], row["y"], row["z"]))
        ddict.printLog(f"-> Saved as CNT{i + 1}.xyz")


def xyz_generator(traj_file, molecules) -> None:
    ddict.printLog("(1) Produce xyz file of the whole simulation box.")
    ddict.printLog("(2) Produce xyz file of empty pore structure.")
    ddict.printLog("(3) Produce xyz file of a CNT.")
    analysis1_choice = int(ddict.get_input("What do you want to do?: ", traj_file.args, "int"))

    if analysis1_choice == 1:
        save_simulation_box(traj_file)
    elif analysis1_choice == 2:
        save_pore_structure(traj_file, molecules)
    elif analysis1_choice == 3:
        save_tube_pictures(traj_file, molecules)
    else:
        ddict.printLog("\nThe analysis you entered is not known.")
    ddict.printLog("")
