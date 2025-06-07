import math
import os
import re
import sys
from collections import Counter
from typing import Tuple

import networkx as nx
import numpy as np
import pandas as pd
from prettytable import PrettyTable
from rdkit import Chem
from rdkit.Chem import Draw
from scipy.spatial import cKDTree

import conan.defdict as ddict


class TrajectoryFile:
    def __init__(self, file, args):
        self.args = args
        self.file = file
        self.file_type = self.get_file_type()
        self.num_atoms, self.lines_per_frame = self.get_num_atoms()
        self.box_size = self.simbox_dimension()
        self.frame0 = self.get_frame()
        self.frame1 = self.get_frame(1)
        self.frame0 = self.frame_comparison(self.frame0, self.frame1)

        (
            self.number_of_frames,
            self.lines_chunk,
            self.lines_last_chunk,
            self.num_chunks,
            self.chunk_size,
            self.last_chunk_size,
        ) = self.traj_chunk_info()

    def get_file_type(self):
        if self.file.endswith(".xyz"):
            return "xyz"
        elif self.file.endswith(".pdb"):
            return "pdb"
        elif self.file.endswith(".lmp") or self.file.endswith(".lammpstrj"):
            return "lmp"
        else:
            return None

    def get_num_atoms(self):
        if self.file_type == "xyz":
            with open(self.file) as f:
                num_atoms = int(f.readline())
                lines_per_frame = num_atoms + 2

        elif self.file_type == "pdb":
            with open(self.file) as f:
                second_line = 0
                next(f)
                for i, line in enumerate(f):
                    if "CRYST1" in line:
                        second_line = i
                        break
                num_atoms = second_line - 1
                lines_per_frame = num_atoms + 2

        elif self.file_type == "lmp":
            with open(self.file, "r") as f:
                for i in range(3):
                    next(f)
                num_atoms = int(f.readline())
                lines_per_frame = num_atoms + 9
        else:
            return None

        ddict.printLog(f"\nNumber of atoms: {num_atoms}\n")

        return num_atoms, lines_per_frame

    def simbox_dimension(self):

        if self.file_type == "xyz":
            same_length = ddict.get_input("Is the simulation box a cube? [y/n]: ", self.args, "str")
            if same_length == "y":
                simbox_x = float(ddict.get_input("What is the edge length of the cube? [\u00c5]: ", self.args, "float"))
                simbox_z = simbox_y = simbox_x
            else:
                ddict.printLog("Enter the dimensions of the simulation box [\u00c5]:")
                simbox_x = float(ddict.get_input("[X]   ", self.args, "float"))
                simbox_y = float(ddict.get_input("[Y]   ", self.args, "float"))
                simbox_z = float(ddict.get_input("[Z]   ", self.args, "float"))
                ddict.printLog("")

        elif self.file_type == "pdb":
            with open(self.file) as f:
                first_line = f.readline()
                box_info = first_line.split()
                simbox_x = float(box_info[1])
                simbox_y = float(box_info[2])
                simbox_z = float(box_info[3])

        elif self.file_type == "lmp":
            with open(self.file, "r") as f:
                for i in range(5):
                    next(f)
                x_dimensions = f.readline().split()
                x_min_boundary = float(x_dimensions[0])
                x_max_boundary = float(x_dimensions[1])
                simbox_x = x_max_boundary - x_min_boundary
                y_dimensions = f.readline().split()
                y_min_boundary = float(y_dimensions[0])
                y_max_boundary = float(y_dimensions[1])
                simbox_y = y_max_boundary - y_min_boundary
                z_dimensions = f.readline().split()
                z_min_boundary = float(z_dimensions[0])
                z_max_boundary = float(z_dimensions[1])
                simbox_z = z_max_boundary - z_min_boundary

        box_size = (simbox_x, simbox_y, simbox_z)
        ddict.printLog(
            f"The simulation box dimensions are [\u00c5]: {float(box_size[0]):.3f} x {float(box_size[1]):.3f} x "
            f"{float(box_size[2]):.3f}"
        )

        return box_size

    def get_frame(self, frame_number=0):
        df_frame = pd.DataFrame()

        try:
            if self.file_type == "xyz":
                # Skip the first two lines of the frame and read the next `num_atoms` lines
                df_frame = pd.read_csv(
                    self.file,
                    sep=r"\s+",
                    header=None,
                    skiprows=2 + frame_number * self.lines_per_frame,
                    nrows=self.num_atoms,
                )

                if df_frame.shape[1] == 5:
                    df_frame.columns = ["Element", "x", "y", "z", "Charge"]
                else:
                    # raise ValueError("Unexpected number of columns in xyz file")
                    # drop all columns except the first 4
                    df_frame = df_frame.iloc[:, :4]
                    df_frame.columns = ["Element", "x", "y", "z"]

            elif self.file_type == "pdb":
                df_frame = pd.read_csv(
                    self.file,
                    sep=r"\s+",
                    nrows=self.num_atoms,
                    header=None,
                    skiprows=1 + frame_number * self.lines_per_frame,
                    names=["Record", "Atom number", "Label", "Molecule", "x", "y", "z", "Charge"],
                )
                df_frame["Element"] = df_frame["Label"].str[0]
                df_frame.drop(columns=["Record", "Atom number", "Label"], inplace=True)

            elif self.file_type == "lmp":
                with open(self.file, "r") as f:
                    try:
                        for i in range(8 + frame_number * self.lines_per_frame):
                            next(f)
                    except StopIteration:
                        raise ValueError(f"Frame {frame_number} does not exist in the file.")

                    header = f.readline().strip().split()
                    # Check if "element", "Element", or "type" is in the header
                    try:
                        atom_type_pos = header.index("element") - 2
                    except ValueError:
                        try:
                            atom_type_pos = header.index("Element") - 2
                        except ValueError:
                            atom_type_pos = header.index("type") - 2

                    x_keywords = ["xu", "x", "ix"]
                    y_keywords = ["yu", "y", "iy"]
                    z_keywords = ["zu", "z", "iz"]

                    def find_index(header, keywords):
                        for keyword in keywords:
                            if keyword in header:
                                return header.index(keyword) - 2
                        raise ValueError("None of the position keywords found in header")

                    atom_x_pos = find_index(header, x_keywords)
                    atom_y_pos = find_index(header, y_keywords)
                    atom_z_pos = find_index(header, z_keywords)
                    atom_mol_pos = header.index("mol") - 2 if "mol" in header else None
                    atom_charge_pos = header.index("q") - 2 if "q" in header else None
                    atom_id_pos = header.index("id") - 2 if "id" in header else None

                    positions = {
                        "id": atom_id_pos,
                        "Element": atom_type_pos,
                        "x": atom_x_pos,
                        "y": atom_y_pos,
                        "z": atom_z_pos,
                        "Molecule": atom_mol_pos,
                        "Charge": atom_charge_pos,
                    }

                    # read the frame
                    df_frame = pd.read_csv(
                        self.file,
                        sep=r"\s+",
                        nrows=self.num_atoms,
                        header=None,
                        skiprows=9 + frame_number * self.lines_per_frame,
                    )

                # Rename the columns according to the positions
                for key, pos in positions.items():
                    if pos is not None:
                        df_frame.rename(columns={pos: key}, inplace=True)

                df_frame.drop(columns=["id"], inplace=True)

            else:
                ddict.printLog("The file is not in a known format. Use the help flag (-h) for more information")
                sys.exit()

            # Check if there is a 'Molecule' or 'Charge' column in the dataframe. If not, add an empty column.
            if "Molecule" not in df_frame.columns:
                df_frame["Molecule"] = None
            if "Charge" not in df_frame.columns:
                df_frame["Charge"] = None

            df_frame = df_frame[["Element", "x", "y", "z", "Molecule", "Charge"]]

        except (pd.errors.EmptyDataError, IndexError) as e:
            # Handle the case where the frame does not exist or cannot be read
            ddict.printLog(f"Warning: Could not read frame {frame_number}. Error: {e}")
            df_frame = pd.DataFrame()

        return df_frame

    def frame_comparison(self, frame0, frame1):
        # Check if frame1 is empty. If so, return frame0 with an empty 'Struc' column.
        if frame1.empty:
            frame0["Struc"] = False
        # Check which atoms did not move, they should have the same x y z coordinates. Label them as True in the Struc
        # column.
        else:
            frame0["Struc"] = (frame0["x"] == frame1["x"]) & (frame0["y"] == frame1["y"]) & (frame0["z"] == frame1["z"])

        return frame0

    def traj_chunk_info(self):
        # GENERAL INFORMATION ON CHUNKS
        ddict.printLog("-> Reading the trajectory.\n")
        trajectory_file_size = os.path.getsize(self.args["trajectoryfile"])
        with open(self.args["trajectoryfile"]) as f:
            number_of_lines = sum(1 for i in f)

        number_of_frames = int(number_of_lines / self.lines_per_frame)

        # Calculate how many bytes each line of the trajectory file has.
        bytes_per_line = trajectory_file_size / (number_of_lines)
        # The number of frames in a chunk. Each chunk is roughly 50 MB large.
        chunk_size = int(100000000 / ((self.lines_per_frame) * bytes_per_line))
        # The number of chunks (always round up).
        number_of_chunks = math.ceil(number_of_frames / chunk_size)
        # The number of frames in the last chunk.
        last_chunk_size = number_of_frames - (number_of_chunks - 1) * chunk_size
        number_of_bytes_per_chunk = chunk_size * (self.lines_per_frame) * bytes_per_line
        number_of_lines_per_chunk = chunk_size * (self.lines_per_frame)
        number_of_lines_last_chunk = last_chunk_size * (self.lines_per_frame)
        # Table with the information on the trajectory file.
        table = PrettyTable(["", "Trajectory", "Chunk(%d)" % (number_of_chunks)])
        table.add_row(
            [
                "Size in MB",
                "%0.1f" % (trajectory_file_size / 1000000),
                "%0.1f (%0.1f)"
                % (
                    number_of_bytes_per_chunk / 1000000,
                    last_chunk_size * (self.lines_per_frame) * bytes_per_line / 1000000,
                ),
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


class Molecule:
    def __init__(self, traj_file):

        self.neglect_atom_kind = self.exclude_atom_kind(traj_file)
        self.all_atoms = self.dataframe_to_list(traj_file.frame0)
        (self.molecules, self.molecule_bonds, self.molecules_sym, self.molecule_bonds_sym, self.molecule_counter) = (
            self.identify_molecules_and_bonds(traj_file)
        )

        (self.molecule_frame, self.unique_molecule_frame, self.molecule_count) = self.get_unique_molecule_frame(
            traj_file.frame0
        )

        self.print_molecule_info()
        self.print_picture()

        (
            self.structure_data,
            self.min_z_pore,
            self.max_z_pore,
            self.length_pore,
            self.center_pore,
            self.CNT_centers,
            self.tuberadii,
            self.CNT_atoms,
            self.Walls_positions,
        ) = self.structure_recognition(traj_file)

    def exclude_atom_kind(self, traj_file):

        exclude_atom_kind = ["Na", "Zn", "Li", "Cl", "Br", "I", "D", "X"]
        neglect_atoms = []
        for atom in exclude_atom_kind:
            if any(traj_file.frame0["Element"] == atom):
                exclude_atom = str(
                    ddict.get_input(
                        f"Should {atom} atoms be excluded from the molecular recognition? [y/n]:  ",
                        traj_file.args,
                        "str",
                    )
                )
                if exclude_atom == "y":
                    neglect_atoms.append(atom)

        return neglect_atoms

    def dataframe_to_list(self, data_frame):
        str_atom_list = []
        for index, row in data_frame.iterrows():
            entry = {
                "Atom": index,
                "Element": str(row["Element"]),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "z": float(row["z"]),
            }
            str_atom_list.append(entry)

        return str_atom_list

    def identify_molecules_and_bonds(self, traj_file) -> Tuple[list, list]:

        atoms = self.all_atoms
        neglect_atoms = self.neglect_atom_kind

        covalent_radii = ddict.dict_covalent()

        bond_distances = {
            (e1, e2): (covalent_radii[e1] + covalent_radii[e2]) * 1.15 for e1 in covalent_radii for e2 in covalent_radii
        }

        # go through all bond distances and set all distances to 0, where a neglect atom is involved
        for key in bond_distances:
            if key[0] in neglect_atoms or key[1] in neglect_atoms:
                bond_distances[key] = 0

        # Add all atoms as nodes to the graph
        atom_positions = np.array([[atom["x"], atom["y"], atom["z"]] for atom in atoms]) % traj_file.box_size
        atom_elements = [atom["Element"] for atom in atoms]

        simbox_G = nx.Graph()
        tree = cKDTree(atom_positions, boxsize=traj_file.box_size)

        # Find pairs within max bond_distance
        max_bond_distance = max(bond_distances.values())
        pairs = tree.query_pairs(max_bond_distance)

        for i, j in pairs:
            bond_distance = bond_distances.get((atom_elements[i], atom_elements[j]), float("inf"))

            # Correct the distance considering the minimum image convention
            distance = minimum_image_distance(atom_positions[i], atom_positions[j], traj_file.box_size)

            if distance <= bond_distance:
                # Add an edge in the graph if atoms are bonded
                simbox_G.add_edge(i, j)

        # Set the atoms[i]['Atom'] column to the index of the atom in the graph.
        for i in range(len(atoms)):
            atoms[i]["Index"] = atoms[i]["Atom"]
            atoms[i]["Atom"] = i

        # Each connected component in the graph represents a molecule
        molecules = [[atoms[i]["Atom"] for i in molecule] for molecule in nx.connected_components(simbox_G)]

        # Determine bonds for each molecule
        molecule_bonds = []
        for molecule in molecules:
            bonds = [sorted((i, j)) for i, j in simbox_G.edges(molecule)]
            molecule_bonds.append(bonds)

        # rename the molecule_bond entries to the original atom index.
        for i in range(len(molecule_bonds)):
            for j in range(len(molecule_bonds[i])):
                molecule_bonds[i][j][0] = atoms[molecule_bonds[i][j][0]]["Index"]
                molecule_bonds[i][j][1] = atoms[molecule_bonds[i][j][1]]["Index"]

        for i in range(len(molecules)):
            for j in range(len(molecules[i])):
                molecules[i][j] = atoms[molecules[i][j]]["Index"]

        # Translate the atom numbers in the molecules and molecule_bonds list of lists
        # to the element symbols (and save in a new file).
        molecules_sym = []
        for molecule in molecules:
            molecule_symloop = []
            for atom in molecule:
                # Check if atoms has an entry for atoms (might not, as neglected atom is removed)
                # if atom < len(atoms) and atoms[atom] is not None:
                #    molecule_symloop.append(atoms[atom]["Element"])
                molecule_symloop.append(atoms[atom]["Element"])
            molecules_sym.append(molecule_symloop)

        molecule_bonds_sym = []
        for molecule in molecule_bonds:
            molecule_bonds_symloop = []
            for bond in molecule:
                # Check if atoms has an entry for atoms (might not, as neglected atom is removed)
                # if bond[0] < len(atoms) and bond[1] < len(atoms):
                #    molecule_bonds_symloop.append((atoms[bond[0]]["Element"], atoms[bond[1]]["Element"]))
                molecule_bonds_symloop.append((atoms[bond[0]]["Element"], atoms[bond[1]]["Element"]))
            molecule_bonds_sym.append(molecule_bonds_symloop)

        # assign molecule numbers to the dataframe
        for i, molecule in enumerate(molecules):
            traj_file.frame0.loc[molecule, "Molecule"] = 1 + i
            molecule_counter = 1 + i

        # Finally assign each atom from each species an individual label
        label_counter = 1
        molecule_counter = traj_file.frame0["Molecule"][0]
        element_counter = traj_file.frame0["Element"][0]

        # Save original index to a new column, before sorting the dataframe
        # (to make sure atom wise and molecule wise sorting is possible)
        traj_file.frame0["original_index"] = traj_file.frame0.index

        traj_file.frame0 = traj_file.frame0.sort_values(by=["Molecule", "Element", "original_index"])

        for index, row in traj_file.frame0.iterrows():

            # Check if the molecule number has changed
            if row["Molecule"] != molecule_counter:
                label_counter = 1
                molecule_counter = row["Molecule"]

            # Check if the element has changed
            if row["Element"] != element_counter:
                label_counter = 1
                element_counter = row["Element"]

            label = row["Element"] + str(label_counter)
            traj_file.frame0.loc[index, "Label"] = label
            label_counter += 1

        # Finally revert the sorting of the dataframe to the original order.
        traj_file.frame0 = traj_file.frame0.sort_values(by="original_index").drop(columns="original_index")

        molecule_counter = traj_file.frame0["Molecule"].max()

        return molecules, molecule_bonds, molecules_sym, molecule_bonds_sym, molecule_counter

    def get_unique_molecule_frame(self, frame0) -> pd.DataFrame:

        molecule_frame = pd.DataFrame(columns=["Molecule", "Atoms", "Bonds", "Atoms_sym", "Bonds_sym"])

        molecule_frame["Molecule"] = range(1, int(self.molecule_counter) + 1)
        molecule_frame["Atoms"] = self.molecules
        molecule_frame["Bonds"] = self.molecule_bonds
        molecule_frame["Atoms_sym"] = self.molecules_sym
        molecule_frame["Bonds_sym"] = self.molecule_bonds_sym
        # Add another column with the Labels. Match the atom numbers in the Atoms column with the row
        # index in the frame0 dataframe.
        molecule_frame["Labels"] = molecule_frame["Atoms"].apply(lambda x: [frame0["Label"][i] for i in x])
        # Sort the lists in the Bonds_sym column to prepare to check for duplicates
        molecule_frame["Bonds_sym"] = molecule_frame["Bonds_sym"].apply(lambda x: sorted(x))

        # Make an independant copy of the molecule_frame dataframe name unique_molecule_frame
        unique_molecule_frame = molecule_frame.copy()

        # Drop all rows that are duplicates in the Bonds_sym column in a new dataframe
        unique_molecule_frame["Bonds_sym"] = unique_molecule_frame["Bonds_sym"].apply(tuple)
        unique_molecule_frame = unique_molecule_frame.drop_duplicates(subset=["Bonds_sym"])
        unique_molecule_frame = unique_molecule_frame.reset_index(drop=True)

        # Get the chemical formulas of the unique molecules by simply counting the number of atoms of each element
        # in the Atoms_sym column
        unique_molecule_frame["Molecule"] = unique_molecule_frame["Atoms_sym"].apply(
            lambda x: "".join(f"{element}{count}" for element, count in Counter(x).items())
        )

        # Adjust the labels in molecule_frame to include the molecule kind.
        for i, row in unique_molecule_frame.iterrows():
            for i2, row2 in molecule_frame.iterrows():
                if sorted(row["Bonds_sym"]) == sorted(row2["Bonds_sym"]):
                    new_labels = [f"{label}_{i + 1}" for label in molecule_frame.loc[i2, "Labels"]]
                    molecule_frame.at[i2, "Labels"] = new_labels
                    max_species = i + 1

        # Change the Labels column in the frame0 dataframe to the new labels
        for i, row in molecule_frame.iterrows():
            frame0.loc[row["Atoms"], "Label"] = row["Labels"]

        frame0["Species"] = frame0["Label"].str.split("_").str[1]

        # Remove the _1/2/3.. from the Labels column in the frame0 dataframe
        frame0["Label"] = frame0["Label"].str.split("_").str[0]

        old_max_species = max_species

        # Finally add all molecules (e.g. all neglected elements/atoms) which were not accounted for yet
        for element in frame0["Element"].unique():
            if frame0[frame0["Element"] == element]["Species"].isnull().any():
                frame0.loc[frame0["Element"] == element, "Species"] = int(max_species) + 1
                max_species += 1
                # add the info to the unique_molecule_frame dataframe
                unique_molecule_frame.loc[len(unique_molecule_frame)] = [
                    element,
                    [unique_molecule_frame.index],
                    None,
                    [element],
                    None,
                    [f"{element}1"],
                ]

        # add the molecule number to the frame0 dataframe if there is no molecule number yet.
        # identify the last molecule number in the frame0 dataframe
        last_molecule_number = frame0["Molecule"].max()
        # now loop over all atoms in the frame0 dataframe, if there is no molecule number yet, assign the
        # last_molecule_number + 1
        for i, row in frame0.iterrows():
            if row["Molecule"] is None:
                frame0.at[i, "Molecule"] = last_molecule_number + 1
                last_molecule_number += 1
                # aslo change the label to "'Element'1" for these atoms
                frame0.at[i, "Label"] = f"{row['Element']}1"

        # The 'Species' column currently is of type 'object' change it to 'int' for later calculations
        frame0["Species"] = frame0["Species"].astype(int)

        # Add correct atom label to the newly adde species in unique_molecule_frame
        for i in range(old_max_species, max_species):
            dummy_atom_list = frame0[frame0["Species"] == (i + 1)].index.tolist()
            unique_molecule_frame.loc[i, "Atoms"] = [dummy_atom_list[0]]

        # Sort the species numbering alphabetically
        unique_molecule_frame.sort_values("Molecule", inplace=True)
        unique_molecule_frame.reset_index(inplace=True, drop=True)

        old_species_list = [0] * len(unique_molecule_frame)
        for i, row in unique_molecule_frame.iterrows():
            old_species_list[i] = frame0["Species"][row["Atoms"][0]]
        for i, row in frame0.iterrows():
            for j in range(0, len(unique_molecule_frame)):
                if row["Species"] == old_species_list[j]:
                    frame0.loc[i, "Species"] = j + 1

        # Count how often each individual molecule/species occurs in the system
        # make a new dataframe with the columns 'Species' and 'Atom_count' and 'Molecule_count'
        molecule_count = pd.DataFrame()
        molecule_count["Atom_count"] = frame0["Species"].value_counts()
        molecule_count.reset_index(inplace=True)

        # Now 'Species' is no longer an index, so we can create the column
        molecule_count.rename(columns={"index": "Species"}, inplace=True)

        molecule_count = molecule_count.sort_values(by="Species").reset_index(drop=True)

        # The number of molecules is the Atom_count divided by the number of atoms in each species.
        # Get the number of atoms in each species. It is the length of the roe 'Atoms' in the
        # unique_molecule_frame dataframe.
        molecule_count["Molecule_count"] = molecule_count["Atom_count"] / unique_molecule_frame["Atoms"].apply(len)

        # Change the molecule count to integer values.
        molecule_count["Molecule_count"] = molecule_count["Molecule_count"].astype(int)

        unique_molecule_frame.index += 1
        molecule_count.index += 1

        return molecule_frame, unique_molecule_frame, molecule_count

    def print_molecule_info(self):
        # Print the information
        table = PrettyTable()
        table.field_names = ["Species", "Chemical formula", "No. molecules", "No. atoms per molecule"]

        for i, row in self.unique_molecule_frame.iterrows():
            table.add_row([i, row["Molecule"], int(self.molecule_count["Molecule_count"][i]), len(row["Atoms"])])

        ddict.printLog(" ")
        ddict.printLog(table)

    def print_picture(self):

        for i, row in self.unique_molecule_frame.iterrows():
            # Create a new graph (if the molecule is smaller than 50)
            if len(row["Atoms"]) < 50 and len(row["Atoms"]) > 1:
                mol = nx.Graph()

                # Combine atoms and labels into a dictionary
                atom_labels = dict(zip(row["Atoms"], row["Labels"]))

                for atom, label in atom_labels.items():
                    dummy_symbol = re.findall("[A-Za-z]+", label)[0]
                    if dummy_symbol not in ["X", "D"]:  # Skip dummy atoms
                        mol.add_node(atom, element=label)

                for bond in row["Bonds"]:
                    atom1, atom2 = bond
                    if atom1 in mol.nodes() and atom2 in mol.nodes():
                        mol.add_edge(atom1, atom2)

                rdkit_mol = Chem.RWMol()

                # Add atoms with labels
                atom_mapping = {}

                for node in mol.nodes():
                    atom_label = mol.nodes[node]["element"]
                    # Use just the element symbol
                    element_symbol = re.findall("[A-Za-z]+", atom_label)[0]
                    atom = Chem.Atom(element_symbol)
                    atom_idx = rdkit_mol.AddAtom(atom)
                    atom_mapping[node] = atom_idx
                    rdkit_mol.GetAtomWithIdx(atom_idx).SetProp("atomNote", atom_label)

                # Add bonds
                for edge in mol.edges():
                    atom_idx1 = atom_mapping[edge[0]]
                    atom_idx2 = atom_mapping[edge[1]]
                    bond_type = Chem.BondType.SINGLE
                    rdkit_mol.AddBond(atom_idx1, atom_idx2, bond_type)

                # Generate a 2D depiction
                rdkit_mol.UpdatePropertyCache(strict=False)
                Draw.rdDepictor.Compute2DCoords(rdkit_mol)

                # Create a drawer with atom options
                drawer = Chem.Draw.rdMolDraw2D.MolDraw2DCairo(500, 500)
                opts = drawer.drawOptions()

                for i in range(rdkit_mol.GetNumAtoms()):
                    atom = rdkit_mol.GetAtomWithIdx(i)
                    opts.atomLabels[i] = atom.GetSymbol()

                drawer.DrawMolecule(rdkit_mol)
                drawer.FinishDrawing()

                # Give the image a unique name of the chemical formula
                with open(f'{row["Molecule"]}.png', "wb") as f:
                    f.write(drawer.GetDrawingText())

    def structure_recognition(self, traj_file):
        structure_frame = traj_file.frame0[traj_file.frame0["Struc"]].copy()

        output = {}
        CNTs = []
        counter_pore = 0
        Walls = []
        Walls_positions = []
        counter_wall = 0
        min_z_pore = []
        max_z_pore = []
        length_pore = []
        center_pore = []
        CNT_centers = []
        tuberadii = []
        CNT_atoms = []
        which_pores = []

        # If the structure_frame is empty, then there are no structures in the simulation box.
        if structure_frame.empty or traj_file.args["manual"]:
            if structure_frame.empty:
                ddict.printLog(
                    "No frozen structures were found in the simulation box. \n",
                    color="red",
                )
                define_struc = ddict.get_input("Manually define structures? [y/n]: ", traj_file.args, "str")
                if define_struc == "n":
                    sys.exit()
            spec_molecule = molecule_choice(traj_file.args, traj_file.frame0, 2)
            structure_frame = traj_file.frame0[traj_file.frame0["Species"].isin(spec_molecule)].copy()

            # output["unique_molecule_frame"] = self.unique_molecule_frame

        # convert atom information to a list of dictionaries
        str_atom_list = []
        for index, row in structure_frame.iterrows():
            entry = {
                "Atom": index,
                "element": str(row["Element"]),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "z": float(row["z"]),
            }
            str_atom_list.append(entry)

        # Identify the structures and the bonds
        # molecules_struc, molecule_bonds_struc = identify_molecules_and_bonds(str_atom_list, traj_file.box_size)
        molecules_struc = structure_frame["Molecule"].unique()

        # Make a copy of the structure frame (to assure pandas treats it as a copy, not a view)
        structure_frame_copy = structure_frame.copy()
        traj_file.frame0["Struc"] = traj_file.frame0["Struc"].astype(str)
        structure_frame["Struc"] = structure_frame["Struc"].astype(str)
        structure_frame_copy["Struc"] = structure_frame_copy["Struc"].astype(str)

        # Consider all Molecules in the structure frame and get the maximum and minimum x, y and z coordinates for each
        # respective one.
        # If the difference in x, y and z is larger than 1.0, it is a pore.
        # If it is smaller in one direction, it is a wall.
        for i, molecule in enumerate(molecules_struc):
            x_max = structure_frame.loc[structure_frame["Molecule"] == molecule, "x"].max()
            x_min = structure_frame.loc[structure_frame["Molecule"] == molecule, "x"].min()
            y_max = structure_frame.loc[structure_frame["Molecule"] == molecule, "y"].max()
            y_min = structure_frame.loc[structure_frame["Molecule"] == molecule, "y"].min()
            z_max = structure_frame.loc[structure_frame["Molecule"] == molecule, "z"].max()
            z_min = structure_frame.loc[structure_frame["Molecule"] == molecule, "z"].min()

            # If the difference in x, y and z is larger than 1.0, it is a pore (not the case for Ag and Au walls)
            if (x_max - x_min) > 1.0 and (y_max - y_min) > 1.0 and (z_max - z_min) > 1.0:

                # If the structure consists of Gold or silver atoms, we define it as a wall (with a certain thickness).
                if structure_frame.loc[structure_frame["Molecule"] == molecule, "Element"].isin(["Au", "Ag"]).any():
                    pass
                    counter_wall += 1
                    ddict.printLog(
                        f"Structure {i} is a wall, labeled Wall{counter_wall} (Species: "
                        f"{structure_frame.loc[structure_frame['Molecule'] == molecule, 'Species'].unique()})\n"
                    )
                    structure_frame_copy.loc[structure_frame["Molecule"] == molecule, "Struc"] = f"Wall{counter_wall}"
                    Walls.append(f"Wall{counter_wall}")
                    Walls_positions.append(z_min)

                    continue

                counter_pore += 1
                ddict.printLog(
                    f"Structure {i} is a pore, labeled Pore{counter_pore} "
                    f"(Species: {structure_frame.loc[structure_frame['Molecule'] == molecule, 'Species'].unique()})\n"
                )

                # Change the structure column to pore{i}
                structure_frame_copy["Struc"] = structure_frame_copy["Struc"].astype(str)
                structure_frame_copy.loc[structure_frame["Molecule"] == molecule, "Struc"] = f"Pore{counter_pore}"
                CNTs.append(f"Pore{counter_pore}")

            # If the difference in x, y and z is smaller than 5.0, it is a wall.
            elif (x_max - x_min) < 5.0 or (y_max - y_min) < 5.0 or (z_max - z_min) < 5.0:
                counter_wall += 1
                ddict.printLog(
                    f"Structure {i} is a wall, labeled Wall{counter_wall} "
                    f"(Species: {structure_frame.loc[structure_frame['Molecule'] == molecule, 'Species'].unique()})"
                )
                if (x_max - x_min) < 1.0:
                    ddict.printLog(f"The wall extends in yz direction at x = {x_min:.2f} \u00c5.\n")
                if (y_max - y_min) < 1.0:
                    ddict.printLog(f"The wall extends in xz direction at y = {y_min:.2f} \u00c5.\n")
                if (z_max - z_min) < 1.0:
                    ddict.printLog(f"The wall extends in xy direction at z = {z_min:.2f} \u00c5.\n")
                    Walls_positions.append(z_min)
                structure_frame_copy.loc[structure_frame["Molecule"] == molecule, "Struc"] = f"Wall{counter_wall}"
                Walls.append(f"Wall{counter_wall}")

        # Copy the structure frame back to the original structure frame.
        structure_frame = structure_frame_copy
        traj_file.frame0["Struc"] = traj_file.frame0["Struc"].astype(structure_frame["Struc"].dtype)
        traj_file.frame0.loc[structure_frame.index, "Struc"] = structure_frame["Struc"]

        # Exchange all the entries in the 'Struc' column saying 'False' with 'Liquid'.
        traj_file.frame0.replace({"Struc": {"False": "Liquid"}}, inplace=True)

        # Print the structure information.
        ddict.printLog(f"\nTotal number of structures: {len(molecules_struc)}")
        ddict.printLog(f"Number of walls: {len(Walls)}")
        ddict.printLog(f"Number of pores: {len(CNTs)}\n")

        if len(CNTs) > 0:
            CNT_pore_question = ddict.get_input(
                "Does one of the pores contain CNTs oriented along the z axis of the simulation box? [y/n]: ",
                traj_file.args,
                "str",
            )
            if CNT_pore_question == "y":
                if len(CNTs) == 1:
                    which_pores = [1]
                    ddict.printLog("Only one Pore in the system.\n")
                else:
                    which_pores = ddict.get_input(
                        f"Which pore contains a CNT? [1-{len(CNTs)}]: ", traj_file.args, "str"
                    )
                    ddict.printLog("")

                    # split the input string into a list of integers. They are divided by a comma.
                    which_pores = [int(i) for i in which_pores.split(",")]
                    # keep all entries, which are equal or smaller than the number of pores.
                    which_pores = [i for i in which_pores if i <= len(CNTs)]

        for i in which_pores:
            pore = traj_file.frame0[traj_file.frame0["Struc"] == f"Pore{i}"].copy()
            min_z_pore.append(pore["z"].min())
            max_z_pore.append(pore["z"].max())

            # The length of each pore is the difference between the maximum and minimum z coordinate.
            length_pore.append(max_z_pore[i - 1] - min_z_pore[i - 1])

            # If the pore length is less than 2.0 Ang smaller than the box size,
            # it can be considered inifinite in z direction.
            # The the length is set to the box size.
            if length_pore[i - 1] > traj_file.box_size[2] - 2.0:
                length_pore[i - 1] = traj_file.box_size[2]
                ddict.printLog(f"Pore{i} is considered infinite in z direction.")
            ddict.printLog(f"The length of Pore{i} is {length_pore[i - 1]:.2f} \u00c5.")

            # The center of each pore is the average of the minimum and maximum z coordinate.
            center_pore.append((max_z_pore[i - 1] + min_z_pore[i - 1]) / 2)

            # A small tolerance is added.
            pore.loc[:, "z_distance"] = abs(pore["z"] - center_pore[i - 1])
            pore = pore.sort_values(by=["z_distance"])
            lowest_z = pore.iloc[0]["z_distance"] + 0.02
            CNT_ring = pore[pore["z_distance"] <= lowest_z].copy()

            # Delete all atoms in the CNT_ring dataframe, which are more than 0.1 angstrom away in the z direction
            # from the first atom in the CNT_ring dataframe.
            CNT_ring.loc[:, "z_distance"] = abs(CNT_ring["z"] - CNT_ring.iloc[0]["z"])
            CNT_ring = CNT_ring[CNT_ring["z_distance"] <= 0.1]

            # Calculate the average x and y coordinate of the atoms in the CNT_ring dataframe.
            x_center = CNT_ring["x"].mean()
            y_center = CNT_ring["y"].mean()
            ddict.printLog(
                (
                    f"The center of the CNT in Pore{i} is at "
                    f"{x_center:.2f}, {y_center:.2f}, {center_pore[i - 1]:.2f}) \u00c5."
                )
            )
            # Combine the x, y and z centers to a numpy array.
            center = np.array([x_center, y_center, center_pore[i - 1]])
            CNT_centers.append(center)

            # Calculate the radius of the CNT_ring.
            tuberadius = np.sqrt((CNT_ring.iloc[0]["x"] - x_center) ** 2 + (CNT_ring.iloc[0]["y"] - y_center) ** 2)
            tuberadii.append(tuberadius)
            ddict.printLog(f"The radius of the CNT in Pore{i} is {tuberadius:.2f} \u00c5.\n")

            # Calculate the xy-distance of the centerpoint of the CNT to all pore atoms.
            # If they are smaller/equal as the tuberadius, they belong to the CNT.
            traj_file.frame0.loc[traj_file.frame0["Struc"] == f"Pore{i}", "xy_distance"] = np.sqrt(
                (traj_file.frame0.loc[traj_file.frame0["Struc"] == f"Pore{i}", "x"] - x_center) ** 2
                + (traj_file.frame0.loc[traj_file.frame0["Struc"] == f"Pore{i}", "y"] - y_center) ** 2
            )
            traj_file.frame0.loc[traj_file.frame0["Struc"] == f"Pore{i}", "xy_distance"] = traj_file.frame0.loc[
                traj_file.frame0["Struc"] == f"Pore{i}", "xy_distance"
            ].round(2)

            # Save the information about the CNT in the structure dataframe, by adding a new column 'CNT' with the CNT
            # number, if the xy_distance is smaller/equal the tuberadius. 0.05 is the tolerance.
            traj_file.frame0.loc[traj_file.frame0["Struc"] == f"Pore{i}", "CNT"] = 0
            traj_file.frame0.loc[
                (traj_file.frame0["Struc"] == f"Pore{i}") & (traj_file.frame0["xy_distance"] <= tuberadius + 0.05),
                "CNT",
            ] = i

            # Delete the xy_distance column again.
            traj_file.frame0.drop(columns=["xy_distance"], inplace=True)

        # check if there is a 'CNT' column in the dataframe (if there are CNTs present). If not, add an empty column.
        if "CNT" not in traj_file.frame0.columns:
            traj_file.frame0["CNT"] = None

        # create a CNT_atoms dataframe with just the CNT atoms (value in the 'CNT' column is larger than 0.)

        CNT_atoms = traj_file.frame0[traj_file.frame0["CNT"] > 0].copy()

        output["id_frame"] = traj_file.frame0
        output["min_z_pore"] = min_z_pore
        output["max_z_pore"] = max_z_pore
        output["length_pore"] = length_pore
        output["CNT_centers"] = CNT_centers
        output["tuberadii"] = tuberadii
        output["CNT_atoms"] = CNT_atoms
        output["Walls_positions"] = Walls_positions

        return (
            output,
            min_z_pore,
            max_z_pore,
            length_pore,
            center_pore,
            CNT_centers,
            tuberadii,
            CNT_atoms,
            Walls_positions,
        )


def read_first_frame(args) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, tuple]:

    file = args["trajectoryfile"]
    traj_file = TrajectoryFile(file, args)

    return traj_file


def molecule_recognition(traj_file) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, tuple]:

    molecules = Molecule(traj_file)

    return molecules


def minimum_image_distance(position1, position2, box_size) -> float:
    # Calculate the minimum distance between two positions.
    d_pos = position1 - position2
    d_pos = d_pos - box_size * np.round(d_pos / box_size)
    return np.linalg.norm(d_pos)


def SortTuple(tup):
    n = len(tup)

    for i in range(n):
        for j in range(n - i - 1):
            if tup[j][0] > tup[j + 1][0]:
                tup[j], tup[j + 1] = tup[j + 1], tup[j]

    return tup


def molecule_choice(args, id_frame, mode) -> Tuple[int, list]:

    species_max = id_frame["Species"].max()
    spec_molecule = []
    spec_atom = []
    ddict.printLog("")

    if mode == 1:
        analysis_spec_molecule = ddict.get_input(
            "Do you want to perform the analysis for a specific molecule kind? (y/n) ", args, "string"
        )
        if analysis_spec_molecule == "y":
            spec_molecule = ddict.get_input(f"Which species to analyze? (1-{species_max}) ", args, "string")
            spec_molecule = spec_molecule.replace(", ", ",").split(",")
            spec_molecule = [int(i) for i in spec_molecule]
            # Ask user for the atom type to analyze. Multiple options are possible, default is 'all'.
            spec_atom = ddict.get_input("Which atoms to analyze? [default:all] ", args, "string")
            if spec_atom == "" or spec_atom == "[default:all]":
                spec_atom = "all"
            # Get the atom type into a list.
            spec_atom = spec_atom.replace(", ", ",").split(",")
            ddict.printLog(f"\n-> Species {spec_molecule} and atom type {spec_atom} will be analyzed.\n")
        else:
            ddict.printLog("-> All species and atoms will be analyzed.\n")

        return spec_molecule, spec_atom, analysis_spec_molecule

    if mode == 2:
        spec_molecule = ddict.get_input(f"Which species are the structures? (1-{species_max}) ", args, "string")
        spec_molecule = spec_molecule.replace(", ", ",").split(",")
        spec_molecule = [int(i) for i in spec_molecule]
        ddict.printLog(f"\n-> Species {spec_molecule} are set as structures.\n")

        return spec_molecule


# Function to identify molecular bonds from a distance search using a k-d tree.
def identify_molecules_and_bonds(atoms, box_size, neglect_atoms=[]) -> Tuple[list, list]:

    # Get the covalent radii
    covalent_radii = ddict.dict_covalent()

    # Define bond_distances
    bond_distances = {
        (e1, e2): (covalent_radii[e1] + covalent_radii[e2]) * 1.15 for e1 in covalent_radii for e2 in covalent_radii
    }

    for neg_atom in neglect_atoms:
        atoms = [atom for atom in atoms if atom["element"] != neg_atom]

    # Add all atoms as nodes to the graph
    atom_positions = np.array([[atom["x"], atom["y"], atom["z"]] for atom in atoms]) % box_size
    atom_elements = [atom["element"] for atom in atoms]

    # Create a graph with atoms as nodes and bonds as edges
    simbox_G = nx.Graph()

    # Create k-d tree for efficient search
    tree = cKDTree(atom_positions, boxsize=box_size)

    # Find pairs within max bond_distance
    pairs = tree.query_pairs(max(bond_distances.values()))

    for i, j in pairs:
        bond_distance = bond_distances.get((atom_elements[i], atom_elements[j]), float("inf"))

        # Correct the distance considering the minimum image convention
        distance = minimum_image_distance(atom_positions[i], atom_positions[j], box_size)

        if distance <= bond_distance:
            # Add an edge in the graph if atoms are bonded
            simbox_G.add_edge(i, j)

    # Set the atoms[i]['Atom'] column to the index of the atom in the graph.
    for i in range(len(atoms)):
        atoms[i]["Index"] = atoms[i]["Atom"]
        atoms[i]["Atom"] = i

    # Each connected component in the graph represents a molecule
    molecules = [[atoms[i]["Atom"] for i in molecule] for molecule in nx.connected_components(simbox_G)]

    # Determine bonds for each molecule
    molecule_bonds = []
    for molecule in molecules:
        bonds = [sorted((i, j)) for i, j in simbox_G.edges(molecule)]
        molecule_bonds.append(bonds)

    # rename the molecule_bond entries to the original atom index.
    for i in range(len(molecule_bonds)):
        for j in range(len(molecule_bonds[i])):
            molecule_bonds[i][j][0] = atoms[molecule_bonds[i][j][0]]["Index"]
            molecule_bonds[i][j][1] = atoms[molecule_bonds[i][j][1]]["Index"]

    for i in range(len(molecules)):
        for j in range(len(molecules[i])):
            molecules[i][j] = atoms[molecules[i][j]]["Index"]

    return molecules, molecule_bonds


# Edit a frame in pdb format.
def pdb(frame, element_masses, id_frame) -> pd.DataFrame:
    # drop the first and last line
    frame = frame.drop(frame.index[[0, len(frame) - 1]])
    # Split the frame into columns  and label them
    split_frame = frame[0].str.split(expand=True)
    split_frame.columns = ["Label", "Count", "Atom", "Molecule", "X", "Y", "Z", "Charge"]
    # now we try to convert the charge column to float
    try:
        split_frame["Charge"] = split_frame["Charge"].astype(float)
    except ValueError:
        # if it fails, we delete the charge column
        split_frame = split_frame.drop(["Charge"], axis=1)
        # and add a new column with the value 0
        split_frame["Charge"] = 0
    # drop label and count column
    split_frame = split_frame.drop(["Label", "Count"], axis=1)

    # Modify the 'Atom' column to contain the first character (and the second character if lowercase)
    split_frame["Atom"] = split_frame["Atom"].apply(
        lambda x: x[0] + (x[1].lower() if len(x) > 1 and x[1].islower() else "")
    )

    # Add a new column with the atomic mass of each atom.
    split_frame["Mass"] = split_frame["Atom"].map(element_masses)
    split_frame.reset_index(drop=True, inplace=True)

    return split_frame


# Edit a frame in xyz format.
def xyz(frame, element_masses, id_frame) -> pd.DataFrame:
    # drop the first two lines
    frame = frame.drop(frame.index[[0, 1]])

    # Split the frame into columns and label them. Also reset the index
    split_frame = frame[0].str.split(expand=True)
    split_frame.columns = ["Atom", "X", "Y", "Z"]
    split_frame.reset_index(drop=True, inplace=True)

    # Modify the 'Atom' column to contain the first character (and the second character if lowercase)
    split_frame["Atom"] = split_frame["Atom"].apply(
        lambda x: x[0] + (x[1].lower() if len(x) > 1 and x[1].islower() else "")
    )

    # Add a new column with the atomic mass of each atom.
    split_frame["Mass"] = split_frame["Atom"].map(element_masses)

    # now add the charge column of the id_frame to the split_frame
    split_frame["Charge"] = id_frame["Charge"]

    return split_frame


# Edit a frame in lammps format.
def lammpstrj(frame, element_masses, id_frame) -> pd.DataFrame:
    # Extract the header information
    header_line = frame.iloc[8, 0].split()
    headers = header_line[2:]
    # Check if "element", "Element", or "type" is in the header
    try:
        atom_type_pos = headers.index("element")
    except ValueError:
        try:
            atom_type_pos = headers.index("Element")
        except ValueError:
            atom_type_pos = headers.index("type")
    try:
        atom_x_pos = headers.index("xu")
        atom_y_pos = headers.index("yu")
        atom_z_pos = headers.index("zu")
    except ValueError:
        atom_x_pos = headers.index("x")
        atom_y_pos = headers.index("y")
        atom_z_pos = headers.index("z")

    atom_id_pos = headers.index("id") if "id" in headers else None
    atom_charge_pos = headers.index("q") if "q" in headers else None

    # Drop the first 9 lines
    frame = frame.drop(frame.index[range(9)])

    split_frame = frame[0].str.split(expand=True)
    split_frame.reset_index(drop=True, inplace=True)

    split_frame = split_frame.rename(columns={atom_type_pos: "Atom", atom_x_pos: "X", atom_y_pos: "Y", atom_z_pos: "Z"})

    # Modify the 'Atom' column to contain the first character (and the second character if lowercase)
    split_frame["Atom"] = split_frame["Atom"].apply(
        lambda x: x[0] + (x[1].lower() if len(x) > 1 and x[1].islower() else "")
    )

    # Add a new column with the atomic mass of each atom.
    split_frame["Mass"] = split_frame["Atom"].map(element_masses)

    # if there are charges provided, add them to the dataframe as a new column.
    if atom_charge_pos:
        split_frame["Charge"] = split_frame[atom_charge_pos].astype(float)

    if atom_id_pos:
        split_frame["ID"] = split_frame[atom_id_pos].astype(str)

    return split_frame


if __name__ == "__main__":
    # ARGUMENTS
    args = ddict.read_commandline()

    pd.options.mode.chained_assignment = (
        None  # default='warn'. Enabling this warning gices a false positive in molecular_recognition().
    )
