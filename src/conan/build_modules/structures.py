from __future__ import annotations

import abc
import copy
import math
import os
import random
from abc import ABC
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy import typing as npt

import conan.defdict as ddict


class Atom:  # ToDo: Evtl. abstrakte Klasse oder Datenklasse?
    """
    Represents an atom in a molecular structure.

    Attributes:
        label (str): The label of the atom.
        x (float): The x-coordinate of the atom.
        y (float): The y-coordinate of the atom.
        z (float): The z-coordinate of the atom.
        adjacent_groups (Union[List[Atom], None]): The list of adjacent atoms or None.
    """

    def __init__(self, label: str, x: float, y: float, z: float):
        """
        Initializes an Atom instance with specified label and coordinates.

        Args:
            label (str): The label of the atom.
            x (float): The x-coordinate of the atom.
            y (float): The y-coordinate of the atom.
            z (float): The z-coordinate of the atom.
        """
        self.label = label
        self.x = x
        self.y = y
        self.z = z
        self.adjacent_groups: Union[List[Atom], None] = None

    def find_adjacent_groups(self, group_list: List[Atom], van_der_waals_radii_dict: Dict[str, float]):
        """
        Determines which atoms from a provided list are adjacent to this atom based on Van der Waals radii.

        Args:
            group_list (List[Atom]): The list of groups to check against.
            van_der_waals_radii_dict (Dict[str, float]): Dictionary of Van der Waals radii.
        """

        # If the group list is empty, there is nothing to do
        if not group_list:
            return

        # Set the adjacent_groups list to empty, so we can fill it
        self.adjacent_groups = []

        # Loop over all groups
        for group in group_list:
            if group == self:
                continue  # Current group cannot be its own neighbor
            elif isinstance(group, Atom):  # Ensures the object is an Atom
                if self._is_atom_neighbor(group, van_der_waals_radii_dict):
                    self.adjacent_groups.append(group)  # Adds the atom to the adjacent list if they are neighbors
            else:
                ddict.printLog(f"WARNING: Unknown group type {group} found in 'find_adjacent_groups'")

    # PRIVATE
    def _is_atom_neighbor(self, atom: Atom, van_der_waals_radii_dict: Dict[str, float]) -> bool:
        """
        Determines if another atom is a neighbor.

        Args:
            atom (Atom): The other atom.
            van_der_waals_radii_dict (Dict[str, float]): Dictionary of Van der Waals radii.

        Returns:
            bool: True if the other atom is a neighbor, False otherwise.
        """
        pass


class FunctionalGroup:  # ToDo: Gedacht für Sauerstoffdotierung?
    """
    Represents a functional group in a molecular structure.

    Attributes:
        group_name (str): The name of the group.
        group_count (int): The number of times the group appears.
        exclusion_radius (float): The exclusion radius for the group.
        atom_positions (List[Tuple[str, float, float, float]]): The positions of the atoms in the group.
    """

    def __init__(self, group_parameters: Dict[str, Union[str, int, float]], structure_library_path: str):
        """
        Initializes a FunctionalGroup instance with specific parameters and a reference to a structure library.

        Args:
            group_parameters (Dict[str, Union[str, int, float]]): Parameters for the functional group.
            structure_library_path (str): Path to the structure library.
        """
        # Extract and set the name of the group from the parameters
        self.group_name = group_parameters["group"]
        # Convert the string count to an integer and set it
        self.group_count = int(group_parameters["group_count"])
        # Set the exclusion radius if specified in the parameters
        if "exclusion_radius" in group_parameters:
            self.exclusion_radius = float(group_parameters["exclusion_radius"])
        # Load the positions of atoms in this functional group from an external library file
        self.atom_positions: List[Tuple[str, float, float, float]] = self.__read_positions_from_library(
            structure_library_path
        )

    def remove_anchors(self) -> List[Tuple[str, float, float, float]]:
        """
        Removes anchor atoms from the functional group.

        Returns:
            List[Tuple[str, float, float, float]]: Atom positions without anchors.
        """
        # atom_positions_without_anchor = self.atom_positions
        # for position in self.atom_positions:
        #     if position[0] == 'X':
        #         atom_positions_without_anchor.remove(position)
        # return atom_positions_without_anchor

        # Filter out atoms that are designates as anchors using a list comprehension (typically marked with 'X')
        return [
            pos for pos in self.atom_positions if pos[0] != "X"
        ]  # ToDo: Sollte etwas effizeinter sein, da remove() innerhalb einer Schleife eine worst case complexity von
        # O(n^2) hat, eine List comprehension hat aber nur eine worst case complexity von O(n)

    # PRIVATE
    def __read_positions_from_library(self, structure_library_path: str) -> List[Tuple[str, float, float, float]]:
        """
        Reads atom positions from the structure library.

        Args:
            structure_library_path (str): Path to the structure library.

        Returns:
            List[Tuple[str, float, float, float]]: List of atom positions.
        """
        # Build the full path to the group's file using its name
        group_path = os.path.join(structure_library_path, f"{self.group_name}.xyz")
        atom_list = []
        # Open the file and read the contents
        with open(group_path, "r") as file:
            # First line is the number of atoms
            number_of_lines = int(file.readline().strip())
            # Skip the comment line
            file.readline().strip()
            # Read each atom's data
            for _ in range(number_of_lines):
                line = file.readline().strip().split()
                # Append the atom's symbol and coordinates converted to float
                atom_list.append((line[0], float(line[1]), float(line[2]), float(line[3])))
        return atom_list


class Structure(
    ABC
):  # ToDo: Klasse sollte wahrscheinlich abstract sein oder, sodass diese nicht selbst instanziiert werden kann, sondern
    # nur die abgeleiteten Klassen
    """
    Represents a molecular structure. This class serves as a base for managing molecular data, designed to be abstract
    to support specific structure types derived from it.

    Attributes:
        _structure_df (pd.DataFrame): DataFrame containing the structure data.
        group_list (List[FunctionalGroup]): List of functional groups within the structure.
    """

    def __init__(self):  # ToDo: Gibt aktuell noch keine init-Methode für diese Klasse. So vielleicht?
        """
        Initializes a Structure instance.
        """
        self._structure_df: pd.DataFrame = pd.DataFrame()  # Empty DataFrame to hold structure data
        self.group_list: List[FunctionalGroup] = []  # Empty list to hold functional groups

    @abc.abstractmethod
    def add(self, parameters: Dict[str, Union[str, int, float]]):
        """
        Adds a functional group to the structure at a specified position, modifying the chemical properties. This method
        ensures that the child classes implement this functionality.

        Args:
            parameters (Dict[str, Union[str, int, float]]): The parameters for the addition.
        """
        pass

    @abc.abstractmethod
    def _add_group_on_position(self, selected_position: List[float]):
        """
        Adds a functional group to the structure at a specified position, with automatic adjustment to ensure proper
        orientation based on the local surface normal. This method ensures that the child classes implement this
        functionality.

        Args:
            selected_position (List[float]): Coordinates [x, y, z] where the group should be added.
        """
        pass

    # INTERFACE
    def print_xyz_file(self, file_name: str):
        """
        Prints the structure to an XYZ file.

        Args:
            file_name (str): The name of the file to write to.
        """
        self._write_xyz_file(file_name, self._structure_df)

    def remove_atom_by_index(self, index: int):
        """
        Removes an atom from the structure by its index.

        Args:
            index (int): The index of the atom to remove.
        """
        self._structure_df.drop([index], inplace=True)  # Removes the specified index from the DataFrame

    # PRIVATE
    def _write_xyz_file(self, name_output: str, coordinates: pd.DataFrame) -> None:
        """
        Writes the structure to an XYZ file.

        Args:
            name_output (str): The name of the output file.
            coordinates (pd.DataFrame): The coordinates to write.
        """
        directory = "structures"  # Directory where the file will be saved
        filename = f"{name_output}.xyz"  # Complete file name
        filepath = os.path.join(directory, filename)

        # Ensure the directory exists
        os.makedirs(directory, exist_ok=True)

        # Handle file creation and writing in a context manager
        with open(filepath, "w") as xyz:
            xyz.write(f"   {len(coordinates)}\n")  # Writes the number of atoms
            xyz.write("# Generated with CONAN\n")  # Comment line in XYZ format
            coordinates.to_csv(xyz, sep="\t", header=False, index=False, float_format="%.3f")  # Writes the coordinates

    def _initialize_functional_groups(self, parameters: Dict[str, Union[str, int, float]]):
        """
        Initializes functional groups based on the given parameters.

        Args:
            parameters (Dict[str, Union[str, int, float]]): The parameters for the functional groups.
        """
        # Depending on whether build_main is called from CONAN.py
        # or as standalone module, the structure library is somewhere else
        base_path = os.path.dirname(os.path.abspath(__file__))  # Base path of the script
        structure_library_path = os.path.join(base_path, "structure_lib")  # Default path to the structure library
        if not os.path.exists(structure_library_path):
            structure_library_path = os.path.join(base_path, "..", "structure_lib")  # Alternative path
        structure_library_path = os.path.normpath(structure_library_path)  # Normalizes the path

        self.group_list = []
        self.group_list.append(FunctionalGroup(parameters, structure_library_path))  # Adds a new functional group

    def rotation_matrix_from_vectors(
        self, vec2: npt.NDArray
    ) -> (
        npt.NDArray
    ):  # ToDo: Hier, oder vielleicht sogar besser in utility Modul oder Utility-Klasse, die verchiedene statische
        # Methoden enthält, da eigentlich nicht sehr objektspezifisch; auf alle Fälle Code-Duplikate vermeiden!
        """
        Computes a rotation matrix to align the first vector (vec1, defaulting to the z-axis [0, 0, 1]) with a second
        vector (vec2).

        Args:
            vec2 (np.ndarray): The target vector to align with the z-axis.

        Returns:
            np.ndarray: The rotation matrix that when multiplied by vec1 results in vec2.
        """
        # Define the default vector vec1 as the positive z-axis
        vec1 = np.array([0, 0, 1])

        # Normalize both vectors to ensure they are unit vectors
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)

        # Compute the cross product of vec1 and vec2 to find the axis of rotation
        v = np.cross(a, b)

        # Calculate the dot product, which gives the cosine of the angle between vec1 and vec2
        c = np.dot(a, b)

        # Calculate the sine of the angle using the magnitude of the cross product vector
        s = np.linalg.norm(v)

        # The skew-symmetric cross-product matrix of vector v
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

        # Calculate the rotation matrix using the Rodrigues' rotation formula:
        # R = I + sin(theta) * K + (1 - cos(theta)) * K^2
        # This formula is derived for rotating one vector onto another.
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))


class Structure1d(Structure):
    """
    Represents a 1-dimensional molecular structure, typically used to model structures like carbon nanotubes.

    Attributes:
        bond_length (float): The bond length between consecutive atoms in the structure.
    """

    def __init__(self, parameters: Dict[str, Union[str, int, float]], keywords: List[str]):
        """
        Initializes a Structure1D instance with specified parameters and keywords.
        Inherits initialization form Structure.

        Args:
            parameters (Dict[str, Union[str, int, float]]): The parameters for the structure.
            keywords (List[str]): The keywords for the structure.
        """
        super().__init__()
        # Set the bond length from parameters
        self.bond_length: float = parameters["bond_length"]
        # Method to construct the CNT based on provided parameters and keywords
        self._build_CNT(parameters, keywords)

    def stack(self, parameters: Dict[str, Union[str, int, float]], keywords: List[str]):
        """
        Stacks multiple instances of carbon nanotubes within the structure based on the provided parameters.

        Args:
            parameters (Dict[str, Union[str, int, float]]): The parameters for the stacking.
            keywords (List[str]): The keywords for the stacking.
        """
        if self._structure_df is not None:  # Ensure there is a structure loaded before attempting to stack
            self._stack_CNTs(parameters, keywords)  # Private method that handles the actual stacking logic

    # INTERFACE
    def add(self, parameters: Dict[str, Union[str, int, float]]):
        """
        Adds a functional group to the structure at a specified position, modifying the chemical properties.

        Args:
            parameters (Dict[str, Union[str, int, float]]):
                Parameters including the position and details about the group to be added.
        """
        parameters["group_count"] = 1  # Assuming only one group is added at a time
        self._initialize_functional_groups(parameters)  # Initializes functional groups based on provided parameters
        # Extract position from the structure DataFrame to place the new group
        position = [
            self._structure_df.iloc[parameters["position"], 1],  # X coordinate
            self._structure_df.iloc[parameters["position"], 2],  # Y coordinate
            self._structure_df.iloc[parameters["position"], 3],
        ]  # Z coordinate
        self._add_group_on_position(position)  # Adds the group at the calculated position

    # PRIVATE
    def _add_group_on_position(self, selected_position: List[float]):
        """
        Adds a functional group to the structure at a specified position, with automatic adjustment to ensure proper
        orientation based on the local surface normal.

        Args:
            selected_position (List[float]): Coordinates [x, y, z] where the group should be added
        """
        # Retrieve the first functional group from the list and remove any anchor atoms
        added_group = self.group_list[0].remove_anchors()
        # Randomly rotate the group to introduce variability in the orientation
        new_atom_coordinates = random_rotate_group_list(added_group.copy())

        # Calculate the normal vector of the surface at the selected position to determine the correct orientation
        normal_vector = self.find_surface_normal_vector(selected_position)
        rotation_matrix = self.rotation_matrix_from_vectors(normal_vector)

        # Rotate the group according to the calculated rotation matrix to align it with the surface normal
        rotated_coordinates = []
        for atom in new_atom_coordinates:
            atom_coords = np.array(atom[1:4], dtype=float)  # ensure that atom_coords has the right datatype
            rotated_coord = np.dot(rotation_matrix, atom_coords)
            rotated_coordinates.append([atom[0], rotated_coord[0], rotated_coord[1], rotated_coord[2], "functional"])

        # Shift the rotated coordinates so that they are correctly positioned at the selected location
        for atom in rotated_coordinates:
            atom[1] += selected_position[0]
            atom[2] += selected_position[1]
            atom[3] += selected_position[2]

        # Create a new DataFrame with the rotated and shifted atom coordinates
        new_atoms_df = pd.DataFrame(
            rotated_coordinates, columns=["Species", "x", "y", "z", "group"]
        )  # Update columns as needed
        # Concatenate this new DataFrame to the main structure DataFrame to update the structure with the new group
        self._structure_df = pd.concat([self._structure_df, new_atoms_df])

    def find_surface_normal_vector(self, position: List[float]) -> npt.NDArray:
        """
        Calculates the surface normal vector at a specified position within the structure.
        This vector is essential for aligning functional groups correctly relative to the structure's surface.

        Args:
            position (List[float]): The coordinates [x, y, z] at which to find the normal vector.

        Returns:
            npt.NDArray: The normalized surface normal vector.
        """

        surface_atoms = []
        # Iterate through each atom in the DataFrame to find atoms near the specified position
        for i, atom in self._structure_df.iterrows():
            # Calculate the Cartesian distance from the current atom to the specified position
            delta_x = atom["x"] - position[0]
            delta_y = atom["y"] - position[1]
            distance = math.sqrt((delta_x) ** 2 + (delta_y) ** 2 + (atom["z"] - position[2]) ** 2)
            # Include atoms that are within a certain threshold distance (e.g., 120% of bond length)
            if (
                distance <= self.bond_length * 1.2
            ):  # ToDo: Warum 1.2? Sollte das wirklich ein harter Wert sein oder sollte das ein Parameter sein?
                # Exclude the position itself to avoid zero vector in calculations
                if distance >= 0.05:  # Ensure it's not the exact same point
                    surface_atoms.append([atom["x"], atom["y"], atom["z"]])

        # Convert list of surface atoms into a NumPy array for vector operations
        surface_atoms = np.array(surface_atoms)
        # Calculate the geometric center (average position) of these surface atoms
        average_position = np.average(surface_atoms, axis=0)

        # this only works on curved surface (selected position and surface atoms are NOT in one plane)
        # on flat surfaces we have to use a different algorithm

        # Calculate the vector from the specified position to the average position
        # This vector points in the direction of the normal to the surface at 'position'
        position = np.array(position)
        normal_vector = average_position - position

        # Normalize the vector to have a magnitude of 1, making it a true normal vector
        normal_magnitude = np.linalg.norm(normal_vector)
        normal_vector /= normal_magnitude

        return normal_vector

    def _build_CNT(self, parameters: Dict[str, Union[str, int, float]], keywords: List[str]) -> pd.DataFrame:
        """
        Builds a carbon nanotube (CNT) structure.

        Args:
            parameters (Dict[str, Union[str, int, float]]): The parameters for the structure, including tube size and
            bond length.
            keywords (List[str]): The keywords defining the type of tube, such as 'armchair' or 'zigzag'.

        Returns:
            pd.DataFrame: The DataFrame containing the structure data.
        """
        # Determine the type of carbon nanotube based on the keywords
        if "armchair" in keywords:
            tube_kind = 1
        elif "zigzag" in keywords:
            tube_kind = 2
        else:
            ddict.printLog("No valid tube kind found in arguments, use 'zigzag' or 'armchair'")
            return None

        tube_size = parameters["tube_size"]  # Number of hexagonal units around the circumference
        tube_length = parameters["tube_length"]  # Length of the tube in the z-direction

        # Load the provided bond length and calculate the distance between two hexagonal vertices
        distance = float(parameters["bond_length"])
        hex_d = distance * math.cos(30 * math.pi / 180) * 2  # Distance between two hexagonal centers in the lattice
        # ToDo: Überprüfen?

        # If the tube is of the armchair configuration
        if tube_kind == 1:
            # Calculate the radius of the tube
            angle_carbon_bond = 360 / (tube_size * 3)
            symmetry_angle = 360 / tube_size
            radius = distance / (2 * math.sin((angle_carbon_bond * math.pi / 180) / 2))

            # Calculate the z distance steps in the tube
            distx = radius - radius * math.cos(angle_carbon_bond / 2 * math.pi / 180)
            disty = 0 - radius * math.sin(angle_carbon_bond / 2 * math.pi / 180)
            zstep = (distance**2 - distx**2 - disty**2) ** 0.5

            # Initialize list for tube positions and angles
            positions_tube = []
            angles = []
            z_max = 0
            counter = 0

            while z_max < tube_length:
                # Calculate the z-coordinate for the current layer of atoms
                # Each layer is placed at intervals determined by zstep, and we consider the distance for the zigzag
                # pattern by multiplying with 2
                z_coordinate = zstep * 2 * counter

                # Loop to create all atoms in the tube
                for i in range(0, tube_size):
                    # Add first position option
                    angle = symmetry_angle * math.pi / 180 * i
                    x = radius * math.cos(angle)
                    y = radius * math.sin(angle)
                    positions_tube.append((x, y, z_coordinate))

                    # Add second position option
                    angle = (symmetry_angle * i + angle_carbon_bond) * math.pi / 180
                    x = radius * math.cos(angle)
                    y = radius * math.sin(angle)
                    positions_tube.append((x, y, z_coordinate))
                    angles.append(angle)

                    # Add third position option
                    angle = (symmetry_angle * i + angle_carbon_bond * 3 / 2) * math.pi / 180
                    x = radius * math.cos(angle)
                    y = radius * math.sin(angle)
                    z = zstep + z_coordinate
                    positions_tube.append((x, y, z))
                    angles.append(angle)

                    # Add fourth position option
                    angle = (symmetry_angle * i + angle_carbon_bond * 5 / 2) * math.pi / 180
                    x = radius * math.cos(angle)
                    y = radius * math.sin(angle)
                    z = zstep + z_coordinate
                    positions_tube.append((x, y, z))
                    angles.append(angle)

                z_max = z_coordinate + zstep  # Update maximum z-coordinate reached
                counter += 1  # Increment the counter

        # If the tube is of the zigzag configuration
        if tube_kind == 2:
            symmetry_angle = 360 / tube_size
            # Calculate the radius of the tube
            radius = hex_d / (2 * math.sin((symmetry_angle * math.pi / 180) / 2))

            # Calculate the z distances in the tube
            distx = radius - radius * math.cos(symmetry_angle / 2 * math.pi / 180)
            disty = 0 - radius * math.sin(symmetry_angle / 2 * math.pi / 180)
            zstep = (distance**2 - distx**2 - disty**2) ** 0.5

            # Initialize list for tube positions and angles
            positions_tube = []
            angles = []
            z_max = 0
            counter = 0

            while z_max < tube_length:
                # Calculate the z-coordinate for the current layer of atoms
                # This combines the vertical step zstep and an additional distance twice the bond length
                # The counter ensures each layer is positioned correctly along the z-axis
                z_coordinate = (2 * zstep + distance * 2) * counter

                # Loop to create the atoms in the tube
                for i in range(0, tube_size):
                    # Add first position option
                    angle = symmetry_angle * math.pi / 180 * i
                    x = radius * math.cos(angle)
                    y = radius * math.sin(angle)
                    positions_tube.append((x, y, z_coordinate))

                    # Add second position option
                    angle = (symmetry_angle * i + symmetry_angle / 2) * math.pi / 180
                    x = radius * math.cos(angle)
                    y = radius * math.sin(angle)
                    z = zstep + z_coordinate
                    positions_tube.append((x, y, z))

                    # Add third position option
                    angle = (symmetry_angle * i + symmetry_angle / 2) * math.pi / 180
                    x = radius * math.cos(angle)
                    y = radius * math.sin(angle)
                    z = zstep + distance + z_coordinate
                    positions_tube.append((x, y, z))

                    # Add fourth position option
                    angle = symmetry_angle * math.pi / 180 * i
                    x = radius * math.cos(angle)
                    y = radius * math.sin(angle)
                    z = 2 * zstep + distance + z_coordinate
                    positions_tube.append((x, y, z))

                z_max = z_coordinate + zstep  # Update maximum z-coordinate reached
                counter += 1  # Increment counter

        # Store the radius of the tube
        self.radius = radius

        # Create DataFrame from the list of positions
        self._structure_df = pd.DataFrame(positions_tube)
        self._structure_df.insert(0, "Species", "C")
        self._structure_df.insert(4, "group", "Structure")
        self._structure_df.insert(5, "Molecule", 1)
        self._structure_df.columns.values[1] = "x"
        self._structure_df.columns.values[2] = "y"
        self._structure_df.columns.values[3] = "z"
        self._structure_df.insert(6, "Label", "X")
        counter = 1
        for i, atom in self._structure_df.iterrows():
            self._structure_df.at[i, "Label"] = f"C{counter}"
            counter = counter + 1
        return self._structure_df

    def _stack_CNTs(
        self, parameters: Dict[str, Union[str, int, float]], keywords: List[str]
    ) -> pd.DataFrame:  # ToDo: Warum ist das in Structure1d? Ist doch jetzt nicht mehr eindimensional?
        """
        Stacks carbon nanotubes in the structure.

        Args:
            parameters (Dict[str, Union[str, int, float]]): The parameters for the stacking.
            keywords (List[str]): The keywords for the stacking.

        Returns:
            pd.DataFrame: The DataFrame containing the stacked structure.
        """
        # Define the distance between the tubes
        distance_tubes = parameters["tube_distance"]
        radius_distance = self.radius + distance_tubes / 2

        # Retrieve the current positions of the tube atoms
        positions_tube = self._structure_df

        # Get the maximum molecule number in the existing structure
        max_molecule = positions_tube.iloc[:, 4].max()

        # Now the position of the second tube is calculated
        # The tube is shifted by the radius_distance in x direction,and by sqrt(3)*radius_distance in y direction.
        tube_two = positions_tube.copy()
        tube_two.iloc[:, 1] = tube_two.iloc[:, 1] + radius_distance
        tube_two.iloc[:, 2] = tube_two.iloc[:, 2] + radius_distance * math.sqrt(3)
        tube_two.iloc[:, 4] = tube_two.iloc[:, 4] + max_molecule

        # Concatenate the two tubes
        unit_cell = pd.DataFrame()
        unit_cell = pd.concat([positions_tube, tube_two], ignore_index=True)

        # Now build the periodic unit cell from the given atoms.
        # The dimensions of the unit cell are 2*radius_distance in x direction and 2*radius_distance*math.sqrt(3) in y
        # direction.
        unit_cell_x = float(2 * radius_distance)
        unit_cell_y = float(2 * radius_distance * math.sqrt(3))

        # Check all atoms in the positions_tube dataframe and shift all atoms that are outside the unit cell to the
        # inside of the unit cell.
        for i in range(0, len(unit_cell)):
            if unit_cell.iloc[i, 1] > unit_cell_x:
                unit_cell.iloc[i, 1] = unit_cell.iloc[i, 1] - unit_cell_x
            if unit_cell.iloc[i, 1] < 0:
                unit_cell.iloc[i, 1] = unit_cell.iloc[i, 1] + unit_cell_x
            if unit_cell.iloc[i, 2] > unit_cell_y:
                unit_cell.iloc[i, 2] = unit_cell.iloc[i, 2] - unit_cell_y
            if unit_cell.iloc[i, 2] < 0:
                unit_cell.iloc[i, 2] = unit_cell.iloc[i, 2] + unit_cell_y

        # Now multiply the unit cell in x and y direction to fill the whole simulation box.
        multiplicity_x = parameters["multiplicity"][0]
        multiplicity_y = parameters["multiplicity"][1]

        # The positions of the atoms in the unit cell are copied and shifted in x and y direction.
        super_cell = unit_cell.copy()
        supercell_x = unit_cell.copy()
        max_molecule = unit_cell.iloc[:, 4].max() if not unit_cell.empty else 0
        for i in range(1, multiplicity_x):
            supercell_x = unit_cell.copy()
            supercell_x.iloc[:, 1] = unit_cell.iloc[:, 1] + i * unit_cell_x
            supercell_x.iloc[:, 4] = supercell_x.iloc[:, 4] + max_molecule * i
            super_cell = pd.concat([super_cell, supercell_x], ignore_index=True)

        supercell_after_x = super_cell.copy()
        max_molecule = super_cell.iloc[:, 4].max() if not super_cell.empty else 0
        for i in range(1, multiplicity_y):
            supercell_y = supercell_after_x.copy()
            supercell_y.iloc[:, 2] = supercell_y.iloc[:, 2] + i * unit_cell_y
            supercell_y.iloc[:, 4] = supercell_y.iloc[:, 4] + max_molecule * i
            super_cell = pd.concat([super_cell, supercell_y], ignore_index=True)

        # check for duplicates in the supercell. If there have been any, give a warning, then drop them.
        duplicates = super_cell.duplicated(subset=["x", "y", "z"], keep="first")
        if duplicates.any():
            ddict.printLog("[WARNING] Duplicates found in the supercell. Dropping them.")
        super_cell = super_cell.drop_duplicates(subset=["x", "y", "z"], keep="first")

        # Now the supercell is written to positions_tube.
        positions_tube = super_cell.copy()
        positions_tube = pd.DataFrame(positions_tube)

        # now within the dataframe, the molcules need to be sorted by the following criteria: Species number -> Molecule
        # number -> Label.
        # The according column names are 'Species', 'Molecule' and 'Label'. The first two are floats, the last one is a
        # string.
        # In case of the label the sorting should be done like C1, C2, C3, ... C10, C11, ... C100, C101, ... C1000,
        # C1001, ...
        # Extract the numerical part from the 'Label' column and convert it to integer
        positions_tube["Label_num"] = positions_tube["Label"].str.extract(r"(\d+)").astype(int)

        # Sort the dataframe by 'Element', 'Molecule', and 'Label_num'
        positions_tube = positions_tube.sort_values(by=["Species", "Molecule", "Label_num"])

        # Drop the 'Label_num' column as it's no longer needed
        positions_tube = positions_tube.drop(columns=["Label_num"])

        # # Finally compute the PBC size of the simulation box. It is given by the multiplicity in x and y direction
        # times the unit cell size.
        # pbc_size_x = multiplicity_x * unit_cell_x
        # pbc_size_y = multiplicity_y * unit_cell_y

        self._structure_df = positions_tube

        return self._structure_df


class Structure2d(Structure):
    """
    Represents a 2-dimensional molecular structure.

    Attributes:
        bond_distance (float): The bond distance in the structure.
        sheet_size (Tuple[float, float]): The size of the sheet.
        _unit_cell_vectors (List[float]): The unit cell vectors.
        _positions_unitcell (List[Tuple[float, float, float]]): The atomic positions in the unit cell.
        _number_of_unit_cells (Tuple[int, int]): The number of unit cells in x and y directions.
    """

    def __init__(
        self, bond_distance: float, sheet_size: Tuple[float, float]
    ):  # ToDo: Der superclass Aufruf hat gefehlt. Habe ich hier mal hinzu gefügt
        """
        Initializes a Structure2D instance.

        Args:
            bond_distance (float): The bond distance in the structure.
            sheet_size (Tuple[float, float]): The size of the sheet.
        """
        super().__init__()
        self.bond_distance = bond_distance
        self.sheet_size = sheet_size
        self._create_sheet()

    def functionalize_sheet(self, parameters: Dict[str, Union[str, int, float]]) -> None:
        """
        Functionalizes the sheet with functional groups.

        Args:
            parameters (Dict[str, Union[str, int, float]]): The parameters for the functionalization.
        """
        self._initialize_functional_groups(parameters)
        self.__add_groups_to_sheet()

    # def available_positions(self):
    #     available_positions = []
    #     for i, position in self._structure_df.iterrows():
    #         available_positions.append([position[1], position[2], position[3]])
    #     return available_positions

    def available_positions(
        self,
    ) -> List[Tuple[float, float, float]]:  # ToDo: Ist Methode für Anbringung der Gruppen? Sicher dann in Structure2d?
        """
        Gets the available positions on the sheet.

        Returns:
            List[Tuple[float, float, float]]: The list of available positions.
        """
        return [
            (position[1], position[2], position[3]) for _, position in self._structure_df.iterrows()
        ]  # ToDo: Sollte dasselbe tun wie Methode oben drüber. Allerdings ist Struktur insgesamt nicht sehr
        # übersichtlich. Evtl. in Datenklasse auslagern?

    def add(
        self, parameters: Dict[str, Union[str, int, float]]
    ):  # ToDo: Das müssten alles Methoden sein, die in der abstrakten Klasse definiert sind und hier implementiert
        # werden; Methode evtl. auch besser wo anders hin verlagern?
        """
        Adds a functional group to the structure.

        Args:
            parameters (Dict[str, Union[str, int, float]]): The parameters for the addition.
        """
        parameters["group_count"] = 1
        self._initialize_functional_groups(parameters)
        position = [
            self._structure_df.iloc[parameters["position"], 1],
            self._structure_df.iloc[parameters["position"], 2],
        ]
        self._add_group_on_position(position)

    # PRIVATE
    def _create_sheet(self):
        """
        Creates the sheet by defining the unit cell and building the sheet.
        """
        self._define_unit_cell()
        self._build_sheet()

    def _define_unit_cell(self):
        """
        Defines the unit cell of the sheet.
        """
        # Calculate the distance between rows of atoms in x-y-directions
        C_C_y_distance = self.bond_distance * math.cos(30 * math.pi / 180)
        C_C_x_distance = self.bond_distance * math.sin(30 * math.pi / 180)

        # Define the unit cell vectors:
        # - x-direction vector length: sum of twice the bond distance and twice the x-distance between rows
        # - y-direction vector length: twice the y-distance between rows
        self._unit_cell_vectors = [
            2 * self.bond_distance + 2 * C_C_x_distance,  # Length of unit vector in x-direction
            2 * C_C_y_distance,  # Length of unit vector in y-direction
        ]

        # Create a list for the atomic positions inside the unit cell.
        self._positions_unitcell: List[Tuple] = [
            (0, 0, 0),  # Position of the first atom
            (C_C_x_distance, C_C_y_distance, 0),  # Position of the second atom
            (C_C_x_distance + self.bond_distance, C_C_y_distance, 0),  # Position of the third atom
            (2 * C_C_x_distance + self.bond_distance, 0, 0),  # Position of the fourth atom
        ]

        # Calculate the number of unit cells that fit into the sheet in the x and y directions
        self._number_of_unit_cells = [
            math.floor(self.sheet_size[0] / self._unit_cell_vectors[0]),  # Number of unit cells in x-direction
            math.floor(self.sheet_size[1] / self._unit_cell_vectors[1]),  # Number of unit cells in y-direction
        ]

        # Adjust the sheet size to fit an integer number of unit cells
        self.sheet_size = [
            self._unit_cell_vectors[0] * self._number_of_unit_cells[0],  # Adjusted size in x-direction
            self._unit_cell_vectors[1] * self._number_of_unit_cells[1],  # Adjusted size in y-direction
        ]

    def __add_groups_to_sheet(self):  # ToDo: wirklich hier in Klasse?
        """
        Adds functional groups to the sheet.
        """
        # Initialize a list to store the functional groups without anchor atoms
        added_groups = []
        for group in self.group_list:
            added_groups.append(group.remove_anchors())

        # Get the list of available positions on the sheet
        position_list = self.available_positions()
        new_atoms = []

        # Seed the random number generator to ensure reproducibility
        random.seed(a=None, version=2)

        # Iterate through each group in the group list
        for i in range(len(self.group_list)):
            group = self.group_list[i]
            number_of_added_groups = 0

            # Try to add the specified number of functional groups to the sheet
            for j in range(group.group_count):
                if position_list:
                    # Add a group to a random position on the sheet
                    new_atoms += self.__add_group_on_random_position(
                        added_groups[i], group.exclusion_radius, position_list
                    )
                else:
                    # If there are no available positions, print a warning
                    print("Sheet size is not large enough!")
                    print(f"Generated sheet is missing {group.group_count - number_of_added_groups} groups")
                    break
                number_of_added_groups += 1

        # Convert the list of new atoms to a DataFrame
        new_atoms_df = pd.DataFrame(new_atoms)
        # Add a column indicating that these atoms are functional groups
        new_atoms_df["group"] = pd.Series(["functional" for x in range(len(new_atoms_df.index))])
        new_atoms_df.columns = ["Species", "x", "y", "z", "group"]

        # Finally, concatenate the new atoms DataFrame to the existing structure DataFrame
        self._structure_df = pd.concat([self._structure_df, new_atoms_df])

    def __add_group_on_random_position(
        self,
        added_group: List[Tuple[str, float, float, float]],
        exclusion_radius: float,
        position_list: List[List[float]],
    ) -> List[Tuple[str, float, float, float]]:
        """
        Adds a functional group to a random position on the sheet.

        Args:
            added_group (List[Tuple[str, float, float, float]]): The functional group to add.
            exclusion_radius (float): The exclusion radius for the group.
            position_list (List[List[float]]): The list of available positions.

        Returns:
            List[Tuple[str, float, float, float]]: The coordinates of the added group.
            # ToDo: Sollte sowas vielleicht auch in Datenklasse ausgelagert werden?
        """
        # select a position to add the group on
        selected_position = position_list[random.randint(0, len(position_list) - 1)]
        # randomly rotate the group
        new_atom_coordinates = random_rotate_group_list(added_group.copy())
        # shift the coordinates to the selected position
        for atom in new_atom_coordinates:
            atom[1] += selected_position[0]
            atom[2] += selected_position[1]
        # Remove positions blocked by the new group
        position_list.remove(selected_position)
        self.__remove_adjacent_positions(position_list, selected_position, exclusion_radius)
        return new_atom_coordinates

    def _add_group_on_position(self, selected_position: List[float]):
        """
        Adds a functional group to the sheet at a specific position.

        Args:
            selected_position (List[float]): The position to add the group to.
        """
        added_group = self.group_list[0].remove_anchors()
        # randomly rotate the group
        new_atom_coordinates = random_rotate_group_list(added_group.copy())
        # shift the coordinates to the selected position
        for atom in new_atom_coordinates:
            atom[1] += selected_position[0]
            atom[2] += selected_position[1]
        new_atoms_df = pd.DataFrame(new_atom_coordinates, columns=["Species", "x", "y", "z"])
        new_atoms_df["group"] = pd.Series(["functional" for x in range(len(new_atoms_df.index))])
        self._structure_df = pd.concat([self._structure_df, new_atoms_df])

    def __remove_adjacent_positions(
        self,
        position_list: List[List[float]],
        selected_position: List[float],
        cutoff_distance: float,
    ):
        """
        Removes positions adjacent to a given position from the list of available positions.

        Args:
            position_list (List[List[float]]): The list of available positions.
            selected_position (List[float]): The position to remove adjacent positions for.
            cutoff_distance (float): The cutoff distance for adjacency.
        """
        # Initialize a list to store positions that are found to be adjacent
        adjacent_positions = []

        # Iterate through each position in the list of available positions
        for position in position_list:
            # Check if the current position is adjacent to the selected position
            if positions_are_adjacent(position, selected_position, cutoff_distance, self.sheet_size):
                # If it is adjacent, add it to the list of adjacent positions
                adjacent_positions.append(position)

        # Remove all adjacent positions from the list of available positions
        for adjacent_position in adjacent_positions:
            position_list.remove(adjacent_position)


class Pore(
    Structure
):  # ToDo: Erbt das wirklich nur von Structure und nicht von Structure2d wie Graphene? -> Pore wird aus Graphenwand
    # und CNT aufgebaut
    """
    Represents a pore structure.

    Attributes:
        bond_length (float): The bond length of the structure.
        sheet_size (List[float]): The size of the sheet.
        pore_center (List[float]): The center of the pore.
        cnt_radius (List[float]): The radius of the carbon nanotube.
    """

    # # CONSTRUCTOR
    # def __init__(self, parameters, keywords):
    #     self._build_pore(parameters, keywords)

    def __init__(self, parameters: Dict[str, Union[str, int, float]], keywords: List[str]):
        """
        Initializes a Pore instance.

        Args:
            parameters (Dict[str, Union[str, int, float]]): The parameters for the pore.
            keywords (List[str]): The keywords for the pore.
        """
        super().__init__()  # ToDo: Superklasse wurde hier mit aufgerufen im Gegensatz zu obigem Code
        self._build_pore(parameters, keywords)

    # INTERFACE
    def add(self, parameters: Dict[str, Union[str, int, float]]):
        """
        Adds a functional group to the pore.

        Args:
            parameters (Dict[str, Union[str, int, float]]): The parameters for the addition.
        """
        # Set the group_count to 1 for the addition of one functional group
        parameters["group_count"] = 1
        # Initialize functional groups based on the given parameters
        self._initialize_functional_groups(parameters)
        # Get the coordinates of the selected position from the structure DataFrame
        selected_position = [
            self._structure_df.iloc[parameters["position"], 1],
            self._structure_df.iloc[parameters["position"], 2],
            self._structure_df.iloc[parameters["position"], 3],
        ]
        # Add the functional group to the selected position
        self._add_group_on_position(selected_position)

    # PRIVATE
    def _build_pore(self, parameters: Dict[str, Union[str, int, float]], keywords: List[str]) -> None:
        """
        Builds a pore structure.

        Args:
            parameters (Dict[str, Union[str, int, float]]): The parameters for the pore.
            keywords (List[str]): The keywords for the pore.
        """
        # Set the bond length and sheet size from the parameters
        self.bond_length = parameters["bond_length"]
        self.sheet_size = parameters["sheet_size"]

        # Determine the type of pore (closed or open) based on the keywords
        if "closed" in keywords:  # ToDo: Evtl. besser über Enums lösen
            pore_kind = 2  # Closed pore
        else:
            pore_kind = 1  # Open pore

        # Generate substructures
        # Create a graphene wall
        wall = Graphene(parameters["bond_length"], parameters["sheet_size"])
        # Initialize a second wall for the open/closed pore
        wall2: Optional[Graphene] = None
        # Create a carbon nanotube (CNT)
        cnt = Structure1d(parameters, keywords)

        # If the user wants a closed pore, we copy the wall now without the hole
        if pore_kind == 2:
            wall2 = copy.deepcopy(wall)

        #  Store the initial structure of the wall
        self._structure_df = wall._structure_df

        # Create a hole in the wall
        # The size of the hole is based on the radius of the CNT plus a margin
        pore_position = wall.make_pores(cnt.radius + 1.0)

        # Shift the CNT position to align with the hole in the wall
        cnt._structure_df["x"] += pore_position[1]
        cnt._structure_df["y"] += pore_position[2]

        # Set the center and radius of the pore
        self.pore_center = [pore_position[1], pore_position[2]]
        self.cnt_radius = [cnt.radius]

        # If the user wants an open pore, we copy it now with the hole
        if pore_kind == 1:
            wall2 = copy.deepcopy(wall)

        # 'Clip off' the ends of the CNT for a smoother transition
        cnt._structure_df = cnt._structure_df[cnt._structure_df["z"] > 0.2]
        max_z = cnt._structure_df["z"].max()
        cnt._structure_df = cnt._structure_df[cnt._structure_df["z"] < (max_z - 0.2)]

        # Move the second wall to the end of the CNT
        wall2._structure_df["z"] += max_z

        # Combine the wall, the CNT, and the second wall to form the complete pore structure
        self._structure_df = pd.concat([wall._structure_df, cnt._structure_df, wall2._structure_df])

        # Correct the sheet_size to reflect the actual size of the sheet
        max_x = self._structure_df["x"].max()  # Determine the maximum x-value
        delta_x = abs(max_x - self.sheet_size[0])  # Minimum image distance in x-direction

        # Adjust the sheet size to ensure the minimum image distance is equal to the bond length
        self.sheet_size[0] -= delta_x - self.bond_length

        # Do the same adjustment in the y direction. The only difference is that the distance should not be equal to one
        # bond length
        max_y = self._structure_df["y"].max()
        delta_y = abs(max_y - self.sheet_size[1])
        self.sheet_size[1] -= delta_y - self.bond_length * math.cos(30 * math.pi / 180)

    def _add_group_on_position(self, selected_position: List[float]) -> None:
        """
        Adds a functional group to the pore at a specific position.

        Args:
            selected_position (List[float]): The position to add the group to.
        """
        # Retrieve and remove anchor atoms from the first functional group in the list
        added_group = self.group_list[0].remove_anchors()

        # Give the group a random orientation first
        new_atom_coordinates = random_rotate_group_list(added_group.copy())

        # Calculate the distance from the selected positioin to the center of the pore
        distance_to_center = math.sqrt(
            (selected_position[0] - self.pore_center[0]) ** 2 + (selected_position[1] - self.pore_center[1]) ** 2
        )

        # Determine if the position is inside the pore or on a wall
        if distance_to_center < self.cnt_radius[0] + 0.4:
            # If inside the pore, add the group inside the pore
            self.add_group_in_pore(new_atom_coordinates, selected_position)
        else:
            # If on the wall, add the group on the wall
            self.add_group_on_wall(new_atom_coordinates, selected_position)

    def add_group_on_wall(self, new_atom_coordinates, selected_position):
        """
        Adds a functional group to the wall of the pore.

        Args:
            new_atom_coordinates (List[Tuple[str, float, float, float]]): The coordinates of the functional group.
            selected_position (List[float]): The position to add the group to.
        """

        # Find out which wall the selected position belongs to by comparing its z-coordinate
        structure_center_z = self._structure_df["z"].max() / 2.0

        # if the position is on the wall with z~0.0, we have to invert the group
        # otherwise we do not change anything
        if selected_position[2] < structure_center_z:
            for atom in new_atom_coordinates:
                atom[1] *= -1.0
                atom[2] *= -1.0
                atom[3] *= -1.0

        # move the group to the position
        for atom in new_atom_coordinates:
            atom[1] += selected_position[0]  # Shift x-coordinate
            atom[2] += selected_position[1]  # Shift y-coordinate
            atom[3] += selected_position[2]  # Shift z-coordinate

        # Create a DataFrame for the new atoms and specify they are functional groups
        new_atoms_df = pd.DataFrame(new_atom_coordinates, columns=["Species", "x", "y", "z"])
        new_atoms_df["group"] = pd.Series(["functional" for x in range(len(new_atoms_df.index))])

        # Concatenate the new atoms with the existing structure
        self._structure_df = pd.concat([self._structure_df, new_atoms_df])

    def add_group_in_pore(
        self, new_atom_coordinates: List[Tuple[str, float, float, float]], selected_position: List[float]
    ) -> None:
        """
        Adds a functional group inside the pore.

        Args:
            new_atom_coordinates (List[Tuple[str, float, float, float]]): The coordinates of the functional group.
            selected_position (List[float]): The position to add the group to.
        """
        # find the right orientation relative to local surface
        normal_vector = self.find_surface_normal_vector(selected_position)
        rotation_matrix = self.rotation_matrix_from_vectors(normal_vector)

        # finally rotate the group
        rotated_coordinates = []
        for atom in new_atom_coordinates:
            atom_coords = np.array(atom[1:4], dtype=float)  # ensure that atom_coords has the right datatype
            rotated_coord = np.dot(rotation_matrix, atom_coords)
            rotated_coordinates.append([atom[0], rotated_coord[0], rotated_coord[1], rotated_coord[2], "functional"])

        # shift the coordinates to the selected position
        for atom in rotated_coordinates:
            atom[1] += selected_position[0]  # Shift x-coordinate
            atom[2] += selected_position[1]  # Shift y-coordinate
            atom[3] += selected_position[2]  # Shift z-coordinate

        # Create a DataFrame for the rotated and shifted atoms and mark them as functional groups
        new_atoms_df = pd.DataFrame(
            rotated_coordinates, columns=["Species", "x", "y", "z", "group"]
        )  # Update columns as needed

        # Concatenate the new atoms with the existing structure
        self._structure_df = pd.concat([self._structure_df, new_atoms_df])

    def find_surface_normal_vector(self, position: List[float]) -> npt.NDArray:
        """
        Finds the surface normal vector at a given position.

        Args:
            position (List[float]): The position to find the normal vector for.

        Returns:
            npt.NDArray: The normal vector.
        """
        # Calculate the vector from the pore center to the given position
        normal_vector = np.array(
            [self.pore_center[0] - position[0], self.pore_center[1] - position[1], 0.0]  # x-component  # y-component
        )  # z-component is 0 because we are assuming a 2D surface in xy-plane

        # Compute the magnitude of the normal vector
        normal_magnitude = np.linalg.norm(normal_vector)

        # Normalize the vector to make it a unit vector
        normal_vector /= normal_magnitude

        return normal_vector


class Graphene(Structure2d):
    """
    Represents a graphene sheet structure.
    """

    # INTERFACE
    def make_pores(self, pore_size: float) -> pd.Series:
        """
        Creates circular pores in the graphene sheet.

        Args:
            pore_size (float): The size of the pore.

        Returns:
            pd.Series: The position of the center of the pore.
        """
        return self._make_circular_pore(pore_size)

    # PRIVATE
    def _make_circular_pore(self, pore_size: float) -> pd.Series:
        """
        Creates a circular pore in the graphene sheet at a specified site.

        Args:
            pore_size (float): The size of the pore.

        Returns:
            pd.Series: The position of the center of the pore.
        """

        # Make a copy of the DataFrame to avoid mutating the original during processing
        atoms_df = self._structure_df.copy()

        # Select the atom closest to the center of the sheet as the position for the pore
        selected_position = center_position(self.sheet_size, atoms_df)  # ToDo: Warum zwangsläufig die Mitte?

        # Prepare a list to keep track of atoms that should be removed
        atoms_to_remove = []

        # Iterate over each atom in the DataFrame
        for i, atom in self._structure_df.iterrows():
            # Determine the position of the current atom
            atom_position = [atom[1], atom[2]]

            # Calculate the minimum image distance from the selected center to the current atom
            if (
                minimum_image_distance(atom_position, [selected_position[1], selected_position[2]], self.sheet_size)
                <= pore_size
            ):
                # If the atom is within the pore size, add it to the removal list
                atoms_to_remove.append(i)
        # Remove the atoms that are marked for removal
        self._structure_df.drop(atoms_to_remove, inplace=True)

        # Return the position of the selected center of the pore
        return selected_position

    def _build_sheet(self) -> None:  # ToDo: Müsste abstrakte Methode sein
        """
        Builds the graphene sheet from multiple unit cells.
        """
        # Set the z-coordinates of all atoms to 0 (planar sheet)
        Z = [0.0] * 4  # All Z-coordinates are 0

        # Initialize an empty list to store the coordinates of all atoms in the sheet
        coords: List = []

        # Build the graphene sheet from multiple unit cells
        for i in range(self._number_of_unit_cells[0]):
            # Calculate the x-coordinates for the four atoms in a single unit cell
            X = [
                self._positions_unitcell[0][0] + i * self._unit_cell_vectors[0],
                self._positions_unitcell[1][0] + i * self._unit_cell_vectors[0],
                self._positions_unitcell[2][0] + i * self._unit_cell_vectors[0],
                self._positions_unitcell[3][0] + i * self._unit_cell_vectors[0],
            ]
            for j in range(self._number_of_unit_cells[1]):
                # Calculate the y-coordinates for the four atoms in a single unit cell
                Y = [
                    self._positions_unitcell[0][1] + j * self._unit_cell_vectors[1],
                    self._positions_unitcell[1][1] + j * self._unit_cell_vectors[1],
                    self._positions_unitcell[2][1] + j * self._unit_cell_vectors[1],
                    self._positions_unitcell[3][1] + j * self._unit_cell_vectors[1],
                ]
                # Add the coordinates for the four atoms in the current unit cell to the list
                coords.append(["C", X[0], Y[0], Z[0], "Structure"])
                coords.append(["C", X[1], Y[1], Z[1], "Structure"])
                coords.append(["C", X[2], Y[2], Z[2], "Structure"])
                coords.append(["C", X[3], Y[3], Z[3], "Structure"])
        # Create a DataFrame from the list of coordinates
        self._structure_df = pd.DataFrame(coords)
        # Set appropriate column names for the DataFrame
        self._structure_df.columns = ["Species", "x", "y", "z", "group"]


class Boronnitride(Structure2d):
    """
    Represents a boron nitride sheet structure.
    """

    # INTERFACE
    def make_pores(self, pore_size: float) -> None:
        """
        Creates triangular pores in the boron nitride sheet.

        Args:
            pore_size (float): The size of the pore.
        """
        self.__make_triangular_pore(pore_size)

    # PRIVATE
    def __make_triangular_pore(self, pore_size: float) -> None:  # ToDo: Warum triangular? Für hexagonales Bornitrid?
        """
        Creates a triangular pore in the boron nitride sheet.

        Args:
            pore_size (float): The size of the pore.
        """
        # Select a starting position based on nitrogen atoms, which typically form one part of the hBN lattice
        atoms_df = self._structure_df.copy()
        dummy_df = atoms_df[atoms_df[0] == "N"]  # Select nitrogen atoms
        selected_position = center_position(self.sheet_size, dummy_df)
        # find nearest atom in x-direction to get orientation of the triangle
        dummy_df = atoms_df[atoms_df["Species"] == "B"]
        dummy_df = dummy_df[dummy_df["y"] > (selected_position[2] - 0.1)]
        dummy_df = dummy_df[dummy_df["y"] < (selected_position[2] + 0.1)]

        # Identify the nearest boron atom to define the orientation of the triangular pore
        dummy_df = atoms_df[atoms_df[0] == "B"]  # Select boron atoms
        dummy_df = dummy_df[dummy_df[2] > (selected_position[2] - 0.1)]  # Narrow down to those close in the y-axis
        # ToDo: dummy_df[2] ist nicht sehr lesbar -> dummy_df['y']?
        dummy_df = dummy_df[dummy_df[2] < (selected_position[2] + 0.1)]
        nearest_atom_df = dummy_df
        nearest_atom_df[1] = nearest_atom_df[1].apply(
            lambda x: abs(x - selected_position[1])
        )  # ToDo: nearest_atom_df['x']?
        nearest_atom = atoms_df.iloc[nearest_atom_df[1].idxmin()]
        nearest_atom_df["x"] = nearest_atom_df["x"].apply(lambda x: abs(x - selected_position[1]))
        nearest_atom = atoms_df.iloc[nearest_atom_df["x"].idxmin()]

        # Calculate the vector for one side of the triangle based on the nearest atom
        orientation_vector = [
            nearest_atom[1] - selected_position[1],  # ToDo: nearest_atom['x']?
            nearest_atom[2] - selected_position[2],  # ToDo: nearest_atom['y']?
        ]
        magnitude = math.sqrt((orientation_vector[0]) ** 2 + (orientation_vector[1]) ** 2) / pore_size
        # orientation_vector[0] /= magnitude
        # orientation_vector[1] /= magnitude
        # orientation_vector[0] *= pore_size
        # orientation_vector[1] *= pore_size
        orientation_vector = [
            component / magnitude * pore_size for component in orientation_vector
        ]  # ToDo: So besser lesbar als oben?

        # Determine the triangle tips based on the starting position and calculated orientation vector
        # triangle_position = [selected_position[1], selected_position[2]]
        tip1 = [selected_position[1] + orientation_vector[0], selected_position[2] + orientation_vector[1]]
        # tip2, tip3 = find_triangle_tips(triangle_position, tip1)
        tip2, tip3 = find_triangle_tips(selected_position.values, tip1)

        # Remove atoms inside the defined triangle
        atoms_to_remove = []
        for i, atom in self._structure_df.iterrows():
            point = (atom[1], atom[2])  # Assuming columns 1 & 2 are x and y coordinates
            if is_point_inside_triangle(tip1, tip2, tip3, point):
                atoms_to_remove.append(i)

        # Update the DataFrame by removing atoms inside the triangle
        self._structure_df.drop(atoms_to_remove, inplace=True)

    def _build_sheet(self):
        """
        Builds the boron nitride sheet from multiple unit cells.

        Boron nitride sheets are composed of alternating boron (B) and nitrogen (N) atoms arranged in a hexagonal
        lattice. This method constructs such a sheet by replicating the defined unit cell across specified dimensions.
        """
        # All Z-coordinates are set to 0 to represent the planar nature of the sheet
        Z = [0.0] * 4  # There are four atoms in each unit cell

        coords = []  # This will hold the coordinates for all atoms in the sheet

        # Loop through each unit cell in the x-direction
        for i in range(self._number_of_unit_cells[0]):
            # Calculate the x-coordinates for the four atoms in one unit cell
            X = [
                self._positions_unitcell[0][0] + i * self._unit_cell_vectors[0],
                self._positions_unitcell[1][0] + i * self._unit_cell_vectors[0],
                self._positions_unitcell[2][0] + i * self._unit_cell_vectors[0],
                self._positions_unitcell[3][0] + i * self._unit_cell_vectors[0],
            ]

            # Loop through each unit cell in the y-direction
            for j in range(self._number_of_unit_cells[1]):
                Y = [
                    self._positions_unitcell[0][1] + j * self._unit_cell_vectors[1],
                    self._positions_unitcell[1][1] + j * self._unit_cell_vectors[1],
                    self._positions_unitcell[2][1] + j * self._unit_cell_vectors[1],
                    self._positions_unitcell[3][1] + j * self._unit_cell_vectors[1],
                ]

                # Append the coordinates for each atom to the coords list
                # Alternating between Boron and Nitrogen based on their positions in the unit cell
                coords.append(["B", X[0], Y[0], Z[0], "Structure"])
                coords.append(["N", X[1], Y[1], Z[1], "Structure"])
                coords.append(["B", X[2], Y[2], Z[2], "Structure"])
                coords.append(["N", X[3], Y[3], Z[3], "Structure"])

        # Create a DataFrame from the list of coordinates
        # This DataFrame represents the complete boron nitride sheet
        self._structure_df = pd.DataFrame(
            coords, columns=["Species", "x", "y", "z", "group"]
        )  # ToDo: Habe hier mal die Spaltennamen hinzugefügt, sodass man verständlicher auf die Elemente zugreifen kann


# def center_position(sheet_size, atoms_df):
#     # This function returns the coordinates of the atom that
#     # is closest to the sheet center
#     center_point = [
#         sheet_size[0] / 2,
#         sheet_size[1] / 2
#     ]
#     distance_to_center_point = []
#     for i, atom in atoms_df.iterrows():
#         distance_to_center_point.append(minimum_image_distance(center_point, [atom[1], atom[2]], sheet_size))
#     center_position_index = distance_to_center_point.index(min(distance_to_center_point))
#     center_position = atoms_df.iloc[int(center_position_index)]
#     return center_position

# ToDo: In einer sauberen Codebasis sollten die folgenden Funktionen in eine utils.py ausgelagert werden


def center_position(
    sheet_size: Tuple[float, float], atoms_df: pd.DataFrame
) -> pd.Series:  # ToDo: Sollte das gleiche tun wie Methode oben drüber, aber evtl. etwas lesbarer
    """
    Returns the coordinates of the atom that is closest to the sheet center.

    Args:
        sheet_size (Tuple[float, float]): The size of the sheet.
        atoms_df (pd.DataFrame): The DataFrame containing the atom coordinates.

    Returns:
        pd.Series: The coordinates of the atom closest to the center.
    """
    # Calculate the center point of the sheet
    center_point = [sheet_size[0] / 2, sheet_size[1] / 2]

    # Compute the distance of each atom to the center point using minimum image distance
    distance_to_center_point = [
        minimum_image_distance(center_point, [atom[1], atom[2]], sheet_size) for _, atom in atoms_df.iterrows()
    ]

    # Find the index of the atom with the minimum distance to the center point
    min_distance_index = distance_to_center_point.index(min(distance_to_center_point))

    # Return the coordinates of the atom closest to the center
    return atoms_df.iloc[int(min_distance_index)]


def rotate_vector(vec: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotates a vector by a given angle.

    Args:
        vec (np.ndarray): The vector to rotate.
        angle (float): The angle by which to rotate the vector.

    Returns:
        np.ndarray: The rotated vector.
    """
    # Convert the angle from degrees to radians
    rad = np.radians(angle)

    # Create a 2D rotation matrix
    rotation_matrix = np.array(
        [
            [np.cos(rad), -np.sin(rad)],  # First row of the rotation matrix
            [np.sin(rad), np.cos(rad)],  # Second row of the rotation matrix
        ]
    )

    # Rotate the vector by multiplying it with the rotation matrix
    return np.dot(rotation_matrix, vec)


def find_triangle_tips(center: np.ndarray, tip1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds the tips of a triangle given the center and one tip.

    Args:
        center (np.ndarray): The center of the triangle.
        tip1 (np.ndarray): One tip of the triangle.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The other two tips of the triangle.
    """
    # Calculate the vector from the center to the first tip
    vec1 = np.array(tip1) - np.array(center)

    # Rotate this vector by 120 degrees to find the second tip
    vec2 = rotate_vector(vec1, 120)

    # Rotate the original vector by 240 degrees to find the third tip
    vec3 = rotate_vector(vec1, 240)

    # Calculate the coordinates of the second and third tips by adding the rotated vectors to the center
    tip2 = center + vec2
    tip3 = center + vec3

    # Return the coordinates of the two additional tips
    return tip2, tip3


# def minimum_image_distance(position1: List[float], position2: List[float], system_size: List[float]) -> float:
#     """
#     Calculates the minimum image distance between two positions in a periodic system.
#
#     Args:
#         position1 (List[float]): The first position.
#         position2 (List[float]): The second position.
#         system_size (List[float]): The size of the periodic system.
#
#     Returns:
#         float: The minimum image distance between the two positions.
#     """
#     delta = np.zeros(2)
#     for i in range(2):
#         delta[i] = position1[i] - position2[i]
#         delta[i] -= system_size[i] * round(delta[i] / system_size[i])
#     return np.sqrt(np.sum(delta ** 2))


def minimum_image_distance(
    position1: List[float], position2: List[float], system_size: Tuple[float, float]
) -> float:  # ToDo: Methode auch leicht abgeändert. Sollte das Gleiche tun wie obige?
    """
    Calculates the minimum image distance between two positions in a periodic system.

    Args:
        position1 (List[float]): The first position.
        position2 (List[float]): The second position.
        system_size (Tuple[float, float]): The size of the periodic system.

    Returns:
        float: The minimum image distance between the two positions.
    """
    # Calculate the difference vector, adjusted for periodic boundaries
    delta = np.array(
        [
            position1[i] - position2[i] - system_size[i] * round((position1[i] - position2[i]) / system_size[i])
            for i in range(2)
        ]
    )
    # Return the Euclidean distance between the two positions
    return np.sqrt(np.sum(delta**2))


def positions_are_adjacent(
    position1: List[float], position2: List[float], cutoff_distance: float, system_size: Tuple[float, float]
) -> bool:
    """
    Checks if two positions are adjacent in a periodic system.

    Args:
        position1 (List[float]): The first position.
        position2 (List[float]): The second position.
        cutoff_distance (float): The cutoff distance for adjacency.
        system_size (Tuple[float, float]): The size of the periodic system.

    Returns:
        bool: True if the positions are adjacent, False otherwise.
    """
    # Calculate the minimum image distance between the two positions
    distance = minimum_image_distance(position1, position2, system_size)

    # Return True if the distance is less than the cutoff distance, otherwise False
    return distance < cutoff_distance


def random_rotation_matrix_2d() -> npt.NDArray:
    """
    Generates a random 2D rotation matrix.

    Returns:
        npt.NDArray: The random 2D rotation matrix.
    """
    # Generate a random angle theta between 0 and 2*pi radians
    theta = np.random.uniform(0, 2 * np.pi)

    # Calculate the cosine and sine of the angle
    cos, sin = np.cos(theta), np.sin(theta)

    # Create the 2D rotation matrix using the cosine and sine values
    return np.array([[cos, -sin], [sin, cos]])


def random_rotate_group_list(group_list: List[List[Union[str, float]]]) -> List[List[Union[str, float]]]:
    """
    Rotates a group of atoms randomly in 2D.

    Args:
        group_list (List[List[Union[str, float]]]): The group of atoms to rotate.

    Returns:
        List[List[Union[str, float]]]: The randomly rotated group.
    """
    # Generate a random 2D rotation matrix
    rotation_matrix = random_rotation_matrix_2d()

    # Apply the rotation to each atom's x and y coordinates in the group list
    rotated_group_list = [[atom[0]] + (rotation_matrix.dot(atom[1:3])).tolist() + [atom[3]] for atom in group_list]

    # Return the list of rotated atoms
    return rotated_group_list


def is_point_inside_triangle(
    tip1: List[float], tip2: List[float], tip3: List[float], point: Tuple[float, float]
) -> bool:
    """
    Checks if a point is inside a triangle.

    Args:
        tip1 (List[float]): The first tip of the triangle.
        tip2 (List[float]): The second tip of the triangle.
        tip3 (List[float]): The third tip of the triangle.
        point (Tuple[float, float]): The point to check.

    Returns:
        bool: True if the point is inside the triangle, False otherwise.
    """

    def area(a: List[float], b: List[float], c: List[float]) -> float:
        # Calculate the area of the triangle formed by points a, b, and c
        return 0.5 * abs((a[0] - c[0]) * (b[1] - a[1]) - (a[0] - b[0]) * (c[1] - a[1]))

    # Calculate the area of the whole triangle
    A = area(tip1, tip2, tip3)
    # Calculate the area of the triangle formed by the point and the tips
    A1 = area(point, tip2, tip3)
    A2 = area(tip1, point, tip3)
    A3 = area(tip1, tip2, point)

    # Calculate the barycentric coordinates
    l1 = A1 / A
    l2 = A2 / A
    l3 = A3 / A

    # Check if the point is inside the triangle
    return 0 <= l1 <= 1 and 0 <= l2 <= 1 and 0 <= l3 <= 1 and abs(l1 + l2 + l3 - 1) < 1e-5
