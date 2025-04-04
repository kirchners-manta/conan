# from typing import TYPE_CHECKING
#
# if TYPE_CHECKING:
#     from conan.playground.structure_optimizer import OptimizationConfig, StructureOptimizer
import copy
import math
import random

# import time
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field
from math import cos, pi, sin
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from networkx.utils import pairwise
from scipy.optimize import minimize
from tqdm import tqdm

from conan.playground.structure_optimizer import OptimizationConfig, StructureOptimizer
from conan.playground.utils import (
    NitrogenSpecies,
    NitrogenSpeciesProperties,
    Position,
    create_position,
    get_color,
    get_neighbors_via_edges,
    minimum_image_distance,
    minimum_image_distance_vectorized,
    write_xyz,
)

# Define a namedtuple for structural components
# This namedtuple will be used to store the atom(s) around which the doping structure is built and its/their neighbors
StructuralComponents = namedtuple("StructuralComponents", ["structure_building_atoms", "structure_building_neighbors"])


class AtomLabeler:
    def __init__(self, graph: nx.Graph, doping_structures: Optional["DopingStructureCollection"] = None):
        self.graph = graph
        """The networkx graph representing the structure of the material (e.g., graphene sheet)."""
        self.doping_structures = doping_structures
        """The collection of doping structures within the structure."""

    def label_atoms(self):
        """
        Label the atoms in the graphene structure based on their doping species and local environment.

        This method assigns labels to atoms based on the doping structures they belong to and their immediate
        environment.
        Atoms that are part of a doping structure get labeled according to their specific nitrogen or carbon species.
        In each doping cycle, the neighboring atoms of a C atom that are also within a cycle are also specified (_CC or
        _CN), as well as a graphitic-N neighbor outside the cycle, if present (_G).
        All other carbon atoms are labeled as "CG" for standard graphene carbon.

        In other words:
        Atoms in the same symmetrically equivalent environment get the same label, while those in different
        environments are labeled differently.
        """
        if not self.doping_structures:
            # Label all atoms as "CG" if there are no doping structures
            for node in self.graph.nodes:
                self.graph.nodes[node]["label"] = "CG"
            return

        # Loop through each doping structure and label the atoms
        for structure in self.doping_structures.structures:
            species = structure.species  # Get the nitrogen species (e.g., PYRIDINIC_1, PYRIDINIC_2, etc.)

            # Determine the appropriate labels for nitrogen and carbon atoms within the doping structure
            if species == NitrogenSpecies.GRAPHITIC:
                nitrogen_label = "NG"
                # Label nitrogen atom in GRAPHITIC species
                for atom in structure.nitrogen_atoms:
                    self.graph.nodes[atom]["label"] = nitrogen_label
            else:
                # For pyridinic species, use NP1, NP2, NP3, NP4 for nitrogen, and CP1, CP2, CP3, CP4 for carbon
                nitrogen_label = f"NP{species.value[-1]}"
                carbon_label_base = f"CP{species.value[-1]}"

                # Label nitrogen atoms within the doping structure
                for atom in structure.nitrogen_atoms:
                    self.graph.nodes[atom]["label"] = nitrogen_label

                # Label carbon atoms in the cycle of the doping structure
                for atom in structure.cycle:
                    if atom not in structure.nitrogen_atoms:
                        # Efficient one-liner to assign the label based on neighbors
                        cycle_neighbors = structure.subgraph.neighbors(atom)
                        self.graph.nodes[atom]["label"] = (
                            f"{carbon_label_base}_CC"
                            if all(self.graph.nodes[n]["element"] == "C" for n in cycle_neighbors)
                            else f"{carbon_label_base}_CN"
                        )

                        # Check for additional cases where a neighboring atom is Graphitic-N
                        neighbors = self.graph.neighbors(atom)
                        if any(
                            self.graph.nodes[n].get("nitrogen_species") == NitrogenSpecies.GRAPHITIC for n in neighbors
                        ):
                            self.graph.nodes[atom]["label"] += "_G"

        # Label remaining carbon atoms as "CG"
        for node in self.graph.nodes:
            if "label" not in self.graph.nodes[node]:  # If the node hasn't been labeled yet
                self.graph.nodes[node]["label"] = "CG"


@dataclass
class DopingStructure:
    """
    Represents a doping structure within the graphene sheet.

    Attributes
    ----------
    species : NitrogenSpecies
        The type of nitrogen doping.
    structural_components : StructuralComponents[List[int], List[int]]
        The structural components of the doping structure. This includes:
        - structure_building_atoms: List of atom IDs that form the structure. In case of graphitic doping, this list
        contains the atom IDs of the atoms that will be changed to nitrogen atoms. In case of pyridinic doping, this
        list contains the atom IDs of the atoms that will be removed to form the pyridinic structure.
        - structure_building_neighbors: List of neighbor atom IDs for the structure building atoms. Some (or all) of
        these neighbors will be replaced by nitrogen atoms to form the respective doping structure.
    nitrogen_atoms : List[int]
        List of atoms that are replaced by nitrogen atoms to form the doping structure.
    cycle : Optional[List[int]]
        List of atom IDs forming the cycle of the doping structure (this is only relevant for pyridinic doping).
    neighboring_atoms : Optional[List[int]]
        List of atom IDs neighboring the cycle of the doping structure (this is only relevant for pyridinic doping).
    subgraph : Optional[nx.Graph]
        The subgraph containing the doping structure.
    additional_edge : Optional[Tuple[int, int]]
        An additional edge added to the doping structure, needed for PYRIDINIC_1 doping.
    """

    species: NitrogenSpecies
    structural_components: StructuralComponents[List[int], List[int]]
    nitrogen_atoms: List[int]
    cycle: Optional[List[int]] = field(default=None)
    neighboring_atoms: Optional[List[int]] = field(default=None)
    subgraph: Optional[nx.Graph] = field(default=None)
    additional_edge: Optional[Tuple[int, int]] = field(default=None)

    @classmethod
    def create_structure(
        cls,
        structure: "MaterialStructure",
        species: NitrogenSpecies,
        structural_components: StructuralComponents[List[int], List[int]],
        start_node: Optional[int] = None,
    ):
        """
        Create a doping structure within the graphene sheet.

        This method creates a doping structure by detecting the cycle in the graph that includes the
        structure-building neighbors, ordering the cycle, adding any necessary edges, and identifying
        neighboring atoms of the cycle.

        Parameters
        ----------
        structure : MaterialStructure
            The carbon structure used for doping (e.g., GrapheneSheet, CNT, ...).
        species : NitrogenSpecies
            The type of nitrogen doping.
        structural_components : StructuralComponents[List[int], List[int]]
            The structural components of the doping structure.
        start_node : Optional[int], optional
            The start node for ordering the cycle. Default is None.

        Returns
        -------
        DopingStructure
            The created doping structure.
        """

        graph = structure.graph

        # Detect the cycle and create the subgraph
        cycle, subgraph = cls._detect_cycle_and_subgraph(graph, structural_components.structure_building_neighbors)

        # Order the cycle
        ordered_cycle = cls._order_cycle(subgraph, cycle, species, start_node)

        # Add edge if needed (only for PYRIDINIC_1 doping)
        additional_edge = None
        if species == NitrogenSpecies.PYRIDINIC_1:
            additional_edge = cls._add_additional_edge(
                structure, subgraph, structural_components.structure_building_neighbors, start_node
            )

        # Identify nitrogen atoms in the ordered cycle
        nitrogen_atoms = [node for node in ordered_cycle if graph.nodes[node]["element"] == "N"]

        # Identify and order neighboring atoms
        neighboring_atoms = cls._get_ordered_neighboring_atoms(graph, ordered_cycle)

        # Create and return the DopingStructure instance
        return cls(
            species=species,
            structural_components=structural_components,
            nitrogen_atoms=nitrogen_atoms,
            cycle=ordered_cycle,
            neighboring_atoms=neighboring_atoms,
            subgraph=subgraph,
            additional_edge=additional_edge,
        )

    @staticmethod
    def _detect_cycle_and_subgraph(graph: nx.Graph, neighbors: List[int]) -> Tuple[List[int], nx.Graph]:
        """
        Detect the cycle including the given neighbors and create the corresponding subgraph.

        Parameters
        ----------
        graph: nx.Graph
            The graph containing the cycle.
        neighbors : List[int]
            List of neighbor atom IDs.

        Returns
        -------
        Tuple[List[int], nx.Graph]
            The detected cycle and the subgraph containing the cycle.
        """

        # Find the shortest cycle that includes all the given neighbors
        cycle = DopingStructure._find_min_cycle_including_neighbors(graph, neighbors)

        # Create a subgraph from the detected cycle
        subgraph = graph.subgraph(cycle).copy()

        # Return the cycle and the corresponding subgraph
        return cycle, subgraph

    @staticmethod
    def _add_additional_edge(
        structure: "MaterialStructure", subgraph: nx.Graph, neighbors: List[int], start_node: int
    ) -> Tuple[int, int]:
        """
        Add an edge between neighbors if the nitrogen species is PYRIDINIC_1.

        Parameters
        ----------
        structure : MaterialStructure
            The carbon structure used for doping (e.g., GrapheneSheet, CNT, ...).
        subgraph : nx.Graph
            The subgraph containing the cycle.
        neighbors : List[int]
            List of neighbor atom IDs.
        start_node: int
            The start node ID.

        Returns
        -------
        Tuple[int, int]
            The nodes between which the additional edge was added.
        """

        graph = structure.graph

        # Remove the start node from the list of neighbors to get the two neighbors to connect
        neighbors.remove(start_node)

        # Get the positions of the two remaining neighbors
        pos1 = graph.nodes[neighbors[0]]["position"]
        pos2 = graph.nodes[neighbors[1]]["position"]

        if isinstance(structure, GrapheneSheet):
            # Calculate the box size for periodic boundary conditions
            box_size = (
                structure.actual_sheet_width + structure.c_c_bond_length,
                structure.actual_sheet_height + structure.cc_y_distance,
                0.0,
            )

            # Calculate the bond length between the two neighbors considering minimum image distance
            bond_length, _ = minimum_image_distance(pos1, pos2, box_size)

            # Add the edge to the main graph and the subgraph with the bond length
            graph.add_edge(neighbors[0], neighbors[1], bond_length=bond_length)
            subgraph.add_edge(neighbors[0], neighbors[1], bond_length=bond_length)
        else:
            # ToDo: If the position adjustment is then also to be performed for the 3D structures, an alternative for
            #  the 'minimum_image_distance' function must be found here in order to calculate the bond_length for
            #  structures that are connected via the periodic edges
            # For CNT or other 3D structures, add the edge without bond length calculation
            graph.add_edge(neighbors[0], neighbors[1])
            subgraph.add_edge(neighbors[0], neighbors[1])

        # Return the nodes between which the edge was added
        return neighbors[0], neighbors[1]

    @staticmethod
    def _order_cycle(
        subgraph: nx.Graph, cycle: List[int], species: NitrogenSpecies, start_node: Optional[int] = None
    ) -> List[int]:
        """
        Order the nodes in the cycle starting from a specified node or a suitable node based on the nitrogen species.

        Parameters
        ----------
        subgraph : nx.Graph
            The subgraph containing the cycle.
        cycle : List[int]
            List of atom IDs forming the cycle.
        species : NitrogenSpecies
            The nitrogen doping species.
        start_node : Optional[int], optional
            The start node ID. If None, a suitable start node will be determined based on the nitrogen species.

        Returns
        -------
        List[int]
            The ordered list of nodes in the cycle.
        """

        if start_node is None:
            # If no start node is provided, find a suitable starting node based on the nitrogen species
            start_node = DopingStructure._find_start_node(subgraph, species)

        # Initialize the list to store the ordered cycle and a set to track visited nodes
        ordered_cycle = []
        current_node = start_node
        visited = set()

        # Continue ordering nodes until all nodes in the cycle are included
        while len(ordered_cycle) < len(cycle):
            # Add the current node to the ordered list and mark it as visited
            ordered_cycle.append(current_node)
            visited.add(current_node)

            # Find the neighbors of the current node that are in the cycle and not yet visited
            neighbors = [node for node in subgraph.neighbors(current_node) if node not in visited]

            # If there are unvisited neighbors, move to the next neighbor; otherwise, break the loop
            if neighbors:
                current_node = neighbors[0]
            else:
                break
        return ordered_cycle

    @staticmethod
    def _find_min_cycle_including_neighbors(graph: nx.Graph, neighbors: List[int]) -> List[int]:
        """
        Find the shortest cycle in the graph that includes all the given neighbors.

        This method uses an iterative approach to expand the subgraph starting from the given neighbors. In each
        iteration, it expands the subgraph by adding edges of the current nodes until a cycle containing all neighbors
        is found. The cycle detection is done using the `cycle_basis` method, which is efficient for small subgraphs
        that are incrementally expanded.

        Parameters
        ----------
        graph: nx.Graph
            The whole graphene sheet graph.
        neighbors : List[int]
            A list of nodes that should be included in the cycle.

        Returns
        -------
        List[int]
            The shortest cycle that includes all the given neighbors, if such a cycle exists. Otherwise, an empty list.
        """

        # Initialize the subgraph with the neighbors
        subgraph = nx.Graph()
        subgraph.add_nodes_from(neighbors)

        # Add edges from each neighbor to the subgraph
        for node in neighbors:
            subgraph.add_edges_from(graph.edges(node))

        # Keep track of visited edges to avoid unwanted cycles
        visited_edges: Set[Tuple[int, int]] = set(subgraph.edges)

        # Expand the subgraph until the cycle is found
        while True:
            # Find all cycles in the current subgraph
            cycles: List[List[int]] = list(nx.cycle_basis(subgraph))
            for cycle in cycles:
                # Check if the current cycle includes all the neighbors
                if all(neighbor in cycle for neighbor in neighbors):
                    return cycle

            # If no cycle is found, expand the subgraph by adding neighbors of the current subgraph
            new_edges: Set[Tuple[int, int]] = set()
            for node in subgraph.nodes:
                new_edges.update(graph.edges(node))

            # Only add new edges that haven't been visited
            new_edges.difference_update(visited_edges)
            if not new_edges:
                return []

            # Add the new edges to the subgraph and update the visited edges
            subgraph.add_edges_from(new_edges)
            visited_edges.update(new_edges)

    @staticmethod
    def _find_start_node(subgraph: nx.Graph, species: NitrogenSpecies) -> int:
        """
        Find a suitable starting node for a given cycle based on the nitrogen species. The starting node is used to
        ensure a consistent iteration order through the cycle, matching the bond lengths and angles correctly.

        Parameters
        ----------
        subgraph : nx.Graph
            The graph containing the cycle.
        species : NitrogenSpecies
            The nitrogen doping species that was inserted for the cycle.

        Returns
        -------
        int
            The starting node ID.

        Raises
        ------
        ValueError
            If no suitable starting node is found in the cycle.
        """

        start_node = None
        if species in {NitrogenSpecies.PYRIDINIC_4, NitrogenSpecies.PYRIDINIC_3}:
            # Find the starting node that has no "N" neighbors within the cycle and is not "N" itself
            for node in subgraph.nodes:
                # Skip the node if it is already a nitrogen atom
                if subgraph.nodes[node]["element"] == "N":
                    continue
                # Get the neighbors of the current node
                neighbors = get_neighbors_via_edges(subgraph, node)
                # Check if none of the neighbors of the node are nitrogen atoms, provided the neighbor is within the
                # cycle
                if all(subgraph.nodes[neighbor]["element"] != "N" for neighbor in neighbors):
                    # If the current node meets all conditions, set it as the start node
                    start_node = node
                    break
        # Raise an error if no suitable start node is found
        if start_node is None:
            raise ValueError("No suitable starting node found in the subgraph.")
        return start_node

    @staticmethod
    def _get_ordered_neighboring_atoms(graph: nx.Graph, ordered_cycle: List[int]) -> List[int]:
        """
        Identify and order neighboring atoms connected to the cycle but not part of it.

        Parameters
        ----------
        graph : nx.Graph
            The main graph containing the structure.
        ordered_cycle : List[int]
            The list of atom IDs forming the ordered cycle.

        Returns
        -------
        List[int]
            The list of neighboring atom IDs ordered based on their connection to the ordered cycle.
        """
        neighboring_atoms = []
        for node in ordered_cycle:
            # Get the neighbor of the node that is not in the cycle
            neighbor_without_cycle = [neighbor for neighbor in graph.neighbors(node) if neighbor not in ordered_cycle]
            neighboring_atoms.extend(neighbor_without_cycle)
        return neighboring_atoms


@dataclass
class DopingStructureCollection:
    """
    Manages a collection of doping structures within the graphene sheet.

    Attributes
    ----------
    structures : List[DopingStructure]
        List of doping structures that are added to the collection.
    chosen_atoms : Dict[NitrogenSpecies, List[int]]
        Dictionary mapping nitrogen species to lists of chosen atom IDs. This is used to keep track of atoms that have
        already been chosen for doping (i.e., replaced by nitrogen atoms) to track the percentage of doping for each
        species.
    """

    structures: List[DopingStructure] = field(default_factory=list)
    chosen_atoms: Dict[NitrogenSpecies, List[int]] = field(default_factory=lambda: defaultdict(list))

    def add_structure(self, doping_structure: DopingStructure):
        """
        Add a doping structure to the collection and update the chosen atoms.
        """

        self.structures.append(doping_structure)
        self.chosen_atoms[doping_structure.species].extend(doping_structure.nitrogen_atoms)

    def get_structures_for_species(self, species: NitrogenSpecies) -> List[DopingStructure]:
        """
        Get a list of doping structures for a specific species.

        Parameters
        ----------
        species : NitrogenSpecies
            The nitrogen species to filter by.

        Returns
        -------
        List[DopingStructure]
            A list of doping structures for the specified species.
        """

        return [structure for structure in self.structures if structure.species == species]


class DopingHandler:
    def __init__(self, carbon_structure: "MaterialStructure"):
        """
        Initialize the DopingHandler with a structure.

        Parameters
        ----------
        carbon_structure : MaterialStructure
            The structure (e.g., GrapheneSheet or CNT) to be doped.
        """
        self.carbon_structure = carbon_structure
        self.graph = carbon_structure.graph

        # Initialize the list of possible carbon atoms
        self._possible_carbon_atoms_needs_update = True
        """Flag to indicate that the list of possible carbon atoms needs to be updated."""
        self._possible_carbon_atoms = []
        """List of possible carbon atoms that can be used for nitrogen doping."""

        self.species_properties = self._initialize_species_properties()
        """A dictionary mapping each NitrogenSpecies to its corresponding NitrogenSpeciesProperties.
        This includes bond lengths and angles characteristic to each species that we aim to achieve in the doping."""

        self.doping_structures = DopingStructureCollection()
        """A dataclass to store information about doping structures in the carbon structure."""

    @property
    def possible_carbon_atoms(self):
        """Get the list of possible carbon atoms for doping."""
        if self._possible_carbon_atoms_needs_update:
            self._update_possible_carbon_atoms()
        return self._possible_carbon_atoms

    def _update_possible_carbon_atoms(self):
        """Update the list of possible carbon atoms for doping."""
        self._possible_carbon_atoms = [
            node for node, data in self.graph.nodes(data=True) if data.get("possible_doping_site")
        ]
        self._possible_carbon_atoms_needs_update = False

    def mark_possible_carbon_atoms_for_update(self):
        """Mark the list of possible carbon atoms as needing an update."""
        self._possible_carbon_atoms_needs_update = True

    @staticmethod
    def _initialize_species_properties() -> Dict[NitrogenSpecies, NitrogenSpeciesProperties]:
        # Initialize properties for PYRIDINIC_4 nitrogen species with target bond lengths and angles
        pyridinic_4_properties = NitrogenSpeciesProperties(
            target_bond_lengths_cycle=[
                1.45,
                1.34,
                1.32,
                1.47,
                1.32,
                1.34,
                1.45,
                1.45,
                1.34,
                1.32,
                1.47,
                1.32,
                1.34,
                1.45,
            ],
            target_angles_cycle=[
                120.26,
                121.02,
                119.3,
                119.3,
                121.02,
                120.26,
                122.91,
                120.26,
                121.02,
                119.3,
                119.3,
                121.02,
                120.26,
                122.91,
            ],
            target_bond_lengths_neighbors=[
                1.43,
                1.43,
                1.42,
                1.42,
                1.43,
                1.43,
                1.43,
                1.42,
                1.42,
                1.43,
            ],
            target_angles_neighbors=[
                118.54,
                118.54,
                118.86,
                120.88,
                122.56,
                118.14,
                118.14,
                122.56,
                120.88,
                118.86,
                118.54,
                118.54,
                118.86,
                120.88,
                122.56,
                118.14,
                118.14,
                122.56,
                120.88,
                118.86,
            ],
        )
        # Initialize properties for PYRIDINIC_3 nitrogen species with target bond lengths and angles
        pyridinic_3_properties = NitrogenSpeciesProperties(
            target_bond_lengths_cycle=[1.45, 1.33, 1.33, 1.45, 1.45, 1.33, 1.33, 1.45, 1.45, 1.33, 1.33, 1.45],
            target_angles_cycle=[
                120.00,
                122.17,
                120.00,
                122.21,
                120.00,
                122.17,
                120.00,
                122.21,
                120.00,
                122.17,
                120.00,
                122.21,
            ],
            target_bond_lengths_neighbors=[
                1.42,
                1.43,
                1.43,
                1.42,
                1.43,
                1.43,
                1.42,
                1.43,
                1.43,
            ],
            target_angles_neighbors=[
                118.88,
                118.88,
                118.92,
                121.10,
                121.10,
                118.92,
                118.88,
                118.88,
                118.92,
                121.10,
                121.10,
                118.92,
                118.88,
                118.88,
                118.92,
                121.10,
                121.10,
                118.92,
            ],
        )
        # Initialize properties for PYRIDINIC_2 nitrogen species with target bond lengths and angles
        pyridinic_2_properties = NitrogenSpeciesProperties(
            target_bond_lengths_cycle=[1.39, 1.42, 1.42, 1.33, 1.35, 1.44, 1.44, 1.35, 1.33, 1.42, 1.42, 1.39],
            target_angles_cycle=[
                125.51,
                118.04,
                117.61,
                120.59,
                121.71,
                122.14,
                121.71,
                120.59,
                117.61,
                118.04,
                125.51,
                125.04,
            ],
            target_bond_lengths_neighbors=[
                1.45,
                1.41,
                1.41,
                1.44,
                1.44,
                1.44,
                1.41,
                1.41,
                1.45,
            ],
            target_angles_neighbors=[
                116.54,
                117.85,
                121.83,
                120.09,
                119.20,
                123.18,
                119.72,
                118.55,
                118.91,
                118.91,
                118.55,
                119.72,
                123.18,
                119.20,
                120.09,
                121.83,
                117.85,
                116.54,
            ],
        )
        # Initialize properties for PYRIDINIC_1 nitrogen species with target bond lengths and angles
        pyridinic_1_properties = NitrogenSpeciesProperties(
            target_bond_lengths_cycle=[1.31, 1.42, 1.45, 1.51, 1.42, 1.40, 1.40, 1.42, 1.51, 1.45, 1.42, 1.31, 1.70],
            target_angles_cycle=[
                115.48,
                118.24,
                128.28,
                109.52,
                112.77,
                110.35,
                112.77,
                109.52,
                128.28,
                118.24,
                115.48,
                120.92,
            ],
            target_bond_lengths_neighbors=[
                1.41,
                1.42,
                1.48,
                1.41,
                1.38,
                1.41,
                1.48,
                1.42,
                1.41,
            ],
            target_angles_neighbors=[
                121.99,
                122.51,
                115.67,
                126.09,
                111.08,
                120.63,
                131.00,
                116.21,
                124.82,
                124.82,
                116.21,
                131.00,
                120.63,
                111.08,
                126.09,
                115.67,
                122.51,
                121.99,
            ],
            target_angles_additional_angles=[
                148.42,
                102.06,
                102.06,
                148.42,
            ],
        )
        # graphitic_properties = NitrogenSpeciesProperties(
        #     target_bond_lengths=[1.42],
        #     target_angles=[120.0],
        # )

        # Initialize a dictionary mapping each NitrogenSpecies to its corresponding properties
        species_properties = {
            NitrogenSpecies.PYRIDINIC_4: pyridinic_4_properties,
            NitrogenSpecies.PYRIDINIC_3: pyridinic_3_properties,
            NitrogenSpecies.PYRIDINIC_2: pyridinic_2_properties,
            NitrogenSpecies.PYRIDINIC_1: pyridinic_1_properties,
            # NitrogenSpecies.GRAPHITIC: graphitic_properties,
        }
        return species_properties

    @staticmethod
    def get_next_possible_carbon_atom(possible_carbon_atoms, tested_atoms):
        """
        Get a randomly selected carbon atom from the list of possible carbon atoms that hasn't been tested yet.

        Parameters
        ----------
        possible_carbon_atoms : list
            The list of possible carbon atoms to select from.
        tested_atoms : set
            A set of atom IDs that have already been tested.

        Returns
        -------
        int or None
            The ID of the selected carbon atom, or None if all atoms have been tested.
        """
        untested_atoms = list(set(possible_carbon_atoms) - tested_atoms)
        if not untested_atoms:
            return None  # Return None if all atoms have been tested
        atom_id = random.choice(untested_atoms)  # Randomly select an untested atom ID
        return atom_id  # Return the selected atom ID

    # @staticmethod
    # def get_next_possible_carbon_atom(atom_list):
    #     """
    #     Get a randomly selected carbon atom from the list of possible carbon atoms.
    #
    #     This method randomly selects a carbon atom from the provided list and removes it from the list.
    #     This ensures that the same atom is not selected more than once.
    #
    #     Parameters
    #     ----------
    #     atom_list : list
    #         The list of possible carbon atoms to select from.
    #
    #     Returns
    #     -------
    #     int or None
    #         The ID of the selected carbon atom, or None if the list is empty.
    #     """
    #
    #     if not atom_list:
    #         return None  # Return None if the list is empty
    #     atom_id = random.choice(atom_list)  # Randomly select an atom ID from the list
    #     atom_list.remove(atom_id)  # Remove the selected atom ID from the list
    #     return atom_id  # Return the selected atom ID

    def add_nitrogen_doping(self, total_percentage: float = None, percentages: dict = None):
        """
        Add nitrogen doping to the structure.

        This method handles the addition of nitrogen doping to the structure using the provided percentages and
        utilizing graph manipulation techniques to insert the doping structures.

        Parameters
        ----------
        total_percentage : float, optional
            The total percentage of carbon atoms to replace with nitrogen atoms.
        percentages : dict, optional
            A dictionary specifying the percentages for each nitrogen species.

        Raises
        ------
        ValueError
            If the specific percentages exceed the total percentage beyond a small tolerance.
        """
        # Validate the input for percentages
        if percentages is not None:
            if not isinstance(percentages, dict):
                raise ValueError(
                    "percentages must be a dictionary with NitrogenSpecies as keys and int or float as values."
                )

            for key, value in percentages.items():
                if not isinstance(key, NitrogenSpecies):
                    raise ValueError(
                        f"Invalid key in percentages dictionary: {key}. Keys must be of type NitrogenSpecies."
                    )
                if not isinstance(value, (int, float)):
                    raise ValueError(
                        f"Invalid value in percentages dictionary for key {key} with value {value}. Values must be int "
                        f"or float."
                    )

        # Validate the input for total_percentage
        if total_percentage is not None and not isinstance(total_percentage, (int, float)):
            raise ValueError("total_percentage must be an int or float.")

        # Copy the percentages dictionary to avoid modifying the input
        percentages = percentages.copy() if percentages else {}

        # Validate specific percentages and calculate the remaining percentage
        if percentages:
            if total_percentage is None:
                # Set total to sum of specific percentages if not provided
                total_percentage = sum(percentages.values())
            else:
                # Sum of provided specific percentages
                specific_total_percentage = sum(percentages.values())
                # Define a small tolerance to account for floating-point errors
                tolerance = 1e-6
                if abs(specific_total_percentage - total_percentage) > tolerance:
                    # Raise an error if the sum of specific percentages exceeds the total percentage beyond the
                    # tolerance
                    raise ValueError(
                        f"The total specific percentages {specific_total_percentage}% are higher than the "
                        f"total_percentage {total_percentage}%. Please adjust your input so that the sum of the "
                        f"'percentages' is less than or equal to 'total_percentage'."
                    )
        else:
            # Set a default total percentage if not provided
            total_percentage = total_percentage if total_percentage is not None else 10.0

            # # Set a default total percentage if not provided
            # if total_percentage is None:
            #     total_percentage = 10  # Default total percentage
            # # Initialize an empty dictionary if no specific percentages are provided
            # percentages = {}

        # Calculate the remaining percentage for other species
        remaining_percentage = total_percentage - sum(percentages.values())

        # Define a tolerance for floating-point comparison
        tolerance = 1e-6

        if remaining_percentage > tolerance:
            # Determine available species not included in the specified percentages
            available_species = [species for species in NitrogenSpecies if species not in percentages]
            # Distribute the remaining percentage equally among available species
            default_distribution = {
                species: remaining_percentage / len(available_species) for species in available_species
            }

            # # Add the default distribution to the specified percentages
            # for species, pct in default_distribution.items():
            #     if species not in percentages:
            #         percentages[species] = pct
            # Add the default distribution to the local percentages dictionary

            percentages.update(default_distribution)
        else:
            # If the remaining percentage is negligible, we ignore it
            pass

        # Calculate the number of nitrogen atoms to add based on the given percentage
        num_atoms = self.graph.number_of_nodes()
        specific_num_nitrogen = {species: int(num_atoms * pct / 100) for species, pct in percentages.items()}

        # Check if all specific_num_nitrogen values are zero
        if all(count == 0 for count in specific_num_nitrogen.values()):
            warnings.warn(
                "The selected doping percentage is too low or the structure is too small to allow for doping.",
                UserWarning,
            )
            return  # Exit the method early if no doping can be done

        # Define the order of nitrogen doping insertion based on the species
        species_order = [
            NitrogenSpecies.PYRIDINIC_4,
            NitrogenSpecies.PYRIDINIC_3,
            NitrogenSpecies.PYRIDINIC_2,
            NitrogenSpecies.PYRIDINIC_1,
            NitrogenSpecies.GRAPHITIC,
        ]
        for species in species_order:
            if species in specific_num_nitrogen:
                num_nitrogen_atoms = specific_num_nitrogen[species]
                # Insert the doping structures for the current species
                self._insert_doping_structures(num_nitrogen_atoms, species)

        # Calculate the actual percentages of added nitrogen species
        total_atoms = self.graph.number_of_nodes()
        actual_percentages = {
            species.value: (
                round((len(self.doping_structures.chosen_atoms[species]) / total_atoms) * 100, 2)
                if total_atoms > 0
                else 0
            )
            for species in NitrogenSpecies
        }

        # Display the results in a DataFrame and add the total doping percentage
        total_doping_percentage = sum(actual_percentages.values())
        doping_percentages_df = pd.DataFrame.from_dict(
            actual_percentages, orient="index", columns=["Actual Percentage"]
        )
        doping_percentages_df.index.name = "Nitrogen Species"
        doping_percentages_df.reset_index(inplace=True)
        total_row = pd.DataFrame([{"Nitrogen Species": "Total Doping", "Actual Percentage": total_doping_percentage}])
        doping_percentages_df = pd.concat([doping_percentages_df, total_row], ignore_index=True)
        print(f"\n{doping_percentages_df}")

    def _insert_doping_structures(self, num_nitrogen: int, nitrogen_species: NitrogenSpecies):
        """
        Insert doping structures of a specific nitrogen species into the graphene sheet.

        Parameters
        ----------
        num_nitrogen : int
            The number of nitrogen atoms of the specified species to add.
        nitrogen_species : NitrogenSpecies
            The type of nitrogen doping to add.

        Notes
        -----
        First, a carbon atom is randomly selected. Then, it is checked whether this atom position is suitable for
        building the doping structure around it (i.e., the new structure to be inserted should not overlap with any
        existing structure). If suitable, the doping structure is built by, for example, removing atoms, replacing
        other C atoms with N atoms, and possibly adding new bonds between atoms (in the case of Pyridinic_1). After
        the structure is inserted, all atoms of this structure are excluded from further doping positions.
        """
        # Create a set to keep track of tested atoms
        tested_atoms = set()

        # Loop until the required number of nitrogen atoms is added or there are no more possible carbon atoms to test
        while len(self.doping_structures.chosen_atoms[nitrogen_species]) < num_nitrogen and len(tested_atoms) < len(
            self.possible_carbon_atoms
        ):
            # Get the next possible carbon atom to test for doping
            atom_id = self.get_next_possible_carbon_atom(self.possible_carbon_atoms, tested_atoms)
            if atom_id is None:
                break  # No more atoms to test

            # Add the atom to tested_atoms
            tested_atoms.add(atom_id)

            # Check if the atom_id is a valid doping position and return the structural components
            is_valid, structural_components = self._is_valid_doping_site(nitrogen_species, atom_id)
            if not is_valid:
                # No valid doping position found, proceed to the next possible carbon atom
                continue

            # The doping position is valid, proceed with nitrogen doping
            if nitrogen_species == NitrogenSpecies.GRAPHITIC:
                # Handle graphitic doping
                self._handle_graphitic_doping(structural_components)
            else:
                # Handle pyridinic doping
                self._handle_pyridinic_doping(structural_components, nitrogen_species)

            # Reset tested_atoms since possible_carbon_atoms has changed
            tested_atoms = set()

        # Warn if not all requested nitrogen atoms could be placed due to proximity constraints
        if len(self.doping_structures.chosen_atoms[nitrogen_species]) < num_nitrogen:
            warning_message = (
                f"\nWarning: Only {len(self.doping_structures.chosen_atoms[nitrogen_species])} nitrogen atoms of "
                f"species {nitrogen_species.value} could be placed due to proximity constraints."
            )
            warnings.warn(warning_message, UserWarning)

    def _handle_graphitic_doping(self, structural_components: StructuralComponents):
        """
        Handle the graphitic nitrogen doping process.

        This method takes the provided structural components and performs the doping process by converting a selected
        carbon atom to a nitrogen atom. It also marks the affected atoms to prevent further doping in those positions
        and updates the internal data structures accordingly.

        Parameters
        ----------
        structural_components : StructuralComponents
            The structural components required to build the graphitic doping structure. This includes the atom that
            will be changed to nitrogen and its neighboring atoms.
        """

        # Get the atom ID of the structure-building atom (the one to be doped with nitrogen)
        atom_id = structural_components.structure_building_atoms[0]
        # Get the neighbors of the structure-building atom
        neighbors = structural_components.structure_building_neighbors

        # Update the selected atom's element to nitrogen and set its nitrogen species
        self.graph.nodes[atom_id]["element"] = "N"
        self.graph.nodes[atom_id]["nitrogen_species"] = NitrogenSpecies.GRAPHITIC

        # Mark this atom as no longer a possible doping site
        self.graph.nodes[atom_id]["possible_doping_site"] = False
        # Iterate through each neighbor and mark them as no longer possible doping sites
        for neighbor in neighbors:
            self.graph.nodes[neighbor]["possible_doping_site"] = False

        # Flag to indicate that the list of possible carbon atoms needs to be updated
        self.mark_possible_carbon_atoms_for_update()

        # Create the doping structure
        doping_structure = DopingStructure(
            species=NitrogenSpecies.GRAPHITIC,  # Set the nitrogen species
            structural_components=structural_components,  # Use the provided structural components
            nitrogen_atoms=[atom_id],  # List of nitrogen atoms in this structure
        )

        # Add the doping structure to the collection
        self.doping_structures.add_structure(doping_structure)

    def _handle_pyridinic_doping(self, structural_components: StructuralComponents, nitrogen_species: NitrogenSpecies):
        """
        Handle the pyridinic nitrogen doping process for the specified nitrogen species.

        This method performs pyridinic doping by removing specific carbon atoms and possibly replacing some neighbors
        with nitrogen atoms, depending on the doping type specified. It also updates internal data structures to reflect
        the changes and ensures no further doping occurs at these locations.

        Parameters
        ----------
        structural_components : StructuralComponents
            The structural components including the atom(s) to be removed and its/their neighboring atoms.
        nitrogen_species : NitrogenSpecies
            The specific type of nitrogen doping to be applied, such as PYRIDINIC_1, PYRIDINIC_2, etc.
        """

        # Remove the carbon atom(s) specified in the structural components from the graph
        for atom in structural_components.structure_building_atoms:
            self.graph.remove_node(atom)  # Remove the atom from the graph
            # Note: The possible_carbon_atoms list is updated later to ensure synchronization with the graph

        # Determine the start node based on the species-specific logic; this is used to order the cycle correctly to
        # ensure the bond lengths and angles are consistent with the target values
        start_node = self._handle_species_specific_logic(
            nitrogen_species, structural_components.structure_building_neighbors
        )

        # Create a new doping structure using the provided nitrogen species and structural components. This involves the
        # creation of a cycle that includes all neighbors of the removed carbon atom(s) and finding a suitable start
        # node for the cycle if not already determined. The cycle is used to build the doping structure. In case of
        # PYRIDINIC_1, an additional edge is added between the neighbors.
        doping_structure = DopingStructure.create_structure(
            self.carbon_structure,
            nitrogen_species,
            structural_components,
            start_node,
        )

        # Add the newly created doping structure to the collection for management and tracking
        self.doping_structures.add_structure(doping_structure)

        # Mark all nodes involved in the newly formed cycle as no longer valid for further doping
        for node in doping_structure.cycle:
            self.graph.nodes[node]["possible_doping_site"] = False

        # Update the list of possible carbon atoms since the doping structure may have affected several nodes and edges
        self.mark_possible_carbon_atoms_for_update()

    def _handle_species_specific_logic(self, nitrogen_species: NitrogenSpecies, neighbors: List[int]) -> Optional[int]:
        """
        Handle species-specific logic for adding nitrogen atoms.

        This method applies the logic specific to each type of nitrogen doping species. It updates the graph by
        replacing certain carbon atoms with nitrogen atoms and determines the start node for the doping structure cycle
        in case of PYRIDINIC_1 and PYRIDINIC_2 species.

        Parameters
        ----------
        nitrogen_species : NitrogenSpecies
            The type of nitrogen doping to add.
        neighbors : List[int]
            List of neighbor atom IDs.

        Returns
        -------
        Optional[int]
            The start node ID if applicable, otherwise None.
        """

        start_node = None  # Initialize the start node as None

        if nitrogen_species == NitrogenSpecies.PYRIDINIC_1:
            # For PYRIDINIC_1, replace one carbon atom with a nitrogen atom
            selected_neighbor = random.choice(neighbors)  # Randomly select one neighbor to replace with nitrogen
            self.graph.nodes[selected_neighbor]["element"] = "N"  # Update the selected neighbor to nitrogen
            self.graph.nodes[selected_neighbor]["nitrogen_species"] = nitrogen_species  # Set its nitrogen species

            # Identify the start node for this cycle as the selected neighbor
            start_node = selected_neighbor

        elif nitrogen_species == NitrogenSpecies.PYRIDINIC_2:
            # For PYRIDINIC_2, replace two carbon atoms with nitrogen atoms
            selected_neighbors = random.sample(neighbors, 2)  # Randomly select two neighbors to replace with nitrogen
            for neighbor in selected_neighbors:
                self.graph.nodes[neighbor]["element"] = "N"  # Update the selected neighbors to nitrogen
                self.graph.nodes[neighbor]["nitrogen_species"] = nitrogen_species  # Set their nitrogen species

            # Identify the start node for this cycle using set difference
            remaining_neighbor = (set(neighbors) - set(selected_neighbors)).pop()  # Find the remaining neighbor
            start_node = remaining_neighbor  # The start node is the remaining neighbor

        elif nitrogen_species == NitrogenSpecies.PYRIDINIC_3 or nitrogen_species == NitrogenSpecies.PYRIDINIC_4:
            # For PYRIDINIC_3 and PYRIDINIC_4, replace three and four carbon atoms respectively with nitrogen atoms
            for neighbor in neighbors:
                self.graph.nodes[neighbor]["element"] = "N"  # Update all neighbors to nitrogen
                self.graph.nodes[neighbor]["nitrogen_species"] = nitrogen_species  # Set their nitrogen species

        return start_node  # Return the determined start node or None if not applicable

    def _is_valid_doping_site(
        self, nitrogen_species: NitrogenSpecies, atom_id: int
    ) -> Tuple[bool, StructuralComponents]:
        """
        Check if a given atom is a valid site for nitrogen doping based on the nitrogen species.

        This method verifies whether the specified carbon atom can be used for doping by checking proximity constraints
        based on the nitrogen species. If the atom is valid for doping, it returns True along with the structural
        components needed for doping. Otherwise, it returns False.

        Parameters
        ----------
        nitrogen_species : NitrogenSpecies
            The type of nitrogen doping to validate.
        atom_id: int
            The atom ID of the carbon atom to test for doping.

        Returns
        -------
        Tuple[bool, StructuralComponents]
            A tuple containing a boolean indicating if the position is valid and the structure components if valid.
            If the position is not valid, returns False and (None, None).

        Notes
        -----
        - For GRAPHITIC nitrogen species, it checks if all neighbors of the selected carbon atom are not nitrogen.
        - For PYRIDINIC nitrogen species (PYRIDINIC_1, PYRIDINIC_2, PYRIDINIC_3), it checks neighbors up to depth 2.
        - For PYRIDINIC_4 species, it checks neighbors up to depth 2 for two atoms and combines the neighbors.
        - It ensures that the selected atom and its neighbors are not part of any existing doping structures.
        """

        def all_neighbors_possible_carbon_atoms(neighbors: List[int]) -> bool:
            """
            Check if all provided neighbors are possible carbon atoms for doping.

            This method verifies whether all neighbors are in the list of possible carbon atoms.
            If any neighbor is not in the list, it indicates that the structure to be added would overlap with the cycle
            of an existing structure, which is not allowed.

            Parameters
            ----------
            neighbors : list
                A list of neighbor atom IDs.

            Returns
            -------
            bool
                True if all neighbors are possible atoms for doping, False otherwise.
            """

            return all(neighbor in self.possible_carbon_atoms for neighbor in neighbors)

        # # Get the next possible carbon atom to test for doping and its neighbors
        # atom_id = self.get_next_possible_carbon_atom(possible_carbon_atoms_to_test)
        # neighbors = get_neighbors_via_edges(self.graph, atom_id)

        # Check if the atom is still in the graph
        if not self.graph.has_node(atom_id):
            return False, (None, None)

        # Check if the atom is still a possible doping site
        if not self.graph.nodes[atom_id].get("possible_doping_site", True):
            return False, (None, None)

        # Get the neighbors of the atom
        neighbors = get_neighbors_via_edges(self.graph, atom_id)

        # Check whether the structure is periodic
        is_periodic = getattr(self.carbon_structure, "periodic", False)

        # Check the proximity constraints based on the nitrogen species
        if nitrogen_species == NitrogenSpecies.GRAPHITIC:
            # Collect elements and nitrogen species of neighbors
            neighbor_elements = [
                (self.graph.nodes[neighbor]["element"], self.graph.nodes[neighbor].get("nitrogen_species"))
                for neighbor in neighbors
            ]
            # Ensure all neighbors are not nitrogen atoms
            if all(elem != "N" for elem, _ in neighbor_elements):
                # Return True if the position is valid for graphitic doping and the structural components
                return True, StructuralComponents(
                    structure_building_atoms=[atom_id], structure_building_neighbors=neighbors
                )
            # Return False if the position is not valid for graphitic doping
            return False, (None, None)

        elif nitrogen_species in {
            NitrogenSpecies.PYRIDINIC_1,
            NitrogenSpecies.PYRIDINIC_2,
            NitrogenSpecies.PYRIDINIC_3,
        }:
            # Get neighbors up to depth 2 for the selected atom
            neighbors_len_2 = get_neighbors_via_edges(self.graph, atom_id, depth=2, inclusive=True)

            # If the neighbors list is less than 9, the doping structure would go beyond the edge
            if not is_periodic and len(neighbors_len_2) < 9:
                return False, (None, None)

            # Ensure all neighbors are possible atoms for doping
            if all_neighbors_possible_carbon_atoms(neighbors_len_2):
                # Return True if the position is valid for pyridinic doping and the structural components
                return True, StructuralComponents(
                    structure_building_atoms=[atom_id], structure_building_neighbors=neighbors
                )
            # Return False if the position is not valid for pyridinic doping
            return False, (None, None)

        elif nitrogen_species == NitrogenSpecies.PYRIDINIC_4:
            # Initialize some variables containing neighbor information
            selected_neighbor = None
            temp_neighbors = neighbors.copy()
            combined_len_2_neighbors = []

            # Iterate over the neighbors of the selected atom to find a direct neighbor that has a valid position
            while temp_neighbors and not selected_neighbor:
                # Find a direct neighbor that also needs to be removed randomly
                temp_neighbor = random.choice(temp_neighbors)
                temp_neighbors.remove(temp_neighbor)

                # Get neighbors up to depth 2 for the selected atom and a neighboring atom (if provided)
                neighbors_len_2_atom = get_neighbors_via_edges(self.graph, atom_id, depth=2, inclusive=True)
                neighbors_len_2_neighbor = get_neighbors_via_edges(self.graph, temp_neighbor, depth=2, inclusive=True)

                # Combine the two lists and remove the atom_id
                combined_len_2_neighbors = list(set(neighbors_len_2_atom + neighbors_len_2_neighbor))
                # Ensure all neighbors (from both atoms) are possible atoms for doping
                if all_neighbors_possible_carbon_atoms(combined_len_2_neighbors):
                    # Valid neighbor found
                    selected_neighbor = temp_neighbor

            if selected_neighbor is None:
                # Return False if no valid neighbor is found for pyridinic 4 doping
                return False, (None, None)

            # Combine the neighbors and remove atom_id and selected_neighbor
            # ToDo: This may be solved better by using an additional flag in get_neighbors_via_edges
            combined_neighbors = list(set(neighbors + get_neighbors_via_edges(self.graph, selected_neighbor)))
            combined_neighbors = [n for n in combined_neighbors if n not in {atom_id, selected_neighbor}]

            # If the neighbors list is less than 9, the doping structure would go beyond the edge
            if not is_periodic and len(combined_len_2_neighbors) < 14:
                return False, (None, None)

            # Return True if the position is valid for pyridinic 4 doping
            return True, StructuralComponents(
                structure_building_atoms=[atom_id, selected_neighbor], structure_building_neighbors=combined_neighbors
            )

        # Return False if the nitrogen species is not recognized
        return False, (None, None)


# Abstract base class for material structures
class MaterialStructure(ABC):
    def __init__(self):
        self.graph = nx.Graph()
        """The networkx graph representing the structure of the material (e.g., graphene sheet)."""
        self.doping_handler = DopingHandler(self)
        """The doping handler for the structure."""

    @staticmethod
    def _validate_bond_length(bond_length):
        if not isinstance(bond_length, (float, int)):
            raise TypeError(f"bond_length must be a float or int, but got {type(bond_length).__name__}.")
        if bond_length <= 0:
            raise ValueError(f"bond_length must be positive, but got {bond_length}.")

        # Return the validated bond_length as a float
        return float(bond_length)

    @abstractmethod
    def build_structure(self):
        """
        Abstract method for building the structure.
        """
        pass

    @abstractmethod
    def plot_structure(
        self, with_labels: bool = False, visualize_periodic_bonds: bool = True, save_path: Union[str, Path, None] = None
    ):
        """
        Abstract method for plotting the structure.

        Parameters
        ----------
        with_labels : bool, optional
            Whether to display labels on the nodes (default is False).
        visualize_periodic_bonds : bool, optional
            Whether to visualize periodic boundary condition edges (default is True).
        save_path : str or pathlib.Path, optional
            The file path to save the plot image. If None, the plot will be displayed interactively.
        """
        pass

    @abstractmethod
    def add_nitrogen_doping(self, *args, **kwargs):
        """
        Abstract method for adding nitrogen doping.

        Accepts any arguments and keyword arguments to allow flexibility in subclasses.
        """
        pass

    def translate(self, x_shift: float = 0.0, y_shift: float = 0.0, z_shift: float = 0.0):
        """
        Translate the structure by shifting all atom positions in the x, y, and z directions.

        Parameters
        ----------
        x_shift : float, optional
            The amount to shift in the x direction. Default is 0.0.
        y_shift : float, optional
            The amount to shift in the y direction. Default is 0.0.
        z_shift : float, optional
            The amount to shift in the z direction. Default is 0.0.
        """
        for node, data in self.graph.nodes(data=True):
            position = data["position"]
            new_position = Position(position.x + x_shift, position.y + y_shift, position.z + z_shift)
            self.graph.nodes[node]["position"] = new_position


# Abstract base class for 2D structures
class Structure2D(MaterialStructure):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def build_structure(self):
        pass

    def plot_structure(
        self, with_labels: bool = False, visualize_periodic_bonds: bool = True, save_path: Union[str, Path, None] = None
    ):
        """
        Plot the structure using networkx and matplotlib in 2D.

        Parameters
        ----------
        with_labels : bool, optional
            Whether to display labels on the nodes (default is False).
        visualize_periodic_bonds : bool, optional
            Whether to visualize periodic boundary condition edges (default is True).
        save_path : str or pathlib.Path, optional
            The file path to save the plot image. If None, the plot will be displayed interactively.


        Notes
        -----
        This method visualizes the sheet structure, optionally with labels indicating the
        element type and node ID. Nodes are colored based on their element type and nitrogen species.
        Periodic boundary condition edges are shown with dashed lines if visualize_periodic_bonds is True.
        """
        # Get positions and elements of nodes, using only x and y for 2D plotting
        pos_2d = {node: (pos[0], pos[1]) for node, pos in nx.get_node_attributes(self.graph, "position").items()}
        elements = nx.get_node_attributes(self.graph, "element")

        # Determine colors for nodes, considering nitrogen species if present
        colors = [
            get_color(elements[node], self.graph.nodes[node].get("nitrogen_species")) for node in self.graph.nodes()
        ]

        # Separate periodic edges and regular edges
        regular_edges = [(u, v) for u, v, d in self.graph.edges(data=True) if not d.get("periodic")]
        periodic_edges = [(u, v) for u, v, d in self.graph.edges(data=True) if d.get("periodic")]

        # Calculate the range of the structure (to scale node/edge size accordingly)
        x_min, x_max = min(x for x, y in pos_2d.values()), max(x for x, y in pos_2d.values())
        y_min, y_max = min(y for x, y in pos_2d.values()), max(y for x, y in pos_2d.values())

        x_range = x_max - x_min
        y_range = y_max - y_min
        grid_size = max(x_range, y_range)

        # Scale node and edge sizes based on grid size (relative to a baseline for 15x15)
        base_node_size = 800
        base_edge_width = 5.0
        base_grid_size = 15

        scaling_factor = base_grid_size / grid_size  # Scaling factor to adjust node/edge size
        node_size_scaled = base_node_size * scaling_factor  # Scaled node size
        edge_width_scaled = base_edge_width * scaling_factor  # Scaled edge width

        # Dynamically set figsize based on grid size
        fig_size_scaled = max(10, grid_size / 2)

        # Initialize plot with dynamically scaled figsize
        fig, ax = plt.subplots(figsize=(fig_size_scaled, fig_size_scaled))

        # Draw regular edges and nodes with scaled sizes
        nx.draw(
            self.graph,
            pos_2d,
            edgelist=regular_edges,
            node_color=colors,
            node_size=node_size_scaled,
            with_labels=False,
            edge_color="gray",
            width=edge_width_scaled,
            ax=ax,
        )

        # Draw periodic edges with dashed lines if visualize_periodic_bonds is True
        if visualize_periodic_bonds:
            nx.draw_networkx_edges(
                self.graph,
                pos_2d,
                edgelist=periodic_edges,
                style="dashed",
                edge_color="gray",
                width=edge_width_scaled,
                ax=ax,
            )

        # Add legend
        unique_colors = set(colors)
        legend_elements = []
        for species in NitrogenSpecies:
            color = get_color("N", species)
            if color in unique_colors:
                legend_elements.append(
                    plt.Line2D(
                        [0], [0], marker="o", color="w", label=species.value, markersize=15, markerfacecolor=color
                    )
                )
        if legend_elements:
            ax.legend(handles=legend_elements, title="Nitrogen Doping Species", fontsize=12, title_fontsize=14)

        # Add labels if specified
        if with_labels:
            labels = {node: f"{elements[node]}{node}" for node in self.graph.nodes()}
            nx.draw_networkx_labels(
                self.graph, pos_2d, labels=labels, font_size=14, font_color="black", font_weight="bold", ax=ax
            )

        # Set equal scaling for the axes to avoid distortion
        ax.set_aspect("equal", "box")  # Ensure both X and Y scales are equal

        # Manually add x- and y-axis labels using ax.text
        ax.text((x_min + x_max) / 2, y_min - (y_max - y_min) * 0.1, "X [Å]", fontsize=14, ha="center")
        ax.text(
            x_min - (x_max - x_min) * 0.1, (y_min + y_max) / 2, "Y [Å]", fontsize=14, va="center", rotation="vertical"
        )

        # Adjust layout to make sure everything fits
        plt.tight_layout()

        if save_path:
            # Ensure save_path is a Path object
            if isinstance(save_path, str):
                save_path = Path(save_path)
            elif not isinstance(save_path, Path):
                raise TypeError("save_path must be a str or pathlib.Path")

            # Create parent directories if they don't exist
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Save the plot to the specified path
            plt.savefig(save_path, bbox_inches="tight")
            plt.close(fig)
        else:
            # Show the plot
            plt.show()


# Abstract base class for 3D structures
class Structure3D(MaterialStructure):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def build_structure(self):
        pass

    def plot_structure(
        self, with_labels: bool = False, visualize_periodic_bonds: bool = True, save_path: Union[str, Path, None] = None
    ):
        """
        Plot the structure in 3D using networkx and matplotlib.

        Parameters
        ----------
        with_labels : bool, optional
            Whether to display labels on the nodes (default is False).
        visualize_periodic_bonds : bool, optional
            Whether to visualize periodic boundary condition edges (default is True).
        save_path : str or pathlib.Path, optional
            The file path to save the plot image. If None, the plot will be displayed interactively.

        Notes
        -----
        This method visualizes the 3D structure, optionally with labels indicating the element type and node ID. Nodes
        are colored based on their element type and nitrogen species.
        Periodic boundary condition edges are shown with dashed lines if visualize_periodic_bonds is True.
        """

        # Get positions and elements of nodes
        pos = nx.get_node_attributes(self.graph, "position")
        elements = nx.get_node_attributes(self.graph, "element")

        # Determine colors for nodes, considering nitrogen species if present
        colors = [
            get_color(elements[node], self.graph.nodes[node].get("nitrogen_species")) for node in self.graph.nodes()
        ]

        # Separate periodic edges and regular edges
        regular_edges = [(u, v) for u, v, d in self.graph.edges(data=True) if not d.get("periodic")]
        periodic_edges = [(u, v) for u, v, d in self.graph.edges(data=True) if d.get("periodic")]

        # Extract node positions
        xs, ys, zs = zip(*[pos[node] for node in self.graph.nodes()])

        # Dynamic scaling based on structure size
        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)
        z_range = max(zs) - min(zs)
        grid_size = max(x_range, y_range, z_range)

        base_node_size = 200  # Reduced node size for 3D visualization
        base_edge_width = 2.0  # Reduced edge width for better visibility
        base_grid_size = 15  # Reference grid size

        scaling_factor = base_grid_size / grid_size
        node_size_scaled = base_node_size * scaling_factor
        edge_width_scaled = base_edge_width * scaling_factor

        # Dynamically set figsize based on grid size
        fig_size_scaled = max(10, grid_size / 2)

        # Initialize 3D plot
        fig = plt.figure(figsize=(fig_size_scaled, fig_size_scaled))
        ax = fig.add_subplot(111, projection="3d")

        # Draw nodes with reduced size
        ax.scatter(xs, ys, zs, color=colors, s=node_size_scaled)

        # Draw regular edges
        if regular_edges:
            regular_segments = np.array(
                [[(pos[u][0], pos[u][1], pos[u][2]), (pos[v][0], pos[v][1], pos[v][2])] for u, v in regular_edges]
            )
            regular_lines = Line3DCollection(regular_segments, colors="gray", linewidths=edge_width_scaled)
            ax.add_collection3d(regular_lines)

        # Draw periodic edges if visualize_periodic_bonds is True
        if visualize_periodic_bonds and periodic_edges:
            periodic_segments = np.array(
                [[(pos[u][0], pos[u][1], pos[u][2]), (pos[v][0], pos[v][1], pos[v][2])] for u, v in periodic_edges]
            )
            periodic_lines = Line3DCollection(
                periodic_segments, colors="gray", linestyles="dashed", linewidths=edge_width_scaled
            )
            ax.add_collection3d(periodic_lines)

        # Calculate the range for each axis
        max_range = np.array([x_range, y_range, z_range]).max() / 2.0

        # Calculate midpoints
        mid_x = (max(xs) + min(xs)) * 0.5
        mid_y = (max(ys) + min(ys)) * 0.5
        mid_z = (max(zs) + min(zs)) * 0.5

        # Set the limits for each axis to ensure equal scaling
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Add labels if specified and position them above the nodes to avoid overlap
        if with_labels:
            for node in self.graph.nodes():
                ax.text(
                    pos[node][0],
                    pos[node][1],
                    pos[node][2] + 0.1,  # Offset the labels to avoid overlap
                    f"{elements[node]}{node}",
                    color="black",
                    fontsize=10,
                )

        # Set the axes labels
        ax.set_xlabel("X [Å]", fontsize=12)
        ax.set_ylabel("Y [Å]", fontsize=12)
        ax.set_zlabel("Z [Å]", fontsize=12)

        # Add a legend for the nitrogen species
        unique_colors = set(colors)
        legend_elements = []
        for species in NitrogenSpecies:
            color = get_color("N", species)
            if color in unique_colors:
                legend_elements.append(
                    plt.Line2D(
                        [0], [0], marker="o", color="w", label=species.value, markersize=15, markerfacecolor=color
                    )
                )
        if legend_elements:
            ax.legend(handles=legend_elements, title="Nitrogen Doping Species", fontsize=10, title_fontsize=12)

        if save_path:
            # Ensure save_path is a Path object
            if isinstance(save_path, str):
                save_path = Path(save_path)
            elif not isinstance(save_path, Path):
                raise TypeError("save_path must be a str or pathlib.Path")

            # Create parent directories if they don't exist
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Save the plot to the specified path
            plt.savefig(save_path, bbox_inches="tight")
            plt.close(fig)
        else:
            # Show the plot
            plt.show()


class GrapheneSheet(Structure2D):
    """
    Represents a graphene sheet structure.
    """

    def __init__(self, bond_length: Union[float, int], sheet_size: Union[Tuple[float, float], Tuple[int, int]]):
        """
        Initialize the GrapheneGraph with given bond distance, sheet size, and whether to adjust positions after doping.

        Parameters
        ----------
        bond_length : Union[float, int]
            The bond distance between carbon atoms in the graphene sheet.
        sheet_size : Optional[Tuple[float, float], Tuple[int, int]]
            The size of the graphene sheet in the x and y directions.

        Raises
        ------
        TypeError
            If the types of `bond_distance` or `sheet_size` are incorrect.
        ValueError
            If `bond_distance` or any element of `sheet_size` is non-positive.
        """
        super().__init__()

        # # Perform validations
        # self._validate_bond_distance(bond_distance)
        # self._validate_sheet_size(sheet_size)

        self.c_c_bond_length = self._validate_bond_length(bond_length)
        """The bond distance between carbon atoms in the graphene sheet."""
        self.c_c_bond_angle = 120
        """The bond angle between carbon atoms in the graphene sheet."""
        self.sheet_size = self.validate_sheet_size(sheet_size)
        """The size of the graphene sheet in the x and y directions."""
        self.positions_adjusted = False
        """Flag to track if positions have been adjusted."""

        # Build the initial graphene sheet structure
        self.build_structure()

    @property
    def cc_x_distance(self):
        """Calculate the distance between atoms in the x direction."""
        return self.c_c_bond_length * sin(pi / 6)

    @property
    def cc_y_distance(self):
        """Calculate the distance between atoms in the y direction."""
        return self.c_c_bond_length * cos(pi / 6)

    @property
    def num_cells_x(self):
        """Calculate the number of unit cells in the x direction based on sheet size and bond distance."""
        return int(self.sheet_size[0] // (2 * self.c_c_bond_length + 2 * self.cc_x_distance))

    @property
    def num_cells_y(self):
        """Calculate the number of unit cells in the y direction based on sheet size and bond distance."""
        return int(self.sheet_size[1] // (2 * self.cc_y_distance))

    @property
    def actual_sheet_width(self):
        """Calculate the actual width of the graphene sheet based on the number of unit cells and bond distance."""
        return self.num_cells_x * (2 * self.c_c_bond_length + 2 * self.cc_x_distance) - self.c_c_bond_length

    @property
    def actual_sheet_height(self):
        """Calculate the actual height of the graphene sheet based on the number of unit cells and bond distance."""
        return self.num_cells_y * (2 * self.cc_y_distance) - self.cc_y_distance

    @staticmethod
    def validate_sheet_size(sheet_size: Tuple[float, float]) -> tuple[float, ...]:
        """
        Validate the sheet_size parameter.

        Parameters
        ----------
        sheet_size : Tuple[float, float]
            The size of the sheet in the x and y dimensions.

        Returns
        -------
        Tuple[float, float]
            The validated sheet_size with elements converted to floats.

        Raises
        ------
        TypeError
            If sheet_size is not a tuple of two numbers.
        ValueError
            If any element of sheet_size is non-positive.
        """
        if not isinstance(sheet_size, tuple):
            raise TypeError("sheet_size must be a tuple of exactly two positive floats or ints.")
        if len(sheet_size) != 2:
            raise TypeError("sheet_size must be a tuple of exactly two positive floats or ints.")
        if not all(isinstance(i, (int, float)) for i in sheet_size):
            raise TypeError("sheet_size must be a tuple of exactly two positive floats or ints.")
        if any(s <= 0 for s in sheet_size):
            raise ValueError(f"All elements of sheet_size must be positive, but got {sheet_size}.")

        # Return the validated sheet_size, converting elements to float
        return tuple(float(dim) for dim in sheet_size)

    def _validate_structure(self):
        """Validate the structure to ensure it can fit within the given sheet size."""
        if self.num_cells_x < 1 or self.num_cells_y < 1:
            raise ValueError(
                f"Sheet size is too small to fit even a single unit cell. Got sheet size {self.sheet_size}."
            )

    def build_structure(self):
        """
        Build the graphene sheet structure.
        """
        self._validate_structure()
        self._build_graphene_sheet()

    def _build_graphene_sheet(self):
        """
        Build the graphene sheet structure by creating nodes and edges (using graph theory via networkx).

        This method iterates over the entire sheet, adding nodes and edges for each unit cell.
        It also connects adjacent unit cells and adds periodic boundary conditions.
        """

        index = 0
        for y in range(self.num_cells_y):
            for x in range(self.num_cells_x):
                x_offset = x * (2 * self.c_c_bond_length + 2 * self.cc_x_distance)
                y_offset = y * (2 * self.cc_y_distance)

                # Add nodes and edges for the unit cell
                self._add_unit_cell(index, x_offset, y_offset)

                # Add horizontal bonds between adjacent unit cells
                if x > 0:
                    self.graph.add_edge(index - 1, index, bond_length=self.c_c_bond_length)

                # Add vertical bonds between unit cells in adjacent rows
                if y > 0:
                    self.graph.add_edge(index - 4 * self.num_cells_x + 1, index, bond_length=self.c_c_bond_length)
                    self.graph.add_edge(index - 4 * self.num_cells_x + 2, index + 3, bond_length=self.c_c_bond_length)

                index += 4

        # Add periodic boundary conditions
        self._add_periodic_boundaries()

    def _add_unit_cell(self, index: int, x_offset: float, y_offset: float):
        """
        Add nodes and internal bonds within a unit cell.

        Parameters
        ----------
        index : int
            The starting index for the nodes in the unit cell.
        x_offset : float
            The x-coordinate offset for the unit cell.
        y_offset : float
            The y-coordinate offset for the unit cell.
        """

        # Define relative positions of atoms within the unit cell
        unit_cell_positions = [
            create_position(x_offset, y_offset),
            create_position(x_offset + self.cc_x_distance, y_offset + self.cc_y_distance),
            create_position(x_offset + self.cc_x_distance + self.c_c_bond_length, y_offset + self.cc_y_distance),
            create_position(x_offset + 2 * self.cc_x_distance + self.c_c_bond_length, y_offset),
        ]

        # Add nodes with positions, element type (carbon) and possible doping site flag
        nodes = [
            (index + i, {"element": "C", "position": pos, "possible_doping_site": True})
            for i, pos in enumerate(unit_cell_positions)
        ]
        self.graph.add_nodes_from(nodes)

        # Add internal bonds within the unit cell
        edges = [
            (index + i, index + i + 1, {"bond_length": self.c_c_bond_length})
            for i in range(len(unit_cell_positions) - 1)
        ]
        self.graph.add_edges_from(edges)

    def _add_periodic_boundaries(self):
        """
        Add periodic boundary conditions to the graphene sheet.

        This method connects the edges of the sheet to simulate an infinite sheet.
        """

        num_nodes_x = self.num_cells_x * 4

        # Generate base indices for horizontal boundaries
        base_indices_y = np.arange(self.num_cells_y) * num_nodes_x
        right_edge_indices = base_indices_y + (self.num_cells_x - 1) * 4 + 3
        left_edge_indices = base_indices_y

        # Add horizontal periodic boundary conditions
        self.graph.add_edges_from(
            zip(right_edge_indices, left_edge_indices), bond_length=self.c_c_bond_length, periodic=True
        )

        # Generate base indices for vertical boundaries
        top_left_indices = np.arange(self.num_cells_x) * 4
        bottom_left_indices = top_left_indices + (self.num_cells_y - 1) * num_nodes_x + 1
        bottom_right_indices = top_left_indices + (self.num_cells_y - 1) * num_nodes_x + 2

        # Add vertical periodic boundary conditions
        self.graph.add_edges_from(
            zip(bottom_left_indices, top_left_indices), bond_length=self.c_c_bond_length, periodic=True
        )
        self.graph.add_edges_from(
            zip(bottom_right_indices, top_left_indices + 3), bond_length=self.c_c_bond_length, periodic=True
        )

    def add_nitrogen_doping(
        self,
        total_percentage: float = None,
        percentages: dict = None,
        adjust_positions: bool = False,
        optimization_config: Optional["OptimizationConfig"] = None,
    ):
        """
        Add nitrogen doping to the graphene sheet.

        This method replaces a specified percentage of carbon atoms with nitrogen atoms in the graphene sheet.
        If specific percentages for different nitrogen species are provided, it ensures the sum does not exceed the
        total percentage. The remaining percentage is distributed equally among the available nitrogen species.

        Parameters
        ----------
        total_percentage : float, optional
            The total percentage of carbon atoms to replace with nitrogen atoms. Default is 10 if not specified.
        percentages : dict, optional
            A dictionary specifying the percentages for each nitrogen species. Keys should be NitrogenSpecies enum
            values and values should be the percentages for the corresponding species.
        adjust_positions : bool, optional
            Whether to adjust the positions of atoms after doping. Default is False.
        optimization_config : OptimizationConfig, optional
            Configuration containing optimization constants for adjusting atom positions. If None, default values are
            used.
            **Note**: This parameter only takes effect if `adjust_positions=True`.

        Raises
        ------
        ValueError
            If the specific percentages exceed the total percentage.
        UserWarning
            If `adjust_positions` is `False` but `optimization_config` is provided.

        Notes
        -----
        - If no total percentage is provided, a default of 10% is used.
        - If specific percentages are provided and their sum exceeds the total percentage, a ValueError is raised.
        - Remaining percentages are distributed equally among the available nitrogen species.
        - Nitrogen species are added in a predefined order: PYRIDINIC_4, PYRIDINIC_3, PYRIDINIC_2, PYRIDINIC_1,
          GRAPHITIC.
        - `optimization_config` is only considered if `adjust_positions` is set to True.
        """
        # Delegate the doping process to the doping handler
        self.doping_handler.add_nitrogen_doping(total_percentage, percentages)

        # Reset the positions_adjusted flag since the structure has changed
        self.positions_adjusted = False

        # Check if optimization_config is provided but adjust_positions is False
        if not adjust_positions and optimization_config is not None:
            warnings.warn(
                "An 'optimization_config' was provided, but 'adjust_positions' is False. "
                "The 'optimization_config' will have no effect. "
                "Set 'adjust_positions=True' to adjust atom positions or call 'adjust_atom_positions()' separately.",
                UserWarning,
            )

        # Adjust atom positions if specified
        if adjust_positions:
            if self.positions_adjusted:
                warnings.warn("Positions have already been adjusted.", UserWarning)
            else:
                if optimization_config is None:
                    optimization_config: "OptimizationConfig" = OptimizationConfig()
                print("\nThe positions of the atoms are now being adjusted. This may take a moment...\n")
                self.adjust_atom_positions(optimization_config=optimization_config)
                print("\nThe positions of the atoms have been adjusted.")
        else:
            print(
                "\nNo position adjustment is being performed. Doping has been applied structurally only.\n"
                "If structural optimization is required to adjust the positions, the 'adjust_positions' flag must be "
                "set to True in the 'add_nitrogen_doping' method or call 'adjust_atom_positions()' separately."
            )

    def adjust_atom_positions(self, optimization_config: Optional["OptimizationConfig"] = None):
        """
        Adjust the positions of atoms in the graphene sheet to optimize the structure including doping while minimizing
        the structural strain.

        Parameters
        ----------
        optimization_config : OptimizationConfig, optional
            Configuration containing optimization constants (the spring constants) for position adjustment. If None,
            default values are used.

        Notes
        -----
        This method uses optimization algorithms to adjust the positions of the atoms in the graphene sheet,
        aiming to achieve target bond lengths and angles, especially after doping has been applied.

        Warnings
        --------
        If positions have already been adjusted, calling this method again will have no effect.
        """
        if self.positions_adjusted:
            warnings.warn("Positions have already been adjusted.", UserWarning)
            return

        if optimization_config is None:
            optimization_config: "OptimizationConfig" = OptimizationConfig()

        # Existing code for position adjustment
        optimizer: "StructureOptimizer" = StructureOptimizer(self, optimization_config)
        optimizer.optimize_positions()

        # Set the flag to indicate positions have been adjusted
        self.positions_adjusted = True

    def _adjust_atom_positions_old(self):
        """
        Adjust the positions of atoms in the graphene sheet to optimize the structure including doping (minimize
        structural strain).

        Notes
        -----
        This method adjusts the positions of atoms in a graphene sheet to optimize the structure based on the doping
        configuration. It uses a combination of bond and angle energies to minimize the total energy of the system.
        """

        # Get all doping structures except graphitic nitrogen (graphitic nitrogen does not affect the structure)
        all_structures = [
            structure
            for structure in self.doping_handler.doping_structures.structures
            if structure.species != NitrogenSpecies.GRAPHITIC
        ]

        # Return if no doping structures are present
        if not all_structures:
            return

        # Get the initial positions of atoms
        positions = {node: self.graph.nodes[node]["position"] for node in self.graph.nodes}
        # Flatten the positions into a 1D array for optimization
        x0 = np.array([coord for node in self.graph.nodes for coord in [positions[node][0], positions[node][1]]])
        # Define the box size for minimum image distance calculation
        box_size = (self.actual_sheet_width + self.c_c_bond_length, self.actual_sheet_height + self.cc_y_distance)

        def bond_strain(x):
            """
            Calculate the bond strain for the given atom positions.

            Parameters
            ----------
            x : ndarray
                Flattened array of positions of all atoms in the cycle.

            Returns
            -------
            total_strain : float
                The total bond strain in the structure.
            """

            total_bond_strain = 0.0

            # Initialize a set to track edges within cycles
            cycle_edges = set()

            # Collect all edges and their properties
            all_edges_in_order = []
            all_target_bond_lengths = []

            for structure in all_structures:
                # Get the target bond lengths for the specific nitrogen species
                properties = self.doping_handler.species_properties[structure.species]
                target_bond_lengths = properties.target_bond_lengths_cycle
                # Extract the ordered cycle of the doping structure to get the current bond lengths in order
                ordered_cycle = structure.cycle

                # Get the graph edges in order, including the additional edge in case of Pyridinic_1
                edges_in_order = list(pairwise(ordered_cycle + [ordered_cycle[0]]))
                if structure.species == NitrogenSpecies.PYRIDINIC_1:
                    edges_in_order.append(structure.additional_edge)

                # Extend the lists of edges and target bond lengths
                all_edges_in_order.extend(edges_in_order)
                all_target_bond_lengths.extend(target_bond_lengths)

            # Convert lists of edges and target bond lengths to numpy arrays
            node_indices = np.array(
                [
                    (list(self.graph.nodes).index(node_i), list(self.graph.nodes).index(node_j))
                    for node_i, node_j in all_edges_in_order
                ]
            )
            target_lengths = np.array(all_target_bond_lengths)

            # Extract the x and y coordinates of the nodes referenced by the first indices in node_indices
            positions_i = x[np.ravel(np.column_stack((node_indices[:, 0] * 2, node_indices[:, 0] * 2 + 1)))]

            # Extract the x and y coordinates of the nodes referenced by the second indices in node_indices
            positions_j = x[np.ravel(np.column_stack((node_indices[:, 1] * 2, node_indices[:, 1] * 2 + 1)))]

            # Reshape the flattened array back to a 2D array where each row contains the [x, y] coordinates of a node
            positions_i = positions_i.reshape(-1, 2)
            positions_j = positions_j.reshape(-1, 2)

            # Calculate bond lengths and energy
            current_lengths, _ = minimum_image_distance_vectorized(positions_i, positions_j, box_size)
            total_bond_strain += 0.5 * self.k_inner_bond * np.sum((current_lengths - target_lengths) ** 2)

            # Update bond lengths in the graph
            edge_updates = {
                (node_i, node_j): {"bond_length": current_lengths[idx]}
                for idx, (node_i, node_j) in enumerate(all_edges_in_order)
            }
            nx.set_edge_attributes(self.graph, edge_updates)
            cycle_edges.update((min(node_i, node_j), max(node_i, node_j)) for node_i, node_j in all_edges_in_order)

            # Handle non-cycle edges in a vectorized manner
            non_cycle_edges = [
                (node_i, node_j)
                for node_i, node_j, data in self.graph.edges(data=True)
                if (min(node_i, node_j), max(node_i, node_j)) not in cycle_edges
            ]
            if non_cycle_edges:
                # Convert non-cycle edge node pairs to a numpy array of indices
                node_indices = np.array(
                    [
                        (list(self.graph.nodes).index(node_i), list(self.graph.nodes).index(node_j))
                        for node_i, node_j in non_cycle_edges
                    ]
                )
                # Extract the x and y coordinates of the nodes referenced by the first indices in node_indices
                positions_i = x[np.ravel(np.column_stack((node_indices[:, 0] * 2, node_indices[:, 0] * 2 + 1)))]

                # Extract the x and y coordinates of the nodes referenced by the second indices in node_indices
                positions_j = x[np.ravel(np.column_stack((node_indices[:, 1] * 2, node_indices[:, 1] * 2 + 1)))]

                # Reshape the flattened array back to a 2D array where each row contains the [x, y] coordinates of a
                # node
                positions_i = positions_i.reshape(-1, 2)
                positions_j = positions_j.reshape(-1, 2)

                # Calculate the current bond lengths using the vectorized minimum image distance function
                current_lengths, _ = minimum_image_distance_vectorized(positions_i, positions_j, box_size)
                # Set the target lengths for non-cycle edges to 1.42 (assumed standard bond length)
                target_lengths = np.full(len(current_lengths), 1.42)

                # Calculate the energy contribution from non-cycle bonds
                total_bond_strain += 0.5 * self.k_outer_bond * np.sum((current_lengths - target_lengths) ** 2)

                # Prepare bond length updates for non-cycle edges
                edge_updates = {
                    (node_i, node_j): {"bond_length": current_lengths[idx]}
                    for idx, (node_i, node_j) in enumerate(non_cycle_edges)
                }
                # Update the bond lengths in the graph for non-cycle edges
                nx.set_edge_attributes(self.graph, edge_updates)

            return total_bond_strain

        def angle_strain(x):
            """
            Calculate the angle strain for the given atom positions.

            Parameters
            ----------
            x : ndarray
                Flattened array of positions of all atoms in the cycle.

            Returns
            -------
            total_strain : float
                The total angular strain in the structure.
            """

            total_angle_strain = 0.0

            # Initialize lists to collect all triplets of nodes and their target angles
            all_triplets = []
            all_target_angles = []

            # Initialize a set to track angles within cycles
            counted_angles = set()  # ToDo: Delete later if outer angles are not needed

            # Iterate over all doping structures to gather triplets and target angles
            for structure in all_structures:
                properties = self.doping_handler.species_properties[structure.species]
                target_angles = properties.target_angles_cycle
                ordered_cycle = structure.cycle

                # Extend the cycle to account for the closed loop by adding the first two nodes at the end
                extended_cycle = ordered_cycle + [ordered_cycle[0], ordered_cycle[1]]

                # Collect node triplets (i, j, k) for angle energy calculations
                triplets = [
                    (list(self.graph.nodes).index(i), list(self.graph.nodes).index(j), list(self.graph.nodes).index(k))
                    for i, j, k in zip(extended_cycle, extended_cycle[1:], extended_cycle[2:])
                ]
                all_triplets.extend(triplets)
                all_target_angles.extend(target_angles)

                if self.include_outer_angles:
                    # Add angles to counted_angles to avoid double-counting
                    for i, j, k in zip(extended_cycle, extended_cycle[1:], extended_cycle[2:]):
                        counted_angles.add((i, j, k))
                        counted_angles.add((k, j, i))

            # Convert lists of triplets and target angles to numpy arrays
            node_indices = np.array(all_triplets)
            target_angles = np.radians(np.array(all_target_angles))

            # Extract the x and y coordinates of the nodes referenced by the first, second, and third indices in
            # node_indices
            positions_i = x[np.ravel(np.column_stack((node_indices[:, 0] * 2, node_indices[:, 0] * 2 + 1)))]
            positions_j = x[np.ravel(np.column_stack((node_indices[:, 1] * 2, node_indices[:, 1] * 2 + 1)))]
            positions_k = x[np.ravel(np.column_stack((node_indices[:, 2] * 2, node_indices[:, 2] * 2 + 1)))]
            # Reshape the flattened arrays back to 2D arrays where each row contains the [x, y] coordinates of a node
            positions_i = positions_i.reshape(-1, 2)
            positions_j = positions_j.reshape(-1, 2)
            positions_k = positions_k.reshape(-1, 2)

            # Calculate vectors v1 (from j to i) and v2 (from j to k)
            _, v1 = minimum_image_distance_vectorized(positions_i, positions_j, box_size)
            _, v2 = minimum_image_distance_vectorized(positions_k, positions_j, box_size)

            # Calculate the cosine of the angle between v1 and v2
            # (cos_theta = dot(v1, v2) / (|v1| * |v2|) element-wise for each pair of vectors)
            cos_theta = np.einsum("ij,ij->i", v1, v2) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))
            # Calculate the angle theta using arccos, ensuring the values are within the valid range for arccos
            theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

            # Calculate the energy contribution from angle deviations and add it to the total energy
            total_angle_strain += 0.5 * self.k_inner_angle * np.sum((theta - target_angles) ** 2)

            if self.include_outer_angles:
                # Calculate angle energy for angles outside the cycles
                for node in self.graph.nodes:
                    neighbors = list(self.graph.neighbors(node))
                    if len(neighbors) < 2:
                        continue
                    for i in range(len(neighbors)):
                        for j in range(i + 1, len(neighbors)):
                            ni = neighbors[i]
                            nj = neighbors[j]

                            # Skip angles that have already been counted
                            if (ni, node, nj) in counted_angles or (nj, node, ni) in counted_angles:
                                continue

                            x_node, y_node = (
                                x[2 * list(self.graph.nodes).index(node)],
                                x[2 * list(self.graph.nodes).index(node) + 1],
                            )
                            x_i, y_i = (
                                x[2 * list(self.graph.nodes).index(ni)],
                                x[2 * list(self.graph.nodes).index(ni) + 1],
                            )
                            x_j, y_j = (
                                x[2 * list(self.graph.nodes).index(nj)],
                                x[2 * list(self.graph.nodes).index(nj) + 1],
                            )
                            pos_node = create_position(x_node, y_node)
                            pos_i = create_position(x_i, y_i)
                            pos_j = create_position(x_j, y_j)
                            if (
                                not isinstance(pos_node, Position)
                                or not isinstance(pos_i, Position)
                                or not isinstance(pos_j, Position)
                            ):
                                raise TypeError("Expected Position, but got a different type")
                            _, v1 = minimum_image_distance(pos_i, pos_node, box_size)
                            _, v2 = minimum_image_distance(pos_j, pos_node, box_size)
                            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                            theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
                            total_angle_strain += (
                                0.5 * self.k_outer_angle * ((theta - np.radians(self.c_c_bond_angle)) ** 2)
                            )

            return total_angle_strain

        def total_strain(x):
            """
            Calculate the total structural strain (bond + angular) for the given positions.

            Parameters
            ----------
            x : ndarray
                Flattened array of positions of all atoms in the cycle.

            Returns
            -------
            total_strain : float
                The total structural strain in the system.
            """

            return bond_strain(x) + angle_strain(x)

        # Initialize the progress bar without a fixed total
        progress_bar = tqdm(total=None, desc="Optimizing positions", unit="iteration")

        def optimization_callback(xk):
            # Update the progress bar by one step
            progress_bar.update(1)

        # Start the optimization process with the callback to update progress
        result = minimize(total_strain, x0, method="L-BFGS-B", callback=optimization_callback)

        # Close the progress bar
        progress_bar.close()

        # Print the number of iterations and final energy
        print(f"\nNumber of iterations: {result.nit}\nFinal structural strain: {result.fun}")

        # Reshape the optimized positions back to the 2D array format
        optimized_positions = result.x.reshape(-1, 2)

        # Update the positions of atoms in the graph with the optimized positions using NetworkX set_node_attributes
        position_dict = {
            node: create_position(optimized_positions[idx][0], optimized_positions[idx][1], positions[node][2])
            for idx, node in enumerate(self.graph.nodes)
        }
        nx.set_node_attributes(self.graph, position_dict, "position")

    def create_hole(self, center: Tuple[float, float], radius: float):
        """
        Removes atoms within a certain radius around a given center using vectorized operations.

        Note: Filtering with a bounding box minimizes the number of distance calculations. Therefore, the method scales
        well with large structures, making it suitable for large graphene sheets.

        Parameters
        ----------
        center : Tuple[float, float]
            The (x, y) coordinates of the center of the hole.
        radius : float
            The radius of the hole.
        """
        # Extract node positions and IDs
        node_positions = nx.get_node_attributes(self.graph, "position")
        node_ids = np.array(list(node_positions.keys()))
        positions = np.array([(pos.x, pos.y) for pos in node_positions.values()])

        # Define the bounding box around the hole
        x0, y0 = center
        r = radius
        within_box = (
            (positions[:, 0] >= x0 - r)
            & (positions[:, 0] <= x0 + r)
            & (positions[:, 1] >= y0 - r)
            & (positions[:, 1] <= y0 + r)
        )

        # Filter positions and node IDs within the bounding box
        positions_in_box = positions[within_box]
        node_ids_in_box = node_ids[within_box]

        # Compute squared distances to the center
        dx = positions_in_box[:, 0] - x0
        dy = positions_in_box[:, 1] - y0
        distances_squared = dx**2 + dy**2

        # Identify nodes within the circle (hole)
        within_circle = distances_squared <= r**2
        nodes_to_remove = node_ids_in_box[within_circle]

        # Remove nodes from the graph
        self.graph.remove_nodes_from(nodes_to_remove.tolist())

        # Update possible doping positions
        self.doping_handler.mark_possible_carbon_atoms_for_update()

    def stack(
        self, interlayer_spacing: float = 3.35, number_of_layers: int = 3, stacking_type: str = "ABA"
    ) -> "StackedGraphene":
        """
        Stack graphene sheets using ABA or ABC stacking.

        Parameters
        ----------
        interlayer_spacing : float, optional
            The shift in the z-direction for each layer. Default is 3.35 Å.
        number_of_layers : int, optional
            The number of layers to stack. Default is 3.
        stacking_type : str, optional
            The type of stacking to use ('ABA' or 'ABC'). Default is 'ABA'.

        Returns
        -------
        StackedGraphene
            The stacked graphene structure.

        Raises
        ------
        ValueError
            If `interlayer_spacing` is non-positive, `number_of_layers` is not a positive integer, or `stacking_type` is
            not 'ABA' or 'ABC'.
        """
        return StackedGraphene(self, interlayer_spacing, number_of_layers, stacking_type)


class StackedGraphene(Structure3D):
    """
    Represents a stacked graphene structure.
    """

    def __init__(
        self,
        graphene_sheet: GrapheneSheet,
        interlayer_spacing: float = 3.35,
        number_of_layers: int = 3,
        stacking_type: str = "ABA",
    ):
        """
        Initialize the StackedGraphene with a base graphene sheet, interlayer spacing, number of layers, and stacking
        type.

        Parameters
        ----------
        graphene_sheet : GrapheneSheet
            The base graphene sheet to be stacked.
        interlayer_spacing : float, optional
            The spacing between layers in the z-direction. Default is 3.35 Å.
        number_of_layers : int, optional
            The number of layers to stack. Default is 3.
        stacking_type : str, optional
            The type of stacking to use ('ABA' or 'ABC'). Default is 'ABA'.

        Raises
        ------
        ValueError
            If `interlayer_spacing` is non-positive, `number_of_layers` is not a positive integer, or `stacking_type` is
            not 'ABA' or 'ABC'.
        """
        super().__init__()

        # Validate interlayer_spacing
        if not isinstance(interlayer_spacing, (int, float)) or interlayer_spacing <= 0:
            raise ValueError(f"interlayer_spacing must be positive number, but got {interlayer_spacing}.")

        # Validate number_of_layers
        if not isinstance(number_of_layers, int) or number_of_layers <= 0:
            raise ValueError(f"number_of_layers must be a positive integer, but got {number_of_layers}.")

        # Ensure stacking_type is a string and validate it
        if not isinstance(stacking_type, str):
            raise ValueError(f"stacking_type must be a string, but got {type(stacking_type).__name__}.")

        # Validate stacking_type after converting it to uppercase
        self.stacking_type = stacking_type.upper()
        """The type of stacking to use ('ABA' or 'ABC')."""
        valid_stacking_types = {"ABA", "ABC"}
        if self.stacking_type not in valid_stacking_types:
            raise ValueError(f"stacking_type must be one of {valid_stacking_types}, but got '{self.stacking_type}'.")

        self.graphene_sheets = []
        """A list to hold individual GrapheneSheet instances."""
        self.interlayer_spacing = interlayer_spacing
        """The spacing between layers in the z-direction."""
        self.number_of_layers = number_of_layers
        """The number of layers to stack."""

        # Add the original graphene sheet as the first layer
        # toggle_dimension(graphene_sheet.graph)
        self.graphene_sheets.append(graphene_sheet)

        # Add additional layers by copying the original graphene sheet
        for layer in range(1, self.number_of_layers):
            # Create a copy of the original graphene sheet and shift it
            new_sheet = copy.deepcopy(graphene_sheet)
            self._shift_sheet(new_sheet, layer)
            self.graphene_sheets.append(new_sheet)

        # Build the structure by combining all graphene sheets
        self.build_structure()

    def _shift_sheet(self, sheet: GrapheneSheet, layer: int):
        """
        Shift the graphene sheet by the appropriate interlayer spacing and x-shift for ABA stacking.

        Parameters
        ----------
        sheet : GrapheneSheet
            The graphene sheet to shift.
        layer : int
            The layer number to determine the shifts.
        """
        interlayer_shift = self.graphene_sheets[0].c_c_bond_length  # Fixed x_shift for ABA stacking

        if self.stacking_type == "ABA":
            x_shift = (layer % 2) * interlayer_shift
        elif self.stacking_type == "ABC":
            x_shift = (layer % 3) * interlayer_shift
        else:
            raise ValueError(f"Unsupported stacking type: {self.stacking_type}. Please use 'ABA' or 'ABC'.")

        z_shift = layer * self.interlayer_spacing

        # Update the positions in the copied sheet
        for node, pos in sheet.graph.nodes(data="position"):
            shifted_pos = Position(pos.x + x_shift, pos.y, z_shift)
            sheet.graph.nodes[node]["position"] = shifted_pos

    def build_structure(self):
        """
        Combine all the graphene sheets into a single structure.
        """
        # Start with the graph of the first layer
        self.graph = self.graphene_sheets[0].graph.copy()

        # Iterate over the remaining layers and combine them into self.graph
        for sheet in self.graphene_sheets[1:]:
            self.graph = nx.disjoint_union(self.graph, sheet.graph)

    def add_nitrogen_doping(
        self,
        total_percentage: float = None,
        percentages: dict = None,
        adjust_positions: bool = False,
        layers: Union[List[int], str] = "all",
        optimization_config: Optional["OptimizationConfig"] = None,
    ):
        """
        Add nitrogen doping to one or multiple layers in the stacked graphene structure.

        Parameters
        ----------
        total_percentage : float, optional
            The total percentage of carbon atoms to replace with nitrogen atoms. Default is 10 if not specified.
        percentages : dict, optional
            A dictionary specifying the percentages for each nitrogen species. Keys should be NitrogenSpecies enum
            values and values should be the percentages for the corresponding species.
        adjust_positions : bool, optional
            Whether to adjust the positions of atoms after doping. Default is False.
        layers : Union[List[int], str], optional
            The layers to apply doping to. Can be a list of layer indices or "all" to apply to all layers. Default is
            "all".
        optimization_config : OptimizationConfig, optional
            Configuration containing optimization constants for adjusting atom positions. If None, default values are
            used.
            **Note**: This parameter only takes effect if `adjust_positions=True`.

        Raises
        ------
        IndexError
            If any of the specified layers are out of range.
        UserWarning
            If `adjust_positions` is `False` but `optimization_config` is provided.

        Notes
        -----
        - After doping, positions may be adjusted by setting `adjust_positions=True` or by calling
        `adjust_atom_positions()`.
        """
        # Determine which layers to dope
        if isinstance(layers, str) and layers.lower() == "all":
            layers = list(range(self.number_of_layers))
        elif isinstance(layers, list):
            if any(layer < 0 or layer >= self.number_of_layers for layer in layers):
                raise IndexError("One or more specified layers are out of range.")
        else:
            raise ValueError("Invalid 'layers' parameter. Must be a list of integers or 'all'.")

        if optimization_config is None and adjust_positions:
            optimization_config: "OptimizationConfig" = OptimizationConfig()

        # Check if optimization_config is provided but adjust_positions is False
        if not adjust_positions and optimization_config is not None:
            warnings.warn(
                "An 'optimization_config' was provided, but 'adjust_positions' is False. "
                "The 'optimization_config' will have no effect. "
                "Set 'adjust_positions=True' to adjust atom positions or call 'adjust_atom_positions()' "
                "separately.",
                UserWarning,
            )

        # Apply doping to each specified layer
        for layer_index in layers:
            self.add_nitrogen_doping_to_layer(
                layer_index=layer_index,
                total_percentage=total_percentage,
                percentages=percentages,
                adjust_positions=adjust_positions,
                optimization_config=optimization_config,
            )

    def add_nitrogen_doping_to_layer(
        self,
        layer_index: int,
        total_percentage: float = None,
        percentages: dict = None,
        adjust_positions: bool = False,
        optimization_config: Optional["OptimizationConfig"] = None,
    ):
        """
        Add nitrogen doping to a specific layer in the stacked graphene structure.

        Parameters
        ----------
        layer_index : int
            The index of the layer to dope.
        total_percentage : float, optional
            The total percentage of carbon atoms to replace with nitrogen atoms. Default is 10 if not specified.
        percentages : dict, optional
            A dictionary specifying the percentages for each nitrogen species. Keys should be NitrogenSpecies enum
            values and values should be the percentages for the corresponding species.
        adjust_positions : bool, optional
            Whether to adjust the positions of atoms after doping. Default is False.
        optimization_config : OptimizationConfig, optional
            Configuration containing optimization constants for adjusting atom positions. If None, default values are
            used.
            **Note**: This parameter only takes effect if `adjust_positions=True`.

        Raises
        ------
        UserWarning
            If `adjust_positions` is `False` but `optimization_config` is provided.
        """
        if 0 <= layer_index < len(self.graphene_sheets):

            if optimization_config is None and adjust_positions:
                optimization_config: "OptimizationConfig" = OptimizationConfig()

            # Perform the doping
            self.graphene_sheets[layer_index].add_nitrogen_doping(
                total_percentage=total_percentage,
                percentages=percentages,
                adjust_positions=adjust_positions,
                optimization_config=optimization_config,
            )

            # Rebuild the main graph in order to update the structure after doping
            self.build_structure()
        else:
            raise IndexError("Layer index out of range.")

    def adjust_atom_positions(
        self, layers: Union[List[int], str] = "all", optimization_config: Optional["OptimizationConfig"] = None
    ):
        """
        Adjust the positions of atoms in specified layers of the stacked graphene structure to optimize the structure.

        Parameters
        ----------
        layers : Union[List[int], str], optional
            The layers to adjust positions for. Can be a list of layer indices or "all".
        optimization_config : OptimizationConfig, optional
            Configuration containing optimization constants for position adjustments. If None, default values are used.

        Notes
        -----
        Positions will only be adjusted for layers where positions have not already been adjusted.

        Warnings
        --------
        If positions have already been adjusted in a layer, a warning will be issued.
        """
        # Determine which layers to adjust
        if isinstance(layers, str) and layers.lower() == "all":
            layers = list(range(self.number_of_layers))
        elif isinstance(layers, list):
            if any(layer < 0 or layer >= self.number_of_layers for layer in layers):
                raise IndexError("One or more specified layers are out of range.")
        else:
            raise ValueError("Invalid 'layers' parameter. Must be a list of integers or 'all'.")

        if optimization_config is None:
            optimization_config: "OptimizationConfig" = OptimizationConfig()

        for layer_index in layers:
            sheet = self.graphene_sheets[layer_index]
            sheet.adjust_atom_positions(optimization_config=optimization_config)

        # Rebuild the main graph to reflect updated positions
        self.build_structure()


class CNT(Structure3D):
    """
    Represents a carbon nanotube structure.
    """

    _warning_shown = False  # Class-level attribute to prevent repeated warnings

    def __init__(
        self,
        bond_length: float,
        tube_length: float,
        tube_size: Optional[int] = None,
        tube_diameter: Optional[float] = None,
        conformation: str = "zigzag",
        periodic: bool = False,
    ):
        """
        Initialize the CarbonNanotube with given parameters.

        Parameters
        ----------
        bond_length : float
            The bond length between carbon atoms in the CNT.
        tube_length : float
            The length of the CNT.
        tube_size : int, optional
            The size of the CNT, i.e., the number of hexagonal units around the circumference.
        tube_diameter: float, optional
            The diameter of the CNT.
        conformation : str, optional
            The conformation of the CNT ('armchair' or 'zigzag'). Default is 'zigzag'.
        periodic : bool, optional
            Whether to apply periodic boundary conditions along the tube axis (default is False).

        Raises
        ------
        ValueError
            If neither tube_size nor tube_diameter is specified.
            If both tube_size and tube_diameter are specified.
            If the conformation is invalid.

        Notes
        -----
        - You must specify either `tube_size` or `tube_diameter`, but not both.
          These parameters are internally converted using standard formulas.
        """
        super().__init__()

        # Input validation and parameter handling
        if tube_size is None and tube_diameter is None:
            raise ValueError("You must specify either 'tube_size' or 'tube_diameter'.")

        if tube_size is not None and tube_diameter is not None:
            raise ValueError("Specify only one of 'tube_size' or 'tube_diameter', not both.")

        self.bond_length = self._validate_bond_length(bond_length)
        self.tube_length = self.validate_tube_length(tube_length)
        self.conformation = self.validate_conformation(conformation)
        self.periodic = self._validate_periodic(periodic)

        # Use 'tube_size' as internal parameter
        if tube_size is not None:
            self.tube_size = self.validate_tube_size(tube_size)
            # `tube_diameter` will be calculated based on `tube_size`
        elif tube_diameter is not None:
            # Calculate `tube_size` based on `tube_diameter`
            self.tube_size = self._calculate_tube_size_from_diameter(tube_diameter)
            # `tube_diameter` will be calculated based on `tube_size`
        self._validate_diameter_from_size(tube_diameter)

        # Build the CNT structure using graph theory
        self.build_structure()

    @property
    def actual_length(self) -> float:
        """
        Calculate the actual length of the CNT by finding the difference between the maximum and minimum z-coordinates.

        Returns
        -------
        float
            The actual length of the CNT in the z direction.
        """
        # Get the z-coordinates of all nodes in the CNT
        z_coordinates = [pos.z for node, pos in nx.get_node_attributes(self.graph, "position").items()]

        # Calculate the difference between the maximum and minimum z-values
        return max(z_coordinates) - min(z_coordinates)

    @property
    def actual_tube_diameter(self) -> float:
        """
        Calculate the diameter of the CNT based on the tube size and bond length.

        The formula differs for zigzag and armchair conformations and is taken from the following sources:
            - https://www.sciencedirect.com/science/article/pii/S0020768306000412
            - https://indico.ictp.it/event/7605/session/12/contribution/72/material/1/0.pdf

        Returns
        -------
        float
            The calculated `tube_diameter` based on `tube_size`.
        """
        return self._calculate_tube_diameter_from_size(self.tube_size)

    def _calculate_tube_diameter_from_size(self, tube_size: int) -> float:
        """
        Calculate the diameter of the CNT based on the given `tube_size` and bond length.

        The formula differs for zigzag and armchair conformations and are taken from the following sources:
            - https://www.sciencedirect.com/science/article/pii/S0020768306000412
            - https://indico.ictp.it/event/7605/session/12/contribution/72/material/1/0.pdf

        Parameters
        ----------
        tube_size : int
            The size of the CNT, i.e., the number of hexagonal units around the circumference.

        Returns
        -------
            The calculated `tube_diameter` based on the given `tube_size`
        """
        len_unit_vec = np.sqrt(3) * self.bond_length
        if self.conformation == "armchair":
            return (np.sqrt(3) * len_unit_vec * tube_size) / np.pi
        elif self.conformation == "zigzag":
            return (len_unit_vec * tube_size) / np.pi

    def _calculate_tube_size_from_diameter(self, tube_diameter: float) -> int:
        """
        Calculates 'tube_size' based on the given 'tube_diameter' and bond length.

        The formula differs for zigzag and armchair conformations and are taken from the following sources:
            - https://www.sciencedirect.com/science/article/pii/S0020768306000412
            - https://indico.ictp.it/event/7605/session/12/contribution/72/material/1/0.pdf

        Parameters
        ----------
        tube_diameter : float
            The desired diameter of the CNT.

        Returns
        -------
        int
            The calculated 'tube_size' based on the given 'tube_diameter'.

        Raises
        ------
        ValueError
            If the calculated 'tube_size' is not a positive integer.
        """
        len_unit_vec = np.sqrt(3) * self.bond_length
        if self.conformation == "armchair":
            tube_size = (tube_diameter * np.pi) / (len_unit_vec * np.sqrt(3))
        else:
            tube_size = (tube_diameter * np.pi) / len_unit_vec

        tube_size = int(round(tube_size))
        if tube_size < 1:
            raise ValueError("The calculated `tube_size` needs to be a positive integer.")

        return tube_size

    def _validate_diameter_from_size(self, provided_diameter: Optional[float] = None):
        """
        Validates and compares the calculated diameter with the provided one and handles warnings.

        Parameters
        ----------
        provided_diameter : float, optional
            The provided 'tube_diameter' to compare with the calculated one.
        """
        actual_diameter = self.actual_tube_diameter
        if provided_diameter is not None and not np.isclose(provided_diameter, actual_diameter, atol=1e-3):
            if not CNT._warning_shown:
                print(
                    f"Note: The provided 'tube_diameter' ({provided_diameter:.3f} Å) does not correspond to an integer "
                    f"'tube_size'. Using 'tube_size' = {self.tube_size}, which results in a 'tube_diameter' of "
                    f"{actual_diameter:.3f} Å."
                )
                CNT._warning_shown = True  # Ensure the message is shown only once

    @staticmethod
    def validate_tube_length(tube_length):
        if not isinstance(tube_length, (float, int)):
            raise TypeError("tube_length must be a float or integer.")
        if tube_length <= 0:
            raise ValueError("tube_length must be positive.")
        return float(tube_length)

    @staticmethod
    def validate_tube_size(tube_size):
        if not isinstance(tube_size, int):
            raise TypeError("tube_size must be an integer.")
        if tube_size <= 0:
            raise ValueError("tube_size must be a positive integer.")
        return tube_size

    @staticmethod
    def validate_conformation(conformation):
        if not isinstance(conformation, str):
            raise TypeError("conformation must be a string.")
        conformation = conformation.lower()
        if conformation not in ["armchair", "zigzag"]:
            raise ValueError("Invalid conformation. Choose either 'armchair' or 'zigzag'.")
        return conformation

    @staticmethod
    def _validate_periodic(periodic):
        if not isinstance(periodic, bool):
            raise TypeError("periodic must be a boolean.")
        return periodic

    def build_structure(self):
        """
        Build the CNT structure based on the given parameters.

        Raises
        ------
        ValueError
            If the conformation is not 'armchair' or 'zigzag'.
        """
        # Check if the conformation is valid
        if self.conformation not in ["armchair", "zigzag"]:
            raise ValueError("Invalid conformation. Choose either 'armchair' or 'zigzag'.")

        # Calculate common parameters
        distance = self.bond_length
        hex_d = distance * math.cos(math.radians(30)) * 2
        symmetry_angle = 360 / self.tube_size

        if self.conformation == "armchair":
            # Calculate the positions for the armchair conformation
            positions, z_max = self._calculate_armchair_positions(distance, symmetry_angle)
        else:
            # Calculate the positions for the zigzag conformation
            positions, z_max = self._calculate_zigzag_positions(distance, hex_d, symmetry_angle)

        # Add nodes to the graph
        self._add_nodes_to_graph(positions)

        # Add internal bonds within unit cells
        self._add_internal_bonds(len(positions))

        # Add connections between unit cells
        self._add_unit_cell_connections(positions)

        # Add connections to complete the end of each cycle
        self._complete_cycle_connections(positions)

        # Create connections between different layers of the CNT
        self._connect_layers(positions)

        # Apply periodic boundary conditions if the 'periodic' attribute is True
        if self.periodic:
            self._add_periodic_boundaries(len(positions))

    def _calculate_armchair_positions(
        self, distance: float, symmetry_angle: float
    ) -> Tuple[List[Tuple[float, float, float]], float]:
        """
        Calculate atom positions for the armchair conformation.

        Parameters
        ----------
        distance : float
            The bond length between carbon atoms in the CNT.
        symmetry_angle : float
            The angle between repeating units around the circumference of the tube.

        Returns
        -------
        positions : List[Tuple[float, float, float]]
            A list of atom positions in the format (x, y, z).
        z_max : float
            The maximum z-coordinate reached by the structure.
        """
        # Calculate the angle between carbon bonds in the armchair configuration
        angle_carbon_bond = 360 / (self.tube_size * 3)
        # Calculate the radius of the CNT
        radius = self.actual_tube_diameter / 2
        # Calculate the horizontal distance between atoms along the x-axis within a unit cell
        distx = radius - radius * math.cos(math.radians(angle_carbon_bond / 2))
        # Calculate the vertical distance between atoms along the y-axis within a unit cell
        disty = -radius * math.sin(math.radians(angle_carbon_bond / 2))
        # Calculate the step size along the z-axis (height) between layers of atoms
        zstep = math.sqrt(distance**2 - distx**2 - disty**2)

        # Initialize a list to store all the calculated positions of atoms
        positions = []
        # Initialize the maximum z-coordinate to 0
        z_max = 0
        # Initialize a counter to track the number of layers
        counter = 0

        # Loop until the maximum z-coordinate reaches or exceeds the desired tube length
        while z_max < self.tube_length:
            # Calculate the current z-coordinate for this layer of atoms
            z_coordinate = zstep * 2 * counter

            # Loop over the number of atoms in one layer (tube_size) to calculate their positions
            for i in range(self.tube_size):
                # Calculate and add the positions of atoms in this unit cell to the list
                positions.extend(
                    self._calculate_armchair_unit_cell_positions(
                        i, radius, symmetry_angle, angle_carbon_bond, zstep, z_coordinate
                    )
                )
            # Update the maximum z-coordinate reached by the structure after this layer
            z_max = z_coordinate + zstep
            # Increment the counter to move to the next layer
            counter += 1

        # Return the list of atom positions and the maximum z-coordinate reached
        return positions, z_max

    def _calculate_zigzag_positions(
        self, distance: float, hex_d: float, symmetry_angle: float
    ) -> Tuple[List[Tuple[float, float, float]], float]:
        """
        Calculate atom positions for the zigzag conformation.

        Parameters
        ----------
        distance : float
            The bond length between carbon atoms in the CNT.
        hex_d : float
            The distance between two atoms in a hexagon.
        symmetry_angle : float
            The angle between repeating units around the circumference of the tube.

        Returns
        -------
        positions : List[Tuple[float, float, float]]
            A list of atom positions in the format (x, y, z).
        z_max : float
            The maximum z-coordinate reached by the structure.
        """
        # Calculate the radius of the CNT
        radius = self.actual_tube_diameter / 2
        # Calculate the horizontal distance between atoms along the x-axis within a unit cell
        distx = radius - radius * math.cos(math.radians(symmetry_angle / 2))
        # Calculate the vertical distance between atoms along the y-axis within a unit cell
        disty = -radius * math.sin(math.radians(symmetry_angle / 2))
        # Calculate the step size along the z-axis (height) between layers of atoms
        zstep = math.sqrt(distance**2 - distx**2 - disty**2)

        # Initialize a list to store all the calculated positions of atoms
        positions = []
        # Initialize the maximum z-coordinate to 0
        z_max = 0
        # Initialize a counter to track the number of layers
        counter = 0

        # Loop until the maximum z-coordinate reaches or exceeds the desired tube length
        while z_max < self.tube_length:
            # Calculate the current z-coordinate for this layer of atoms, including the vertical offset
            z_coordinate = (2 * zstep + distance * 2) * counter

            # Loop over the number of cells in one layer (tube_size) to calculate their positions
            for i in range(self.tube_size):
                # Calculate and add the positions of atoms in this unit cell to the list
                positions.extend(
                    self._calculate_zigzag_unit_cell_positions(i, radius, symmetry_angle, zstep, distance, z_coordinate)
                )

            # Update the maximum z-coordinate reached by the structure after this layer
            z_max = z_coordinate + 2 * zstep + distance * 2
            # Increment the counter to move to the next layer
            counter += 1

        # Return the list of atom positions and the maximum z-coordinate reached
        return positions, z_max

    @staticmethod
    def _calculate_armchair_unit_cell_positions(
        i: int, radius: float, symmetry_angle: float, angle_carbon_bond: float, zstep: float, z_coordinate: float
    ) -> List[Tuple[float, float, float]]:
        """
        Calculate the positions of atoms in one armchair unit cell.

        Parameters
        ----------
        i : int
            The index of the unit cell around the circumference.
        radius : float
            The radius of the CNT.
        symmetry_angle : float
            The angle between repeating units around the circumference of the tube.
        angle_carbon_bond : float
            The bond angle between carbon atoms.
        zstep : float
            The step size along the z-axis between layers.
        z_coordinate : float
            The z-coordinate of the current layer.

        Returns
        -------
        positions : List[Tuple[float, float, float]]
            A list of atom positions in the format (x, y, z).
        """
        # Initialize an empty list to store the positions of atoms in this unit cell
        positions: List[Tuple[float, float, float]] = []

        # Calculate the position of the first atom in the unit cell
        angle1 = math.radians(symmetry_angle * i)  # Convert the angle to radians
        x1 = radius * math.cos(angle1)  # Calculate the x-coordinate using the radius and angle
        y1 = radius * math.sin(angle1)  # Calculate the y-coordinate using the radius and angle
        positions.append((x1, y1, z_coordinate))  # Append the (x, y, z) position of the first atom

        # Calculate the position of the second atom in the unit cell
        angle2 = math.radians(symmetry_angle * i + angle_carbon_bond)  # Adjust the angle for the second bond
        x2 = radius * math.cos(angle2)  # Calculate the x-coordinate of the second atom
        y2 = radius * math.sin(angle2)  # Calculate the y-coordinate of the second atom
        positions.append((x2, y2, z_coordinate))  # Append the (x, y, z) position of the second atom

        # Calculate the position of the third atom in the unit cell, which is shifted along the z-axis
        angle3 = math.radians(symmetry_angle * i + angle_carbon_bond * 1.5)  # Adjust the angle for the third bond
        x3 = radius * math.cos(angle3)  # Calculate the x-coordinate of the third atom
        y3 = radius * math.sin(angle3)  # Calculate the y-coordinate of the third atom
        z3 = zstep + z_coordinate  # Add the z-step to the z-coordinate for the third atom
        positions.append((x3, y3, z3))  # Append the (x, y, z) position of the third atom

        # Calculate the position of the fourth atom in the unit cell, also shifted along the z-axis
        angle4 = math.radians(symmetry_angle * i + angle_carbon_bond * 2.5)  # Adjust the angle for the fourth bond
        x4 = radius * math.cos(angle4)  # Calculate the x-coordinate of the fourth atom
        y4 = radius * math.sin(angle4)  # Calculate the y-coordinate of the fourth atom
        z4 = zstep + z_coordinate  # Add the z-step to the z-coordinate for the fourth atom
        positions.append((x4, y4, z4))  # Append the (x, y, z) position of the fourth atom

        # Return the list of atom positions calculated for this unit cell
        return positions

    @staticmethod
    def _calculate_zigzag_unit_cell_positions(
        i: int, radius: float, symmetry_angle: float, zstep: float, distance: float, z_coordinate: float
    ) -> List[Tuple[float, float, float]]:
        """
        Calculate the positions of atoms in one zigzag unit cell.

        Parameters
        ----------
        i : int
            The index of the unit cell around the circumference.
        radius : float
            The radius of the CNT.
        symmetry_angle : float
            The angle between repeating units around the circumference of the tube.
        zstep : float
            The step size along the z-axis between layers.
        distance : float
            The bond length between carbon atoms in the CNT.
        z_coordinate : float
            The z-coordinate of the current layer.

        Returns
        -------
        positions : List[Tuple[float, float, float]]
            A list of atom positions in the format (x, y, z).
        """
        # Initialize an empty list to store the positions of atoms in this unit cell
        positions: List[Tuple[float, float, float]] = []

        # Calculate the position of the first atom in the unit cell
        angle1 = math.radians(symmetry_angle * i)  # Convert the angle to radians
        x1 = radius * math.cos(angle1)  # Calculate the x-coordinate using the radius and angle
        y1 = radius * math.sin(angle1)  # Calculate the y-coordinate using the radius and angle
        positions.append((x1, y1, z_coordinate))  # Append the (x, y, z) position of the first atom

        # Calculate the position of the second atom in the unit cell
        angle2 = math.radians(symmetry_angle * i + symmetry_angle / 2)  # Adjust the angle by half the symmetry angle
        x2 = radius * math.cos(angle2)  # Calculate the x-coordinate of the second atom
        y2 = radius * math.sin(angle2)  # Calculate the y-coordinate of the second atom
        z2 = zstep + z_coordinate  # Add the z-step to the z-coordinate for the second atom
        positions.append((x2, y2, z2))  # Append the (x, y, z) position of the second atom

        # The third atom shares the same angular position as the second but is further along the z-axis
        angle3 = angle2  # The angle remains the same as the second atom
        x3 = radius * math.cos(angle3)  # Calculate the x-coordinate of the third atom
        y3 = radius * math.sin(angle3)  # Calculate the y-coordinate of the third atom
        z3 = zstep + distance + z_coordinate  # Add the z-step and bond distance to the z-coordinate
        positions.append((x3, y3, z3))  # Append the (x, y, z) position of the third atom

        # The fourth atom returns to the angular position of the first atom but is at a different z-coordinate
        angle4 = angle1  # The angle is the same as the first atom
        x4 = radius * math.cos(angle4)  # Calculate the x-coordinate of the fourth atom
        y4 = radius * math.sin(angle4)  # Calculate the y-coordinate of the fourth atom
        z4 = 2 * zstep + distance + z_coordinate  # Add twice the z-step and bond distance to the z-coordinate
        positions.append((x4, y4, z4))  # Append the (x, y, z) position of the fourth atom

        # Return the list of atom positions calculated for this unit cell
        return positions

    def _add_nodes_to_graph(self, positions: List[Tuple[float, float, float]]):
        """
        Add the calculated positions as nodes to the graph.

        Parameters
        ----------
        positions : List[Tuple[float, float, float]]
            A list of atom positions in the format (x, y, z).
        """
        # Check if the CNT conformation is "armchair"
        if self.conformation == "armchair":
            # Calculate the index shift for armchair conformation to ensure correct node indexing
            idx_shift = 4 * self.tube_size

            # Generate node indices with a specific shift applied for the first node in each unit cell
            node_indices = [i + idx_shift - 1 if i % idx_shift == 0 else i - 1 for i in range(len(positions))]
        else:
            # For "zigzag" conformation, use a simple sequence of indices
            node_indices = list(range(len(positions)))

        # Create a dictionary of nodes, mapping each node index to its attributes
        nodes = {
            idx: {
                "element": "C",  # Set the element type to carbon ("C")
                "position": Position(*pos),  # Convert position tuple to Position object
                "possible_doping_site": True,  # Mark the node as a possible doping site
            }
            for idx, pos in zip(node_indices, positions)  # Iterate over node indices and positions
        }

        # Add all nodes to the graph using the dictionary created
        self.graph.add_nodes_from(nodes.items())

    def _add_internal_bonds(self, num_positions: int) -> None:
        """
        Add internal bonds within unit cells.

        Parameters
        ----------
        num_positions : int
            The number of atom positions in the CNT structure.
        """
        # Connect atoms within the same unit cell
        edges = [(idx, idx + 1) for idx in range(0, num_positions, 4)]
        edges += [(idx + 1, idx + 2) for idx in range(0, num_positions, 4)]
        edges += [(idx + 2, idx + 3) for idx in range(0, num_positions, 4)]
        self.graph.add_edges_from(edges, bond_length=self.bond_length)

    def _add_unit_cell_connections(self, positions: List[Tuple[float, float, float]]):
        """
        Add connections between unit cells.

        Parameters
        ----------
        positions : List[Tuple[float, float, float]]
            A list of atom positions in the format (x, y, z).
        """
        edges = []
        for idx in range(0, len(positions) - 4, 4):
            # Check if the atoms are in the same layer
            if positions[idx][2] == positions[idx + 4][2]:
                if self.conformation == "armchair":
                    # Connect the last atom of the current unit cell with the first atom of the next unit cell
                    edges.append((idx + 3, idx + 4))
                else:
                    # Connect the second atom of the current unit cell with the first atom of the next unit cell
                    edges.append((idx + 1, idx + 4))
                    # Connect the third atom of the current unit cell with the fourth atom of the next unit cell
                    edges.append((idx + 2, idx + 7))
        self.graph.add_edges_from(edges, bond_length=self.bond_length)

    def _complete_cycle_connections(self, positions: List[Tuple[float, float, float]]) -> None:
        """
        Complete connections at the end of each cycle.

        Parameters
        ----------
        positions : List[Tuple[float, float, float]]
            A list of atom positions in the format (x, y, z).
        """
        edges: List[Tuple[int, int]] = []
        for idx in range(0, len(positions), 4 * self.tube_size):
            # Get indices of atoms in the first and last cells of the cycle (in the same layer)
            first_idx_first_cell_of_cycle = idx
            first_idx_last_cell_of_cycle = idx + 4 * self.tube_size - 4
            last_idx_last_cell_of_cycle = idx + 4 * self.tube_size - 1

            if self.conformation == "armchair":
                # Connect the last atom of the last unit cell with the first atom of the first unit cell
                edges.append((last_idx_last_cell_of_cycle, first_idx_first_cell_of_cycle))
            else:
                # Connect the second atom of the last unit cell with the first atom of the first unit cell
                edges.append((first_idx_last_cell_of_cycle + 1, first_idx_first_cell_of_cycle))
                # Connect the third atom of the last unit cell with the fourth atom of the first unit cell
                edges.append((first_idx_last_cell_of_cycle + 2, first_idx_first_cell_of_cycle + 3))
        self.graph.add_edges_from(edges, bond_length=self.bond_length)

    def _connect_layers(self, positions: List[Tuple[float, float, float]]):
        """
        Create connections between different layers of the CNT.

        Parameters
        ----------
        positions : List[Tuple[float, float, float]]
            A list of atom positions in the format (x, y, z).
        """
        edges = []
        # Loop over the positions to create connections between different layers
        # The loop iterates over the positions with a step of 4 * tube_size (this ensures that we are jumping from one
        # "layer" to the next)
        for idx in range(0, len(positions) - 4 * self.tube_size, 4 * self.tube_size):
            # Loop over the number of cells in one layer (tube_size)
            for i in range(self.tube_size):
                # For the "armchair" conformation
                if self.conformation == "armchair":
                    # Connect the third atom of a cell of one layer to the fourth atom of a cell in the layer above
                    edges.append((idx + 2 + 4 * i, idx + 3 + 4 * i + 4 * self.tube_size))
                    # Connect the second atom of a cell of one layer to the first atom of a cell in the layer above
                    edges.append((idx + 1 + 4 * i, idx + 4 * i + 4 * self.tube_size))
                # For the "zigzag" conformation
                else:
                    # Connect the last atom of a cell of one layer to the first atom of a cell in the layer above
                    edges.append((idx + 3 + 4 * i, idx + 4 * i + 4 * self.tube_size))
        self.graph.add_edges_from(edges, bond_length=self.bond_length)

    def _add_periodic_boundaries(self, num_atoms: int):
        """
        Add periodic boundary conditions along the z-axis of the CNT.

        Parameters
        ----------
        num_atoms : int
            The number of atoms in the CNT structure.
        """
        edges: List[Tuple[int, int]] = []

        # Loop through the atoms to connect the first layer with the last, closing the cylinder along the z-axis
        for i in range(self.tube_size):
            # Determine the indices for the first and last layers in the positions list
            first_idx_z_coordinate_first_cell = i * 4
            first_idx_z_coordinate_last_cell = first_idx_z_coordinate_first_cell + num_atoms - 4 * self.tube_size

            if self.conformation == "armchair":
                # Connect the second atom of the last cell in one column of cells with the first atom of the first cell
                edges.append((first_idx_z_coordinate_last_cell + 1, first_idx_z_coordinate_first_cell))
                # Connect the third atom of the last cell in one column of cells with the fourth atom of the first cell
                edges.append((first_idx_z_coordinate_last_cell + 2, first_idx_z_coordinate_first_cell + 3))
            else:
                # Connect the last atom of the last cell in one column of cells with the first atom of the first cell
                edges.append((first_idx_z_coordinate_last_cell + 3, first_idx_z_coordinate_first_cell))

        # Add these periodic edges to the graph, marking them as periodic
        self.graph.add_edges_from(edges, bond_length=self.bond_length, periodic=True)

    def add_nitrogen_doping(self, total_percentage: float = None, percentages: dict = None):
        """
        Add nitrogen doping to the CNT.

        This method replaces a specified percentage of carbon atoms with nitrogen atoms in the CNT.
        If specific percentages for different nitrogen species are provided, it ensures the sum does not exceed the
        total percentage. The remaining percentage is distributed equally among the available nitrogen species. Note
        that no position adjustment is implemented for three-dimensional structures and therefore not supported for
        CNTs as well.

        Parameters
        ----------
        total_percentage : float, optional
            The total percentage of carbon atoms to replace with nitrogen atoms. Default is 10 if not specified.
        percentages : dict, optional
            A dictionary specifying the percentages for each nitrogen species. Keys should be NitrogenSpecies enum
            values and values should be the percentages for the corresponding species.

        Raises
        ------
        ValueError
            If the specific percentages exceed the total percentage.

        Notes
        -----
        - If no total percentage is provided, a default of 10% is used.
        - If specific percentages are provided and their sum exceeds the total percentage, a ValueError is raised.
        - Remaining percentages are distributed equally among the available nitrogen species.
        - Nitrogen species are added in a predefined order: PYRIDINIC_4, PYRIDINIC_3, PYRIDINIC_2, PYRIDINIC_1,
          GRAPHITIC.

        Warnings
        --------
        Note that three-dimensional position adjustment is currently not implemented in CONAN. Therefore, the generated
        doped structure should be used as a preliminary model and is recommended for further refinement using DFT or
        other computational methods. Future versions may include 3D position optimization.
        """
        # Delegate the doping process to the DopingHandler
        self.doping_handler.add_nitrogen_doping(total_percentage, percentages)

        # Issue a user warning about the lack of 3D position adjustment
        warnings.warn(
            "3D position adjustment is not currently supported in CONAN. "
            "The generated doped structure should be treated with care and may be used as a basis for further DFT or "
            "other computational calculations."
            "Future versions may include 3D position optimization.",
            UserWarning,
        )


class Pore(Structure3D):
    """
    Represents a Pore structure consisting of two graphene sheets connected by a CNT.
    """

    def __init__(
        self,
        bond_length: Union[float, int],
        sheet_size: Union[Tuple[float, float], Tuple[int, int]],
        tube_length: float,
        tube_size: Optional[int] = None,
        tube_diameter: Optional[float] = None,
        conformation: str = "zigzag",
    ):
        """
        Initialize the Pore with two graphene sheets and a CNT in between.

        Parameters
        ----------
        bond_length : Union[float, int]
            The bond length between carbon atoms.
        sheet_size : Optional[Tuple[float, float], Tuple[int, int]]
            The size of the graphene sheets (x, y dimensions).
        tube_length : float
            The length of the CNT.
        tube_size : int, optional
            The size of the CNT (number of hexagonal units around the circumference).
        tube_diameter : float, optional
            The diameter of the CNT.
        conformation : str, optional
            The conformation of the CNT ('armchair' or 'zigzag'). Default is 'zigzag'.
        """
        super().__init__()

        # Input validation and parameter handling
        if tube_size is None and tube_diameter is None:
            raise ValueError("You must specify either 'tube_size' or 'tube_diameter' for the CNT within the pore.")

        if tube_size is not None and tube_diameter is not None:
            raise ValueError(
                "Specify only one of 'tube_size' or 'tube_diameter' for the CNT within the pore, not " "both."
            )

        self.bond_length = self._validate_bond_length(bond_length)
        self.sheet_size = GrapheneSheet.validate_sheet_size(sheet_size)
        self.tube_length = CNT.validate_tube_length(tube_length)
        self.conformation = CNT.validate_conformation(conformation)

        # Use 'tube_size' as internal parameter
        if tube_size is not None:
            self.tube_size = CNT.validate_tube_size(tube_size)
        elif tube_diameter is not None:
            # Create a temporary CNT to calculate 'tube_size' from 'tube_diameter'
            temp_cnt = CNT(
                bond_length=self.bond_length,
                tube_length=self.tube_length,
                tube_diameter=tube_diameter,
                conformation=self.conformation,
            )
            self.tube_size = temp_cnt.tube_size

        # Create the graphene sheets and CNT
        self.graphene1 = GrapheneSheet(bond_length, sheet_size)
        self.graphene2 = GrapheneSheet(bond_length, sheet_size)
        self.cnt = CNT(
            bond_length=bond_length,
            tube_length=tube_length,
            tube_size=tube_size,
            tube_diameter=tube_diameter,
            conformation=conformation,
        )

        # Build the structure
        self.build_structure()

    def build_structure(self):
        """
        Build the Pore structure by connecting the two graphene sheets with the CNT.
        """
        # Calculate the x and y shift to center the CNT in the middle of the first graphene sheet
        x_shift = self.graphene1.actual_sheet_width / 2
        y_shift = self.graphene1.actual_sheet_height / 2

        # Position the CNT exactly in the center of the first graphene sheet in the x and y directions
        self.cnt.translate(x_shift=x_shift, y_shift=y_shift)

        # Shift the second graphene sheet along the z-axis by the length of the CNT
        self.graphene2.translate(z_shift=self.cnt.actual_length)

        # Create holes in the graphene sheets
        center = (x_shift, y_shift)
        radius = self.cnt.actual_tube_diameter / 2 + self.bond_length
        self.graphene1.create_hole(center, radius)
        self.graphene2.create_hole(center, radius)

        # Merge the three structures (graphene1, CNT, graphene2)
        self._merge_structures()

    def _merge_structures(self):
        """
        Merge the two graphene sheets and the CNT into a single structure.
        """
        # self.graph = nx.compose_all([self.graphene1.graph, self.cnt.graph, self.graphene2.graph])
        self.graph = nx.disjoint_union_all([self.graphene1.graph, self.cnt.graph, self.graphene2.graph])
        # self._connect_graphene_to_cnt()

    def add_nitrogen_doping(self, total_percentage: float = 10):
        """
        Add nitrogen doping to the Pore structure.

        Parameters
        ----------
        total_percentage : float
            Percentage of carbon atoms to replace with nitrogen.
        """
        self.graphene1.add_nitrogen_doping(total_percentage)
        self.graphene2.add_nitrogen_doping(total_percentage)
        self.cnt.add_nitrogen_doping(total_percentage)


def main():
    # Set seed for reproducibility
    # random.seed(42)
    # random.seed(3)
    random.seed(1)

    ####################################################################################################################
    # # CREATE A GRAPHENE SHEET
    # sheet_size = (10, 10)
    #
    # graphene = GrapheneSheet(bond_distance=1.42, sheet_size=sheet_size)
    # graphene.plot_structure(with_labels=True, visualize_periodic_bonds=False)
    # write_xyz(graphene.graph, "graphene_sheet.xyz")

    ####################################################################################################################
    # # CREATE A GRAPHENE SHEET AND LABEL THE ATOMS
    # sheet_size = (10, 10)
    #
    # graphene = GrapheneSheet(bond_distance=1.42, sheet_size=sheet_size)
    # graphene.plot_structure(with_labels=True, visualize_periodic_bonds=False)
    #
    # # Label atoms before writing to XYZ file
    # labeler = AtomLabeler(graphene.graph, graphene.doping_handler.doping_structures)
    # labeler.label_atoms()
    #
    # write_xyz(graphene.graph, "graphene_sheet.xyz")

    ####################################################################################################################
    # CREATE A GRAPHENE SHEET, DOPE IT AND ADJUST POSITIONS VIA ADD_NITROGEN_DOPING METHOD
    sheet_size = (15, 15)

    graphene = GrapheneSheet(bond_length=1.42, sheet_size=sheet_size)
    graphene.add_nitrogen_doping(total_percentage=10, adjust_positions=True)
    # graphene.add_nitrogen_doping(percentages={NitrogenSpecies.PYRIDINIC_4: 1})
    graphene.plot_structure(with_labels=True, visualize_periodic_bonds=False)

    write_xyz(graphene.graph, "graphene_sheet_doped.xyz")

    ####################################################################################################################
    # # CREATE A GRAPHENE SHEET, DOPE IT AND ADJUST POSITIONS
    # sheet_size = (20, 20)
    #
    # # Create a graphene sheet
    # graphene = GrapheneSheet(bond_distance=1.42, sheet_size=sheet_size)
    #
    # # Add nitrogen doping without adjusting positions
    # graphene.add_nitrogen_doping(total_percentage=10, adjust_positions=False)
    # # graphene.add_nitrogen_doping(percentages={NitrogenSpecies.PYRIDINIC_4: 1})
    #
    # # Adjust positions separately
    # graphene.adjust_atom_positions()
    # # Positions are now adjusted
    #
    # # Attempt to adjust positions again
    # graphene.adjust_atom_positions()
    # # Warning: Positions have already been adjusted
    #
    # # Plot structure
    # graphene.plot_structure(with_labels=True, visualize_periodic_bonds=False)
    #
    # write_xyz(graphene.graph, "graphene_sheet_doped.xyz")

    ####################################################################################################################
    # # CREATE A GRAPHENE SHEET, DOPE IT AND LABEL THE ATOMS
    # sheet_size = (20, 20)
    #
    # graphene = GrapheneSheet(bond_distance=1.42, sheet_size=sheet_size)
    # graphene.add_nitrogen_doping(total_percentage=10, adjust_positions=False)
    # graphene.plot_structure(with_labels=True, visualize_periodic_bonds=False)
    #
    # # Label atoms before writing to XYZ file
    # labeler = AtomLabeler(graphene.graph, graphene.doping_handler.doping_structures)
    # labeler.label_atoms()
    #
    # write_xyz(graphene.graph, "graphene_sheet_doped.xyz")

    ####################################################################################################################
    # # VERSION 1: CREATE A GRAPHENE SHEET, DOPE AND STACK IT
    # import time
    # sheet_size = (20, 20)
    #
    # # Create a graphene sheet
    # graphene = GrapheneSheet(bond_distance=1.42, sheet_size=sheet_size)
    #
    # # Add nitrogen doping to the graphene sheet
    # start_time = time.time()  # Time the nitrogen doping process
    # graphene.add_nitrogen_doping(total_percentage=15, adjust_positions=True)
    # end_time = time.time()
    #
    # # Calculate the elapsed time
    # elapsed_time = end_time - start_time
    # print(f"Time taken for nitrogen doping for a sheet of size {sheet_size}: {elapsed_time:.2f} seconds")
    #
    # # Plot the graphene sheet with nitrogen doping
    # graphene.plot_structure(with_labels=True, visualize_periodic_bonds=False)
    #
    # # Stack the graphene sheet
    # stacked_graphene = graphene.stack(interlayer_spacing=3.35, number_of_layers=5)
    #
    # # Plot the stacked structure
    # stacked_graphene.plot_structure(with_labels=True, visualize_periodic_bonds=False)
    #
    # # Save the structure to a .xyz file
    # write_xyz(stacked_graphene.graph, "ABA_stacking.xyz")

    ####################################################################################################################
    # # VERSION 2: DIRECTLY USE THE STACKED GRAPHENE SHEET AND ADJUST POSITIONS VIA ADD_NITROGEN_DOPING METHOD
    #
    # # Create a graphene sheet
    # graphene_sheet = GrapheneSheet(bond_length=1.42, sheet_size=(40, 40))
    #
    # # Create stacked graphene using the graphene sheet
    # stacked_graphene = StackedGraphene(graphene_sheet, number_of_layers=5, stacking_type="ABA")
    #
    # # Add nitrogen doping to the specified graphene sheets
    # stacked_graphene.add_nitrogen_doping(total_percentage=8, adjust_positions=True, layers="all")
    #
    # # Plot the stacked structure
    # stacked_graphene.plot_structure(with_labels=False, visualize_periodic_bonds=False)
    #
    # write_xyz(stacked_graphene.graph, "ABA_stacking.xyz")

    ####################################################################################################################
    # # VERSION 2: DIRECTLY USE THE STACKED GRAPHENE SHEET AND ADJUST POSITIONS OF SPECIFIC LAYERS
    #
    # # Create a base graphene sheet
    # base_graphene = GrapheneSheet(bond_distance=1.42, sheet_size=(20, 20))
    #
    # # Create a stacked graphene structure
    # stacked_graphene = StackedGraphene(base_graphene, number_of_layers=3)
    #
    # # Add nitrogen doping to layers 0 and 1 without adjusting positions
    # stacked_graphene.add_nitrogen_doping(total_percentage=10, adjust_positions=False, layers=[0, 1])
    # # No positions adjusted
    #
    # # Adjust positions for layers 0 and 1
    # stacked_graphene.adjust_atom_positions(layers=[0, 1])
    # # Positions are now adjusted for layers 0 and 1
    #
    # # Attempt to adjust positions again
    # stacked_graphene.adjust_atom_positions(layers=[0, 1])
    # # Warnings: Positions have already been adjusted in layers 0 and 1

    ####################################################################################################################
    # # Example: Only dope the first and last layer (both will have the same doping percentage but different ordering)
    # import time
    #
    # sheet_size = (20, 20)
    #
    # # Create a graphene sheet
    # graphene = GrapheneSheet(bond_distance=1.42, sheet_size=sheet_size)
    #
    # # Stack the graphene sheet
    # stacked_graphene = graphene.stack(interlayer_spacing=3.34, number_of_layers=5, stacking_type="ABC")
    #
    # # Add individual nitrogen doping only to the first and last layer
    # start_time = time.time()  # Time the nitrogen doping process
    # stacked_graphene.add_nitrogen_doping_to_layer(layer_index=0, total_percentage=15, adjust_positions=True)
    # stacked_graphene.add_nitrogen_doping_to_layer(layer_index=4, total_percentage=15, adjust_positions=True)
    # end_time = time.time()
    #
    # # Calculate the elapsed time
    # elapsed_time = end_time - start_time
    # print(f"Time taken for nitrogen doping for a sheet of size {sheet_size}: {elapsed_time:.2f} seconds")
    #
    # # Plot the stacked structure
    # stacked_graphene.plot_structure(with_labels=True, visualize_periodic_bonds=False)
    #
    # # Save the structure to a .xyz file
    # write_xyz(stacked_graphene.graph, "ABC_stacking.xyz")

    ####################################################################################################################
    # # CREATE A CNT STRUCTURE
    #
    # # cnt = CNT(bond_length=1.42, tube_length=10.0, tube_size=8, conformation="zigzag", periodic=False)
    # cnt = CNT(bond_length=1.42, tube_length=10.0, tube_diameter=6, conformation="zigzag", periodic=False)
    # # cnt.add_nitrogen_doping(total_percentage=10)
    # cnt.plot_structure(with_labels=True, visualize_periodic_bonds=False)
    #
    # # Save the CNT structure to a file
    # write_xyz(cnt.graph, "CNT_structure_zigzag_doped.xyz")

    ####################################################################################################################
    # CREATE A PORE STRUCTURE
    # Define parameters for the graphene sheets and CNT
    bond_length = 1.42  # Bond length for carbon atoms
    sheet_size = (20, 20)  # Size of the graphene sheets
    tube_length = 10.0  # Length of the CNT
    # tube_size = 8  # Number of hexagonal units around the CNT circumference
    tube_diameter = 7  # Diameter of the CNT
    conformation = "zigzag"  # Conformation of the CNT (can be "zigzag" or "armchair")

    # Create a Pore structure
    pore = Pore(
        bond_length=bond_length,
        sheet_size=sheet_size,
        tube_length=tube_length,
        # tube_size=tube_size,
        tube_diameter=tube_diameter,
        conformation=conformation,
    )

    # Add optional nitrogen doping (if needed)
    # pore.add_nitrogen_doping(total_percentage=10)

    # Visualize the structure with labels (without showing periodic bonds)
    pore.plot_structure(with_labels=True, visualize_periodic_bonds=False)

    # Save the Pore structure to a file
    write_xyz(pore.graph, "pore_structure.xyz")


if __name__ == "__main__":
    main()
