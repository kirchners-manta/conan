from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conan.playground.structures import MaterialStructure
    from conan.playground.utils import get_neighbors_via_edges, minimum_image_distance

import random
import warnings
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import pandas as pd

from conan.playground.utils import get_neighbors_via_edges, minimum_image_distance

# Define a namedtuple for structural components
# This namedtuple will be used to store the atom(s) around which the doping structure is built and its/their neighbors
StructuralComponents = namedtuple("StructuralComponents", ["structure_building_atoms", "structure_building_neighbors"])


@dataclass
class NitrogenSpeciesProperties:
    """
    Define data class for nitrogen species properties.

    Attributes
    ----------
    target_bond_lengths_cycle : List[float]
        A list of bond lengths inside the doping structure (cycle).
    target_bond_lengths_neighbors : List[float]
        A list of bond lengths between doping structure and neighbors outside the cycle.
    target_angles_cycle : List[float]
        A list of bond angles inside the doping structure (cycle).
    target_angles_neighbors : List[float]
        A list of bond angles between doping structure and neighbors outside the cycle.
    target_angles_additional_angles : Optional[List[float]]
        A list of angles that are added by adding the additional_edge in the PYRIDINIC_1 case
    """

    target_bond_lengths_cycle: List[float]
    target_bond_lengths_neighbors: List[float]
    target_angles_cycle: List[float]
    target_angles_neighbors: List[float]
    target_angles_additional_angles: Optional[List[float]] = field(default=None)


class NitrogenSpecies(Enum):
    GRAPHITIC = "Graphitic-N"
    # PYRIDINIC = "pyridinic"
    PYRIDINIC_1 = "Pyridinic-N 1"
    PYRIDINIC_2 = "Pyridinic-N 2"
    PYRIDINIC_3 = "Pyridinic-N 3"
    PYRIDINIC_4 = "Pyridinic-N 4"
    # PYRROLIC = "pyrrolic"
    # PYRAZOLE = "pyrazole"

    @staticmethod
    def get_color(element: str, nitrogen_species: "NitrogenSpecies" = None) -> str:
        """
        Get the color based on the element and type of nitrogen.

        Parameters
        ----------
        element : str
            The chemical element of the atom (e.g., 'C' for carbon, 'N' for nitrogen).
        nitrogen_species : NitrogenSpecies, optional
            The type of nitrogen doping (default is None).

        Returns
        -------
        str
            The color associated with the element and nitrogen species.

        Notes
        -----
        The color mapping is defined for different elements and nitrogen species to visually
        distinguish them in plots.
        """
        colors = {"C": "gray"}
        nitrogen_colors = {
            # NitrogenSpecies.PYRIDINIC: "blue",
            NitrogenSpecies.PYRIDINIC_1: "violet",
            NitrogenSpecies.PYRIDINIC_2: "orange",
            NitrogenSpecies.PYRIDINIC_3: "lime",
            NitrogenSpecies.PYRIDINIC_4: "cyan",
            NitrogenSpecies.GRAPHITIC: "tomato",
            # NitrogenSpecies.PYRROLIC: "cyan",
            # NitrogenSpecies.PYRAZOLE: "green",
        }
        if nitrogen_species in nitrogen_colors:
            return nitrogen_colors[nitrogen_species]
        return colors.get(element, "pink")


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

        # if isinstance(structure, GrapheneSheet):
        if hasattr(structure, "actual_sheet_width") and hasattr(structure, "actual_sheet_height"):
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
            If the specific percentages exceed the total percentage beyond a small tolerance, or if any percentage is
            negative.
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
                if value <= 0:
                    raise ValueError(f"Percentage for {key} must be positive. Received {value}.")

        # Validate the input for total_percentage
        if total_percentage is not None:
            if not isinstance(total_percentage, (int, float)):
                raise ValueError("total_percentage must be an int or float.")
            if total_percentage <= 0:
                raise ValueError("total_percentage must be positive.")

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
