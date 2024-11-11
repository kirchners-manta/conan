from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conan.playground.structures import MaterialStructure
    from conan.playground.utils import get_neighbors_via_edges, minimum_image_distance

# import math
import random
import warnings
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

import networkx as nx
import pandas as pd
from pulp import PULP_CBC_CMD, LpMinimize, LpProblem, LpStatusOptimal, LpVariable, lpSum

from conan.playground.utils import get_neighbors_via_edges, minimum_image_distance

# Define a namedtuple for structural components
# This namedtuple will be used to store the atom(s) around which the doping structure is built and its/their neighbors
StructuralComponents = namedtuple("StructuralComponents", ["structure_building_atoms", "structure_building_neighbors"])


class OptimizationWeights(NamedTuple):
    """
    Represents the weights used in the optimization of nitrogen doping to achieve the desired nitrogen percentage while
    maintaining an equal distribution among species.

    In the linear programming model, the objective function is: Minimize w1 * z1 + w2 * z2, where z1 represents the
    absolute deviation from the desired total nitrogen percentage and z2 represents the absolute deviation from an equal
    distribution of nitrogen atoms among the different species.

    Attributes
    ----------
    nitrogen_percentage_weight : float
        Weight for the deviation from the desired nitrogen percentage (w1).
    equal_distribution_weight : float
        Weight for the deviation from equal distribution among species (w2).

    Notes
    -----
    * Effects of high w1 and w2 values:

        High nitrogen_percentage_weight: The optimization will prioritize achieving the desired nitrogen percentage so
        that the solution will closely match the desired nitrogen percentage, potentially at the expense of an equal
        distribution of nitrogen atoms among species.

        High equal_distribution_weight: The optimization will prioritize achieving an equal distribution of nitrogen
        atoms among species, potentially at the expense of deviating from the desired nitrogen percentage.

    * Trade-off between w1 and w2:

        Balancing Act: Adjusting w1 and w2 allows you to balance between achieving the exact nitrogen percentage and
        maintaining an equal distribution among species.

        Scenario Examples:
         - w1 >> w2: Focus on matching the nitrogen percentage; distribution among species is less critical.
         - w2 >> w1: Focus on equal distribution; some deviation from the nitrogen percentage is acceptable.
         - w1 ≈ w2: The model seeks a compromise between the two objectives.
    """

    nitrogen_percentage_weight: float = 1000.0
    equal_distribution_weight: float = 1.0

    def validate_weights(self):
        if self.nitrogen_percentage_weight <= 0 or self.equal_distribution_weight <= 0:
            raise ValueError("Weights must be positive numbers.")


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

    @staticmethod
    def get_num_nitrogen_atoms_to_add(nitrogen_species: "NitrogenSpecies") -> int:
        """
        Get the number of nitrogen atoms to add for a specific nitrogen species.

        Parameters
        ----------
        nitrogen_species : NitrogenSpecies
            The type of nitrogen doping.

        Returns
        -------
        int
            The number of nitrogen atoms to add for the specified nitrogen species.
        """
        if nitrogen_species in {NitrogenSpecies.PYRIDINIC_1, NitrogenSpecies.GRAPHITIC}:
            return 1
        if nitrogen_species == NitrogenSpecies.PYRIDINIC_2:
            return 2
        if nitrogen_species == NitrogenSpecies.PYRIDINIC_3:
            return 3
        if nitrogen_species == NitrogenSpecies.PYRIDINIC_4:
            return 4
        return 0

    @staticmethod
    def get_num_carbon_atoms_to_remove(nitrogen_species: "NitrogenSpecies") -> int:
        """
        Get the number of carbon atoms to remove in order to insert a specific nitrogen species.

        Parameters
        ----------
        nitrogen_species : NitrogenSpecies
            The type of nitrogen doping.

        Returns
        -------
        int
            The number of carbon atoms to remove for the specified nitrogen species.
        """
        if nitrogen_species == NitrogenSpecies.GRAPHITIC:
            return 0
        if nitrogen_species in {NitrogenSpecies.PYRIDINIC_1, NitrogenSpecies.PYRIDINIC_2, NitrogenSpecies.PYRIDINIC_3}:
            return 1
        if nitrogen_species == NitrogenSpecies.PYRIDINIC_4:
            return 2
        return 0


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

    # def add_nitrogen_doping(
    #     self,
    #     total_percentage: Optional[float] = None,
    #     percentages: Optional[dict] = None,
    #     optimization_weights: Optional[OptimizationWeights] = None,
    # ):
    #     """
    #     Add nitrogen doping to the structure using linear programming optimization and utilizing graph manipulation
    #     techniques to insert the doping structures.
    #
    #     This method adds nitrogen doping to the structure by determining the optimal number of doping structures
    #     for each nitrogen species to achieve the desired total nitrogen percentage and an equal distribution
    #     among species. It uses a linear programming model solved with PuLP.
    #
    #     Parameters
    #     ----------
    #     total_percentage : float, optional
    #         The total percentage of carbon atoms to replace with nitrogen atoms.
    #     percentages : dict, optional
    #         A dictionary specifying the percentages for each nitrogen species.
    #     optimization_weights : OptimizationWeights, optional
    #         An instance containing weights for the optimization objective function to balance the trade-off between
    #         gaining the desired nitrogen percentage and achieving an equal distribution among species.
    #
    #         - nitrogen_percentage_weight: Weight for the deviation from the desired nitrogen percentage in the
    #         objective function.
    #
    #         - equal_distribution_weight: Weight for the deviation from equal distribution among species in the
    #         objective function.
    #
    #     Raises
    #     ------
    #     ValueError
    #         If the specific percentages exceed the total percentage beyond a small tolerance, or if any percentage is
    #         negative.
    #     """
    #
    #     # Step 1: Validate inputs and prepare percentages
    #     total_percentage, percentages = self._validate_and_set_percentages(total_percentage, percentages)
    #
    #     # Step 2: Get initial number of carbon atoms in the structure
    #     num_initial_atoms = self.graph.number_of_nodes()
    #     if num_initial_atoms == 0:  # ToDo: Sollte hoffentlich nicht vorkommen
    #         warnings.warn("The structure has no carbon atoms to dope.", UserWarning)
    #         return
    #
    #     # Step 3: Calculate desired number of structures for each species using optimization
    #     desired_num_structures_per_species = self._calculate_num_desired_structures(
    #         num_initial_atoms, total_percentage, percentages, optimization_weights
    #     )
    #
    #     # Step 4: Insert doping structures
    #     self._insert_doping_structures(desired_num_structures_per_species)
    #
    #     # Step 5: Adjust if actual doping percentage falls short of desired
    #     self._adjust_for_shortfall_in_doping(total_percentage)
    #
    #     # Step 6: Display the final results of the doping process
    #     self._display_doping_results()

    def add_nitrogen_doping(
        self,
        total_percentage: Optional[float] = None,
        percentages: Optional[Dict[NitrogenSpecies, float]] = None,
        optimization_weights: Optional[OptimizationWeights] = None,
    ):
        """
        Add nitrogen doping to the structure using linear programming optimization.

        Parameters
        ----------
        total_percentage : float, optional
            The total percentage of carbon atoms to replace with nitrogen atoms.
        percentages : dict, optional
            A dictionary specifying the percentages for each nitrogen species.
        optimization_weights : OptimizationWeights, optional
            An instance containing weights for the optimization objective function to balance the trade-off between
            gaining the desired nitrogen percentage and achieving an equal distribution among species.

            - nitrogen_percentage_weight: Weight for the deviation from the desired nitrogen percentage in the
            objective function.

            - equal_distribution_weight: Weight for the deviation from equal distribution among species in the
            objective function.

        Raises
        ------
        ValueError
            If the specific percentages exceed the total percentage beyond a small tolerance, or if any percentage is
            negative.
        """
        # Step 1: Validate inputs and prepare percentages
        total_percentage, percentages = self._validate_and_prepare_percentages(total_percentage, percentages)

        # Step 2: Get initial number of carbon atoms in the structure
        num_initial_atoms = self.graph.number_of_nodes()
        if num_initial_atoms == 0:
            warnings.warn("The structure has no carbon atoms to dope.", UserWarning)
            return

        # Step 3: Calculate structures for specified percentages
        fixed_structures = {}
        num_atoms_remaining = num_initial_atoms
        specific_total_percentage = sum(percentages.values())
        tolerance = 1e-6

        if percentages and abs(total_percentage - specific_total_percentage) <= tolerance:
            # Only specified percentages are given and sum up to total_percentage
            fixed_structures, num_atoms_remaining = self._calculate_num_structures_for_specified_percentages(
                num_initial_atoms, percentages
            )
            desired_num_structures = fixed_structures
        else:
            # Calculate structures for specified percentages
            if percentages:
                fixed_structures, num_atoms_remaining = self._calculate_num_structures_for_specified_percentages(
                    num_initial_atoms, percentages
                )
            else:
                fixed_structures = {}
                num_atoms_remaining = num_initial_atoms

            # Remaining percentage
            remaining_percentage = total_percentage - sum(percentages.values())

            # Step 4: Use optimization for remaining percentages
            desired_structures_remaining = self._calculate_num_desired_structures(
                num_atoms_remaining, remaining_percentage, optimization_weights, fixed_structures
            )

            # Combine the results
            desired_num_structures = {**fixed_structures, **desired_structures_remaining}

        # Step 5: Insert doping structures
        self._insert_doping_structures(desired_num_structures)

        # Step 6: Adjust if actual doping percentage falls short of desired
        self._adjust_for_shortfall_in_doping(total_percentage)

        # Step 7: Display the final results of the doping process
        self._display_doping_results()

    @staticmethod
    def _validate_and_prepare_percentages(
        total_percentage: Optional[float], percentages: Optional[Dict[NitrogenSpecies, float]]
    ) -> (float, Dict[NitrogenSpecies, float], List[NitrogenSpecies]):
        """
        Validate the input percentages and total percentage, and set default values if necessary.

        Parameters
        ----------
        total_percentage : float
            The total nitrogen doping percentage.
        percentages : Dict[NitrogenSpecies, float]
            A dictionary specifying the desired percentages for each nitrogen species.

        Returns
        -------
        Tuple(float, Dict[NitrogenSpecies, float])
            The validated total nitrogen doping percentage as well as the validated and prepared percentages per
            nitrogen species.

         Raises
        ------
        ValueError
            If input values are invalid or inconsistent.

        Notes
        -----
        This method ensures that the input percentages and total_percentage are valid and consistent.
        If total_percentage is not provided, it defaults to 10%.
        If percentages are provided, it checks that they are valid and do not exceed the total_percentage.
        """
        # Validate the input for percentages
        if percentages is not None:
            if not isinstance(percentages, dict):
                raise ValueError(
                    "percentages must be a dictionary with NitrogenSpecies as keys and int or float as values."
                )
            if len(percentages) == 0:
                raise ValueError("percentages dictionary cannot be empty. Define at least one positive percentage.")

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

        # Calculate the sum of specified percentages
        specific_total_percentage = sum(percentages.values())
        tolerance = 1e-6

        # Validate specific percentages and total_percentage
        if percentages:
            if total_percentage is None:
                # Set total to sum of specific percentages if not provided
                total_percentage = specific_total_percentage
            else:
                # total_percentage is provided; check against specific_total_percentage
                if specific_total_percentage > total_percentage + tolerance:
                    # Raise an error if the sum of specific percentages exceeds the total percentage beyond the
                    # tolerance
                    raise ValueError(
                        f"The total specific percentages {specific_total_percentage}% exceed the total_percentage "
                        f"{total_percentage}%. Please adjust your input so that the sum of the 'percentages' is less "
                        f"than or equal to 'total_percentage'."
                    )
                elif specific_total_percentage < total_percentage - tolerance:
                    # Do not pre-distribute; let the optimization model handle it
                    warnings.warn(
                        f"The sum of specified percentages {specific_total_percentage}% is lower than the "
                        f"total_percentage {total_percentage}%. Remaining percentage will be distributed among other "
                        f"available species.",
                        UserWarning,
                    )
                    pass  # The optimization model will handle the remaining percentage
        else:
            # Set a default total percentage if not provided
            total_percentage = total_percentage if total_percentage is not None else 10.0

        return total_percentage, percentages

    @staticmethod
    def _calculate_num_desired_structures_old(
        num_initial_atoms: int,
        total_percentage: float,
        percentages: Dict[NitrogenSpecies, float],
        optimization_weights: Optional[OptimizationWeights],
    ) -> Dict[NitrogenSpecies, int]:
        """
        Calculate the desired number of structures for each species using linear programming optimization.

        This method sets up and solves a linear programming problem to determine the optimal number of doping
        structures for each nitrogen species to achieve the desired total nitrogen percentage while distributing
        the nitrogen atoms as equally as possible among the species.

        Parameters
        ----------
        num_initial_atoms : int
            The initial number of carbon atoms in the structure.
        total_percentage : float
            The desired total nitrogen doping percentage.
        percentages : Dict[NitrogenSpecies, float]
            The desired percentages for each nitrogen species.
        optimization_weights : OptimizationWeights, optional
            An instance containing weights for the optimization objective function to balance the trade-off between
            gaining the desired nitrogen percentage and achieving an equal distribution among species.

            - nitrogen_percentage_weight: Weight for the deviation from the desired nitrogen percentage in the
            objective function.

            - equal_distribution_weight: Weight for the deviation from equal distribution among species in the
            objective function.

        Returns
        -------
        Dict[NitrogenSpecies, int]
            The calculated number of structures to insert for each nitrogen species.
        """
        # Use default weights if none provided
        if optimization_weights is None:
            optimization_weights = OptimizationWeights()

        w1 = optimization_weights.nitrogen_percentage_weight
        w2 = optimization_weights.equal_distribution_weight

        # Convert desired percentage to fractions
        total_percentage_fraction = total_percentage / 100.0

        # Map species to indices for consistent ordering
        species_list = list(percentages.keys())
        # species_indices = {species: idx for idx, species in enumerate(species_list)}
        num_doping_types = len(species_list)  # Number of doping types

        # Constants for each doping type
        # Number of carbon atoms removed by doping type i
        ci = [NitrogenSpecies.get_num_carbon_atoms_to_remove(s) for s in species_list]
        # Number of nitrogen atoms added by doping type i
        ri = [NitrogenSpecies.get_num_nitrogen_atoms_to_add(s) for s in species_list]

        # Compute ki values (effective nitrogen contribution of doping type i, accounting for nitrogen added and the
        # effect of carbon atoms removed on the overall nitrogen percentage)
        ki = [ri_i + total_percentage_fraction * ci_i for ri_i, ci_i in zip(ri, ci)]
        # The right-hand side of the equation, representing the desired total nitrogen atoms based on the initial total
        # atoms
        rhs = total_percentage_fraction * num_initial_atoms

        # Initialize the problem
        prob = LpProblem("Nitrogen_Doping_Optimization", LpMinimize)

        # Decision variables
        # Integer variable representing the number of doping structures of type i to insert
        xi = [LpVariable(f"x_{i}", lowBound=0, cat="Integer") for i in range(num_doping_types)]
        # Continuous variable representing the average number of nitrogen atoms per doping type
        num_nitrogen_avg = LpVariable("N_avg", lowBound=0, cat="Continuous")
        # Continuous variables for positive and negative deviations of nitrogen atoms added by doping type i from N_avg
        p_num_nitrogen_dev_i = [LpVariable(f"P_d_{i}", lowBound=0, cat="Continuous") for i in range(num_doping_types)]
        n_num_nitrogen_dev_i = [LpVariable(f"N_d_{i}", lowBound=0, cat="Continuous") for i in range(num_doping_types)]
        # Continuous variables for positive and negative deviations (in nitrogen atom units) from the desired total
        # nitrogen atoms
        p_perc_dev = LpVariable("P", lowBound=0, cat="Continuous")
        n_perc_dev = LpVariable("N", lowBound=0, cat="Continuous")
        z1 = LpVariable("z1", lowBound=0, cat="Continuous")
        z2 = LpVariable("z2", lowBound=0, cat="Continuous")

        # Objective function
        prob += w1 * z1 + w2 * z2, "Minimize total deviation"

        # Nitrogen percentage constraint (replacing upper and lower bound constraints) to ensure that the total
        # effective nitrogen contribution from all doping types matches the desired total nitrogen atoms (rhs),
        # accounting for deviations (P, N)
        prob += (
            lpSum([ki[i] * xi[i] for i in range(num_doping_types)]) + p_perc_dev - n_perc_dev == rhs,
            "Nitrogen deviation constraint",
        )
        # Define z1 as the sum of positive and negative deviations, effectively capturing the absolute deviation in
        # nitrogen atoms
        prob += z1 == p_perc_dev + n_perc_dev, "Absolute deviation constraint"

        # Constraint to calculate the average number of nitrogen atoms added per doping type
        prob += (
            num_nitrogen_avg == lpSum([ri[i] * xi[i] for i in range(num_doping_types)]) / num_doping_types,
            "Average nitrogen atoms constraint",
        )

        # Constraints for deviations in nitrogen atoms from the average
        for i in range(num_doping_types):
            prob += (
                ri[i] * xi[i] + p_num_nitrogen_dev_i[i] - n_num_nitrogen_dev_i[i] == num_nitrogen_avg,
                f"Nitrogen deviation for type {i}",
            )
        # Define z2 as the sum of all individual deviations, representing the total deviation from equal nitrogen
        # distribution
        prob += (
            z2 == lpSum([p_num_nitrogen_dev_i[i] + n_num_nitrogen_dev_i[i] for i in range(num_doping_types)]),
            "Total deviation in nitrogen atoms",
        )

        # Solve the problem
        prob.solve(PULP_CBC_CMD(msg=False))

        # Check if an optimal solution was found
        if prob.status != LpStatusOptimal:
            warnings.warn("Optimal solution not found. Doping may not meet desired specifications.", UserWarning)

        # Retrieve the solution
        xi_values = [int(xi[i].varValue) for i in range(num_doping_types)]
        desired_num_structures = {species_list[i]: xi_values[i] for i in range(num_doping_types)}

        return desired_num_structures

    @staticmethod
    def _calculate_num_desired_structures(
        num_initial_atoms: int,
        total_percentage: float,
        optimization_weights: Optional[OptimizationWeights],
        fixed_structures: Dict[NitrogenSpecies, int],
    ) -> Dict[NitrogenSpecies, int]:
        """
        Calculate the desired number of structures for the remaining species using linear programming optimization.

        Parameters
        ----------
        num_initial_atoms : int
            The initial number of carbon atoms in the structure (possibly after already inserting specified structures)
        total_percentage : float
            The desired total nitrogen doping percentage to be distributed among the remaining species.
        optimization_weights : OptimizationWeights, optional
            Weights for the optimization objective function.
        fixed_structures : Dict[NitrogenSpecies, int]
            The number of structures already assigned to specified species.

        Returns
        -------
        desired_num_structures : Dict[NitrogenSpecies, int]
            The calculated number of structures to insert for each remaining nitrogen species.
        """
        # Use default weights if none provided
        if optimization_weights is None:
            optimization_weights = OptimizationWeights()

        w1 = optimization_weights.nitrogen_percentage_weight
        w2 = optimization_weights.equal_distribution_weight

        # Remaining species (excluding those in fixed_structures)
        remaining_species = [s for s in NitrogenSpecies if s not in fixed_structures]
        num_doping_types = len(remaining_species)

        if num_doping_types == 0:
            return {}

        # Constants for each doping type
        # Number of carbon atoms removed by doping type i
        ci = [NitrogenSpecies.get_num_carbon_atoms_to_remove(s) for s in remaining_species]
        # Number of nitrogen atoms added by doping type i
        ri = [NitrogenSpecies.get_num_nitrogen_atoms_to_add(s) for s in remaining_species]

        # Convert desired percentage to fractions
        total_percentage_fraction = total_percentage / 100.0

        # Compute ki values (effective nitrogen contribution of doping type i, accounting for nitrogen added and the
        # effect of carbon atoms removed on the overall nitrogen percentage)
        ki = [ri_i + total_percentage_fraction * ci_i for ri_i, ci_i in zip(ri, ci)]

        # The right-hand side of the equation, representing the desired total nitrogen atoms based on the initial total
        # atoms
        rhs = total_percentage_fraction * num_initial_atoms

        # Initialize the problem
        prob = LpProblem("Nitrogen_Doping_Optimization_Remaining", LpMinimize)

        # Decision variables
        # Integer variable representing the number of doping structures of type i to insert
        xi = [LpVariable(f"x_{i}", lowBound=0, cat="Integer") for i in range(num_doping_types)]
        # Continuous variable representing the average number of nitrogen atoms per doping type
        num_nitrogen_avg = LpVariable("N_avg", lowBound=0, cat="Continuous")
        # Continuous variables for positive and negative deviations of nitrogen atoms added by doping type i from N_avg
        p_num_nitrogen_dev_i = [LpVariable(f"P_{i}", lowBound=0, cat="Continuous") for i in range(num_doping_types)]
        n_num_nitrogen_dev_i = [LpVariable(f"N_{i}", lowBound=0, cat="Continuous") for i in range(num_doping_types)]
        # Continuous variables for positive and negative deviations (in nitrogen atom units) from the desired total
        # nitrogen atoms
        p_perc_dev = LpVariable("P", lowBound=0, cat="Continuous")
        n_perc_dev = LpVariable("N", lowBound=0, cat="Continuous")
        # Continuous variable representing the deviation in nitrogen percentage from the desired value
        z1 = LpVariable("z1", lowBound=0, cat="Continuous")
        # Continuous variable representing the total deviation in nitrogen atoms from equal distribution among doping
        # types
        z2 = LpVariable("z2", lowBound=0, cat="Continuous")

        # Objective function
        prob += w1 * z1 + w2 * z2, "Minimize total deviation"

        # Nitrogen percentage constraint (replacing upper and lower bound constraints) to ensure that the total
        # effective nitrogen contribution from all doping types matches the desired total nitrogen atoms (rhs),
        # accounting for deviations (P, N)
        prob += (
            lpSum([ki[i] * xi[i] for i in range(num_doping_types)]) + p_perc_dev - n_perc_dev == rhs,
            "Nitrogen deviation constraint",
        )
        # Define z1 as the sum of positive and negative deviations, effectively capturing the absolute deviation in
        # nitrogen atoms
        prob += z1 == p_perc_dev + n_perc_dev, "Absolute deviation constraint"

        # Constraint to calculate the average number of nitrogen atoms added per doping type
        prob += (
            num_nitrogen_avg == lpSum([ri[i] * xi[i] for i in range(num_doping_types)]) / num_doping_types,
            "Average nitrogen atoms constraint",
        )

        # Constraints for deviations in nitrogen atoms from the average
        for i in range(num_doping_types):
            prob += (
                ri[i] * xi[i] + p_num_nitrogen_dev_i[i] - n_num_nitrogen_dev_i[i] == num_nitrogen_avg,
                f"Nitrogen deviation for type {i}",
            )
            # Define z2 as the sum of all individual deviations, representing the total deviation from equal nitrogen
            # distribution
        prob += (
            z2 == lpSum([p_num_nitrogen_dev_i[i] + n_num_nitrogen_dev_i[i] for i in range(num_doping_types)]),
            "Total deviation in nitrogen atoms",
        )

        # Solve the problem
        prob.solve(PULP_CBC_CMD(msg=False))

        # Check if an optimal solution was found
        if prob.status != LpStatusOptimal:
            warnings.warn("Optimal solution not found. Doping may not meet desired specifications.", UserWarning)

        # Retrieve the solution
        xi_values = [int(xi[i].varValue) for i in range(num_doping_types)]
        desired_num_structures = {remaining_species[i]: xi_values[i] for i in range(num_doping_types)}

        return desired_num_structures

    @staticmethod
    def _calculate_num_structures_for_specified_percentages(
        num_initial_atoms: int, percentages: Dict[NitrogenSpecies, float]
    ) -> Tuple[Dict[NitrogenSpecies, int], int]:
        """
        Calculate the number of structures needed for each specified species to achieve their desired percentages
        as closely as possible.

        Parameters
        ----------
        num_initial_atoms : int
            The initial number of carbon atoms in the structure.
        percentages : Dict[NitrogenSpecies, float]
            The desired percentages for each specified nitrogen species.

        Returns
        -------
        desired_num_structures : Dict[NitrogenSpecies, int]
            The calculated number of structures for each specified species.
        num_atoms_remaining : int
            The number of carbon atoms remaining after inserting the specified structures.
        """
        desired_num_structures = {}
        num_atoms_remaining = num_initial_atoms

        for species, desired_percentage in percentages.items():
            # Get the number of nitrogen atoms added and carbon atoms removed per structure
            ri = NitrogenSpecies.get_num_nitrogen_atoms_to_add(species)
            ci = NitrogenSpecies.get_num_carbon_atoms_to_remove(species)

            # Desired percentage as a fraction
            desired_percentage_fraction = desired_percentage / 100.0

            # Calculate n using the derived formula
            numerator = desired_percentage_fraction * num_atoms_remaining
            denominator = ri + desired_percentage_fraction * ci

            # Avoid division by zero
            if denominator == 0:
                n = 0
            else:
                n = numerator / denominator

            # Since n must be an integer, consider floor and ceiling values
            n_floor = int(n)
            n_ceil = n_floor + 1

            # Initialize variables to track the best n
            min_delta = float("inf")
            best_n = n_floor

            # Evaluate both floor and ceiling candidates
            for n_candidate in [n_floor, n_ceil]:
                if n_candidate < 0:
                    continue

                # Calculate total nitrogen atoms added
                total_ri = n_candidate * ri

                # Calculate the total number of atoms after doping
                num_atoms_after_doping = num_atoms_remaining - n_candidate * ci

                if num_atoms_after_doping <= 0:
                    continue  # Skip invalid candidate

                # Calculate the achieved percentage
                achieved_percentage = (total_ri / num_atoms_after_doping) * 100.0

                # Calculate the absolute difference from desired percentage
                delta = abs(achieved_percentage - desired_percentage)

                # Update the best_n if this delta is smaller
                if delta < min_delta:
                    min_delta = delta
                    best_n = n_candidate

            # Assign the best_n for this species
            desired_num_structures[species] = best_n

            # Update the number of atoms remaining
            num_atoms_removed = best_n * ci
            num_atoms_remaining -= num_atoms_removed

        return desired_num_structures, num_atoms_remaining

    def _insert_doping_structures(self, desired_structures: Dict[NitrogenSpecies, int]):
        """
        Insert a specified number of doping structures of a specific nitrogen species into the graphene sheet.

        Parameters
        ----------
        desired_structures : Dict[NitrogenSpecies, int]
            A dictionary where the keys are NitrogenSpecies and the values are the number of structures to insert for
            that species.

        Notes
        -----
        First, a carbon atom is randomly selected. Then, it is checked whether this atom position is suitable for
        building the doping structure around it (i.e., the new structure to be inserted should not overlap with any
        existing structure). If suitable, the doping structure is built by, for example, removing atoms, replacing
        other C atoms with N atoms, and possibly adding new bonds between atoms (in the case of Pyridinic_1). After
        the structure is inserted, all atoms of this structure are excluded from further doping positions.
        """
        # Sort the species by the number of removed carbon atoms as well as the number of contributing nitrogen atoms in
        # decreasing order to insert larger structures first
        species_sorted_by_size = sorted(
            desired_structures.keys(),
            key=lambda s: NitrogenSpecies.get_num_carbon_atoms_to_remove(s)
            and NitrogenSpecies.get_num_nitrogen_atoms_to_add(s),
            reverse=True,
        )

        for species in species_sorted_by_size:
            num_structures = desired_structures.get(species, 0)
            num_structures_inserted = self._attempt_insertion_for_species(species, num_structures)

            # Warn if not all requested structures could be placed due to space constraints
            if num_structures_inserted < num_structures:
                warnings.warn(
                    f"Only {num_structures_inserted} out of the desired {num_structures} structures of "
                    f"species {species.value} could be placed due to space constraints.",
                    UserWarning,
                )

    def _attempt_insertion_for_species(self, species: NitrogenSpecies, num_structures: int):
        """
        Insert a specific number of doping structures for a given species.

        Parameters
        ----------
        species : NitrogenSpecies
            The nitrogen species to insert.
        num_structures : int
            The desired number of structures to insert.

        Notes
        -----
        First, a carbon atom is randomly selected. Then, it is checked whether this atom position is suitable for
        building the doping structure around it (i.e., the new structure to be inserted should not overlap with any
        existing structure). If suitable, the doping structure is built by, for example, removing atoms, replacing
        other C atoms with N atoms, and possibly adding new bonds between atoms (in the case of Pyridinic_1). After
        the structure is inserted, all atoms of this structure are excluded from further doping positions.

        Returns
        -------
        int
            The actual number of structures successfully inserted for the species.
        """
        tested_atoms: Set[int] = set()  # Set to keep track of tested atoms
        structures_inserted = 0  # Counter for the number of structures inserted

        while structures_inserted < num_structures and len(tested_atoms) < len(self.possible_carbon_atoms):
            # Get the next possible carbon atom to test for doping
            atom_id = self.get_next_possible_carbon_atom(self.possible_carbon_atoms, tested_atoms)
            if atom_id is None:
                break  # No more atoms to test

            # Add the atom to the already tested atoms
            tested_atoms.add(atom_id)

            # Check if the atom_id is a valid doping position and return the structural components
            is_valid, structural_components = self._is_valid_doping_site(species, atom_id)
            if not is_valid:
                # No valid doping position found, proceed to the next possible carbon atom
                continue

            # The doping position is valid, proceed with nitrogen doping
            if species == NitrogenSpecies.GRAPHITIC:
                # Handle graphitic doping
                self._handle_graphitic_doping(structural_components)
            else:
                # Handle pyridinic doping
                self._handle_pyridinic_doping(structural_components, species)

            structures_inserted += 1

            # Reset tested_atoms since possible_carbon_atoms may have changed
            tested_atoms = set()

        return structures_inserted

    def _handle_graphitic_doping(self, structural_components: StructuralComponents):
        """
        Handle the graphitic nitrogen doping process.

        This method takes the provided structural components and performs the doping process by converting a selected
        carbon atom to a nitrogen atom. It also marks the affected atoms to prevent further doping in those positions
        and updates the internal data structures accordingly.

        Parameters
        ----------
        structural_components : StructuralComponents
            The structural components required to build the graphitic doping structure. This includes the atom that will
            be changed to nitrogen and its neighboring atoms.
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
            species=NitrogenSpecies.GRAPHITIC,
            structural_components=structural_components,
            nitrogen_atoms=[atom_id],
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

    def _adjust_for_shortfall_in_doping(self, total_percentage: float):
        """
        Insert additional Graphitic-N structures if actual doping percentage falls short of the target.

        Parameters
        ----------
        total_percentage : float
            The target total doping percentage.
        """
        total_atoms_after_doping = self.graph.number_of_nodes()
        total_nitrogen_atoms = sum(len(atoms) for atoms in self.doping_structures.chosen_atoms.values())
        actual_doping_percentage = round((total_nitrogen_atoms / total_atoms_after_doping) * 100, 2)

        if actual_doping_percentage < total_percentage:
            warn_message = (
                "For reasons of space, it is not possible to come closer to the desired total doping "
                "percentage if we strive for an equal distribution of structures. Additional Graphitic-N structures "
                "will be inserted to adjust for the shortfall and get closer to the desired doping "
                "percentage.\nPlease consider whether such a high doping percentage is still practical, as "
                "achieving it may compromise the structural integrity of the material."
            )
            warnings.warn(warn_message, UserWarning)
            shortfall = total_percentage - actual_doping_percentage
            doping_per_graphitic = (
                NitrogenSpecies.get_num_nitrogen_atoms_to_add(NitrogenSpecies.GRAPHITIC) / total_atoms_after_doping
            ) * 100
            additional_graphitic_needed = int(round(shortfall / doping_per_graphitic))

            inserted = self._attempt_insertion_for_species(NitrogenSpecies.GRAPHITIC, additional_graphitic_needed)
            if inserted < additional_graphitic_needed:
                warnings.warn(
                    f"Only {inserted} out of {additional_graphitic_needed} additional Graphitic-N structures could be "
                    "placed due to space constraints.",
                    UserWarning,
                )

    def _update_actual_doping_percentages(self, additional_structures_inserted: int, species: NitrogenSpecies):
        """
        Update the actual doping percentages after adding additional structures.
        """
        total_atoms_after_doping = self.graph.number_of_nodes()
        actual_percentages = {
            species.value: (
                round(
                    (len(self.doping_structures.chosen_atoms.get(species, [])) / total_atoms_after_doping) * 100,
                    2,
                )
                if total_atoms_after_doping > 0
                else 0
            )
            for species in NitrogenSpecies
        }

        # Update Graphitic-N Doping
        if species == NitrogenSpecies.GRAPHITIC:
            actual_percentages[species.value] += round(
                (additional_structures_inserted / total_atoms_after_doping) * 100, 2
            )

        # Recalculate total nitrogen and total doping percentage
        total_nitrogen_atoms = sum(len(atoms) for atoms in self.doping_structures.chosen_atoms.values())
        total_doping_percentage = round((total_nitrogen_atoms / total_atoms_after_doping) * 100, 2)

        return actual_percentages, total_doping_percentage

    def _display_doping_results(self):
        """
        Display the final doping results, including actual percentages, absolute counts of nitrogen atoms added,
        and the number of doping structures per species.
        """
        total_atoms_after_doping = self.graph.number_of_nodes()

        # Calculate nitrogen atom counts and actual percentages
        nitrogen_atom_counts = {
            species: len(self.doping_structures.chosen_atoms.get(species, [])) for species in NitrogenSpecies
        }
        actual_percentages = {
            species.value: (round((count / total_atoms_after_doping) * 100, 2) if total_atoms_after_doping > 0 else 0)
            for species, count in nitrogen_atom_counts.items()
        }

        # Calculate total counts and percentages
        total_nitrogen_atoms = sum(nitrogen_atom_counts.values())
        total_doping_percentage = round((total_nitrogen_atoms / total_atoms_after_doping) * 100, 2)

        # Calculate the number of structures per species using the get_structures_for_species method
        doping_structure_counts = {
            species: len(self.doping_structures.get_structures_for_species(species)) for species in NitrogenSpecies
        }

        # Prepare the DataFrame
        doping_results = {
            "Nitrogen Species": [species.value for species in NitrogenSpecies] + ["Total Doping"],
            "Actual Percentage": list(actual_percentages.values()) + [total_doping_percentage],
            "Nitrogen Atom Count": list(nitrogen_atom_counts.values()) + [total_nitrogen_atoms],
            "Doping Structure Count": list(doping_structure_counts.values()) + [sum(doping_structure_counts.values())],
        }
        doping_results_df = pd.DataFrame(doping_results)

        # Set pandas display options to show all columns without truncation
        pd.set_option("display.max_columns", None)  # Shows all columns
        pd.set_option("display.width", None)  # Adjusts output to the screen width

        print("\nDoping Results:")
        print(doping_results_df)
