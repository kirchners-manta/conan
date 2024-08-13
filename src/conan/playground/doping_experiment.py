import copy
import math
import random

# import time
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field
from math import cos, pi, sin
from typing import Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from networkx.utils import pairwise
from scipy.optimize import minimize
from scipy.spatial import KDTree

from conan.playground.graph_utils import (
    NitrogenSpecies,
    NitrogenSpeciesProperties,
    Position3D,
    create_position,
    get_color,
    get_neighbors_via_edges,
    minimum_image_distance,
    minimum_image_distance_vectorized,
    toggle_dimension,
    write_xyz,
)

# Define a namedtuple for structural components
# This namedtuple will be used to store the atom(s) around which the doping structure is built and its/their neighbors
StructuralComponents = namedtuple("StructuralComponents", ["structure_building_atoms", "structure_building_neighbors"])


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
        List of atom IDs forming the cycle of the doping structure.
    subgraph : Optional[nx.Graph]
        The subgraph containing the doping structure.
    additional_edge : Optional[Tuple[int, int]]
        An additional edge added to the doping structure, needed for PYRIDINIC_1 doping.
    """

    species: NitrogenSpecies
    structural_components: StructuralComponents[List[int], List[int]]
    nitrogen_atoms: List[int]
    cycle: Optional[List[int]] = field(default=None)
    subgraph: Optional[nx.Graph] = field(default=None)
    additional_edge: Optional[Tuple[int, int]] = field(default=None)

    @classmethod
    def create_structure(
        cls,
        graphene: "GrapheneSheet",
        species: NitrogenSpecies,
        structural_components: StructuralComponents[List[int], List[int]],
        start_node: Optional[int] = None,
    ):
        """
        Create a doping structure within the graphene sheet.

        This method creates a doping structure by detecting the cycle in the graph that includes the
        structure-building neighbors, ordering the cycle, and adding any necessary edges.

        Parameters
        ----------
        graphene : GrapheneSheet
            The graphene sheet.
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

        graph = graphene.graph

        # Detect the cycle and create the subgraph
        cycle, subgraph = cls._detect_cycle_and_subgraph(graph, structural_components.structure_building_neighbors)

        # Order the cycle
        ordered_cycle = cls._order_cycle(subgraph, cycle, species, start_node)

        # Add edge if needed (only for PYRIDINIC_1 doping)
        additional_edge = None
        if species == NitrogenSpecies.PYRIDINIC_1:
            additional_edge = cls._add_additional_edge(
                graphene, subgraph, structural_components.structure_building_neighbors, start_node
            )

        # Identify nitrogen atoms in the ordered cycle
        nitrogen_atoms = [node for node in ordered_cycle if graph.nodes[node]["element"] == "N"]

        # Create and return the DopingStructure instance
        return cls(species, structural_components, nitrogen_atoms, ordered_cycle, subgraph, additional_edge)

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
        graphene: "GrapheneSheet", subgraph: nx.Graph, neighbors: List[int], start_node: int
    ) -> Tuple[int, int]:
        """
        Add an edge between neighbors if the nitrogen species is PYRIDINIC_1.

        Parameters
        ----------
        graphene : GrapheneSheet
            The graphene sheet.
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

        graph = graphene.graph

        # Remove the start node from the list of neighbors to get the two neighbors to connect
        neighbors.remove(start_node)

        # Get the positions of the two remaining neighbors
        pos1 = graph.nodes[neighbors[0]]["position"]
        pos2 = graph.nodes[neighbors[1]]["position"]

        # Calculate the box size for periodic boundary conditions
        box_size = (
            graphene.actual_sheet_width + graphene.c_c_bond_distance,
            graphene.actual_sheet_height + graphene.cc_y_distance,
        )

        # Calculate the bond length between the two neighbors considering minimum image distance
        bond_length, _ = minimum_image_distance(pos1, pos2, box_size)

        # Add the edge to the main graph and the subgraph
        graph.add_edge(neighbors[0], neighbors[1], bond_length=bond_length)
        subgraph.add_edge(neighbors[0], neighbors[1], bond_length=bond_length)

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

        # Initialize the subgraph with the neighbors and their edges
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

    def add_structure(self, dopings_structure: DopingStructure):
        """
        Add a doping structure to the collection and update the chosen atoms.
        """

        self.structures.append(dopings_structure)
        self.chosen_atoms[dopings_structure.species].extend(dopings_structure.nitrogen_atoms)

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


# Abstract base class for material structures
class MaterialStructure(ABC):
    def __init__(self):
        self.graph = nx.Graph()
        """The networkx graph representing the structure of the material (e.g., graphene sheet)."""

    @abstractmethod
    def build_structure(self):
        pass

    @abstractmethod
    def plot_structure(self, with_labels: bool = False, visualize_periodic_bonds: bool = True):
        pass


# Abstract base class for 2D structures
class Structure2D(MaterialStructure):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def build_structure(self):
        pass

    def plot_structure(self, with_labels: bool = False, visualize_periodic_bonds: bool = True):
        """
        Plot the structure using networkx and matplotlib in 2D.

        Parameters
        ----------
        with_labels : bool, optional
            Whether to display labels on the nodes (default is False).
        visualize_periodic_bonds : bool, optional
            Whether to visualize periodic boundary condition edges (default is True).

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

        # Initialize plot with an Axes object
        fig, ax = plt.subplots(figsize=(12, 12))

        # Draw the regular edges
        nx.draw(self.graph, pos_2d, edgelist=regular_edges, node_color=colors, node_size=200, with_labels=False, ax=ax)

        # Draw periodic edges with dashed lines if visualize_periodic_bonds is True
        if visualize_periodic_bonds:
            nx.draw_networkx_edges(
                self.graph, pos_2d, edgelist=periodic_edges, style="dashed", edge_color="gray", ax=ax
            )

        # Add legend
        unique_colors = set(colors)
        legend_elements = []
        for species in NitrogenSpecies:
            color = get_color("N", species)
            if color in unique_colors:
                legend_elements.append(
                    plt.Line2D(
                        [0], [0], marker="o", color="w", label=species.value, markersize=10, markerfacecolor=color
                    )
                )
        ax.legend(handles=legend_elements, title="Nitrogen Doping Species")

        # Add labels if specified
        if with_labels:
            labels = {node: f"{elements[node]}{node}" for node in self.graph.nodes()}
            nx.draw_networkx_labels(
                self.graph, pos_2d, labels=labels, font_size=10, font_color="cyan", font_weight="bold", ax=ax
            )

        # Manually add x- and y-axis labels using ax.text
        x_min, x_max = min(x for x, y in pos_2d.values()), max(x for x, y in pos_2d.values())
        y_min, y_max = min(y for x, y in pos_2d.values()), max(y for x, y in pos_2d.values())

        ax.text((x_min + x_max) / 2, y_min - (y_max - y_min) * 0.1, "X [Å]", fontsize=14, ha="center")
        ax.text(
            x_min - (x_max - x_min) * 0.1, (y_min + y_max) / 2, "Y [Å]", fontsize=14, va="center", rotation="vertical"
        )

        # Adjust layout to make sure everything fits
        plt.tight_layout()

        # Show the plot
        plt.show()


# Abstract base class for 3D structures
class Structure3D(MaterialStructure):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def build_structure(self):
        pass

    def plot_structure(self, with_labels: bool = False, visualize_periodic_bonds: bool = True):
        """
        Plot the structure in 3D using networkx and matplotlib.

        Parameters
        ----------
        with_labels : bool, optional
            Whether to display labels on the nodes (default is False).
        visualize_periodic_bonds : bool, optional
            Whether to visualize periodic boundary condition edges (default is True).

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

        # Initialize 3D plot
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection="3d")

        # Extract node positions
        xs, ys, zs = zip(*[pos[node] for node in self.graph.nodes()])

        # Draw nodes in one step
        ax.scatter(xs, ys, zs, color=colors, s=20)

        # Create line segments for regular edges
        regular_segments = np.array(
            [[(pos[u][0], pos[u][1], pos[u][2]), (pos[v][0], pos[v][1], pos[v][2])] for u, v in regular_edges]
        )
        regular_lines = Line3DCollection(regular_segments, colors="black")
        ax.add_collection3d(regular_lines)

        # Create line segments for periodic edges if visualize_periodic_bonds is True
        if visualize_periodic_bonds:
            periodic_segments = np.array(
                [[(pos[u][0], pos[u][1], pos[u][2]), (pos[v][0], pos[v][1], pos[v][2])] for u, v in periodic_edges]
            )
            periodic_lines = Line3DCollection(periodic_segments, colors="gray", linestyles="dashed")
            ax.add_collection3d(periodic_lines)

        # Add labels if specified
        if with_labels:
            for node in self.graph.nodes():
                ax.text(pos[node][0], pos[node][1], pos[node][2], f"{elements[node]}{node}", color="cyan")

        # Set the axes labels
        ax.set_xlabel("X [Å]")
        ax.set_ylabel("Y [Å]")
        ax.set_zlabel("Z [Å]")

        # Add a legend for the nitrogen species
        unique_colors = set(colors)
        legend_elements = []
        for species in NitrogenSpecies:
            color = get_color("N", species)
            if color in unique_colors:
                legend_elements.append(
                    plt.Line2D(
                        [0], [0], marker="o", color="w", label=species.value, markersize=10, markerfacecolor=color
                    )
                )
        ax.legend(handles=legend_elements, title="Nitrogen Doping Species")

        # Show the plot
        plt.show()


class GrapheneSheet(Structure2D):
    """
    Represents a graphene sheet structure and manages nitrogen doping within the sheet.
    """

    def __init__(self, bond_distance: Union[float, int], sheet_size: Union[Tuple[float, float], Tuple[int, int]]):
        """
        Initialize the GrapheneGraph with given bond distance and sheet size.

        Parameters
        ----------
        bond_distance : Union[float, int]
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

        # Perform validations
        self._validate_bond_distance(bond_distance)
        self._validate_sheet_size(sheet_size)

        self.c_c_bond_distance = bond_distance
        """The bond distance between carbon atoms in the graphene sheet."""
        self.c_c_bond_angle = 120
        """The bond angle between carbon atoms in the graphene sheet."""
        self.sheet_size = sheet_size
        """The size of the graphene sheet in the x and y directions."""
        self.k_inner_bond = 10
        """The spring constant for bonds within the doping structure."""
        self.k_outer_bond = 0.1
        """The spring constant for bonds outside the doping structure."""
        self.k_inner_angle = 10
        """The spring constant for angles within the doping structure."""
        self.k_outer_angle = 0.1
        """The spring constant for angles outside the doping structure."""
        # self.graph = nx.Graph()
        # """The networkx graph representing the graphene sheet structure."""
        # self._build_graphene_sheet()  # Build the initial graphene sheet structure

        # Build the initial graphene sheet structure
        self.build_structure()

        # Initialize the list of possible carbon atoms
        self._possible_carbon_atoms_needs_update = True
        """Flag to indicate that the list of possible carbon atoms needs to be updated."""
        self._possible_carbon_atoms = []
        """List of possible carbon atoms that can be used for nitrogen doping."""

        self.species_properties = self._initialize_species_properties()
        """A dictionary mapping each NitrogenSpecies to its corresponding NitrogenSpeciesProperties.
        This includes bond lengths and angles characteristic to each species that we aim to achieve in the doping."""

        self.doping_structures = DopingStructureCollection()
        """A dataclass to store information about doping structures in the graphene sheet."""

        # Initialize positions and KDTree for efficient neighbor search
        self._positions = np.array([self.graph.nodes[node]["position"] for node in self.graph.nodes])
        # self._positions = np.array([self.graph.nodes[node]['position'].to_tuple() for node in self.graph.nodes])
        """The positions of atoms in the graphene sheet."""
        self.kdtree = KDTree(self._positions)  # ToDo: Solve problem with periodic boundary conditions
        """The KDTree data structure for efficient nearest neighbor search. A KDTree is particularly efficient for
        spatial queries, such as searching for neighbors within a certain Euclidean distance. Such queries are often
        computationally intensive when performed over a graph, especially when dealing with direct distance rather than
        path lengths in the graph."""

        self.include_outer_angles = False  # ToDo: Delete later; just for testing purposes

    @property
    def positions(self):
        return self._positions

    @positions.setter
    def positions(self, new_positions):
        """Update the positions of atoms and rebuild the KDTree for efficient spatial queries."""
        self._positions = new_positions
        self.kdtree = KDTree(new_positions)

    @property
    def cc_x_distance(self):
        """Calculate the distance between atoms in the x direction."""
        return self.c_c_bond_distance * sin(pi / 6)

    @property
    def cc_y_distance(self):
        """Calculate the distance between atoms in the y direction."""
        return self.c_c_bond_distance * cos(pi / 6)

    @property
    def num_cells_x(self):
        """Calculate the number of unit cells in the x direction based on sheet size and bond distance."""
        return int(self.sheet_size[0] // (2 * self.c_c_bond_distance + 2 * self.cc_x_distance))

    @property
    def num_cells_y(self):
        """Calculate the number of unit cells in the y direction based on sheet size and bond distance."""
        return int(self.sheet_size[1] // (2 * self.cc_y_distance))

    @property
    def actual_sheet_width(self):
        """Calculate the actual width of the graphene sheet based on the number of unit cells and bond distance."""
        return self.num_cells_x * (2 * self.c_c_bond_distance + 2 * self.cc_x_distance) - self.c_c_bond_distance

    @property
    def actual_sheet_height(self):
        """Calculate the actual height of the graphene sheet based on the number of unit cells and bond distance."""
        return self.num_cells_y * (2 * self.cc_y_distance) - self.cc_y_distance

    @property
    def possible_carbon_atoms(self):
        """Get the list of possible carbon atoms for doping."""
        if self._possible_carbon_atoms_needs_update:
            self._update_possible_carbon_atoms()
        return self._possible_carbon_atoms

    @staticmethod
    def _validate_bond_distance(bond_distance: float):
        """Validate the bond distance."""
        if not isinstance(bond_distance, (int, float)):
            raise TypeError(f"bond_distance must be a float or int, but got {type(bond_distance).__name__}.")
        if bond_distance <= 0:
            raise ValueError(f"bond_distance must be positive, but got {bond_distance}.")

    @staticmethod
    def _validate_sheet_size(sheet_size: Tuple[float, float]):
        """Validate the sheet size."""
        if not isinstance(sheet_size, tuple):
            raise TypeError("sheet_size must be a tuple of exactly two positive floats or ints.")
        if len(sheet_size) != 2:  # Überprüfen, ob das Tupel genau zwei Elemente hat
            raise TypeError("sheet_size must be a tuple of exactly two positive floats or ints.")
        if not all(isinstance(i, (int, float)) for i in sheet_size):
            raise TypeError("sheet_size must be a tuple of exactly two positive floats or ints.")
        if any(s <= 0 for s in sheet_size):
            raise ValueError(f"All elements of sheet_size must be positive, but got {sheet_size}.")

    def _validate_structure(self):
        """Validate the structure to ensure it can fit within the given sheet size."""
        if self.num_cells_x < 1 or self.num_cells_y < 1:
            raise ValueError(
                f"Sheet size is too small to fit even a single unit cell. Got sheet size {self.sheet_size}."
            )

    def _update_possible_carbon_atoms(self):
        """Update the list of possible carbon atoms for doping."""
        self._possible_carbon_atoms = [
            node for node, data in self.graph.nodes(data=True) if data.get("possible_doping_site")
        ]
        self._possible_carbon_atoms_needs_update = False

    def mark_possible_carbon_atoms_for_update(self):
        """Mark the list of possible carbon atoms as needing an update."""
        self._possible_carbon_atoms_needs_update = True

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
                x_offset = x * (2 * self.c_c_bond_distance + 2 * self.cc_x_distance)
                y_offset = y * (2 * self.cc_y_distance)

                # Add nodes and edges for the unit cell
                self._add_unit_cell(index, x_offset, y_offset)

                # Add horizontal bonds between adjacent unit cells
                if x > 0:
                    self.graph.add_edge(index - 1, index, bond_length=self.c_c_bond_distance)

                # Add vertical bonds between unit cells in adjacent rows
                if y > 0:
                    self.graph.add_edge(index - 4 * self.num_cells_x + 1, index, bond_length=self.c_c_bond_distance)
                    self.graph.add_edge(index - 4 * self.num_cells_x + 2, index + 3, bond_length=self.c_c_bond_distance)

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
            create_position(x_offset + self.cc_x_distance + self.c_c_bond_distance, y_offset + self.cc_y_distance),
            create_position(x_offset + 2 * self.cc_x_distance + self.c_c_bond_distance, y_offset),
        ]

        # Add nodes with positions, element type (carbon) and possible doping site flag
        nodes = [
            (index + i, {"element": "C", "position": pos, "possible_doping_site": True})
            for i, pos in enumerate(unit_cell_positions)
        ]
        self.graph.add_nodes_from(nodes)

        # Add internal bonds within the unit cell
        edges = [
            (index + i, index + i + 1, {"bond_length": self.c_c_bond_distance})
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
            zip(right_edge_indices, left_edge_indices), bond_length=self.c_c_bond_distance, periodic=True
        )

        # Generate base indices for vertical boundaries
        top_left_indices = np.arange(self.num_cells_x) * 4
        bottom_left_indices = top_left_indices + (self.num_cells_y - 1) * num_nodes_x + 1
        bottom_right_indices = top_left_indices + (self.num_cells_y - 1) * num_nodes_x + 2

        # Add vertical periodic boundary conditions
        self.graph.add_edges_from(
            zip(bottom_left_indices, top_left_indices), bond_length=self.c_c_bond_distance, periodic=True
        )
        self.graph.add_edges_from(
            zip(bottom_right_indices, top_left_indices + 3), bond_length=self.c_c_bond_distance, periodic=True
        )

    @staticmethod
    def _initialize_species_properties() -> Dict[NitrogenSpecies, NitrogenSpeciesProperties]:
        # Initialize properties for PYRIDINIC_4 nitrogen species with target bond lengths and angles
        pyridinic_4_properties = NitrogenSpeciesProperties(
            target_bond_lengths=[1.45, 1.34, 1.32, 1.47, 1.32, 1.34, 1.45, 1.45, 1.34, 1.32, 1.47, 1.32, 1.34, 1.45],
            target_angles=[
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
        )
        # Initialize properties for PYRIDINIC_3 nitrogen species with target bond lengths and angles
        pyridinic_3_properties = NitrogenSpeciesProperties(
            target_bond_lengths=[1.45, 1.33, 1.33, 1.45, 1.45, 1.33, 1.33, 1.45, 1.45, 1.33, 1.33, 1.45],
            target_angles=[
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
        )
        # Initialize properties for PYRIDINIC_2 nitrogen species with target bond lengths and angles
        pyridinic_2_properties = NitrogenSpeciesProperties(
            target_bond_lengths=[1.39, 1.42, 1.42, 1.33, 1.35, 1.44, 1.44, 1.35, 1.33, 1.42, 1.42, 1.39],
            target_angles=[
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
        )
        # Initialize properties for PYRIDINIC_1 nitrogen species with target bond lengths and angles
        pyridinic_1_properties = NitrogenSpeciesProperties(
            target_bond_lengths=[1.31, 1.42, 1.45, 1.51, 1.42, 1.40, 1.40, 1.42, 1.51, 1.45, 1.42, 1.31, 1.70],
            target_angles=[
                115.48,
                118.24,
                128.28,
                109.52,
                112.77,
                110.35,
                112.76,
                109.52,
                128.28,
                118.24,
                115.48,
                120.92,
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
    def get_next_possible_carbon_atom(atom_list):
        """
        Get a randomly selected carbon atom from the list of possible carbon atoms.

        This method randomly selects a carbon atom from the provided list and removes it from the list.
        This ensures that the same atom is not selected more than once.

        Parameters
        ----------
        atom_list : list
            The list of possible carbon atoms to select from.

        Returns
        -------
        int or None
            The ID of the selected carbon atom, or None if the list is empty.
        """

        if not atom_list:
            return None  # Return None if the list is empty
        atom_id = random.choice(atom_list)  # Randomly select an atom ID from the list
        atom_list.remove(atom_id)  # Remove the selected atom ID from the list
        return atom_id  # Return the selected atom ID

    def add_nitrogen_doping(self, total_percentage: float = None, percentages: dict = None):
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
                        f"Invalid value in percentages dictionary for key {key}: {value}. Values must be int or float."
                    )

        # Validate the input for total_percentage
        if total_percentage is not None and not isinstance(total_percentage, (int, float)):
            raise ValueError("total_percentage must be an int or float.")

        # Validate specific percentages and calculate the remaining percentage
        if percentages:
            if total_percentage is None:
                # Set total to sum of specific percentages if not provided
                total_percentage = sum(percentages.values())
            else:
                # Sum of provided specific percentages
                specific_total_percentage = sum(percentages.values())
                if specific_total_percentage > total_percentage:
                    # Raise an error if the sum of specific percentages exceeds the total percentage
                    raise ValueError(
                        f"The total specific percentages {specific_total_percentage}% are higher than the "
                        f"total_percentage {total_percentage}%. Please adjust your input so that the sum of the "
                        f"'percentages' is less than or equal to 'total_percentage'."
                    )
        else:
            # Set a default total percentage if not provided
            if total_percentage is None:
                total_percentage = 10  # Default total percentage
            # Initialize an empty dictionary if no specific percentages are provided
            percentages = {}

        # Calculate the remaining percentage for other species
        remaining_percentage = total_percentage - sum(percentages.values())

        if remaining_percentage > 0:
            # Determine available species not included in the specified percentages
            available_species = [species for species in NitrogenSpecies if species not in percentages]
            # Distribute the remaining percentage equally among available species
            default_distribution = {
                species: remaining_percentage / len(available_species) for species in available_species
            }
            # Add the default distribution to the specified percentages
            for species, pct in default_distribution.items():
                if species not in percentages:
                    percentages[species] = pct

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
        for species in [
            NitrogenSpecies.PYRIDINIC_4,
            NitrogenSpecies.PYRIDINIC_3,
            NitrogenSpecies.PYRIDINIC_2,
            NitrogenSpecies.PYRIDINIC_1,
            NitrogenSpecies.GRAPHITIC,
        ]:
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

        # Adjust the positions of atoms in all cycles to optimize the structure
        if any(self.doping_structures.structures):
            self._adjust_atom_positions()

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

        # Create a copy of the possible carbon atoms to test for doping
        possible_carbon_atoms_to_test = self.possible_carbon_atoms.copy()

        # Loop until the required number of nitrogen atoms is added or there are no more possible carbon atoms to test
        while (
            len(self.doping_structures.chosen_atoms[nitrogen_species]) < num_nitrogen and possible_carbon_atoms_to_test
        ):
            # Get a valid doping placement for the current nitrogen species and return the structural components
            is_valid, structural_components = self._find_valid_doping_position(
                nitrogen_species, possible_carbon_atoms_to_test
            )
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
            self,
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

    def _adjust_atom_positions(self):
        """
        Adjust the positions of atoms in the graphene sheet to optimize the structure including doping.

        Notes
        -----
        This method adjusts the positions of atoms in a graphene sheet to optimize the structure based on the doping
        configuration. It uses a combination of bond and angle energies to minimize the total energy of the system.
        """

        # Get all doping structures except graphitic nitrogen (graphitic nitrogen does not affect the structure)
        all_structures = [
            structure
            for structure in self.doping_structures.structures
            if structure.species != NitrogenSpecies.GRAPHITIC
        ]

        # Return if no doping structures are present
        if not all_structures:
            return

        # Get the initial positions of atoms
        positions = {node: self.graph.nodes[node]["position"] for node in self.graph.nodes}
        # Flatten the positions into a 1D array for optimization
        x0 = np.array([coord for node in self.graph.nodes for coord in [positions[node].x, positions[node].y]])
        # Define the box size for minimum image distance calculation
        box_size = (self.actual_sheet_width + self.c_c_bond_distance, self.actual_sheet_height + self.cc_y_distance)

        def bond_energy(x):
            """
            Calculate the bond energy for the given positions.

            Parameters
            ----------
            x : ndarray
                Flattened array of positions of all atoms in the cycle.

            Returns
            -------
            energy : float
                The total bond energy.
            """

            energy = 0.0

            # Initialize a set to track edges within cycles
            cycle_edges = set()

            # Collect all edges and their properties
            all_edges_in_order = []
            all_target_bond_lengths = []

            for structure in all_structures:
                # Get the target bond lengths for the specific nitrogen species
                properties = self.species_properties[structure.species]
                target_bond_lengths = properties.target_bond_lengths
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
            energy += 0.5 * self.k_inner_bond * np.sum((current_lengths - target_lengths) ** 2)

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
                energy += 0.5 * self.k_outer_bond * np.sum((current_lengths - target_lengths) ** 2)

                # Prepare bond length updates for non-cycle edges
                edge_updates = {
                    (node_i, node_j): {"bond_length": current_lengths[idx]}
                    for idx, (node_i, node_j) in enumerate(non_cycle_edges)
                }
                # Update the bond lengths in the graph for non-cycle edges
                nx.set_edge_attributes(self.graph, edge_updates)

            return energy

        def angle_energy(x):
            """
            Calculate the angle energy for the given positions.

            Parameters
            ----------
            x : ndarray
                Flattened array of positions of all atoms in the cycle.

            Returns
            -------
            energy : float
                The total angle energy.
            """

            energy = 0.0

            # Initialize lists to collect all triplets of nodes and their target angles
            all_triplets = []
            all_target_angles = []

            # Initialize a set to track angles within cycles
            counted_angles = set()  # ToDo: Delete later if outer angles are not needed

            # Iterate over all doping structures to gather triplets and target angles
            for structure in all_structures:
                properties = self.species_properties[structure.species]
                target_angles = properties.target_angles
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
            energy += 0.5 * self.k_inner_angle * np.sum((theta - target_angles) ** 2)

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
                            _, v1 = minimum_image_distance(pos_i, pos_node, box_size)
                            _, v2 = minimum_image_distance(pos_j, pos_node, box_size)
                            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                            theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
                            energy += 0.5 * self.k_outer_angle * ((theta - np.radians(self.c_c_bond_angle)) ** 2)

            return energy

        def total_energy(x):
            """
            Calculate the total energy (bond energy + angle energy) for the given positions.

            Parameters
            ----------
            x : ndarray
                Flattened array of positions of all atoms in the cycle.

            Returns
            -------
            energy : float
                The total energy.
            """

            return bond_energy(x) + angle_energy(x)

        # Use L-BFGS-B optimization method to minimize the total energy
        result = minimize(total_energy, x0, method="L-BFGS-B")
        print(f"Number of iterations: {result.nit}\nFinal energy: {result.fun}")

        # Reshape the optimized positions back to the 2D array format
        optimized_positions = result.x.reshape(-1, 2)

        # Update the positions of atoms in the graph with the optimized positions using NetworkX set_node_attributes
        position_dict = {
            node: create_position(optimized_positions[idx][0], optimized_positions[idx][1])
            for idx, node in enumerate(self.graph.nodes)
        }
        nx.set_node_attributes(self.graph, position_dict, "position")

    def _find_valid_doping_position(
        self, nitrogen_species: NitrogenSpecies, possible_carbon_atoms_to_test: List[int]
    ) -> Tuple[bool, StructuralComponents]:
        """
        Determine if a given position is valid for nitrogen doping based on the nitrogen species and atom position.

        This method tests possible carbon atoms for doping by checking their proximity constraints
        based on the type of nitrogen species. If a valid position is found, it returns True along with
        the structural components needed for doping. Otherwise, it returns False.

        Parameters
        ----------
        nitrogen_species : NitrogenSpecies
            The type of nitrogen doping to validate.
        possible_carbon_atoms_to_test : List[int]
            The list of possible carbon atoms to test.

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

        def all_neighbors_possible_carbon_atoms(neighbors: List[int]):
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

        # Get the next possible carbon atom to test for doping and its neighbors
        atom_id = self.get_next_possible_carbon_atom(possible_carbon_atoms_to_test)
        neighbors = get_neighbors_via_edges(self.graph, atom_id)

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
            # Ensure all neighbors are possible atoms for doping
            if all_neighbors_possible_carbon_atoms(neighbors_len_2):
                # Return True if the position is valid for pyridinic doping and the structural components
                return True, StructuralComponents(
                    structure_building_atoms=[atom_id], structure_building_neighbors=neighbors
                )
            # Return False if the position is not valid for pyridinic doping
            return False, (None, None)

        elif nitrogen_species == NitrogenSpecies.PYRIDINIC_4:
            # Iterate over the neighbors of the selected atom to find a direct neighbor that has a valid position
            selected_neighbor = None
            temp_neighbors = neighbors.copy()

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

            # Return True if the position is valid for pyridinic 4 doping
            return True, StructuralComponents(
                structure_building_atoms=[atom_id, selected_neighbor], structure_building_neighbors=combined_neighbors
            )

        # Return False if the nitrogen species is not recognized
        return False, (None, None)

    def stack(self, interlayer_spacing: float, number_of_layers: int) -> "StackedGraphene":
        """
        Stack graphene sheets using ABA stacking.

        Parameters
        ----------
        interlayer_spacing : float
            The shift in the z-direction for each layer.
        number_of_layers : int
            The number of layers to stack.

        Returns
        -------
        StackedGraphene
            The stacked graphene structure.
        """
        return StackedGraphene(self, interlayer_spacing, number_of_layers)


class StackedGraphene(Structure3D):
    """
    Represents a stacked graphene structure.
    """

    def __init__(self, graphene_sheet: GrapheneSheet, interlayer_spacing: float, number_of_layers: int):
        """
        Initialize the StackedGraphene with a base graphene sheet, interlayer spacing, and number of layers.

        Parameters
        ----------
        graphene_sheet : GrapheneSheet
            The base graphene sheet to be stacked.
        interlayer_spacing : float
            The spacing between layers in the z-direction.
        number_of_layers : int
            The number of layers to stack.
        """
        super().__init__()
        self.graphene_sheets = []
        """A list to hold individual GrapheneSheet instances."""
        self.interlayer_spacing = interlayer_spacing
        """The spacing between layers in the z-direction."""
        self.number_of_layers = number_of_layers
        """The number of layers to stack."""

        # Add the original graphene sheet as the first layer
        toggle_dimension(graphene_sheet.graph)
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
        interlayer_shift = 1.42  # Fixed x_shift for ABA stacking  # ToDo: Anpassen, dass hier self.bond_distance steht
        x_shift = (layer % 2) * interlayer_shift
        z_shift = layer * self.interlayer_spacing

        # Update the positions in the copied sheet
        for node, pos in sheet.graph.nodes(data="position"):
            shifted_pos = Position3D(pos.x + x_shift, pos.y, pos.z + z_shift)
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

    def add_nitrogen_doping_to_layer(self, layer_index: int, total_percentage: float):
        """
        Add nitrogen doping to a specific layer in the stacked graphene structure.

        Parameters
        ----------
        layer_index : int
            The index of the layer to dope.
        total_percentage : float
            The total percentage of carbon atoms to replace with nitrogen atoms.
        """
        if 0 <= layer_index < len(self.graphene_sheets):
            # Convert to 2D before doping
            toggle_dimension(self.graphene_sheets[layer_index].graph)

            # Perform the doping
            self.graphene_sheets[layer_index].add_nitrogen_doping(total_percentage=total_percentage)

            # Convert back to 3D after doping
            toggle_dimension(self.graphene_sheets[layer_index].graph)
            self._shift_sheet(self.graphene_sheets[layer_index], layer_index)

            # Rebuild the main graph in order to update the structure after doping
            self.build_structure()
        else:
            raise IndexError("Layer index out of range.")


class CNT(Structure3D):
    """
    Represents a carbon nanotube structure.
    """

    def __init__(self, bond_length: float, tube_length: float, tube_size: int, conformation: str):
        """
        Initialize the CarbonNanotube with given parameters.

        Parameters
        ----------
        bond_length : float
            The bond length between carbon atoms in the CNT.
        tube_length : float
            The length of the CNT.
        tube_size : int
            The size of the CNT, i.e., the number of hexagonal units around the circumference.
        conformation : str
            The conformation of the CNT ('armchair' or 'zigzag').
        """
        super().__init__()
        self.bond_length = bond_length
        self.tube_length = tube_length
        self.tube_size = tube_size
        self.conformation = conformation.lower()

        # Build the CNT structure using graph theory
        self.build_structure()

    def build_structure(self):
        """
        Build the CNT structure based on the given parameters.
        """
        if self.conformation not in ["armchair", "zigzag"]:
            raise ValueError("Invalid conformation. Choose either 'armchair' or 'zigzag'.")

        # Berechne Abstand und Radius basierend auf der Bindungslänge
        distance = self.bond_length
        hex_d = distance * math.cos(30 * math.pi / 180) * 2
        symmetry_angle = 360 / self.tube_size

        # Initialisiere Variablen
        z_max = 0
        counter = 0
        positions = []

        if self.conformation == "armchair":
            angle_carbon_bond = 360 / (self.tube_size * 3)
            radius = distance / (2 * math.sin((angle_carbon_bond * math.pi / 180) / 2))
            distx = radius - radius * math.cos(angle_carbon_bond / 2 * math.pi / 180)
            disty = 0 - radius * math.sin(angle_carbon_bond / 2 * math.pi / 180)
            zstep = (distance**2 - distx**2 - disty**2) ** 0.5

            while z_max < self.tube_length:
                z_coordinate = zstep * 2 * counter

                for i in range(self.tube_size):
                    angle = symmetry_angle * math.pi / 180 * i
                    x = radius * math.cos(angle)
                    y = radius * math.sin(angle)
                    positions.append((x, y, z_coordinate))

                    angle = (symmetry_angle * i + angle_carbon_bond) * math.pi / 180
                    x = radius * math.cos(angle)
                    y = radius * math.sin(angle)
                    positions.append((x, y, z_coordinate))

                    angle = (symmetry_angle * i + angle_carbon_bond * 3 / 2) * math.pi / 180
                    x = radius * math.cos(angle)
                    y = radius * math.sin(angle)
                    z = zstep + z_coordinate
                    positions.append((x, y, z))

                    angle = (symmetry_angle * i + angle_carbon_bond * 5 / 2) * math.pi / 180
                    x = radius * math.cos(angle)
                    y = radius * math.sin(angle)
                    z = zstep + z_coordinate
                    positions.append((x, y, z))

                z_max = z_coordinate + zstep
                counter += 1

        elif self.conformation == "zigzag":
            radius = hex_d / (2 * math.sin((symmetry_angle * math.pi / 180) / 2))
            distx = radius - radius * math.cos(symmetry_angle / 2 * math.pi / 180)
            disty = 0 - radius * math.sin(symmetry_angle / 2 * math.pi / 180)
            zstep = (distance**2 - distx**2 - disty**2) ** 0.5

            while z_max < self.tube_length:
                z_coordinate = (2 * zstep + distance * 2) * counter

                for i in range(self.tube_size):
                    angle = symmetry_angle * math.pi / 180 * i
                    x = radius * math.cos(angle)
                    y = radius * math.sin(angle)
                    positions.append((x, y, z_coordinate))

                    angle = (symmetry_angle * i + symmetry_angle / 2) * math.pi / 180
                    x = radius * math.cos(angle)
                    y = radius * math.sin(angle)
                    z = zstep + z_coordinate
                    positions.append((x, y, z))

                    angle = (symmetry_angle * i + symmetry_angle / 2) * math.pi / 180
                    x = radius * math.cos(angle)
                    y = radius * math.sin(angle)
                    z = zstep + distance + z_coordinate
                    positions.append((x, y, z))

                    angle = symmetry_angle * math.pi / 180 * i
                    x = radius * math.cos(angle)
                    y = radius * math.sin(angle)
                    z = 2 * zstep + distance + z_coordinate
                    positions.append((x, y, z))

                z_max = z_coordinate + zstep
                counter += 1

        # Positionen im Graph speichern
        for idx, (x, y, z) in enumerate(positions):
            pos = Position3D(x, y, z)
            self.graph.add_node(idx, element="C", position=pos)

        # Interne Bindungen innerhalb der Einheitszellen
        for idx in range(0, len(positions), 4):
            self.graph.add_edge(idx, idx + 1, bond_length=self.bond_length)
            self.graph.add_edge(idx + 1, idx + 2, bond_length=self.bond_length)
            self.graph.add_edge(idx + 2, idx + 3, bond_length=self.bond_length)

        # Zusätzliche Verbindungen zwischen den Einheitszellen nur innerhalb derselben Ebene
        for idx in range(0, len(positions) - 4, 4):
            if positions[idx][2] == positions[idx + 4][2]:  # Verbindungen nur innerhalb derselben z-Ebene
                self.graph.add_edge(idx + 1, idx + 4, bond_length=self.bond_length)
                self.graph.add_edge(idx + 2, idx + 7, bond_length=self.bond_length)

        # Abschluss der Ebenen-Verbindungen, um den Kreis zu schließen
        for idx in range(0, len(positions), 4 * self.tube_size):
            # Verbinde den vorletzten und vorvorletzten Knoten der letzten Einheitszelle
            # mit dem ersten und letzten Knoten der ersten Einheitszelle
            first_idx_first_cell_of_cycle = idx
            first_idx_last_cell_of_cycle = idx + 4 * self.tube_size - 4

            self.graph.add_edge(
                first_idx_last_cell_of_cycle + 1, first_idx_first_cell_of_cycle, bond_length=self.bond_length
            )  # vorletzter Knoten mit dem ersten Knoten
            self.graph.add_edge(
                first_idx_last_cell_of_cycle + 2, first_idx_first_cell_of_cycle + 3, bond_length=self.bond_length
            )  # letzter Knoten mit dem letzten Knoten

        # Verbindungen zwischen den Ebenen herstellen
        for idx in range(0, len(positions) - 4 * self.tube_size, 4 * self.tube_size):
            for i in range(self.tube_size):
                # Verbindung des letzten Knotens einer Zelle in der aktuellen Ebene
                # mit dem ersten Knoten der entsprechenden Zelle in der darüberliegenden Ebene
                self.graph.add_edge(idx + 3 + 4 * i, idx + 4 * i + 4 * self.tube_size, bond_length=self.bond_length)

    def plot_structure(self, with_labels=False):
        """
        Plot the CNT structure in 3D using matplotlib.

        Parameters
        ----------
        with_labels : bool, optional
            If True, labels the nodes with their index.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        pos = nx.get_node_attributes(self.graph, "position")
        xs, ys, zs = [], [], []

        for node in self.graph.nodes():
            p = pos[node]
            xs.append(p.x)
            ys.append(p.y)
            zs.append(p.z)

        xs = np.array(xs)
        ys = np.array(ys)
        zs = np.array(zs)

        ax.scatter(xs, ys, zs, c="r", marker="o")

        for edge in self.graph.edges():
            p1 = pos[edge[0]]
            p2 = pos[edge[1]]
            ax.plot([p1.x, p2.x], [p1.y, p2.y], [p1.z, p2.z], color="b")

        # Set equal aspect ratio for all axes
        ax.set_box_aspect([1, 1, 1])

        # Optional: Change the perspective
        ax.view_init(elev=20, azim=30)

        # Autoscale the plot based on the positions
        ax.auto_scale_xyz(xs, ys, zs)

        if with_labels:
            for node, (x, y, z) in zip(self.graph.nodes(), zip(xs, ys, zs)):
                ax.text(x, y, z, s=str(node), fontsize=10)

        plt.show()


def main():
    # Set seed for reproducibility
    # random.seed(42)
    # random.seed(3)
    random.seed(0)

    # sheet_size = (5, 5)

    ####################################################################################################################
    # # VERSION 1:
    #
    # # Create a graphene sheet
    # graphene = GrapheneSheet(bond_distance=1.42, sheet_size=sheet_size)
    #
    # # Add nitrogen doping to the graphene sheet
    # start_time = time.time()  # Time the nitrogen doping process
    # graphene.add_nitrogen_doping(total_percentage=15)
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
    # # VERSION 2:
    # # Create individual GrapheneSheet instances
    # graphene = GrapheneSheet(bond_distance=1.42, sheet_size=sheet_size)
    #
    # # Add nitrogen doping to the graphene sheet
    # start_time = time.time()  # Time the nitrogen doping process
    # graphene.add_nitrogen_doping(total_percentage=15)
    # end_time = time.time()
    #
    # # Calculate the elapsed time
    # elapsed_time = end_time - start_time
    # print(f"Time taken for nitrogen doping for a sheet of size {sheet_size}: {elapsed_time:.2f} seconds")
    #
    # # Stack sheets into a 3D structure
    # graphene.stack(interlayer_spacing=3.35, number_of_layers=3)
    #
    # # Plot the stacked structure
    # graphene.plot_structure(with_labels=True, visualize_periodic_bonds=False)
    #
    # write_xyz(graphene.graph, "ABA_stacking.xyz")

    ####################################################################################################################
    # # Example: Only dope the first and last layer (both will have the same doping percentage but different ordering)
    #
    # # Create a graphene sheet
    # graphene = GrapheneSheet(bond_distance=1.42, sheet_size=sheet_size)
    #
    # # Stack the graphene sheet
    # stacked_graphene = graphene.stack(interlayer_spacing=3.35, number_of_layers=5)
    #
    # # Add individual nitrogen doping only to the first and last layer
    # start_time = time.time()  # Time the nitrogen doping process
    # stacked_graphene.add_nitrogen_doping_to_layer(layer_index=0, total_percentage=15)
    # stacked_graphene.add_nitrogen_doping_to_layer(layer_index=4, total_percentage=15)
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
    # write_xyz(stacked_graphene.graph, "ABA_stacking.xyz")

    ####################################################################################################################
    # Example of creating a CNT
    cnt = CNT(bond_length=1.42, tube_length=10.0, tube_size=8, conformation="zigzag")
    # cnt.add_nitrogen_doping(total_percentage=10)
    cnt.plot_structure(with_labels=True)

    # Save the CNT structure to a file
    write_xyz(cnt.graph, "CNT_structure.xyz")


if __name__ == "__main__":
    main()
