import random
from dataclasses import dataclass, field
from math import cos, pi, sin
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial import KDTree

from conan.playground.graph_utils import (
    NitrogenSpecies,
    NitrogenSpeciesProperties,
    Position,
    get_neighbors_via_edges,
    minimum_image_distance,
    plot_graphene,
    print_warning,
    write_xyz,
)


@dataclass
class CycleData:
    cycles: Dict[NitrogenSpecies, List[List[int]]] = field(default_factory=dict)

    def add_cycle(self, species: NitrogenSpecies, cycle: List[int]):
        if species not in self.cycles:
            self.cycles[species] = []
        self.cycles[species].append(cycle)

    def _order_single_cycle(
        self, graph: nx.Graph, cycle: List[int], species: NitrogenSpecies, start_node: Optional[int] = None
    ) -> List[int]:
        if start_node is None:
            # If no start node is provided, find a suitable starting node based on the nitrogen species
            start_node = self._find_start_node(graph, cycle, species)

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
            neighbors = [node for node in graph.neighbors(current_node) if node in cycle and node not in visited]

            # If there are unvisited neighbors, move to the next neighbor; otherwise, break the loop
            if neighbors:
                current_node = neighbors[0]
            else:
                break
        return ordered_cycle

    def detect_and_register_cycle(
        self, graph, species: NitrogenSpecies, neighbors: List[int], start_node: Optional[int] = None
    ):
        cycle = self._find_min_cycle_including_neighbors(graph, neighbors)
        ordered_cycle = self._order_single_cycle(graph, cycle, species, start_node)
        self.add_cycle(species, ordered_cycle)
        return ordered_cycle

    @staticmethod
    def _find_min_cycle_including_neighbors(graph: nx.Graph, neighbors: List[int]) -> List[int]:
        """
        Find the shortest cycle in the graph that includes all the given neighbors.

        This method uses an iterative approach to expand the subgraph starting from the given neighbors. In each
        iteration, it expands the subgraph by adding edges of the current nodes until a cycle containing all neighbors
        is found.
        The cycle detection is done using the `cycle_basis` method, which is efficient for small subgraphs that are
        incrementally expanded.

        Parameters
        ----------
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
    def _find_start_node(graph: nx.Graph, cycle: List[int], species: NitrogenSpecies) -> int:
        """
        Find a suitable starting node for a given cycle based on the nitrogen species. The starting node is used to
        ensure a consistent iteration order through the cycle, matching the bond lengths and angles correctly.

        Parameters
        ----------
        graph : nx.Graph
            The graph containing the cycle.
        cycle : List[int]
            A list containing the atom IDs forming the cycle.
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
            for node in cycle:
                # Skip the node if it is already a nitrogen atom
                if graph.nodes[node]["element"] == "N":
                    continue
                # Get the neighbors of the current node
                neighbors = get_neighbors_via_edges(graph, node)
                # Check if none of the neighbors of the node are nitrogen atoms, provided the neighbor is within the
                # cycle
                if all(graph.nodes[neighbor]["element"] != "N" for neighbor in neighbors if neighbor in cycle):
                    # If the current node meets all conditions, set it as the start node
                    start_node = node
                    break
            # Raise an error if no suitable start node is found
        if start_node is None:
            raise ValueError(f"No suitable starting node found in the cycle {cycle}.")
        return start_node


@dataclass  # ToDo: Comment
class DopingStructuralComponents:
    atoms: List[int]
    neighbors: List[int]


class Graphene:
    def __init__(self, bond_distance: float, sheet_size: Tuple[float, float]):
        """
        Initialize the GrapheneGraph with given bond distance and sheet size.

        Parameters
        ----------
        bond_distance : float
            The bond distance between carbon atoms in the graphene sheet.
        sheet_size : Tuple[float, float]
            The size of the graphene sheet in the x and y directions.
        """
        self.c_c_bond_distance = bond_distance
        """The bond distance between carbon atoms in the graphene sheet."""
        self.c_c_bond_angle = 120
        """The bond angle between carbon atoms in the graphene sheet."""
        self.sheet_size = sheet_size
        """The size of the graphene sheet in the x and y directions."""
        self.k_inner = 1000
        """The spring constant for bonds and angles within the doping structure."""
        self.k_outer = 0.1
        """The spring constant for bonds and angles outside the doping structure."""
        self.graph = nx.Graph()
        """The networkx graph representing the graphene sheet structure."""
        self._build_graphene_sheet()

        self.possible_carbon_atoms = [node for node, data in self.graph.nodes(data=True) if data["element"] == "C"]
        """A list of all possible carbon atoms in the graphene sheet for nitrogen doping."""
        # ToDo: Kann evtl. über Property automatisch geupdatet werden, sodass die atom ids nicht zusätzlich zum
        #  Entfernen aus dem Graphen auch noch aus der possible_carbon_atoms Liste entfernt werden müssen;
        #  Es könnte auch über die gefundenen cycles festgestellt werden, ob ein Atom als neue atom_id in Frage kommt
        #  Evtl. wird dann possible_carbon_atoms gar nicht mehr benötigt? Sondern alle im Graphen noch verfügbaren Atome
        #  sind potentielle Kandidaten für neue atom_ids (außer die, die schon in einem cycle sind)
        self.chosen_atoms = {species: [] for species in NitrogenSpecies}
        """A dictionary to keep track of chosen atoms for nitrogen doping for each species."""

        self.species_properties = self._initialize_species_properties()
        """A dictionary mapping each NitrogenSpecies to its corresponding NitrogenSpeciesProperties.
        This includes bond lengths and angles characteristic to each species."""

        self.cycle_data = CycleData()
        """A dataclass to store information about cycles (doping structures) in the graphene sheet."""

        # Initialize positions and KDTree for efficient neighbor search
        self._positions = np.array([self.graph.nodes[node]["position"] for node in self.graph.nodes])
        """The positions of atoms in the graphene sheet."""
        self.kdtree = KDTree(self._positions)  # ToDo: Solve problem with periodic boundary conditions
        """The KDTree data structure for efficient nearest neighbor search. A KDTree is particularly efficient for
        spatial queries, such as searching for neighbors within a certain Euclidean distance. Such queries are often
        computationally intensive when performed over a graph, especially when dealing with direct distance rather than
        path lengths in the graph."""

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
            Position(x_offset, y_offset),
            Position(x_offset + self.cc_x_distance, y_offset + self.cc_y_distance),
            Position(x_offset + self.cc_x_distance + self.c_c_bond_distance, y_offset + self.cc_y_distance),
            Position(x_offset + 2 * self.cc_x_distance + self.c_c_bond_distance, y_offset),
        ]

        # Add nodes with positions and element type (carbon)
        nodes = [(index + i, {"element": "C", "position": pos}) for i, pos in enumerate(unit_cell_positions)]
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

        # Initialize other species similarly
        species_properties = {
            NitrogenSpecies.PYRIDINIC_4: pyridinic_4_properties,
            NitrogenSpecies.PYRIDINIC_3: pyridinic_3_properties,
            NitrogenSpecies.PYRIDINIC_2: pyridinic_2_properties,
            NitrogenSpecies.PYRIDINIC_1: pyridinic_1_properties,
        }
        return species_properties

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
        # Validate specific percentages and calculate the remaining percentage
        if percentages:
            if total_percentage is None:
                total_percentage = sum(percentages.values())  # Set total to sum of specific percentages if not provided
            else:
                specific_total_percentage = sum(percentages.values())  # Sum of provided specific percentages
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
            percentages = {}  # Initialize an empty dictionary if no specific percentages are provided

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

        # Dictionary to keep track of actually added nitrogen atoms
        added_nitrogen_counts = {species: 0 for species in NitrogenSpecies}

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
                self._insert_doping_structures(num_nitrogen_atoms, species)
                added_nitrogen_counts[species] += len(self.chosen_atoms[species])

        # Calculate the actual percentages of added nitrogen species
        total_atoms = self.graph.number_of_nodes()
        actual_percentages = {
            species.value: round((count / total_atoms) * 100, 2) if total_atoms > 0 else 0
            for species, count in added_nitrogen_counts.items()
        }

        # Adjust the positions of atoms in all cycles to optimize the structure
        # if any(self.cycle_data.cycles.values()):
        #     self._adjust_atom_positions()

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

    def _insert_doping_structures(self, num_nitrogen: int, nitrogen_species: NitrogenSpecies) -> int:
        """
        Insert doping structures of a specific nitrogen species into the graphene sheet.

        Parameters
        ----------
        num_nitrogen : int
            The number of nitrogen atoms to add.
        nitrogen_species : NitrogenSpecies
            The type of nitrogen doping to add.

        Returns
        -------
         int
            The number of successfully added nitrogen atoms.

        Notes
        -----
        First, a carbon atom is randomly selected. Then, it is checked whether this atom position is suitable for
        building the doping structure around it (i.e., the new structure to be inserted should not overlap with any
        existing structure). If suitable, the doping structure is built by, for example, removing atoms, replacing
        other C atoms with N atoms, and possibly adding new bonds between atoms (in the case of Pyridinic_1). After
        the structure is inserted, all atoms of this structure are excluded from further doping positions.
        """

        # Shuffle the list of possible carbon atoms
        possible_carbon_atoms_shuffled = random.sample(self.possible_carbon_atoms, len(self.possible_carbon_atoms))

        while len(self.chosen_atoms[nitrogen_species]) < num_nitrogen and possible_carbon_atoms_shuffled:
            # Randomly select a carbon atom from the shuffled list without replacement and compute its neighbors
            atom_id = possible_carbon_atoms_shuffled.pop(0)

            # Check if the selected atom is a possible carbon atom
            if atom_id not in self.possible_carbon_atoms:
                continue  # ToDo: Super blöde Lösung, dass man hier das nochmal überprüfen muss; evtl. möglich, dass das
                #  possible_carbon_atoms_shuffled geschickter mit self.possible_carbon_atoms funktioniert?

            neighbors = get_neighbors_via_edges(self.graph, atom_id)

            # Check if the selected atom is a valid doping position
            is_valid, doping_structural_components = self._valid_doping_position(nitrogen_species, atom_id, neighbors)
            if not is_valid:
                continue

            # Atom is valid, proceed with nitrogen doping
            if nitrogen_species == NitrogenSpecies.GRAPHITIC:
                self._handle_graphitic_doping(doping_structural_components, nitrogen_species)
            else:
                self._handle_pyridinic_doping(doping_structural_components, nitrogen_species)

        # Warn if not all requested nitrogen atoms could be placed
        if len(self.chosen_atoms[nitrogen_species]) < num_nitrogen:
            warning_message = (
                f"\nWarning: Only {len(self.chosen_atoms[nitrogen_species])} nitrogen atoms of species "
                f"{nitrogen_species.value} could be placed due to proximity constraints."
            )
            print_warning(warning_message)

    def _handle_graphitic_doping(
        self, doping_structural_components: DopingStructuralComponents, nitrogen_species: NitrogenSpecies
    ):
        atom_id = doping_structural_components.atoms[0]
        neighbors = doping_structural_components.neighbors

        # Add the selected atom to the list of chosen atoms
        self.chosen_atoms[nitrogen_species].append(atom_id)
        # Update the selected atom's element to nitrogen and set its nitrogen species
        self.graph.nodes[atom_id]["element"] = "N"
        self.graph.nodes[atom_id]["nitrogen_species"] = NitrogenSpecies.GRAPHITIC

        # Remove the selected atom and its neighbors from the list of potential carbon atoms
        self.possible_carbon_atoms.remove(atom_id)
        for neighbor in neighbors:
            if neighbor in self.possible_carbon_atoms:
                self.possible_carbon_atoms.remove(neighbor)

    def _handle_pyridinic_doping(
        self, doping_structural_components: DopingStructuralComponents, nitrogen_species: NitrogenSpecies
    ):

        # Remove the selected atom(s) from the graph
        for atom in doping_structural_components.atoms:
            self.graph.remove_node(atom)
            self.possible_carbon_atoms.remove(atom)

        start_node = self._handle_species_specific_logic(nitrogen_species, doping_structural_components.neighbors)
        nodes_to_exclude = self.cycle_data.detect_and_register_cycle(
            self.graph, nitrogen_species, doping_structural_components.neighbors, start_node
        )

        for node in nodes_to_exclude:
            if node in self.possible_carbon_atoms:
                self.possible_carbon_atoms.remove(node)
        self._add_edge_if_needed(nitrogen_species, doping_structural_components.neighbors)

    def _handle_species_specific_logic(self, nitrogen_species: NitrogenSpecies, neighbors: List[int]) -> Optional[int]:
        """
        Handle species-specific logic for adding nitrogen atoms.

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
        start_node = None

        if nitrogen_species == NitrogenSpecies.PYRIDINIC_1:
            # Replace 1 carbon atom to form pyridinic nitrogen structure
            selected_neighbor = random.choice(neighbors)
            self.graph.nodes[selected_neighbor]["element"] = "N"
            self.graph.nodes[selected_neighbor]["nitrogen_species"] = nitrogen_species
            # Add the selected atom to the list of chosen atoms
            self.chosen_atoms[nitrogen_species].append(selected_neighbor)

            # # Remove the selected neighbor from the list of neighbors
            # neighbors.remove(selected_neighbor)

            # Identify the start node for this cycle as the selected neighbor
            start_node = selected_neighbor

        elif nitrogen_species == NitrogenSpecies.PYRIDINIC_2:
            # Replace 2 carbon atoms to form pyridinic nitrogen structure
            selected_neighbors = random.sample(neighbors, 2)
            for neighbor in selected_neighbors:
                self.graph.nodes[neighbor]["element"] = "N"
                self.graph.nodes[neighbor]["nitrogen_species"] = nitrogen_species
                # Add the neighbor to the list of chosen atoms
                self.chosen_atoms[nitrogen_species].append(neighbor)

            # Identify the start node for this cycle using set difference
            remaining_neighbor = (set(neighbors) - set(selected_neighbors)).pop()
            start_node = remaining_neighbor

        elif nitrogen_species == NitrogenSpecies.PYRIDINIC_3 or nitrogen_species == NitrogenSpecies.PYRIDINIC_4:
            # Replace 3 resp. 4 carbon atoms to form pyridinic nitrogen structure
            for neighbor in neighbors:
                self.graph.nodes[neighbor]["element"] = "N"
                self.graph.nodes[neighbor]["nitrogen_species"] = nitrogen_species
                # Add the neighbor to the list of chosen atoms
                self.chosen_atoms[nitrogen_species].append(neighbor)

        # elif nitrogen_species == NitrogenSpecies.PYRIDINIC_4:
        #
        #     # Replace 4 carbon atoms to form pyridinic nitrogen structure
        #     for neighbor in neighbors:
        #         self.graph.nodes[neighbor]["element"] = "N"
        #         self.graph.nodes[neighbor]["nitrogen_species"] = nitrogen_species
        #         # Add the neighbor to the list of chosen atoms
        #         chosen_atoms.append(neighbor)

        return start_node

    def _add_edge_if_needed(self, nitrogen_species: NitrogenSpecies, neighbors: List[int]):
        """
        Add an edge between neighbors if the nitrogen species is PYRIDINIC_1.

        Parameters
        ----------
        nitrogen_species : NitrogenSpecies
            The type of nitrogen doping.
        neighbors : List[int]
            List of neighbor atom IDs.
        """
        if nitrogen_species == NitrogenSpecies.PYRIDINIC_1:
            # Calculate the bond length between neighbors[0] and neighbors[1]
            pos1 = self.graph.nodes[neighbors[0]]["position"]
            pos2 = self.graph.nodes[neighbors[1]]["position"]
            box_size = (
                self.actual_sheet_width + self.c_c_bond_distance,
                self.actual_sheet_height + self.cc_y_distance,
            )
            bond_length, _ = minimum_image_distance(pos1, pos2, box_size)

            # Insert a new binding between the `neighbors_of_neighbor` with the calculated bond length
            self.graph.add_edge(neighbors[0], neighbors[1], bond_length=bond_length)

    def _adjust_atom_positions(self):
        """
        Adjust the positions of atoms in the graphene sheet to optimize the structure including doping.

        Notes
        -----
        This method adjusts the positions of atoms in a graphene sheet to optimize the structure based on the doping
        configuration. It uses a combination of bond and angle energies to minimize the total energy of the system.
        """
        # ToDo: Refactoring is urgently needed here. Totally stupid solution with Dict and then separate lists.
        all_cycles = []
        species_for_cycles = []

        for species, cycle_list in self.cycle_data.cycles.items():
            for cycle in cycle_list:
                all_cycles.append(cycle)
                species_for_cycles.append(species)

        if not all_cycles:
            return

        # Initial positions (use existing positions if available)
        positions = {node: self.graph.nodes[node]["position"] for node in self.graph.nodes}

        # Flatten initial positions for optimization
        x0 = np.array([coord for node in self.graph.nodes for coord in positions[node]])

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

            # Iterate over each cycle
            for idx, ordered_cycle in enumerate(all_cycles):
                # Get species properties for the current cycle
                species = species_for_cycles[idx]
                properties = self.species_properties[species]
                target_bond_lengths = properties.target_bond_lengths

                # Create a subgraph for the current cycle
                subgraph = self.graph.subgraph(ordered_cycle).copy()

                # Calculate bond energy for edges within the cycle
                cycle_length = len(ordered_cycle)
                for i in range(cycle_length):
                    node_i = ordered_cycle[i]
                    node_j = ordered_cycle[(i + 1) % cycle_length]  # Ensure the last node connects to the first node
                    xi, yi = (
                        x[2 * list(self.graph.nodes).index(node_i)],
                        x[2 * list(self.graph.nodes).index(node_i) + 1],
                    )
                    xj, yj = (
                        x[2 * list(self.graph.nodes).index(node_j)],
                        x[2 * list(self.graph.nodes).index(node_j) + 1],
                    )
                    pos_i = Position(xi, yi)
                    pos_j = Position(xj, yj)

                    # Calculate the current bond length and target bond length
                    current_length, _ = minimum_image_distance(pos_i, pos_j, box_size)
                    target_length = target_bond_lengths[ordered_cycle.index(node_i)]
                    energy += 0.5 * self.k_inner * ((current_length - target_length) ** 2)

                    # Update bond length in the graph during optimization
                    self.graph.edges[node_i, node_j]["bond_length"] = current_length

                    # Add edge to cycle_edges set
                    cycle_edges.add((min(node_i, node_j), max(node_i, node_j)))

                if species == NitrogenSpecies.PYRIDINIC_1:
                    # ToDo: See if you can optimize this so that you don't have to repeat a lot of logic from above and
                    #  maybe you can also iterate directly over the subgraph.edges in the for loop above and thus save
                    #  redundant code
                    for i, j in subgraph.edges():
                        if (min(i, j), max(i, j)) not in cycle_edges:
                            xi, yi = (
                                x[2 * list(self.graph.nodes).index(i)],
                                x[2 * list(self.graph.nodes).index(i) + 1],
                            )
                            xj, yj = (
                                x[2 * list(self.graph.nodes).index(j)],
                                x[2 * list(self.graph.nodes).index(j) + 1],
                            )
                            pos_i = Position(xi, yi)
                            pos_j = Position(xj, yj)

                            current_length, _ = minimum_image_distance(pos_i, pos_j, box_size)
                            target_length = target_bond_lengths[-1]  # Last bond length for Pyridinic_1
                            energy += 0.5 * self.k_inner * ((current_length - target_length) ** 2)

                            # Update bond length in the graph during optimization
                            self.graph.edges[i, j]["bond_length"] = current_length

                            # Add edge to cycle_edges set
                            cycle_edges.add((min(i, j), max(i, j)))

            # Calculate bond energy for edges outside the cycles
            for i, j, data in self.graph.edges(data=True):
                if (min(i, j), max(i, j)) not in cycle_edges:
                    xi, yi = x[2 * list(self.graph.nodes).index(i)], x[2 * list(self.graph.nodes).index(i) + 1]
                    xj, yj = x[2 * list(self.graph.nodes).index(j)], x[2 * list(self.graph.nodes).index(j) + 1]
                    pos_i = Position(xi, yi)
                    pos_j = Position(xj, yj)

                    # Calculate the current bond length and set default target length
                    current_length, _ = minimum_image_distance(pos_i, pos_j, box_size)
                    target_length = 1.42
                    energy += 0.5 * self.k_outer * ((current_length - target_length) ** 2)

                    # Update bond length in the graph during optimization
                    self.graph.edges[i, j]["bond_length"] = current_length

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

            # Initialize a set to track angles within cycles
            counted_angles = set()

            # Iterate over each cycle
            for idx, ordered_cycle in enumerate(all_cycles):
                # Get species properties for the current cycle
                species = species_for_cycles[idx]
                properties = self.species_properties[species]
                target_angles = properties.target_angles

                for (i, j, k), angle in zip(zip(ordered_cycle, ordered_cycle[1:], ordered_cycle[2:]), target_angles):
                    xi, yi = x[2 * list(self.graph.nodes).index(i)], x[2 * list(self.graph.nodes).index(i) + 1]
                    xj, yj = x[2 * list(self.graph.nodes).index(j)], x[2 * list(self.graph.nodes).index(j) + 1]
                    xk, yk = x[2 * list(self.graph.nodes).index(k)], x[2 * list(self.graph.nodes).index(k) + 1]

                    pos_i = Position(xi, yi)
                    pos_j = Position(xj, yj)
                    pos_k = Position(xk, yk)

                    _, v1 = minimum_image_distance(pos_i, pos_j, box_size)
                    _, v2 = minimum_image_distance(pos_k, pos_j, box_size)

                    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
                    energy += 0.5 * self.k_inner * ((theta - np.radians(angle)) ** 2)

                    # Add angles to counted_angles to avoid double-counting
                    counted_angles.add((i, j, k))
                    counted_angles.add((k, j, i))

            # # Calculate angle energy for angles outside the cycles
            # for node in self.graph.nodes:
            #     neighbors = list(self.graph.neighbors(node))
            #     if len(neighbors) < 2:
            #         continue
            #     for i in range(len(neighbors)):
            #         for j in range(i + 1, len(neighbors)):
            #             ni = neighbors[i]
            #             nj = neighbors[j]
            #
            #             # Skip angles that have already been counted
            #             if (ni, node, nj) in counted_angles or (nj, node, ni) in counted_angles:
            #                 continue
            #
            #             x_node, y_node = (
            #                 x[2 * list(self.graph.nodes).index(node)],
            #                 x[2 * list(self.graph.nodes).index(node) + 1],
            #             )
            #             x_i, y_i = (x[2 * list(self.graph.nodes).index(ni)],
            #                         x[2 * list(self.graph.nodes).index(ni) + 1])
            #             x_j, y_j = (x[2 * list(self.graph.nodes).index(nj)],
            #                         x[2 * list(self.graph.nodes).index(nj) + 1])
            #             pos_node = Position(x_node, y_node)
            #             pos_i = Position(x_i, y_i)
            #             pos_j = Position(x_j, y_j)
            #             _, v1 = minimum_image_distance(pos_i, pos_node, box_size)
            #             _, v2 = minimum_image_distance(pos_j, pos_node, box_size)
            #             cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            #             theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
            #             energy += 0.5 * self.k_outer * ((theta - np.radians(self.c_c_bond_angle)) ** 2)

            return energy

        def total_energy(x):
            """
            Calculate the total energy (bond energy + angle energy).

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

        # Optimize positions to minimize energy
        result = minimize(total_energy, x0, method="L-BFGS-B")

        # Show the number of iterations and the final energy
        print(f"Number of iterations: {result.nit}\nFinal energy: {result.fun}")

        # Reshape the 1D array result to 2D coordinates and update positions in the graph
        optimized_positions = result.x.reshape(-1, 2)

        # Update positions in the original graph based on the optimized positions
        for idx, node in enumerate(self.graph.nodes):
            optimized_position = optimized_positions[idx]
            adjusted_position = Position(x=optimized_position[0], y=optimized_position[1])
            self.graph.nodes[node]["position"] = adjusted_position

    # ToDo: evtl. Methode etwas umbenennen, da sie ja auch die DopingStructuralComponents setzt
    def _valid_doping_position(
        self, nitrogen_species: NitrogenSpecies, atom_id: int, neighbors: List[int]
    ) -> Tuple[bool, Optional[DopingStructuralComponents]]:
        """
        Determine if a given position is valid for nitrogen doping based on the nitrogen species and atom position.

        This method checks if there is enough space around the randomly selected atom_id to add a doping structure
        of the type nitrogen_species without overlapping with existing doping structures. It ensures that the new
        structure does not interfere with the cycle of any existing structures.

        Parameters
        ----------
        nitrogen_species : NitrogenSpecies
            The type of nitrogen doping to validate.
        atom_id : int
            The ID of the atom to check for doping suitability.

        Returns
        -------
        Tuple[bool, Optional[DopingStructuralComponents]]
            A tuple containing a boolean indicating if the position is valid and a DopingStructuralComponents instance
            with atoms and neighbors.
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

        # Check the proximity constraints based on the nitrogen species
        if nitrogen_species == NitrogenSpecies.GRAPHITIC:
            # Retrieve elements and nitrogen species of neighbors
            neighbor_elements = [
                (self.graph.nodes[neighbor]["element"], self.graph.nodes[neighbor].get("nitrogen_species"))
                for neighbor in neighbors
            ]
            # Ensure all neighbors are not nitrogen atoms
            if all(elem != "N" for elem, _ in neighbor_elements):
                return True, DopingStructuralComponents(atoms=[atom_id], neighbors=neighbors)
            return False, None

        elif nitrogen_species in {
            NitrogenSpecies.PYRIDINIC_1,
            NitrogenSpecies.PYRIDINIC_2,
            NitrogenSpecies.PYRIDINIC_3,
        }:
            # Get neighbors up to depth 2 for the selected atom
            neighbors_len_2 = get_neighbors_via_edges(self.graph, atom_id, depth=2, inclusive=True)
            # Ensure all neighbors are possible atoms for doping
            # return all_neighbors_possible_carbon_atoms(neighbors_len_2), None
            if all_neighbors_possible_carbon_atoms(neighbors_len_2):
                return True, DopingStructuralComponents(atoms=[atom_id], neighbors=neighbors_len_2)
            return False, None

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
                # Ensure all neighbors (from both atoms) are possible atoms for doping
                if all_neighbors_possible_carbon_atoms(list(set(neighbors_len_2_atom + neighbors_len_2_neighbor))):
                    # Valid neighbor found
                    selected_neighbor = temp_neighbor

            # # Get neighbors up to depth 2 for the selected atom and a neighboring atom (if provided)
            # neighbors_len_2_atom = get_neighbors_via_edges(self.graph, atom_id, depth=2, inclusive=True)
            # neighbors_len_2_neighbor = (
            #     get_neighbors_via_edges(self.graph, neighbor_id, depth=2, inclusive=True) if neighbor_id else []
            # )
            # # Ensure all neighbors (from both atoms) are possible atoms for doping
            # return all_neighbors_possible_carbon_atoms(neighbors_len_2_atom + neighbors_len_2_neighbor)

            if selected_neighbor is None:
                return False, None

            combined_neighbors = list(set(neighbors + get_neighbors_via_edges(self.graph, selected_neighbor)))
            combined_neighbors = [n for n in combined_neighbors if n not in {atom_id, selected_neighbor}]
            # ToDo: Lösung gefällt mir nicht. Evtl. ist es doch besser bei "get_neighbors_via_edges" über eine Flag auch
            #  zu ermöglichen, dass der start node selbst auch mit in die Liste aufgenommen wird, wenn inclusive=True
            #  ist
            return True, DopingStructuralComponents(atoms=[atom_id, selected_neighbor], neighbors=combined_neighbors)

        # Return False if none of the conditions are met which should not happen
        return False, None


def main():
    # Set seed for reproducibility
    # random.seed(42)
    # random.seed(3)
    random.seed(0)

    graphene = Graphene(bond_distance=1.42, sheet_size=(20, 20))

    # write_xyz(graphene.graph, 'graphene.xyz')
    # graphene.plot_graphene(with_labels=True)

    # Find direct neighbors of a node (depth=1)
    direct_neighbors = get_neighbors_via_edges(graphene.graph, atom_id=0, depth=1)
    print(f"Direct neighbors of C_0: {direct_neighbors}")

    # Find neighbors of a node at an exact depth (depth=2)
    depth_neighbors = get_neighbors_via_edges(graphene.graph, atom_id=0, depth=2)
    print(f"Neighbors of C_0 at depth 2: {depth_neighbors}")

    # Find neighbors of a node up to a certain depth (inclusive=True)
    inclusive_neighbors = get_neighbors_via_edges(graphene.graph, atom_id=0, depth=2, inclusive=True)
    print(f"Neighbors of C_0 up to depth 2 (inclusive): {inclusive_neighbors}")

    # graphene.add_nitrogen_doping_old(10, NitrogenSpecies.GRAPHITIC)
    # plot_graphene(graphene.graph, with_labels=True, visualize_periodic_bonds=False)

    # graphene.add_nitrogen_doping(percentages={NitrogenSpecies.PYRIDINIC_2: 2})
    # plot_graphene(graphene.graph, with_labels=True, visualize_periodic_bonds=False)

    # graphene.add_nitrogen_doping(percentages={NitrogenSpecies.PYRIDINIC_3: 2})
    # plot_graphene(graphene.graph, with_labels=True, visualize_periodic_bonds=False)

    # graphene.add_nitrogen_doping(
    #     percentages={NitrogenSpecies.PYRIDINIC_2: 10, NitrogenSpecies.PYRIDINIC_3: 10, NitrogenSpecies.GRAPHITIC: 20}
    # )
    # plot_graphene(graphene.graph, with_labels=True, visualize_periodic_bonds=False)

    # graphene.add_nitrogen_doping(
    #     percentages={
    #         NitrogenSpecies.PYRIDINIC_2: 3,
    #         NitrogenSpecies.PYRIDINIC_3: 3,
    #         NitrogenSpecies.GRAPHITIC: 20,
    #         NitrogenSpecies.PYRIDINIC_4: 5,
    #         NitrogenSpecies.PYRIDINIC_1: 5,
    #     }
    # )
    # plot_graphene(graphene.graph, with_labels=True, visualize_periodic_bonds=False)

    graphene.add_nitrogen_doping(percentages={NitrogenSpecies.GRAPHITIC: 50, NitrogenSpecies.PYRIDINIC_4: 20})
    plot_graphene(graphene.graph, with_labels=True, visualize_periodic_bonds=False)

    # graphene.add_nitrogen_doping(percentages={NitrogenSpecies.PYRIDINIC_4: 30})
    # plot_graphene(graphene.graph, with_labels=True, visualize_periodic_bonds=False)

    # graphene.add_nitrogen_doping(percentages={NitrogenSpecies.PYRIDINIC_1: 1})
    # plot_graphene(graphene.graph, with_labels=True, visualize_periodic_bonds=False)

    # graphene.add_nitrogen_doping(total_percentage=20, percentages={NitrogenSpecies.GRAPHITIC: 10})
    # plot_graphene(graphene.graph, with_labels=True, visualize_periodic_bonds=False)

    # graphene.add_nitrogen_doping(percentages={NitrogenSpecies.GRAPHITIC: 10, NitrogenSpecies.PYRIDINIC_3: 5})
    # plot_graphene(graphene.graph, with_labels=True, visualize_periodic_bonds=False)

    # graphene.add_nitrogen_doping(total_percentage=15)
    # plot_graphene(graphene.graph, with_labels=True, visualize_periodic_bonds=False)

    # write_xyz(
    #     graphene.graph,
    #     f"graphene_doping_k_inner_{graphene.k_inner}_k_outer_{graphene.k_outer}_including_angles_outside_cycle.xyz",
    # )

    # write_xyz(graphene.graph, f"graphene_doping_k_inner_{graphene.k_inner}_k_outer_{graphene.k_outer}.xyz")

    write_xyz(
        graphene.graph,
        f"pyridinic_4_doping_k_inner_{graphene.k_inner}_k_outer_{graphene.k_outer}_refactored.xyz",
    )

    # write_xyz(graphene.graph, f"pyridinic_4_doping_k_inner_{graphene.k_inner}_k_outer_{graphene.k_outer}.xyz")

    # source = 0
    # target = 10
    # path = get_shortest_path(graphene.graph, source, target)
    # print(f"Shortest path from C_{source} to C_{target}: {path}")
    # plot_graphene_with_path(graphene.graph, path)
    #
    # plot_graphene_with_depth_neighbors_based_on_bond_length(graphene.graph, 0, 4)
    #
    # # Find nodes within a certain distance from a source node
    # atom_id = 5
    # max_distance = 5
    # nodes_within_distance = get_neighbors_within_distance(graphene.graph, graphene.kdtree, atom_id, max_distance)
    # print(f"Nodes within {max_distance} distance from node {atom_id}: {nodes_within_distance}")
    #
    # # Plot the nodes within the specified distance
    # plot_nodes_within_distance(graphene.graph, nodes_within_distance)


if __name__ == "__main__":
    main()
