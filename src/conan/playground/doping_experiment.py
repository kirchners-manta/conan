import random
from enum import Enum
from math import cos, pi, sin
from typing import List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.spatial import KDTree


class NitrogenSpecies(Enum):
    GRAPHITIC = "graphitic"
    PYRIDINIC = "pyridinic"
    PYRIDINIC_1 = "pyridinic_1"
    PYRIDINIC_2 = "pyridinic_2"
    PYRIDINIC_3 = "pyridinic_3"
    PYRIDINIC_4 = "pyridinic_4"
    PYRROLIC = "pyrrolic"
    PYRAZOLE = "pyrazole"


class GrapheneGraph:
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
        self.bond_distance = bond_distance
        """The bond distance between carbon atoms in the graphene sheet."""
        self.sheet_size = sheet_size
        """The size of the graphene sheet in the x and y directions."""
        self.graph = nx.Graph()
        """The networkx graph representing the graphene sheet structure."""
        self._build_graphene_sheet()

        # Initialize positions and KDTree for efficient neighbor search
        self._positions = np.array([self.graph.nodes[node]["position"] for node in self.graph.nodes])
        """The positions of atoms in the graphene sheet."""
        self._kdtree = KDTree(self._positions)  # ToDo: Solve problem with periodic boundary conditions
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
        self._kdtree = KDTree(new_positions)

    @property
    def cc_x_distance(self):
        """Calculate the distance between atoms in the x direction."""
        return self.bond_distance * sin(pi / 6)

    @property
    def cc_y_distance(self):
        """Calculate the distance between atoms in the y direction."""
        return self.bond_distance * cos(pi / 6)

    @property
    def num_cells_x(self):
        """Calculate the number of unit cells in the x direction based on sheet size and bond distance."""
        return int(self.sheet_size[0] // (2 * self.bond_distance + 2 * self.cc_x_distance))

    @property
    def num_cells_y(self):
        """Calculate the number of unit cells in the y direction based on sheet size and bond distance."""
        return int(self.sheet_size[1] // (2 * self.cc_y_distance))

    def _build_graphene_sheet(self):
        """
        Build the graphene sheet structure by creating nodes and edges (using graph theory via networkx).

        This method iterates over the entire sheet, adding nodes and edges for each unit cell.
        It also connects adjacent unit cells and adds periodic boundary conditions.
        """
        index = 0
        for y in range(self.num_cells_y):
            for x in range(self.num_cells_x):
                x_offset = x * (2 * self.bond_distance + 2 * self.cc_x_distance)
                y_offset = y * (2 * self.cc_y_distance)

                # Add nodes and edges for the unit cell
                self._add_unit_cell(index, x_offset, y_offset)

                # Add horizontal bonds between adjacent unit cells
                if x > 0:
                    self.graph.add_edge(index - 1, index, bond_length=self.bond_distance)

                # Add vertical bonds between unit cells in adjacent rows
                if y > 0:
                    self.graph.add_edge(index - 4 * self.num_cells_x + 1, index, bond_length=self.bond_distance)
                    self.graph.add_edge(index - 4 * self.num_cells_x + 2, index + 3, bond_length=self.bond_distance)

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
            (x_offset, y_offset),
            (x_offset + self.cc_x_distance, y_offset + self.cc_y_distance),
            (x_offset + self.cc_x_distance + self.bond_distance, y_offset + self.cc_y_distance),
            (x_offset + 2 * self.cc_x_distance + self.bond_distance, y_offset),
        ]

        # Add nodes with positions and element type (carbon)
        nodes = [(index + i, {"element": "C", "position": pos}) for i, pos in enumerate(unit_cell_positions)]
        self.graph.add_nodes_from(nodes)

        # Add internal bonds within the unit cell
        edges = [
            (index + i, index + i + 1, {"bond_length": self.bond_distance}) for i in range(len(unit_cell_positions) - 1)
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
            zip(right_edge_indices, left_edge_indices), bond_length=self.bond_distance, periodic=True
        )

        # Generate base indices for vertical boundaries
        top_left_indices = np.arange(self.num_cells_x) * 4
        bottom_left_indices = top_left_indices + (self.num_cells_y - 1) * num_nodes_x + 1
        bottom_right_indices = top_left_indices + (self.num_cells_y - 1) * num_nodes_x + 2

        # Add vertical periodic boundary conditions
        self.graph.add_edges_from(
            zip(bottom_left_indices, top_left_indices), bond_length=self.bond_distance, periodic=True
        )
        self.graph.add_edges_from(
            zip(bottom_right_indices, top_left_indices + 3), bond_length=self.bond_distance, periodic=True
        )

    def add_nitrogen_doping_old(self, percentage: float, nitrogen_species: NitrogenSpecies = NitrogenSpecies.GRAPHITIC):
        """
        Add nitrogen doping to the graphene sheet.

        This method randomly replaces a specified percentage of carbon atoms with nitrogen atoms of a given species.
        It ensures that there is always at least one carbon atom between two graphitic nitrogen atoms.

        Parameters
        ----------
        percentage : float
            The percentage of carbon atoms to replace with nitrogen atoms.
        nitrogen_species : NitrogenSpecies, optional
            The type of nitrogen doping to add. Default is NitrogenSpecies.GRAPHITIC.

        Raises
        ------
        ValueError
            If the nitrogen_species is not a valid NitrogenSpecies enum value.

        Notes
        -----
        If the specified percentage of nitrogen atoms cannot be placed due to proximity constraints,
        a warning will be printed.
        """
        # Check if the provided nitrogen_species is valid
        if not isinstance(nitrogen_species, NitrogenSpecies):
            raise ValueError(
                f"Invalid nitrogen type: {nitrogen_species}. Valid types are: "
                f"{', '.join([e.value for e in NitrogenSpecies])}"
            )

        # Calculate the number of nitrogen atoms to add based on the given percentage
        num_atoms = self.graph.number_of_nodes()
        num_nitrogen = int(num_atoms * percentage / 100)

        # Get a list of all carbon atoms in the graphene sheet
        carbon_atoms = [node for node, data in self.graph.nodes(data=True) if data["element"] == "C"]

        # Initialize an empty list to store the chosen atoms for nitrogen doping
        chosen_atoms = []

        # Randomly select carbon atoms to replace with nitrogen, ensuring proximity constraints
        while len(chosen_atoms) < num_nitrogen and carbon_atoms:
            # Randomly select a carbon atom from the list
            atom_id = random.choice(carbon_atoms)
            # Get the direct neighbors of the selected atom
            neighbors = self.get_neighbors_via_edges(atom_id)
            # Get the elements and nitrogen species of the neighbors
            neighbor_elements = [
                (self.graph.nodes[neighbor]["element"], self.graph.nodes[neighbor].get("nitrogen_species"))
                for neighbor in neighbors
            ]

            # Check if all neighbors are not graphitic nitrogen atoms
            if all(
                elem != "N" or (elem == "N" and n_type != NitrogenSpecies.GRAPHITIC)
                for elem, n_type in neighbor_elements
            ):
                # Add the selected atom to the list of chosen atoms
                chosen_atoms.append(atom_id)
                # Update the selected atom's element to nitrogen and set its nitrogen species
                self.graph.nodes[atom_id]["element"] = "N"
                self.graph.nodes[atom_id]["nitrogen_species"] = nitrogen_species
                # Remove the selected atom and its neighbors from the list of potential carbon atoms
                carbon_atoms.remove(atom_id)
                for neighbor in neighbors:
                    if neighbor in carbon_atoms:
                        carbon_atoms.remove(neighbor)

        # Warn if not all requested nitrogen atoms could be placed
        if len(chosen_atoms) < num_nitrogen:
            print(f"Warning: Only {len(chosen_atoms)} nitrogen atoms could be placed due to proximity constraints.")

        # Implement specific changes for other types of nitrogen doping if needed
        for atom_id in chosen_atoms:
            if nitrogen_species == NitrogenSpecies.GRAPHITIC:
                continue
            elif nitrogen_species == NitrogenSpecies.PYRIDINIC:
                # Implement pyridinic nitrogen replacement
                pass
            elif nitrogen_species == NitrogenSpecies.PYRROLIC:
                # Implement pyrrolic nitrogen replacement
                pass
            elif nitrogen_species == NitrogenSpecies.PYRAZOLE:
                # Implement pyrazole nitrogen replacement
                pass

    def add_nitrogen_doping(self, total_percentage: float = None, percentages: dict = None):
        """
        Add nitrogen doping to the graphene sheet.

        Parameters
        ----------
        total_percentage : float, optional
            The total percentage of carbon atoms to replace with nitrogen atoms.
        percentages : dict, optional
            A dictionary specifying the percentages for each nitrogen species. Keys should be NitrogenSpecies enum
            values and values should be the percentages for the corresponding species.

        Raises
        ------
        ValueError
            If the specific percentages are not valid.
        """
        if percentages:
            if total_percentage is None:
                total_percentage = sum(percentages.values())
            else:
                specific_total_percentage = sum(percentages.values())
                if specific_total_percentage != total_percentage:
                    raise ValueError(
                        f"The total specific percentages {specific_total_percentage}% do not match the total_percentage"
                        f" {total_percentage}%."
                    )
        else:
            if total_percentage is None:
                total_percentage = 10  # Default total percentage
            # Default distribution if no specific percentages are provided
            percentages = {
                NitrogenSpecies.GRAPHITIC: total_percentage * 0.5,
                NitrogenSpecies.PYRIDINIC_1: total_percentage * 0.075,
                NitrogenSpecies.PYRIDINIC_2: total_percentage * 0.075,
                NitrogenSpecies.PYRIDINIC_3: total_percentage * 0.075,
                NitrogenSpecies.PYRIDINIC_4: total_percentage * 0.075,
                NitrogenSpecies.PYRROLIC: total_percentage * 0.1,
                NitrogenSpecies.PYRAZOLE: total_percentage * 0.05,
            }

        # Calculate the number of nitrogen atoms to add based on the given percentage
        num_atoms = self.graph.number_of_nodes()
        specific_num_nitrogen = {species: int(num_atoms * pct / 100) for species, pct in percentages.items()}

        # Add nitrogen atoms for each specified nitrogen species
        for species, num_nitrogen_atoms in specific_num_nitrogen.items():
            self._add_nitrogen_atoms(num_nitrogen_atoms, species)

    def _add_nitrogen_atoms(self, num_nitrogen: int, nitrogen_species: NitrogenSpecies):
        """
        Add nitrogen atoms of a specific species to the graphene sheet.

        Parameters
        ----------
        num_nitrogen : int
            The number of nitrogen atoms to add.
        nitrogen_species : NitrogenSpecies
            The type of nitrogen doping to add.

        Notes
        -----
        This method randomly replaces carbon atoms with nitrogen atoms of the specified species.
        """
        # Get a list of all carbon atoms in the graphene sheet
        carbon_atoms = [node for node, data in self.graph.nodes(data=True) if data["element"] == "C"]

        # Initialize an empty list to store the chosen atoms for nitrogen doping
        chosen_atoms = []

        # Randomly select carbon atoms to replace with nitrogen, ensuring proximity constraints
        while len(chosen_atoms) < num_nitrogen and carbon_atoms:
            # Randomly select a carbon atom from the list
            atom_id = random.choice(carbon_atoms)
            # Get the direct neighbors of the selected atom
            neighbors = self.get_neighbors_via_edges(atom_id)
            # Get the elements and nitrogen species of the neighbors
            neighbor_elements = [
                (self.graph.nodes[neighbor]["element"], self.graph.nodes[neighbor].get("nitrogen_species"))
                for neighbor in neighbors
            ]

            # Implement species-specific changes for nitrogen doping
            if nitrogen_species == NitrogenSpecies.GRAPHITIC:
                # Check if all neighbors are not nitrogen atoms
                if all(elem != "N" for elem, _ in neighbor_elements):
                    # Add the selected atom to the list of chosen atoms
                    chosen_atoms.append(atom_id)
                    # Update the selected atom's element to nitrogen and set its nitrogen species
                    self.graph.nodes[atom_id]["element"] = "N"
                    self.graph.nodes[atom_id]["nitrogen_species"] = nitrogen_species
                    # Remove the selected atom and its neighbors from the list of potential carbon atoms
                    carbon_atoms.remove(atom_id)
                    for neighbor in neighbors:
                        if neighbor in carbon_atoms:
                            carbon_atoms.remove(neighbor)
            elif nitrogen_species in {
                NitrogenSpecies.PYRIDINIC_1,
                NitrogenSpecies.PYRIDINIC_2,
                NitrogenSpecies.PYRIDINIC_3,
                NitrogenSpecies.PYRIDINIC_4,
            }:
                # Get the neighbors until length two of the selected atom
                neighbors_len_2 = self.get_neighbors_via_edges(atom_id, depth=2, inclusive=True)
                # Check if all neighbors until length two are not nitrogen atoms
                if all(elem != "N" for elem in [self.graph.nodes[neighbor]["element"] for neighbor in neighbors_len_2]):
                    # Remove the selected atom and its neighbors of length two from the list of potential carbon atoms
                    carbon_atoms.remove(atom_id)
                    for neighbor in neighbors_len_2:
                        if neighbor in carbon_atoms:
                            carbon_atoms.remove(neighbor)

                    # Remove the selected atom from the graph
                    self.graph.remove_node(atom_id)

                    if nitrogen_species == NitrogenSpecies.PYRIDINIC_1:
                        # Replace 1 carbon atom to form pyridinic nitrogen structure
                        pass
                    elif nitrogen_species == NitrogenSpecies.PYRIDINIC_2:
                        # Replace 2 carbon atoms to form pyridinic nitrogen structure
                        selected_neighbors = random.sample(neighbors, 2)
                        for neighbor in selected_neighbors:
                            self.graph.nodes[neighbor]["element"] = "N"
                            self.graph.nodes[neighbor]["nitrogen_species"] = nitrogen_species
                            # Add the neighbor to the list of chosen atoms
                            chosen_atoms.append(neighbor)
                    elif nitrogen_species == NitrogenSpecies.PYRIDINIC_3:
                        # Replace 3 carbon atoms to form pyridinic nitrogen structure
                        for neighbor in neighbors:
                            self.graph.nodes[neighbor]["element"] = "N"
                            self.graph.nodes[neighbor]["nitrogen_species"] = nitrogen_species
                            # Add the neighbor to the list of chosen atoms
                            chosen_atoms.append(neighbor)
                    elif nitrogen_species == NitrogenSpecies.PYRIDINIC_4:
                        # Replace 4 carbon atoms to form pyridinic nitrogen structure
                        pass

        # Warn if not all requested nitrogen atoms could be placed
        if len(chosen_atoms) < num_nitrogen:
            print(f"Warning: Only {len(chosen_atoms)} nitrogen atoms could be placed due to proximity constraints.")

    def _implement_species_specific_changes(self, chosen_atoms, nitrogen_species):
        """
        Implement species-specific changes for nitrogen doping.

        Parameters
        ----------
        chosen_atoms : list
            List of atom IDs where nitrogen doping has been applied.
        nitrogen_species : NitrogenSpecies
            The type of nitrogen doping applied.

        Notes
        -----
        This method modifies the graphene structure based on the type of nitrogen doping.
        """
        if nitrogen_species == NitrogenSpecies.GRAPHITIC:
            return
        elif nitrogen_species in {
            NitrogenSpecies.PYRIDINIC_1,
            NitrogenSpecies.PYRIDINIC_2,
            NitrogenSpecies.PYRIDINIC_3,
            NitrogenSpecies.PYRIDINIC_4,
        }:
            for atom_id in chosen_atoms:
                # neighbors = self.get_neighbors_via_edges(atom_id)
                if nitrogen_species == NitrogenSpecies.PYRIDINIC_1:
                    # Replace 1 carbon atom to form pyridinic nitrogen structure
                    pass
                elif nitrogen_species == NitrogenSpecies.PYRIDINIC_2:
                    # Replace 2 carbon atoms to form pyridinic nitrogen structure
                    pass
                elif nitrogen_species == NitrogenSpecies.PYRIDINIC_3:
                    # Replace 3 carbon atoms to form pyridinic nitrogen structure
                    pass
                elif nitrogen_species == NitrogenSpecies.PYRIDINIC_4:
                    # Replace 4 carbon atoms to form pyridinic nitrogen structure
                    pass
        elif nitrogen_species == NitrogenSpecies.PYRROLIC:
            # Implement pyrrolic nitrogen replacement
            pass
        elif nitrogen_species == NitrogenSpecies.PYRAZOLE:
            # Implement pyrazole nitrogen replacement
            pass

    def get_neighbors_within_distance(self, atom_id: int, distance: float) -> List[int]:
        """
        Find all neighbors within a given distance from the specified atom.

        Parameters
        ----------
        atom_id : int
            The ID of the atom (node) from which distances are measured.
        distance : float
            The maximum distance to search for neighbors.

        Returns
        -------
        List[int]
            A list of IDs representing the neighbors within the given distance from the source node.
        """
        atom_position = self.graph.nodes[atom_id]["position"]
        indices = self._kdtree.query_ball_point(atom_position, distance)
        return [list(self.graph.nodes)[index] for index in indices]

    def get_neighbors_via_edges(self, atom_id: int, depth: int = 1, inclusive: bool = False) -> List[int]:
        """
        Get connected neighbors of a given atom up to a certain depth.

        Parameters
        ----------
        atom_id : int
            The ID of the atom (node) whose neighbors are to be found.
        depth : int, optional
            The depth up to which neighbors are to be found (default is 1).
        inclusive : bool, optional
            If True, return all neighbors up to the specified depth. If False, return only the neighbors
            at the exact specified depth (default is False).

        Returns
        -------
        List[int]
            A list of IDs representing the neighbors of the specified atom up to the given depth.

        Notes
        -----
        - If `depth` is 1, this method uses the `neighbors` function from networkx to find the immediate neighbors.
        - If `depth` is greater than 1:
          The function uses `nx.single_source_shortest_path_length(self.graph, atom_id, cutoff=depth)` from networkx.
          This function computes the shortest path lengths from the source node (atom_id) to all other nodes in the
          graph,up to the specified cutoff depth.

          - If `inclusive` is True, the function returns all neighbors up to the specified depth, meaning it includes
            neighbors at depth 1, 2, ..., up to the given depth.
          - If `inclusive` is False, the function returns only the neighbors at the exact specified depth.
        """

        if depth == 1:
            # Get immediate neighbors (directly connected nodes)
            return list(self.graph.neighbors(atom_id))
        else:
            # Get neighbors up to the specified depth using shortest path lengths
            paths = nx.single_source_shortest_path_length(self.graph, atom_id, cutoff=depth)
            if inclusive:
                # Include all neighbors up to the specified depth
                return [node for node in paths.keys() if node != atom_id]  # Exclude the atom itself (depth 0)
            else:
                # Include only neighbors at the exact specified depth
                return [node for node, length in paths.items() if length == depth]

    def get_neighbors_paths(self, atom_id: int, depth: int = 1) -> List[Tuple[int, int]]:
        """
        Get edges of paths to connected neighbors up to a certain depth.

        Parameters
        ----------
        atom_id : int
            The ID of the atom (node) whose neighbors' paths are to be found.
        depth : int, optional
            The depth up to which neighbors' paths are to be found (default is 1).

        Returns
        -------
        List[Tuple[int, int]]
            A list of tuples representing the edges of paths to neighbors up to the given depth.

        Notes
        -----
        This method uses the `single_source_shortest_path` function from networkx to find
        all paths up to the specified depth and then extracts the edges from these paths.
        """
        paths = nx.single_source_shortest_path(self.graph, atom_id, cutoff=depth)
        edges = []
        for path in paths.values():
            edges.extend([(path[i], path[i + 1]) for i in range(len(path) - 1)])
        return edges

    def get_shortest_path_length(self, source: int, target: int) -> float:
        """
        Get the shortest path length between two connected atoms based on bond lengths.

        Parameters
        ----------
        source : int
            The ID of the source atom (node).
        target : int
            The ID of the target atom (node).

        Returns
        -------
        float
            The shortest path length between the source and target atoms.

        Notes
        -----
        This method uses the Dijkstra algorithm implemented in networkx to find the shortest path length.
        """
        return nx.dijkstra_path_length(self.graph, source, target, weight="bond_length")

    def get_shortest_path(self, source: int, target: int) -> List[int]:
        """
        Get the shortest path between two connected atoms based on bond lengths.

        Parameters
        ----------
        source : int
            The ID of the source atom (node).
        target : int
            The ID of the target atom (node).

        Returns
        -------
        List[int]
            A list of IDs representing the nodes in the shortest path from the source to the target atom.

        Notes
        -----
        This method uses the Dijkstra algorithm implemented in networkx to find the shortest path.
        """
        return nx.dijkstra_path(self.graph, source, target, weight="bond_length")

    def get_color(self, element: str, nitrogen_species: NitrogenSpecies = None) -> str:
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
        colors = {"C": "black"}
        nitrogen_colors = {
            NitrogenSpecies.PYRIDINIC: "blue",
            NitrogenSpecies.GRAPHITIC: "red",
            NitrogenSpecies.PYRROLIC: "cyan",
            NitrogenSpecies.PYRAZOLE: "green",
        }
        if nitrogen_species in nitrogen_colors:
            return nitrogen_colors[nitrogen_species]
        return colors.get(element, "pink")

    def plot_graphene(self, with_labels: bool = False, visualize_periodic_bonds: bool = True):
        """
        Plot the graphene structure using networkx and matplotlib.

        Parameters
        ----------
        with_labels : bool, optional
            Whether to display labels on the nodes (default is False).
        visualize_periodic_bonds : bool, optional
            Whether to visualize periodic boundary condition edges (default is True).

        Notes
        -----
        This method visualizes the graphene structure, optionally with labels indicating the
        element type and node ID. Nodes are colored based on their element type and nitrogen species.
        Periodic boundary condition edges are shown with dashed lines if visualize_periodic_bonds is True.
        """
        # Get positions and elements of nodes
        pos = nx.get_node_attributes(self.graph, "position")
        elements = nx.get_node_attributes(self.graph, "element")

        # Determine colors for nodes, considering nitrogen species if present
        colors = [
            self.get_color(elements[node], self.graph.nodes[node].get("nitrogen_species"))
            for node in self.graph.nodes()
        ]

        # Separate periodic edges and regular edges
        regular_edges = [(u, v) for u, v, d in self.graph.edges(data=True) if not d.get("periodic")]
        periodic_edges = [(u, v) for u, v, d in self.graph.edges(data=True) if d.get("periodic")]

        # Initialize plot
        plt.figure(figsize=(12, 12))

        # Draw the regular edges
        nx.draw(self.graph, pos, edgelist=regular_edges, node_color=colors, node_size=200, with_labels=False)

        # Draw periodic edges with dashed lines if visualize_periodic_bonds is True
        if visualize_periodic_bonds:
            nx.draw_networkx_edges(self.graph, pos, edgelist=periodic_edges, style="dashed", edge_color="gray")

        # Add labels if specified
        if with_labels:
            labels = {node: f"{elements[node]}{node}" for node in self.graph.nodes()}
            nx.draw_networkx_labels(self.graph, pos, labels=labels, font_size=10, font_color="cyan", font_weight="bold")

        # Show plot
        plt.show()

    def plot_graphene_with_path(self, path: List[int], visualize_periodic_bonds: bool = True):
        """
        Plot the graphene structure with a highlighted path.

        This method plots the entire graphene structure and highlights a specific path
        between two nodes using a different color.

        Parameters
        ----------
        path : List[int]
            A list of node IDs representing the path to be highlighted.
        visualize_periodic_bonds : bool, optional
            Whether to visualize periodic boundary condition edges (default is True).

        Notes
        -----
        The path is highlighted in yellow, while the rest of the graphene structure
        is displayed in its default colors. Periodic boundary condition edges are
        shown with dashed lines if visualize_periodic_bonds is True.
        """
        # Get positions and elements of nodes
        pos = nx.get_node_attributes(self.graph, "position")
        elements = nx.get_node_attributes(self.graph, "element")

        # Determine colors for nodes, considering nitrogen species if present
        colors = [
            self.get_color(elements[node], self.graph.nodes[node].get("nitrogen_species"))
            for node in self.graph.nodes()
        ]
        labels = {node: f"{elements[node]}{node}" for node in self.graph.nodes()}

        # Separate periodic edges and regular edges
        regular_edges = [(u, v) for u, v, d in self.graph.edges(data=True) if not d.get("periodic")]
        periodic_edges = [(u, v) for u, v, d in self.graph.edges(data=True) if d.get("periodic")]

        # Initialize plot
        plt.figure(figsize=(12, 12))

        # Draw the regular edges
        nx.draw(self.graph, pos, edgelist=regular_edges, node_color=colors, node_size=200, with_labels=False)

        # Draw periodic edges with dashed lines if visualize_periodic_bonds is True
        if visualize_periodic_bonds:
            nx.draw_networkx_edges(self.graph, pos, edgelist=periodic_edges, style="dashed", edge_color="gray")

        # Highlight the nodes and edges in the specified path
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_nodes(self.graph, pos, nodelist=path, node_color="yellow", node_size=300)
        nx.draw_networkx_edges(self.graph, pos, edgelist=path_edges, edge_color="yellow", width=2)

        # Draw labels for nodes
        nx.draw_networkx_labels(self.graph, pos, labels=labels, font_size=10, font_color="cyan", font_weight="bold")

        plt.show()

    def plot_graphene_with_depth_neighbors_based_on_bond_length(
        self, atom_id: int, max_distance: float, visualize_periodic_bonds: bool = True
    ):
        """
        Plot the graphene structure with neighbors highlighted based on bond length.

        This method plots the entire graphene structure and highlights nodes that are within
        a specified maximum distance from a given atom, using the bond length as the distance metric.

        Parameters
        ----------
        atom_id : int
            The ID of the atom from which distances are measured.
        max_distance : float
            The maximum bond length distance within which neighbors are highlighted.
        visualize_periodic_bonds : bool, optional
            Whether to visualize periodic boundary condition edges (default is True).

        Notes
        -----
        The neighbors within the specified distance are highlighted in yellow, while the rest
        of the graphene structure is displayed in its default colors. Periodic boundary condition edges are
        shown with dashed lines if visualize_periodic_bonds is True.
        """
        # Get positions and elements of nodes
        pos = nx.get_node_attributes(self.graph, "position")
        elements = nx.get_node_attributes(self.graph, "element")

        # Determine colors for nodes, considering nitrogen species if present
        colors = [
            self.get_color(elements[node], self.graph.nodes[node].get("nitrogen_species"))
            for node in self.graph.nodes()
        ]
        labels = {node: f"{elements[node]}{node}" for node in self.graph.nodes()}

        # Separate periodic edges and regular edges
        regular_edges = [(u, v) for u, v, d in self.graph.edges(data=True) if not d.get("periodic")]
        periodic_edges = [(u, v) for u, v, d in self.graph.edges(data=True) if d.get("periodic")]

        # Initialize plot
        plt.figure(figsize=(12, 12))

        # Draw the regular edges
        nx.draw(self.graph, pos, edgelist=regular_edges, node_color=colors, node_size=200, with_labels=False)

        # Draw periodic edges with dashed lines if visualize_periodic_bonds is True
        if visualize_periodic_bonds:
            nx.draw_networkx_edges(self.graph, pos, edgelist=periodic_edges, style="dashed", edge_color="gray")

        # Compute shortest path lengths from the specified atom using bond lengths
        paths = nx.single_source_dijkstra_path_length(self.graph, atom_id, cutoff=max_distance, weight="bond_length")

        # Identify neighbors within the specified maximum distance
        depth_neighbors = [node for node, length in paths.items() if length <= max_distance]
        path_edges = [(u, v) for u in depth_neighbors for v in self.graph.neighbors(u) if v in depth_neighbors]

        # Highlight the identified neighbors and their connecting edges
        nx.draw_networkx_nodes(self.graph, pos, nodelist=depth_neighbors, node_color="yellow", node_size=300)
        nx.draw_networkx_edges(self.graph, pos, edgelist=path_edges, edge_color="yellow", width=2)

        # Draw labels for nodes
        nx.draw_networkx_labels(self.graph, pos, labels=labels, font_size=10, font_color="cyan", font_weight="bold")

        # Show plot
        plt.show()

    def plot_nodes_within_distance(self, nodes_within_distance: List[int], visualize_periodic_bonds: bool = True):
        """
        Plot the graphene structure with neighbors highlighted based on distance.

        This method plots the entire graphene structure and highlights nodes that are within
        a specified maximum distance from a given atom, using the bond length as the distance metric.

        Parameters
        ----------
        nodes_within_distance : List[int]
            A list of node IDs representing the neighbors within the given distance.
        visualize_periodic_bonds : bool, optional
            Whether to visualize periodic boundary condition edges (default is True).

        Notes
        -----
        The neighbors within the specified distance are highlighted in yellow, while the rest
        of the graphene structure is displayed in its default colors. Periodic boundary condition edges are
        shown with dashed lines if visualize_periodic_bonds is True.
        """
        # Get positions and elements of nodes
        pos = nx.get_node_attributes(self.graph, "position")
        elements = nx.get_node_attributes(self.graph, "element")

        # Determine colors for nodes, considering nitrogen species if present
        colors = [
            self.get_color(elements[node], self.graph.nodes[node].get("nitrogen_species"))
            for node in self.graph.nodes()
        ]
        labels = {node: f"{elements[node]}{node}" for node in self.graph.nodes()}

        # Separate periodic edges and regular edges
        regular_edges = [(u, v) for u, v, d in self.graph.edges(data=True) if not d.get("periodic")]
        periodic_edges = [(u, v) for u, v, d in self.graph.edges(data=True) if d.get("periodic")]

        # Initialize plot
        plt.figure(figsize=(12, 12))

        # Draw the regular edges
        nx.draw(self.graph, pos, edgelist=regular_edges, node_color=colors, node_size=200, with_labels=False)

        # Draw periodic edges with dashed lines if visualize_periodic_bonds is True
        if visualize_periodic_bonds:
            nx.draw_networkx_edges(self.graph, pos, edgelist=periodic_edges, style="dashed", edge_color="gray")

        # Compute edges within the specified distance
        path_edges = [
            (u, v) for u in nodes_within_distance for v in self.graph.neighbors(u) if v in nodes_within_distance
        ]

        # Highlight the identified neighbors and their connecting edges
        nx.draw_networkx_nodes(self.graph, pos, nodelist=nodes_within_distance, node_color="yellow", node_size=300)
        nx.draw_networkx_edges(self.graph, pos, edgelist=path_edges, edge_color="yellow", width=2)

        # Draw labels for nodes
        nx.draw_networkx_labels(self.graph, pos, labels=labels, font_size=10, font_color="cyan", font_weight="bold")

        # Show plot
        plt.show()


def write_xyz(graph, filename):
    with open(filename, "w") as file:
        file.write(f"{graph.number_of_nodes()}\n")
        file.write("XYZ file generated from GrapheneGraph\n")
        for node_id, node_data in graph.nodes(data=True):
            x, y = node_data["position"]
            element = node_data["element"]
            file.write(f"{element} {x:.3f} {y:.3f} 0.000\n")


def main():
    # Set seed for reproducibility
    # random.seed(42)

    graphene = GrapheneGraph(bond_distance=1.42, sheet_size=(20, 20))

    # write_xyz(graphene.graph, 'graphene.xyz')
    graphene.plot_graphene(with_labels=True)

    # Find direct neighbors of a node (depth=1)
    direct_neighbors = graphene.get_neighbors_via_edges(atom_id=0, depth=1)
    print(f"Direct neighbors of C_0: {direct_neighbors}")

    # Find neighbors of a node at an exact depth (depth=2)
    depth_neighbors = graphene.get_neighbors_via_edges(atom_id=0, depth=2)
    print(f"Neighbors of C_0 at depth 2: {depth_neighbors}")

    # Find neighbors of a node up to a certain depth (inclusive=True)
    inclusive_neighbors = graphene.get_neighbors_via_edges(atom_id=0, depth=2, inclusive=True)
    print(f"Neighbors of C_0 up to depth 2 (inclusive): {inclusive_neighbors}")

    # graphene.add_nitrogen_doping_old(10, NitrogenSpecies.GRAPHITIC)
    # graphene.plot_graphene(with_labels=True, visualize_periodic_bonds=False)

    graphene.add_nitrogen_doping(percentages={NitrogenSpecies.PYRIDINIC_2: 10})
    graphene.plot_graphene(with_labels=True, visualize_periodic_bonds=False)

    source = 0
    target = 10
    path = graphene.get_shortest_path(source, target)
    print(f"Shortest path from C_{source} to C_{target}: {path}")
    graphene.plot_graphene_with_path(path)

    graphene.plot_graphene_with_depth_neighbors_based_on_bond_length(0, 5)

    # Find nodes within a certain distance from a source node
    atom_id = 5
    max_distance = 5
    nodes_within_distance = graphene.get_neighbors_within_distance(atom_id, max_distance)
    print(f"Nodes within {max_distance} distance from node {atom_id}: {nodes_within_distance}")

    # Plot the nodes within the specified distance
    graphene.plot_nodes_within_distance(nodes_within_distance)


if __name__ == "__main__":
    main()
