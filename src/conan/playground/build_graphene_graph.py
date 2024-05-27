from math import cos, pi, sin
from typing import List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class GrapheneGraph:
    def __init__(self, bond_distance: float, sheet_size: Tuple[float, float]):
        self.bond_distance = bond_distance
        self.sheet_size = sheet_size
        self.graph = nx.Graph()
        self._build_graphene_sheet()

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

    def get_direct_neighbors(self, atom_id: int) -> List[int]:
        """Get the direct neighbors of a node based on its ID."""
        return list(self.graph.neighbors(atom_id))

    def get_neighbors(self, atom_id: int, depth: int = 1) -> List[int]:
        """Get neighbors of a given atom up to a certain depth."""
        paths = nx.single_source_shortest_path_length(self.graph, atom_id, cutoff=depth)
        return [node for node, length in paths.items() if length == depth]

    def get_neighbors_paths(self, atom_id: int, depth: int = 1) -> List[Tuple[int, int]]:
        """Get edges of paths to neighbors up to a certain depth."""
        paths = nx.single_source_shortest_path(self.graph, atom_id, cutoff=depth)
        edges = []
        for path in paths.values():
            edges.extend([(path[i], path[i + 1]) for i in range(len(path) - 1)])
        return edges

    def get_shortest_path_length(self, source: int, target: int) -> float:
        """Get the shortest path length between two atoms based on bond lengths."""
        return nx.dijkstra_path_length(self.graph, source, target, weight="bond_length")

    def get_shortest_path(self, source: int, target: int) -> List[int]:
        """Get the shortest path between two atoms based on bond lengths."""
        return nx.dijkstra_path(self.graph, source, target, weight="bond_length")

    def get_color(self, element: str) -> str:
        """Get the color of an element for plotting."""
        colors = {"C": "black"}
        return colors.get(element, "red")

    def plot_graphene(self, with_labels: bool = False):
        """Plot the graphene structure using networkx and matplotlib."""
        pos = nx.get_node_attributes(self.graph, "position")
        elements = nx.get_node_attributes(self.graph, "element")
        colors = [self.get_color(elements[node]) for node in self.graph.nodes()]

        plt.figure(figsize=(12, 12))
        if with_labels:
            labels = {node: f"{elements[node]}{node}" for node in self.graph.nodes()}
            nx.draw(
                self.graph, pos, labels=elements, with_labels=with_labels, node_color=colors, node_size=200, font_size=8
            )
            nx.draw_networkx_labels(self.graph, pos, labels=labels, font_size=10, font_color="cyan", font_weight="bold")
        else:
            nx.draw(self.graph, pos, with_labels=with_labels, node_color=colors, node_size=200)
        plt.show()

    def plot_graphene_with_depth_neighbors(self, atom_id: int, depth: int):
        """Plot the graphene structure with neighbors up to a certain depth highlighted."""
        pos = nx.get_node_attributes(self.graph, "position")
        elements = nx.get_node_attributes(self.graph, "element")
        colors = [self.get_color(elements[node]) for node in self.graph.nodes()]
        labels = {node: f"{elements[node]}{node}" for node in self.graph.nodes()}

        # Draw the entire graphene structure
        plt.figure(figsize=(12, 12))
        nx.draw(self.graph, pos, node_color=colors, node_size=200, with_labels=False)

        # Get neighbors and paths up to a certain depth
        neighbors = self.get_neighbors(atom_id, depth)
        path_edges = self.get_neighbors_paths(atom_id, depth)

        # Highlight the nodes and edges in the shortest path
        nx.draw_networkx_nodes(self.graph, pos, nodelist=neighbors, node_color="yellow", node_size=300)
        nx.draw_networkx_edges(self.graph, pos, edgelist=path_edges, edge_color="yellow", width=2)
        nx.draw_networkx_labels(self.graph, pos, labels=labels, font_size=10, font_color="cyan", font_weight="bold")

        plt.show()

    def plot_graphene_with_path(self, path: List[int]):
        """Plot the graphene structure with a highlighted path using networkx and matplotlib."""
        pos = nx.get_node_attributes(self.graph, "position")
        elements = nx.get_node_attributes(self.graph, "element")
        colors = [self.get_color(elements[node]) for node in self.graph.nodes()]
        labels = {node: f"{elements[node]}{node}" for node in self.graph.nodes()}

        # Draw the entire graphene structure
        plt.figure(figsize=(12, 12))
        nx.draw(self.graph, pos, node_color=colors, node_size=200, with_labels=False)

        # Highlight the nodes and edges in the shortest path
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_nodes(self.graph, pos, nodelist=path, node_color="yellow", node_size=300)
        nx.draw_networkx_edges(self.graph, pos, edgelist=path_edges, edge_color="yellow", width=2)
        nx.draw_networkx_labels(self.graph, pos, labels=labels, font_size=10, font_color="cyan", font_weight="bold")

        plt.show()

    def plot_graphene_with_neighbors_based_on_bond_length(self, atom_id: int, max_distance: float):
        """Plot the graphene structure with neighbors up to a certain distance highlighted based on bond lengths."""
        pos = nx.get_node_attributes(self.graph, "position")
        elements = nx.get_node_attributes(self.graph, "element")
        colors = [self.get_color(elements[node]) for node in self.graph.nodes()]
        labels = {node: f"{elements[node]}{node}" for node in self.graph.nodes()}

        # Draw the entire graphene structure
        plt.figure(figsize=(12, 12))
        nx.draw(self.graph, pos, node_color=colors, node_size=200, with_labels=False)

        # Get neighbors up to a certain distance
        paths = nx.single_source_dijkstra_path(self.graph, atom_id, cutoff=max_distance, weight="bond_length")
        depth_neighbors = [
            node
            for node, path in paths.items()
            if sum(
                nx.get_edge_attributes(self.graph, "bond_length").get((path[i], path[i + 1]), 0)
                for i in range(len(path) - 1)
            )
            <= max_distance
        ]
        path_edges = [
            (path[i], path[i + 1])
            for path in paths.values()
            for i in range(len(path) - 1)
            if sum(
                nx.get_edge_attributes(self.graph, "bond_length").get((path[i], path[i + 1]), 0)
                for i in range(len(path) - 1)
            )
            <= max_distance
        ]

        # Highlight the nodes and edges in the shortest path
        nx.draw_networkx_nodes(self.graph, pos, nodelist=depth_neighbors, node_color="yellow", node_size=300)
        nx.draw_networkx_edges(self.graph, pos, edgelist=path_edges, edge_color="yellow", width=2)
        nx.draw_networkx_labels(self.graph, pos, labels=labels, font_size=10, font_color="cyan", font_weight="bold")

        plt.show()


def write_xyz(graph, filename):
    with open(filename, "w") as file:
        # Write the number of atoms to the beginning of the file
        file.write(f"{graph.number_of_nodes()}\n")
        file.write("XYZ file generated from GrapheneGraph\n")  # Write a comment line
        for node_id, node_data in graph.nodes(data=True):
            x, y = node_data["position"]
            element = node_data["element"]
            file.write(f"{element} {x:.3f} {y:.3f} 0.000\n")  # z-coordinate is set to 0


# def draw_graph(G):  # ToDo: Methode könnte in Mutterklasse ausgelagert werden später
#     # Get node positions from the graph object
#     pos = nx.get_node_attributes(G, 'position')
#     nx.draw(G, pos, node_size=50, node_color='black', with_labels=False)
#
#     plt.gca().set_aspect('equal', adjustable='datalim')
#     plt.title('Graphene Lattice Structure')
#     plt.show()


def main():
    graphene = GrapheneGraph(bond_distance=1.42, sheet_size=(20, 20))

    # Save and plot the graphene structure
    write_xyz(graphene.graph, "graphene.xyz")
    graphene.plot_graphene(with_labels=True)

    ##############################################################################################

    # Find direct neighbors of a node
    print("Neighbors of C_0:", graphene.get_direct_neighbors(0))
    print("Neighbors of C_5:", graphene.get_direct_neighbors(5))

    # Find neighbors of a node up to a certain depth
    print("Neighbors of C_0 up to depth 2:", graphene.get_neighbors(0, depth=2))
    print("Neighbors of C_5 up to depth 2:", graphene.get_neighbors(5, depth=2))

    # Plot the graphene structure with neighbors up to a certain depth highlighted
    graphene.plot_graphene_with_depth_neighbors(0, depth=2)
    graphene.plot_graphene_with_depth_neighbors(5, depth=2)

    ##############################################################################################

    # Find the shortest path between two atoms using the dijkstra algorithm
    source = 0
    target = 10

    # Find the shortest path length based on bond lengths
    path_length = graphene.get_shortest_path_length(source, target)
    print(f"Shortest path length from C_{source} to C_{target}: {path_length}")

    # Find the shortest path based on bond lengths
    path = graphene.get_shortest_path(source, target)
    print(f"Shortest path from C_{source} to C_{target}: {path}")

    # Plot the graphene structure with the shortest path highlighted
    graphene.plot_graphene_with_path(path)

    ##############################################################################################

    # Plot the graphene structure with neighbors up to a certain distance highlighted
    max_distance = 5  # Example maximum distance (3 bonds)
    graphene.plot_graphene_with_neighbors_based_on_bond_length(5, max_distance)


if __name__ == "__main__":
    main()
