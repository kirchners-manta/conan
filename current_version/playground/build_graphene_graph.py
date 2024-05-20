import networkx as nx
import matplotlib.pyplot as plt
from typing import Tuple, List
from math import cos, sin, pi


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
        """Build the graphene sheet structure by creating nodes and edges."""
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

    def _add_unit_cell(self, index: int, x_offset: float, y_offset: float):
        """Add nodes and internal bonds within a unit cell."""
        unit_cell_positions = [
            (x_offset, y_offset),
            (x_offset + self.cc_x_distance, y_offset + self.cc_y_distance),
            (x_offset + self.cc_x_distance + self.bond_distance, y_offset + self.cc_y_distance),
            (x_offset + 2 * self.cc_x_distance + self.bond_distance, y_offset)
        ]

        # Add nodes
        for i, pos in enumerate(unit_cell_positions):
            self.graph.add_node(index + i, element='C', position=pos)

        # Add internal bonds within the unit cell
        for i in range(len(unit_cell_positions) - 1):
            self.graph.add_edge(index + i, index + i + 1, bond_length=self.bond_distance)

    def get_direct_neighbors(self, atom_id: int) -> List[int]:
        """Get the direct neighbors of a node based on its ID."""
        return list(self.graph.neighbors(atom_id))

    def get_neighbors(self, atom_id: int, depth: int = 1) -> List[int]:
        """Get neighbors of a given atom up to a certain depth."""
        paths = nx.single_source_shortest_path_length(self.graph, atom_id, cutoff=depth)
        return [node for node, length in paths.items() if length == depth]

    def get_color(self, element: str) -> str:
        """Get the color of an element for plotting."""
        colors = {'C': 'black'}
        return colors.get(element, 'red')

    def plot_graphene(self, with_labels: bool = False):
        """Plot the graphene structure using networkx and matplotlib."""
        pos = nx.get_node_attributes(self.graph, 'position')
        elements = nx.get_node_attributes(self.graph, 'element')
        colors = [self.get_color(elements[node]) for node in self.graph.nodes()]

        plt.figure(figsize=(12, 12))
        if with_labels:
            labels = {node: f'{elements[node]}{node}' for node in self.graph.nodes()}
            nx.draw(self.graph, pos, labels=elements, with_labels=with_labels, node_color=colors, node_size=200,
                    font_size=8)
            nx.draw_networkx_labels(self.graph, pos, labels=labels, font_size=10, font_color='cyan', font_weight='bold')
        else:
            nx.draw(self.graph, pos, with_labels=with_labels, node_color=colors, node_size=200)
        plt.show()


def write_xyz(graph, filename):
    with open(filename, 'w') as file:
        # Write the number of atoms to the beginning of the file
        file.write(f"{graph.number_of_nodes()}\n")
        file.write("XYZ file generated from GrapheneGraph\n")  # Write a comment line
        for node_id, node_data in graph.nodes(data=True):
            x, y = node_data['position']
            element = node_data['element']
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

    # Find direct neighbors of a node
    print(f"Neighbors of C_0:", graphene.get_direct_neighbors(0))
    print(f"Neighbors of C_5:", graphene.get_direct_neighbors(5))

    # Find neighbors of a node up to a certain depth
    print(f"Neighbors of C_0 up to depth 2:", graphene.get_neighbors(0, depth=2))
    print(f"Neighbors of C_5 up to depth 2:", graphene.get_neighbors(5, depth=2))

    # Save and plot the graphene structure
    write_xyz(graphene.graph, 'graphene.xyz')
    graphene.plot_graphene(with_labels=True)


if __name__ == "__main__":
    main()
