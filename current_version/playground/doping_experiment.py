import networkx as nx
import matplotlib.pyplot as plt
from typing import Tuple, List
from math import cos, sin, pi
import random
from enum import Enum


class NitrogenType(Enum):
    GRAPHITIC = 'graphitic'
    PYRIDINIC = 'pyridinic'
    PYRROLIC = 'pyrrolic'
    PYRAZOLE = 'pyrazole'


class GrapheneGraph:
    def __init__(self, bond_distance: float, sheet_size: Tuple[float, float]):
        self.bond_distance = bond_distance
        self.sheet_size = sheet_size
        self.graph = nx.Graph()
        self._build_graphene_sheet()

    @property
    def cc_x_distance(self):
        return self.bond_distance * sin(pi / 6)

    @property
    def cc_y_distance(self):
        return self.bond_distance * cos(pi / 6)

    @property
    def num_cells_x(self):
        return int(self.sheet_size[0] // (2 * self.bond_distance + 2 * self.cc_x_distance))

    @property
    def num_cells_y(self):
        return int(self.sheet_size[1] // (2 * self.cc_y_distance))

    def _build_graphene_sheet(self):
        index = 0
        for y in range(self.num_cells_y):
            for x in range(self.num_cells_x):
                x_offset = x * (2 * self.bond_distance + 2 * self.cc_x_distance)
                y_offset = y * (2 * self.cc_y_distance)

                self._add_unit_cell(index, x_offset, y_offset)

                if x > 0:
                    self.graph.add_edge(index - 1, index, bond_length=self.bond_distance)
                if y > 0:
                    self.graph.add_edge(index - 4 * self.num_cells_x + 1, index, bond_length=self.bond_distance)
                    self.graph.add_edge(index - 4 * self.num_cells_x + 2, index + 3, bond_length=self.bond_distance)

                index += 4

    def _add_unit_cell(self, index: int, x_offset: float, y_offset: float):
        unit_cell_positions = [
            (x_offset, y_offset),
            (x_offset + self.cc_x_distance, y_offset + self.cc_y_distance),
            (x_offset + self.cc_x_distance + self.bond_distance, y_offset + self.cc_y_distance),
            (x_offset + 2 * self.cc_x_distance + self.bond_distance, y_offset)
        ]

        for i, pos in enumerate(unit_cell_positions):
            self.graph.add_node(index + i, element='C', position=pos)

        for i in range(len(unit_cell_positions) - 1):
            self.graph.add_edge(index + i, index + i + 1, bond_length=self.bond_distance)

    def add_nitrogen_doping(self, percentage: float, type_of_nitrogen: NitrogenType = NitrogenType.GRAPHITIC):
        if not isinstance(type_of_nitrogen, NitrogenType):
            raise ValueError(
                f"Invalid nitrogen type: {type_of_nitrogen}. Valid types are: "
                f"{', '.join([e.value for e in NitrogenType])}")

        num_atoms = self.graph.number_of_nodes()
        num_nitrogen = int(num_atoms * percentage / 100)

        carbon_atoms = [node for node, data in self.graph.nodes(data=True) if data['element'] == 'C']
        chosen_atoms = random.sample(carbon_atoms, num_nitrogen)

        for atom_id in chosen_atoms:
            self.graph.nodes[atom_id]['element'] = 'N'
            self.graph.nodes[atom_id]['type_of_nitrogen'] = type_of_nitrogen
            if type_of_nitrogen == NitrogenType.GRAPHITIC:
                continue
            elif type_of_nitrogen == NitrogenType.PYRIDINIC:
                # Implement pyridinic nitrogen replacement
                pass
            elif type_of_nitrogen == NitrogenType.PYRROLIC:
                # Implement pyrrolic nitrogen replacement
                pass
            elif type_of_nitrogen == NitrogenType.PYRAZOLE:
                # Implement pyrazole nitrogen replacement
                pass

    def get_direct_neighbors(self, atom_id: int) -> List[int]:
        return list(self.graph.neighbors(atom_id))

    def get_neighbors(self, atom_id: int, depth: int = 1) -> List[int]:
        paths = nx.single_source_shortest_path_length(self.graph, atom_id, cutoff=depth)
        return [node for node, length in paths.items() if length == depth]

    def get_shortest_path_length(self, source: int, target: int) -> float:
        return nx.dijkstra_path_length(self.graph, source, target, weight='bond_length')

    def get_shortest_path(self, source: int, target: int) -> List[int]:
        return nx.dijkstra_path(self.graph, source, target, weight='bond_length')

    def get_color(self, element: str, type_of_nitrogen: NitrogenType = None) -> str:
        colors = {'C': 'black'}
        nitrogen_colors = {
            NitrogenType.PYRIDINIC: 'blue',
            NitrogenType.GRAPHITIC: 'red',
            NitrogenType.PYRROLIC: 'cyan',
            NitrogenType.PYRAZOLE: 'green'
        }
        if type_of_nitrogen in nitrogen_colors:
            return nitrogen_colors[type_of_nitrogen]
        return colors.get(element, 'pink')

    def plot_graphene(self, with_labels: bool = False):
        pos = nx.get_node_attributes(self.graph, 'position')
        elements = nx.get_node_attributes(self.graph, 'element')
        colors = [self.get_color(elements[node], self.graph.nodes[node].get('type_of_nitrogen')) for node in
                  self.graph.nodes()]

        plt.figure(figsize=(12, 12))
        if with_labels:
            labels = {node: f'{elements[node]}{node}' for node in self.graph.nodes()}
            nx.draw(self.graph, pos, labels=elements, with_labels=with_labels, node_color=colors, node_size=200,
                    font_size=8)
            nx.draw_networkx_labels(self.graph, pos, labels=labels, font_size=10, font_color='cyan', font_weight='bold')
        else:
            nx.draw(self.graph, pos, with_labels=with_labels, node_color=colors, node_size=200)
        plt.show()

    def plot_graphene_with_path(self, path: List[int]):
        pos = nx.get_node_attributes(self.graph, 'position')
        elements = nx.get_node_attributes(self.graph, 'element')
        colors = [self.get_color(elements[node], self.graph.nodes[node].get('type_of_nitrogen')) for node in
                  self.graph.nodes()]
        labels = {node: f'{elements[node]}{node}' for node in self.graph.nodes()}

        plt.figure(figsize=(12, 12))
        nx.draw(self.graph, pos, node_color=colors, node_size=200, with_labels=False)

        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_nodes(self.graph, pos, nodelist=path, node_color='yellow', node_size=300)
        nx.draw_networkx_edges(self.graph, pos, edgelist=path_edges, edge_color='yellow', width=2)
        nx.draw_networkx_labels(self.graph, pos, labels=labels, font_size=10, font_color='cyan', font_weight='bold')

        plt.show()

    def plot_graphene_with_depth_neighbors_based_on_bond_length(self, atom_id: int, max_distance: float):
        pos = nx.get_node_attributes(self.graph, 'position')
        elements = nx.get_node_attributes(self.graph, 'element')
        colors = [self.get_color(elements[node], self.graph.nodes[node].get('type_of_nitrogen')) for node in
                  self.graph.nodes()]
        labels = {node: f'{elements[node]}{node}' for node in self.graph.nodes()}

        plt.figure(figsize=(12, 12))
        nx.draw(self.graph, pos, node_color=colors, node_size=200, with_labels=False)

        paths = nx.single_source_dijkstra_path_length(self.graph, atom_id, cutoff=max_distance, weight='bond_length')
        depth_neighbors = [node for node, length in paths.items() if length <= max_distance]
        path_edges = [(u, v) for u in depth_neighbors for v in self.graph.neighbors(u) if v in depth_neighbors]

        nx.draw_networkx_nodes(self.graph, pos, nodelist=depth_neighbors, node_color='yellow', node_size=300)
        nx.draw_networkx_edges(self.graph, pos, edgelist=path_edges, edge_color='yellow', width=2)
        nx.draw_networkx_labels(self.graph, pos, labels=labels, font_size=10, font_color='cyan', font_weight='bold')

        plt.show()


def write_xyz(graph, filename):
    with open(filename, 'w') as file:
        file.write(f"{graph.number_of_nodes()}\n")
        file.write("XYZ file generated from GrapheneGraph\n")
        for node_id, node_data in graph.nodes(data=True):
            x, y = node_data['position']
            element = node_data['element']
            file.write(f"{element} {x:.3f} {y:.3f} 0.000\n")


def main():
    # Set seed for reproducibility
    random.seed(42)

    graphene = GrapheneGraph(bond_distance=1.42, sheet_size=(20, 20))

    write_xyz(graphene.graph, 'graphene.xyz')
    graphene.plot_graphene(with_labels=True)

    graphene.add_nitrogen_doping(10, NitrogenType.GRAPHITIC)
    graphene.plot_graphene(with_labels=True)

    source = 0
    target = 10
    path = graphene.get_shortest_path(source, target)
    print(f"Shortest path from C_{source} to C_{target}: {path}")
    graphene.plot_graphene_with_path(path)

    graphene.plot_graphene_with_depth_neighbors_based_on_bond_length(0, 5)


if __name__ == "__main__":
    main()
