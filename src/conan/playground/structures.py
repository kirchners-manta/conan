# from typing import TYPE_CHECKING
#
# if TYPE_CHECKING:
#     from conan.playground.doping import DopingStructureCollection
import copy
import math
import warnings
from abc import ABC, abstractmethod
from functools import cached_property

# from functools import cache
from math import cos, pi, sin
from pathlib import Path
from typing import List, Optional, Tuple, Union

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from conan.playground.doping import (
    DopingHandler,
    DopingStructureCollection,
    NitrogenSpecies,
    OptimizationWeights,
    StructuralComponents,
)
from conan.playground.structure_optimizer import OptimizationConfig, StructureOptimizer
from conan.playground.utils import Position, create_position


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
class Structure2D(MaterialStructure, ABC):
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
            NitrogenSpecies.get_color(elements[node], self.graph.nodes[node].get("nitrogen_species"))
            for node in self.graph.nodes()
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
            color = NitrogenSpecies.get_color("N", species)
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
class Structure3D(MaterialStructure, ABC):
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
            NitrogenSpecies.get_color(elements[node], self.graph.nodes[node].get("nitrogen_species"))
            for node in self.graph.nodes()
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
            color = NitrogenSpecies.get_color("N", species)
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


class CombinedStructure(MaterialStructure, ABC):
    """
    Abstract base class for combined structures.
    Provides shared logic for combined structures like StackedGraphene and Pore.
    """

    def __init__(self):
        super().__init__()
        self._combined_graph = None
        self._combined_doping_handler = None

    @cached_property
    def graph(self):
        """
        Returns the combined graph of all substructures in the stack.
        Caches the result to avoid re-computation unless invalidated.
        """
        if self._combined_graph is None:
            self.build_structure()
        return self._combined_graph

    @cached_property
    def doping_handler(self):
        """
        Returns a DopingHandler that contains all doping structures from all substructures.
        Caches the result to avoid re-computation unless invalidated.
        """
        if self._combined_doping_handler is None:
            self._build_doping_handler()
        return self._combined_doping_handler

    def _invalidate_cache(self):
        """Invalidate the cached properties."""
        self._combined_graph = None
        self._combined_doping_handler = None
        if "graph" in self.__dict__:
            del self.__dict__["graph"]
        if "doping_handler" in self.__dict__:
            del self.__dict__["doping_handler"]

    @abstractmethod
    def build_structure(self):
        """
        Abstract method to build the combined structure.
        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def get_components(self):
        """
        Abstract method to retrieve the individual components of the combined structure.
        Must be implemented by subclasses.
        """
        pass

    def _build_doping_handler(self):
        """
        Build the doping handler for the stacked material structure.
        """
        # Combine doping structures from all material structures
        combined_doping_handler = DopingHandler(self)
        combined_doping_structures = DopingStructureCollection()

        for component in self.get_components():
            # Add doping structures from the material structure
            for doping_structure in component.doping_handler.doping_structures:
                combined_doping_structures.add_structure(doping_structure)
            # Combine chosen_atoms
            for species, atoms in component.doping_handler.doping_structures.chosen_atoms.items():
                combined_doping_structures.chosen_atoms[species].extend(atoms)

        combined_doping_handler.doping_structures = combined_doping_structures
        self._combined_doping_handler = combined_doping_handler

    @staticmethod
    def _adjust_node_ids(mat_structure: MaterialStructure, node_id_offset: int):
        """
        Adjust the node IDs of a structure by a given offset.

        Parameters
        ----------
        mat_structure: MaterialStructure
            The material structure whose node IDs need to be adjusted.
        node_id_offset: int
            The offset to add to each node ID.
        """
        # Create a mapping from old to new node IDs
        mapping = {node: node + node_id_offset for node in mat_structure.graph.nodes()}
        # Update the node IDs in the graph
        mat_structure.graph = nx.relabel_nodes(mat_structure.graph, mapping)
        mat_structure.doping_handler.graph = mat_structure.graph
        # Update the node IDs in the doping structures
        for doping_structure in mat_structure.doping_handler.doping_structures:
            # Adjust nitrogen atoms
            doping_structure.nitrogen_atoms = [node + node_id_offset for node in doping_structure.nitrogen_atoms]
            # Adjust cycle
            if doping_structure.cycle is not None:
                doping_structure.cycle = [node + node_id_offset for node in doping_structure.cycle]
            # Adjust neighboring atoms
            if doping_structure.neighboring_atoms is not None:
                doping_structure.neighboring_atoms = [
                    node + node_id_offset for node in doping_structure.neighboring_atoms
                ]
            # Adjust structural components
            if doping_structure.structural_components is not None:
                components = StructuralComponents([], [])
                components.structure_building_atoms.extend(
                    [node + node_id_offset for node in doping_structure.structural_components.structure_building_atoms]
                )
                components.structure_building_neighbors.extend(
                    [
                        node + node_id_offset
                        for node in doping_structure.structural_components.structure_building_neighbors
                    ]
                )
                doping_structure.structural_components = components
            # Adjust additional edge
            if doping_structure.additional_edge is not None:
                doping_structure.additional_edge = (
                    doping_structure.additional_edge[0] + node_id_offset,
                    doping_structure.additional_edge[1] + node_id_offset,
                )
            # Adjust subgraph node IDs
            if doping_structure.subgraph is not None:
                # Relabel the nodes in the subgraph
                doping_structure.subgraph = nx.relabel_nodes(doping_structure.subgraph, mapping)
            # Adjust chosen_atoms in DopingStructureCollection
            chosen_atoms = mat_structure.doping_handler.doping_structures.chosen_atoms
            for species, atoms in chosen_atoms.items():
                chosen_atoms[species] = [node + node_id_offset for node in atoms]


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
        optimization_weights: Optional["OptimizationWeights"] = None,
        adjust_positions: bool = False,
        optimization_config: Optional["OptimizationConfig"] = None,
    ):
        """
        Add nitrogen doping to the graphene sheet.

        This method calculates the optimal nitrogen doping distribution across various nitrogen species to achieve a
        target nitrogen percentage while balancing deviation from an even species distribution and the desired overall
        nitrogen concentration. If specified, it also optimizes atom positions after doping.

        Parameters
        ----------
        total_percentage : float, optional
            The total percentage of carbon atoms to replace with nitrogen atoms. Default is 10 if not specified.
        percentages : dict, optional
            A dictionary specifying the percentages for each nitrogen species. Keys should be NitrogenSpecies enum
            values and values should be the percentages for the corresponding species.
        optimization_weights : OptimizationWeights, optional
            An instance containing weights for the optimization objective function to balance the trade-off between
            gaining the desired nitrogen percentage and achieving an equal distribution among species.

            - nitrogen_percentage_weight: Weight for the deviation from the desired nitrogen percentage in the
              objective function.

            - equal_distribution_weight: Weight for the deviation from equal distribution among species in the
              objective function.

            **Note**: `optimization_weights` only have an effect if `total_percentage` is provided and is greater than
            the sum of specified `percentages`. If `total_percentage` is equal to or less than the sum of the individual
            `percentages`, the optimization solver will not be used, and an alternative method is employed to meet
            the specified percentages exactly.
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
        The doping process includes multiple stages:
          1. **Validation:** Ensures provided percentages are feasible and adjusts if necessary.
          2. **Optimization:** Solves a mixed-integer linear programming (MILP) problem to determine the best
             distribution of doping structures, respecting both the target nitrogen percentage and species distribution.
          3. **Insertion:** Incorporates the calculated nitrogen doping structures into the material utilizing a graph
             theoretical approach.
          4. **Adjustment (if needed):** Compensates for shortfalls in actual nitrogen levels if the desired target
             percentage is not reached due to space constraints.
          5. **Position Adjustment:** Optionally adjusts atom positions based on optimization configuration.
        """
        if not isinstance(adjust_positions, bool):
            raise ValueError(f"adjust_positions must be a Boolean, but got {type(adjust_positions).__name__}")

        # Delegate the doping process to the doping handler
        self.doping_handler.add_nitrogen_doping(total_percentage, percentages, optimization_weights)

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

        # Validate optimization_config
        if optimization_config is not None and not isinstance(optimization_config, OptimizationConfig):
            raise TypeError("optimization_config must be an instance of OptimizationConfig.")

        if optimization_config is None:
            optimization_config: "OptimizationConfig" = OptimizationConfig()  # Use default config if none provided

        # Existing code for position adjustment
        optimizer: "StructureOptimizer" = StructureOptimizer(self, optimization_config)
        optimizer.optimize_positions()

        # Set the flag to indicate positions have been adjusted
        self.positions_adjusted = True

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

        Raises
        ------
        ValueError
            If the hole radius is too large and would remove too much of the graphene sheet.
        """
        # Validate center
        if not isinstance(center, tuple) or len(center) != 2 or not all(isinstance(c, (int, float)) for c in center):
            raise ValueError("center must be a tuple of two numbers representing coordinates.")

        # Validate radius
        if not isinstance(radius, (int, float)) or radius <= 0:
            raise ValueError("radius must be a positive number.")

        # Check if the radius is appropriate for the sheet dimensions
        sheet_width, sheet_height = self.sheet_size
        max_radius = min(sheet_width, sheet_height) / 2 - 2 * self.c_c_bond_length  # Adding bond length as a buffer
        if radius > max_radius:
            raise ValueError(
                f"Hole radius {radius} is too large for the graphene sheet dimensions ({sheet_width}, {sheet_height}). "
                f"The radius must be small enough to leave a border around the sheet."
            )

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


class StackedGraphene(CombinedStructure, Structure3D):
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
        CombinedStructure.__init__(self)
        Structure3D.__init__(self)

        # Validate interlayer_spacing
        if not isinstance(interlayer_spacing, (int, float)):
            raise ValueError(f"interlayer_spacing must be a float or int, but got {type(interlayer_spacing).__name__}.")
        if interlayer_spacing <= 0:
            raise ValueError(f"interlayer_spacing must be positive, but got {interlayer_spacing}.")

        # Validate number_of_layers
        if not isinstance(number_of_layers, int):
            raise ValueError(f"number_of_layers must be an integer, but got {type(number_of_layers).__name__}.")
        if number_of_layers <= 0:
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

        self.graphene_sheets = [graphene_sheet]
        """A list to hold individual GrapheneSheet instances."""
        self.interlayer_spacing = interlayer_spacing
        """The spacing between layers in the z-direction."""
        self.number_of_layers = number_of_layers
        """The number of layers to stack."""

        # Generate additional layers by copying the base graphene sheet and shifting it
        self._generate_layers()

        # Build the initial structure by combining all graphene sheets
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

    def _generate_layers(self):
        # Initialize total number of nodes
        total_nodes = self.graphene_sheets[0].graph.number_of_nodes()

        # Add additional layers by copying the original graphene sheet
        for layer in range(1, self.number_of_layers):
            # Create a copy of the original graphene sheet and shift it
            new_sheet = copy.deepcopy(self.graphene_sheets[0])
            # Adjust the node IDs of the new sheet
            self._adjust_node_ids(new_sheet, total_nodes)
            # Shift the sheet according to the stacking type
            self._shift_sheet(new_sheet, layer)
            # Add the new sheet to the list
            self.graphene_sheets.append(new_sheet)
            # Update the total number of nodes
            total_nodes += new_sheet.graph.number_of_nodes()

    def build_structure(self):
        """
        Build the stacked graphene structure by combining all graphene sheets.
        """
        # Invalidate cached properties if any
        self._invalidate_cache()

        # Combine the graphs of all graphene sheets
        combined_graph = nx.Graph()
        for sheet in self.graphene_sheets:
            combined_graph.update(sheet.graph)
        self._combined_graph = combined_graph

    def get_components(self):
        return self.graphene_sheets

    def add_nitrogen_doping(
        self,
        total_percentage: float = None,
        percentages: dict = None,
        optimization_weights: Optional["OptimizationWeights"] = None,
        adjust_positions: bool = False,
        layers: Union[List[int], str] = "all",
        optimization_config: Optional["OptimizationConfig"] = None,
    ):
        """
        Add nitrogen doping to one or multiple layers in the stacked graphene structure.

        This method calculates the optimal nitrogen doping distribution across various nitrogen species to achieve a
        target nitrogen percentage while balancing deviation from an even species distribution and the desired overall
        nitrogen concentration. If specified, it also optimizes atom positions after doping.

        Parameters
        ----------
        total_percentage : float, optional
            The total percentage of carbon atoms to replace with nitrogen atoms. Default is 10 if not specified.
        percentages : dict, optional
            A dictionary specifying the percentages for each nitrogen species. Keys should be NitrogenSpecies enum
            values and values should be the percentages for the corresponding species.
        optimization_weights : OptimizationWeights, optional
            An instance containing weights for the optimization objective function to balance the trade-off between
            gaining the desired nitrogen percentage and achieving an equal distribution among species.

            - nitrogen_percentage_weight: Weight for the deviation from the desired nitrogen percentage in the
              objective function.

            - equal_distribution_weight: Weight for the deviation from equal distribution among species in the
              objective function.

            **Note**: `optimization_weights` only have an effect if `total_percentage` is provided and is greater than
            the sum of specified `percentages`. If `total_percentage` is equal to or less than the sum of the individual
            `percentages`, the optimization solver will not be used, and an alternative method is employed to meet
            the specified percentages exactly.
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
        - The doping process includes multiple stages:
              1. **Validation:** Ensures provided percentages are feasible and adjusts if necessary.
              2. **Optimization:** Solves a mixed-integer linear programming (MILP) problem to determine the best
                 distribution of doping structures, respecting both the target nitrogen percentage and species
                 distribution.
              3. **Insertion:** Incorporates the calculated nitrogen doping structures into the material utilizing a
                 graph theoretical approach.
              4. **Adjustment (if needed):** Compensates for shortfalls in actual nitrogen levels if the desired target
                 percentage is not reached due to space constraints.
              5. **Position Adjustment:** Optionally adjusts atom positions based on optimization configuration.
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
                optimization_weights=optimization_weights,
                adjust_positions=adjust_positions,
                optimization_config=optimization_config,
            )

    def add_nitrogen_doping_to_layer(
        self,
        layer_index: int,
        total_percentage: float = None,
        percentages: dict = None,
        optimization_weights: Optional["OptimizationWeights"] = None,
        adjust_positions: bool = False,
        optimization_config: Optional["OptimizationConfig"] = None,
    ):
        """
        Add nitrogen doping to a specific layer in the stacked graphene structure.

        This method calculates the optimal nitrogen doping distribution across various nitrogen species to achieve a
        target nitrogen percentage while balancing deviation from an even species distribution and the desired overall
        nitrogen concentration. If specified, it also optimizes atom positions after doping.

        Parameters
        ----------
        layer_index : int
            The index of the layer to dope.
        total_percentage : float, optional
            The total percentage of carbon atoms to replace with nitrogen atoms. Default is 10 if not specified.
        percentages : dict, optional
            A dictionary specifying the percentages for each nitrogen species. Keys should be NitrogenSpecies enum
            values and values should be the percentages for the corresponding species.
        optimization_weights : OptimizationWeights, optional
            An instance containing weights for the optimization objective function to balance the trade-off between
            gaining the desired nitrogen percentage and achieving an equal distribution among species.

            - nitrogen_percentage_weight: Weight for the deviation from the desired nitrogen percentage in the
              objective function.

            - equal_distribution_weight: Weight for the deviation from equal distribution among species in the
              objective function.

            **Note**: `optimization_weights` only have an effect if `total_percentage` is provided and is greater than
            the sum of specified `percentages`. If `total_percentage` is equal to or less than the sum of the individual
            `percentages`, the optimization solver will not be used, and an alternative method is employed to meet
            the specified percentages exactly.
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

        Notes
        -----
        The doping process includes multiple stages:
          1. **Validation:** Ensures provided percentages are feasible and adjusts if necessary.
          2. **Optimization:** Solves a mixed-integer linear programming (MILP) problem to determine the best
             distribution of doping structures, respecting both the target nitrogen percentage and species distribution.
          3. **Insertion:** Incorporates the calculated nitrogen doping structures into the material utilizing a graph
             theoretical approach.
          4. **Adjustment (if needed):** Compensates for shortfalls in actual nitrogen levels if the desired target
             percentage is not reached due to space constraints.
          5. **Position Adjustment:** Optionally adjusts atom positions based on optimization configuration.
        """
        if 0 <= layer_index < len(self.graphene_sheets):

            if optimization_config is None and adjust_positions:
                optimization_config: "OptimizationConfig" = OptimizationConfig()

            # Perform the doping
            self.graphene_sheets[layer_index].add_nitrogen_doping(
                total_percentage=total_percentage,
                percentages=percentages,
                optimization_weights=optimization_weights,
                adjust_positions=adjust_positions,
                optimization_config=optimization_config,
            )

            # Invalidate the cache after modifying the sheet in order to update the structure after doping
            self._invalidate_cache()
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

        # Invalidate the cache after modifying the sheet in order to update the structure after doping
        self._invalidate_cache()


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
        symmetry_angle = 360 / self.tube_size

        if self.conformation == "armchair":
            # Calculate the positions for the armchair conformation
            positions, z_max = self._calculate_armchair_positions(distance, symmetry_angle)
        else:
            # Calculate the positions for the zigzag conformation
            positions, z_max = self._calculate_zigzag_positions(distance, symmetry_angle)

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
        self, distance: float, symmetry_angle: float
    ) -> Tuple[List[Tuple[float, float, float]], float]:
        """
        Calculate atom positions for the zigzag conformation.

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

    def add_nitrogen_doping(
        self,
        total_percentage: float = None,
        percentages: dict = None,
        optimization_weights: Optional["OptimizationWeights"] = None,
    ):
        """
        Add nitrogen doping to the CNT.

        This method calculates the optimal nitrogen doping distribution across various nitrogen species to achieve a
        target nitrogen percentage while balancing deviation from an even species distribution and the desired overall
        nitrogen concentration.
        Note that no position adjustment is implemented for three-dimensional structures and therefore not supported for
        CNTs as well.

        Parameters
        ----------
        total_percentage : float, optional
            The total percentage of carbon atoms to replace with nitrogen atoms. Default is 10 if not specified.
        percentages : dict, optional
            A dictionary specifying the percentages for each nitrogen species. Keys should be NitrogenSpecies enum
            values and values should be the percentages for the corresponding species.
        optimization_weights : OptimizationWeights, optional
            An instance containing weights for the optimization objective function to balance the trade-off between
            gaining the desired nitrogen percentage and achieving an equal distribution among species.

            - nitrogen_percentage_weight: Weight for the deviation from the desired nitrogen percentage in the
              objective function.

            - equal_distribution_weight: Weight for the deviation from equal distribution among species in the
              objective function.

            **Note**: `optimization_weights` only have an effect if `total_percentage` is provided and is greater than
            the sum of specified `percentages`. If `total_percentage` is equal to or less than the sum of the individual
            `percentages`, the optimization solver will not be used, and an alternative method is employed to meet
            the specified percentages exactly.

        Raises
        ------
        ValueError
            If the specific percentages exceed the total percentage.

        Notes
        -----
        The doping process includes multiple stages:
          1. **Validation:** Ensures provided percentages are feasible and adjusts if necessary.
          2. **Optimization:** Solves a mixed-integer linear programming (MILP) problem to determine the best
             distribution of doping structures, respecting both the target nitrogen percentage and species distribution.
          3. **Insertion:** Incorporates the calculated nitrogen doping structures into the material utilizing a graph
             theoretical approach.
          4. **Adjustment (if needed):** Compensates for shortfalls in actual nitrogen levels if the desired target
             percentage is not reached due to space constraints.
          5. **Position Adjustment:** Optionally adjusts atom positions based on optimization configuration.

        Warnings
        --------
        Note that three-dimensional position adjustment is currently not implemented in CONAN. Therefore, the generated
        doped structure should be used as a preliminary model and is recommended for further refinement using DFT or
        other computational methods. Future versions may include 3D position optimization.
        """
        # Delegate the doping process to the DopingHandler
        self.doping_handler.add_nitrogen_doping(total_percentage, percentages, optimization_weights)

        # Issue a user warning about the lack of 3D position adjustment
        warnings.warn(
            "3D position adjustment is not currently supported in CONAN. "
            "The generated doped structure should be treated with care and may be used as a basis for further DFT or "
            "other computational calculations."
            "Future versions may include 3D position optimization.",
            UserWarning,
        )


class Pore(CombinedStructure, Structure3D):
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
        CombinedStructure.__init__(self)
        Structure3D.__init__(self)

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

        # Assemble the components of the pore
        self._assemble_components()

        # Build the structure
        self.build_structure()

    def _assemble_components(self):
        """
        Assemble the components of the pore: two graphene sheets and a CNT.
        """
        # Create the graphene sheets and CNT
        self.graphene1 = GrapheneSheet(self.bond_length, self.sheet_size)
        self.graphene2 = GrapheneSheet(self.bond_length, self.sheet_size)
        self.cnt = CNT(
            bond_length=self.bond_length,
            tube_length=self.tube_length,
            tube_size=self.tube_size,
            conformation=self.conformation,
        )

        # Calculate the x and y shift to center the CNT in the middle of the graphene sheets
        x_shift = self.graphene1.actual_sheet_width / 2
        y_shift = self.graphene1.actual_sheet_height / 2

        # Position the CNT exactly in the center of the graphene sheets in the x and y directions
        self.cnt.translate(x_shift=x_shift, y_shift=y_shift)

        # Shift the second graphene sheet along the z-axis by the length of the CNT
        self.graphene2.translate(z_shift=self.cnt.actual_length)

        # Create holes in the graphene sheets
        center = (x_shift, y_shift)
        radius = self.cnt.actual_tube_diameter / 2 + self.bond_length
        self.graphene1.create_hole(center, radius)
        self.graphene2.create_hole(center, radius)

        # Adjust node IDs to prevent overlaps
        total_nodes = self.graphene1.graph.number_of_nodes()
        self._adjust_node_ids(self.cnt, total_nodes)
        total_nodes += self.cnt.graph.number_of_nodes()
        self._adjust_node_ids(self.graphene2, total_nodes)

    def build_structure(self):
        """
        Build the Pore structure by combining the graphene sheets and the CNT.
        """
        self._invalidate_cache()
        self._combined_graph = nx.disjoint_union_all([self.graphene1.graph, self.cnt.graph, self.graphene2.graph])

    def get_components(self):
        return [self.graphene1, self.cnt, self.graphene2]

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
