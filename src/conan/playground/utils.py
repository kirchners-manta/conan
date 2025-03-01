from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conan.playground.doping import NitrogenSpecies

from typing import List, NamedTuple, Tuple, Union

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from numba import jit
from numpy import typing as npt
from scipy.spatial import KDTree


class Position(NamedTuple):
    """
    Position: Named tuple to represent 3D coordinates of atoms.

    Attributes
    ----------
    x : float
        The x-coordinate of the atom.
    y : float
        The y-coordinate of the atom.
    z : float
        The z-coordinate of the atom.
    """

    x: float
    y: float
    z: float


# class Position2D(NamedTuple):
#     """
#     Position2D: Named tuple to represent 2D coordinates of atoms.
#
#     Attributes
#     ----------
#     x : float
#         The x-coordinate of the atom.
#     y : float
#         The y-coordinate of the atom.
#     """
#
#     x: float
#     y: float
#
#
# class Position3D(NamedTuple):
#     """
#     Position3D: Named tuple to represent 3D coordinates of atoms.
#
#     Attributes
#     ----------
#     x : float
#         The x-coordinate of the atom.
#     y : float
#         The y-coordinate of the atom.
#     z : float
#         The z-coordinate of the atom.
#     """
#
#     x: float
#     y: float
#     z: float


class Vector(NamedTuple):
    """
    Vector: Named tuple to represent the displacement between atoms.

    Attributes
    ----------
    dx : float
        The x-component of the displacement.
    dy : float
        The y-component of the displacement.
    dz : float
        The z-component of the displacement.
    """

    dx: float
    dy: float
    dz: float


def create_position(*args: Union[float, Tuple[float, float], Tuple[float, float, float]]):
    """
    Create a Position3D instance with an optional default z-coordinate (0.0 if not provided).

    Parameters
    ----------
    args
        Variable length argument list.
        - If two floats are provided, they are treated as x and y coordinates, with z=0.0 by default.
        - If three floats are provided, they are treated as x, y, and z coordinates.
        - If a single tuple of two or three floats is provided, it is unpacked to x, y, and z coordinates.

    Returns
    -------
    Position
        A Position object based on the input.
    """
    if len(args) == 1 and isinstance(args[0], tuple):
        args = args[0]  # Unpack tuple if a single tuple argument is passed

    if len(args) == 2:
        return Position(args[0], args[1], 0.0)  # Create 3D position with z=0.0 by default
    elif len(args) == 3:
        return Position(args[0], args[1], args[2])  # Create 3D position with provided z-coordinate
    else:
        raise ValueError("Invalid number of arguments for creating a Position. Expected 2 or 3 values.")


def minimum_image_distance(
    pos1: Position, pos2: Position, box_size: Union[Tuple[float, float], Tuple[float, float, float]]
) -> Tuple[float, Vector]:
    """
    Calculate the minimum distance between two positions considering periodic boundary conditions.

    Parameters
    ----------
    pos1 : Position
        Position of the first atom.
    pos2 : Position
        Position of the second atom.
    box_size : Union[Tuple[float, float], Tuple[float, float, float]]
        Size of the box in the x, y and optionally z dimensions (box_width, box_height, box_depth).

    Returns
    -------
    Tuple[float, Vector]
        A tuple containing:
        - The minimum distance between the two positions as a float.
        - The displacement vector accounting for periodic boundary conditions as a named tuple (dx, dy, dz).
    """
    # Check if the dimensions of pos1/pos2 and box_size match
    pos_dim = len(pos1)
    box_dim = len(box_size)

    if pos_dim != box_dim:
        raise ValueError(f"Dimension mismatch: positions are {pos_dim}D, but box_size is {box_dim}D.")

    # Convert named tuples to numpy arrays for vector operations
    pos1 = np.array(pos1, dtype=np.float64)
    pos2 = np.array(pos2, dtype=np.float64)
    box_size = np.array(box_size, dtype=np.float64)

    # Check for division by zero in the box size and replace with a large value if necessary
    box_size_safe = np.where(box_size == 0, np.inf, box_size)

    # Calculate the vector difference between the two positions
    d_pos = pos1 - pos2

    # Adjust the difference vector for periodic boundary conditions
    d_pos = d_pos - np.array(box_size) * np.round(d_pos / np.array(box_size_safe))

    # Calculate the Euclidean distance using the adjusted difference vector
    distance = np.linalg.norm(d_pos)
    displacement = Vector(float(d_pos[0]), float(d_pos[1]), float(d_pos[2]))

    return distance, displacement


@jit(nopython=True)
def minimum_image_distance_vectorized(
    pos1: npt.NDArray, pos2: npt.NDArray, box_size: Union[Tuple[float, float], Tuple[float, float, float]]
) -> (npt.NDArray, npt.NDArray):
    """
    Calculate the minimum distance between two sets of positions considering periodic boundary conditions.

    Parameters
    ----------
    pos1 : npt.NDArray
        Array of positions of the first set of atoms (N x 3).
    pos2 : npt.NDArray
        Array of positions of the second set of atoms (N x 3).
    box_size : Union[Tuple[float, float], Tuple[float, float, float]]
        Size of the box in the x, y and optionally z dimensions (box_width, box_height, box_depth).

    Returns
    -------
    Tuple[npt.NDArray, npt.NDArray]
        A tuple containing:
        - The minimum distances between the sets of positions as a numpy array.
        - The displacement vectors accounting for periodic boundary conditions as a numpy array (N x 3).
    """
    # Check if the dimensions of pos1/pos2 and box_size match
    pos_dim = pos1.shape[1]
    box_dim = len(box_size)

    if pos_dim != box_dim:
        raise ValueError(f"Dimension mismatch: positions are {pos_dim}D, but box_size is {box_dim}D.")

    # Calculate the vector difference between the two positions
    delta = pos2 - pos1

    # Convert the tuple to a numpy array for broadcasting
    box_size_arr = np.array(box_size)

    # Adjust the difference vector for periodic boundary conditions
    # This ensures that the atoms are considered within the bounds of the box
    delta -= np.round(delta / box_size_arr) * box_size_arr

    # Calculate the Euclidean distance using the adjusted difference vector
    # np.sum(delta**2, axis=1) computes the squared distances for each pair of points
    # np.sqrt(...) computes the Euclidean distance
    dist = np.sqrt(np.sum(delta**2, axis=1))

    return dist, delta


# def toggle_dimension(sheet_graph: nx.Graph):
#     """
#     Toggle the graph positions between 2D and 3D.
#
#     If the positions are in 2D (Position2D), they will be converted to 3D (Position3D) by adding a z-coordinate of 0.
#     If the positions are in 3D (Position3D), they will be converted to 2D (Position2D) by removing the z-coordinate.
#
#     Parameters
#     ----------
#     sheet_graph : nx.Graph
#         The graph containing the sheet structure to convert.
#     """
#     for node, pos in sheet_graph.nodes(data="position"):
#         if isinstance(pos, Position2D):
#             # Convert from 2D to 3D
#             sheet_graph.nodes[node]["position"] = Position3D(pos.x, pos.y, 0.0)
#         elif isinstance(pos, Position3D):
#             # Convert from 3D to 2D
#             sheet_graph.nodes[node]["position"] = Position2D(pos.x, pos.y)


def write_xyz(graph: nx.Graph, filename: str):
    """
    Write the atomic positions and elements to an XYZ file.

    Parameters
    ----------
    graph : nx.Graph
        The graph representing the atomic structure.
    filename : str
        The name of the XYZ file to write to.
    """

    with open(filename, "w") as file:
        num_atoms = len(graph.nodes)
        file.write(f"{num_atoms}\n")
        file.write("Atoms\n")

        for node in graph.nodes(data=True):
            label = node[1].get("label", node[1].get("element", "X"))  # Fallback to 'X' if no element or label is set
            pos = node[1]["position"]
            file.write(f"{label} {pos.x:.3f} {pos.y:.3f} {pos.z:.3f}\n")


def print_warning(message: str):
    # ANSI escape code for red color
    RED = "\033[91m"
    # ANSI escape code to reset color
    RESET = "\033[0m"
    print(f"{RED}{message}{RESET}")


def get_neighbors_within_distance(graph: nx.Graph, kdtree: KDTree, atom_id: int, distance: float) -> List[int]:
    """
    Find all neighbors within a given distance from the specified atom.

    Parameters
    ----------
    graph : nx.Graph
        The graph representing the graphene sheet.
    kdtree : KDTree
        The KDTree object used for efficient neighbor search.
    atom_id : int
        The ID of the atom (node) from which distances are measured.
    distance : float
        The maximum distance to search for neighbors.

    Returns
    -------
    List[int]
        A list of IDs representing the neighbors within the given distance from the source node.
    """
    atom_position = graph.nodes[atom_id]["position"]
    indices = kdtree.query_ball_point(atom_position, distance)
    return [list(graph.nodes)[index] for index in indices]


def get_neighbors_via_edges(graph: nx.Graph, atom_id: int, depth: int = 1, inclusive: bool = False) -> List[int]:
    """
    Get connected neighbors of a given atom up to a certain depth.

    Parameters
    ----------
    graph: nx.Graph
        The graph representing the graphene sheet.
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
        return list(graph.neighbors(atom_id))
    else:
        # Get neighbors up to the specified depth using shortest path lengths
        paths = nx.single_source_shortest_path_length(graph, atom_id, cutoff=depth)
        if inclusive:
            # Include all neighbors up to the specified depth
            return [node for node in paths.keys() if node != atom_id]  # Exclude the atom itself (depth 0)
        else:
            # Include only neighbors at the exact specified depth
            return [node for node, length in paths.items() if length == depth]


def get_neighbors_paths(graph: nx.Graph, atom_id: int, depth: int = 1) -> List[Tuple[int, int]]:
    """
    Get edges of paths to connected neighbors up to a certain depth.

    Parameters
    ----------
    graph : nx.Graph
        The graph representing the graphene sheet.
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
    paths = nx.single_source_shortest_path(graph, atom_id, cutoff=depth)
    edges = []
    for path in paths.values():
        edges.extend([(path[i], path[i + 1]) for i in range(len(path) - 1)])
    return edges


def get_shortest_path_length(graph: nx.Graph, source: int, target: int) -> float:
    """
    Get the shortest path length between two connected atoms based on bond lengths.

    Parameters
    ----------
    graph: nx.Graph
        The graph representing the graphene sheet.
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
    return nx.dijkstra_path_length(graph, source, target, weight="bond_length")


def get_shortest_path(graph: nx.Graph, source: int, target: int) -> List[int]:
    """
    Get the shortest path between two connected atoms based on bond lengths.

    Parameters
    ----------
    graph: nx.Graph
        The graph representing the graphene sheet.
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
    return nx.dijkstra_path(graph, source, target, weight="bond_length")


def plot_graphene_with_path(graph: nx.Graph, path: List[int], visualize_periodic_bonds: bool = True):
    """
    Plot the graphene structure with a highlighted path.

    This method plots the entire graphene structure and highlights a specific path
    between two nodes using a different color.

    Parameters
    ----------
    graph : nx.Graph
        The graph representing the graphene sheet.
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
    pos = nx.get_node_attributes(graph, "position")
    elements = nx.get_node_attributes(graph, "element")

    # Determine colors for nodes, considering nitrogen species if present
    colors = [
        NitrogenSpecies.get_color(elements[node], graph.nodes[node].get("nitrogen_species")) for node in graph.nodes()
    ]
    labels = {node: f"{elements[node]}{node}" for node in graph.nodes()}

    # Separate periodic edges and regular edges
    regular_edges = [(u, v) for u, v, d in graph.edges(data=True) if not d.get("periodic")]
    periodic_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get("periodic")]

    # Initialize plot
    plt.figure(figsize=(12, 12))

    # Draw the regular edges
    nx.draw(graph, pos, edgelist=regular_edges, node_color=colors, node_size=200, with_labels=False)

    # Draw periodic edges with dashed lines if visualize_periodic_bonds is True
    if visualize_periodic_bonds:
        nx.draw_networkx_edges(graph, pos, edgelist=periodic_edges, style="dashed", edge_color="gray")

    # Highlight the nodes and edges in the specified path
    path_edges = list(zip(path, path[1:]))
    nx.draw_networkx_nodes(graph, pos, nodelist=path, node_color="yellow", node_size=300)
    nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color="yellow", width=2)

    # Draw labels for nodes
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=10, font_color="cyan", font_weight="bold")

    plt.show()


def plot_graphene_with_depth_neighbors_based_on_bond_length(
    graph: nx.Graph, atom_id: int, max_distance: float, visualize_periodic_bonds: bool = True
):
    """
    Plot the graphene structure with neighbors highlighted based on bond length.

    This method plots the entire graphene structure and highlights nodes that are within
    a specified maximum distance from a given atom, using the bond length as the distance metric.

    Parameters
    ----------
    graph : nx.Graph
        The graph representing the graphene sheet.
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
    pos = nx.get_node_attributes(graph, "position")
    elements = nx.get_node_attributes(graph, "element")

    # Determine colors for nodes, considering nitrogen species if present
    colors = [
        NitrogenSpecies.get_color(elements[node], graph.nodes[node].get("nitrogen_species")) for node in graph.nodes()
    ]
    labels = {node: f"{elements[node]}{node}" for node in graph.nodes()}

    # Separate periodic edges and regular edges
    regular_edges = [(u, v) for u, v, d in graph.edges(data=True) if not d.get("periodic")]
    periodic_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get("periodic")]

    # Initialize plot
    plt.figure(figsize=(12, 12))

    # Draw the regular edges
    nx.draw(graph, pos, edgelist=regular_edges, node_color=colors, node_size=200, with_labels=False)

    # Draw periodic edges with dashed lines if visualize_periodic_bonds is True
    if visualize_periodic_bonds:
        nx.draw_networkx_edges(graph, pos, edgelist=periodic_edges, style="dashed", edge_color="gray")

    # Compute shortest path lengths from the specified atom using bond lengths
    paths = nx.single_source_dijkstra_path_length(graph, atom_id, cutoff=max_distance, weight="bond_length")

    # Identify neighbors within the specified maximum distance
    depth_neighbors = [node for node, length in paths.items() if length <= max_distance]
    path_edges = [(u, v) for u in depth_neighbors for v in graph.neighbors(u) if v in depth_neighbors]

    # Highlight the identified neighbors and their connecting edges
    nx.draw_networkx_nodes(graph, pos, nodelist=depth_neighbors, node_color="yellow", node_size=300)
    nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color="yellow", width=2)

    # Draw labels for nodes
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=10, font_color="cyan", font_weight="bold")

    # Show plot
    plt.show()


def plot_nodes_within_distance(
    graph: nx.Graph, nodes_within_distance: List[int], visualize_periodic_bonds: bool = True
):
    """
    Plot the graphene structure with neighbors highlighted based on distance.

    This method plots the entire graphene structure and highlights nodes that are within
    a specified maximum distance from a given atom, using the bond length as the distance metric.

    Parameters
    ----------
    graph : nx.Graph
        The graph representing the graphene sheet.
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
    pos = nx.get_node_attributes(graph, "position")
    elements = nx.get_node_attributes(graph, "element")

    # Determine colors for nodes, considering nitrogen species if present
    colors = [
        NitrogenSpecies.get_color(elements[node], graph.nodes[node].get("nitrogen_species")) for node in graph.nodes()
    ]
    labels = {node: f"{elements[node]}{node}" for node in graph.nodes()}

    # Separate periodic edges and regular edges
    regular_edges = [(u, v) for u, v, d in graph.edges(data=True) if not d.get("periodic")]
    periodic_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get("periodic")]

    # Initialize plot
    plt.figure(figsize=(12, 12))

    # Draw the regular edges
    nx.draw(graph, pos, edgelist=regular_edges, node_color=colors, node_size=200, with_labels=False)

    # Draw periodic edges with dashed lines if visualize_periodic_bonds is True
    if visualize_periodic_bonds:
        nx.draw_networkx_edges(graph, pos, edgelist=periodic_edges, style="dashed", edge_color="gray")

    # Compute edges within the specified distance
    path_edges = [(u, v) for u in nodes_within_distance for v in graph.neighbors(u) if v in nodes_within_distance]

    # Highlight the identified neighbors and their connecting edges
    nx.draw_networkx_nodes(graph, pos, nodelist=nodes_within_distance, node_color="yellow", node_size=300)
    nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color="yellow", width=2)

    # Draw labels for nodes
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=10, font_color="cyan", font_weight="bold")

    # Show plot
    plt.show()


# def _adjust_for_periodic_boundaries(
#         positions: Dict[int, Tuple[float, float]],
#         subgraph: nx.Graph,
#         reference_position: Tuple[float, float],
#         graphene_graph: GrapheneGraph,
# ) -> Dict[int, Tuple[float, float]]:
#     """
#     Adjust positions for periodic boundary conditions when a dopant spans periodic boundaries in the graphene
#     lattice.
#
#     This method ensures that the positions of atoms are correctly adjusted when a dopant affects nodes connected via
#     periodic boundaries. It identifies nodes connected via periodic boundaries and propagates these adjustments to
#     all connected nodes, ensuring the continuity and integrity of the graphene sheet structure. It modifies the
#     positions of atoms in the subgraph, considering the positions of nodes that were connected to the reference node
#     before its deletion.
#
#     Parameters
#     ----------
#     positions : dict
#         Dictionary of positions of atoms where keys are node IDs and values are (x, y) coordinates.
#     subgraph : nx.Graph
#         The subgraph containing the cycle of nodes to be adjusted.
#     reference_position : tuple
#         The position of the reference node before deletion, used to determine the direction of adjustment.
#     graphene_graph : GrapheneGraph
#         The GrapheneGraph object containing the graphene sheet and its properties.
#
#     Returns
#     -------
#     dict
#         Dictionary of adjusted positions where keys are node IDs and values are adjusted (x, y) coordinates.
#
#     Notes
#     -----
#     This method involves three main steps:
#     1. Identifying nodes that are connected via periodic boundaries.
#     2. Performing a depth-first search (DFS) to propagate boundary adjustments to all connected nodes.
#     3. Adjusting the positions of all identified nodes to account for the periodic boundaries.
#     """
#
#     # Copy the original positions to avoid modifying the input directly
#     adjusted_positions = positions.copy()
#     # Store each node together with the boundaries where they should be moved to for position optimization
#     nodes_with_boundaries = {}
#
#     # Step 1: Identify nodes that need to be adjusted and are connected via periodic boundaries
#     for edge in subgraph.edges(data=True):
#         # Check if the edge is periodic
#         if edge[2].get("periodic"):
#             node1, node2 = edge[0], edge[1]
#
#             # Ensure node1 is always the node with the smaller ID
#             if node1 > node2:
#                 node1, node2 = node2, node1
#
#             # Get positions of the nodes
#             pos1, pos2 = (adjusted_positions[node1], adjusted_positions[node2])
#             # Determine the boundary based on the reference position and positions of the nodes
#             boundary = determine_boundary(reference_position, pos1, pos2)
#
#             # Add the boundary adjustment to the appropriate node
#             if boundary in ["left", "bottom"]:
#                 nodes_with_boundaries.setdefault(node2, set()).add(boundary)
#             elif boundary in ["right", "top"]:
#                 nodes_with_boundaries.setdefault(node1, set()).add(boundary)
#
#     # Step 2: Find all the remaining nodes that need to be adjusted via a depth-first search
#     def dfs(node: int, visited: Set[int]):
#         """
#         Perform a depth-first search (DFS) to find and adjust all nodes connected via non-periodic edges.
#
#         Parameters
#         ----------
#         node : int
#             The current node to start the DFS from.
#         visited : set
#             A set to keep track of all visited nodes.
#
#         Notes
#         -----
#         The DFS will propagate boundary adjustments from nodes with periodic boundaries to all connected nodes
#         without periodic boundaries, ensuring proper adjustment of all related positions.
#         """
#         stack = [node]  # Initialize the stack with the starting node
#         while stack:
#             current_node = stack.pop()  # Get the last node added to the stack
#             if current_node not in visited:
#                 visited.add(current_node)  # Mark the current node as visited
#                 for neighbor in subgraph.neighbors(current_node):
#                     # Only proceed if the neighbor is not visited and the edge is not periodic
#                     if neighbor not in visited and not subgraph.edges[current_node, neighbor].get("periodic"):
#                         stack.append(neighbor)  # Add the neighbor to the stack for further exploration
#                         if neighbor not in nodes_with_boundaries:
#                             # Copy boundary adjustments from the current node to the neighbor
#                             nodes_with_boundaries[neighbor] = nodes_with_boundaries[current_node].copy()
#                         else:
#                             # Update boundary adjustments to ensure all necessary boundaries are included
#                             nodes_with_boundaries[current_node].update(nodes_with_boundaries[neighbor])
#                             nodes_with_boundaries[neighbor].update(nodes_with_boundaries[current_node])
#
#     # Initialize visited set to keep track of all nodes that have been visited during the DFS
#     visited = set()
#     # List of nodes that need boundary adjustments based on periodic boundaries
#     confining_nodes = list(nodes_with_boundaries.keys())
#     # Run DFS for each node that has boundary adjustments to propagate these adjustments to all connected nodes
#     for node in confining_nodes:
#         if node not in visited:
#             dfs(node, visited)
#
#     # Step 3: Adjust the positions of the nodes in nodes_with_boundaries
#     for node, boundaries in nodes_with_boundaries.items():
#         node_pos = np.array(adjusted_positions[node])
#
#         # Adjust positions based on identified boundaries
#         if "left" in boundaries:
#             node_pos[0] -= graphene_graph.actual_sheet_width + graphene_graph.bond_distance
#         elif "right" in boundaries:
#             node_pos[0] += graphene_graph.actual_sheet_width + graphene_graph.bond_distance
#         if "top" in boundaries:
#             node_pos[1] += graphene_graph.actual_sheet_height + graphene_graph.cc_y_distance
#         elif "bottom" in boundaries:
#             node_pos[1] -= graphene_graph.actual_sheet_height + graphene_graph.cc_y_distance
#
#         # Update the adjusted positions
#         adjusted_positions[node] = (float(node_pos[0]), float(node_pos[1]))
#
#     return adjusted_positions


def determine_boundary(
    reference_position: Tuple[float, float], pos1: Tuple[float, float], pos2: Tuple[float, float]
) -> str:
    """
    Determine the boundary direction for periodic continuation of doping.

    This method calculates which boundary (left, right, top, bottom) is crossed when inserting a doping structure
    with a periodic boundary condition between two positions (pos1 and pos2) relative to a given reference position
    (the atom id). The reference_position is used to determine which position (pos1 or pos2) is closer to it.
    The doping structure is built from this closer position, and thus, it calculates at which boundary or boundaries
    the doping structure should be continued. This information is used to shift the remaining doping atoms connected
    directly or indirectly via periodic boundary conditions to the correct position for optimization.

    Parameters
    ----------
    reference_position : tuple
        The reference position used to determine the boundary direction.
        Typically, this is the position of the reference node before deletion.
    pos1 : tuple
        The first position to compare.
    pos2 : tuple
        The second position to compare.

    Returns
    -------
    str
        The boundary direction ('left', 'right', 'top', or 'bottom') indicating where the doping structure should
        be continued.
    """
    # Determine which position is left and which is right based on the x-coordinate
    left, right = (pos1, pos2) if pos2[0] > pos1[0] else (pos2, pos1)
    # Determine which position is down and which is up based on the y-coordinate
    down, up = (pos1, pos2) if pos2[1] > pos1[1] else (pos2, pos1)

    # Calculate the difference in distance from the reference position to the left and right positions
    x_diff = abs(reference_position[0] - left[0]) - abs(reference_position[0] - right[0])
    # Calculate the difference in distance from the reference position to the down and up positions
    y_diff = abs(reference_position[1] - down[1]) - abs(reference_position[1] - up[1])

    # Determine the primary boundary direction based on the larger of the two differences
    if abs(x_diff) > abs(y_diff):
        # If the x-difference is larger, the boundary is horizontal (left or right)
        return "left" if x_diff < 0 else "right"
    else:
        # If the y-difference is larger, the boundary is vertical (top or bottom)
        return "bottom" if y_diff < 0 else "top"
