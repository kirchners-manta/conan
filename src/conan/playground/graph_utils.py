from typing import Dict, Set, Tuple

import networkx as nx
import numpy as np

from conan.playground.doping_experiment import GrapheneGraph


def _adjust_for_periodic_boundaries(
    positions: Dict[int, Tuple[float, float]],
    subgraph: nx.Graph,
    reference_position: Tuple[float, float],
    graphene_graph: GrapheneGraph,
) -> Dict[int, Tuple[float, float]]:
    """
    Adjust positions for periodic boundary conditions when a dopant spans periodic boundaries in the graphene
    lattice.

    This method ensures that the positions of atoms are correctly adjusted when a dopant affects nodes connected via
    periodic boundaries. It identifies nodes connected via periodic boundaries and propagates these adjustments to
    all connected nodes, ensuring the continuity and integrity of the graphene sheet structure. It modifies the
    positions of atoms in the subgraph, considering the positions of nodes that were connected to the reference node
    before its deletion.

    Parameters
    ----------
    positions : dict
        Dictionary of positions of atoms where keys are node IDs and values are (x, y) coordinates.
    subgraph : nx.Graph
        The subgraph containing the cycle of nodes to be adjusted.
    reference_position : tuple
        The position of the reference node before deletion, used to determine the direction of adjustment.
    graphene_graph : GrapheneGraph
        The GrapheneGraph object containing the graphene sheet and its properties.

    Returns
    -------
    dict
        Dictionary of adjusted positions where keys are node IDs and values are adjusted (x, y) coordinates.

    Notes
    -----
    This method involves three main steps:
    1. Identifying nodes that are connected via periodic boundaries.
    2. Performing a depth-first search (DFS) to propagate boundary adjustments to all connected nodes.
    3. Adjusting the positions of all identified nodes to account for the periodic boundaries.
    """

    # Copy the original positions to avoid modifying the input directly
    adjusted_positions = positions.copy()
    # Store each node together with the boundaries where they should be moved to for position optimization
    nodes_with_boundaries = {}

    # Step 1: Identify nodes that need to be adjusted and are connected via periodic boundaries
    for edge in subgraph.edges(data=True):
        # Check if the edge is periodic
        if edge[2].get("periodic"):
            node1, node2 = edge[0], edge[1]

            # Ensure node1 is always the node with the smaller ID
            if node1 > node2:
                node1, node2 = node2, node1

            # Get positions of the nodes
            pos1, pos2 = (adjusted_positions[node1], adjusted_positions[node2])
            # Determine the boundary based on the reference position and positions of the nodes
            boundary = determine_boundary(reference_position, pos1, pos2)

            # Add the boundary adjustment to the appropriate node
            if boundary in ["left", "bottom"]:
                nodes_with_boundaries.setdefault(node2, set()).add(boundary)
            elif boundary in ["right", "top"]:
                nodes_with_boundaries.setdefault(node1, set()).add(boundary)

    # Step 2: Find all the remaining nodes that need to be adjusted via a depth-first search
    def dfs(node: int, visited: Set[int]):
        """
        Perform a depth-first search (DFS) to find and adjust all nodes connected via non-periodic edges.

        Parameters
        ----------
        node : int
            The current node to start the DFS from.
        visited : set
            A set to keep track of all visited nodes.

        Notes
        -----
        The DFS will propagate boundary adjustments from nodes with periodic boundaries to all connected nodes
        without periodic boundaries, ensuring proper adjustment of all related positions.
        """
        stack = [node]  # Initialize the stack with the starting node
        while stack:
            current_node = stack.pop()  # Get the last node added to the stack
            if current_node not in visited:
                visited.add(current_node)  # Mark the current node as visited
                for neighbor in subgraph.neighbors(current_node):
                    # Only proceed if the neighbor is not visited and the edge is not periodic
                    if neighbor not in visited and not subgraph.edges[current_node, neighbor].get("periodic"):
                        stack.append(neighbor)  # Add the neighbor to the stack for further exploration
                        if neighbor not in nodes_with_boundaries:
                            # Copy boundary adjustments from the current node to the neighbor
                            nodes_with_boundaries[neighbor] = nodes_with_boundaries[current_node].copy()
                        else:
                            # Update boundary adjustments to ensure all necessary boundaries are included
                            nodes_with_boundaries[current_node].update(nodes_with_boundaries[neighbor])
                            nodes_with_boundaries[neighbor].update(nodes_with_boundaries[current_node])

    # Initialize visited set to keep track of all nodes that have been visited during the DFS
    visited = set()
    # List of nodes that need boundary adjustments based on periodic boundaries
    confining_nodes = list(nodes_with_boundaries.keys())
    # Run DFS for each node that has boundary adjustments to propagate these adjustments to all connected nodes
    for node in confining_nodes:
        if node not in visited:
            dfs(node, visited)

    # Step 3: Adjust the positions of the nodes in nodes_with_boundaries
    for node, boundaries in nodes_with_boundaries.items():
        node_pos = np.array(adjusted_positions[node])

        # Adjust positions based on identified boundaries
        if "left" in boundaries:
            node_pos[0] -= graphene_graph.actual_sheet_width + graphene_graph.bond_distance
        elif "right" in boundaries:
            node_pos[0] += graphene_graph.actual_sheet_width + graphene_graph.bond_distance
        if "top" in boundaries:
            node_pos[1] += graphene_graph.actual_sheet_height + graphene_graph.cc_y_distance
        elif "bottom" in boundaries:
            node_pos[1] -= graphene_graph.actual_sheet_height + graphene_graph.cc_y_distance

        # Update the adjusted positions
        adjusted_positions[node] = (float(node_pos[0]), float(node_pos[1]))

    return adjusted_positions


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
