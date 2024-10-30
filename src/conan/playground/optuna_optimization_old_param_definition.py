import json
import os
import random
from typing import Optional, Tuple

import networkx as nx
import numpy as np
import optuna
import pandas as pd
from networkx.utils import pairwise
from optuna.visualization import (
    plot_contour,
    plot_edf,
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_slice,
)
from scipy.optimize import minimize

from conan.playground.doping_experiment_deprecated import GrapheneSheet, NitrogenSpecies
from conan.playground.utils import create_position, minimum_image_distance, minimum_image_distance_vectorized, write_xyz


def calculate_minimal_total_energy(
    graphene: GrapheneSheet, include_outer_angles: bool = False
) -> (Tuple)[float, Optional[GrapheneSheet]]:
    """
    Calculate the total energy of the graphene sheet, considering bond and angle energies.

    Parameters
    ----------
    graphene : GrapheneSheet
        The graphene sheet to calculate the total energy.
    include_outer_angles : bool (optional)
        Whether to include angles outside the cycles in the calculation.

    Returns
    -------
    float
        The total energy of the graphene sheet.
    GrapheneSheet
        The graphene sheet object with optimized positions.
    """
    # Get all doping structures except graphitic nitrogen (graphitic nitrogen does not affect the structure)
    all_structures = [
        structure
        for structure in graphene.doping_structures.structures
        if structure.species != NitrogenSpecies.GRAPHITIC
    ]

    # Get the initial positions of atoms
    positions = {node: graphene.graph.nodes[node]["position"] for node in graphene.graph.nodes}
    # Flatten the positions into a 1D array for optimization
    x0 = np.array([coord for node in graphene.graph.nodes for coord in [positions[node].x, positions[node].y]])
    # Define the box size for minimum image distance calculation
    box_size = (
        graphene.actual_sheet_width + graphene.c_c_bond_length,
        graphene.actual_sheet_height + graphene.cc_y_distance,
    )

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
            properties = graphene.species_properties[structure.species]
            target_bond_lengths = properties.target_bond_lengths_cycle
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
                (list(graphene.graph.nodes).index(node_i), list(graphene.graph.nodes).index(node_j))
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
        energy += 0.5 * graphene.k_inner_bond * np.sum((current_lengths - target_lengths) ** 2)

        # Update bond lengths in the graph
        edge_updates = {
            (node_i, node_j): {"bond_length": current_lengths[idx]}
            for idx, (node_i, node_j) in enumerate(all_edges_in_order)
        }
        nx.set_edge_attributes(graphene.graph, edge_updates)
        cycle_edges.update((min(node_i, node_j), max(node_i, node_j)) for node_i, node_j in all_edges_in_order)

        # Handle non-cycle edges in a vectorized manner
        non_cycle_edges = [
            (node_i, node_j)
            for node_i, node_j, data in graphene.graph.edges(data=True)
            if (min(node_i, node_j), max(node_i, node_j)) not in cycle_edges
        ]
        if non_cycle_edges:
            # Convert non-cycle edge node pairs to a numpy array of indices
            node_indices = np.array(
                [
                    (list(graphene.graph.nodes).index(node_i), list(graphene.graph.nodes).index(node_j))
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
            energy += 0.5 * graphene.k_outer_bond * np.sum((current_lengths - target_lengths) ** 2)

            # Prepare bond length updates for non-cycle edges
            edge_updates = {
                (node_i, node_j): {"bond_length": current_lengths[idx]}
                for idx, (node_i, node_j) in enumerate(non_cycle_edges)
            }
            # Update the bond lengths in the graph for non-cycle edges
            nx.set_edge_attributes(graphene.graph, edge_updates)

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
        counted_angles = set()

        # Iterate over all doping structures to gather triplets and target angles
        for structure in all_structures:
            properties = graphene.species_properties[structure.species]
            target_angles = properties.target_angles_cycle
            ordered_cycle = structure.cycle

            # Extend the cycle to account for the closed loop by adding the first two nodes at the end
            extended_cycle = ordered_cycle + [ordered_cycle[0], ordered_cycle[1]]

            # Collect node triplets (i, j, k) for angle energy calculations
            triplets = [
                (
                    list(graphene.graph.nodes).index(i),
                    list(graphene.graph.nodes).index(j),
                    list(graphene.graph.nodes).index(k),
                )
                for i, j, k in zip(extended_cycle, extended_cycle[1:], extended_cycle[2:])
            ]
            all_triplets.extend(triplets)
            all_target_angles.extend(target_angles)

            if include_outer_angles:
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
        energy += 0.5 * graphene.k_inner_angle * np.sum((theta - target_angles) ** 2)

        if include_outer_angles:
            # Calculate angle energy for angles outside the cycles
            for node in graphene.graph.nodes:
                neighbors = list(graphene.graph.neighbors(node))
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
                            x[2 * list(graphene.graph.nodes).index(node)],
                            x[2 * list(graphene.graph.nodes).index(node) + 1],
                        )
                        x_i, y_i = (
                            x[2 * list(graphene.graph.nodes).index(ni)],
                            x[2 * list(graphene.graph.nodes).index(ni) + 1],
                        )
                        x_j, y_j = (
                            x[2 * list(graphene.graph.nodes).index(nj)],
                            x[2 * list(graphene.graph.nodes).index(nj) + 1],
                        )
                        pos_node = create_position(x_node, y_node)
                        pos_i = create_position(x_i, y_i)
                        pos_j = create_position(x_j, y_j)
                        _, v1 = minimum_image_distance(pos_i, pos_node, box_size)
                        _, v2 = minimum_image_distance(pos_j, pos_node, box_size)
                        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
                        energy += 0.5 * graphene.k_outer_angle * ((theta - np.radians(graphene.c_c_bond_angle)) ** 2)

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

    # Reshape the optimized positions back to the 2D array format
    optimized_positions = result.x.reshape(-1, 2)

    # Update the positions of atoms in the graph with the optimized positions using NetworkX set_node_attributes
    position_dict = {
        node: create_position(optimized_positions[idx][0], optimized_positions[idx][1])
        for idx, node in enumerate(graphene.graph.nodes)
    }
    nx.set_node_attributes(graphene.graph, position_dict, "position")

    return result.fun, graphene


def calculate_bond_angle_accuracy(graphene: GrapheneSheet) -> Tuple[float, float]:
    all_cycles = []
    species_for_cycles = []

    for structure in graphene.doping_structures.structures:
        if structure.species != NitrogenSpecies.GRAPHITIC:  # Skip GRAPHITIC species
            all_cycles.append(structure.cycle)
            species_for_cycles.append(structure.species)

    bond_accuracy = 0.0
    angle_accuracy = 0.0

    for idx, ordered_cycle in enumerate(all_cycles):
        species = species_for_cycles[idx]
        properties = graphene.species_properties[species]
        target_bond_lengths = properties.target_bond_lengths_cycle
        target_angles = properties.target_angles_cycle
        # subgraph = graphene.graph.subgraph(ordered_cycle).copy()

        for i in range(len(ordered_cycle)):
            node_i = ordered_cycle[i]
            node_j = ordered_cycle[(i + 1) % len(ordered_cycle)]
            pos_i = graphene.graph.nodes[node_i]["position"]
            pos_j = graphene.graph.nodes[node_j]["position"]
            current_length, _ = minimum_image_distance(
                pos_i,
                pos_j,
                (
                    graphene.actual_sheet_width + graphene.c_c_bond_length,
                    graphene.actual_sheet_height + graphene.cc_y_distance,
                ),
            )
            target_length = target_bond_lengths[i]
            bond_accuracy += abs(current_length - target_length)

        for (i, j, k), angle in zip(zip(ordered_cycle, ordered_cycle[1:], ordered_cycle[2:]), target_angles):
            pos_i = graphene.graph.nodes[i]["position"]
            pos_j = graphene.graph.nodes[j]["position"]
            pos_k = graphene.graph.nodes[k]["position"]
            _, v1 = minimum_image_distance(
                pos_i,
                pos_j,
                (
                    graphene.actual_sheet_width + graphene.c_c_bond_length,
                    graphene.actual_sheet_height + graphene.cc_y_distance,
                ),
            )
            _, v2 = minimum_image_distance(
                pos_k,
                pos_j,
                (
                    graphene.actual_sheet_width + graphene.c_c_bond_length,
                    graphene.actual_sheet_height + graphene.cc_y_distance,
                ),
            )
            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
            angle_accuracy += abs(np.degrees(theta) - angle)

    bond_accuracy /= len(all_cycles)
    angle_accuracy /= len(all_cycles)

    return bond_accuracy, angle_accuracy


def total_energy_correctness_check(trial):
    random.seed(0)

    # Create Graphene instance and set k_inner_bond and k_outer_bond
    graphene = GrapheneSheet(bond_length=1.42, sheet_size=(20, 20))
    graphene.k_inner_bond = 23.359776202184758
    graphene.k_outer_bond = 0.014112166829508662
    graphene.k_inner_angle = 79.55711394238168
    graphene.k_outer_angle = 0.019431203948375452

    # Add nitrogen doping to the graphene sheet
    graphene.add_nitrogen_doping(total_percentage=15, adjust_positions=False)

    # Calculate the total energy of the graphene sheet
    total_energy, graphene = calculate_minimal_total_energy(graphene, include_outer_angles=True)

    # Write the optimized structure to an XYZ file for comparison to validate the optimization
    write_xyz(
        graphene.graph,
        f"test_all_structures_k_inner_bond_{graphene.k_inner_bond}_k_outer_bond_{graphene.k_outer_bond}_"
        f"k_inner_angle_{graphene.k_inner_angle}.xyz",
    )

    # Return the total energy as the objective value
    return total_energy


def objective_total_energy_pyridinic_4(trial):
    random.seed(0)

    # Sample k_inner and k_outer
    k_inner_bond = trial.suggest_float("k_inner_bond", 1.0, 1000.0, log=True)
    k_inner_angle = trial.suggest_float("k_inner_angle", 1.0, 1000.0, log=True)
    k_outer_bond = trial.suggest_float("k_outer_bond", 0.01, 10.0, log=True)
    # k_outer_angle = trial.suggest_float("k_outer_angle", 0.01, 10.0, log=True)

    # Create Graphene instance and set k_inner_bond and k_outer_bond
    graphene = GrapheneSheet(bond_length=1.42, sheet_size=(20, 20))
    graphene.k_inner_bond = k_inner_bond
    graphene.k_outer_bond = k_outer_bond
    graphene.k_inner_angle = k_inner_angle
    # graphene.k_outer_angle = k_outer_angle

    # Add nitrogen doping to the graphene sheet
    graphene.add_nitrogen_doping(percentages={NitrogenSpecies.PYRIDINIC_4: 3}, adjust_positions=False)

    # Calculate the total energy of the graphene sheet
    total_energy, _ = calculate_minimal_total_energy(graphene)

    # Return the total energy as the objective value
    return total_energy


def objective_total_energy_all_structures(trial):
    random.seed(0)

    # Sample k_inner and k_outer
    k_inner_bond = trial.suggest_float("k_inner_bond", 0.01, 100.0, log=True)
    k_inner_angle = trial.suggest_float("k_inner_angle", 0.01, 100.0, log=True)
    k_outer_bond = trial.suggest_float("k_outer_bond", 0.01, 100.0, log=True)
    # k_outer_angle = trial.suggest_float("k_outer_angle", 0.01, 100.0, log=True)

    # Create Graphene instance and set k_inner_bond and k_outer_bond
    graphene = GrapheneSheet(bond_length=1.42, sheet_size=(20, 20))
    graphene.k_inner_bond = k_inner_bond
    graphene.k_outer_bond = k_outer_bond
    graphene.k_inner_angle = k_inner_angle
    # graphene.k_outer_angle = k_outer_angle

    # Add nitrogen doping to the graphene sheet
    graphene.add_nitrogen_doping(total_percentage=15, adjust_positions=False)

    # Calculate the total energy of the graphene sheet
    total_energy, _ = calculate_minimal_total_energy(graphene)

    # Return the total energy as the objective value
    return total_energy


def objective_combined_pyridinic_4(trial):
    random.seed(0)

    # Sample k_inner and k_outer
    k_inner_bond = trial.suggest_float("k_inner_bond", 0.01, 100.0, log=True)
    k_inner_angle = trial.suggest_float("k_inner_angle", 0.01, 100.0, log=True)
    k_outer_bond = trial.suggest_float("k_outer_bond", 0.01, 100.0, log=True)
    # k_outer_angle = trial.suggest_float("k_outer_angle", 0.01, 100.0, log=True)

    # Create Graphene instance and set k_inner_bond and k_outer_bond
    graphene = GrapheneSheet(bond_length=1.42, sheet_size=(20, 20))
    graphene.k_inner_bond = k_inner_bond
    graphene.k_outer_bond = k_outer_bond
    graphene.k_inner_angle = k_inner_angle
    # graphene.k_outer_angle = k_outer_angle

    # Add nitrogen doping to the graphene sheet
    graphene.add_nitrogen_doping(percentages={NitrogenSpecies.PYRIDINIC_4: 3}, adjust_positions=False)

    # Calculate the total energy of the graphene sheet
    total_energy, updated_graphene = calculate_minimal_total_energy(graphene)

    # Calculate bond and angle accuracy within cycles (additional objectives can be added)
    bond_accuracy, angle_accuracy = calculate_bond_angle_accuracy(updated_graphene)

    # Combine objectives
    objective_value = total_energy + bond_accuracy + angle_accuracy

    return objective_value


def objective_combined_all_structures(trial):
    random.seed(0)

    # Sample k_inner and k_outer
    k_inner_bond = trial.suggest_float("k_inner_bond", 0.01, 100.0, log=True)
    k_inner_angle = trial.suggest_float("k_inner_angle", 0.01, 100.0, log=True)
    k_outer_bond = trial.suggest_float("k_outer_bond", 0.01, 100.0, log=True)
    # k_outer_angle = trial.suggest_float("k_outer_angle", 0.01, 100.0, log=True)

    # Create Graphene instance and set k_inner_bond and k_outer_bond
    graphene = GrapheneSheet(bond_length=1.42, sheet_size=(20, 20))
    graphene.k_inner_bond = k_inner_bond
    graphene.k_outer_bond = k_outer_bond
    graphene.k_inner_angle = k_inner_angle
    # graphene.k_outer_angle = k_outer_angle

    # Add nitrogen doping to the graphene sheet
    graphene.add_nitrogen_doping(total_percentage=15, adjust_positions=False)

    # Calculate the total energy of the graphene sheet
    total_energy, updated_graphene = calculate_minimal_total_energy(graphene)

    # Calculate bond and angle accuracy within cycles (additional objectives can be added)
    bond_accuracy, angle_accuracy = calculate_bond_angle_accuracy(updated_graphene)

    # Combine objectives
    objective_value = total_energy + bond_accuracy + angle_accuracy

    return objective_value


def objective_bond_angle_accuracy_pyridinic_4(trial):
    random.seed(0)

    # Sample k_inner and k_outer
    k_inner_bond = trial.suggest_float("k_inner_bond", 0.01, 100.0, log=True)
    k_inner_angle = trial.suggest_float("k_inner_angle", 0.01, 100.0, log=True)
    k_outer_bond = trial.suggest_float("k_outer_bond", 0.01, 100.0, log=True)
    # k_outer_angle = trial.suggest_float("k_outer_angle", 0.01, 100.0, log=True)

    # Create Graphene instance and set k_inner_bond and k_outer_bond
    graphene = GrapheneSheet(bond_length=1.42, sheet_size=(20, 20))
    graphene.k_inner_bond = k_inner_bond
    graphene.k_outer_bond = k_outer_bond
    graphene.k_inner_angle = k_inner_angle
    # graphene.k_outer_angle = k_outer_angle

    # Add nitrogen doping to the graphene sheet
    graphene.add_nitrogen_doping(percentages={NitrogenSpecies.PYRIDINIC_4: 3})

    # Calculate bond and angle accuracy within cycles
    bond_accuracy, angle_accuracy = calculate_bond_angle_accuracy(graphene)

    # Combine objectives
    objective_value = bond_accuracy + angle_accuracy

    return objective_value


def objective_bond_angle_accuracy_all_structures(trial):
    random.seed(0)

    # Sample k_inner and k_outer
    k_inner_bond = trial.suggest_float("k_inner_bond", 0.01, 100.0, log=True)
    k_inner_angle = trial.suggest_float("k_inner_angle", 0.01, 100.0, log=True)
    k_outer_bond = trial.suggest_float("k_outer_bond", 0.01, 100.0, log=True)
    # k_outer_angle = trial.suggest_float("k_outer_angle", 0.01, 100.0, log=True)

    # Create Graphene instance and set k_inner_bond and k_outer_bond
    graphene = GrapheneSheet(bond_length=1.42, sheet_size=(20, 20))
    graphene.k_inner_bond = k_inner_bond
    graphene.k_outer_bond = k_outer_bond
    graphene.k_inner_angle = k_inner_angle
    # graphene.k_outer_angle = k_outer_angle

    # Add nitrogen doping to the graphene sheet
    graphene.add_nitrogen_doping(total_percentage=15)

    # Calculate bond and angle accuracy within cycles
    bond_accuracy, angle_accuracy = calculate_bond_angle_accuracy(graphene)

    # Combine objectives
    objective_value = bond_accuracy + angle_accuracy

    return objective_value


def objective_total_energy_pyridinic_4_with_outer_angles(trial):
    random.seed(0)

    k_inner_bond = trial.suggest_float("k_inner_bond", 0.01, 100.0, log=True)
    k_inner_angle = trial.suggest_float("k_inner_angle", 0.01, 100.0, log=True)
    k_outer_bond = trial.suggest_float("k_outer_bond", 0.01, 100.0, log=True)
    k_outer_angle = trial.suggest_float("k_outer_angle", 0.01, 100.0, log=True)

    graphene = GrapheneSheet(bond_length=1.42, sheet_size=(20, 20))
    graphene.k_inner_bond = k_inner_bond
    graphene.k_outer_bond = k_outer_bond
    graphene.k_inner_angle = k_inner_angle
    graphene.k_outer_angle = k_outer_angle

    graphene.add_nitrogen_doping(percentages={NitrogenSpecies.PYRIDINIC_4: 3}, adjust_positions=False)

    total_energy, _ = calculate_minimal_total_energy(graphene, include_outer_angles=True)
    return total_energy


def objective_total_energy_all_structures_with_outer_angles(trial):
    random.seed(0)

    k_inner_bond = trial.suggest_float("k_inner_bond", 0.01, 100.0, log=True)
    k_inner_angle = trial.suggest_float("k_inner_angle", 0.01, 100.0, log=True)
    k_outer_bond = trial.suggest_float("k_outer_bond", 0.01, 100.0, log=True)
    k_outer_angle = trial.suggest_float("k_outer_angle", 0.01, 100.0, log=True)

    graphene = GrapheneSheet(bond_length=1.42, sheet_size=(20, 20))
    graphene.k_inner_bond = k_inner_bond
    graphene.k_outer_bond = k_outer_bond
    graphene.k_inner_angle = k_inner_angle
    graphene.k_outer_angle = k_outer_angle

    graphene.add_nitrogen_doping(total_percentage=15, adjust_positions=False)

    total_energy, _ = calculate_minimal_total_energy(graphene, include_outer_angles=True)
    return total_energy


def objective_combined_pyridinic_4_with_outer_angles(trial):
    random.seed(0)

    k_inner_bond = trial.suggest_float("k_inner_bond", 0.01, 100.0, log=True)
    k_inner_angle = trial.suggest_float("k_inner_angle", 0.01, 100.0, log=True)
    k_outer_bond = trial.suggest_float("k_outer_bond", 0.01, 100.0, log=True)
    k_outer_angle = trial.suggest_float("k_outer_angle", 0.01, 100.0, log=True)

    graphene = GrapheneSheet(bond_length=1.42, sheet_size=(20, 20))
    graphene.k_inner_bond = k_inner_bond
    graphene.k_outer_bond = k_outer_bond
    graphene.k_inner_angle = k_inner_angle
    graphene.k_outer_angle = k_outer_angle

    graphene.add_nitrogen_doping(percentages={NitrogenSpecies.PYRIDINIC_4: 3}, adjust_positions=False)

    total_energy, updated_graphene = calculate_minimal_total_energy(graphene, include_outer_angles=True)
    bond_accuracy, angle_accuracy = calculate_bond_angle_accuracy(updated_graphene)
    objective_value = total_energy + bond_accuracy + angle_accuracy

    return objective_value


def objective_combined_all_structures_with_outer_angles(trial):
    random.seed(0)

    k_inner_bond = trial.suggest_float("k_inner_bond", 0.01, 100.0, log=True)
    k_inner_angle = trial.suggest_float("k_inner_angle", 0.01, 100.0, log=True)
    k_outer_bond = trial.suggest_float("k_outer_bond", 0.01, 100.0, log=True)
    k_outer_angle = trial.suggest_float("k_outer_angle", 0.01, 100.0, log=True)

    graphene = GrapheneSheet(bond_length=1.42, sheet_size=(20, 20))
    graphene.k_inner_bond = k_inner_bond
    graphene.k_outer_bond = k_outer_bond
    graphene.k_inner_angle = k_inner_angle
    graphene.k_outer_angle = k_outer_angle

    graphene.add_nitrogen_doping(total_percentage=15, adjust_positions=False)

    total_energy, updated_graphene = calculate_minimal_total_energy(graphene, include_outer_angles=True)
    bond_accuracy, angle_accuracy = calculate_bond_angle_accuracy(updated_graphene)
    objective_value = total_energy + bond_accuracy + angle_accuracy

    return objective_value


def save_study_results(study, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df = study.trials_dataframe()
    df = df.map(lambda x: str(x) if isinstance(x, (pd.Timestamp, pd.Timedelta)) else x)
    with open(filename, "w") as f:
        json.dump(df.to_dict(), f, indent=4)


def save_best_trial(study, filepath):
    best_trial = study.best_trial
    best_params = best_trial.params
    best_value = best_trial.value
    best_trial_number = best_trial.number

    best_trial_data = {"params": best_params, "value": best_value, "trial_number": best_trial_number}

    with open(filepath, "w") as f:
        json.dump(best_trial_data, f, indent=4)


def save_study_visualizations(study, results_dir):
    visualizations_dir = f"{results_dir}/visualizations"
    os.makedirs(visualizations_dir, exist_ok=True)

    optimization_history = plot_optimization_history(study)
    param_importances = plot_param_importances(study)
    parallel_coordinate = plot_parallel_coordinate(study)
    slice_plot = plot_slice(study)
    contour_plot = plot_contour(study)
    edf_plot = plot_edf(study)

    optimization_history.write_image(f"{visualizations_dir}/optimization_history.png")
    param_importances.write_image(f"{visualizations_dir}/param_importances.png")
    parallel_coordinate.write_image(f"{visualizations_dir}/parallel_coordinate.png")
    slice_plot.write_image(f"{visualizations_dir}/slice_plot.png")
    contour_plot.write_image(f"{visualizations_dir}/contour_plot.png")
    edf_plot.write_image(f"{visualizations_dir}/edf_plot.png")


# Conducting and saving multiple studies
def conduct_study(objective_function, study_name, n_trials=100):
    study = optuna.create_study(direction="minimize")
    study.optimize(objective_function, n_trials=n_trials)

    results_dir = f"optuna_results/{study_name}"
    os.makedirs(results_dir, exist_ok=True)

    save_study_results(study, f"{results_dir}/results.json")
    save_best_trial(study, f"{results_dir}/best_trial.json")
    save_study_visualizations(study, results_dir)
    print(f"Best trial for {study_name}:")
    print(study.best_trial)


# Example usage
if __name__ == "__main__":
    # random.seed(0)

    os.makedirs("optuna_results", exist_ok=True)

    # # Conduct study for total energy with Pyridinic_4
    # conduct_study(objective_total_energy_pyridinic_4, "total_energy_pyridinic_4")
    #
    # # Conduct study for total energy with all structures
    # conduct_study(objective_total_energy_all_structures, "total_energy_all_structures")
    #
    # # Conduct study for combined objective with Pyridinic_4
    # conduct_study(objective_combined_pyridinic_4, "combined_pyridinic_4")
    #
    # # Conduct study for combined objective with all structures
    # conduct_study(objective_combined_all_structures, "combined_all_structures")
    #
    # # Conduct study for total energy with Pyridinic_4 including outer angles
    # conduct_study(
    #     objective_total_energy_pyridinic_4_with_outer_angles, "total_energy_pyridinic_4_including_outer_angles"
    # )
    #
    # # Conduct study for total energy with all structures including outer angles
    # conduct_study(
    #     objective_total_energy_all_structures_with_outer_angles, "total_energy_all_structures_including_outer_angles"
    # )
    #
    # # Conduct study for combined objective with Pyridinic_4 including outer angles
    # conduct_study(objective_combined_pyridinic_4_with_outer_angles, "combined_pyridinic_4_including_outer_angles")
    #
    # # Conduct study for combined objective with all structures including outer angles
    # conduct_study(objective_combined_all_structures_with_outer_angles,
    # "combined_all_structures_including_outer_angles")

    # # Conduct study for combined objective with Pyridinic_4
    # conduct_study(objective_combined_pyridinic_4, "combined_pyridinic_4_test", n_trials=2)

    # # Validate correctness of the optimization objectives
    # total_energy_correctness_check(1)

    # Conduct study for bond and angle accuracy with Pyridinic_4
    conduct_study(objective_bond_angle_accuracy_pyridinic_4, "bond_angle_accuracy_pyridinic_4", n_trials=100)

    # Conduct study for bond and angle accuracy with all structures
    conduct_study(objective_bond_angle_accuracy_all_structures, "bond_angle_accuracy_all_structures", n_trials=100)
