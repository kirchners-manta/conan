import copy
import random
from itertools import pairwise

import numpy as np
import optuna
from tqdm import tqdm

from conan.playground.doping_experiment import GrapheneSheet
from conan.playground.generate_doped_graphene_sheets import create_graphene_sheets
from conan.playground.graph_utils import NitrogenSpecies, minimum_image_distance_vectorized


# Function to calculate total error
def calculate_total_error(graphene_sheet: GrapheneSheet) -> float:
    """
    Calculate the total error for the graphene sheet, considering both bond lengths and angles,
    and taking into account the target values for inner and outer bonds and angles.

    Parameters
    ----------
    graphene_sheet : GrapheneSheet
        The graphene sheet object with positions adjusted.

    Returns
    -------
    float
        The combined total error.
    """
    # Get all nodes and their positions
    positions = {node: graphene_sheet.graph.nodes[node]["position"] for node in graphene_sheet.graph.nodes()}

    box_size = (
        graphene_sheet.actual_sheet_width + graphene_sheet.c_c_bond_distance,
        graphene_sheet.actual_sheet_height + graphene_sheet.cc_y_distance,
    )

    # Dictionaries to store bond and angle properties
    bond_target_lengths = {}
    angle_target_angles = {}
    inner_bond_set = set()
    inner_angle_set = set()

    # Collect inner bonds and angles from doping structures
    all_structures = [
        structure
        for structure in graphene_sheet.doping_handler.doping_structures.structures
        if structure.species != NitrogenSpecies.GRAPHITIC
    ]

    for structure in all_structures:
        properties = graphene_sheet.doping_handler.species_properties[structure.species]

        cycle_atoms = structure.cycle
        neighbor_atoms = structure.neighboring_atoms

        neighbor_atom_indices = {node: idx for idx, node in enumerate(neighbor_atoms)}

        # Bonds within the cycle
        cycle_edges = list(pairwise(cycle_atoms + [cycle_atoms[0]]))
        if structure.species == NitrogenSpecies.PYRIDINIC_1 and structure.additional_edge:
            cycle_edges.append(structure.additional_edge)

        for idx, (node_i, node_j) in enumerate(cycle_edges):
            bond = (min(node_i, node_j), max(node_i, node_j))
            inner_bond_set.add(bond)
            bond_target_lengths[bond] = properties.target_bond_lengths_cycle[idx]

        # Bonds between cycle atoms and their neighbors
        for idx, node_i in enumerate(cycle_atoms):
            neighbors = [n for n in graphene_sheet.graph.neighbors(node_i) if n not in cycle_atoms]
            for neighbor in neighbors:
                bond = (min(node_i, neighbor), max(node_i, neighbor))
                inner_bond_set.add(bond)
                idx_in_neighbors = neighbor_atom_indices.get(neighbor, None)
                if idx_in_neighbors is not None and idx_in_neighbors < len(properties.target_bond_lengths_neighbors):
                    target_length = properties.target_bond_lengths_neighbors[idx_in_neighbors]
                else:
                    # Default to standard bond length
                    target_length = graphene_sheet.c_c_bond_distance
                bond_target_lengths[bond] = target_length

        # Angles within the cycle
        extended_cycle = cycle_atoms + [cycle_atoms[0], cycle_atoms[1]]
        for idx in range(len(cycle_atoms)):
            node_i = extended_cycle[idx]
            node_j = extended_cycle[idx + 1]
            node_k = extended_cycle[idx + 2]
            angle = (min(node_i, node_k), node_j, max(node_i, node_k))
            inner_angle_set.add(angle)
            angle_target_angles[angle] = properties.target_angles_cycle[idx]

        # Handle additional angles for PYRIDINIC_1
        if structure.species == NitrogenSpecies.PYRIDINIC_1 and structure.additional_edge:
            node_a, node_b = structure.additional_edge
            idx_a = cycle_atoms.index(node_a)
            idx_b = cycle_atoms.index(node_b)

            # Ensure node_a is before node_b in the cycle
            if idx_a > idx_b:
                node_a, node_b = node_b, node_a
                idx_a, idx_b = idx_b, idx_a

            # Find the additional angles
            additional_angles = []

            # Angles involving node a
            prev_node_a = cycle_atoms[(idx_a - 1) % len(cycle_atoms)]
            next_node_a = cycle_atoms[(idx_a + 1) % len(cycle_atoms)]
            prev_angle = (min(prev_node_a, node_b), node_a, max(prev_node_a, node_b))
            next_angle = (min(node_b, next_node_a), node_a, max(node_b, next_node_a))
            additional_angles.extend([prev_angle, next_angle])

            # Angles involving node b
            prev_node_b = cycle_atoms[(idx_b - 1) % len(cycle_atoms)]
            next_node_b = cycle_atoms[(idx_b + 1) % len(cycle_atoms)]
            prev_angle = (min(prev_node_b, node_a), node_b, max(prev_node_b, node_a))
            next_angle = (min(node_a, next_node_b), node_b, max(node_a, next_node_b))
            additional_angles.extend([prev_angle, next_angle])

            # Assign target angles from properties.target_angles_additional_angles
            for idx, angle in enumerate(additional_angles):
                inner_angle_set.add(angle)
                angle_target_angles[angle] = properties.target_angles_additional_angles[idx]

        # Angles involving neighboring atoms
        for idx_j, node_j in enumerate(cycle_atoms):
            neighbors = [n for n in graphene_sheet.graph.neighbors(node_j) if n not in cycle_atoms]
            node_i_prev = cycle_atoms[idx_j - 1]  # Wrap-around to get the previous node
            node_k_next = cycle_atoms[(idx_j + 1) % len(cycle_atoms)]  # Wrap-around for the next node

            for neighbor in neighbors:
                # Angle: previous node in cycle - node_j - neighbor
                angle1 = (min(node_i_prev, neighbor), node_j, max(node_i_prev, neighbor))
                inner_angle_set.add(angle1)
                idx_in_neighbors = neighbor_atom_indices.get(neighbor, None)
                if idx_in_neighbors is not None and idx_in_neighbors < len(properties.target_angles_neighbors):
                    target_angle = properties.target_angles_neighbors[2 * idx_in_neighbors]
                else:
                    # Default to standard bond angle
                    target_angle = graphene_sheet.c_c_bond_angle
                angle_target_angles[angle1] = target_angle

                # Angle: neighbor - node_j - next node in cycle
                angle2 = (min(neighbor, node_k_next), node_j, max(neighbor, node_k_next))
                inner_angle_set.add(angle2)
                if idx_in_neighbors is not None and idx_in_neighbors < len(properties.target_angles_neighbors):
                    target_angle = properties.target_angles_neighbors[2 * idx_in_neighbors + 1]
                else:
                    # Default to standard bond angle
                    target_angle = graphene_sheet.c_c_bond_angle
                angle_target_angles[angle2] = target_angle

    # Collect all bonds in the graph
    all_bonds = [(min(node_i, node_j), max(node_i, node_j)) for node_i, node_j in graphene_sheet.graph.edges()]

    # Outer bonds are those not in inner_bond_set
    outer_bonds = [bond for bond in all_bonds if bond not in inner_bond_set]

    # Assign target lengths for outer bonds
    for bond in outer_bonds:
        bond_target_lengths[bond] = graphene_sheet.c_c_bond_distance

    # Collect all angles in the graph
    all_angle_set = set()
    for node_j in graphene_sheet.graph.nodes():
        neighbors = list(graphene_sheet.graph.neighbors(node_j))
        for idx_i in range(len(neighbors)):
            for idx_k in range(idx_i + 1, len(neighbors)):
                node_i = neighbors[idx_i]
                node_k = neighbors[idx_k]
                angle = (min(node_i, node_k), node_j, max(node_i, node_k))
                all_angle_set.add(angle)

    # Outer angles are those not in inner_angle_set
    outer_angle_set = all_angle_set - inner_angle_set

    # Assign target angles for outer angles
    for angle in outer_angle_set:
        angle_target_angles[angle] = graphene_sheet.c_c_bond_angle  # 120 degrees

    # Now calculate the errors
    bond_errors = []
    for bond, target_length in bond_target_lengths.items():
        node_i, node_j = bond
        position_i = positions[node_i]
        position_j = positions[node_j]
        # Calculate bond length
        calculated_length, _ = minimum_image_distance_vectorized(
            np.array([[position_i.x, position_i.y]]),
            np.array([[position_j.x, position_j.y]]),
            box_size,
        )
        absolute_error = abs(calculated_length[0] - target_length)
        normalized_error = absolute_error / target_length
        bond_errors.append(normalized_error)

    angle_errors = []
    for angle, target_angle in angle_target_angles.items():
        node_i, node_j, node_k = angle
        position_i = positions[node_i]
        position_j = positions[node_j]
        position_k = positions[node_k]
        # Calculate angle
        _, v1 = minimum_image_distance_vectorized(
            np.array([[position_i.x, position_i.y]]),
            np.array([[position_j.x, position_j.y]]),
            box_size,
        )
        _, v2 = minimum_image_distance_vectorized(
            np.array([[position_k.x, position_k.y]]),
            np.array([[position_j.x, position_j.y]]),
            box_size,
        )
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            # Avoid division by zero
            continue
        cos_theta = np.dot(v1, v2.T) / (norm_v1 * norm_v2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.degrees(np.arccos(cos_theta[0][0]))
        absolute_error = abs(theta - target_angle)
        normalized_error = absolute_error / target_angle
        angle_errors.append(normalized_error)

    # Compute average normalized errors
    average_bond_error = np.mean(bond_errors) if bond_errors else 0.0
    average_angle_error = np.mean(angle_errors) if angle_errors else 0.0

    # Total score is the average of bond and angle errors
    total_score = (average_bond_error + average_angle_error) / 2

    return total_score


def objective(trial, graphene_sheets):
    """
    Objective function for Optuna optimization.

    Parameters
    ----------
    trial : optuna.Trial
        An Optuna trial object to suggest parameter values.
    graphene_sheets : List[GrapheneSheet]
        A list of graphene sheet objects.

    Returns
    -------
    float
        The average total error across all graphene sheets for the given parameter values.
    """
    # Suggest k-parameters
    k_inner_bond = trial.suggest_float("k_inner_bond", 0.01, 100.0, log=True)
    k_outer_bond = trial.suggest_float("k_outer_bond", 0.01, 100.0, log=True)
    k_inner_angle = trial.suggest_float("k_inner_angle", 0.01, 100.0, log=True)
    k_outer_angle = trial.suggest_float("k_outer_angle", 0.01, 100.0, log=True)

    total_scores = []

    # Use tqdm to show progress
    for graphene_sheet in tqdm(graphene_sheets, desc="Processing Sheets"):
        # Create a deep copy to avoid modifying the original sheet
        graphene = copy.deepcopy(graphene_sheet)

        # Set the k-parameters
        graphene.k_inner_bond = k_inner_bond
        graphene.k_outer_bond = k_outer_bond
        graphene.k_inner_angle = k_inner_angle
        graphene.k_outer_angle = k_outer_angle

        # Adjust atom positions
        graphene.adjust_atom_positions()

        # Calculate the total error
        score = calculate_total_error(graphene)
        total_scores.append(score)

    # Compute the average total score across all sheets
    average_total_score = np.mean(total_scores)

    return average_total_score


def main():
    random.seed(0)

    # Generate graphene sheets once
    num_sheets = 5
    print("Generating graphene sheets...")
    graphene_sheets = create_graphene_sheets(num_sheets, write_to_file=True)

    # Define the Optuna study
    study = optuna.create_study(direction="minimize")

    # Optimize using the objective function, passing graphene_sheets as an additional argument
    study.optimize(lambda trial: objective(trial, graphene_sheets), n_trials=3)

    # Print the best parameters
    print("Best parameters:")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")

    # Print the best value
    print(f"Best average total error: {study.best_value}")


if __name__ == "__main__":
    main()
