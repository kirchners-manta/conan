import json
import os
import random
from typing import Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from doping_experiment import Graphene, NitrogenSpecies
from graph_utils import Position, minimum_image_distance
from optuna.visualization import (
    plot_contour,
    plot_edf,
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_slice,
)
from scipy.optimize import minimize


def calculate_minimal_total_energy(
    graphene: Graphene, include_outer_angles: bool = False
) -> (Tuple)[float, Optional[Graphene]]:
    """
    Calculate the total energy of the graphene sheet, considering bond and angle energies.

    Parameters
    ----------
    graphene : Graphene
        The graphene sheet to calculate the total energy.
    include_outer_angles : bool (optional)
        Whether to include angles outside the cycles in the calculation.

    Returns
    -------
    float
        The total energy of the graphene sheet.
    Graphene
        The graphene sheet object with optimized positions.
    """
    all_cycles = []
    species_for_cycles = []

    for species, cycle_list in graphene.cycle_data.cycles.items():
        for cycle in cycle_list:
            all_cycles.append(cycle)
            species_for_cycles.append(species)

    if not all_cycles:
        return 0.0, None  # No cycles to optimize

    # Initial positions (use existing positions if available)
    positions = {node: graphene.graph.nodes[node]["position"] for node in graphene.graph.nodes}

    # Flatten initial positions for optimization
    x0 = np.array([coord for node in graphene.graph.nodes for coord in positions[node]])

    box_size = (
        graphene.actual_sheet_width + graphene.c_c_bond_distance,
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

        # Iterate over each cycle
        for idx, ordered_cycle in enumerate(all_cycles):
            # Get species properties for the current cycle
            species = species_for_cycles[idx]
            properties = graphene.species_properties[species]
            target_bond_lengths = properties.target_bond_lengths

            # Create a subgraph for the current cycle
            subgraph = graphene.graph.subgraph(ordered_cycle).copy()

            # Calculate bond energy for edges within the cycle
            cycle_length = len(ordered_cycle)
            for i in range(cycle_length):
                node_i = ordered_cycle[i]
                node_j = ordered_cycle[(i + 1) % cycle_length]  # Ensure the last node connects to the first node
                xi, yi = (
                    x[2 * list(graphene.graph.nodes).index(node_i)],
                    x[2 * list(graphene.graph.nodes).index(node_i) + 1],
                )
                xj, yj = (
                    x[2 * list(graphene.graph.nodes).index(node_j)],
                    x[2 * list(graphene.graph.nodes).index(node_j) + 1],
                )
                pos_i = Position(xi, yi)
                pos_j = Position(xj, yj)

                # Calculate the current bond length and target bond length
                current_length, _ = minimum_image_distance(pos_i, pos_j, box_size)
                target_length = target_bond_lengths[ordered_cycle.index(node_i)]
                energy += 0.5 * graphene.k_inner_bond * ((current_length - target_length) ** 2)

                # Update bond length in the graph during optimization
                graphene.graph.edges[node_i, node_j]["bond_length"] = current_length

                # Add edge to cycle_edges set
                cycle_edges.add((min(node_i, node_j), max(node_i, node_j)))

            if species == NitrogenSpecies.PYRIDINIC_1:
                for i, j in subgraph.edges():
                    if (min(i, j), max(i, j)) not in cycle_edges:
                        xi, yi = (
                            x[2 * list(graphene.graph.nodes).index(i)],
                            x[2 * list(graphene.graph.nodes).index(i) + 1],
                        )
                        xj, yj = (
                            x[2 * list(graphene.graph.nodes).index(j)],
                            x[2 * list(graphene.graph.nodes).index(j) + 1],
                        )
                        pos_i = Position(xi, yi)
                        pos_j = Position(xj, yj)

                        current_length, _ = minimum_image_distance(pos_i, pos_j, box_size)
                        target_length = target_bond_lengths[-1]  # Last bond length for Pyridinic_1
                        energy += 0.5 * graphene.k_inner_bond * ((current_length - target_length) ** 2)

                        # Update bond length in the graph during optimization
                        graphene.graph.edges[i, j]["bond_length"] = current_length

                        # Add edge to cycle_edges set
                        cycle_edges.add((min(i, j), max(i, j)))

        # Calculate bond energy for edges outside the cycles
        for i, j, data in graphene.graph.edges(data=True):
            if (min(i, j), max(i, j)) not in cycle_edges:
                xi, yi = x[2 * list(graphene.graph.nodes).index(i)], x[2 * list(graphene.graph.nodes).index(i) + 1]
                xj, yj = x[2 * list(graphene.graph.nodes).index(j)], x[2 * list(graphene.graph.nodes).index(j) + 1]
                pos_i = Position(xi, yi)
                pos_j = Position(xj, yj)

                # Calculate the current bond length and set default target length
                current_length, _ = minimum_image_distance(pos_i, pos_j, box_size)
                target_length = 1.42
                energy += 0.5 * graphene.k_outer_bond * ((current_length - target_length) ** 2)

                # Update bond length in the graph during optimization
                graphene.graph.edges[i, j]["bond_length"] = current_length

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
            properties = graphene.species_properties[species]
            target_angles = properties.target_angles

            for (i, j, k), angle in zip(zip(ordered_cycle, ordered_cycle[1:], ordered_cycle[2:]), target_angles):
                xi, yi = x[2 * list(graphene.graph.nodes).index(i)], x[2 * list(graphene.graph.nodes).index(i) + 1]
                xj, yj = x[2 * list(graphene.graph.nodes).index(j)], x[2 * list(graphene.graph.nodes).index(j) + 1]
                xk, yk = x[2 * list(graphene.graph.nodes).index(k)], x[2 * list(graphene.graph.nodes).index(k) + 1]

                pos_i = Position(xi, yi)
                pos_j = Position(xj, yj)
                pos_k = Position(xk, yk)

                _, v1 = minimum_image_distance(pos_i, pos_j, box_size)
                _, v2 = minimum_image_distance(pos_k, pos_j, box_size)

                cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
                energy += 0.5 * graphene.k_inner_angle * ((theta - np.radians(angle)) ** 2)

                # Add angles to counted_angles to avoid double-counting
                counted_angles.add((i, j, k))
                counted_angles.add((k, j, i))

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
                        pos_node = Position(x_node, y_node)
                        pos_i = Position(x_i, y_i)
                        pos_j = Position(x_j, y_j)
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

    # Update the positions in the graphene object with the optimized positions
    optimized_positions = result.x.reshape((-1, 2))
    for i, node in enumerate(graphene.graph.nodes):
        graphene.graph.nodes[node]["position"] = (optimized_positions[i][0], optimized_positions[i][1])

    return result.fun, graphene


def calculate_bond_angle_accuracy(graphene: Graphene) -> Tuple[float, float]:
    all_cycles = []
    species_for_cycles = []

    for species, cycle_list in graphene.cycle_data.cycles.items():
        for cycle in cycle_list:
            all_cycles.append(cycle)
            species_for_cycles.append(species)

    bond_accuracy = 0.0
    angle_accuracy = 0.0

    for idx, ordered_cycle in enumerate(all_cycles):
        species = species_for_cycles[idx]
        properties = graphene.species_properties[species]
        target_bond_lengths = properties.target_bond_lengths
        target_angles = properties.target_angles
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
                    graphene.actual_sheet_width + graphene.c_c_bond_distance,
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
                    graphene.actual_sheet_width + graphene.c_c_bond_distance,
                    graphene.actual_sheet_height + graphene.cc_y_distance,
                ),
            )
            _, v2 = minimum_image_distance(
                pos_k,
                pos_j,
                (
                    graphene.actual_sheet_width + graphene.c_c_bond_distance,
                    graphene.actual_sheet_height + graphene.cc_y_distance,
                ),
            )
            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
            angle_accuracy += abs(np.degrees(theta) - angle)

    bond_accuracy /= len(all_cycles)
    angle_accuracy /= len(all_cycles)

    return bond_accuracy, angle_accuracy


def objective_total_energy_pyridinic_4(trial):
    # Sample k_inner and k_outer
    k_inner_bond = trial.suggest_float("k_inner_bond", 1.0, 1000.0, log=True)
    k_inner_angle = trial.suggest_float("k_inner_angle", 1.0, 1000.0, log=True)
    k_outer_bond = trial.suggest_float("k_outer_bond", 0.01, 10.0, log=True)
    # k_outer_angle = trial.suggest_float("k_outer_angle", 0.01, 10.0, log=True)

    # Create Graphene instance and set k_inner_bond and k_outer_bond
    graphene = Graphene(bond_distance=1.42, sheet_size=(20, 20))
    graphene.k_inner_bond = k_inner_bond
    graphene.k_outer_bond = k_outer_bond
    graphene.k_inner_angle = k_inner_angle
    # graphene.k_outer_angle = k_outer_angle

    # Add nitrogen doping to the graphene sheet
    graphene.add_nitrogen_doping(percentages={NitrogenSpecies.PYRIDINIC_4: 3})

    # Calculate the total energy of the graphene sheet
    total_energy, _ = calculate_minimal_total_energy(graphene)

    # Return the total energy as the objective value
    return total_energy


def objective_total_energy_all_structures(trial):
    # Sample k_inner and k_outer
    k_inner_bond = trial.suggest_float("k_inner_bond", 0.01, 100.0, log=True)
    k_inner_angle = trial.suggest_float("k_inner_angle", 0.01, 100.0, log=True)
    k_outer_bond = trial.suggest_float("k_outer_bond", 0.01, 100.0, log=True)
    # k_outer_angle = trial.suggest_float("k_outer_angle", 0.01, 100.0, log=True)

    # Create Graphene instance and set k_inner_bond and k_outer_bond
    graphene = Graphene(bond_distance=1.42, sheet_size=(20, 20))
    graphene.k_inner_bond = k_inner_bond
    graphene.k_outer_bond = k_outer_bond
    graphene.k_inner_angle = k_inner_angle
    # graphene.k_outer_angle = k_outer_angle

    # Add nitrogen doping to the graphene sheet
    graphene.add_nitrogen_doping(total_percentage=15)

    # Calculate the total energy of the graphene sheet
    total_energy, _ = calculate_minimal_total_energy(graphene)

    # Return the total energy as the objective value
    return total_energy


def objective_combined_pyridinic_4(trial):
    # Sample k_inner and k_outer
    k_inner_bond = trial.suggest_float("k_inner_bond", 0.01, 100.0, log=True)
    k_inner_angle = trial.suggest_float("k_inner_angle", 0.01, 100.0, log=True)
    k_outer_bond = trial.suggest_float("k_outer_bond", 0.01, 100.0, log=True)
    # k_outer_angle = trial.suggest_float("k_outer_angle", 0.01, 100.0, log=True)

    # Create Graphene instance and set k_inner_bond and k_outer_bond
    graphene = Graphene(bond_distance=1.42, sheet_size=(20, 20))
    graphene.k_inner_bond = k_inner_bond
    graphene.k_outer_bond = k_outer_bond
    graphene.k_inner_angle = k_inner_angle
    # graphene.k_outer_angle = k_outer_angle

    # Add nitrogen doping to the graphene sheet
    graphene.add_nitrogen_doping(percentages={NitrogenSpecies.PYRIDINIC_4: 3})

    # Calculate the total energy of the graphene sheet
    total_energy, updated_graphene = calculate_minimal_total_energy(graphene)

    # Calculate bond and angle accuracy within cycles (additional objectives can be added)
    bond_accuracy, angle_accuracy = calculate_bond_angle_accuracy(updated_graphene)

    # Combine objectives
    objective_value = total_energy + bond_accuracy + angle_accuracy

    return objective_value


def objective_combined_all_structures(trial):
    # Sample k_inner and k_outer
    k_inner_bond = trial.suggest_float("k_inner_bond", 0.01, 100.0, log=True)
    k_inner_angle = trial.suggest_float("k_inner_angle", 0.01, 100.0, log=True)
    k_outer_bond = trial.suggest_float("k_outer_bond", 0.01, 100.0, log=True)
    # k_outer_angle = trial.suggest_float("k_outer_angle", 0.01, 100.0, log=True)

    # Create Graphene instance and set k_inner_bond and k_outer_bond
    graphene = Graphene(bond_distance=1.42, sheet_size=(20, 20))
    graphene.k_inner_bond = k_inner_bond
    graphene.k_outer_bond = k_outer_bond
    graphene.k_inner_angle = k_inner_angle
    # graphene.k_outer_angle = k_outer_angle

    # Add nitrogen doping to the graphene sheet
    graphene.add_nitrogen_doping(total_percentage=15)

    # Calculate the total energy of the graphene sheet
    total_energy, updated_graphene = calculate_minimal_total_energy(graphene)

    # Calculate bond and angle accuracy within cycles (additional objectives can be added)
    bond_accuracy, angle_accuracy = calculate_bond_angle_accuracy(updated_graphene)

    # Combine objectives
    objective_value = total_energy + bond_accuracy + angle_accuracy

    return objective_value


def objective_total_energy_pyridinic_4_with_outer_angles(trial):
    k_inner_bond = trial.suggest_float("k_inner_bond", 0.01, 100.0, log=True)
    k_inner_angle = trial.suggest_float("k_inner_angle", 0.01, 100.0, log=True)
    k_outer_bond = trial.suggest_float("k_outer_bond", 0.01, 100.0, log=True)
    k_outer_angle = trial.suggest_float("k_outer_angle", 0.01, 100.0, log=True)

    graphene = Graphene(bond_distance=1.42, sheet_size=(20, 20))
    graphene.k_inner_bond = k_inner_bond
    graphene.k_outer_bond = k_outer_bond
    graphene.k_inner_angle = k_inner_angle
    graphene.k_outer_angle = k_outer_angle

    graphene.add_nitrogen_doping(percentages={NitrogenSpecies.PYRIDINIC_4: 3})

    total_energy, _ = calculate_minimal_total_energy(graphene, include_outer_angles=True)
    return total_energy


def objective_total_energy_all_structures_with_outer_angles(trial):
    k_inner_bond = trial.suggest_float("k_inner_bond", 0.01, 100.0, log=True)
    k_inner_angle = trial.suggest_float("k_inner_angle", 0.01, 100.0, log=True)
    k_outer_bond = trial.suggest_float("k_outer_bond", 0.01, 100.0, log=True)
    k_outer_angle = trial.suggest_float("k_outer_angle", 0.01, 100.0, log=True)

    graphene = Graphene(bond_distance=1.42, sheet_size=(20, 20))
    graphene.k_inner_bond = k_inner_bond
    graphene.k_outer_bond = k_outer_bond
    graphene.k_inner_angle = k_inner_angle
    graphene.k_outer_angle = k_outer_angle

    graphene.add_nitrogen_doping(total_percentage=15)

    total_energy, _ = calculate_minimal_total_energy(graphene, include_outer_angles=True)
    return total_energy


def objective_combined_pyridinic_4_with_outer_angles(trial):
    k_inner_bond = trial.suggest_float("k_inner_bond", 0.01, 100.0, log=True)
    k_inner_angle = trial.suggest_float("k_inner_angle", 0.01, 100.0, log=True)
    k_outer_bond = trial.suggest_float("k_outer_bond", 0.01, 100.0, log=True)
    k_outer_angle = trial.suggest_float("k_outer_angle", 0.01, 100.0, log=True)

    graphene = Graphene(bond_distance=1.42, sheet_size=(20, 20))
    graphene.k_inner_bond = k_inner_bond
    graphene.k_outer_bond = k_outer_bond
    graphene.k_inner_angle = k_inner_angle
    graphene.k_outer_angle = k_outer_angle

    graphene.add_nitrogen_doping(percentages={NitrogenSpecies.PYRIDINIC_4: 3})

    total_energy, updated_graphene = calculate_minimal_total_energy(graphene, include_outer_angles=True)
    bond_accuracy, angle_accuracy = calculate_bond_angle_accuracy(updated_graphene)
    objective_value = total_energy + bond_accuracy + angle_accuracy

    return objective_value


def objective_combined_all_structures_with_outer_angles(trial):
    k_inner_bond = trial.suggest_float("k_inner_bond", 0.01, 100.0, log=True)
    k_inner_angle = trial.suggest_float("k_inner_angle", 0.01, 100.0, log=True)
    k_outer_bond = trial.suggest_float("k_outer_bond", 0.01, 100.0, log=True)
    k_outer_angle = trial.suggest_float("k_outer_angle", 0.01, 100.0, log=True)

    graphene = Graphene(bond_distance=1.42, sheet_size=(20, 20))
    graphene.k_inner_bond = k_inner_bond
    graphene.k_outer_bond = k_outer_bond
    graphene.k_inner_angle = k_inner_angle
    graphene.k_outer_angle = k_outer_angle

    graphene.add_nitrogen_doping(total_percentage=15)

    total_energy, updated_graphene = calculate_minimal_total_energy(graphene, include_outer_angles=True)
    bond_accuracy, angle_accuracy = calculate_bond_angle_accuracy(updated_graphene)
    objective_value = total_energy + bond_accuracy + angle_accuracy

    return objective_value


def save_study_results(study, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df = study.trials_dataframe()
    df = df.applymap(lambda x: str(x) if isinstance(x, (pd.Timestamp, pd.Timedelta)) else x)
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
def conduct_study(objective_function, study_name, n_trials=200):
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
    random.seed(0)

    os.makedirs("optuna_results", exist_ok=True)

    # Conduct study for total energy with Pyridinic_4
    conduct_study(objective_total_energy_pyridinic_4, "total_energy_pyridinic_4")

    # Conduct study for total energy with all structures
    conduct_study(objective_total_energy_all_structures, "total_energy_all_structures")

    # Conduct study for combined objective with Pyridinic_4
    conduct_study(objective_combined_pyridinic_4, "combined_pyridinic_4")

    # Conduct study for combined objective with all structures
    conduct_study(objective_combined_all_structures, "combined_all_structures")

    # Conduct study for total energy with Pyridinic_4 including outer angles
    conduct_study(
        objective_total_energy_pyridinic_4_with_outer_angles, "total_energy_pyridinic_4_including_outer_angles"
    )

    # Conduct study for total energy with all structures including outer angles
    conduct_study(
        objective_total_energy_all_structures_with_outer_angles, "total_energy_all_structures_including_outer_angles"
    )

    # Conduct study for combined objective with Pyridinic_4 including outer angles
    conduct_study(objective_combined_pyridinic_4_with_outer_angles, "combined_pyridinic_4_including_outer_angles")

    # Conduct study for combined objective with all structures including outer angles
    conduct_study(objective_combined_all_structures_with_outer_angles, "combined_all_structures_including_outer_angles")

    # # Conduct study for combined objective with Pyridinic_4
    # conduct_study(objective_combined_pyridinic_4, "combined_pyridinic_4_test", n_trials=2)
