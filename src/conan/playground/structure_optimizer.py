from dataclasses import dataclass
from itertools import pairwise

import networkx as nx
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm

from conan.playground.doping_experiment import MaterialStructure
from conan.playground.graph_utils import NitrogenSpecies, Position, minimum_image_distance_vectorized


@dataclass
class OptimizationConfig:
    """
    Configuration of spring constants for the structure optimization process.
    """

    k_inner_bond: float = 90.0
    # self.k_inner_bond = 10
    """The spring constant for bonds within the doping structure (cycle) as well as the direct bonds from the cycle
    atoms to their neighbors in the graphene sheet."""
    k_outer_bond: float = 75.0
    # self.k_outer_bond = 0.1
    """The spring constant for bonds outside the doping structure (cycle) and not directly connected to it."""
    k_inner_angle: float = 11.6
    # self.k_inner_angle = 10
    """The spring constant for angles within the doping structure (cycle) as well as the angles between the cycle
    atoms and their neighbors in the graphene sheet."""
    k_outer_angle: float = 11.6
    # self.k_outer_angle = 0.1
    """The spring constant for angles outside the doping structure (cycle) and not directly connected to it."""


class StructureOptimizer:
    def __init__(self, structure: MaterialStructure, config: OptimizationConfig):
        """
        Initialize the StructureOptimizer with the given carbon structure and optimization configuration.

        Parameters
        ----------
        structure : MaterialStructure
            The structure to optimize.
        config : OptimizationConfig
            Configuration containing optimization constants.
        """
        self.structure = structure
        """The material structure to optimize."""
        self.graph = structure.graph
        """The networkx graph representing the material structure."""
        self.doping_handler = structure.doping_handler
        """The doping handler for the material structure."""

        # Assign constants from config
        self.k_inner_bond = config.k_inner_bond
        self.k_outer_bond = config.k_outer_bond
        self.k_inner_angle = config.k_inner_angle
        self.k_outer_angle = config.k_outer_angle

    def optimize_positions(self):
        """
        Adjust the positions of atoms in the material structure to optimize the structure and minimize structural
        strain.
        """
        self._adjust_atom_positions()

    def _adjust_atom_positions(self):
        """
        Adjust the positions of atoms in the graphene sheet to optimize the structure including doping (minimize
        structural strain).

        Notes
        -----
        This method adjusts the positions of atoms in a graphene sheet to optimize the structure based on the doping
        configuration. It uses a combination of bond and angle energies to minimize the total energy of the system,
        following the specified definitions for bond and angle spring constants.

        It handles cases where bonds are included in multiple doping structures by averaging the target
        lengths assigned to them. Each bond has a single force constant `k`, determined by whether it is an inner
        or outer bond. Angles are assigned target values based on their involvement in doping structures or are set
        to the default bond angle (120 degrees) otherwise.
        """

        # Get all doping structures except graphitic nitrogen (graphitic nitrogen does not affect the structure)
        all_structures = [
            structure
            for structure in self.doping_handler.doping_structures.structures
            if structure.species != NitrogenSpecies.GRAPHITIC
        ]

        # Return if no doping structures are present
        if not all_structures:
            return

        # Ensure consistent ordering of nodes
        all_nodes = sorted(self.graph.nodes())
        node_index_map = {node: idx for idx, node in enumerate(all_nodes)}

        # Get the initial positions of atoms, ordered consistently
        positions = {node: self.graph.nodes[node]["position"] for node in all_nodes}

        # Flatten the positions into a 1D array for optimization (alternating x and y)
        x0 = np.array([coord for node in all_nodes for coord in (positions[node][0], positions[node][1])])

        # Define the box size for minimum image distance calculation
        box_size = (
            self.structure.actual_sheet_width + self.structure.c_c_bond_distance,
            self.structure.actual_sheet_height + self.structure.cc_y_distance,
        )

        # Dictionaries to store bond and angle properties
        bond_target_lengths = {}  # key: (node_i, node_j), value: list of target lengths
        bond_k_values = {}  # key: (node_i, node_j), value: k value
        angle_target_angles = {}  # key: (node_i, node_j, node_k), value: target angle in degrees
        angle_k_values = {}  # key: (node_i, node_j, node_k), value: k value

        # Sets to keep track of inner bonds and angles
        inner_bond_set = set()
        inner_angle_set = set()

        # Collect inner bonds and angles from doping structures
        for structure in all_structures:
            properties = self.doping_handler.species_properties[structure.species]

            cycle_atoms = structure.cycle
            neighbor_atoms = structure.neighboring_atoms

            # Map node IDs to indices in the neighbors list
            neighbor_atom_indices = {node: idx for idx, node in enumerate(neighbor_atoms)}

            # Bonds within the cycle
            cycle_edges = list(pairwise(cycle_atoms + [cycle_atoms[0]]))
            if structure.species == NitrogenSpecies.PYRIDINIC_1 and structure.additional_edge:
                cycle_edges.append(structure.additional_edge)

            for idx, (node_i, node_j) in enumerate(cycle_edges):
                bond = (min(node_i, node_j), max(node_i, node_j))
                inner_bond_set.add(bond)

                # Append target length
                bond_target_lengths.setdefault(bond, []).append(properties.target_bond_lengths_cycle[idx])
                # Assign k value (k_inner_bond)
                bond_k_values[bond] = self.k_inner_bond

            # Bonds between cycle atoms and their neighbors
            for idx, node_i in enumerate(cycle_atoms):
                neighbors = [n for n in self.graph.neighbors(node_i) if n not in cycle_atoms]
                for neighbor in neighbors:
                    bond = (min(node_i, neighbor), max(node_i, neighbor))
                    inner_bond_set.add(bond)
                    idx_in_neighbors = neighbor_atom_indices.get(neighbor, None)
                    if idx_in_neighbors is not None and idx_in_neighbors < len(
                        properties.target_bond_lengths_neighbors
                    ):
                        target_length = properties.target_bond_lengths_neighbors[idx_in_neighbors]
                    else:
                        raise ValueError(
                            f"Error when assigning the target bond length: Neighbor atom {neighbor} "
                            f"(index {idx_in_neighbors}) has no corresponding target length in "
                            f"target_bond_lengths_neighbors."
                        )
                    bond_target_lengths.setdefault(bond, []).append(target_length)
                    # Assign k value (k_inner_bond)
                    bond_k_values[bond] = self.k_inner_bond

            # Angles within the cycle
            # Extend the cycle to account for the closed loop by adding the first two nodes at the end
            extended_cycle = cycle_atoms + [cycle_atoms[0], cycle_atoms[1]]
            for idx in range(len(cycle_atoms)):
                node_i = extended_cycle[idx]
                node_j = extended_cycle[idx + 1]
                node_k = extended_cycle[idx + 2]
                angle = (min(node_i, node_k), node_j, max(node_i, node_k))
                inner_angle_set.add(angle)
                angle_target_angles[angle] = properties.target_angles_cycle[idx]
                # Assign k value (k_inner_angle)
                angle_k_values[angle] = self.k_inner_angle

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
                    angle_k_values[angle] = self.k_inner_angle

            # Angles involving neighboring atoms
            for idx_j, node_j in enumerate(cycle_atoms):
                neighbors = [n for n in self.graph.neighbors(node_j) if n not in cycle_atoms]
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
                        raise ValueError(
                            f"Error when assigning the target angle: Neighbor atom {neighbor} (index "
                            f"{idx_in_neighbors}) has no corresponding target angle in 'target_angles_neighbors'."
                        )
                    angle_target_angles[angle1] = target_angle
                    angle_k_values[angle1] = self.k_inner_angle

                    # Angle: neighbor - node_j - next node in cycle
                    angle2 = (min(neighbor, node_k_next), node_j, max(neighbor, node_k_next))
                    inner_angle_set.add(angle2)
                    if idx_in_neighbors is not None and idx_in_neighbors < len(properties.target_angles_neighbors):
                        target_angle = properties.target_angles_neighbors[2 * idx_in_neighbors + 1]
                    else:
                        raise ValueError(
                            f"Error when assigning the target angle: Neighbor atom {neighbor} (index "
                            f"{idx_in_neighbors}) has no corresponding target angle in 'target_angles_neighbors'."
                        )
                    angle_target_angles[angle2] = target_angle
                    angle_k_values[angle2] = self.k_inner_angle

        # Collect all bonds in the graph
        all_bonds = [(min(node_i, node_j), max(node_i, node_j)) for node_i, node_j in self.graph.edges()]

        # Outer bonds are those not in inner_bond_set
        outer_bonds = [bond for bond in all_bonds if bond not in inner_bond_set]

        # Assign target lengths and k values for outer bonds
        for bond in outer_bonds:
            bond_target_lengths.setdefault(bond, []).append(self.structure.c_c_bond_distance)
            bond_k_values[bond] = self.k_outer_bond

        # Collect all angles in the graph
        all_angle_set = set()
        for node_j in self.graph.nodes():
            neighbors = list(self.graph.neighbors(node_j))
            for idx_i in range(len(neighbors)):
                for idx_k in range(idx_i + 1, len(neighbors)):
                    node_i = neighbors[idx_i]
                    node_k = neighbors[idx_k]
                    # angle = (node_i, node_j, node_k)
                    angle = (min(node_i, node_k), node_j, max(node_i, node_k))
                    all_angle_set.add(angle)

        # Outer angles are those not in inner_angle_set
        outer_angle_set = all_angle_set - inner_angle_set

        # Assign target angles for outer angles
        for angle in outer_angle_set:
            angle_target_angles[angle] = self.structure.c_c_bond_angle  # 120 degrees
            angle_k_values[angle] = self.k_outer_angle

        # Prepare data for bond strain calculation
        bond_list = []
        for bond in bond_target_lengths:
            node_i, node_j = bond
            idx_i = node_index_map[node_i]
            idx_j = node_index_map[node_j]
            # Average target lengths
            avg_target_length = np.mean(bond_target_lengths[bond])
            k_value = bond_k_values[bond]
            bond_list.append((idx_i, idx_j, avg_target_length, k_value))

        bond_array = np.array(bond_list, dtype=[("idx_i", int), ("idx_j", int), ("target_length", float), ("k", float)])

        # Prepare data for angle strain calculation
        angle_list = []
        for angle in angle_target_angles:
            node_i, node_j, node_k = angle
            idx_i = node_index_map[node_i]
            idx_j = node_index_map[node_j]
            idx_k = node_index_map[node_k]
            target_angle = angle_target_angles[angle]
            k_value = angle_k_values[angle]
            angle_list.append((idx_i, idx_j, idx_k, target_angle, k_value))

        angle_array = np.array(
            angle_list,
            dtype=[("idx_i", int), ("idx_j", int), ("idx_k", int), ("target_angle", float), ("k", float)],
        )

        def bond_strain(x):
            """
            Calculate the bond strain for the given atom positions.

            Parameters
            ----------
            x : ndarray
                Flattened array of positions of all atoms.

            Returns
            -------
            total_strain : float
                The total bond strain in the structure.
            """
            # Extract positions
            idx_i_array = bond_array["idx_i"]
            idx_j_array = bond_array["idx_j"]
            positions_i = x[np.ravel(np.column_stack((idx_i_array * 2, idx_i_array * 2 + 1)))]
            positions_j = x[np.ravel(np.column_stack((idx_j_array * 2, idx_j_array * 2 + 1)))]
            positions_i = positions_i.reshape(-1, 2)
            positions_j = positions_j.reshape(-1, 2)

            # Calculate bond lengths
            current_lengths, _ = minimum_image_distance_vectorized(positions_i, positions_j, box_size)

            # Calculate bond strain
            target_lengths = bond_array["target_length"]
            k_values = bond_array["k"]
            total_bond_strain = 0.5 * np.sum(k_values * (current_lengths - target_lengths) ** 2)

            return total_bond_strain

        def angle_strain(x):
            """
            Calculate the angle strain for the given atom positions.

            Parameters
            ----------
            x : ndarray
                Flattened array of positions of all atoms.

            Returns
            -------
            total_strain : float
                The total angular strain in the structure.
            """
            # Extract positions
            idx_i_array = angle_array["idx_i"]
            idx_j_array = angle_array["idx_j"]
            idx_k_array = angle_array["idx_k"]
            positions_i = x[np.ravel(np.column_stack((idx_i_array * 2, idx_i_array * 2 + 1)))]
            positions_j = x[np.ravel(np.column_stack((idx_j_array * 2, idx_j_array * 2 + 1)))]
            positions_k = x[np.ravel(np.column_stack((idx_k_array * 2, idx_k_array * 2 + 1)))]
            positions_i = positions_i.reshape(-1, 2)
            positions_j = positions_j.reshape(-1, 2)
            positions_k = positions_k.reshape(-1, 2)

            # Calculate vectors
            _, v1 = minimum_image_distance_vectorized(positions_i, positions_j, box_size)
            _, v2 = minimum_image_distance_vectorized(positions_k, positions_j, box_size)

            # Calculate norms
            norm_v1 = np.linalg.norm(v1, axis=1)
            norm_v2 = np.linalg.norm(v2, axis=1)

            # Prevent division by zero
            norm_v1 = np.where(norm_v1 == 0, 1e-8, norm_v1)
            norm_v2 = np.where(norm_v2 == 0, 1e-8, norm_v2)

            # Calculate cos_theta safely
            cos_theta = np.einsum("ij,ij->i", v1, v2) / (norm_v1 * norm_v2)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)

            # Calculate angles
            theta = np.arccos(cos_theta)

            # Calculate angle strain
            target_angles = np.radians(angle_array["target_angle"])
            delta_theta = theta - target_angles
            k_values = angle_array["k"]
            total_bond_strain = 0.5 * np.sum(k_values * delta_theta**2)

            return total_bond_strain

        def total_strain(x):
            """
            Calculate the total structural strain (bond + angular) for the given positions.

            Parameters
            ----------
            x : ndarray
                Flattened array of positions of all atoms.

            Returns
            -------
            total_strain : float
                The total structural strain in the system.
            """
            return bond_strain(x) + angle_strain(x)

        # Initialize the progress bar
        progress_bar = tqdm(total=None, desc="Optimizing positions", unit="iteration")

        def optimization_callback(xk):
            # Update the progress bar by one step
            progress_bar.update(1)

        # Start the optimization process with the callback to update progress
        result = minimize(total_strain, x0, method="L-BFGS-B", callback=optimization_callback, options={"disp": True})

        # Close the progress bar
        progress_bar.close()

        # Print the number of iterations and final energy
        print(f"\nNumber of iterations: {result.nit}\nFinal structural strain: {result.fun}")

        # Reshape the optimized positions back to the 2D array format
        optimized_positions = result.x.reshape(-1, 2)

        # Update the positions of atoms in the graph with the optimized positions
        position_dict = {
            node: Position(optimized_positions[idx][0], optimized_positions[idx][1], positions[node][2])
            for idx, node in enumerate(all_nodes)
        }
        nx.set_node_attributes(self.graph, position_dict, "position")

        # Extract positions for bond length calculation
        positions_array = optimized_positions  # Shape: (num_nodes, 2)

        idx_i_array = bond_array["idx_i"]
        idx_j_array = bond_array["idx_j"]
        positions_i = positions_array[idx_i_array]
        positions_j = positions_array[idx_j_array]

        # Calculate bond lengths
        current_lengths, _ = minimum_image_distance_vectorized(positions_i, positions_j, box_size)

        # Prepare bond length updates for all bonds
        edge_updates = {
            (all_nodes[idx_i_array[idx]], all_nodes[idx_j_array[idx]]): {"bond_length": current_lengths[idx]}
            for idx in range(len(idx_i_array))
        }

        # Update the bond lengths in the graph
        nx.set_edge_attributes(self.graph, edge_updates)
