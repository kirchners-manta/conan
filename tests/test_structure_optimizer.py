import os
import platform
import random
import tempfile
import warnings

import numpy as np
import numpy.testing as npt
import pytest
from ase.io import read

from conan.playground.doping import NitrogenSpecies, OptimizationWeights
from conan.playground.structure_optimizer import OptimizationConfig, StructureOptimizer
from conan.playground.structures import GrapheneSheet
from conan.playground.utils import write_xyz


def read_optimized_structure(file_path):
    """
    Read the optimized structure from an .xyz file.

    Parameters
    ----------
    file_path : str
        The path to the .xyz file containing the optimized structure.

    Returns
    -------
    optimized_positions : np.ndarray
        An array of positions with shape (n_atoms, 3).
    elements : List[str]
        A list of element symbols corresponding to each atom.
    """
    atoms = read(file_path)
    positions = atoms.get_positions()
    elements = atoms.get_chemical_symbols()
    return positions, elements


class TestStructureOptimizer:

    @pytest.fixture
    def setup_structure_optimizer_small_system(self):
        # Set up the graphene sheet
        random.seed(1)
        sheet_size = (15, 15)
        graphene = GrapheneSheet(bond_length=1.42, sheet_size=sheet_size)
        graphene.add_nitrogen_doping(
            percentages={
                NitrogenSpecies.GRAPHITIC: 1.49,
                NitrogenSpecies.PYRIDINIC_1: 1.49,
                NitrogenSpecies.PYRIDINIC_2: 2.99,
                NitrogenSpecies.PYRIDINIC_3: 4.48,
                NitrogenSpecies.PYRIDINIC_4: 5.97,
            }
        )

        # Create the optimizer
        config = OptimizationConfig(
            k_inner_bond=10.0,
            k_middle_bond=5,
            k_outer_bond=0.1,
            k_inner_angle=10.0,
            k_middle_angle=5,
            k_outer_angle=0.1,
        )
        optimizer = StructureOptimizer(graphene, config)
        return optimizer

    @pytest.fixture
    def setup_structure_optimizer(self):
        # Set the random seed for reproducibility
        random.seed(0)
        np.random.seed(0)

        # Set up the graphene sheet with the same parameters as the optimized reference structure
        sheet_size = (20, 20)
        weights = OptimizationWeights(nitrogen_percentage_weight=1, equal_distribution_weight=1)
        graphene = GrapheneSheet(bond_length=1.42, sheet_size=sheet_size)

        # Apply nitrogen doping with 10% total percentage
        graphene.add_nitrogen_doping(total_percentage=10, optimization_weights=weights)

        # Create the optimizer
        config = OptimizationConfig(
            k_inner_bond=10.0,
            k_middle_bond=5,
            k_outer_bond=0.1,
            k_inner_angle=10.0,
            k_middle_angle=5,
            k_outer_angle=0.1,
        )
        optimizer = StructureOptimizer(graphene, config)
        return optimizer

    @pytest.fixture
    def optimized_reference_structure(self):
        """
        Fixture to load the optimized reference structure from the .xyz file.

        Returns
        -------
        optimized_positions : np.ndarray
            The positions from the optimized structure.
        elements : List[str]
            The element symbols from the optimized structure.
        """
        # Get the directory where the current script is located
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the path to the reference file relative to the script's directory
        file_path = os.path.join(base_dir, "..", "structures", "optimized_structure.xyz")
        optimized_positions, elements = read_optimized_structure(file_path)
        return optimized_positions, elements

    def test_assign_target_bond_lengths_and_k_values(self, setup_structure_optimizer_small_system):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # Ignore expected warnings in this test

        optimizer = setup_structure_optimizer_small_system

        # Get all doping structures except graphitic nitrogen (graphitic nitrogen does not affect the structure)
        all_structures = [
            structure
            for structure in optimizer.doping_handler.doping_structures.structures
            if structure.species != NitrogenSpecies.GRAPHITIC
        ]

        # Return if no doping structures are present
        if not all_structures:
            return

        # Ensure consistent ordering of nodes
        all_nodes = sorted(optimizer.graph.nodes())
        node_index_map = {node: idx for idx, node in enumerate(all_nodes)}

        # Get the target bond lengths for the graphene sheet edges
        bond_array = optimizer._assign_target_bond_lengths(node_index_map, all_structures)

        # Now we can compare bond_array with the expected data
        expected_bond_list = [
            # (node_i, node_j, target_length, k_value)
            (5, 6, 1.45, 10.0),
            (5, 16, 1.34, 10.0),
            (15, 16, 1.32, 10.0),
            (14, 15, 1.47, 10.0),
            (14, 26, 1.32, 10.0),
            (25, 26, 1.34, 10.0),
            (25, 36, 1.45, 10.0),
            (36, 37, 1.45, 10.0),
            (27, 37, 1.34, 10.0),
            (27, 28, 1.32, 10.0),
            (28, 29, 1.47, 10.0),
            (17, 29, 1.32, 10.0),
            (17, 18, 1.34, 10.0),
            (6, 18, 1.45, 10.0),
            (6, 7, 1.43, 5.0),
            (4, 5, 1.455, 5.0),
            (2, 15, 1.42, 5.0),
            (13, 14, 1.42, 5.0),
            (24, 25, 1.42, 5.0),
            (35, 36, 1.43, 5.0),
            (37, 38, 1.43, 5.0),
            (28, 40, 1.42, 5.0),
            (29, 30, 1.43, 5.0),
            (18, 19, 1.43, 5.0),
            (8, 9, 1.45, 10.0),
            (8, 65, 1.33, 10.0),
            (64, 65, 1.33, 10.0),
            (54, 64, 1.45, 10.0),
            (54, 55, 1.45, 10.0),
            (55, 66, 1.33, 10.0),
            (57, 66, 1.33, 10.0),
            (57, 58, 1.45, 10.0),
            (0, 58, 1.45, 10.0),
            (0, 11, 1.33, 10.0),
            (10, 11, 1.33, 10.0),
            (9, 10, 1.45, 10.0),
            (9, 19, 1.42, 5.0),
            (7, 8, 1.43, 5.0),
            (63, 64, 1.405, 5.0),
            (53, 54, 1.42, 5.0),
            (55, 56, 1.43, 5.0),
            (46, 57, 1.43, 5.0),
            (58, 59, 1.415, 5.0),
            (0, 1, 1.43, 5.0),
            (10, 22, 1.43, 5.0),
            (33, 44, 1.39, 10.0),
            (24, 33, 1.42, 10.0),
            (23, 24, 1.42, 10.0),
            (23, 32, 1.33, 10.0),
            (21, 32, 1.35, 10.0),
            (20, 21, 1.44, 10.0),
            (20, 30, 1.44, 10.0),
            (30, 31, 1.35, 10.0),
            (31, 41, 1.33, 10.0),
            (41, 42, 1.42, 10.0),
            (42, 43, 1.42, 10.0),
            (43, 44, 1.39, 10.0),
            (33, 34, 1.45, 5.0),
            (13, 23, 1.41, 5.0),
            (21, 22, 1.44, 5.0),
            (19, 20, 1.44, 5.0),
            (40, 41, 1.41, 5.0),
            (42, 53, 1.41, 5.0),
            (43, 56, 1.45, 5.0),
            (47, 60, 1.31, 10.0),
            (47, 48, 1.42, 10.0),
            (48, 49, 1.45, 10.0),
            (49, 50, 1.51, 10.0),
            (50, 51, 1.42, 10.0),
            (51, 63, 1.4, 10.0),
            (62, 63, 1.4, 10.0),
            (61, 62, 1.42, 10.0),
            (4, 61, 1.51, 10.0),
            (3, 4, 1.45, 10.0),
            (3, 59, 1.42, 10.0),
            (59, 60, 1.31, 10.0),
            (50, 61, 1.7, 10.0),
            (46, 47, 1.41, 5.0),
            (35, 48, 1.42, 5.0),
            (38, 49, 1.48, 5.0),
            (51, 52, 1.41, 5.0),
            (7, 62, 1.41, 5.0),
            (2, 3, 1.42, 5.0),
            (1, 2, 1.42, 0.1),
            (1, 12, 1.42, 0.1),
            (12, 13, 1.42, 0.1),
            (12, 22, 1.42, 0.1),
            (34, 35, 1.42, 0.1),
            (34, 45, 1.42, 0.1),
            (38, 39, 1.42, 0.1),
            (39, 40, 1.42, 0.1),
            (39, 52, 1.42, 0.1),
            (45, 46, 1.42, 0.1),
            (45, 56, 1.42, 0.1),
            (52, 53, 1.42, 0.1),
        ]

        # Convert the expected list to an array with indices
        expected_bond_array = np.array(
            expected_bond_list, dtype=[("idx_i", int), ("idx_j", int), ("target_length", float), ("k", float)]
        )

        # Sort both arrays for comparison
        bond_array_sorted = np.sort(bond_array, order=["idx_i", "idx_j"])
        expected_bond_array_sorted = np.sort(expected_bond_array, order=["idx_i", "idx_j"])

        # Compare the arrays
        assert np.allclose(
            bond_array_sorted["target_length"], expected_bond_array_sorted["target_length"]
        ), "Bond target lengths do not match expected values."
        assert np.allclose(
            bond_array_sorted["k"], expected_bond_array_sorted["k"]
        ), "Bond k values do not match expected values."

    def test_assign_target_angles_and_k_values(self, setup_structure_optimizer_small_system):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # Ignore expected warnings in this test

        optimizer = setup_structure_optimizer_small_system

        # Get all doping structures except graphitic nitrogen (graphitic nitrogen does not affect the structure)
        all_structures = [
            structure
            for structure in optimizer.doping_handler.doping_structures.structures
            if structure.species != NitrogenSpecies.GRAPHITIC
        ]

        # Return if no doping structures are present
        if not all_structures:
            return

        # Ensure consistent ordering of nodes
        all_nodes = sorted(optimizer.graph.nodes())
        node_index_map = {node: idx for idx, node in enumerate(all_nodes)}

        # Get the target angles for the graphene sheet edges
        angle_array = optimizer._assign_target_angles(node_index_map, all_structures)

        # Now we can compare angle_array with the expected data
        expected_angle_list = [
            # (node_i, node_j, node_k, target_angle, k_value)
            (6, 5, 16, 120.26, 10.0),
            (5, 16, 15, 121.02, 10.0),
            (14, 15, 16, 119.3, 10.0),
            (15, 14, 26, 119.3, 10.0),
            (14, 26, 25, 121.02, 10.0),
            (26, 25, 36, 120.26, 10.0),
            (25, 36, 37, 122.91, 10.0),
            (27, 37, 36, 120.26, 10.0),
            (28, 27, 37, 121.02, 10.0),
            (27, 28, 29, 119.3, 10.0),
            (17, 29, 28, 119.3, 10.0),
            (18, 17, 29, 121.02, 10.0),
            (6, 18, 17, 120.26, 10.0),
            (5, 6, 18, 122.91, 10.0),
            (7, 6, 18, 118.54, 5.0),
            (5, 6, 7, 118.54, 5.0),
            (4, 5, 6, 118.86, 5.0),
            (4, 5, 16, 120.88, 5.0),
            (2, 15, 16, 122.56, 5.0),
            (2, 15, 14, 118.14, 5.0),
            (13, 14, 15, 118.14, 5.0),
            (13, 14, 26, 122.56, 5.0),
            (24, 25, 26, 120.88, 5.0),
            (24, 25, 36, 118.86, 5.0),
            (25, 36, 35, 118.54, 5.0),
            (35, 36, 37, 118.54, 5.0),
            (36, 37, 38, 118.86, 5.0),
            (27, 37, 38, 120.88, 5.0),
            (27, 28, 40, 122.56, 5.0),
            (29, 28, 40, 118.14, 5.0),
            (28, 29, 30, 118.14, 5.0),
            (17, 29, 30, 122.56, 5.0),
            (17, 18, 19, 120.88, 5.0),
            (6, 18, 19, 118.86, 5.0),
            (9, 8, 65, 120.0, 10.0),
            (8, 65, 64, 122.17, 10.0),
            (54, 64, 65, 120.0, 10.0),
            (55, 54, 64, 122.21, 10.0),
            (54, 55, 66, 120.0, 10.0),
            (55, 66, 57, 122.17, 10.0),
            (58, 57, 66, 120.0, 10.0),
            (0, 58, 57, 122.21, 10.0),
            (11, 0, 58, 120.0, 10.0),
            (0, 11, 10, 122.17, 10.0),
            (9, 10, 11, 120.0, 10.0),
            (8, 9, 10, 122.21, 10.0),
            (10, 9, 19, 118.88, 5.0),
            (8, 9, 19, 118.88, 5.0),
            (7, 8, 9, 118.92, 5.0),
            (7, 8, 65, 121.1, 5.0),
            (63, 64, 65, 121.1, 5.0),
            (54, 64, 63, 118.92, 5.0),
            (53, 54, 64, 118.88, 5.0),
            (53, 54, 55, 118.88, 5.0),
            (54, 55, 56, 118.92, 5.0),
            (56, 55, 66, 121.1, 5.0),
            (46, 57, 66, 121.1, 5.0),
            (46, 57, 58, 118.92, 5.0),
            (57, 58, 59, 118.88, 5.0),
            (0, 58, 59, 118.88, 5.0),
            (1, 0, 58, 118.92, 5.0),
            (1, 0, 11, 121.1, 5.0),
            (11, 10, 22, 121.1, 5.0),
            (9, 10, 22, 118.92, 5.0),
            (24, 33, 44, 125.51, 10.0),
            (23, 24, 33, 118.04, 10.0),
            (24, 23, 32, 117.61, 10.0),
            (21, 32, 23, 120.59, 10.0),
            (20, 21, 32, 121.71, 10.0),
            (21, 20, 30, 122.14, 10.0),
            (20, 30, 31, 121.71, 10.0),
            (30, 31, 41, 120.59, 10.0),
            (31, 41, 42, 117.61, 10.0),
            (41, 42, 43, 118.04, 10.0),
            (42, 43, 44, 125.51, 10.0),
            (33, 44, 43, 125.04, 10.0),
            (34, 33, 44, 116.54, 5.0),
            (24, 33, 34, 117.85, 5.0),
            (25, 24, 33, 121.83, 5.0),
            (23, 24, 25, 120.09, 5.0),
            (13, 23, 24, 119.2, 5.0),
            (13, 23, 32, 123.18, 5.0),
            (22, 21, 32, 119.72, 5.0),
            (20, 21, 22, 118.55, 5.0),
            (19, 20, 21, 118.91, 5.0),
            (19, 20, 30, 118.91, 5.0),
            (20, 30, 29, 118.55, 5.0),
            (29, 30, 31, 119.72, 5.0),
            (31, 41, 40, 123.18, 5.0),
            (40, 41, 42, 119.2, 5.0),
            (41, 42, 53, 120.09, 5.0),
            (43, 42, 53, 121.83, 5.0),
            (42, 43, 56, 117.85, 5.0),
            (44, 43, 56, 116.54, 5.0),
            (48, 47, 60, 115.48, 10.0),
            (47, 48, 49, 118.24, 10.0),
            (48, 49, 50, 128.28, 10.0),
            (49, 50, 51, 109.52, 10.0),
            (50, 51, 63, 112.77, 10.0),
            (51, 63, 62, 110.35, 10.0),
            (61, 62, 63, 112.77, 10.0),
            (4, 61, 62, 109.52, 10.0),
            (3, 4, 61, 128.28, 10.0),
            (4, 3, 59, 118.24, 10.0),
            (3, 59, 60, 115.48, 10.0),
            (47, 60, 59, 120.92, 10.0),
            (49, 50, 61, 148.42, 10.0),
            (51, 50, 61, 102.06, 10.0),
            (50, 61, 62, 102.06, 10.0),
            (4, 61, 50, 148.42, 10.0),
            (46, 47, 60, 121.99, 5.0),
            (46, 47, 48, 122.51, 5.0),
            (35, 48, 47, 115.67, 5.0),
            (35, 48, 49, 126.09, 5.0),
            (38, 49, 48, 111.08, 5.0),
            (38, 49, 50, 120.63, 5.0),
            (50, 51, 52, 131.0, 5.0),
            (52, 51, 63, 116.21, 5.0),
            (51, 63, 64, 124.82, 5.0),
            (62, 63, 64, 124.82, 5.0),
            (7, 62, 63, 116.21, 5.0),
            (7, 62, 61, 131.0, 5.0),
            (5, 4, 61, 120.63, 5.0),
            (3, 4, 5, 111.08, 5.0),
            (2, 3, 4, 126.09, 5.0),
            (2, 3, 59, 115.67, 5.0),
            (3, 59, 58, 122.51, 5.0),
            (58, 59, 60, 121.99, 5.0),
            (8, 7, 62, 120.0, 0.1),
            (38, 39, 52, 120.0, 0.1),
            (37, 38, 39, 120.0, 0.1),
            (0, 1, 12, 120.0, 0.1),
            (2, 1, 12, 120.0, 0.1),
            (40, 39, 52, 120.0, 0.1),
            (43, 56, 45, 120.0, 0.1),
            (33, 34, 45, 120.0, 0.1),
            (28, 40, 39, 120.0, 0.1),
            (52, 53, 54, 120.0, 0.1),
            (45, 46, 47, 120.0, 0.1),
            (18, 19, 20, 120.0, 0.1),
            (39, 40, 41, 120.0, 0.1),
            (13, 12, 22, 120.0, 0.1),
            (10, 22, 21, 120.0, 0.1),
            (6, 7, 62, 120.0, 0.1),
            (6, 7, 8, 120.0, 0.1),
            (9, 19, 18, 120.0, 0.1),
            (39, 38, 49, 120.0, 0.1),
            (33, 34, 35, 120.0, 0.1),
            (39, 52, 51, 120.0, 0.1),
            (0, 1, 2, 120.0, 0.1),
            (36, 35, 48, 120.0, 0.1),
            (34, 45, 56, 120.0, 0.1),
            (42, 53, 52, 120.0, 0.1),
            (47, 46, 57, 120.0, 0.1),
            (28, 40, 41, 120.0, 0.1),
            (10, 22, 12, 120.0, 0.1),
            (3, 2, 15, 120.0, 0.1),
            (1, 12, 22, 120.0, 0.1),
            (46, 45, 56, 120.0, 0.1),
            (34, 35, 36, 120.0, 0.1),
            (37, 38, 49, 120.0, 0.1),
            (12, 13, 23, 120.0, 0.1),
            (34, 45, 46, 120.0, 0.1),
            (9, 19, 20, 120.0, 0.1),
            (34, 35, 48, 120.0, 0.1),
            (51, 52, 53, 120.0, 0.1),
            (35, 34, 45, 120.0, 0.1),
            (1, 2, 3, 120.0, 0.1),
            (39, 52, 53, 120.0, 0.1),
            (1, 12, 13, 120.0, 0.1),
            (14, 13, 23, 120.0, 0.1),
            (45, 56, 55, 120.0, 0.1),
            (42, 53, 54, 120.0, 0.1),
            (43, 56, 55, 120.0, 0.1),
            (1, 2, 15, 120.0, 0.1),
            (45, 46, 57, 120.0, 0.1),
            (38, 39, 40, 120.0, 0.1),
            (12, 13, 14, 120.0, 0.1),
            (12, 22, 21, 120.0, 0.1),
        ]

        # Convert the expected list directly to an array
        expected_angle_array = np.array(
            expected_angle_list,
            dtype=[("idx_i", int), ("idx_j", int), ("idx_k", int), ("target_angle", float), ("k", float)],
        )

        # Sort both arrays for comparison
        angle_array_sorted = np.sort(angle_array, order=["idx_i", "idx_j", "idx_k"])
        expected_angle_array_sorted = np.sort(expected_angle_array, order=["idx_i", "idx_j", "idx_k"])

        # Compare both arrays
        assert np.allclose(
            angle_array_sorted["target_angle"], expected_angle_array_sorted["target_angle"]
        ), "Angle target angles do not match expected values."
        assert np.allclose(
            angle_array_sorted["k"], expected_angle_array_sorted["k"]
        ), "Angle k values do not match expected values."

    def test_prepare_optimization_no_doping(self, setup_structure_optimizer_small_system):
        optimizer = setup_structure_optimizer_small_system

        # Remove all doping structures
        optimizer.doping_handler.doping_structures.structures = []

        result = optimizer._prepare_optimization()
        assert result is None, "Expected None when no doping structures are present."

    def test_prepare_optimization_with_doping(self, setup_structure_optimizer_small_system):
        optimizer = setup_structure_optimizer_small_system

        result = optimizer._prepare_optimization()
        assert result is not None, "Expected data when doping structures are present."

        x0, bond_array, angle_array, box_size, all_nodes, positions = result

        # Check the types of the returned values
        assert isinstance(x0, np.ndarray), "x0 should be a numpy array."
        assert isinstance(bond_array, np.ndarray), "bond_array should be a numpy array."
        assert isinstance(angle_array, np.ndarray), "angle_array should be a numpy array."
        assert isinstance(box_size, tuple), "box_size should be a tuple."
        assert isinstance(all_nodes, list), "all_nodes should be a list."
        assert isinstance(positions, dict), "positions should be a dict."

        # Check whether the lengths of the arrays are correct
        num_nodes = len(all_nodes)
        assert x0.shape[0] == num_nodes * 2, "x0 should have length 2 * number of nodes."

    def test_bond_strain_small_system(self):
        # Create a very small system
        # Atoms: A (0, 0), B (1.0, 0), C (2.1, 0)
        # Bonds: A-B, B-C
        # Target bond lengths: 1.0 for both bonds
        # Actual bond lengths: 1.0 for A-B, 1.1 for B-C (Delta = 0.1)

        # Create positions dict
        positions = {
            0: (0.0, 0.0),  # Atom A
            1: (1.0, 0.0),  # Atom B
            2: (2.1, 0.0),  # Atom C (shifted to obtain a difference)
        }

        # Create list of nodes and map the nodes to indices
        all_nodes = [0, 1, 2]
        node_index_map = {node: idx for idx, node in enumerate(all_nodes)}

        # Flattened positions array x0
        x0 = np.array([coord for node in all_nodes for coord in positions[node]])

        # Create target bond array
        bond_list = [
            (0, 1, 1.0, 10.0),  # Bond A-B
            (1, 2, 1.0, 10.0),  # Bond B-C
        ]
        bond_array = np.array(
            [(node_index_map[i], node_index_map[j], target_length, k) for i, j, target_length, k in bond_list],
            dtype=[("idx_i", int), ("idx_j", int), ("target_length", float), ("k", float)],
        )

        # Box size for periodic boundary conditions (not relevant here)
        box_size = (10.0, 10.0)

        # Calculate the bond strain using the function under test
        strain = StructureOptimizer._bond_strain(x0, bond_array, box_size)

        # Calculate expected strain manually
        # Bond A-B: Length = 1.0, Target = 1.0, Difference = 0.0, Strain = 0.0
        # Bond B-C: Length = 1.1, Target = 1.0, Difference = 0.1, Strain = 0.5 * 10.0 * (0.1)^2 = 0.05
        expected_strain = 0.05

        # Check if the calculated strain matches the expected value
        assert np.isclose(strain, expected_strain), f"Expected strain {expected_strain}, got {strain}."

    def test_angle_strain_small_system_without_deviation(self):
        # Create positions dict
        positions = {
            0: (0.0, 0.0),  # Atom A
            1: (1.0, 0.0),  # Atom B
            2: (1.0, 1.0),  # Atom C
        }

        # Create list of nodes and map the nodes to indices
        all_nodes = [0, 1, 2]
        node_index_map = {node: idx for idx, node in enumerate(all_nodes)}

        # Flattened positions array x0
        x0 = np.array([coord for node in all_nodes for coord in positions[node]])

        # Create target angle array
        angle_list = [
            (0, 1, 2, 90.0, 10.0),  # Angle at atom B between A-B and C-B
        ]
        angle_array = np.array(
            [
                (node_index_map[i], node_index_map[j], node_index_map[k], target_angle, k_value)
                for i, j, k, target_angle, k_value in angle_list
            ],
            dtype=[("idx_i", int), ("idx_j", int), ("idx_k", int), ("target_angle", float), ("k", float)],
        )

        # Box size for periodic boundary conditions (not relevant here)
        box_size = (10.0, 10.0)

        # Calculate the angle strain using the function under test
        strain = StructureOptimizer._angle_strain(x0, angle_array, box_size)

        # Calculate expected strain manually
        # Actual angle between vectors A-B and C-B is 90 degrees, hence difference = 0
        # Strain = 0.5 * k * (Difference in radians)^2 = 0
        expected_strain = 0.0

        # Check if the calculated strain matches the expected value
        assert np.isclose(strain, expected_strain), f"Expected strain {expected_strain}, got {strain}"

    def test_angle_strain_small_system_with_deviation(self):
        # Create positions dict with a slight deviation to introduce angle strain
        positions = {
            0: (0.0, 0.0),  # Atom A
            1: (1.0, 0.0),  # Atom B
            2: (2.0, 0.1),  # Atom C (slightly off the x-axis to create an angle)
            3: (3.0, -0.1),  # Atom D (slightly below the x-axis)
        }

        # Create list of nodes and map the nodes to indices
        all_nodes = [0, 1, 2, 3]
        node_index_map = {node: idx for idx, node in enumerate(all_nodes)}

        # Flattened positions array x0
        x0 = np.array([coord for node in all_nodes for coord in positions[node]])

        # Create target angle array
        angle_list = [
            (0, 1, 2, 180.0, 10.0),  # Target angle at atom B between A-B and C-B is 180 degrees
            (1, 2, 3, 180.0, 10.0),  # Target angle at atom C between B-C and D-C is 180 degrees
        ]
        angle_array = np.array(
            [
                (node_index_map[i], node_index_map[j], node_index_map[k], target_angle, k_value)
                for i, j, k, target_angle, k_value in angle_list
            ],
            dtype=[("idx_i", int), ("idx_j", int), ("idx_k", int), ("target_angle", float), ("k", float)],
        )

        # Box size for periodic boundary conditions (not relevant here)
        box_size = (10.0, 10.0)

        # Calculate the angle strain using the function under test
        strain = StructureOptimizer._angle_strain(x0, angle_array, box_size)

        # Calculate expected strain manually
        # Calculate the actual angles
        def calculate_angle(pos_i, pos_j, pos_k):
            v1 = np.array(pos_i) - np.array(pos_j)
            v2 = np.array(pos_k) - np.array(pos_j)
            # Calculate angle in radians
            cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Avoid numerical errors
            theta = np.arccos(cos_theta)
            return np.degrees(theta)

        angle_B = calculate_angle(positions[0], positions[1], positions[2])
        angle_C = calculate_angle(positions[1], positions[2], positions[3])

        # Expected strain calculation
        # Difference between actual and target angles
        delta_theta_B = np.radians(angle_B - 180.0)
        delta_theta_C = np.radians(angle_C - 180.0)

        # Strain = 0.5 * k * delta_theta^2
        k = 10.0
        expected_strain_B = 0.5 * k * delta_theta_B**2
        expected_strain_C = 0.5 * k * delta_theta_C**2
        expected_strain = expected_strain_B + expected_strain_C

        # Check if the calculated strain matches the expected value
        assert np.isclose(strain, expected_strain), f"Expected strain {expected_strain}, got {strain}."

    def test_optimize_positions(self, setup_structure_optimizer, optimized_reference_structure):
        """
        Test that the adjusted atom positions closely match the optimized reference structure.

        Parameters
        ----------
        setup_structure_optimizer : StructureOptimizer
            The optimizer instance set up for the test.
        optimized_reference_structure : tuple
            A tuple containing the optimized positions and element symbols obtained from force field optimization.

        Raises
        ------
        AssertionError
            If the positions or elements do not match within acceptable tolerances.
        """
        # Unpack the optimized reference structure
        ref_positions, ref_elements = optimized_reference_structure

        optimizer = setup_structure_optimizer

        # Perform the position optimization
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # Ignore optimization warnings
            optimizer.optimize_positions()

        # Write the optimized structure to a temporary XYZ file
        graphene_sheet = optimizer.structure

        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as tmpfile:
            tmp_filename = tmpfile.name
            write_xyz(graphene_sheet.graph, tmp_filename)

        # Read the optimized structure from the temporary XYZ file
        optimized_atoms = read(tmp_filename)
        optimized_positions = optimized_atoms.get_positions()
        optimized_elements = optimized_atoms.get_chemical_symbols()

        # Remove the temporary file
        os.remove(tmp_filename)

        # Ensure that the number of atoms matches
        assert len(optimized_positions) == len(ref_positions), "Number of atoms does not match."

        # Compare elements
        assert optimized_elements == ref_elements, "Element symbols do not match."

        # Compare positions with tolerances
        try:
            if platform.system() in ["Windows", "Darwin"]:  # Darwin ist macOS
                npt.assert_allclose(optimized_positions, ref_positions, atol=1e-3, rtol=1e-3)
            else:
                npt.assert_allclose(optimized_positions, ref_positions, atol=1e-5, rtol=1e-5)
        except AssertionError as e:
            for i, (opt_pos, ref_pos) in enumerate(zip(optimized_positions, ref_positions)):
                print(f"Atom {i}: Optimized position = {opt_pos}, Reference position = {ref_pos}")
            raise AssertionError("Positions do not match within tolerance.") from e

    def test_assign_target_bond_lengths_missing_target_length(self, setup_structure_optimizer_small_system):
        optimizer = setup_structure_optimizer_small_system

        # Manipulate the doping structures to create a neighbor atom without a corresponding target bond length
        all_structures = [
            structure
            for structure in optimizer.doping_handler.doping_structures.structures
            if structure.species != NitrogenSpecies.GRAPHITIC
        ]

        # Modify the properties to have insufficient target bond lengths
        for structure in all_structures:
            properties = optimizer.doping_handler.species_properties[structure.species]
            # Remove the target bond lengths to simulate missing data
            properties.target_bond_lengths_neighbors = []

        # Ensure consistent ordering of nodes
        all_nodes = sorted(optimizer.graph.nodes())
        node_index_map = {node: idx for idx, node in enumerate(all_nodes)}

        # Expect a ValueError when calling _assign_target_bond_lengths
        with pytest.raises(ValueError) as excinfo:
            optimizer._assign_target_bond_lengths(node_index_map, all_structures)

        assert "Error when assigning the target bond length" in str(excinfo.value)

    def test_assign_target_angles_missing_target_angle(self, setup_structure_optimizer_small_system):
        optimizer = setup_structure_optimizer_small_system

        # Manipulate the doping structures to create a neighbor atom without a corresponding target angle
        all_structures = [
            structure
            for structure in optimizer.doping_handler.doping_structures.structures
            if structure.species != NitrogenSpecies.GRAPHITIC
        ]

        # Modify the properties to have insufficient target angles
        for structure in all_structures:
            properties = optimizer.doping_handler.species_properties[structure.species]
            # Remove the target angles to simulate missing data
            properties.target_angles_neighbors = []

        # Ensure consistent ordering of nodes
        all_nodes = sorted(optimizer.graph.nodes())
        node_index_map = {node: idx for idx, node in enumerate(all_nodes)}

        # Expect a ValueError when calling _assign_target_angles
        with pytest.raises(ValueError) as excinfo:
            optimizer._assign_target_angles(node_index_map, all_structures)

        assert "Error when assigning the target angle" in str(excinfo.value)

    def test_optimize_positions_no_doping(self, setup_structure_optimizer_small_system):
        optimizer = setup_structure_optimizer_small_system

        # Remove all doping structures
        optimizer.doping_handler.doping_structures.structures = []

        # Call optimize_positions and ensure it returns without error
        optimizer.optimize_positions()

        # Since there are no doping structures, positions should remain unchanged
        # You can check that positions are the same as before
        initial_positions = {node: data["position"] for node, data in optimizer.graph.nodes(data=True)}
        optimizer.optimize_positions()
        final_positions = {node: data["position"] for node, data in optimizer.graph.nodes(data=True)}
        assert initial_positions == final_positions, "Positions should remain unchanged when no doping is present."

    def test_assign_target_angles_missing_second_target_angle(self, setup_structure_optimizer_small_system):
        optimizer = setup_structure_optimizer_small_system

        # Get all doping structures
        all_structures = [
            structure
            for structure in optimizer.doping_handler.doping_structures.structures
            if structure.species != NitrogenSpecies.GRAPHITIC
        ]

        # Modify the properties to have insufficient target angles for angle2
        for structure in all_structures:
            properties = optimizer.doping_handler.species_properties[structure.species]
            # Ensure there is at least one neighbor
            if properties.target_angles_neighbors:
                # Keep only the first angle to make length insufficient for angle2
                properties.target_angles_neighbors = properties.target_angles_neighbors[:1]

        # Ensure consistent ordering of nodes
        all_nodes = sorted(optimizer.graph.nodes())
        node_index_map = {node: idx for idx, node in enumerate(all_nodes)}

        # Expect a ValueError when calling _assign_target_angles
        with pytest.raises(ValueError) as excinfo:
            optimizer._assign_target_angles(node_index_map, all_structures)

        assert "Error when assigning the target angle" in str(excinfo.value)

    def test_assign_target_angles_pyridinic_1_with_additional_edge_reverse_order(self):
        # Create a mock doping structure with PYRIDINIC_1 and an additional edge
        from conan.playground.doping import DopingHandler, DopingStructure, StructuralComponents
        from conan.playground.structures import GrapheneSheet

        # Set up a small graphene sheet
        graphene = GrapheneSheet(bond_length=1.42, sheet_size=(10, 10))

        # Create a doping handler with the species properties
        doping_handler = DopingHandler(graphene)

        # Define node IDs for the doping structure
        cycle_atoms = [9, 8, 1, 2, 11, 12, 13, 20, 19, 18, 17, 16]
        neighboring_atoms = [15, 0, 3, 5, 14, 21, 27, 24, 23]
        # Set additional_edge with node_a index less than node_b index
        additional_edge = (11, 19)
        structural_components = StructuralComponents
        structural_components.structure_building_atoms = [9, 11, 19]
        structural_components.structure_building_neighbors = [8, 16, 2, 12, 18, 20]
        nitrogen_atoms = [9]  # For example, node 1 is a nitrogen atom
        graphene.graph.remove_node(10)  # Remove node to create pyridinic 1 structure

        # Define a doping structure with PYRIDINIC_1 species
        doping_structure = DopingStructure(
            species=NitrogenSpecies.PYRIDINIC_1,
            cycle=cycle_atoms,
            neighboring_atoms=neighboring_atoms,
            additional_edge=additional_edge,
            structural_components=structural_components,
            nitrogen_atoms=nitrogen_atoms,
        )

        # Add the doping structure to the handler
        doping_handler.doping_structures.structures = [doping_structure]
        optimizer = StructureOptimizer(graphene, OptimizationConfig())

        # Assign the custom doping handler
        optimizer.doping_handler = doping_handler

        # Ensure consistent ordering of nodes
        all_nodes = sorted(optimizer.graph.nodes())
        node_index_map = {node: idx for idx, node in enumerate(all_nodes)}

        # Call the method to ensure the code path is covered
        angle_array = optimizer._assign_target_angles(node_index_map, [doping_structure])

        # Check that angle_array is not empty
        assert len(angle_array) > 0, "Angle array should not be empty."


class TestOptimizationConfigValidations:

    @pytest.mark.parametrize(
        "attr_name, value, expected_message",
        [
            ("k_inner_bond", "invalid", "k_inner_bond must be a float or int."),
            ("k_middle_bond", -5, "k_middle_bond must be positive. Got -5."),
            ("k_outer_bond", 0, "k_outer_bond must be positive. Got 0."),
        ],
    )
    def test_optimization_config_invalid_values(self, attr_name, value, expected_message):
        """
        Test that invalid values for spring constants raise appropriate exceptions.
        """
        kwargs = {
            "k_inner_bond": 10,
            "k_middle_bond": 5,
            "k_outer_bond": 0.1,
            "k_inner_angle": 10,
            "k_middle_angle": 5,
            "k_outer_angle": 0.1,
        }
        kwargs[attr_name] = value
        with pytest.raises((ValueError, TypeError), match=expected_message):
            OptimizationConfig(**kwargs)

    def test_optimization_config_valid_values(self):
        """
        Test that valid spring constants do not raise exceptions.
        """
        config = OptimizationConfig(
            k_inner_bond=10,
            k_middle_bond=5,
            k_outer_bond=0.1,
            k_inner_angle=10,
            k_middle_angle=5,
            k_outer_angle=0.1,
        )
        assert config.k_inner_bond == 10
        assert config.k_outer_angle == 0.1
