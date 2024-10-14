import random

import numpy as np
import pytest

from conan.playground.doping_experiment import GrapheneSheet, OptimizationConfig, StructureOptimizer
from conan.playground.graph_utils import NitrogenSpecies


class TestStructureOptimizer:

    @pytest.fixture
    def setup_structure_optimizer(self):
        # Set up the graphene sheet
        random.seed(1)
        sheet_size = (15, 15)
        graphene = GrapheneSheet(bond_length=1.42, sheet_size=sheet_size)
        graphene.add_nitrogen_doping(total_percentage=10)

        # Create the optimizer
        config = OptimizationConfig(
            k_inner_bond=10.0,
            k_outer_bond=0.1,
            k_inner_angle=10.0,
            k_outer_angle=0.1,
        )
        optimizer = StructureOptimizer(graphene, config)
        return optimizer

    def test_assign_target_bond_lengths_and_k_values(self, setup_structure_optimizer):
        optimizer = setup_structure_optimizer

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
            (6, 7, 1.43, 10.0),
            (4, 5, 1.455, 10.0),
            (2, 15, 1.42, 10.0),
            (13, 14, 1.42, 10.0),
            (24, 25, 1.42, 10.0),
            (35, 36, 1.43, 10.0),
            (37, 38, 1.43, 10.0),
            (28, 40, 1.42, 10.0),
            (29, 30, 1.43, 10.0),
            (18, 19, 1.43, 10.0),
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
            (9, 19, 1.42, 10.0),
            (7, 8, 1.43, 10.0),
            (63, 64, 1.405, 10.0),
            (53, 54, 1.42, 10.0),
            (55, 56, 1.43, 10.0),
            (46, 57, 1.43, 10.0),
            (58, 59, 1.415, 10.0),
            (0, 1, 1.43, 10.0),
            (10, 22, 1.43, 10.0),
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
            (33, 34, 1.45, 10.0),
            (13, 23, 1.41, 10.0),
            (21, 22, 1.44, 10.0),
            (19, 20, 1.44, 10.0),
            (40, 41, 1.41, 10.0),
            (42, 53, 1.41, 10.0),
            (43, 56, 1.45, 10.0),
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
            (46, 47, 1.41, 10.0),
            (35, 48, 1.42, 10.0),
            (38, 49, 1.48, 10.0),
            (51, 52, 1.41, 10.0),
            (7, 62, 1.41, 10.0),
            (2, 3, 1.42, 10.0),
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

    def test_assign_target_angles_and_k_values(self, setup_structure_optimizer):
        optimizer = setup_structure_optimizer

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
            (7, 6, 18, 118.54, 10.0),
            (5, 6, 7, 118.54, 10.0),
            (4, 5, 6, 118.86, 10.0),
            (4, 5, 16, 120.88, 10.0),
            (2, 15, 16, 122.56, 10.0),
            (2, 15, 14, 118.14, 10.0),
            (13, 14, 15, 118.14, 10.0),
            (13, 14, 26, 122.56, 10.0),
            (24, 25, 26, 120.88, 10.0),
            (24, 25, 36, 118.86, 10.0),
            (25, 36, 35, 118.54, 10.0),
            (35, 36, 37, 118.54, 10.0),
            (36, 37, 38, 118.86, 10.0),
            (27, 37, 38, 120.88, 10.0),
            (27, 28, 40, 122.56, 10.0),
            (29, 28, 40, 118.14, 10.0),
            (28, 29, 30, 118.14, 10.0),
            (17, 29, 30, 122.56, 10.0),
            (17, 18, 19, 120.88, 10.0),
            (6, 18, 19, 118.86, 10.0),
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
            (10, 9, 19, 118.88, 10.0),
            (8, 9, 19, 118.88, 10.0),
            (7, 8, 9, 118.92, 10.0),
            (7, 8, 65, 121.1, 10.0),
            (63, 64, 65, 121.1, 10.0),
            (54, 64, 63, 118.92, 10.0),
            (53, 54, 64, 118.88, 10.0),
            (53, 54, 55, 118.88, 10.0),
            (54, 55, 56, 118.92, 10.0),
            (56, 55, 66, 121.1, 10.0),
            (46, 57, 66, 121.1, 10.0),
            (46, 57, 58, 118.92, 10.0),
            (57, 58, 59, 118.88, 10.0),
            (0, 58, 59, 118.88, 10.0),
            (1, 0, 58, 118.92, 10.0),
            (1, 0, 11, 121.1, 10.0),
            (11, 10, 22, 121.1, 10.0),
            (9, 10, 22, 118.92, 10.0),
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
            (34, 33, 44, 116.54, 10.0),
            (24, 33, 34, 117.85, 10.0),
            (25, 24, 33, 121.83, 10.0),
            (23, 24, 25, 120.09, 10.0),
            (13, 23, 24, 119.2, 10.0),
            (13, 23, 32, 123.18, 10.0),
            (22, 21, 32, 119.72, 10.0),
            (20, 21, 22, 118.55, 10.0),
            (19, 20, 21, 118.91, 10.0),
            (19, 20, 30, 118.91, 10.0),
            (20, 30, 29, 118.55, 10.0),
            (29, 30, 31, 119.72, 10.0),
            (31, 41, 40, 123.18, 10.0),
            (40, 41, 42, 119.2, 10.0),
            (41, 42, 53, 120.09, 10.0),
            (43, 42, 53, 121.83, 10.0),
            (42, 43, 56, 117.85, 10.0),
            (44, 43, 56, 116.54, 10.0),
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
            (46, 47, 60, 121.99, 10.0),
            (46, 47, 48, 122.51, 10.0),
            (35, 48, 47, 115.67, 10.0),
            (35, 48, 49, 126.09, 10.0),
            (38, 49, 48, 111.08, 10.0),
            (38, 49, 50, 120.63, 10.0),
            (50, 51, 52, 131.0, 10.0),
            (52, 51, 63, 116.21, 10.0),
            (51, 63, 64, 124.82, 10.0),
            (62, 63, 64, 124.82, 10.0),
            (7, 62, 63, 116.21, 10.0),
            (7, 62, 61, 131.0, 10.0),
            (5, 4, 61, 120.63, 10.0),
            (3, 4, 5, 111.08, 10.0),
            (2, 3, 4, 126.09, 10.0),
            (2, 3, 59, 115.67, 10.0),
            (3, 59, 58, 122.51, 10.0),
            (58, 59, 60, 121.99, 10.0),
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
