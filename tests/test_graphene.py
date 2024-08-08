import random

import numpy as np
import pytest

from conan.playground.doping_experiment import GrapheneSheet
from conan.playground.graph_utils import get_neighbors_via_edges, get_neighbors_within_distance


class TestGraphene:
    @pytest.fixture
    def graphene(self):
        """
        Fixture to set up a GrapheneGraph instance with a fixed random seed for reproducibility.

        Returns
        -------
        GrapheneSheet
            An instance of the GrapheneGraph class with a predefined sheet size and bond distance.
        """
        # Set seed for reproducibility
        random.seed(42)
        graphene = GrapheneSheet(bond_distance=1.42, sheet_size=(20, 20))
        return graphene

    def test_get_direct_neighbors_via_bonds(self, graphene: GrapheneSheet):
        """
        Test to verify the direct connected neighbors of an atom while considering periodic boundary conditions.

        Parameters
        ----------
        graphene : GrapheneSheet
            The graphene fixture providing the initialized GrapheneGraph instance.

        Asserts
        -------
        Checks if the direct connected neighbors of the atom with ID 0 match the expected neighbors.
        """
        direct_neighbors = get_neighbors_via_edges(graphene.graph, atom_id=0, depth=1)
        expected_neighbors = get_neighbors_via_edges(graphene.graph, 0)
        assert set(direct_neighbors) == set(
            expected_neighbors
        ), f"Expected {expected_neighbors}, but got {direct_neighbors}"

    def test_get_neighbors_at_exact_depth_via_bonds(self, graphene: GrapheneSheet):
        """
        Test to verify the connected neighbors of an atom at an exact depth while considering periodic boundary
        conditions.

        Parameters
        ----------
        graphene : GrapheneSheet
            The graphene fixture providing the initialized GrapheneGraph instance.

        Asserts
        -------
        Checks if the connected neighbors of the atom with ID 0 at depth 2 match the expected neighbors.
        """
        depth_neighbors = get_neighbors_via_edges(graphene.graph, atom_id=0, depth=2, inclusive=False)
        expected_neighbors = [2, 16, 14, 126, 112, 114]
        assert set(depth_neighbors) == set(
            expected_neighbors
        ), f"Expected {expected_neighbors}, but got {depth_neighbors}"

    def test_get_neighbors_up_to_depth_via_bonds(self, graphene: GrapheneSheet):
        """
        Test to verify the connected neighbors of an atom up to a certain depth while considering periodic boundary
        conditions.

        Parameters
        ----------
        graphene : GrapheneSheet
            The graphene fixture providing the initialized GrapheneGraph instance.

        Asserts
        -------
        Checks if the connected neighbors of the atom with ID 0 up to depth 2 (inclusive) match the expected neighbors.
        """
        graphene = graphene
        inclusive_neighbors = get_neighbors_via_edges(graphene.graph, atom_id=0, depth=2, inclusive=True)
        expected_neighbors = [1, 15, 113, 2, 16, 14, 126, 112, 114]
        assert set(inclusive_neighbors) == set(
            expected_neighbors
        ), f"Expected {expected_neighbors}, but got {inclusive_neighbors}"

    def test_get_neighbors_within_distance(self, graphene: GrapheneSheet):
        """
        Test to verify neighbors within a given distance using KDTree.

        Parameters
        ----------
        graphene : GrapheneSheet
            The graphene fixture providing the initialized GrapheneGraph instance.

        Asserts
        -------
        Checks if the neighbors within a specified distance are correct by calculating the Euclidean distance.
        """
        atom_id = 56
        max_distance = 5
        neighbors = get_neighbors_within_distance(graphene.graph, graphene.kdtree, atom_id, max_distance)

        atom_position = np.array(graphene.graph.nodes[atom_id]["position"])
        for neighbor in neighbors:
            neighbor_position = np.array(graphene.graph.nodes[neighbor]["position"])
            distance = np.linalg.norm(atom_position - neighbor_position)
            assert distance <= max_distance, (
                f"Neighbor {neighbor} at distance {distance} exceeds max_distance " f"{max_distance}"
            )
