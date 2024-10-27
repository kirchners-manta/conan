import random

import pytest

from conan.playground.structures import GrapheneSheet
from conan.playground.utils import get_neighbors_via_edges


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
        graphene = GrapheneSheet(bond_length=1.42, sheet_size=(20, 20))
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

    # def test_get_neighbors_within_distance(self, graphene: GrapheneSheet):
    #     """
    #     Test to verify neighbors within a given distance using KDTree.
    #
    #     Parameters
    #     ----------
    #     graphene : GrapheneSheet
    #         The graphene fixture providing the initialized GrapheneGraph instance.
    #
    #     Asserts
    #     -------
    #     Checks if the neighbors within a specified distance are correct by calculating the Euclidean distance.
    #     """
    #     atom_id = 56
    #     max_distance = 5
    #     neighbors = get_neighbors_within_distance(graphene.graph, graphene.kdtree, atom_id, max_distance)
    #
    #     atom_position = np.array(graphene.graph.nodes[atom_id]["position"])
    #     for neighbor in neighbors:
    #         neighbor_position = np.array(graphene.graph.nodes[neighbor]["position"])
    #         distance = np.linalg.norm(atom_position - neighbor_position)
    #         assert distance <= max_distance, (
    #             f"Neighbor {neighbor} at distance {distance} exceeds max_distance " f"{max_distance}"
    #         )


class TestGrapheneValidations:

    def test_bond_distance_type_error(self):
        """
        Test that a TypeError is raised when bond_length is not a float or int.
        """
        with pytest.raises(TypeError, match=r"bond_length must be a float or int, but got str."):
            GrapheneSheet(bond_length="invalid", sheet_size=(20, 20))

    @pytest.mark.parametrize("invalid_bond_distance", [-1.42, 0])
    def test_bond_distance_value_error(self, invalid_bond_distance):
        """
        Test that a ValueError is raised when bond_length is not positive.
        """
        with pytest.raises(ValueError, match=rf"bond_length must be positive, but got {invalid_bond_distance}."):
            GrapheneSheet(bond_length=invalid_bond_distance, sheet_size=(20, 20))

    @pytest.mark.parametrize(
        "invalid_sheet_size",
        [
            "invalid",  # Not a tuple and not a number
            (20, "invalid"),  # Second element is not a number
            (20, 20, 20),  # Too many elements in the tuple
            20,  # Not a tuple
        ],
    )
    def test_sheet_size_type_error(self, invalid_sheet_size):
        """
        Test that a TypeError is raised when sheet_size is not a tuple of two positive floats or ints.

        Parameters
        ----------
        invalid_sheet_size : various
            Different invalid values for sheet_size that should trigger a TypeError.
        """
        with pytest.raises(TypeError, match=r"sheet_size must be a tuple of exactly two positive floats or ints."):
            GrapheneSheet(bond_length=1.42, sheet_size=invalid_sheet_size)

    @pytest.mark.parametrize(
        "invalid_sheet_size, expected_message",
        [
            ((-20, 20), r"All elements of sheet_size must be positive, but got \(-20, 20\)."),
            ((20, 0), r"All elements of sheet_size must be positive, but got \(20, 0\)."),
        ],
    )
    def test_sheet_size_value_error(self, invalid_sheet_size, expected_message):
        """
        Test that a ValueError is raised when sheet_size has non-positive values.

        Parameters
        ----------
        invalid_sheet_size : tuple
            Invalid sheet_size values to test.
        expected_message : str
            The expected error message.
        """
        with pytest.raises(ValueError, match=expected_message):
            GrapheneSheet(bond_length=1.42, sheet_size=invalid_sheet_size)

    def test_invalid_structure(self):
        """
        Test that a ValueError is raised when the sheet size is too small to fit a unit cell.
        """
        with pytest.raises(ValueError, match=r"Sheet size is too small to fit even a single unit cell."):
            GrapheneSheet(bond_length=1.42, sheet_size=(0.1, 0.1))


class TestGrapheneAdjustAtomPositions:

    def test_adjust_atom_positions_invalid_config_type(self):
        """
        Test that a TypeError is raised when optimization_config is not an instance of OptimizationConfig.
        """
        graphene = GrapheneSheet(bond_length=1.42, sheet_size=(20, 20))

        with pytest.raises(TypeError, match="optimization_config must be an instance of OptimizationConfig."):
            graphene.adjust_atom_positions(optimization_config="invalid_config")

    def test_adjust_atom_positions_with_default_config(self):
        """
        Test that positions can be adjusted using the default optimization config.
        """
        graphene = GrapheneSheet(bond_length=1.42, sheet_size=(20, 20))
        graphene.adjust_atom_positions()  # Should not raise any exceptions

    def test_adjust_atom_positions_repeated_call(self):
        """
        Test that a warning is issued if adjust_atom_positions is called twice.
        """
        graphene = GrapheneSheet(bond_length=1.42, sheet_size=(20, 20))
        graphene.adjust_atom_positions()

        with pytest.warns(UserWarning, match="Positions have already been adjusted."):
            graphene.adjust_atom_positions()
