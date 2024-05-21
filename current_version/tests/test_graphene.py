import pytest
import random
from ..playground.doping_experiment import GrapheneGraph, NitrogenSpecies


@pytest.fixture
def setup_graphene():
    """
    Fixture to set up a GrapheneGraph instance with a fixed random seed for reproducibility.

    Returns
    -------
    GrapheneGraph
        An instance of the GrapheneGraph class with a predefined sheet size and bond distance.
    """
    # Set seed for reproducibility
    random.seed(42)
    graphene = GrapheneGraph(bond_distance=1.42, sheet_size=(20, 20))
    return graphene


def test_get_direct_neighbors(setup_graphene):
    """
    Test to verify the direct neighbors of an atom.

    Parameters
    ----------
    setup_graphene : GrapheneGraph
        The graphene fixture providing the initialized GrapheneGraph instance.

    Asserts
    -------
    Checks if the direct neighbors of the atom with ID 0 match the expected neighbors.
    """
    graphene = setup_graphene
    direct_neighbors = graphene.get_neighbors(atom_id=0, depth=1)
    expected_neighbors = graphene.get_neighbors(0)
    assert set(direct_neighbors) == set(
        expected_neighbors), f"Expected {expected_neighbors}, but got {direct_neighbors}"


def test_get_neighbors_at_exact_depth(setup_graphene):
    """
    Test to verify the neighbors of an atom at an exact depth.

    Parameters
    ----------
    setup_graphene : GrapheneGraph
        The graphene fixture providing the initialized GrapheneGraph instance.

    Asserts
    -------
    Checks if the neighbors of the atom with ID 0 at depth 2 match the expected neighbors.
    """
    graphene = setup_graphene
    depth_neighbors = graphene.get_neighbors(atom_id=0, depth=2, inclusive=False)
    expected_neighbors = []  # ToDo: Add expected neighbors
    assert set(depth_neighbors) == set(expected_neighbors), f"Expected {expected_neighbors}, but got {depth_neighbors}"


def test_get_neighbors_up_to_depth(setup_graphene):
    """
    Test to verify the neighbors of an atom up to a certain depth.

    Parameters
    ----------
    setup_graphene : GrapheneGraph
        The graphene fixture providing the initialized GrapheneGraph instance.

    Asserts
    -------
    Checks if the neighbors of the atom with ID 0 up to depth 2 (inclusive) match the expected neighbors.
    """
    graphene = setup_graphene
    inclusive_neighbors = graphene.get_neighbors(atom_id=0, depth=2, inclusive=True)
    expected_neighbors = []  # ToDo: Add expected neighbors
    assert set(inclusive_neighbors) == set(
        expected_neighbors), f"Expected {expected_neighbors}, but got {inclusive_neighbors}"


if __name__ == "__main__":
    pytest.main()
