import random
import warnings

import pytest

from conan.playground.doping import DopingStructure, DopingStructureCollection, NitrogenSpecies, StructuralComponents
from conan.playground.labeling import AtomLabeler
from conan.playground.structures import GrapheneSheet


@pytest.fixture
def mock_graph():
    """
    Fixture for creating a 20x20 graphene sheet structure graph with a fixed random seed.
    """
    random.seed(100)
    sheet_size = (20, 20)
    graphene = GrapheneSheet(bond_length=1.42, sheet_size=sheet_size)

    # Temporarily ignore warnings for this setup
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        graphene.add_nitrogen_doping(percentages={NitrogenSpecies.PYRIDINIC_4: 1})

    return graphene.graph


@pytest.fixture
def mock_doping_structures(mock_graph):
    """
    Fixture for creating a mock DopingStructureCollection with one Pyridinic-N 4 doping structure.
    """
    doping_structures = DopingStructureCollection()

    # Create example Pyridinic-N 4 doping structures
    pyridinic_4_doping_1 = DopingStructure(
        species=NitrogenSpecies.PYRIDINIC_4,
        structural_components=StructuralComponents([37, 38], [52, 55, 36, 39]),
        nitrogen_atoms=[52, 55, 36, 39],
        cycle=[34, 51, 52, 53, 54, 55, 56, 41, 40, 39, 22, 21, 36, 35],
        neighboring_atoms=[33, 50, 68, 71, 57, 42, 25, 23, 20, 18],
        subgraph=mock_graph.subgraph([34, 51, 52, 53, 54, 55, 56, 41, 40, 39, 22, 21, 36, 35]).copy(),
    )

    # Add doping structures to the collection
    doping_structures.add_structure(pyridinic_4_doping_1)

    return doping_structures


class TestAtomLabeler:

    # ToDo: Test labeling procedure

    def test_label_atoms_without_doping(self, mock_graph):
        """
        Tests if all atoms are labeled as 'CG' when no doping structures are present.
        """
        labeler = AtomLabeler(mock_graph)
        labeler.label_atoms()

        for node in mock_graph.nodes:
            assert mock_graph.nodes[node]["label"] == "CG", f"Atom {node} should be labeled as 'CG'."

    def test_invalid_graph_type(self):
        """
        Tests if an error is raised when an invalid type is passed for `graph`.
        """
        with pytest.raises(TypeError, match="Expected graph to be a networkx Graph instance"):
            AtomLabeler(graph="invalid_graph")

    def test_invalid_doping_structure_type(self, mock_graph):
        """
        Tests if an error is raised when an invalid type is passed for `doping_structures`.
        """
        with pytest.raises(
            TypeError, match="Expected doping_structures to be a DopingStructureCollection instance or None"
        ):
            AtomLabeler(graph=mock_graph, doping_structures="invalid_doping_structures")

    def test_label_atoms_with_no_label_key(self, mock_graph):
        """
        Tests if the AtomLabeler correctly adds labels to atoms when the graph previously had no 'label' keys.
        """
        # Ensure no labels are present before labeling
        for node in mock_graph.nodes:
            assert "label" not in mock_graph.nodes[node], f"Atom {node} should not have a label."

        labeler = AtomLabeler(mock_graph)
        labeler.label_atoms()

        # Check if labels are added correctly
        for node in mock_graph.nodes:
            assert mock_graph.nodes[node]["label"] == "CG", f"Atom {node} should be labeled as 'CG'."
