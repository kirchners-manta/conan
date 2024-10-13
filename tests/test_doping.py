import warnings

# import numpy as np
import pytest
from ase.io import read

from conan.playground.doping_experiment import GrapheneSheet, NitrogenSpecies


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
    optimized_positions = atoms.get_positions()
    elements = atoms.get_chemical_symbols()
    return optimized_positions, elements


@pytest.fixture
def graphene_sheet():
    """
    Fixture to create a GrapheneSheet instance with predefined parameters.
    """
    return GrapheneSheet(bond_length=1.42, sheet_size=(10, 10))


@pytest.fixture
def optimized_structure():
    """
    Fixture to load the optimized structure from the .xyz file.

    Returns
    -------
    optimized_positions : np.ndarray
        The positions from the optimized structure.
    elements : List[str]
        The element symbols from the optimized structure.
    """
    file_path = "../structures/optimized_structure.xyz"
    optimized_positions, elements = read_optimized_structure(file_path)
    return optimized_positions, elements


class TestDopingValidations:

    # def test_adjust_atom_positions(self, optimized_reference_structure):
    #     """
    #     Test that the adjusted atom positions closely match the optimized reference structure.
    #
    #     Parameters
    #     ----------
    #     optimized_reference_structure : tuple
    #         A tuple containing the optimized positions and element symbols obtained from force field optimization.
    #
    #     Raises
    #     ------
    #     AssertionError
    #         If the positions or elements do not match within acceptable tolerances.
    #     """
    #     # Unpack the optimized reference structure
    #     optimized_reference_positions, optimized_reference_elements = optimized_reference_structure
    #
    #     # Set the random seed for reproducibility
    #     random.seed(0)
    #
    #     # Create the graphene sheet with the same parameters as the optimized reference structure
    #     graphene_sheet = GrapheneSheet(bond_distance=1.42, sheet_size=(20, 20))
    #
    #     # Apply nitrogen doping with 10% total percentage
    #     graphene_sheet.add_nitrogen_doping(total_percentage=10, adjust_positions=True)
    #
    #     # Extract positions and elements from the optimized structure
    #     optimized_positions = np.array(
    #         [graphene_sheet.graph.nodes[node]["position"].to_tuple() for node in sorted(graphene_sheet.graph.nodes())]
    #     )
    #     optimized_elements = [
    #         graphene_sheet.graph.nodes[node]["element"] for node in sorted(graphene_sheet.graph.nodes())
    #     ]
    #
    #     # Ensure that the number of atoms matches
    #     assert len(optimized_positions) == len(optimized_reference_positions), "Number of atoms does not match."
    #
    #     # Compare elements
    #     assert optimized_elements == optimized_reference_elements, "Element symbols do not match."
    #
    #     # Compare positions with tolerances
    #     npt.assert_allclose(optimized_positions, optimized_reference_positions, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("invalid_total_percentage", ["invalid", [], {}])
    def test_total_percentage_type_error(self, graphene_sheet, invalid_total_percentage):
        """
        Test that a ValueError is raised when total_percentage is not a float or int.
        """
        with pytest.raises(ValueError, match="total_percentage must be an int or float"):
            graphene_sheet.add_nitrogen_doping(total_percentage=invalid_total_percentage)

    @pytest.mark.parametrize(
        "invalid_percentages, expected_message",
        [
            ("invalid", "percentages must be a dictionary with NitrogenSpecies as keys and int or float as values."),
            # Not a dictionary
            (
                {"InvalidKey": 50},
                "Invalid key in percentages dictionary: InvalidKey. Keys must be of type NitrogenSpecies.",
            ),
            # Invalid key type
            (
                {NitrogenSpecies.GRAPHITIC: "invalid"},
                "Invalid value in percentages dictionary for key NitrogenSpecies.GRAPHITIC with value invalid. Values "
                "must be int or float.",
            ),
            # Invalid value type
        ],
    )
    def test_percentages_type_error(self, graphene_sheet, invalid_percentages, expected_message):
        """
        Test that a ValueError is raised when percentages is not a dictionary with NitrogenSpecies keys
        and int/float values.
        """
        with pytest.raises(ValueError, match=expected_message):
            graphene_sheet.add_nitrogen_doping(percentages=invalid_percentages)

    @pytest.mark.parametrize(
        "small_sheet_size, expected_warning",
        [
            ((5, 5), "The selected doping percentage is too low or the structure is too small to allow for doping."),
        ],
    )
    def test_doping_too_small(self, small_sheet_size, expected_warning):
        """
        Test that a warning is raised when the structure is too small to allow for doping.
        """
        graphene = GrapheneSheet(bond_length=1.42, sheet_size=small_sheet_size)
        with pytest.warns(UserWarning, match=expected_warning):
            graphene.add_nitrogen_doping(total_percentage=0.01)

    @pytest.mark.parametrize(
        "valid_total_percentage, valid_percentages",
        [
            (15, None),  # Valid total_percentage with no specific percentages
            (None, {NitrogenSpecies.GRAPHITIC: 50}),  # Valid percentages with no total_percentage
        ],
    )
    def test_doping_valid_input(self, graphene_sheet, valid_total_percentage, valid_percentages):
        """
        Test that add_nitrogen_doping works with valid inputs.
        """
        # Ignore UserWarnings for this test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                graphene_sheet.add_nitrogen_doping(
                    total_percentage=valid_total_percentage, percentages=valid_percentages
                )
            except ValueError:
                pytest.fail("add_nitrogen_doping raised ValueError unexpectedly!")

    def test_warning_when_not_all_atoms_placed(self, graphene_sheet):
        with pytest.warns(
            UserWarning, match=r"Only \d+ nitrogen atoms of species .* could be placed due to proximity constraints"
        ):
            graphene_sheet.add_nitrogen_doping(total_percentage=50)
