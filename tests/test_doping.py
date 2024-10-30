import warnings

import pytest
from ase.io import read

from conan.playground.doping import NitrogenSpecies
from conan.playground.structures import GrapheneSheet


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

    @pytest.mark.parametrize("invalid_total_percentage", ["invalid", [], {}])
    def test_total_percentage_type_error(self, graphene_sheet, invalid_total_percentage):
        """
        Test that a ValueError is raised when total_percentage is not a float or int.
        """
        with pytest.raises(ValueError, match="total_percentage must be an int or float"):
            graphene_sheet.add_nitrogen_doping(total_percentage=invalid_total_percentage)

    @pytest.mark.parametrize("invalid_total_percentage", [-10, 0])
    def test_invalid_total_percentage(self, graphene_sheet, invalid_total_percentage):
        """
        Test that a ValueError is raised when total_percentage is non-positive.
        """
        with pytest.raises(ValueError, match="total_percentage must be positive."):
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
            (
                {NitrogenSpecies.GRAPHITIC: -10},
                "Percentage for NitrogenSpecies.GRAPHITIC must be positive. Received -10.",
            ),  # Negative value
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

    def test_empty_percentages_dict(self, graphene_sheet):
        """
        Test that a ValueError is raised when percentages is an empty dictionary.
        """
        with pytest.raises(
            ValueError, match="percentages dictionary cannot be empty. Define at least one positive percentage."
        ):
            graphene_sheet.add_nitrogen_doping(total_percentage=10, percentages={})

    def test_warning_for_incomplete_percentages(self, graphene_sheet):
        """
        Test that a UserWarning is raised when the sum of specific percentages is lower than total_percentage.
        """
        with pytest.warns(UserWarning, match="Remaining percentage will be distributed among other available species."):
            graphene_sheet.add_nitrogen_doping(total_percentage=10, percentages={NitrogenSpecies.GRAPHITIC: 3})

    # ToDo: Dieser Test muss dringend nochmal getestet werden, sobald Problem mit prozentualem Dotierungsgehalt gelöst
    # def test_remaining_percentage_distribution(self, graphene_sheet):
    #     """
    #     Test that the remaining percentage is correctly distributed among other available species.
    #     """
    #     # Set total_percentage to 10% with 3% specified for GRAPHITIC nitrogen.
    #     graphene_sheet.add_nitrogen_doping(total_percentage=10, percentages={NitrogenSpecies.GRAPHITIC: 3})
    #
    #     # Count the number of nitrogen atoms of each species in the structure.
    #     total_atoms = graphene_sheet.graph.number_of_nodes()
    #     nitrogen_counts = {species: 0 for species in NitrogenSpecies}
    #
    #     for node in graphene_sheet.graph.nodes(data=True):
    #         element = node[1].get("element")
    #         nitrogen_species = node[1].get("nitrogen_species")
    #
    #         if element == "N" and nitrogen_species in NitrogenSpecies:
    #             nitrogen_counts[nitrogen_species] += 1
    #
    #     # Calculate actual percentages based on the number of nitrogen atoms.
    #     actual_distribution = {
    #         species: round((count / total_atoms) * 100, 2) for species, count in nitrogen_counts.items()
    #     }
    #
    #     # Check that the GRAPHITIC nitrogen percentage is 3% as specified.
    #     assert actual_distribution[NitrogenSpecies.GRAPHITIC] == 3, "Expected GRAPHITIC doping percentage to be 3%."
    #
    #     # Calculate the remaining percentage and expected distribution per remaining species.
    #     remaining_species = set(NitrogenSpecies) - {NitrogenSpecies.GRAPHITIC}
    #     remaining_percentage = 10 - actual_distribution[NitrogenSpecies.GRAPHITIC]
    #     expected_remaining_percentage_per_species = remaining_percentage / len(remaining_species)
    #
    #     # Check that each remaining species has an approximately equal share of the remaining percentage.
    #     for species in remaining_species:
    #         assert actual_distribution[species] == pytest.approx(expected_remaining_percentage_per_species, 0.01), \
    #             f"Expected {expected_remaining_percentage_per_species}% for {species}, but got
    #             {actual_distribution[species]}."
