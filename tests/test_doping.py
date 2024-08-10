import pytest

from conan.playground.doping_experiment import GrapheneSheet, NitrogenSpecies


@pytest.fixture
def graphene_sheet():
    """
    Fixture to create a GrapheneSheet instance with predefined parameters.
    """
    return GrapheneSheet(bond_distance=1.42, sheet_size=(10, 10))


class TestDopingValidations:

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
                "Invalid value in percentages dictionary for key NitrogenSpecies.GRAPHITIC: invalid. Values must be "
                "int or float.",
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
        graphene = GrapheneSheet(bond_distance=1.42, sheet_size=small_sheet_size)
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
        try:
            graphene_sheet.add_nitrogen_doping(total_percentage=valid_total_percentage, percentages=valid_percentages)
        except ValueError:
            pytest.fail("add_nitrogen_doping raised ValueError unexpectedly!")

    def test_warning_when_not_all_atoms_placed(self, graphene_sheet):
        with pytest.warns(
            UserWarning, match=r"Only \d+ nitrogen atoms of species .* could be placed due to proximity constraints"
        ):
            graphene_sheet.add_nitrogen_doping(total_percentage=50)
