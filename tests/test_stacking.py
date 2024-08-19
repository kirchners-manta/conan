import pytest

from conan.playground.doping_experiment import GrapheneSheet, StackedGraphene


@pytest.fixture
def graphene_sheet():
    """
    Fixture to create a GrapheneSheet instance with predefined parameters.
    """
    return GrapheneSheet(bond_distance=1.42, sheet_size=(10, 10))


class TestStackedGrapheneValidations:

    @pytest.mark.parametrize("invalid_spacing", ["invalid", [], {}, -1, 0])
    def test_interlayer_spacing_type_and_value_error(self, graphene_sheet, invalid_spacing):
        """
        Test that a ValueError is raised when interlayer_spacing is not a positive float or int.
        """
        with pytest.raises(ValueError, match="interlayer_spacing must be positive number."):
            StackedGraphene(graphene_sheet, interlayer_spacing=invalid_spacing, number_of_layers=3, stacking_type="ABA")

    @pytest.mark.parametrize("invalid_layers", ["invalid", [], {}, -1, 0, 1.5])
    def test_number_of_layers_type_and_value_error(self, graphene_sheet, invalid_layers):
        """
        Test that a ValueError is raised when number_of_layers is not a positive integer.
        """
        with pytest.raises(ValueError, match="number_of_layers must be a positive integer"):
            StackedGraphene(
                graphene_sheet, interlayer_spacing=3.34, number_of_layers=invalid_layers, stacking_type="ABA"
            )

    @pytest.mark.parametrize("invalid_stacking_type", ["invalid", 123, [], {}, None])
    def test_stacking_type_value_error(graphene_sheet, invalid_stacking_type):
        """
        Test that a ValueError is raised when stacking_type is not 'ABA' or 'ABC'.
        """
        if isinstance(invalid_stacking_type, str):
            expected_message = r"stacking_type must be one of \{'ABC', 'ABA'\}, but got"
        else:
            expected_message = "stacking_type must be a string"

        with pytest.raises(ValueError, match=expected_message):
            StackedGraphene(
                graphene_sheet, interlayer_spacing=3.34, number_of_layers=3, stacking_type=invalid_stacking_type
            )

    def test_valid_stacking(self, graphene_sheet):
        """
        Test that a valid stacking operation works without errors.
        """
        try:
            stacked_graphene = StackedGraphene(
                graphene_sheet, interlayer_spacing=3.34, number_of_layers=3, stacking_type="ABC"
            )
            assert stacked_graphene.number_of_layers == 3
            assert stacked_graphene.interlayer_spacing == 3.34
            assert stacked_graphene.stacking_type == "ABC"
        except ValueError:
            pytest.fail("StackedGraphene raised ValueError unexpectedly!")


if __name__ == "__main__":
    pytest.main()
