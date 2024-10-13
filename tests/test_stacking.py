import pytest

from conan.playground.doping_experiment import GrapheneSheet, StackedGraphene


@pytest.fixture
def graphene_sheet():
    """
    Fixture to create a GrapheneSheet instance with predefined parameters.
    """
    return GrapheneSheet(bond_length=1.42, sheet_size=(20, 20))


@pytest.fixture
def stacked_graphene(graphene_sheet):
    """
    Fixture to create a StackedGraphene instance with predefined parameters.
    """
    return StackedGraphene(graphene_sheet, interlayer_spacing=3.34, number_of_layers=3, stacking_type="ABA")


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
    def test_stacking_type_value_error(self, graphene_sheet, invalid_stacking_type):
        """
        Test that a ValueError is raised when stacking_type is not 'ABA' or 'ABC'.
        """
        if isinstance(invalid_stacking_type, str):
            # Two possible regex patterns to match either 'ABA', 'ABC' or 'ABC', 'ABA' order
            possible_patterns = [
                r"stacking_type must be one of \{'ABA', 'ABC'\}, but got",
                r"stacking_type must be one of \{'ABC', 'ABA'\}, but got",
            ]
        else:
            possible_patterns = ["stacking_type must be a string"]

        for pattern in possible_patterns:
            try:
                with pytest.raises(ValueError, match=pattern):
                    StackedGraphene(
                        graphene_sheet, interlayer_spacing=3.34, number_of_layers=3, stacking_type=invalid_stacking_type
                    )
                return  # If a pattern matches, exit the test successfully
            except AssertionError:
                continue  # If the pattern doesn't match, try the next one

        # If no pattern matched, fail the test
        pytest.fail(f"No matching pattern found for the error message. Expected patterns: {possible_patterns}")

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

    @pytest.mark.parametrize("invalid_layers", ["invalid", -1, [0, 10], 1.5])
    def test_invalid_layers(self, stacked_graphene, invalid_layers):
        """
        Test that a ValueError or IndexError is raised for invalid layers input in add_nitrogen_doping.
        """
        if isinstance(invalid_layers, list) and any(
            layer >= stacked_graphene.number_of_layers for layer in invalid_layers
        ):
            with pytest.raises(IndexError, match="One or more specified layers are out of range."):
                stacked_graphene.add_nitrogen_doping(layers=invalid_layers)
        else:
            with pytest.raises(ValueError, match="Invalid 'layers' parameter. Must be a list of integers or 'all'."):
                stacked_graphene.add_nitrogen_doping(layers=invalid_layers)

    # @pytest.mark.parametrize("invalid_total_percentage", ["invalid", -10, 150])
    # def test_invalid_total_percentage(self, stacked_graphene, invalid_total_percentage):
    #     """
    #     Test that a ValueError is raised for invalid total_percentage input in add_nitrogen_doping.
    #     """
    #     with pytest.raises(ValueError, match="total_percentage must be a positive number less than or equal to 100."):
    #         stacked_graphene.add_nitrogen_doping(total_percentage=invalid_total_percentage)
    #
    # @pytest.mark.parametrize("invalid_percentages", [
    #     {"PYRIDINIC_1": "invalid"},  # Non-numeric value
    #     {"PYRIDINIC_1": -5},  # Negative percentage
    #     {"PYRIDINIC_1": 110},  # Percentage greater than 100
    # ])
    # def test_invalid_percentages(self, stacked_graphene, invalid_percentages):
    #     """
    #     Test that a ValueError is raised for invalid percentages input in add_nitrogen_doping.
    #     """
    #     with pytest.raises(ValueError,
    #                        match="Each percentage in 'percentages' must be a positive number less than or equal to
    #                        100."):
    #         stacked_graphene.add_nitrogen_doping(percentages=invalid_percentages)

    def test_valid_doping(self, stacked_graphene):
        """
        Test that valid nitrogen doping operation works without errors.
        """
        try:
            stacked_graphene.add_nitrogen_doping(total_percentage=10)
            # No assertions are made since we are only testing for the absence of exceptions
        except (ValueError, IndexError):
            pytest.fail("add_nitrogen_doping raised an error unexpectedly!")


if __name__ == "__main__":
    pytest.main()
