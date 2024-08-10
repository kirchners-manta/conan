import pytest

from conan.playground.doping_experiment import GrapheneSheet


class TestGrapheneValidations:

    def test_bond_distance_type_error(self):
        """
        Test that a TypeError is raised when bond_distance is not a float or int.
        """
        with pytest.raises(TypeError, match=r"bond_distance must be a float or int, but got str."):
            GrapheneSheet(bond_distance="invalid", sheet_size=(20, 20))

    def test_bond_distance_value_error(self):
        """
        Test that a ValueError is raised when bond_distance is not positive.
        """
        with pytest.raises(ValueError, match=r"bond_distance must be positive, but got -1.42."):
            GrapheneSheet(bond_distance=-1.42, sheet_size=(20, 20))

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
        with pytest.raises(TypeError, match=r"sheet_size must be a tuple of two positive floats or ints."):
            GrapheneSheet(bond_distance=1.42, sheet_size=invalid_sheet_size)

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
            GrapheneSheet(bond_distance=1.42, sheet_size=invalid_sheet_size)

    def test_invalid_structure(self):
        """
        Test that a ValueError is raised when the sheet size is too small to fit a unit cell.
        """
        with pytest.raises(ValueError, match=r"Sheet size is too small to fit even a single unit cell."):
            GrapheneSheet(bond_distance=1.42, sheet_size=(0.1, 0.1))
