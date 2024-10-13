import numpy as np
import pytest

from conan.playground.doping_experiment import CNT


@pytest.fixture
def bond_length():
    """Fixture for bond length."""
    return 1.42


@pytest.fixture
def tube_length():
    """Fixture for tube length."""
    return 10.0


@pytest.mark.parametrize(
    "tube_size, conformation",
    [
        (8, "zigzag"),
        (10, "armchair"),
    ],
)
def test_cnt_with_tube_size(bond_length, tube_length, tube_size, conformation):
    """Test CNT initialization with tube_size specified."""
    cnt = CNT(
        bond_length=bond_length,
        tube_length=tube_length,
        tube_size=tube_size,
        conformation=conformation,
    )
    expected_diameter = cnt._calculate_tube_diameter_from_size(tube_size)
    assert cnt.actual_tube_diameter == pytest.approx(expected_diameter, abs=1e-3)
    assert cnt.tube_size == tube_size


@pytest.mark.parametrize(
    "tube_diameter, conformation",
    [
        (6.3, "zigzag"),
        (5.5, "armchair"),
    ],
)
def test_cnt_with_tube_diameter(bond_length, tube_length, tube_diameter, conformation):
    """Test CNT initialization with tube_diameter specified."""
    cnt = CNT(
        bond_length=bond_length,
        tube_length=tube_length,
        tube_diameter=tube_diameter,
        conformation=conformation,
    )
    calculated_size = cnt._calculate_tube_size_from_diameter(tube_diameter)
    expected_diameter = cnt._calculate_tube_diameter_from_size(calculated_size)
    assert cnt.tube_size == calculated_size
    assert cnt.actual_tube_diameter == pytest.approx(expected_diameter, abs=1e-3)
    # Check if the actual diameter is close to the provided diameter
    if not np.isclose(tube_diameter, expected_diameter, atol=1e-3):
        assert tube_diameter != expected_diameter


def test_cnt_with_both_tube_size_and_diameter(bond_length, tube_length):
    """Test CNT initialization with both tube_size and tube_diameter specified."""
    with pytest.raises(ValueError):
        CNT(
            bond_length=bond_length,
            tube_length=tube_length,
            tube_size=8,
            tube_diameter=6.3,
            conformation="zigzag",
        )


def test_cnt_with_neither_tube_size_nor_diameter(bond_length, tube_length):
    """Test CNT initialization with neither tube_size nor tube_diameter specified."""
    with pytest.raises(ValueError):
        CNT(
            bond_length=bond_length,
            tube_length=tube_length,
            conformation="zigzag",
        )


@pytest.mark.parametrize(
    "tube_diameter, conformation",
    [
        (6.0, "armchair"),
        (7.0, "zigzag"),
    ],
)
def test_tube_size_calculation_accuracy(bond_length, tube_length, tube_diameter, conformation):
    """Test the accuracy of tube_size calculation from tube_diameter."""
    cnt = CNT(
        bond_length=bond_length,
        tube_length=tube_length,
        tube_diameter=tube_diameter,
        conformation=conformation,
    )
    expected_size = cnt._calculate_tube_size_from_diameter(tube_diameter)
    assert cnt.tube_size == expected_size


@pytest.mark.parametrize(
    "tube_size, conformation",
    [
        (10, "armchair"),
        (7, "zigzag"),
    ],
)
def test_tube_diameter_calculation_accuracy(bond_length, tube_length, tube_size, conformation):
    """Test the accuracy of tube_diameter calculation from tube_size."""
    cnt = CNT(
        bond_length=bond_length,
        tube_length=tube_length,
        tube_size=tube_size,
        conformation=conformation,
    )
    expected_diameter = cnt._calculate_tube_diameter_from_size(tube_size)
    assert cnt.actual_tube_diameter == pytest.approx(expected_diameter, abs=1e-3)


@pytest.mark.parametrize(
    "invalid_bond_length, expected_exception",
    [
        ("1.42", TypeError),  # Not a float or int
        (-1.0, ValueError),  # Non-positive value
        (0, ValueError),  # Zero value
        (None, TypeError),  # NoneType
    ],
)
def test_invalid_bond_length(invalid_bond_length, expected_exception, tube_length):
    """Test invalid bond_length inputs for CNT."""
    with pytest.raises(expected_exception):
        CNT(
            bond_length=invalid_bond_length,
            tube_length=tube_length,
            tube_size=8,
            conformation="zigzag",
        )


@pytest.mark.parametrize(
    "invalid_tube_length, expected_exception",
    [
        ("10.0", TypeError),  # Not a float or int
        (-5.0, ValueError),  # Non-positive value
        (0, ValueError),  # Zero value
        (None, TypeError),  # NoneType
    ],
)
def test_invalid_tube_length(bond_length, invalid_tube_length, expected_exception):
    """Test invalid tube_length inputs for CNT."""
    with pytest.raises(expected_exception):
        CNT(
            bond_length=bond_length,
            tube_length=invalid_tube_length,
            tube_size=8,
            conformation="zigzag",
        )


@pytest.mark.parametrize(
    "invalid_tube_size, expected_exception",
    [
        ("8", TypeError),  # Not an int
        (-1, ValueError),  # Non-positive value
        (0, ValueError),  # Zero value
        (5.5, TypeError),  # Float instead of int
        (None, ValueError),  # NoneType
    ],
)
def test_invalid_tube_size(bond_length, tube_length, invalid_tube_size, expected_exception):
    """Test invalid tube_size inputs for CNT."""
    with pytest.raises(expected_exception):
        CNT(
            bond_length=bond_length,
            tube_length=tube_length,
            tube_size=invalid_tube_size,
            conformation="zigzag",
        )


@pytest.mark.parametrize(
    "invalid_tube_diameter, expected_exception",
    [
        ("6.3", TypeError),  # Not a float or int
        (-6.3, ValueError),  # Non-positive value
        (0, ValueError),  # Zero value
        (None, ValueError),  # NoneType
    ],
)
def test_invalid_tube_diameter(bond_length, tube_length, invalid_tube_diameter, expected_exception):
    """Test invalid tube_diameter inputs for CNT."""
    with pytest.raises(expected_exception):
        CNT(
            bond_length=bond_length,
            tube_length=tube_length,
            tube_diameter=invalid_tube_diameter,
            conformation="zigzag",
        )


@pytest.mark.parametrize(
    "invalid_conformation, expected_exception",
    [
        (123, TypeError),  # Not a string
        (None, TypeError),  # NoneType
        ("invalid_conformation", ValueError),  # Invalid value
    ],
)
def test_invalid_conformation(bond_length, tube_length, invalid_conformation, expected_exception):
    """Test invalid conformation inputs for CNT."""
    with pytest.raises(expected_exception):
        CNT(
            bond_length=bond_length,
            tube_length=tube_length,
            tube_size=8,
            conformation=invalid_conformation,
        )


@pytest.mark.parametrize(
    "invalid_periodic, expected_exception",
    [
        ("False", TypeError),  # Not a boolean
        (1, TypeError),  # Not a boolean
        (None, TypeError),  # NoneType
    ],
)
def test_invalid_periodic(bond_length, tube_length, invalid_periodic, expected_exception):
    """Test invalid periodic inputs for CNT."""
    with pytest.raises(expected_exception):
        CNT(
            bond_length=bond_length,
            tube_length=tube_length,
            tube_size=8,
            conformation="zigzag",
            periodic=invalid_periodic,
        )
