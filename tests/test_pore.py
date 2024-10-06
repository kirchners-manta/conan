import numpy as np
import pytest

from conan.playground.doping_experiment import Pore


@pytest.fixture
def bond_length():
    """Fixture for bond length."""
    return 1.42


@pytest.fixture
def sheet_size():
    """Fixture for sheet size."""
    return 20.0, 20.0


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
def test_pore_with_tube_size(bond_length, sheet_size, tube_length, tube_size, conformation):
    """Test Pore initialization with tube_size specified."""
    pore = Pore(
        bond_length=bond_length,
        sheet_size=sheet_size,
        tube_length=tube_length,
        tube_size=tube_size,
        conformation=conformation,
    )
    expected_diameter = pore.cnt._calculate_tube_diameter_from_size(tube_size)
    assert pore.cnt.actual_tube_diameter == pytest.approx(expected_diameter, abs=1e-3)
    assert pore.cnt.tube_size == tube_size


@pytest.mark.parametrize(
    "tube_diameter, conformation",
    [
        (6.3, "zigzag"),
        (5.5, "armchair"),
    ],
)
def test_pore_with_tube_diameter(bond_length, sheet_size, tube_length, tube_diameter, conformation):
    """Test Pore initialization with tube_diameter specified."""
    pore = Pore(
        bond_length=bond_length,
        sheet_size=sheet_size,
        tube_length=tube_length,
        tube_diameter=tube_diameter,
        conformation=conformation,
    )
    calculated_size = pore.cnt._calculate_tube_size_from_diameter(tube_diameter)
    expected_diameter = pore.cnt._calculate_tube_diameter_from_size(calculated_size)
    assert pore.cnt.tube_size == calculated_size
    assert pore.cnt.actual_tube_diameter == pytest.approx(expected_diameter, abs=1e-3)
    # Check if the actual diameter is close to the provided diameter
    if not np.isclose(tube_diameter, expected_diameter, atol=1e-3):
        assert tube_diameter != expected_diameter


def test_pore_with_both_tube_size_and_diameter(bond_length, sheet_size, tube_length):
    """Test Pore initialization with both tube_size and tube_diameter specified."""
    with pytest.raises(ValueError):
        Pore(
            bond_length=bond_length,
            sheet_size=sheet_size,
            tube_length=tube_length,
            tube_size=8,
            tube_diameter=6.3,
            conformation="zigzag",
        )


def test_pore_with_neither_tube_size_nor_diameter(bond_length, sheet_size, tube_length):
    """Test Pore initialization with neither tube_size nor tube_diameter specified."""
    with pytest.raises(ValueError):
        Pore(
            bond_length=bond_length,
            sheet_size=sheet_size,
            tube_length=tube_length,
            conformation="zigzag",
        )


def test_pore_invalid_conformation(bond_length, sheet_size, tube_length):
    """Test Pore initialization with invalid conformation."""
    with pytest.raises(ValueError):
        Pore(
            bond_length=bond_length,
            sheet_size=sheet_size,
            tube_length=tube_length,
            tube_size=8,
            conformation="invalid_conformation",
        )
