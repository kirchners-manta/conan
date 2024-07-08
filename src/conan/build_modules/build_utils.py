from typing import List, Tuple, Union

import numpy as np
import pandas as pd


def rotate_3d_vector(vec: np.ndarray, rotational_axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotates a vector around a specified rotational axis by a given angle using
    Rodrigues' rotation formula.

    Parameters
    ----------
    vec : np.ndarray
        The vector to be rotated.
    rotational_axis : np.ndarray
        The axis around which the vector `vec` is rotated. Should be a 3D vector.
    angle : float
        The rotation angle in radians.

    Returns
    -------
    np.ndarray
        The rotated vector.
    """
    # Ensure the inputs are numpy arrays
    vec = np.array(vec)

    # Calculate the rotated vector using Rodrigues' rotation formula
    return (
        vec * np.cos(angle)
        + np.cross(rotational_axis, vec) * np.sin(angle)
        + rotational_axis * np.dot(rotational_axis, vec) * (1 - np.cos(angle))
    )


def center_position(sheet_size: Tuple[float, float], atoms_df: pd.DataFrame) -> pd.Series:
    """
    Returns the coordinates of the atom that is closest to the sheet center.

    Parameters
    ----------
    sheet_size : Tuple[float, float]
        The size of the sheet as a tuple of two floats (width, height).
    atoms_df : pd.DataFrame
        The DataFrame containing the atom coordinates. Assumes the DataFrame
        has columns where the first column is an identifier and the next two
        columns are the x and y coordinates.

    Returns
    -------
    pd.Series
        The coordinates of the atom closest to the center as a Pandas Series.
    """
    # Calculate the center point of the sheet
    center_point = [sheet_size[0] / 2, sheet_size[1] / 2]

    # Compute the distance of each atom to the center point using minimum image distance
    distance_to_center_point = [
        minimum_image_distance_2d(center_point, [atom.iloc[1], atom.iloc[2]], sheet_size)
        for _, atom in atoms_df.iterrows()
    ]

    # Find the index of the atom with the minimum distance to the center point
    min_distance_index = distance_to_center_point.index(min(distance_to_center_point))

    # Return the coordinates of the atom closest to the center
    return atoms_df.iloc[min_distance_index]


def rotate_vector(vec: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotates a vector by a given angle.

    Parameters
    ----------
    vec : np.ndarray
        The vector to rotate.
    angle : float
        The angle by which to rotate the vector, in degrees.

    Returns
    -------
    np.ndarray
        The rotated vector.
    """
    # Convert the angle from degrees to radians
    rad = np.radians(angle)

    # Create a 2D rotation matrix
    rotation_matrix = np.array(
        [
            [np.cos(rad), -np.sin(rad)],  # First row of the rotation matrix
            [np.sin(rad), np.cos(rad)],  # Second row of the rotation matrix
        ]
    )

    # Rotate the vector by multiplying it with the rotation matrix
    return np.dot(rotation_matrix, vec)


def minimum_image_distance_3d(
    position1: List[float], position2: List[float], system_size: Tuple[float, float, float]
) -> float:
    """
    Calculates the minimum image distance between two positions in a 3D periodic system.

    Parameters
    ----------
    position1 : List[float]
        The first position as a list of three floats [x, y, z].
    position2 : List[float]
        The second position as a list of three floats [x, y, z].
    system_size : Tuple[float, float, float]
        The size of the periodic system as a tuple of three floats (width, height, depth).

    Returns
    -------
    float
        The minimum image distance between the two positions.
    """
    # Calculate the difference vector, adjusted for periodic boundaries
    delta = np.array(
        [
            position1[i] - position2[i] - system_size[i] * round((position1[i] - position2[i]) / system_size[i])
            for i in range(3)
        ]
    )
    # Return the minimum image distance between the two positions
    return np.sqrt(np.sum(delta**2))


def minimum_image_distance_2d(
    position1: List[float], position2: List[float], system_size: Tuple[float, float]
) -> float:
    """
    Calculates the minimum image distance between two positions in a periodic system.

    Parameters
    ----------
    position1 : List[float]
        The first position as a list of two floats [x, y].
    position2 : List[float]
        The second position as a list of two floats [x, y].
    system_size : Tuple[float, float]
        The size of the periodic system as a tuple of two floats (width, height).

    Returns
    -------
    float
        The minimum image distance between the two positions.
    """
    # Calculate the difference vector, adjusted for periodic boundaries
    delta = np.array(
        [
            position1[i] - position2[i] - system_size[i] * round((position1[i] - position2[i]) / system_size[i])
            for i in range(2)
        ]
    )
    # Return the minimum image distance between the two positions
    return np.sqrt(np.sum(delta**2))


def positions_are_adjacent(
    position1: List[float], position2: List[float], cutoff_distance: float, system_size: Tuple[float, float]
) -> bool:
    """
    Checks if two positions are adjacent in a periodic system.

    Parameters
    ----------
    position1 : List[float]
        The first position as a list of two floats [x, y].
    position2 : List[float]
        The second position as a list of two floats [x, y].
    cutoff_distance : float
        The cutoff distance for adjacency.
    system_size : Tuple[float, float]
        The size of the periodic system as a tuple of two floats (width, height).

    Returns
    -------
    bool
        True if the positions are adjacent, False otherwise.
    """
    # Calculate the minimum image distance between the two positions
    distance = minimum_image_distance_2d(position1, position2, system_size)

    # Return True if the distance is less than the cutoff distance, otherwise False
    return distance < cutoff_distance


def random_rotation_matrix_2d() -> np.ndarray:
    """
    Generates a random 2D rotation matrix.

    Returns
    -------
    np.ndarray
        The random 2D rotation matrix.
    """
    # Generate a random angle between 0 and 2*pi radians
    angle = np.random.uniform(0, 2 * np.pi)

    # Calculate the cosine and sine of the angle
    cos, sin = np.cos(angle), np.sin(angle)

    # Create the 2D rotation matrix using the cosine and sine values
    return np.array([[cos, -sin], [sin, cos]])


def random_rotate_group_list(group_list: List[List[Union[str, float]]]) -> List[List[Union[str, float]]]:
    """
    Randomly rotates a group of atoms around the z-axis.

    Parameters
    ----------
    group_list : List[List[Union[str, float]]]
        A list of atoms to rotate, where each atom is itself represented by a list.
        The first element is a the element, followed by x and y coordinates.

    Returns
    -------
    List[List[Union[str, float]]]
        The randomly rotated group, with each atom's x and y coordinates rotated.
    """
    # Generate a random 2D rotation matrix
    rotation_matrix = random_rotation_matrix_2d()

    # Apply the rotation to each atom's x and y coordinates in the group list
    rotated_group_list = [[atom[0]] + rotation_matrix.dot(atom[1:3]).tolist() + [atom[3]] for atom in group_list]
    # Return the list of rotated atoms
    return rotated_group_list


def find_triangle_tips(center: np.ndarray, tip1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds the tips of a triangle given the center and one tip.

    Parameters
    ----------
    center : np.ndarray
        The center of the triangle.
    tip1 : np.ndarray
        One tip of the triangle.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The other two tips of the triangle.
    """
    # Calculate the vector from the center to the first tip
    vec1 = tip1 - center

    # Rotate this vector by 120 degrees to find the second tip
    vec2 = rotate_vector(vec1, 120)

    # Rotate the original vector by 240 degrees to find the third tip
    vec3 = rotate_vector(vec1, 240)

    # Calculate the coordinates of the second and third tips by adding the rotated vectors to the center
    tip2 = center + vec2
    tip3 = center + vec3

    # Return the coordinates of the two additional tips
    return tip2, tip3


def point_is_inside_triangle(tip1: np.ndarray, tip2: np.ndarray, tip3: np.ndarray, point: np.ndarray) -> bool:
    """
    Checks if a point is inside a triangle.

    Parameters
    ----------
    tip1 : np.ndarray
        The first tip of the triangle as a numpy array.
    tip2 : np.ndarray
        The second tip of the triangle as a numpy array.
    tip3 : np.ndarray
        The third tip of the triangle as a numpy array.
    point : Tuple[float, float]
        The point to check as a numpy array.

    Returns
    -------
    bool
        True if the point is inside the triangle, False otherwise.
    """

    # Calculate the area of the whole triangle
    A = area(tip1, tip2, tip3)
    # Calculate the area of the triangles formed by the point and the tips
    A1 = area(point, tip2, tip3)
    A2 = area(tip1, point, tip3)
    A3 = area(tip1, tip2, point)

    # Calculate the barycentric coordinates
    l1 = A1 / A
    l2 = A2 / A
    l3 = A3 / A

    # Check if the point is inside the triangle
    return 0 <= l1 <= 1 and 0 <= l2 <= 1 and 0 <= l3 <= 1 and abs(l1 + l2 + l3 - 1) < 1e-5


def area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Calculate the area of the triangle formed by points a, b, and c.

    Parameters
    ----------
    a : np.ndarray
        The first point of the triangle.
    b : np.ndarray
        The second point of the triangle.
    c : np.ndarray
        The third point of the triangle.

    Returns
    -------
    float
        The area of the triangle.
    """
    return 0.5 * abs((a[0] - c[0]) * (b[1] - a[1]) - (a[0] - b[0]) * (c[1] - a[1]))
