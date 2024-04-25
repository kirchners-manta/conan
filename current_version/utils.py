import numpy as np

def minimum_image_distance(box_dimension, coordinates_reference, coordinates_observed):
    # coordinates_reference and coordinates_observed are pandas dataframes, that contain the columns
    # 'X_COM', 'Y_COM' and 'Z_COM' for the coordinates. For all molecules in 'coordinates_reference',
    # the distance to all molecules in 'coordinates_observed' is calculated

    dx = coordinates_reference['X_COM'].values[:, np.newaxis] - coordinates_observed['X_COM'].values
    dy = coordinates_reference['Y_COM'].values[:, np.newaxis] - coordinates_observed['Y_COM'].values
    dz = coordinates_reference['Z_COM'].values[:, np.newaxis] - coordinates_observed['Z_COM'].values

    dx = dx - box_dimension[0] * np.round(dx / box_dimension[0])
    dy = dy - box_dimension[1] * np.round(dy / box_dimension[1])
    dz = dz - box_dimension[2] * np.round(dz / box_dimension[2])

    distances = np.sqrt(dx**2 + dy**2 + dz**2)

    return distances

def COM_calculation(frame):

    # We now calculate the center of mass (COM) for each molecule
    # Convert all values to float (this is needed so that the agg-function works)
    frame['X'] = frame['X'].astype(float)
    frame['Y'] = frame['Y'].astype(float)
    frame['Z'] = frame['Z'].astype(float)
    frame['Mass'] = frame['Mass'].astype(float)

    # Precompute total mass for each molecule
    total_mass_per_molecule = frame.groupby('Molecule')['Mass'].transform('sum')

    # Calculate mass weighted coordinates
    frame['X_COM'] = (frame['X'] * frame['Mass']) / total_mass_per_molecule
    frame['Y_COM'] = (frame['Y'] * frame['Mass']) / total_mass_per_molecule
    frame['Z_COM'] = (frame['Z'] * frame['Mass']) / total_mass_per_molecule

    # Calculate the center of mass for each molecule
    mol_com = frame.groupby('Molecule').agg(
        Species=('Species', 'first'),
        X_COM=('X_COM', 'sum'),
        Y_COM=('Y_COM', 'sum'),
        Z_COM=('Z_COM', 'sum')
    ).reset_index()

    return mol_com

