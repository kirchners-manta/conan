import numpy as np
import pandas as pd
import sys
import math
import matplotlib.pyplot as plt

import defdict as ddict

'''This function computes the occurence of finding atom Y at a distance to atom X. The maximum distance is set by te user.
The function returns a dictionary with the occurence of finding atom Y at a distance to atom X.
The occurence function thus gives a porbability of finding an atom Y at a distance to atom X.
'''


# Occurence function

def occurrence_prep(inputdict):
    args = inputdict['args']
    #ask the user up to which distance the occurence should be computed
    max_dist = ddict.get_input("Up to which distance should the occurence be computed?  ", args, 'float')

    # ask the user how many intervals the distance should be divided into
    intervals = ddict.get_input("How many intervals should the distance be divided into?  ", args, 'int')

    #set up an array with all the distance increments.
    dist_array = np.linspace(0, max_dist, intervals)

    #set up an array with 2 entries per distance increment. The first entry is the distance increment and the second entry is the total number of atoms found at that distance.
    occ_array = np.zeros((intervals, 2))

    # ask the user which species should be used as reference species
    species_ref = ddict.get_input("Which species should be used as reference species?  ", args, 'int')
    #ask the user which atom should be used as reference atom from the given species
    atom_ref = ddict.get_input("Which atom should be used as reference atom?  ", args, 'string')

    #ask the user which species should be used as observable species
    species_obs = ddict.get_input("Which species should be used as observable species?  ", args, 'int')
    # ask the user which atom should be used as observable
    atom_obs = ddict.get_input("Which atom should be used as observable?  ", args, 'string')

    outputdict = {
        'max_dist': max_dist,
        'intervals': intervals,
        'species_ref': species_ref,
        'atom_ref': atom_ref,
        'species_obs': species_obs,
        'atom_obs': atom_obs,
        'dist_array': dist_array,
        'occ_array': occ_array
    }

    inputdict.update(outputdict)
    return inputdict


def occurrence_analysis(inputdict):
    #get the data from the inputdict
    args = inputdict['args']
    max_dist = inputdict['max_dist']
    intervals = inputdict['intervals']
    species_ref = inputdict['species_ref']
    atom_ref = inputdict['atom_ref']
    species_obs = inputdict['species_obs']
    atom_obs = inputdict['atom_obs']
    split_frame = inputdict['split_frame']
    dist_array = inputdict['dist_array']
    occ_array = inputdict['occ_array']

    # make a new datafram with only the reference atoms from the reference species
    ref_frame = split_frame[split_frame['Species'] == species_ref]
    ref_frame = ref_frame[ref_frame['Label'] == atom_ref]

    # make a new dataframe with only the observable atoms from the observable species
    obs_frame = split_frame[split_frame['Species'] == species_obs]
    obs_frame = obs_frame[obs_frame['Label'] == atom_obs]

    #make sure the x,y and z coordinates are floats (column 2,3 and 4)
    ref_frame.iloc[:, 2:5] = ref_frame.iloc[:, 2:5].astype(float)
    obs_frame.iloc[:, 2:5] = obs_frame.iloc[:, 2:5].astype(float)

    # Convert the coordinates to NumPy arrays
    ref_coords = ref_frame.iloc[:, 2:5].values
    obs_coords = obs_frame.iloc[:, 2:5].values

    # Compute the distances between each pair of atoms
    distances = np.linalg.norm(ref_coords - obs_coords, axis=1)

    print(distances)

    return inputdict
