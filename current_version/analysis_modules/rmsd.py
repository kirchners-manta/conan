import numpy as np
import pandas as pd

import utils as ut
import defdict as ddict

import cProfile

# Create a profiler object
profiler = cProfile.Profile()

# Start the profiler
profiler.enable()


# with this module the rmsd of the molecules in the system is computed (for all liquid species)
def rmsd_prep(inputdict):
    
    number_of_frames = inputdict['number_of_frames']
    first_frame = inputdict['id_frame']
    # rename the x, y and z column to X, Y and Z 
    first_frame = first_frame.rename(columns={'x': 'X', 'y': 'Y', 'z': 'Z'})
    box_size = inputdict['box_size']

    #add a new column to the first_frame by adding the masses of each atom. Stored in dict_mass in ddict.
    first_frame['Mass'] = first_frame['Element'].map(ddict.dict_mass())
    inputdict['molecule_indices'] = {}

    # Set up 3D arrays to store the calulated displacements in x, y and z direction.
    # We check the id_frame by finding all different numbers in the Species column, where Liquid is written in the Struc column.
    # This way we can find out how many liquid species are in the system.
    num_liq_species_dict = {}
    species_mol_numbers = {}
    num_liq_species = first_frame[first_frame['Struc'] == 'Liquid']['Species'].unique()

    for species in num_liq_species:
        num_molecules = len(first_frame[(first_frame['Struc'] == 'Liquid') & (first_frame['Species'] == species)]['Molecule'].unique())
        num_liq_species_dict[species] = num_molecules
        #also store the exact molecule numbers in a list and add them to the dictionary
        molecule_numbers = first_frame[(first_frame['Struc'] == 'Liquid') & (first_frame['Species'] == species)]['Molecule'].unique()
        species_mol_numbers[species] = molecule_numbers

        #add the molecule indices to the inputdict
        inputdict['molecule_indices'][species] = {molecule: i for i, molecule in enumerate(molecule_numbers)}

        # set up the 3D arrays
        displacements = np.zeros((num_molecules, number_of_frames, 3))
        inputdict['displacements_'+str(species)] = displacements
        print('new array called displacements_' + str(species) + ' created of size ' + str(displacements.shape))

    #now set up a dataframe containing the COM coordinated of each liquid molecule in the system
    #the dataframe is structured as follows column wise: species, molecule number, x, y z coordinated of the COM 
    #the dataframe is called COM_frame_initial.
    COM_frame_initial = pd.DataFrame(columns = ['Species', 'Molecule', 'X', 'Y', 'Z'])

    # Loop over all species and all molecules of the species and calculate the COM of each molecule using the calculate_com function.
    # The calculated COM is then added to the COM_frame_initial dataframe.
    first_frame_grouped = first_frame[first_frame['Struc'] == 'Liquid'].groupby(['Species', 'Molecule'])
    COM_dicts_initial = []
    for (species, molecule), molecule_data in first_frame_grouped:
        com = ut.calculate_com(molecule_data, box_size)
        molecule_dict = {
            'Species': species,
            'Molecule': molecule,
            'X': com[0],
            'Y': com[1],
            'Z': com[2]
        }
        COM_dicts_initial.append(molecule_dict)
    COM_frame_initial = pd.DataFrame(COM_dicts_initial)


    print(COM_frame_initial)

    inputdict['COM_frame_initial'] = COM_frame_initial  
    inputdict['num_liq_species_dict'] = num_liq_species_dict
    inputdict['species_mol_numbers'] = species_mol_numbers

    print(first_frame)

    return inputdict

def rmsd_calc(inputdict):

    current_frame = inputdict['split_frame']
    COM_frame_initial = inputdict['COM_frame_initial']
    # make the x, y, z and mass column floats
    current_frame['X'] = current_frame['X'].astype(float)
    current_frame['Y'] = current_frame['Y'].astype(float)
    current_frame['Z'] = current_frame['Z'].astype(float)
    current_frame['Mass'] = current_frame['Mass'].astype(float)

    box_size = inputdict['box_size']

    # Calculate the COM of all liquid molecules in the current frame and store them in a dataframe.
    COM_frame_current = pd.DataFrame(columns = ['Species', 'Molecule', 'X', 'Y', 'Z'])

    current_frame_grouped = current_frame.groupby(['Species', 'Molecule'])
    COM_dicts_current = []
    for (species, molecule), molecule_data in current_frame_grouped:
        com = ut.calculate_com(molecule_data, box_size)
        molecule_dict = {
            'Species': species,
            'Molecule': molecule,
            'X': com[0],
            'Y': com[1],
            'Z': com[2]
        }
        COM_dicts_current.append(molecule_dict)
    COM_frame_current = pd.DataFrame(COM_dicts_current)

    #print(COM_frame_initial)
    #print(COM_frame_current)


    # Calculate the displacement of each liquid molecule in the current frame compared to the initial frame.
    # The displacement is stored in the displacements array.
    for species in inputdict['num_liq_species_dict']:
        for molecule in inputdict['species_mol_numbers'][species]:
            COM_initial = COM_frame_initial[(COM_frame_initial['Species'] == species) & (COM_frame_initial['Molecule'] == molecule)]
            COM_current = COM_frame_current[(COM_frame_current['Species'] == species) & (COM_frame_current['Molecule'] == molecule)]
            displacement = np.array([COM_current['X'].values[0] - COM_initial['X'].values[0], COM_current['Y'].values[0] - COM_initial['Y'].values[0], COM_current['Z'].values[0] - COM_initial['Z'].values[0]])
            mapped_index = inputdict['molecule_indices'][species][molecule]
            inputdict['displacements_'+str(species)][mapped_index, inputdict['counter']] = displacement

            '''
            if inputdict['counter'] == inputdict['number_of_frames']-1:
                print('current_frame for last frame:')
                print(current_frame)
                print('COM_frame_current for last frame:')
                print(COM_frame_current)
                print('displacement for last frame:')
                print(displacement)  
            '''
    inputdict['COM_frame_current'] = COM_frame_current

    '''
    # Print the calculated displacements
    print('Displacements:')
    for species in inputdict['num_liq_species_dict']:
        print('Species: ' + str(species))
        for molecule in inputdict['species_mol_numbers'][species]:
            print('Molecule: ' + str(molecule))

            molecule_indices = {molecule: i for i, molecule in enumerate(inputdict['species_mol_numbers'][species])}

            print(inputdict['displacements_'+str(species)][molecule_indices[molecule], inputdict['counter']])
    '''
    return inputdict

def rmsd_processing(inputdict):

    # Calculate the RMSD of each liquid molecule in the system.
    # The RMSD is stored in the rmsd array.
    for species in inputdict['num_liq_species_dict']:
        for molecule in inputdict['species_mol_numbers'][species]:
            molecule_indices = {molecule: i for i, molecule in enumerate(inputdict['species_mol_numbers'][species])}
            displacements = inputdict['displacements_'+str(species)][molecule_indices[molecule]]
            rmsd = np.sqrt(np.sum(np.square(displacements), axis=1))
            inputdict['rmsd_'+str(species)+'_'+str(molecule)] = rmsd

            print(displacements)


    '''
    #print the rmsd values
    print('RMSD:')
    for species in inputdict['num_liq_species_dict']:
        print('Species: ' + str(species))
        for molecule in inputdict['species_mol_numbers'][species]:
            print('Molecule: ' + str(molecule))
            print(inputdict['rmsd_'+str(species)+'_'+str(molecule)])

    '''

    return inputdict