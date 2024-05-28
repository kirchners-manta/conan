import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import conan.defdict as ddict
from conan.analysis_modules import utils as ut

# with this module the rmsd of the molecules in the system is computed (for all liquid species)
def msd_prep(inputdict):
    
    number_of_frames = inputdict['number_of_frames']
    first_frame = inputdict['id_frame']
    # rename the x, y and z column to X, Y and Z 
    first_frame = first_frame.rename(columns={'x': 'X', 'y': 'Y', 'z': 'Z'})
    box_size = inputdict['box_size']

    # add a new column to the first_frame by adding the masses of each atom. Stored in dict_mass in ddict.
    first_frame['Mass'] = first_frame['Element'].map(ddict.dict_mass())
    inputdict['molecule_indices'] = {}

    # Set up 3D arrays to store the calulated displacements in x, y and z direction.
    # check the id_frame by finding all different numbers in the Species column, where Liquid is written in the Struc column.
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
        ddict.printLog('New array called displacements_' + str(species) + ' created of size ' + str(displacements.shape))

    # Set up the initial COM dataframe containing the coordinates of each liquid molecule in the system
    # Loop over all species and all molecules and calculate the COM of each molecule using the calculate_com function in utils.
    # The calculated COM is then added to the COM_frame_initial dataframe.
    COM_frame_initial = pd.DataFrame(columns = ['Species', 'Molecule', 'X', 'Y', 'Z'])
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

    inputdict['COM_frame_reference'] = COM_frame_initial  
    inputdict['num_liq_species_dict'] = num_liq_species_dict
    inputdict['species_mol_numbers'] = species_mol_numbers

    return inputdict

def msd_calc(inputdict):

    box_size = inputdict['box_size']
    current_frame = inputdict['split_frame']
    COM_frame_reference = inputdict['COM_frame_reference']
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

    # Calculate the displacement of each liquid molecule in the current frame compared to the initial frame.
    # The minimum image convention is used to calculate the correct  displacement (to account for the PBC).
    # The displacement is stored in the displacements array.
    for species in inputdict['num_liq_species_dict']:
        for molecule in inputdict['species_mol_numbers'][species]:
            COM_initial = COM_frame_reference[(COM_frame_reference['Species'] == species) & (COM_frame_reference['Molecule'] == molecule)]
            COM_current = COM_frame_current[(COM_frame_current['Species'] == species) & (COM_frame_current['Molecule'] == molecule)]
            displacement = np.array([COM_current['X'].values[0] - COM_initial['X'].values[0], COM_current['Y'].values[0] - COM_initial['Y'].values[0], COM_current['Z'].values[0] - COM_initial['Z'].values[0]])
            displacement = displacement - box_size * np.round(displacement / box_size)
            mapped_index = inputdict['molecule_indices'][species][molecule]
            inputdict['displacements_'+str(species)][mapped_index, inputdict['counter']] = displacement


    inputdict['COM_frame_reference'] = COM_frame_current

    return inputdict

def msd_processing(inputdict):
    
    num_frames = inputdict['number_of_frames']
    args = inputdict['args']
    # ask if the msd or the rmsd should be calculated
    msd_or_rmsd = int(ddict.get_input('Do you want to calculate the [1] MSD or the [2] RMSD?  ', args, 'int'))
    
    
    if msd_or_rmsd == 1:
        ddict.printLog('Calculating MSD values for each species...')
        for species in inputdict['num_liq_species_dict']:
            num_molecules = inputdict['num_liq_species_dict'][species]
            displacements = inputdict['displacements_' + str(species)]

            msd_all_timesteps = np.zeros((num_frames, num_frames))

            msd_all_timesteps_x = np.zeros((num_frames, num_frames))
            msd_all_timesteps_y = np.zeros((num_frames, num_frames))
            msd_all_timesteps_z = np.zeros((num_frames, num_frames))

            for t0 in range(num_frames):
                for t1 in range(t0, num_frames):
                    displacement_t0_t1 = displacements[:, t1, :] - displacements[:, t0, :]
                    msd_t0_t1 = np.sum(np.square(displacement_t0_t1), axis=1)

                    msd_t0_t1_x = np.square(displacement_t0_t1[:, 0])
                    msd_t0_t1_y = np.square(displacement_t0_t1[:, 1])
                    msd_t0_t1_z = np.square(displacement_t0_t1[:, 2])

                    msd_all_timesteps[t0, t1] = np.mean(msd_t0_t1)
                    msd_all_timesteps_x[t0, t1] = np.mean(msd_t0_t1_x)
                    msd_all_timesteps_y[t0, t1] = np.mean(msd_t0_t1_y)
                    msd_all_timesteps_z[t0, t1] = np.mean(msd_t0_t1_z)

            inputdict['msd_' + str(species)] = np.mean(msd_all_timesteps, axis=0)

            inputdict['msd_' + str(species) + '_x'] = np.mean(msd_all_timesteps_x, axis=0)
            inputdict['msd_' + str(species) + '_y'] = np.mean(msd_all_timesteps_y, axis=0)
            inputdict['msd_' + str(species) + '_z'] = np.mean(msd_all_timesteps_z, axis=0)

        for species in inputdict['num_liq_species_dict']:
            ddict.printLog('MSD values for species ' + str(species) + ':')
            ddict.printLog(inputdict['msd_' + str(species)])

        fig, axs = plt.subplots(2, 2)
        for species in inputdict['num_liq_species_dict']:
            axs[0, 0].plot(inputdict['msd_' + str(species)], label=species)
            axs[0, 1].plot(inputdict['msd_' + str(species) + '_x'], label=species)
            axs[1, 0].plot(inputdict['msd_' + str(species) + '_y'], label=species)
            axs[1, 1].plot(inputdict['msd_' + str(species) + '_z'], label=species)

        axs[0, 0].set(xlabel='Index', ylabel='MSD', title='Overall MSD values for each species')
        axs[0, 1].set(xlabel='Index', ylabel='MSD', title='MSD values for each species in X')
        axs[1, 0].set(xlabel='Index', ylabel='MSD', title='MSD values for each species in Y')
        axs[1, 1].set(xlabel='Index', ylabel='MSD', title='MSD values for each species in Z')

        for ax in axs.flat:
            ax.legend()
            ax.grid()

        plt.tight_layout()
        plt.show()
        fig.savefig('msd.png')    
    
    elif msd_or_rmsd == 2:
        ddict.printLog('Calculating RMSD values for each species...')

        for species in inputdict['num_liq_species_dict']:
            num_molecules = inputdict['num_liq_species_dict'][species]
            displacements = inputdict['displacements_' + str(species)]

            rmsd_all_timesteps = np.zeros((num_frames, num_frames))

            rmsd_all_timesteps_x = np.zeros((num_frames, num_frames))
            rmsd_all_timesteps_y = np.zeros((num_frames, num_frames))
            rmsd_all_timesteps_z = np.zeros((num_frames, num_frames))


            for t0 in range(num_frames):
                for t1 in range(t0, num_frames):
                    displacement_t0_t1 = displacements[:, t1, :] - displacements[:, t0, :]

                    rmsd_t0_t1 = np.sqrt(np.sum(np.square(displacement_t0_t1), axis=1))

                    rmsd_t0_t1_x = np.sqrt(np.square(displacement_t0_t1[:, 0]))
                    rmsd_t0_t1_y = np.sqrt(np.square(displacement_t0_t1[:, 1]))
                    rmsd_t0_t1_z = np.sqrt(np.square(displacement_t0_t1[:, 2]))

                    rmsd_all_timesteps[t0, t1] = np.mean(rmsd_t0_t1)
                    rmsd_all_timesteps_x[t0, t1] = np.mean(rmsd_t0_t1_x)
                    rmsd_all_timesteps_y[t0, t1] = np.mean(rmsd_t0_t1_y)
                    rmsd_all_timesteps_z[t0, t1] = np.mean(rmsd_t0_t1_z)


            inputdict['rmsd_' + str(species)] = np.mean(rmsd_all_timesteps, axis=0)

            inputdict['rmsd_' + str(species) + '_x'] = np.mean(rmsd_all_timesteps_x, axis=0)
            inputdict['rmsd_' + str(species) + '_y'] = np.mean(rmsd_all_timesteps_y, axis=0)
            inputdict['rmsd_' + str(species) + '_z'] = np.mean(rmsd_all_timesteps_z, axis=0)


        for species in inputdict['num_liq_species_dict']:
            ddict.printLog('RMSD values for species ' + str(species) + ':')
            ddict.printLog(inputdict['rmsd_' + str(species)])
            ddict.printLog(inputdict['rmsd_' + str(species) + '_x'])
            ddict.printLog(inputdict['rmsd_' + str(species) + '_y'])
            ddict.printLog(inputdict['rmsd_' + str(species) + '_z'])


        fig, axs = plt.subplots(2, 2)
        for species in inputdict['num_liq_species_dict']:
            axs[0, 0].plot(inputdict['rmsd_' + str(species)], label=species)
            axs[0, 1].plot(inputdict['rmsd_' + str(species) + '_x'], label=species)
            axs[1, 0].plot(inputdict['rmsd_' + str(species) + '_y'], label=species)
            axs[1, 1].plot(inputdict['rmsd_' + str(species) + '_z'], label=species)

        axs[0, 0].set(xlabel='Index', ylabel='RMSD', title='Overall RMSD values for each species')
        axs[0, 1].set(xlabel='Index', ylabel='RMSD', title='RMSD values for each species in X')
        axs[1, 0].set(xlabel='Index', ylabel='RMSD', title='RMSD values for each species in Y')
        axs[1, 1].set(xlabel='Index', ylabel='RMSD', title='RMSD values for each species in Z')

        for ax in axs.flat:
            ax.legend()
            ax.grid()

        plt.tight_layout()
        plt.show()
        fig.savefig('rmsd.png')

    return inputdict