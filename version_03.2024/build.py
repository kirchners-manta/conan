# The program is written by Leonard Dick, 2023

# MODULES
import time                                                    
import os
import math
import numpy as np
import pandas as pd
from typing import Tuple
import random
from prettytable import PrettyTable
# ----- Own modules ----- #
import defdict as ddict
import traj_info

# RUNTIME
build_time = time.time()

# ARGUMENTS
args = ddict.read_commandline()

# Define bond length and interlayer distance.
def parameters(build) -> Tuple[float, float]:
    if build != 5:
        layer_distance = 3.35
        bond_distance = 1.42
        ddict.printLog("Default Parameters:\nC-C distance: {:.2f} Ang\nInterlayer distance of carbon sheets: {:.2f} Ang".format(bond_distance, layer_distance))
    else:
        layer_distance = 3.33
        bond_distance = 1.446
        ddict.printLog("Default Parameters:\nB-N distance: {:.3f} Ang\nInterlayer distance of boron nitride sheets: {:.2f} Ang".format(bond_distance, layer_distance))
    
    change_paras = ddict.get_input("Change values? [y/n] ", args, 'string')
    if change_paras == 'y':
        bond_distance = float(ddict.get_input("New bond distance: ", args, 'float'))
        layer_distance = float(ddict.get_input("New interlayer distance: ", args, 'float'))

    return bond_distance, layer_distance


# Wall section.
def wall_structure(carbon_distance, interlayer_distance, boronnitride, size_graphsheet_x=None, size_graphsheet_y=None, num_layer=None) -> pd.DataFrame:

    # General questions.
    ddict.printLog('')
    if size_graphsheet_x is None:
        size_graphsheet_x = float(ddict.get_input('How far should the walls extend in x direction? [Ang]  ', args, 'float'))
    if size_graphsheet_y is None:
        size_graphsheet_y = float(ddict.get_input('How far should the walls extend in y direction? [Ang]  ', args, 'float'))
    if num_layer is None:
        num_layer = int(ddict.get_input('How many layers in z direction?   ', args, 'int'))

    # Load the provided distances and bond lengths.
    distance = float(carbon_distance)
    interlayer_distance = float(interlayer_distance)
    distance_row_y = distance * math.cos(30 * math.pi / 180)

    # Define an array for the atom unti cell positions.
    positions_unitcell = [(distance * 0.5,0,0),(distance * 1.5,0,0),(0,distance_row_y,0),(distance * 2,distance_row_y,0)]
    positions = []

    # Loop over the number of layers and the number of unit cells in x and y direction, using AB stacking.
    for i in range(0,num_layer):
        position_z = interlayer_distance * i
        if i%2 == 0:
            AB_shift = 0
        else:
            # If a boronnitride layer is to be build, deactivate the alternating shift.
            if boronnitride == True:
                AB_shift = 0
            else:
                AB_shift = distance
        # Unit cell in y direction. Maximal number of unit cells direction is 1000.
        for k in range(0,1000):
        # Repeat the unit cell in x direction.
            position_y = distance_row_y * 2 * k
            # Add the unit cell to the array until the distance is larger than the size of the carbon wall.
            for i in range(0,1000):
                position_x = distance * 3 * i + AB_shift
                for j in range(0,4):
                    positions.append((positions_unitcell[j][0] + position_x, positions_unitcell[j][1] + position_y, positions_unitcell[j][2] + position_z))
                if positions_unitcell[j][0] + distance * 3 * i > size_graphsheet_x:
                    break
            if position_y > size_graphsheet_y:
                break    

    positions = pd.DataFrame(positions)

    # Set the element type for each atom.
    if boronnitride == True:
        # Identify all x values in the dataframe and write them to an array.
        x_values = []
        for i in range(0,len(positions)):
            if positions.at[i,0] not in x_values:
                x_values.append(positions.at[i,0])
        x_values.sort()

        # Assign the boron and nitrogen element type for each atom. For the second layer, the positions are switched -> ABA stacking.
        # First set all atoms to nitrogen.
        positions.insert(0, "Atom", "N")

        # Then: All atoms with an even x value and an even layer number: Change the element type to boron.
        for i in range(0,len(x_values)):
            for j in range(0,num_layer):
                if i%2 == 0 and j%2 == 0:
                    positions.loc[(positions[0] == x_values[i]) & (positions[2] == interlayer_distance*j), 'Atom'] = 'B'
                    
                elif i%2 == 1 and j%2 == 1:
                    positions.loc[(positions[0] == x_values[i]) & (positions[2] == interlayer_distance*j), 'Atom'] = 'B'
    else:
        positions.insert(0, "Atom", "C")
    

    # Name the columns Atom, x, y and z.
    positions.columns = ["Atom", "x", "y", "z"]

    return positions


# Multiple CNTs
def stacked_CNTs(radius, positions_tube) -> pd.DataFrame:

        #First set a distance between the tubes.
        distance_tubes = float(ddict.get_input('Which distance between the tubes? [Ang]  ', args, 'float'))
        radius_distance = radius + distance_tubes/2

        # get the maximum molecule number
        max_molecule = positions_tube.iloc[:,4].max()

        # Now the position of the second tube is calculated
        # The tube is shifted by the radius_distance in x direction,and by sqrt(3)*radius_distance in y direction.
        tube_two = positions_tube.copy()
        tube_two.iloc[:,1] = tube_two.iloc[:,1] + radius_distance
        tube_two.iloc[:,2] = tube_two.iloc[:,2] + radius_distance*math.sqrt(3)
        tube_two.iloc[:,4] = tube_two.iloc[:,4] + max_molecule

        # Concatenate the two tubes
        unit_cell = pd.DataFrame()
        unit_cell = pd.concat([positions_tube, tube_two], ignore_index=True)

        # Now build the periodic unit cell from the given atoms.
        # The dimensions of the unit cell are 2*radius_distance in x direction and 2*radius_distance*math.sqrt(3) in y direction.
        unit_cell_x = float(2*radius_distance)
        unit_cell_y = float(2*radius_distance*math.sqrt(3))

        # Now multiply the unit cell in x and y direction to fill the whole simulation box.
        multiplicity_x = int(ddict.get_input("Multiplicity in x direction:  ", args, 'int'))
        multiplicity_y = int(ddict.get_input("Multiplicity in y direction:  ", args, 'int'))

        # The positions of the atoms in the unit cell are copied and shifted in x and y direction.
        super_cell = unit_cell.copy()
        supercell_x = unit_cell.copy()
        max_molecule = unit_cell.iloc[:,4].max() if not unit_cell.empty else 0
        for i in range(1, multiplicity_x):
            supercell_x = unit_cell.copy()
            supercell_x.iloc[:,1] = unit_cell.iloc[:,1] + i*unit_cell_x
            supercell_x.iloc[:,4] = supercell_x.iloc[:,4] + max_molecule*i
            super_cell = pd.concat([super_cell, supercell_x], ignore_index=True)

        supercell_after_x = super_cell.copy()
        max_molecule = super_cell.iloc[:,4].max() if not super_cell.empty else 0
        for i in range(1, multiplicity_y):
            supercell_y = supercell_after_x.copy()
            supercell_y.iloc[:,2] = supercell_y.iloc[:,2] + i*unit_cell_y
            supercell_y.iloc[:,4] = supercell_y.iloc[:,4] + max_molecule*i
            super_cell = pd.concat([super_cell, supercell_y], ignore_index=True)

        #check for duplicates in the supercell. If there have been any, give a warning, then drop them.
        duplicates = super_cell.duplicated(subset = ['x', 'y', 'z'], keep = 'first')
        if duplicates.any():
            ddict.printLog(f'[WARNING] Duplicates found in the supercell. Dropping them.')
        super_cell = super_cell.drop_duplicates(subset = ['x', 'y', 'z'], keep = 'first')
    
        # Now the supercell is written to positions_tube.
        positions_tube = pd.DataFrame(super_cell.copy())

        # Finally compute the PBC size of the simulation box. It is given by the multiplicity in x and y direction times the unit cell size.
        pbc_size_x = multiplicity_x * unit_cell_x
        pbc_size_y = multiplicity_y * unit_cell_y
        
        # shift all atoms with x or y coordinates larger than the pbc_size to the inside of the pbc
        for i in range(0, len(positions_tube)):
            if positions_tube.iloc[i,1] > pbc_size_x:
                positions_tube.iloc[i,1] = positions_tube.iloc[i,1] - pbc_size_x
            if positions_tube.iloc[i,1] < 0:
                positions_tube.iloc[i,1] = positions_tube.iloc[i,1] + pbc_size_x
            if positions_tube.iloc[i,2] > pbc_size_y:
                positions_tube.iloc[i,2] = positions_tube.iloc[i,2] - pbc_size_y
            if positions_tube.iloc[i,2] < 0:
                positions_tube.iloc[i,2] = positions_tube.iloc[i,2] + pbc_size_y
        
        ddict.printLog(positions_tube)
        
        # now within the dataframe, the molcules need to be sorted by the following criteria: Species number -> Molecule number -> Label.
        # The according column names are 'Species', 'Molecule' and 'Label'. The first two are floats, the last one is a string.
        # In case of the label the sorting should be done like C1, C2, C3, ... C10, C11, ... C100, C101, ... C1000, C1001, ...
        # Extract the numerical part from the 'Label' column and convert it to integer
        positions_tube['Label_num'] = positions_tube['Label'].str.extract('(\d+)').astype(int)

        # Sort the dataframe by 'Element', 'Molecule', and 'Label_num'
        positions_tube = positions_tube.sort_values(by=['Species', 'Molecule', 'Label_num'])

        # Drop the 'Label_num' column as it's no longer needed
        positions_tube = positions_tube.drop(columns=['Label_num'])

        if question_build != 6:
            # Print the information about the stacked CNTs (not if built from input).
            ddict.printLog(f'')
            ddict.printLog(f'[INFO] The stacked CNTs have the following dimensions:')
            ddict.printLog(f'Number of CNTs: {multiplicity_x * multiplicity_y * 2}')
            ddict.printLog(f'Radius: {round(radius, 3)}')
            # Print the periodic boundary conditions with 3 decimal places
            ddict.printLog(f'PBC in x direction: {round(pbc_size_x, 3)}')
            ddict.printLog(f'PBC in y direction: {round(pbc_size_y, 3)}')
        
        return positions_tube, pbc_size_x, pbc_size_y, multiplicity_x, multiplicity_y
    

# Multiple stacked CNTs from given file
def stacked_from_input() -> pd.DataFrame:

    #check if an input file is given from the input arguments (-i flag). It is listed in the args dictionary, under 'input' and has to be not none

    if args['trajectoryfile'] is not None:
        # Read the input file from the command line arguments.
        struc_file = args['trajectoryfile']
    else:
        struc_file = ddict.get_input('What is the name of the coordinate file?  ', args, 'string')

    # Read the input file. Skip the first two lines.
    input_file = open(struc_file, "r")
    input_file.readline()
    input_file.readline()
    positions = pd.DataFrame([line.split() for line in input_file], columns = ["Element", "x", "y", "z"])

    # Delete all empty rows and reset the index.
    positions = positions.dropna()
    positions = positions.reset_index(drop=True)    

    # Change the x, y and z values to float.
    positions[positions.columns[1:4]] = positions[positions.columns[1:4]].astype(float)

    # Now we identify all different molecules in the input file using the molecule recognition function from the traj_info module.
    # Set an arbitary box size of 1000 Ang.
    box_size=(1000, 1000, 1000)

    positions, unique_molecule_frame = traj_info.molecule_recognition(positions, box_size)
    # Make a list of all different molecules.
    molecules = positions.iloc[:,4].unique()

    # Make a list of all different x and y coordinates.
    x_y_values = []
    for i in range(0, len(positions)):
        x_y_values.append((positions.iloc[i,1], positions.iloc[i,2]))
    x_y_values = list(set(x_y_values))

    # Make a list of all different molecules with repeated x and y coordinates.
    repeated_molecules = []
    unique_molecule_atoms = []
    for molecule in molecules:
        # Make a list of all atoms of the current molecule.
        molecule_atoms = []
        for i in range(0, len(positions)):
            if positions.iloc[i,4] == molecule:
                molecule_atoms.append((positions.iloc[i,1], positions.iloc[i,2]))

        # Check for duplicates in the molecule_atoms list.
        unique_molecule_atoms = list(set(molecule_atoms))

        if len(unique_molecule_atoms) < len(molecule_atoms):
            repeated_molecules.append(int(molecule))
    
    # Assign the species number to each identified structure molecule.
    # Set up a dictionary with the molecule number as key and the species number as value.
    species_dict = {}
    for i in range(0, len(repeated_molecules)):
        species_dict[repeated_molecules[i]] = positions.loc[positions.iloc[:,4] == repeated_molecules[i], 'Species'].unique()[0]

    ddict.printLog(f'')
    for i in range(0, len(repeated_molecules)):
        ddict.printLog(f'[INFO] Molecule {repeated_molecules[i]} (species {species_dict[repeated_molecules[i]]}) has repeated x and y coordinates and is set as CNT.\n')
    
    # Identify the radius of the CNT. It is the distance between the center of the CNT and the first atom of the CNT.
    # The center of the CNT is the average of the x and y coordinates of the atoms of the CNT.

    # Make a list of all atoms of the CNT.
    cnt_atoms = []
    for i in range(0, len(positions)):
        if positions.iloc[i,4] == repeated_molecules[0]:
            cnt_atoms.append((positions.iloc[i,0], positions.iloc[i,1], positions.iloc[i,2], positions.iloc[i,3]))

    # Calculate the center of the CNT.
    centerx = 0
    centery = 0
    for i in range(0, len(cnt_atoms)):
        centerx += float(cnt_atoms[i][1])
        centery += float(cnt_atoms[i][2])
    centerx = centerx / len(cnt_atoms)
    centery = centery / len(cnt_atoms)


    # Calculate the distance between the center of the CNT and the first atom of the CNT.
    radius = ((centerx - float(cnt_atoms[0][1])) ** 2 + (centery - float(cnt_atoms[0][2])) ** 2) ** 0.5

    # Make sure the positions dataframe is in the right format, with the x and y values as format float.
    positions[positions.columns[1:4]] = positions[positions.columns[1:4]].astype(float)

    stacked_question = ddict.get_input('Do you want to build stacked CNTs? [y/n]  ', args, 'string')
    if stacked_question == 'y':
        # Now build the stacked CNTs.
        positions_stacked, pbc_x, pbc_y, mult_x, mult_y = stacked_CNTs(radius, positions)


        #relabeling the atoms
        positions_stacked['Label_num'] = positions_stacked['Label'].str.extract('(\d+)').astype(int)

        # Sort the dataframe by 'Element', 'Molecule', and 'Label_num'
        positions_stacked = positions_stacked.sort_values(by=['Species', 'Molecule', 'Label_num'])

        # Drop the 'Label_num' column as it's no longer needed
        positions_stacked = positions_stacked.drop(columns=['Label_num'])

        # Calculate the total number of CNTs 
        number_of_CNTs = 2 * mult_x * mult_y	
    else:
        #relabeling the atoms
        positions['Label_num'] = positions['Label'].str.extract('(\d+)').astype(int)

        # Sort the dataframe by 'Element', 'Molecule', and 'Label_num'
        positions = positions.sort_values(by=['Species', 'Molecule', 'Label_num'])

        # Drop the 'Label_num' column as it's no longer needed
        positions = positions.drop(columns=['Label_num'])
        
        positions_stacked = positions
        number_of_CNTs = 1

    # Now we have to add the periodic boundary conditions in the z direction.
    # Make a list of all different z coordinates.

    z_values = []
    for i in range(0, len(cnt_atoms)):
        z_values.append(float(cnt_atoms[i][3]))

    # Order the z_values list by increasing z values and just keep the unique ones.
    z_values = list(set(z_values))
    z_values = sorted(z_values)

    # Calculate the distance between the first and second z value and 2 and 3 and so on.
    zstep = []
    for i in range(0, len(z_values) - 1):
        zstep.append(z_values[i + 1] - z_values[i])

    zstep_unique = list(set(zstep))

    # If the difference between the smallest and largest distance is below the threshold of 0.01, we assume that the zstep is constant.
    if max(zstep_unique) - min(zstep_unique) < 0.01:
        zstep_unique = max(zstep_unique)
        tube_kind = 1
    else:
        tube_kind = 2

    # If the CNT is armchair, the periodic boundary conditions are just the maximum z value plus the zstep_unique.
    if tube_kind == 1:
        pbc_z = z_values[-1] + zstep_unique
    else:
        pbc_z = z_values[-1] + 1.42

    # Now adjust the z coordinates of the atoms in the positions_stacked dataframe to fit the periodic boundary conditions, if desired.
    pbc_z_question = ddict.get_input('Adjust the z coordinates of the atoms to fit the periodic boundary conditions? [y/n]  ', args, 'string')
    if pbc_z_question == 'y':
        for i in range(0, len(positions_stacked)):
            if positions_stacked.iloc[i,3] > pbc_z:
                positions_stacked.iloc[i,3] = positions_stacked.iloc[i,3] - pbc_z
            if positions_stacked.iloc[i,3] < 0:
                positions_stacked.iloc[i,3] = positions_stacked.iloc[i,3] + pbc_z

    labels_or_elements = ddict.get_input('Do you want to change the atomic element symbol to the given label? [y/n]  ', args, 'string')
    if labels_or_elements == 'y':
        # change the labels of the the atoms of each species (and all the molecules), if wanted. The user can choose freely.
        change_labels = ddict.get_input('Change the labels of the atoms? [y/n]  ', args, 'string')
        if change_labels == 'y':
            # First identify how many different species are in the dataframe.
            species = positions_stacked.iloc[:,6].unique()
            # Then identify all unique labels of the atoms for each species.
            for i in range(0, len(species)):
                labels = positions_stacked.loc[positions_stacked.iloc[:,6] == species[i], 'Label'].unique()
                # Now the user can choose a new label for each atom of each species.
                # Same label for all atoms of the same species(?).
                ddict.printLog(f'')
                #ddict.printLog(f'[INFO] Species {species[i]} has the following labels: {labels}')
                ddict.printLog(f'')
                same_label = ddict.get_input(f'Give all atoms of species {species[i]} the same label? [y/n]  ', args, 'string')
                if same_label == 'y':
                    new_label = ddict.get_input(f'What is the new label for the atoms of species {species[i]}?  ', args, 'string')
                    positions_stacked.loc[positions_stacked.iloc[:,6] == species[i], 'Label'] = new_label
                else:
                    for j in range(0, len(labels)):
                        new_label = ddict.get_input(f'What is the new label for the atoms of species {species[i]} with the label {labels[j]}?  ', args, 'string')
                        positions_stacked.loc[(positions_stacked.iloc[:,6] == species[i]) & (positions_stacked.iloc[:,5] == labels[j]), 'Label'] = new_label

        # Now drop all columns except for the label, x, y and z columns.
        positions_stacked = positions_stacked.drop(columns=['Element', 'Species', 'Molecule'])

        #rename the 'Label' column to the 'Element' column
        positions_stacked = positions_stacked.rename(columns={"Label": "Element"})

    # reordering the columns to element, x, y, z
    positions_stacked = positions_stacked[['Element', 'x', 'y', 'z']]

    #Finally print the information about the stacked CNTs.
    ddict.printLog(f'')
    ddict.printLog(f'[INFO] The stacked CNTs have the following dimensions:')
    ddict.printLog(f'Number of CNTs: {number_of_CNTs}')
    ddict.printLog(f'Radius: {round(radius, 3)}')
    # Print the periodic boundary conditions
    if stacked_question == 'y':
        ddict.printLog(f'PBC in x direction: {round(pbc_x, 3)}')
        ddict.printLog(f'PBC in y direction: {round(pbc_y, 3)}')
    ddict.printLog(f'PBC in z direction: {round(pbc_z, 3)}')

    return positions_stacked

# Tube section.
def CNT(distance, boronnitride) -> Tuple[pd.DataFrame, float, int, float, float]:

    # General questions.
    ddict.printLog('')
    tube_kind = float(ddict.get_input('What kind of tube? [1]armchair [2]zigzag   ', args, 'float'))
    if tube_kind == 1:
       tube_size = int(ddict.get_input('What size should the armchair tube be? [Give m in (m,m) CNT]   ', args, 'int'))

    elif tube_kind == 2:
         tube_size = int(ddict.get_input('What size should the zigzag tube be? [Give m in (m,0) CNT]   ', args, 'int'))
    tube_length = float(ddict.get_input('How long should the tube be? [Ang]   ', args, 'float'))

    # Load the provided distances and bond lengths.
    distance = float(distance)
    hex_d=distance * math.cos(30 * math.pi / 180) * 2

    #Armchair configuration.
    if tube_kind == 1:
        # Calculate the radius of the tube.
        angle_carbon_bond = 360 / (tube_size * 3)
        symmetry_angle = 360 / tube_size
        radius = distance / (2 * math.sin((angle_carbon_bond * math.pi / 180) / 2))
        
        # Calculate the z distances in the tube. 
        distx = (radius - radius * math.cos(angle_carbon_bond / 2  *math.pi / 180))
        disty = (0 - radius * math.sin(angle_carbon_bond / 2 * math.pi / 180))
        zstep = (distance ** 2 - distx ** 2 - disty ** 2) ** 0.5

        # Make a list of the first set of positions.
        positions_tube = []
        angles = []
        z_max = 0
        counter = 0

        while z_max < tube_length:
            z_coordinate = zstep * 2 * counter

            # Loop to create all atoms of the tube.
            for i in range(0, tube_size):

                # Add first position option.
                angle = symmetry_angle * math.pi / 180 * i
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                positions_tube.append((x, y, z_coordinate))

                # Add second position option.
                angle = (symmetry_angle * i + angle_carbon_bond) * math.pi / 180
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                positions_tube.append((x, y, z_coordinate))
                angles.append(angle)

                # Add third position option.
                angle = (symmetry_angle * i + angle_carbon_bond * 3 / 2) * math.pi / 180
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                z = zstep + z_coordinate
                positions_tube.append((x, y, z))
                angles.append(angle)

                # Add fourth position option.
                angle = (symmetry_angle * i + angle_carbon_bond * 5 / 2) * math.pi / 180
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                z = zstep + z_coordinate
                positions_tube.append((x, y, z))
                angles.append(angle)

            z_max = z_coordinate + zstep
            counter += 1

    # Zigzag configuration.
    if tube_kind == 2:
        symmetry_angle = 360 / tube_size
        # Calculate the radius of the tube.
        radius = hex_d / (2 * math.sin((symmetry_angle * math.pi / 180) / 2))
        
        # Calculate the z distances in the tube.
        distx = (radius - radius * math.cos(symmetry_angle / 2 * math.pi / 180))
        disty = (0 - radius * math.sin(symmetry_angle / 2 * math.pi / 180))
        zstep = (distance ** 2 - distx ** 2 - disty ** 2) ** 0.5

        # Make a list of the first set of positions.
        positions_tube= []
        angles = []
        z_max = 0
        counter = 0

        while z_max < tube_length:
            z_coordinate = ( 2 * zstep + distance * 2) * counter

            # Loop to create the atoms in the tube.
            for i in range(0, tube_size):

                # Add first position option.
                angle = symmetry_angle * math.pi / 180 * i
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                positions_tube.append((x, y, z_coordinate))

                # Add second position option.
                angle = (symmetry_angle * i + symmetry_angle / 2) * math.pi / 180
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                z = zstep + z_coordinate
                positions_tube.append((x, y, z))

                # Add third position option.
                angle = (symmetry_angle * i + symmetry_angle / 2) * math.pi / 180
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                z = zstep + distance + z_coordinate
                positions_tube.append((x, y, z))   

                # Add fourth position option.
                angle = symmetry_angle * math.pi / 180 * i
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                z = 2 * zstep + distance + z_coordinate
                positions_tube.append((x, y ,z))         

            z_max = z_coordinate+zstep
            counter += 1

    positions_tube = pd.DataFrame(positions_tube)

    # Insert the atom names to C.
    positions_tube.insert(0, "Atom", "C")

    # Name the columns x, y and z.
    positions_tube.columns = ["Atom", "x", "y", "z"]

    # close the CNT at one side, if the user requests it 
    close_CNT = ddict.get_input('Close the CNT at one side? [y/n]  ', args, 'string')
    if close_CNT == 'y':
        # first create a wall which is larger than the CNT opening 
        wall = wall_structure(distance, distance, boronnitride, size_graphsheet_x=1000, size_graphsheet_y=1000, num_layer=1)
        # shift the wall center to 0 0 0 
        wall.iloc[:,1] = wall.iloc[:,1] - 500
        wall.iloc[:,2] = wall.iloc[:,2] - 500
        # now identify the atoms in the wall dataframe which are in the CNT opening. They are at a distance which is less than the radius-1.42 from the center of the CNT, which is 0 0 0 
        # delete all other atoms
        wall['distance'] = (wall.iloc[:,1] ** 2 + wall.iloc[:,2] ** 2) ** 0.5
        print(wall)
        # drop all lines where the distance is larger than the radius-1.42
        wall = wall.drop(wall[wall['distance'] > radius - 1.42].index)
        # drop the distance column
        wall = wall.drop(columns=['distance'])

        close_at_min_or_max = ddict.get_input('Close the CNT at the minimum or maximum z value? [min/max]  ', args, 'string')
        if close_at_min_or_max == 'min':
            # now add the wall to the CNT
            positions_tube = pd.concat([positions_tube, wall], ignore_index=True)
        elif close_at_min_or_max == 'max':
            # shift the wall to the maximum z value of the CNT
            wall.iloc[:,3] = wall.iloc[:,3] + positions_tube.iloc[:,3].max()
            # now add the wall to the CNT
            positions_tube = pd.concat([positions_tube, wall], ignore_index=True)





    # Change the elements of the nanotube to boronnitride (if needed). 
    if boronnitride == True:

        # It needs to be differentiated between the tube_kind.
        if tube_kind == 1:

            # Make a list of all present x and y combinations in the tube.
            x_y_values_tube = []
            for i in range(0, len(positions_tube)):
                x_y_values_tube.append((positions_tube.iloc[i,1], positions_tube.iloc[i,2]))
            x_y_values_tube = list(set(x_y_values_tube))

            # Lists for the nitrogen and boron atoms.
            nitrogen_atoms = []
            boron_atoms = []

            # Change of the element type of the first atom to nitrogen together with all atoms with the same x and y value.
            positions_tube.loc[(positions_tube.iloc[:,1] == x_y_values_tube[0][0]) & (positions_tube.iloc[:,2] == x_y_values_tube[0][1]), "Atom"] = "N"

            # Add the set nitrogen atoms to the nitrogen_atoms list.
            for i in range(0,len(positions_tube)):
                if positions_tube.iloc[i,0] == "N":
                    nitrogen_atoms.append((positions_tube.iloc[i,1],positions_tube.iloc[i,2]))

            # Loop as long as there are nitrogen atoms in the nitrogen_atoms list and boron atoms in the boron_atoms list.
            while len(nitrogen_atoms) != 0 or len(boron_atoms) != 0:
                
                # Delete all boron atoms from the boron_atoms list.
                boron_atoms = []
                # Go through all nitrogen atoms and change the carbon atoms at a bond distance to boron.
                for i in range(0, len(nitrogen_atoms)):
                    for j in range(0, len(positions_tube)):
                        if positions_tube.iloc[j,0] == "C":
                            if ((nitrogen_atoms[i][0] - positions_tube.iloc[j,1]) ** 2 + (nitrogen_atoms[i][1] - positions_tube.iloc[j,2]) ** 2) ** 0.5 <= distance + 0.001:
                                positions_tube.iloc[j,0] = "B"
                                # Change all atoms with the same x and y values as the boron atom to boron. As they then belong to the same row.
                                positions_tube.loc[(positions_tube.iloc[:,1] == positions_tube.iloc[j,1]) & (positions_tube.iloc[:,2] == positions_tube.iloc[j,2]), "Atom"] = "B"
                                # Add the set boron atoms to the boron_atoms list.
                                boron_atoms.append((positions_tube.iloc[j,1], positions_tube.iloc[j,2]))
                
                # Delete all nitrogen atoms from the boron_atoms list.
                nitrogen_atoms = []
                # Go through all boron atoms and change the carbon atoms at a bond distance to nitrogen.
                for i in range(0, len(boron_atoms)):
                    for j in range(0, len(positions_tube)):
                        if positions_tube.iloc[j,0] == "C":
                            if ((boron_atoms[i][0] - positions_tube.iloc[j,1]) ** 2 + (boron_atoms[i][1] - positions_tube.iloc[j,2]) ** 2) ** 0.5 <= distance + 0.001:
                                positions_tube.iloc[j,0] = "N"
                                # Change all atoms with the same x and y values as the nitrogen atom to nitrogen.
                                positions_tube.loc[(positions_tube.iloc[:,1] == positions_tube.iloc[j,1]) & (positions_tube.iloc[:,2] == positions_tube.iloc[j,2]), "Atom"] = "N"
                                # Add the set nitrogen atoms to the nitrogen_atoms list.
                                nitrogen_atoms.append((positions_tube.iloc[j,1], positions_tube.iloc[j,2]))

        if tube_kind == 2:
            # Make a list of all different z values of the atoms of the nanotube.
            z_values = positions_tube.iloc[:,3].unique()
            for i in range(0,len(z_values)):
                # Identify all atoms(rows) with the same z value of the current iteration.
                # Change the element type of the found atoms to boron. 
                if i%2 == 0:
                    positions_tube.loc[positions_tube.iloc[:,3] == z_values[i], "Atom"] = "B"
            # Change all remaining C atoms to nitrogen.
            positions_tube.loc[positions_tube.iloc[:,0] == "C", "Atom"] = "N"
    
    # If the analysis option 'pore' is chosen, stacked CNTs can not be build.
    if question_build != 4:
        # Multiple CNTs in densest packing
        multiple_tubes = ddict.get_input("Multiple stacked CNTs? [y/n]  ", args, 'string')
        if multiple_tubes == "y":
            # rename the 'Atom' column to the 'Element' column
            positions_tube = positions_tube.rename(columns={"Atom": "Element"})
            #make a new column with the species number and molecule number, all set to 1
            positions_tube['Species'] = 1
            positions_tube['Molecule'] = 1
            # also set a Label column with the Element+atom number
            positions_tube['Label'] = positions_tube['Element'] + positions_tube.index.astype(str)
            ddict.printLog(positions_tube)
            positions_tube, pbc_x, pbc_y, multi_x, multi_y = stacked_CNTs(radius, positions_tube)




    return positions_tube, radius, tube_kind, z_max, zstep


# Pore section.
def Pore(carbon_distance: float, interlayer_distance: float, boronnitride: bool) -> Tuple[int, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, float, int, float, float]:
    
    # Parameters
    if boronnitride:
        distance = 1.446
        interlayer_distance = 3.33
    else:
        distance = float(carbon_distance)
        interlayer_distance = float(interlayer_distance)    

    # Generate the carbon wall and CNT using other functions
    wall = wall_structure(distance, interlayer_distance, boronnitride)
    cnt, radius, tube_kind, max_z, zstep = CNT(distance, boronnitride)

    # Cut a hole in the carbon wall where the CNT will be placed by deleting all atoms within the radius of the CNT plus the specified distance
    centerx = wall['x'].max() / 2
    centery = wall['y'].max() / 2
    wall_hole = wall[np.sqrt((wall['x'] - centerx) ** 2 + (wall['y'] - centery) ** 2) > radius + distance]
    
    # Shift the CNT to the center of the carbon wall
    cnt['x'] += centerx
    cnt['y'] += centery

    # Combine the carbon wall and CNT
    pore = pd.concat([wall_hole, cnt])
    ddict.printLog('\nOptions:\n[1] Pore with walls on both ends\n[2] Closed Pore with one wall')
    pore_kind = int(ddict.get_input('Which kind of pore to build?: ', args, 'int')) 
    
    # Create a pore with walls on both ends
    if pore_kind == 1:
        # Add a second carbon wall to the other end of the CNT
        wall_zshift = wall_hole.copy()
        zmax = cnt['z'].max()
        zmax_wall = wall_zshift['z'].max()
        zdiff = zmax - zmax_wall
        wall_zshift['z'] += zdiff
        pore = pd.concat([pore, wall_zshift])
        pore2 = pore.copy()
    
    # Create a closed pore with one wall
    elif pore_kind == 2:
        # Close the pore by adding a second wall of the size of the hole to the other side of the cnt - distance
        wall_zshift = pd.DataFrame.copy(wall)
        # Find the maximum z coordinate of the cnt and the carbon wall
        zmax = max(cnt.iloc[:,3].max(), wall_zshift.iloc[:,3].max())
        # Shift the carbon wall by the difference between zmax and its current maximum z coordinate
        wall_zshift.iloc[:,3] += zmax - wall_zshift.iloc[:,3].max()
        # Remove all layers with smaller z coordinates than the maximum z layer
        wall_zshift = wall_zshift[wall_zshift.iloc[:,3] == zmax]
        # Find the atoms that are closer than radius-distance to the center of the carbon wall
        wall_zshift = wall_zshift[np.sqrt((wall_zshift.iloc[:,1]-centerx) ** 2 + (wall_zshift.iloc[:,2] - centery) ** 2) < radius - distance]
        # Add the carbon wall to the pore and create the second pore by mirroring the z coordinates
        pore = pd.DataFrame(np.concatenate((pore, wall_zshift), axis = 0))
        pore2 = pd.DataFrame.copy(pore)
        pore2.iloc[:,3] = (pore2.iloc[:,3]-zmax) * (-1)

    # Define column 2, 3 and 4 as float
    pore[pore.columns[1:4]] = pore[pore.columns[1:4]].astype(float)
    pore2[pore2.columns[1:4]] = pore2[pore2.columns[1:4]].astype(float)

    return pore_kind, pore, pore2, wall, cnt, radius, tube_kind, max_z, zstep


# Print section.
def xyz_file(name_output, coordinates) -> None:

    # Check if the folder carbon_structures exists, if not it is created.
    if not os.path.exists("structures"):
        os.makedirs("structures")

    # Change the working directory to the carbon_structures folder.
    os.chdir("structures")
    # If the file already exists, delete it.
    if os.path.exists("{}.xyz".format(name_output)):
        os.remove("{}.xyz".format(name_output))

    # Create the xyz file.
    xyz = open("{}.xyz".format(name_output), "x")
    xyz.write("   {}\n".format(len(coordinates)))
    xyz.write("#Made by CONAN\n")
    # Add the coordinates to the xyz file, using the write to csv function. Just use 3 decimal places.
    coordinates.to_csv(xyz, sep = '\t', header = False, index = False, float_format = '%.3f')
    xyz.close()
    os.chdir("..")


# Print tables.
def table_atoms_print(name_output, coordinates) -> None:

    atoms = coordinates.iloc[:,0].unique()
    # List of tuples with the atom kind and the number of atoms for each atom.
    num_atomkind = []
    for atom in atoms:
        num_atomkind.append((atom, len(coordinates[coordinates.iloc[:,0] == atom])))  
    
    # Print the table.
    table = PrettyTable()
    table.title = name_output + " atoms"
    table.field_names = ["Element", "Number of atoms", "Percentage"]
    for i in range(0, len(atoms)):
        # Print the number of atoms and the percentage of the total number of atoms.
        table.add_row([atoms[i], num_atomkind[i][1], round(num_atomkind[i][1] / len(coordinates) * 100, 3)])
    table.add_row(['Total', len(coordinates), "-"])
    ddict.printLog('')
    ddict.printLog(table)
    ddict.printLog('')


# Print wall information.
def table_dimensions_wall(name_output, coordinates, bond_dist, layer_dist, boronnitride) -> None:
    # Get the general info from the coordinates dataframe, the xyz coordinates.
    size_x = coordinates.iloc[:,1].max()
    size_y = coordinates.iloc[:,2].max()
    size_z = coordinates.iloc[:,3].max()

    # Include periodic boundary conditions.
    if boronnitride == True:
        pbc_size_x = size_x + bond_dist
    else:
        # For PBC in z, there have to be multiple walls and an even number of layers
        if size_z > 0:
            pbc_size_x = size_x
        else:
            pbc_size_x = size_x + bond_dist
    pbc_size_y = size_y + bond_dist*math.cos(30*math.pi/180)
    pbc_size_z = size_z + layer_dist  

    wall_table = PrettyTable()
    wall_table.title = name_output + " dimensions"
    wall_table.field_names = ["Coordinate", "Structure size", "Periodic boundary conditions"]
    wall_table.add_row(['x', round(size_x,3), round(pbc_size_x,3)])
    wall_table.add_row(['y', round(size_y,3), round(pbc_size_y,3)])
    wall_table.add_row(['z', round(size_z,3), round(pbc_size_z,3)])
    ddict.printLog(wall_table)

    #First check if there ar multiple layers in the z direction. Find the number of different z coordinates
    num_layer_z = len(coordinates.iloc[:,3].unique())
    if num_layer_z%2 != 0 and boronnitride == False:
        ddict.printLog('\nWarning: The number of layers in z direction is odd. PBC in z direction are not fullfilled due to the AB stacking of the layers.', color='red')


# Print tube information.
def table_dimensions_cnt(name_output, radius, tube_kind, max_z, bond_distance, zstep) -> None:

    # Periodic boundary conditions in z direction.
    if tube_kind == 1:
        pbc_size_z = max_z + zstep
        tube_name = "armchair"

    if tube_kind == 2:
        pbc_size_z = max_z + bond_distance
        tube_name = "zigzag"
        
    tube_table = PrettyTable()
    tube_table.title = name_output + " dimensions [Ang]"
    tube_table.field_names = ["Parameter", "Value"]
    tube_table.add_row(['configuration', tube_name])
    tube_table.add_row(['radius', round(radius, 3)])
    tube_table.add_row(['diameter', round(radius * 2, 3)])
    tube_table.add_row(['length', round(max_z, 3)])
    tube_table.add_row(['PBC length', round(pbc_size_z, 3)])

    ddict.printLog(tube_table)


# MAIN SECTION
# General questions.
ddict.printLog('BUILD mode', color='red')
ddict.printLog('\nOptions:\n[1]Carbon wall\n[2]CNT\n[3]Carbon wall and CNT\n[4]Pore structure\n[5]Boron nitride structures\n[6]Change atom order or build stacked CNTs from input file')
question_build = int(ddict.get_input('What do you want to build? [1-6]: ', args, 'int'))
ddict.printLog('')
# Repeat the question if the input is not valid.
while question_build not in [1, 2, 3, 4, 5, 6]:
    ddict.printLog('Invalid input. Try again.\n', color='red')
    question_build = int(ddict.get_input('What do you want to build? [1-6]: ', args, 'int'))
    ddict.printLog('')

# Parameter adjustment.
if question_build != 6:
    bond_distance, layer_distance = parameters(question_build)

# Building section.
# Predefine all structure dataframes.
positions = pd.DataFrame()
positions_tube = pd.DataFrame()
pore = pd.DataFrame()
pore2 = pd.DataFrame()

# Check if a carbon structure or a hexagonal boron nitride structure is built.
if question_build == 5:
    boronnitride = True
else:
    boronnitride = False

# Building the structures.
if question_build == 1 or question_build == 3:
    positions = wall_structure(bond_distance, layer_distance, boronnitride)

if question_build == 2 or question_build == 3:
    positions_tube, radius, tube_kind, max_z, zstep = CNT(bond_distance, boronnitride)

if question_build == 4:
    pore_kind, pore, pore2, positions, positions_tube, radius, tube_kind, max_z, zstep = Pore(bond_distance, layer_distance, boronnitride)

if question_build == 5:
    question_what_structure = int(ddict.get_input('\nWhat do you want to build?\n[1]Boron nitride wall\n[2]Boron nitride nanotube\n[3]Boron nitride pore\nInput [1-3]: ', args, 'int'))
    
    if question_what_structure == 1:
        positions = wall_structure(bond_distance, layer_distance, boronnitride)
    
    if question_what_structure == 2:
        positions_tube, radius, tube_kind, max_z, zstep = CNT(bond_distance, boronnitride)
    
    if question_what_structure == 3:
        pore_kind, pore, pore2, positions, positions_tube, radius, tube_kind, max_z, zstep = Pore(bond_distance, layer_distance, boronnitride)

if question_build == 6:
    positions_stacked = stacked_from_input()

# Units and doping section.
ddict.printLog('\nDo you want to change units [Default is Ang] or add doping?\n')
ddict.printLog('Doping is only possible for carbon structures.\n', color='red')

question_doping_units = 0
if question_build != 5:
    ddict.printLog('Options:\n[1]Change units\n[2]Add doping\n[3]Change units and add doping\n[4]No changes')
    question_doping_units=int(ddict.get_input('Change the structure? [1-4]: ', args, 'int'))


# Units section.
if question_doping_units == 1 or question_doping_units == 3:
    
    # Ask for units.
    ddict.printLog('\nOptions:\n[1]Bohr\n[2]New')
    question_units=int(ddict.get_input('What units do you want to use? [1-2]: ', args, 'int'))
    ddict.printLog("\n")

    if question_units == 1:
        ddict.printLog('Converting units to Bohr...')
        arbunits=1/0.529177
    else:
        arbunits=float(ddict.get_input('What conversion factor from Ang do you want to use? ', args, 'float'))

    # Convert units.
    if question_build == 1 or question_build == 3 or question_build == 4 or [question_build == 5 and question_what_structure == 1]:
        positions[positions.select_dtypes(['number']).columns] = positions.select_dtypes(['number'])*arbunits

    if question_build == 2 or question_build == 3 or question_build == 4 or [question_build == 5 and question_what_structure == 2]:
        positions_tube[positions_tube.select_dtypes(['number']).columns] = positions_tube.select_dtypes(['number'])*arbunits

    if question_build == 4 or [question_build == 5 and question_what_structure == 3]:
        pore[pore.select_dtypes(float).columns] = pore.select_dtypes(float)*arbunits
        pore2[pore2.select_dtypes(float).columns] = pore2.select_dtypes(float)*arbunits


# Add doping.
if question_doping_units == 2 or question_doping_units == 3:

    if boronnitride == True:
        ddict.printLog_red('\nDoping is only possible for carbon structures. Skipping...\n')
    else:
        ddict.printLog('\nOptions:\n[1]Graphitic nitrogen\n[2]None')
        question_doping = int(ddict.get_input('What kind of atoms do you want to add? [1-2]: ', args, 'int'))
        if question_doping == 1:
            # The percentage of doped atoms is given by the user, as a float between 0 and 100. The value is divided by 100 to get a percentage.
            ddict.printLog('')
            question_doping_amount = float(ddict.get_input('What percentage of atoms do you want to change? ', args, 'float')) / 100

            # Defined by the precentage of doped atoms, change the atom name for a given amount of atoms to N.  
            if question_build == 1 or question_build == 3 or question_build == 4:
                # Randomly change the atoms to N, using the random module.
                for i in range(int(len(positions) * question_doping_amount)):
                    positions.iloc[random.randint(0, len(positions) - 1), 0] = 'N'

            if question_build == 2 or question_build == 3 or question_build == 4:
                for i in range(int(len(positions_tube) * question_doping_amount)):
                    positions_tube.iloc[random.randint(0, len(positions_tube) - 1),0] = 'N'

            if question_build == 4:
                for i in range(int(len(pore) * question_doping_amount)):
                    pore.iloc[random.randint(0,len(pore) - 1), 0] = 'N'
                    pore2.iloc[random.randint(0,len(pore2) - 1), 0] = 'N'
            ddict.printLog('')

        elif question_doping == 2:
            ddict.printLog('')


# Print and save section.
if question_build == 1 or question_build == 3 or question_build == 4:
    xyz_file("wall", positions)
    table_atoms_print("Wall", positions)
    table_dimensions_wall("Wall", positions, bond_distance, layer_distance, boronnitride)

if question_build == 2 or question_build == 3 or question_build == 4:
    xyz_file("cnt", positions_tube)
    table_atoms_print("CNT", positions_tube)
    table_dimensions_cnt("CNT", radius, tube_kind, max_z, bond_distance, zstep)

if question_build == 4 and pore_kind == 1:
    xyz_file("pore", pore)
    table_atoms_print("Pore", pore)

elif question_build == 4 and pore_kind == 2:
    xyz_file("pore_left", pore2)
    table_atoms_print("Left pore", pore2)
    xyz_file("pore_right", pore)
    table_atoms_print("Right pore", pore)

# Boron nitride structures.
if question_build == 5:

    if question_what_structure == 1 or question_what_structure == 3:
        xyz_file("boronnitride_wall", positions)
        table_atoms_print("Boron nitride wall", positions)
        table_dimensions_wall("Boron nitride wall", positions, bond_distance, layer_distance, boronnitride)

    if question_what_structure == 2 or question_what_structure == 3:
        xyz_file("boronnitride_nanotube", positions_tube)
        table_atoms_print("Boron nitride nanotube", positions_tube)
        table_dimensions_cnt("Boron nitride tube", radius, tube_kind, max_z, bond_distance, zstep)

    if question_what_structure == 3 and pore_kind == 1:
        xyz_file("boronnitride_pore", pore)
        table_atoms_print("Boron nitride pore", pore)

    elif question_what_structure == 3 and pore_kind == 2:
        xyz_file("boronnitride_pore_left", pore2)
        table_atoms_print("Boron nitride left pore", pore2)
        xyz_file("boronnitride_pore_right", pore)
        table_atoms_print("Boron nitride right pore", pore)

if question_build == 6:
    xyz_file("stacked_CNTs", positions_stacked)
    table_atoms_print("Stacked CNTs", positions_stacked)

# Print the time the module needed.
ddict.printLog("\nCBuild mode finished in %0.3f seconds" % (time.time()-build_time))
