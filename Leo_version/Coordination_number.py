import numpy as np
import pandas as pd
import sys, time
import scipy


def Coord_number_prep(inputdict):
    import defdict as ddict

    # Get values from inputdict
    min_z_pore = inputdict['min_z_pore'] 
    max_z_pore = inputdict['max_z_pore']
    box_size = inputdict['box_size']
    CNT_centers = inputdict['CNT_centers']
    Walls_positions = inputdict['Walls_positions']
    args = inputdict['args']


    ddict.printLog('')
    if len(CNT_centers) > 1:
        ddict.printLog('-> Multiple CNTs detected. The analysis will be conducted on the first CNT.\n', color = 'red')
    if len(CNT_centers) == 0:
        ddict.printLog('No CNTs detected, performing bulk analysis.\n')

    do_xyz_analysis = ddict.get_input('Do you want to perform a 3d analysis of the coordination number? [y/n]', args, 'string')

    # Set up reference point
    z_referencepoint = [0]*4
    referencepoint = 'n'
    poresonly = 'n'
    structure_as_reference = 'n'
    if do_xyz_analysis == 'n':        
        referencepoint = ddict.get_input('Should the distance to a reference point (e.g. a Wall or CNT) be calculated? [y/n]', args, 'string')
        if referencepoint == 'y':
            structure_as_reference = ddict.get_input('Should an existing structure be used as reference point? [y/n]', args, 'string')           
            if structure_as_reference == 'n':
                ddict.printLog('Setting user-defined reference point for the analysis.\n')
                # Ask user for the wanted reference point
                z_referencepoint[1] = ddict.get_input('Enter the z-values of the reference point point (x- and y- dependent analysis not implemented yet, sorry D:))', args, 'float')
                
                # Set other reference values
                z_referencepoint[2] = z_referencepoint[1]
                z_referencepoint[0] = z_referencepoint[1] - box_size[2]
                z_referencepoint[3] = z_referencepoint[2] + box_size[2]        
            elif structure_as_reference == 'y':
                poresonly = ddict.get_input('Do you want to calculte the coordination number only inside a pore? [y/n]', args, 'string')
                if poresonly == 'n':
                    #Calculate z-coordinates of all pores that need to be considered (4 pores for one CNT)
                    z_referencepoint = [0] * 4
                    if len(CNT_centers) != 0:
                        z_referencepoint[0] = max_z_pore[0] - box_size[2] # z-coordinate of first pore (outside of the box)
                        z_referencepoint[1] = min_z_pore[0]               # z-coordinate of first pore inside the box
                        z_referencepoint[2] = max_z_pore[0]               # z-coordinate of second pore inside the box
                        z_referencepoint[3] = box_size[2] + min_z_pore[0] # z-coordinate of second pore outside of the box
                    elif len(Walls_positions) != 0:
                        z_referencepoint[0] = Walls_positions[0] - box_size[2] # z-coordinate of first Wall (outside of the box)
                        z_referencepoint[1] = Walls_positions[0]               # z-coordinate of first Wall inside the box
                        z_referencepoint[2] = Walls_positions[1]               # z-coordinate of second Wall inside the box
                        z_referencepoint[3] = box_size[2] + Walls_positions[1] # z-coordinate of second Wall outside of the box
                        # All of this just assumes two walls / one pore at the moment. If both are present only the distance to the pore is analyzed.
                    else:
                        ddict.printLog("No structural coordinates found. aborting run...")
                        sys.exit(1)
            else:
                ddict.printLog('Invalid input.\n')
                sys.exit(1)
        elif (referencepoint == 'n') & (len(CNT_centers) != 0):
            poresonly = ddict.get_input('Do you want to calculte the coordination number only inside a pore? [y/n]', args, 'string')
        else:
            ddict.printLog('Invalid input.\n')
            sys.exit(1)
    elif do_xyz_analysis == 'y':
        number_xbins = int(ddict.get_input('How many bins in x direction? ', args, 'int'))
        number_ybins = int(ddict.get_input('How many bins in y direction? ', args, 'int'))
        number_zbins = int(ddict.get_input('How many bins in z direction? ', args, 'int'))
        xbin_edges = np.linspace(0, box_size[0], number_xbins+1)
        ybin_edges = np.linspace(0, box_size[1], number_ybins+1)
        zbin_edges = np.linspace(0, box_size[2], number_zbins+1)
    else:
        ddict.printLog('Invalid input.\n')
        sys.exit(1)


    # Ask user up to which distance the coordination number should be calculated.
    coord_dist = float(ddict.get_input('Up to which distance should the coordination number be calculated? ', args, 'float'))
    coord_bin_edges = np.linspace(0, coord_dist, 101) 
    coord_bin_edges = coord_bin_edges.round(2)

    # Ask user how the bul should be incremented
    coord_bulk_bin_edges = 0
    if referencepoint == 'y':
        number_of_bulk_increments = int(ddict.get_input('Number of increments? ', args, 'int'))
        if poresonly == 'n':
            coord_bulk_bin_edges = np.linspace(0, max((z_referencepoint[3]-z_referencepoint[2])/2,(z_referencepoint[1]-z_referencepoint[0])/2), number_of_bulk_increments)
        else:
            coord_bulk_bin_edges = np.linspace(0, inputdict['tuberadii'][0], number_of_bulk_increments)
            ddict.printLog(f'Bin size:{coord_bulk_bin_edges[1]-coord_bulk_bin_edges[0]} Ang.\n')
    
    # Initialize an empty DataFrame to store distances for each chunk
    chunk_distances_df = pd.DataFrame(columns=['Frame', 'Species1', 'Molecule1', 'Species2', 'Molecule2', 'Distance'])
    Processed_coord_df = pd.DataFrame() 

    # Prepare output
    outputdict = inputdict
    outputdict['z_referencepoint'] = z_referencepoint
    outputdict['coord_bin_edges'] = coord_bin_edges
    outputdict['coord_bulk_bin_edges'] = coord_bulk_bin_edges
    outputdict['chunk_distances_df'] = chunk_distances_df
    outputdict['referencepoint'] = referencepoint
    outputdict['poresonly'] = poresonly
    outputdict['coord_dist'] = coord_dist
    outputdict['processed_coord_df'] = Processed_coord_df
    outputdict['do_xyz_analysis'] = do_xyz_analysis


    if do_xyz_analysis == 'y':
        outputdict['xbin_edges'] = xbin_edges
        outputdict['ybin_edges'] = ybin_edges
        outputdict['zbin_edges'] = zbin_edges

    return outputdict







def Coord_number_analysis(inputdict):

    if (inputdict['do_xyz_analysis'])=='y':
        outputdict = Coord_number_xyz_analysis(inputdict)
        return outputdict
    elif (inputdict['referencepoint'] == 'y') & (inputdict['poresonly'] == 'y'):
        outputdict = Coord_number_pore_analysis(inputdict)
        return outputdict

    # Get values from inputdict
    min_z_pore = inputdict['min_z_pore'] 
    max_z_pore = inputdict['max_z_pore']
    z_referencepoint = inputdict['z_referencepoint']
    CNT_centers = inputdict['CNT_centers']
    regional = inputdict['regional']
    regions = inputdict['regions']
    referencepoint = inputdict['referencepoint']
    poresonly = inputdict['poresonly']
    split_frame = inputdict['split_frame']
    chunk_distances_df = inputdict['chunk_distances_df']
    box_size = inputdict['box_size']
    coord_dist = inputdict['coord_dist']
    counter = inputdict['counter']


    # Remove any unwanted atoms
    if poresonly == 'y':
        split_frame = split_frame[split_frame['Z'].astype(float) <= max_z_pore[0]]
        split_frame = split_frame[split_frame['Z'].astype(float) >= min_z_pore[0]]

    # Now make an array of all molecules which are outside the tube.
    mol_array = np.array(split_frame['Molecule'].unique())

    # Now drop all atoms in the split_frame which are not in the mol_list.  
    split_frame = split_frame[split_frame['Molecule'].isin(mol_array)]

    # We now calculate the center of mass (COM) for each molecule
    # Convert all values to float (this is needed so that the agg-function works)
    split_frame['X'] = split_frame['X'].astype(float)
    split_frame['Y'] = split_frame['Y'].astype(float)
    split_frame['Z'] = split_frame['Z'].astype(float)
    split_frame['Mass'] = split_frame['Mass'].astype(float)

    # Precompute total mass for each molecule
    total_mass_per_molecule = split_frame.groupby('Molecule')['Mass'].transform('sum')

    # Calculate mass weighted coordinates
    split_frame['X_COM'] = (split_frame['X'] * split_frame['Mass']) / total_mass_per_molecule
    split_frame['Y_COM'] = (split_frame['Y'] * split_frame['Mass']) / total_mass_per_molecule
    split_frame['Z_COM'] = (split_frame['Z'] * split_frame['Mass']) / total_mass_per_molecule

    # Calculate the center of mass for each molecule
    mol_com = split_frame.groupby('Molecule').agg(
        Species=('Species', 'first'),
        X_COM=('X_COM', 'sum'),
        Y_COM=('Y_COM', 'sum'),
        Z_COM=('Z_COM', 'sum')
    ).reset_index()

    # Make dataframe with only the molecules that are analyzed
    if regional == 'y':
        mol_com_reference = mol_com[mol_com['X_COM'].astype(float) >= regions[0]]
        mol_com_reference = mol_com_reference[mol_com_reference['X_COM'].astype(float) <= regions[1]]
        mol_com_reference = mol_com_reference[mol_com_reference['Y_COM'].astype(float) >= regions[2]]
        mol_com_reference = mol_com_reference[mol_com_reference['Y_COM'].astype(float) <= regions[3]]
        mol_com_reference = mol_com_reference[mol_com_reference['Z_COM'].astype(float) >= regions[4]]
        mol_com_reference = mol_com_reference[mol_com_reference['Z_COM'].astype(float) <= regions[5]]
    elif poresonly == 'y':
        mol_com_reference = mol_com[mol_com['Z_COM'] <= (max_z_pore[0] - coord_dist)]
        mol_com_reference = mol_com_reference[mol_com_reference['Z_COM'] >= (min_z_pore[0] + coord_dist)]       
    else:
        mol_com_reference = mol_com

    # We need to account for periodic boundary conditions. Otherwise, molecules close to the box faces will have a CN that
    # is too low.
    mol_com = mirror_PBC(box_size, coord_dist, mol_com)

    # Now we have to calculate the distance between the COM of each molecule in the mol_com_reference and the COM of all other molecules in the mol_com dataframe.
    # The distance are saved, if they are smaller than coord_dist and the species are different.
    distances = np.sqrt(
    (mol_com_reference['X_COM'].values[:, np.newaxis] - mol_com['X_COM'].values) ** 2 +
    (mol_com_reference['Y_COM'].values[:, np.newaxis] - mol_com['Y_COM'].values) ** 2 +
    (mol_com_reference['Z_COM'].values[:, np.newaxis] - mol_com['Z_COM'].values) ** 2
    )    

    # Create a mask to filter the distances
    mask = (mol_com_reference['Species'].values[:, np.newaxis] != mol_com['Species'].values) & (distances <= coord_dist)

    # Get the indices of the filtered distances
    indices = np.where(mask)

    # Create a DataFrame with the results
    data = {
        'Frame': counter + 1,
        'Species1': mol_com_reference['Species'].values[indices[0]],
        'Molecule1': mol_com_reference['Molecule'].values[indices[0]],
        'Species2': mol_com['Species'].values[indices[1]],
        'Molecule2': mol_com['Molecule'].values[indices[1]],
        'Distance': distances[indices],
    }

    if referencepoint == 'y':
        # Calculate the distance to the nearest pore
        distance_to_referencepoint = np.abs(mol_com_reference['Z_COM'].values[:, np.newaxis] - z_referencepoint).min(axis=1)
        data['Distance_to_referencepoint'] = distance_to_referencepoint[indices[0]]

    distances_df = pd.DataFrame(data)

    # add the new values to the dataframe
    chunk_distances_df = pd.concat([chunk_distances_df, distances_df])


    # generate outputdict
    outputdict = inputdict
    outputdict['chunk_distances_df'] = chunk_distances_df

    return outputdict


def mirror_PBC(box_size, coord_dist, mol_com):
    new_entries = []

    # Take care of faces
    for dim in range(3):
        # Calculate distances to box faces in the current dimension
        dist_to_max = box_size[dim] - mol_com.iloc[:, dim + 2]
        dist_to_min = mol_com.iloc[:, dim + 2]

        # Check proximity to box boundaries and mirror molecules into the neighboring box
        close_to_max = dist_to_max < coord_dist
        close_to_min = dist_to_min < coord_dist

        # Mirror molecules close to boundary
        mirrored_max = mol_com.loc[close_to_max].copy()
        mirrored_max[mol_com.columns[dim + 2]] -= box_size[dim]
        new_entries.append(mirrored_max)

        mirrored_min = mol_com.loc[close_to_min].copy()
        mirrored_min[mol_com.columns[dim + 2]] += box_size[dim]
        new_entries.append(mirrored_min)

        # Take care of edges. The if conditions are there to prevent molecules from getting mirrored twice (This is also the only reason
        # why this is in the same loop with the box faces).
        for i in range(3):
            if i != dim:
                for j in range(i + 1, 3):
                    if j != dim:
                        # Calculate distances to box boundaries. For edges we have to do this for two dimensions at the same time
                        dist_to_max_edge_i = box_size[i] - mol_com.iloc[:, i + 2]
                        dist_to_min_edge_i = mol_com.iloc[:, i + 2]
                        dist_to_max_edge_j = box_size[j] - mol_com.iloc[:, j + 2]
                        dist_to_min_edge_j = mol_com.iloc[:, j + 2]

                        # Check proximity to box edges and mirror molecules into the neighboring box
                        close_to_max_edge_i = dist_to_max_edge_i < coord_dist
                        close_to_min_edge_i = dist_to_min_edge_i < coord_dist
                        close_to_max_edge_j = dist_to_max_edge_j < coord_dist
                        close_to_min_edge_j = dist_to_min_edge_j < coord_dist

                        mirrored_max_edge = mol_com.loc[
                            close_to_max_edge_i & close_to_max_edge_j
                        ].copy()
                        mirrored_max_edge[mol_com.columns[i + 2]] -= box_size[i]
                        mirrored_max_edge[mol_com.columns[j + 2]] -= box_size[j]
                        new_entries.append(mirrored_max_edge)

                        mirrored_min_edge = mol_com.loc[
                            close_to_min_edge_i & close_to_min_edge_j
                        ].copy()
                        mirrored_min_edge[mol_com.columns[i + 2]] += box_size[i]
                        mirrored_min_edge[mol_com.columns[j + 2]] += box_size[j]
                        new_entries.append(mirrored_min_edge)

                        mirrored_min_max_edge = mol_com.loc[
                            close_to_min_edge_i & close_to_max_edge_j
                        ].copy()
                        mirrored_min_max_edge[mol_com.columns[i + 2]] += box_size[i]
                        mirrored_min_max_edge[mol_com.columns[j + 2]] -= box_size[j]
                        new_entries.append(mirrored_min_max_edge)

                        mirrored_max_min_edge = mol_com.loc[
                            close_to_max_edge_i & close_to_min_edge_j
                        ].copy()
                        mirrored_max_min_edge[mol_com.columns[i + 2]] -= box_size[i]
                        mirrored_max_min_edge[mol_com.columns[j + 2]] += box_size[j]
                        new_entries.append(mirrored_max_min_edge)

    # Take care of corners. This is super ugly as we have to take care of all the corners separately.
    for i in range(3):
        for j in range(i + 1, 3):
            for k in range(j + 1, 3):
                # Calculate distances to box boundaries
                dist_to_max_corner_i = box_size[i] - mol_com.iloc[:, i + 2]
                dist_to_min_corner_i = mol_com.iloc[:, i + 2]
                dist_to_max_corner_j = box_size[j] - mol_com.iloc[:, j + 2]
                dist_to_min_corner_j = mol_com.iloc[:, j + 2]
                dist_to_max_corner_k = box_size[k] - mol_com.iloc[:, k + 2]
                dist_to_min_corner_k = mol_com.iloc[:, k + 2]

                # Check proximity to box boundaries
                close_to_max_corner_i = dist_to_max_corner_i < coord_dist
                close_to_min_corner_i = dist_to_min_corner_i < coord_dist
                close_to_max_corner_j = dist_to_max_corner_j < coord_dist
                close_to_min_corner_j = dist_to_min_corner_j < coord_dist
                close_to_max_corner_k = dist_to_max_corner_k < coord_dist
                close_to_min_corner_k = dist_to_min_corner_k < coord_dist


                # Now we check for each corner if any molecules are close
                mirrored_max_corner = mol_com.loc[
                    close_to_max_corner_i & close_to_max_corner_j & close_to_max_corner_k
                ].copy()

                # New coordinates for all molecuels in close proximity are calculated and appended to mirrored_max_corner
                mirrored_max_corner[mol_com.columns[i + 2]] -= box_size[i]
                mirrored_max_corner[mol_com.columns[j + 2]] -= box_size[j]
                mirrored_max_corner[mol_com.columns[k + 2]] -= box_size[k]
                new_entries.append(mirrored_max_corner)

                mirrored_min_corner = mol_com.loc[
                    close_to_min_corner_i & close_to_min_corner_j & close_to_min_corner_k
                ].copy()
                mirrored_min_corner[mol_com.columns[i + 2]] += box_size[i]
                mirrored_min_corner[mol_com.columns[j + 2]] += box_size[j]
                mirrored_min_corner[mol_com.columns[k + 2]] += box_size[k]
                new_entries.append(mirrored_min_corner)

                mirrored_min_corner = mol_com.loc[
                    close_to_min_corner_i & close_to_min_corner_j & close_to_max_corner_k
                ].copy()
                mirrored_min_corner[mol_com.columns[i + 2]] += box_size[i]
                mirrored_min_corner[mol_com.columns[j + 2]] += box_size[j]
                mirrored_min_corner[mol_com.columns[k + 2]] -= box_size[k]
                new_entries.append(mirrored_min_corner)

                mirrored_min_corner = mol_com.loc[
                    close_to_min_corner_i & close_to_max_corner_j & close_to_min_corner_k
                ].copy()
                mirrored_min_corner[mol_com.columns[i + 2]] += box_size[i]
                mirrored_min_corner[mol_com.columns[j + 2]] -= box_size[j]
                mirrored_min_corner[mol_com.columns[k + 2]] += box_size[k]
                new_entries.append(mirrored_min_corner)

                mirrored_min_corner = mol_com.loc[
                    close_to_min_corner_i & close_to_max_corner_j & close_to_max_corner_k
                ].copy()
                mirrored_min_corner[mol_com.columns[i + 2]] += box_size[i]
                mirrored_min_corner[mol_com.columns[j + 2]] -= box_size[j]
                mirrored_min_corner[mol_com.columns[k + 2]] -= box_size[k]
                new_entries.append(mirrored_min_corner)

                mirrored_min_corner = mol_com.loc[
                    close_to_max_corner_i & close_to_min_corner_j & close_to_min_corner_k
                ].copy()
                mirrored_min_corner[mol_com.columns[i + 2]] -= box_size[i]
                mirrored_min_corner[mol_com.columns[j + 2]] += box_size[j]
                mirrored_min_corner[mol_com.columns[k + 2]] += box_size[k]
                new_entries.append(mirrored_min_corner)

                mirrored_min_corner = mol_com.loc[
                    close_to_max_corner_i & close_to_min_corner_j & close_to_max_corner_k
                ].copy()
                mirrored_min_corner[mol_com.columns[i + 2]] -= box_size[i]
                mirrored_min_corner[mol_com.columns[j + 2]] += box_size[j]
                mirrored_min_corner[mol_com.columns[k + 2]] -= box_size[k]
                new_entries.append(mirrored_min_corner)

                mirrored_min_corner = mol_com.loc[
                    close_to_max_corner_i & close_to_max_corner_j & close_to_min_corner_k
                ].copy()
                mirrored_min_corner[mol_com.columns[i + 2]] -= box_size[i]
                mirrored_min_corner[mol_com.columns[j + 2]] -= box_size[j]
                mirrored_min_corner[mol_com.columns[k + 2]] += box_size[k]
                new_entries.append(mirrored_min_corner)

                mirrored_min_corner = mol_com.loc[
                    close_to_max_corner_i & close_to_max_corner_j & close_to_max_corner_k
                ].copy()
                mirrored_min_corner[mol_com.columns[i + 2]] -= box_size[i]
                mirrored_min_corner[mol_com.columns[j + 2]] -= box_size[j]
                mirrored_min_corner[mol_com.columns[k + 2]] -= box_size[k]
                new_entries.append(mirrored_min_corner)

                

    # Concatenate new entries to the original DataFrame
    if new_entries:
        new_entries = pd.concat(new_entries, ignore_index=True)
        mol_com = pd.concat([mol_com, new_entries], ignore_index=True)

    return mol_com


def Coord_chunk_processing(inputdict):
    print('Processing Chunk...')

    if (inputdict['do_xyz_analysis'])=='y':
        outputdict = Coord_xyz_chunk_processing(inputdict)
        return outputdict
    elif (inputdict['referencepoint'] == 'y') & (inputdict['poresonly'] == 'y'):
        outputdict = Coord_pore_chunk_processing(inputdict)
        return outputdict

    # Get values from inputdict
    chunk_distances_df = inputdict['chunk_distances_df']
    coord_bin_edges = inputdict['coord_bin_edges']
    coord_bulk_bin_edges = inputdict['coord_bulk_bin_edges']
    chunk_number = inputdict['chunk_number']
    processed_coord_df = inputdict['processed_coord_df']

    # Set up a DataFrame for the results
    coord_df = pd.DataFrame(columns=["Frame","Reference","Observable", "Molecule","Zbin"] + list(coord_bin_edges[:-1]))

    # Extract unique frames
    unique_frames = chunk_distances_df['Frame'].unique()

    counts = np.zeros(100)

    # Loop over each unique frame
    for frame in unique_frames:

        # Filter distances for the current frame
        frame_distances = chunk_distances_df[chunk_distances_df['Frame'] == frame]

        # Create an empty list to store the data
        coord_data = []

        # initialize Zbin (holds the distance to the next structure) now so we do not have to initialize it inside the loop
        Zbin = 0

        for species_pair in frame_distances[['Species1', 'Species2']].drop_duplicates().values:

            # Filter distances for the current species pair in the current frame
            if 'Distance_to_referencepoint' in frame_distances:
                distances = frame_distances.loc[(frame_distances['Species1'] == species_pair[0]) &
                                                (frame_distances['Species2'] == species_pair[1]), ['Molecule1', 'Distance', 'Distance_to_referencepoint']]
            else:
                distances = frame_distances.loc[(frame_distances['Species1'] == species_pair[0]) &
                                                (frame_distances['Species2'] == species_pair[1]), ['Molecule1', 'Distance']]

            # group the data by molecule
            grouped_distances = distances.groupby(['Molecule1'])           

            # Loop over each molecule
            for molecule, molecule_distances in grouped_distances:

                if 'Distance_to_referencepoint' in molecule_distances:
                    # First we take care of the distance to the next pore, as it stays the same for each molecule.
                    # Since all values are identical we can just take the first
                    Distance_to_referencepoint = molecule_distances['Distance_to_referencepoint'].values[0]

                    # Now we sort it into a bin
                    Zbin = coord_bulk_bin_edges[np.digitize(Distance_to_referencepoint, coord_bulk_bin_edges)].round(2)

                # Next we take care of the coo rdination number

                # np.histogram sorts all values in 'molecule_distance['Distance']' into bins defined by 'bins=coord_bin_edges'
                # the [0] at the end gives us a list of all y-values of the generated histograms.
                counts[:] = np.histogram(molecule_distances['Distance'], bins=coord_bin_edges)[0]   

                # Now we calculate the cumulative sum of all bins
                counts = np.cumsum(counts)         

                # Add the data for the current molecule into coord_data
                coord_data.append({
                    "Chunk": chunk_number,
                    "Frame": frame, 
                    "Reference": species_pair[0], 
                    "Observable": species_pair[1], 
                    "Molecule": molecule_distances['Molecule1'].values[0],
                    "Zbin": Zbin,
                    **dict(zip(coord_bin_edges[:-1], counts))
                })

    # convert coord_data into a pandas DataFrame for further processing
    coord_df = pd.DataFrame(coord_data)

    # Combine the Reference and Observable columns
    coord_df['Reference_Observable'] = coord_df['Reference'].astype(str) + '_' + coord_df['Observable'].astype(str)

    # Drop any columns that we do not need anymore
    coord_df = coord_df.drop(columns=['Reference', 'Observable', 'Frame', 'Molecule'])       

    # Reorder columns
    coord_df = coord_df[['Chunk','Reference_Observable', 'Zbin'] + list(coord_bin_edges[:-1])]

    # Set up a DataFrame for the average results. If the column 'Distance_to_referencepoint' exists (meaning we do not calculate the coordination
    # number w.r.t. a reference point), we do not need to keep the Zbins and just average over all collected distances.
    if 'Distance_to_referencepoint' in frame_distances:
        avg_coord_df = coord_df.groupby(['Reference_Observable', 'Zbin'])
    else:
        coord_df = coord_df.drop(columns=('Zbin'))
        avg_coord_df = coord_df.groupby(['Reference_Observable'])

    # Average the results for the current chunk
    avg_coord_df = avg_coord_df.mean().reset_index()

    # Since we need to later average over all chunks we need to save the count of how many datapoints we averaged over. The counts will
    # serve as weights for the last averaging process.
    if 'Distance_to_referencepoint' in frame_distances:
        count_df = coord_df.groupby(['Reference_Observable', 'Zbin']).size().reset_index(name='Count')
    else:
        count_df = coord_df.groupby(['Reference_Observable']).size().reset_index(name='Count')

    # Save the counts in our averaged dataframe
    avg_coord_df['Count'] = count_df['Count']

    # Now we save the processed results in 'processed_coords_df'.
    processed_coord_df = pd.concat([processed_coord_df, avg_coord_df])

    outputdict = inputdict
    outputdict['processed_coord_df'] = processed_coord_df

    # Empty chunk_distances_df to free up memory
    outputdict['chunk_distances_df'] = pd.DataFrame()

    return outputdict


def Coord_post_processing(inputdict):

    if (inputdict['do_xyz_analysis'])=='y':
        outputdict = Coord_xyz_post_processing(inputdict)
        return outputdict
    elif (inputdict['referencepoint'] == 'y') & (inputdict['poresonly'] == 'y'):
        outputdict = Coord_pore_post_processing(inputdict)
        return outputdict

    import matplotlib.pyplot as plt
    import os   

    processed_coord_df = inputdict['processed_coord_df']

    # Average over all Chunks. Each Chunk has to be weighted by the amount of datapoints that are contained in it.
    def weighted_average(group):
        return (group.iloc[:, 3:-1].mul(group['Count'], axis=0).sum() / group['Count'].sum())
    
    # If the column 'Zbin' does not exist in 'processed_coord_df' we dropped it earlier as the user chose to average the coordination
    # number over all collected points (and not w.r.t the distance to a reference point)
    if 'Zbin' in processed_coord_df:
        processed_coord_df = processed_coord_df.groupby(['Reference_Observable', 'Zbin']).apply(weighted_average).reset_index()
    else:
        processed_coord_df = processed_coord_df.groupby(['Reference_Observable']).apply(weighted_average).reset_index()

    # Now we iterate over each species pair and write the data into separate files
    # If an output dir does not exist, we create one
    output_directory = "coord_distance_files"

    # If the directory already exists we do not need to make a new one
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Find minimum and maximum coordination numbers. We need this to make all plots have the same scale
    if 'Zbin' in processed_coord_df:
        dummy_df = processed_coord_df.drop(columns=['Reference_Observable', 'Zbin'])
    else:
        dummy_df = processed_coord_df.drop(columns=['Reference_Observable'])
    min_CN_value = (dummy_df.values).min()
    max_CN_value = (dummy_df.values).max()

    avg_coord_df = processed_coord_df

    for species_pair in avg_coord_df['Reference_Observable'].unique():

        # make a dummy dataframe with values from the current species pair
        dummy_df = avg_coord_df[avg_coord_df['Reference_Observable'] == species_pair]

        # Remove the 'Reference_Observable' column since we do not need it anymore
        dummy_df = dummy_df.drop(columns=['Reference_Observable'])

        if 'Zbin' in avg_coord_df:
            dummy_df.set_index(['Zbin'], inplace=True)
        else:
            dummy_df = dummy_df.transpose()        

        # make the data look nicer by rounding the values
        dummy_df = dummy_df.round(decimals=4)

        # Finally print the data in .csv format
        output_file_path = os.path.join(output_directory, f"{species_pair}.csv")
        dummy_df.to_csv(output_file_path)

        # if the column 'Zbin' exists in the processed data (meaning the user wants to see the coordination number in realtion to 
        # the distance to a reference point), we plot the data as heatmap
        if 'Zbin' in avg_coord_df:
            # Extract the column labels from the first row
            column_labels = dummy_df.columns

            # Extract the row labels (index)
            row_labels = dummy_df.index

            # Extract the values for the heatmap
            values = dummy_df.values

            # Create a heatmap plot
            plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
            plt.imshow(values, cmap='viridis', aspect='auto', origin='lower')

            # Set a common colorbar for all species pairs
            plt.clim(min_CN_value, max_CN_value)
            plt.colorbar(label="Coordination number")  # Add a colorbar

            # Show every fifth tick on the x-axis and y-axis for better readability
            x_ticks = range(0, len(column_labels), 5)
            y_ticks = range(0, len(row_labels), 5)

            plt.xticks(x_ticks, [column_labels[i] for i in x_ticks], rotation=90)
            plt.yticks(y_ticks, [row_labels[i].round(decimals=1) for i in y_ticks])

            plt.xlabel("Coordination distance")
            plt.ylabel("Distance to structure")

            plt.title(" ")

            # Save the plot
            plt.savefig(f"{output_directory}/{species_pair}.png")

    # close all figures
    plt.close('all')




def Coord_number_xyz_analysis(inputdict):

    # Get values from inputdict
    regional = inputdict['regional']
    regions = inputdict['regions']
    split_frame = inputdict['split_frame']
    chunk_distances_df = inputdict['chunk_distances_df']
    box_size = inputdict['box_size']
    coord_dist = inputdict['coord_dist']
    counter = inputdict['counter']

    # Remove any unwanted atoms
    if regional == 'y':
        split_frame = split_frame[split_frame['X'].astype(float) >= regions[0]]
        split_frame = split_frame[split_frame['X'].astype(float) <= regions[1]]
        split_frame = split_frame[split_frame['Y'].astype(float) >= regions[2]]
        split_frame = split_frame[split_frame['Y'].astype(float) <= regions[3]]
        split_frame = split_frame[split_frame['Z'].astype(float) >= regions[4]]
        split_frame = split_frame[split_frame['Z'].astype(float) <= regions[5]]

    # Now make an array of all molecules which are outside the tube.
    mol_array = np.array(split_frame['Molecule'].unique())

    # Now drop all atoms in the split_frame which are not in the mol_list.  
    split_frame = split_frame[split_frame['Molecule'].isin(mol_array)]

    # We now calculate the center of mass (COM) for each molecule
    # Convert all values to float (this is needed so that the agg-function works)
    split_frame['X'] = split_frame['X'].astype(float)
    split_frame['Y'] = split_frame['Y'].astype(float)
    split_frame['Z'] = split_frame['Z'].astype(float)
    split_frame['Mass'] = split_frame['Mass'].astype(float)

    # Precompute total mass for each molecule
    total_mass_per_molecule = split_frame.groupby('Molecule')['Mass'].transform('sum')

    # Calculate mass weighted coordinates
    split_frame['X_COM'] = (split_frame['X'] * split_frame['Mass']) / total_mass_per_molecule
    split_frame['Y_COM'] = (split_frame['Y'] * split_frame['Mass']) / total_mass_per_molecule
    split_frame['Z_COM'] = (split_frame['Z'] * split_frame['Mass']) / total_mass_per_molecule

    # Calculate the center of mass for each molecule
    mol_com = split_frame.groupby('Molecule').agg(
        Species=('Species', 'first'),
        X_COM=('X_COM', 'sum'),
        Y_COM=('Y_COM', 'sum'),
        Z_COM=('Z_COM', 'sum')
    ).reset_index()

    # Make dataframe with only the molecules that are analyzed
    if regional == 'y':
        mol_com_reference = mol_com[mol_com['X_COM'].astype(float) >= regions[0]]
        mol_com_reference = mol_com_reference[mol_com_reference['X_COM'].astype(float) <= regions[1]]
        mol_com_reference = mol_com_reference[mol_com_reference['Y_COM'].astype(float) >= regions[2]]
        mol_com_reference = mol_com_reference[mol_com_reference['Y_COM'].astype(float) <= regions[3]]
        mol_com_reference = mol_com_reference[mol_com_reference['Z_COM'].astype(float) >= regions[4]]
        mol_com_reference = mol_com_reference[mol_com_reference['Z_COM'].astype(float) <= regions[5]]
    else:
        mol_com_reference = mol_com

    # To correctly account for periodic boundary conditions, we add another layer
    # of molecules on each face of the box (also)
    mol_com = mirror_PBC(box_size, coord_dist, mol_com)

    # Now we have to calculate the distance between the COM of each molecule in the mol_com_reference and the COM of all other molecules in the mol_com dataframe.
    # The distance are saved, if they are smaller than coord_dist and the species are different.
    distances = np.sqrt(
    (mol_com_reference['X_COM'].values[:, np.newaxis] - mol_com['X_COM'].values) ** 2 +
    (mol_com_reference['Y_COM'].values[:, np.newaxis] - mol_com['Y_COM'].values) ** 2 +
    (mol_com_reference['Z_COM'].values[:, np.newaxis] - mol_com['Z_COM'].values) ** 2
    )    

    # Create a mask to filter the distances
    mask = (mol_com_reference['Species'].values[:, np.newaxis] != mol_com['Species'].values) & (distances <= coord_dist)

    # Get the indices of the filtered distances
    indices = np.where(mask)

    # Create a DataFrame with the results
    data = {
        'Frame': counter + 1,
        'Species1': mol_com_reference['Species'].values[indices[0]],
        'Molecule1': mol_com_reference['Molecule'].values[indices[0]],
        'X_COM': mol_com_reference['X_COM'].values[indices[0]],
        'Y_COM': mol_com_reference['Y_COM'].values[indices[0]],
        'Z_COM': mol_com_reference['Z_COM'].values[indices[0]],
        'Species2': mol_com['Species'].values[indices[1]],
        'Molecule2': mol_com['Molecule'].values[indices[1]],
        'Distance': distances[indices],
    }

    distances_df = pd.DataFrame(data)

    # add the new values to the dataframe
    chunk_distances_df = pd.concat([chunk_distances_df, distances_df])


    # generate outputdict
    outputdict = inputdict
    outputdict['chunk_distances_df'] = chunk_distances_df

    return outputdict




def Coord_xyz_chunk_processing(inputdict):

    # Get values from inputdict
    chunk_distances_df = inputdict['chunk_distances_df']
    coord_bin_edges = inputdict['coord_bin_edges']
    xbin_edges = inputdict['xbin_edges']
    ybin_edges = inputdict['ybin_edges']
    zbin_edges = inputdict['zbin_edges']
    chunk_number = inputdict['chunk_number']
    processed_coord_df = inputdict['processed_coord_df']

    # Set up a DataFrame for the results
    coord_df = pd.DataFrame(columns=["Frame","Reference","Observable", "Molecule","Xbin","Ybin","Zbin"] + list(coord_bin_edges[:-1]))

    # Extract unique frames
    unique_frames = chunk_distances_df['Frame'].unique()

    counts = np.zeros(100)

    # Loop over each unique frame
    for frame in unique_frames:

        # Filter distances for the current frame
        frame_distances = chunk_distances_df[chunk_distances_df['Frame'] == frame]

        # Create an empty list to store the data
        coord_data = []

        # initialize Zbin (holds the distance to the next structure) now so we do not have to initialize it inside the loop

        for species_pair in frame_distances[['Species1', 'Species2']].drop_duplicates().values:

            distances = frame_distances.loc[(frame_distances['Species1'] == species_pair[0]) &
                                            (frame_distances['Species2'] == species_pair[1]), ['Molecule1', 'Distance', 'X_COM', 'Y_COM', 'Z_COM']]

            # group the data by molecule
            grouped_distances = distances.groupby(['Molecule1'])           

            # Loop over each molecule
            for molecule, molecule_distances in grouped_distances:
                # Sort molecule into XYZ bins
                Xbin = np.digitize(molecule_distances['X_COM'].values[0], xbin_edges)
                Ybin = np.digitize(molecule_distances['Y_COM'].values[0], ybin_edges)
                Zbin = np.digitize(molecule_distances['Z_COM'].values[0], zbin_edges)

                # Next we take care of the coordination number

                # np.histogram sorts all values in 'molecule_distance['Distance']' into bins defined by 'bins=coord_bin_edges'
                # the [0] at the end gives us a list of all y-values of the generated histograms.
                counts[:] = np.histogram(molecule_distances['Distance'], bins=coord_bin_edges)[0]   

                # Now we calculate the cumulative sum of all bins
                counts = np.cumsum(counts)         

                # Add the data for the current molecule into coord_data
                coord_data.append({
                    "Chunk": chunk_number,
                    "Frame": frame, 
                    "Reference": species_pair[0], 
                    "Observable": species_pair[1], 
                    "Molecule": molecule_distances['Molecule1'].values[0],
                    "Xbin": Xbin,
                    "Ybin": Ybin,
                    "Zbin": Zbin,
                    **dict(zip(coord_bin_edges[:-1], counts))
                })

    # convert coord_data into a pandas DataFrame for further processing
    coord_df = pd.DataFrame(coord_data)

    # Combine the Reference and Observable columns
    coord_df['Reference_Observable'] = coord_df['Reference'].astype(str) + '_' + coord_df['Observable'].astype(str)

    # Drop any columns that we do not need anymore
    coord_df = coord_df.drop(columns=['Reference', 'Observable', 'Frame', 'Molecule'])       

    # Reorder columns
    coord_df = coord_df[['Chunk','Reference_Observable', 'Xbin','Ybin', 'Zbin'] + list(coord_bin_edges[:-1])]

    # Set up a DataFrame for the average results. If the column 'Distance_to_referencepoint' exists (meaning we do not calculate the coordination
    # number w.r.t. a reference point), we do not need to keep the Zbins and just average over all collected distances.

    avg_coord_df = coord_df.groupby(['Reference_Observable', 'Xbin', 'Ybin', 'Zbin'])

    # Average the results for the current chunk
    avg_coord_df = avg_coord_df.mean().reset_index()

    # Since we need to later average over all chunks we need to save the count of how many datapoints we averaged over. The counts will
    # serve as weights for the last averaging process.
    count_df = coord_df.groupby(['Reference_Observable', 'Xbin', 'Ybin','Zbin']).size().reset_index(name='Count')

    # Save the counts in our averaged dataframe
    avg_coord_df['Count'] = count_df['Count']

    # Now we save the processed results in 'processed_coords_df'.
    processed_coord_df = pd.concat([processed_coord_df, avg_coord_df])

    outputdict = inputdict
    outputdict['processed_coord_df'] = processed_coord_df

    # Empty chunk_distances_df to free up memory
    outputdict['chunk_distances_df'] = pd.DataFrame()

    return outputdict



def Coord_xyz_post_processing(inputdict):
    import matplotlib.pyplot as plt
    import os   

    processed_coord_df = inputdict['processed_coord_df']
    coord_bin_edges = inputdict['coord_bin_edges']
    xbin_edges = inputdict['xbin_edges']
    ybin_edges = inputdict['ybin_edges']
    zbin_edges = inputdict['zbin_edges']
    
    # Get structures
    structures_df = inputdict['id_frame']
    structures_df = structures_df[structures_df['Struc'] != 'Liquid']

    # Average over all Chunks. Each Chunk has to be weighted by the amount of datapoints that are contained in it.
    def weighted_average(group):
        return (group.iloc[:, 3:-1].mul(group['Count'], axis=0).sum() / group['Count'].sum())
    
    # If the column 'Zbin' does not exist in 'processed_coord_df' we dropped it earlier as the user chose to average the coordination
    # number over all collected points (and not w.r.t the distance to a reference point)
    processed_coord_df = processed_coord_df.groupby(['Reference_Observable', 'Xbin', 'Ybin','Zbin']).apply(weighted_average)
    
    # For some reason there is still a 'Zbin' column after the grouping so we need to drop it before we reset the index
    processed_coord_df = processed_coord_df.drop(columns=['Zbin']).reset_index()


    # Save raw data in csv file
    output_directory = "coord_distance_files"

    # If the directory already exists we do not need to make a new one
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for species_pair in processed_coord_df['Reference_Observable'].unique():

        # make a dummy dataframe with values from the current species pair
        dummy_df = processed_coord_df[processed_coord_df['Reference_Observable'] == species_pair].copy()

        # Remove the 'Reference_Observable' column since we do not need it anymore
        dummy_df = dummy_df.drop(columns=['Reference_Observable'])

        # make the data look nicer by rounding the values
        dummy_df = dummy_df.round(decimals=4)

        # Finally print the data in .csv format
        output_file_path = os.path.join(output_directory, f"{species_pair}.csv")
        dummy_df.to_csv(output_file_path)

    # Now take care of the plotting

    # Since we can only include the CN at a certain distance and not all distances into the isosurface plot we need to 
    # choose one now and drop the rest

    # Create a dataframe that only contains the data that we want to plot
    plot_df = pd.DataFrame()

    # Copy all needed columns
    plot_df['Reference_Observable'] = processed_coord_df['Reference_Observable']
    plot_df['Xbin'] = processed_coord_df['Xbin']
    plot_df['Ybin'] = processed_coord_df['Ybin']
    plot_df['Zbin'] = processed_coord_df['Zbin']

    # Copy the wanted coord distance as coordination number HARDCODED VALUE NOW, DONT FORGET TO CHANGE AFTER TESTING
    plot_df['CN'] = processed_coord_df[coord_bin_edges[99]]

    for species_pair in plot_df['Reference_Observable'].unique():
        
        # make a dummy dataframe with values from the current species pair
        dummy_df = plot_df[plot_df['Reference_Observable'] == species_pair].copy()
        
        # Drop the species identifier
        dummy_df = dummy_df.drop(columns=['Reference_Observable'])
        
        # we need to transfer the data to a 3d numpy array for isosurface plot

        # create an empty datagrid
        grid = np.zeros((len(xbin_edges), len(ybin_edges), len(zbin_edges)))

        for index, datapoint in dummy_df.iterrows():
            grid[int(datapoint['Xbin'])-1, int(datapoint['Ybin'])-1, int(datapoint['Zbin'])-1] = datapoint['CN']

        # Output grid data in Gaussian cube format
        cube_filename = f"{species_pair}_output.cube"
        with open(cube_filename, 'w') as f:

            # The first two lines of a .cube file are comments
            f.write(f"Cube file generated using CONAn\n")
            f.write(f"#\n")

            # The third line contains the number of atoms as well as the coordinates of the origin
            f.write(f"{len(structures_df)} 0.0 0.0 0.0\n")

            # The next three lines give the number of bins along each axis (x, y, z) followed by the axis vector (basically size of the bins).
            # We need to use e.g. len(xbin_edges) - 1 because xbin_edges is just the number of bin EDGES, not the number of BINS.
            f.write(f"{len(xbin_edges)-1} {xbin_edges[1]} 0.0 0.0\n")
            f.write(f"{len(ybin_edges)-1} 0.0 {ybin_edges[1]} 0.0\n")
            f.write(f"{len(zbin_edges)-1} 0.0 0.0 {zbin_edges[1]}\n")
            
            # The next section contains one line for eacht atom. Each line has 5 columns with: Atomic number, Charge and coordinates.
            for index, structure in structures_df.iterrows():
                f.write(f"12 0.0 {structure['x']} {structure['y']} {structure['z']}\n")
            # The last section contains the volumetric data
            for x in range(len(xbin_edges)-1):
                for y in range(len(ybin_edges)-1):
                    for z in range(len(zbin_edges)-1):
                        f.write(f"{grid[x, y, z]} ")
                        if (z) % 6 == 5:  # Limit 6 values per line for better readability
                            f.write("\n")

        print(f"Grid data for {species_pair} has been saved to {cube_filename}.")

    # close all figures
    plt.close('all')


def Coord_number_pore_analysis(inputdict):


    # Get values from inputdict
    min_z_pore = inputdict['min_z_pore'] 
    max_z_pore = inputdict['max_z_pore']
    CNT_centers = inputdict['CNT_centers']
    referencepoint = inputdict['referencepoint']
    split_frame = inputdict['split_frame']
    chunk_distances_df = inputdict['chunk_distances_df']
    box_size = inputdict['box_size']
    coord_dist = inputdict['coord_dist']
    counter = inputdict['counter']

    # Remove any unwanted atoms
    split_frame = split_frame[split_frame['Z'].astype(float) <= max_z_pore[0]]
    split_frame = split_frame[split_frame['Z'].astype(float) >= min_z_pore[0]]

    # Now make an array of all molecules which are outside the tube.
    mol_array = np.array(split_frame['Molecule'].unique())

    # Now drop all atoms in the split_frame which are not in the mol_list.  
    split_frame = split_frame[split_frame['Molecule'].isin(mol_array)]

    # We now calculate the center of mass (COM) for each molecule
    # Convert all values to float (this is needed so that the agg-function works)
    split_frame['X'] = split_frame['X'].astype(float)
    split_frame['Y'] = split_frame['Y'].astype(float)
    split_frame['Z'] = split_frame['Z'].astype(float)
    split_frame['Mass'] = split_frame['Mass'].astype(float)

    # Precompute total mass for each molecule
    total_mass_per_molecule = split_frame.groupby('Molecule')['Mass'].transform('sum')

    # Calculate mass weighted coordinates
    split_frame['X_COM'] = (split_frame['X'] * split_frame['Mass']) / total_mass_per_molecule
    split_frame['Y_COM'] = (split_frame['Y'] * split_frame['Mass']) / total_mass_per_molecule
    split_frame['Z_COM'] = (split_frame['Z'] * split_frame['Mass']) / total_mass_per_molecule

    # Calculate the center of mass for each molecule
    mol_com = split_frame.groupby('Molecule').agg(
        Species=('Species', 'first'),
        X_COM=('X_COM', 'sum'),
        Y_COM=('Y_COM', 'sum'),
        Z_COM=('Z_COM', 'sum')
    ).reset_index()

    # Make dataframe with only the molecules that are analyzed
    mol_com_reference = mol_com[mol_com['Z_COM'] <= (max_z_pore[0] - coord_dist)]
    mol_com_reference = mol_com_reference[mol_com_reference['Z_COM'] >= (min_z_pore[0] + coord_dist)]       

    # We need to account for periodic boundary conditions. Otherwise, molecules close to the box faces will have a CN that
    # is too low.
    mol_com = mirror_PBC(box_size, coord_dist, mol_com)

    # Now we have to calculate the distance between the COM of each molecule in the mol_com_reference and the COM of all other molecules in the mol_com dataframe.
    # The distance are saved, if they are smaller than coord_dist and the species are different.
    distances = np.sqrt(
    (mol_com_reference['X_COM'].values[:, np.newaxis] - mol_com['X_COM'].values) ** 2 +
    (mol_com_reference['Y_COM'].values[:, np.newaxis] - mol_com['Y_COM'].values) ** 2 +
    (mol_com_reference['Z_COM'].values[:, np.newaxis] - mol_com['Z_COM'].values) ** 2
    )    

    # Create a mask to filter the distances
    mask = (mol_com_reference['Species'].values[:, np.newaxis] != mol_com['Species'].values) & (distances <= coord_dist)

    # Get the indices of the filtered distances
    indices = np.where(mask)

    # Create a DataFrame with the results
    data = {
        'Frame': counter + 1,
        'Species1': mol_com_reference['Species'].values[indices[0]],
        'Molecule1': mol_com_reference['Molecule'].values[indices[0]],
        'Species2': mol_com['Species'].values[indices[1]],
        'Molecule2': mol_com['Molecule'].values[indices[1]],
        'Distance': distances[indices],
    }

    DistSeatchInp = {
        'X': mol_com_reference['X_COM'].values[indices[0]],
        'Y': mol_com_reference['Y_COM'].values[indices[0]],
        'Z': mol_com_reference['Z_COM'].values[indices[0]]
    }

    DistInp_df = pd.DataFrame(DistSeatchInp)

    # Calculate the distance to the nearest pore
    # first get the input structure atoms from the id_frame in the inputdict
    structure_atoms = inputdict['id_frame']
    # drop all rows, which are labeled 'Liquid' in the 'Struc' column
    structure_atoms = structure_atoms[structure_atoms['Struc'] != 'Liquid']

    # now transform the CNT atoms into a kd-tree
    structure_atoms_tree = scipy.spatial.KDTree(structure_atoms[['x', 'y', 'z']].values)

    # now get the coordinates of the split_frame
    split_frame_coords = DistInp_df[['X', 'Y', 'Z']].values

    # now query the structure_atoms_tree for the closest atom to each atom in the split_frame
    closest_atom_dist, closest_atom_idx = structure_atoms_tree.query(split_frame_coords)

    data['Distance_to_referencepoint'] = closest_atom_dist

    distances_df = pd.DataFrame(data)

    # add the new values to the dataframe
    chunk_distances_df = pd.concat([chunk_distances_df, distances_df])


    # generate outputdict
    outputdict = inputdict
    outputdict['chunk_distances_df'] = chunk_distances_df

    return outputdict


def Coord_pore_chunk_processing(inputdict):
    print('Processing Chunk...')

    # Get values from inputdict
    chunk_distances_df = inputdict['chunk_distances_df']
    coord_bin_edges = inputdict['coord_bin_edges']
    coord_bulk_bin_edges = inputdict['coord_bulk_bin_edges']
    chunk_number = inputdict['chunk_number']
    processed_coord_df = inputdict['processed_coord_df']

    # Set up a DataFrame for the results
    coord_df = pd.DataFrame(columns=["Frame","Reference","Observable", "Molecule","Zbin"] + list(coord_bin_edges[:-1]))

    # Extract unique frames
    unique_frames = chunk_distances_df['Frame'].unique()

    counts = np.zeros(100)

    # Loop over each unique frame
    for frame in unique_frames:

        # Filter distances for the current frame
        frame_distances = chunk_distances_df[chunk_distances_df['Frame'] == frame]

        # Create an empty list to store the data
        coord_data = []

        # initialize Zbin (holds the distance to the next structure) now so we do not have to initialize it inside the loop
        Zbin = 0

        for species_pair in frame_distances[['Species1', 'Species2']].drop_duplicates().values:

            # Filter distances for the current species pair in the current frame
            distances = frame_distances.loc[(frame_distances['Species1'] == species_pair[0]) &
                                            (frame_distances['Species2'] == species_pair[1]), ['Molecule1', 'Distance', 'Distance_to_referencepoint']]
            # group the data by molecule
            grouped_distances = distances.groupby(['Molecule1'])           

            # Loop over each molecule
            for molecule, molecule_distances in grouped_distances:

                if 'Distance_to_referencepoint' in molecule_distances:
                    # First we take care of the distance to the next pore, as it stays the same for each molecule.
                    # Since all values are identical we can just take the first
                    Distance_to_referencepoint = molecule_distances['Distance_to_referencepoint'].values[0]

                    # Now we sort it into a bin
                    Zbin = coord_bulk_bin_edges[np.digitize(Distance_to_referencepoint, coord_bulk_bin_edges)-1].round(2)

                # Next we take care of the coo rdination number

                # np.histogram sorts all values in 'molecule_distance['Distance']' into bins defined by 'bins=coord_bin_edges'
                # the [0] at the end gives us a list of all y-values of the generated histograms.
                counts[:] = np.histogram(molecule_distances['Distance'], bins=coord_bin_edges)[0]   

                # Now we calculate the cumulative sum of all bins
                counts = np.cumsum(counts)         

                # Add the data for the current molecule into coord_data
                coord_data.append({
                    "Chunk": chunk_number,
                    "Frame": frame, 
                    "Reference": species_pair[0], 
                    "Observable": species_pair[1], 
                    "Molecule": molecule_distances['Molecule1'].values[0],
                    "Zbin": Zbin,
                    **dict(zip(coord_bin_edges[:-1], counts))
                })

    # convert coord_data into a pandas DataFrame for further processing
    coord_df = pd.DataFrame(coord_data)

    # Combine the Reference and Observable columns
    coord_df['Reference_Observable'] = coord_df['Reference'].astype(str) + '_' + coord_df['Observable'].astype(str)

    # Drop any columns that we do not need anymore
    coord_df = coord_df.drop(columns=['Reference', 'Observable', 'Frame', 'Molecule'])       

    # Reorder columns
    coord_df = coord_df[['Chunk','Reference_Observable', 'Zbin'] + list(coord_bin_edges[:-1])]

    # Set up a DataFrame for the average results. If the column 'Distance_to_referencepoint' exists (meaning we do not calculate the coordination
    # number w.r.t. a reference point), we do not need to keep the Zbins and just average over all collected distances.
    if 'Distance_to_referencepoint' in frame_distances:
        avg_coord_df = coord_df.groupby(['Reference_Observable', 'Zbin'])
    else:
        coord_df = coord_df.drop(columns=('Zbin'))
        avg_coord_df = coord_df.groupby(['Reference_Observable'])

    # Average the results for the current chunk
    avg_coord_df = avg_coord_df.mean().reset_index()

    # Since we need to later average over all chunks we need to save the count of how many datapoints we averaged over. The counts will
    # serve as weights for the last averaging process.
    if 'Distance_to_referencepoint' in frame_distances:
        count_df = coord_df.groupby(['Reference_Observable', 'Zbin']).size().reset_index(name='Count')
    else:
        count_df = coord_df.groupby(['Reference_Observable']).size().reset_index(name='Count')

    # Save the counts in our averaged dataframe
    avg_coord_df['Count'] = count_df['Count']

    # Now we save the processed results in 'processed_coords_df'.
    processed_coord_df = pd.concat([processed_coord_df, avg_coord_df])

    outputdict = inputdict
    outputdict['processed_coord_df'] = processed_coord_df

    # Empty chunk_distances_df to free up memory
    outputdict['chunk_distances_df'] = pd.DataFrame()

    return outputdict



def Coord_pore_post_processing(inputdict):
    import matplotlib.pyplot as plt
    import os   

    processed_coord_df = inputdict['processed_coord_df']
    coord_bulk_bin_edges = inputdict['coord_bulk_bin_edges']

    # Average over all Chunks. Each Chunk has to be weighted by the amount of datapoints that are contained in it.
    def weighted_average(group):
        return (group.iloc[:, 3:-1].mul(group['Count'], axis=0).sum() / group['Count'].sum())
    
    # If the column 'Zbin' does not exist in 'processed_coord_df' we dropped it earlier as the user chose to average the coordination
    # number over all collected points (and not w.r.t the distance to a reference point)
    processed_coord_df = processed_coord_df.groupby(['Reference_Observable', 'Zbin']).apply(weighted_average).reset_index()

    # Now we iterate over each species pair and write the data into separate files
    # If an output dir does not exist, we create one
    output_directory = "coord_distance_files"

    # If the directory already exists we do not need to make a new one
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Find minimum and maximum coordination numbers. We need this to make all plots have the same scale
    if 'Zbin' in processed_coord_df:
        dummy_df = processed_coord_df.drop(columns=['Reference_Observable', 'Zbin'])
    else:
        dummy_df = processed_coord_df.drop(columns=['Reference_Observable'])
    min_CN_value = (dummy_df.values).min()
    max_CN_value = (dummy_df.values).max()

    avg_coord_df = processed_coord_df

    for species_pair in avg_coord_df['Reference_Observable'].unique():

        # make a dummy dataframe with values from the current species pair
        dummy_df = avg_coord_df[avg_coord_df['Reference_Observable'] == species_pair]

        # Remove the 'Reference_Observable' column since we do not need it anymore
        dummy_df = dummy_df.drop(columns=['Reference_Observable'])

        if 'Zbin' in avg_coord_df:
            dummy_df.set_index(['Zbin'], inplace=True)
        else:
            dummy_df = dummy_df.transpose()        

        # make the data look nicer by rounding the values
        dummy_df = dummy_df.round(decimals=4)

        # Finally print the data in .csv format
        output_file_path = os.path.join(output_directory, f"{species_pair}.csv")
        dummy_df.to_csv(output_file_path)

        # if the column 'Zbin' exists in the processed data (meaning the user wants to see the coordination number in realtion to 
        # the distance to a reference point), we plot the data as heatmap
        if 'Zbin' in avg_coord_df:
            # Extract the column labels from the first row
            column_labels = dummy_df.columns

            # Extract the row labels (index)
            row_labels = dummy_df.index

            # Extract the values for the heatmap
            values = dummy_df.values

            # Create a heatmap plot
            plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
            plt.imshow(values, cmap='viridis', aspect='auto', origin='lower')

            # Set a common colorbar for all species pairs
            plt.clim(min_CN_value, max_CN_value)
            plt.colorbar(label="Coordination number")  # Add a colorbar

            # Show every fifth tick on the x-axis and y-axis for better readability
            x_ticks = range(0, len(column_labels), 5)
            y_ticks = range(0, len(row_labels))

            plt.xticks(x_ticks, [column_labels[i] for i in x_ticks], rotation=90)
            plt.yticks(y_ticks, [row_labels[i].round(decimals=1) for i in y_ticks])

            plt.xlabel("Coordination distance")
            plt.ylabel("Distance to Pore Wall")

            plt.title(" ")

            # Save the plot
            plt.savefig(f"{output_directory}/{species_pair}.png")

    # close all figures
    plt.close('all')
