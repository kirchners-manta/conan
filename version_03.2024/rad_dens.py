import numpy as np
import pandas as pd
import sys
import math
import matplotlib.pyplot as plt

import defdict as ddict



# Radial density 

def raddens_prep(inputdict):
    CNT_centers = inputdict['CNT_centers']
    tuberadii = inputdict['tuberadii']
    number_of_frames = inputdict['number_of_frames']
    args = inputdict['args']
    ddict.printLog('')
    if len(CNT_centers) > 1:
        ddict.printLog('-> Multiple CNTs detected. The analysis will be conducted on the first CNT.\n', color = 'red')
    if len(CNT_centers) == 0:
        ddict.printLog('-> No CNTs detected. Aborting...\n', color = 'red')
        sys.exit(1)
    for i in range(len(CNT_centers)):
        ddict.printLog(f'\n-> CNT{i + 1}')
        num_increments = int(ddict.get_input('How many increments do you want to use to calculate the density profile? ', args, 'int'))
        rad_increment = tuberadii[i] / num_increments
        # Make an array which start at 0 and end at tuberadius with num_increments + 1 steps.
        raddens_bin_edges = np.linspace(0, tuberadii[0], num_increments + 1)
        # Define raddens_bin_labels, they are a counter for the bin edges.
        raddens_bin_labels = np.arange(1, len(raddens_bin_edges), 1)
        ddict.printLog('Increment distance: %0.3f angstrom' % (rad_increment))
    raddens_bin_labels = np.arange(0, num_increments, 1)	
    # Make new dataframe with the number of frames of the trajectory.
    raddens_df_dummy = pd.DataFrame(np.arange(1,number_of_frames + 1),columns = ['Frame'])            
    # Add a column to the dataframe for each increment.
    for i in range(num_increments):        
        raddens_df_dummy['Bin %d' % (i+1)] = 0
        raddens_df_dummy = raddens_df_dummy.copy()
    raddens_df = raddens_df_dummy.copy()

     # Prepare output dict
    outputdict = {
        'rad_increment': rad_increment,
        'raddens_bin_edges': raddens_bin_edges,
        'raddens_bin_labels': raddens_bin_labels,
        'raddens_df': raddens_df,
        'num_increments': num_increments
    }

    id_frame = inputdict['id_frame']
    analysis_choice2 = inputdict['analysis_choice2']
    if analysis_choice2 == 2:
        #check if the charge column is empty (None) of if there are actual values given
        if id_frame['Charge'].isnull().values.all():
            ddict.printLog('-> No charges detected.')
            charge_add = ddict.get_input('Do you want to add charges? (y/n) ', args, 'string')
            if charge_add == 'y':
                #first make a new column with the combination of 'Struc'_'Species'_'Label'
                id_frame['Struc_Species_Label'] = id_frame['Struc'].astype(str) + '_' + id_frame['Species'].astype(str) + '_' + id_frame['Label'].astype(str)
                id_frame['Struc_Species_Label'] = id_frame['Struc_Species_Label'].astype(str)
                # then we find all unique labels
                unique_labels = id_frame['Struc_Species_Label'].unique()
                #convert unique_labels back to a pandas Series
                unique_labels_series = pd.Series(unique_labels)
                #throw away all entries that are not liquid
                unique_labels = unique_labels_series[unique_labels_series.str.contains('Liquid')]
                # now we loop over all unique labels and ask the user for a charge
                charge_dict = {}
                for label in unique_labels:
                    charge = ddict.get_input(f'What is the charge of {label}? ', args, 'float')
                    charge_dict[label] = charge
                # now we loop over all liquid atoms in the system and assign the charge to the atom
                for label in unique_labels:
                    id_frame.loc[id_frame['Struc_Species_Label'] == label, 'Charge'] = charge_dict[label]
                    #all remaining entries in the column which are not defined yet are set to 0
                    id_frame['Charge'].fillna(0, inplace = True)
                print(id_frame)
                # now we remove the column 'Struc_Species_Label'
                id_frame.drop('Struc_Species_Label', axis = 1, inplace = True)
        else:
            ddict.printLog('-> Charges detected.\n')


     # Prepare output dict
    outputdict = {
        'rad_increment': rad_increment,
        'raddens_bin_edges': raddens_bin_edges,
        'raddens_bin_labels': raddens_bin_labels,
        'raddens_df': raddens_df,
        'num_increments': num_increments,
        'atom_charges': id_frame['Charge'].values,
    }

    # Load inputdict into new dict
    outputdict.update(**inputdict)

    return outputdict

def radial_density_analysis(inputdict):

    split_frame = inputdict['split_frame']
    raddens_df = inputdict['raddens_df']
    raddens_bin_edges = inputdict['raddens_bin_edges']
    raddens_bin_labels = inputdict['raddens_bin_labels']
    num_increments = inputdict['num_increments']
    counter = inputdict['counter']
    CNT_centers = inputdict['CNT_centers']
    max_z_pore = inputdict['max_z_pore']
    min_z_pore = inputdict['min_z_pore']

    split_frame = split_frame[split_frame['Z'].astype(float) <= max_z_pore[0]]   
    split_frame = split_frame[split_frame['Z'].astype(float) >= min_z_pore[0]]

    # Calculate the radial density function with the remaining atoms.
    split_frame['X_adjust'] = split_frame['X'].astype(float) - CNT_centers[0][0]
    split_frame['Y_adjust'] = split_frame['Y'].astype(float) - CNT_centers[0][1]
    # Calculate the distance of each atom to the center of the CNT.
    split_frame['Distance'] = np.sqrt(split_frame['X_adjust'] ** 2 + split_frame['Y_adjust'] ** 2)
    split_frame['Distance_bin'] = pd.cut(split_frame['Distance'], bins = raddens_bin_edges, labels = raddens_bin_labels+1)

    # Add all masses of the atoms in each bin to the corresponding bin.
    raddens_df_temp = split_frame.groupby(pd.cut(split_frame['Distance'], raddens_bin_edges))['Mass'].sum().reset_index(name = 'Weighted_counts')
    raddens_df_temp = pd.DataFrame(raddens_df_temp)

    # Add a new first column with the index+1 of the bin.
    raddens_df_temp.insert(0, 'Bin', raddens_df_temp.index+1)          

    # Write the results into the raddens_df dataframe. The row is defined by the frame number.
    for i in range(num_increments):
        raddens_df.loc[counter,'Bin %d' % (i + 1)] = raddens_df_temp.loc[i,'Weighted_counts']

    # Remove the raddens_df_temp dataframe every loop.
    del raddens_df_temp    
    # Prepare output dict
    outputdict = inputdict
    outputdict['raddens_df'] = raddens_df

    return outputdict

def raddens_post_processing(inputdict):


    raddens_df = inputdict['raddens_df']
    raddens_bin_edges = inputdict['raddens_bin_edges']
    number_of_frames = inputdict['number_of_frames']
    analysis_choice2 = inputdict['analysis_choice2']
    args = inputdict['args']
    tuberadii = inputdict['tuberadii']
    length_pore = inputdict['length_pore']

    CNT_length = length_pore[0]
    radius_tube = tuberadii[0]

    # Ceck the analysis choice. -> If mass or charge density is plotted.
    if analysis_choice2 == 1:
        choice='Mass'
    elif analysis_choice2 == 2:
        choice = 'Charge'
    ddict.printLog('')
    results_rd_df = pd.DataFrame(raddens_df.iloc[:,1:].sum(axis = 0) / number_of_frames)
    results_rd_df.columns = [choice]

    # Add the bin edges to the results_rd_df dataframe.
    results_rd_df['Bin_lowedge'] = raddens_bin_edges[:-1]
    results_rd_df['Bin_highedge'] = raddens_bin_edges[1:]
    
    # The center of the bin is the average of the bin edges.
    results_rd_df['Bin_center'] = (raddens_bin_edges[1:] + raddens_bin_edges[:-1]) / 2                
    
    # Calculate the Volume of each bin. By setting the length of the CNT as length of a cylinder, and the radius of the bin as the radius of the cylinder.
    # Subtract the volume of the smaller cylinder from the volume of the larger cylinder (next larger bin). The volume of a cylinder is pi*r^2*h.
    vol_increment=math.pi * (raddens_bin_edges[1:] ** 2 - raddens_bin_edges[:-1] ** 2) * CNT_length     
    results_rd_df['Volume'] = vol_increment
    
    if choice == 'Mass':
        # Calculate the density of each bin by dividing the average mass by the volume.                                             
        results_rd_df['Density [u/Ang^3]'] = results_rd_df[choice] / results_rd_df['Volume']
        # Calculate the density in g/cm^3.   
        results_rd_df['Density [g/cm^3]'] = results_rd_df['Density [u/Ang^3]'] * 1.66053907  

    if choice == 'Charge':
        # Calculate the charge density in e/Ang^3.
        results_rd_df['Charge density [e/Ang^3]'] = results_rd_df[choice] / results_rd_df['Volume'] 

    # Reset the index of the dataframe.
    results_rd_df.reset_index(drop = True, inplace = True)   
    # Add a new first column with the index+1 of the bin.
    results_rd_df.insert(0, 'Bin', results_rd_df.index + 1)  

    # Now for the initial raw_data frame raddens_df -> Make new dataframe with the mass/charge densities for each bin.
    # To do this we divide the mass/charge of each bin (column in raddens_df) by the volume of the bin it is in (row in results_rd_df).
    raddens_df_density = pd.DataFrame()
    for i in range(len(results_rd_df)):
        raddens_df_density[i] = raddens_df.iloc[:,i+1] / results_rd_df.loc[i, 'Volume']
    raddens_df_density = raddens_df_density.copy()

    # Calculate the variance for all columns in raddens_df_density.
    results_rd_df['Variance'] = pd.DataFrame(raddens_df_density.var(axis = 0))
    # Calculate the standard deviation for all columns in raddens_df_density.
    results_rd_df['Standard dev.'] = pd.DataFrame(raddens_df_density.std(axis = 0))
    # Calculate the standard error for all columns in raddens_df_density.
    results_rd_df['Standard error'] = pd.DataFrame(raddens_df_density.sem(axis = 0))

    # Change the column names to the according bins, as in raddens_df and add the frame number.
    raddens_df_density.columns = raddens_df.columns[1:]
    raddens_df_density.insert(0, 'Frame', raddens_df['Frame'])

    # Plot the data.
    raddens_data_preparation = ddict.get_input('Do you want to plot the data? (y/n) ', args, 'string')
    # Adjusting the resulting dataframe for plotting, as the user set it.
    if raddens_data_preparation == 'y':
        results_rd_df_copy = results_rd_df.copy()

        # Normalization of the data.
        normalize=ddict.get_input('Do you want to normalize the increments with respect to the CNTs\' radius? (y/n) ', args, 'string')
        if normalize == 'y':
            results_rd_df['Bin_center'] = results_rd_df['Bin_center'] / radius_tube

        # Mirroring the data.
        mirror=ddict.get_input('Do you want to mirror the plot? (y/n) ', args, 'string')
        if mirror == 'y':
            # Mirror the data by multiplying the bin center by -1. Then sort the dataframe by the bin center values and combine the dataframes.
            results_rd_dummy = results_rd_df.copy()
            results_rd_dummy['Bin_center'] = results_rd_df['Bin_center'] * (-1)               
            results_rd_dummy.sort_values(by = ['Bin_center'], inplace = True)
            results_rd_df = pd.concat([results_rd_dummy, results_rd_df], ignore_index=True)                    

        # Generate the plot.
        fig, ax = plt.subplots()
        if choice == 'Mass':
            ax.plot(results_rd_df['Bin_center'], results_rd_df['Density [g/cm^3]'], '-' , label = 'Radial density function', color = 'black')
            ax.set(xlabel = 'Distance from tube center [Ang]', ylabel = 'Density [g/cm^3]', title = 'Radial density function')
            
        if choice == 'Charge':
            ax.plot(results_rd_df['Bin_center'], results_rd_df['Charge density [e/Ang^3]'], '-' , label = 'Radial density function', color = 'black')
            ax.set(xlabel = 'Distance from tube center [Ang]', ylabel = 'Charge density [e/Ang^3]', title = 'Radial density function')

        ax.grid()  
        fig.savefig("Radial_density_function.pdf")
        ddict.printLog('-> Radial density function saved as Radial_density_function.pdf\n')

        # Save the data.
        results_rd_df.to_csv('Radial_density_function.csv', sep = ';', index = False, header = True, float_format = '%.5f')

        # Radial contour plot.
        results_rd_df = results_rd_df_copy.copy()
        radial_plot = ddict.get_input('Do you also want to create a radial countour plot? (y/n) ', args, 'string')
        if radial_plot == 'y':
            theta = np.linspace(0, 2 * np.pi, 500)
            if normalize == 'n':
                r = np.linspace(0, radius_tube, len(results_rd_df['Bin_center']))
            else:
                r = np.linspace(0, 1, len(results_rd_df['Bin_center']))
                
            Theta, R = np.meshgrid(theta, r)

            if choice == 'Mass':
                values = np.tile(results_rd_df['Density [g/cm^3]'].values,
                                (len(theta), 1)).T
            elif choice == 'Charge':
                values = np.tile(results_rd_df['Charge density [e/Ang^3]'].values,
                                (len(theta), 1)).T

            fig = plt.figure()
            ax = fig.add_subplot(projection='polar')  
            c = ax.contourf(Theta, R, values, cmap='Reds')  

            ax.spines['polar'].set_visible(False)  

            # Remove angle labels.
            ax.set_xticklabels([]) 

            # Set radial gridlines and their labels.
            r_ticks = np.linspace(0, radius_tube if normalize == 'n' else 1, 5)
            ax.set_yticks(r_ticks)  
            ax.set_yticklabels(['{:.2f}'.format(x) for x in r_ticks])

            # Set the position of radial labels.
            ax.set_rlabel_position(22.5)  
            ax.grid(color = 'black', linestyle = '--', alpha = 0.5)  
            
            # Set title and add a colorbar.
            plt.title('Radial Density Contour Plot', fontsize = 20, pad = 20)
            cbar = fig.colorbar(c, ax = ax, pad = 0.10, fraction = 0.046, orientation = "horizontal")

            if choice == 'Mass':
                cbar.set_ticklabels(['{:.2f}'.format(x) for x in cbar.get_ticks()])
                cbar.set_label(r'Mass density $[g/cm^{3}]$', fontsize = 15)

            elif choice == 'Charge':
                cbar.set_ticklabels(['{:.3f}'.format(x) for x in cbar.get_ticks()])
                cbar.set_label(r'Charge density $[e/Ang^{3}]$', fontsize = 15)

            # Set the y label.
            if normalize == 'n':
                ax.set_ylabel(r'$d_{rad}$', labelpad = 10, fontsize = 20)  
            else:
                ax.set_ylabel(r'$d_{rad}$/$r_{CNT}$', labelpad = 10, fontsize = 20)

            # Save the data.
            fig.savefig("Radial_density_function_polar.pdf")
            ddict.printLog('-> Radial density function countour plot saved as Radial_density_function_polar.pdf\n')

    raw_data = ddict.get_input('Do you want to save the raw data? (y/n) ', args, 'string')
    if raw_data == 'y':
        raddens_df.to_csv('Radial_mass_dist_raw.csv', sep = ';', index = False, header = True, float_format = '%.5f')
        ddict.printLog('Raw mass distribution data saved as Radial_mass_dist_raw.csv')
        raddens_df_density.to_csv('Radial_density_raw.csv', sep = ';', index = False, header = True, float_format = '%.5f')
        ddict.printLog('Raw density data saved as Radial_density_raw.csv')    



# Radial charge density analysis

def radial_charge_density_analysis(inputdict):

    split_frame = inputdict['split_frame']
    raddens_df = inputdict['raddens_df']
    raddens_bin_edges = inputdict['raddens_bin_edges']
    raddens_bin_labels = inputdict['raddens_bin_labels']
    num_increments = inputdict['num_increments']
    counter = inputdict['counter']
    CNT_centers = inputdict['CNT_centers']
    max_z_pore = inputdict['max_z_pore']
    min_z_pore = inputdict['min_z_pore']

    split_frame = split_frame[split_frame['Z'].astype(float) <= max_z_pore[0]]   
    split_frame = split_frame[split_frame['Z'].astype(float) >= min_z_pore[0]]

    # Calculate the radial density function with the remaining atoms.
    split_frame['X_adjust'] = split_frame['X'].astype(float) - CNT_centers[0][0]
    split_frame['Y_adjust'] = split_frame['Y'].astype(float) - CNT_centers[0][1]
    
    # Calculate the distance of each atom to the center of the CNT.
    split_frame['Distance'] = np.sqrt(split_frame['X_adjust'] ** 2 + split_frame['Y_adjust'] ** 2)
    split_frame['Distance_bin'] = pd.cut(split_frame['Distance'], bins = raddens_bin_edges, labels = raddens_bin_labels+1)

    # Add all masses of the atoms in each bin to the corresponding bin.
    raddens_df_temp = split_frame.groupby(pd.cut(split_frame['Distance'], raddens_bin_edges))['Charge'].sum().reset_index(name='Weighted_counts')
    raddens_df_temp = pd.DataFrame(raddens_df_temp)

    # Add a new first column with the index+1 of the bin.
    raddens_df_temp.insert(0, 'Bin', raddens_df_temp.index+1)      

    # Write the results into the raddens_df dataframe. The row is defined by the frame number.
    for i in range(num_increments):
        raddens_df.loc[counter,'Bin %d' % (i + 1)] = raddens_df_temp.loc[i,'Weighted_counts']

    # Remove the raddens_df_temp dataframe every loop.
    del raddens_df_temp    


    # Prepare output dict
    outputdict = inputdict
    outputdict['raddens_df'] = raddens_df

    return outputdict


