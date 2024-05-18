import math
import os
import defdict as ddict
import copy
import sys
import random
import numpy as np
import pandas as pd
import analysis_modules.utils as utils

class Atom:
    # INTERFACE
    def find_adjacent_groups(self,group_list,van_der_waals_radii_dict):

        # if the group list is empty, there is nothing to do
        if not group_list:
            return

        # set the adjacent_groups lsit to empty, so we can fill it
        self.adjacent_groups = []
        # now we loop over all groups
        for group in group_list:
            if group == self:
                continue # current group cannot be its own neighbor
            # next we check if the current group is an Atom, if yes we determine if its a neighbor
            elif isinstance(group,Atom):
                if self._is_atom_neighbor(group,van_der_waals_radii_dict):
                    self.adjacent_groups.append(group)
            else:
                ddict.printLog(f"WARNING: Unknown group type {group} found in 'find_adjacent_groups'")

    # CONSTRUCTOR
    def __init__(self,label,x,y,z):
        self.label = label
        self.x = x
        self.y = y
        self.z = z
        self.adjacent_groups = None

    # PRIVATE
    def _is_atom_neighbor(self,atom,van_der_waals_radii_dict):
        pass

class Functional_Group:

    # INTERFACE
    def remove_anchors(self):
        atom_positions_without_anchor = self.atom_positions
        for position in self.atom_positions:
            if position[0] == 'X':
                atom_positions_without_anchor.remove(position)
        return atom_positions_without_anchor

    # CONSTRUCTOR
    def __init__(self,group_parameters,structure_library_path):
        self.group_name = group_parameters['group']
        self.group_count = int(group_parameters['group_count'])
        if 'exclusion_radius' in group_parameters:
            self.exclusion_radius = float(group_parameters['exclusion_radius'])
        self.atom_positions = self.__read_positions_from_library(structure_library_path)

    # PRIVATE
    def __read_positions_from_library(self,structure_library_path):
        group_path = os.path.join(structure_library_path, f'{self.group_name}.xyz')
        atom_list = []
        # Open and read the library file
        with open(group_path, 'r') as file:
            number_of_lines = int(file.readline().strip())
            file.readline().strip() #skip comment line
            for i in range(number_of_lines):
                line = file.readline().strip().split()
                atom_list.append([line[0],float(line[1]),float(line[2]),float(line[3])])
        return atom_list
    
class Structure:
    
    # INTERFACE
    def print_xyz_file(self,file_name):
        self.__xyz_file(file_name,self._structure_df)

    def remove_atom_by_index(self,index):
        self._structure_df.drop([index], inplace=True)
    
    # PRIVATE
    def __xyz_file(self, name_output, coordinates) -> None:
        directory = "structures"
        filename = f"{name_output}.xyz"
        filepath = os.path.join(directory, filename)
        os.makedirs(directory, exist_ok=True)
        with open(filepath, "w") as xyz:
            xyz.write(f"   {len(coordinates)}\n")
            xyz.write("#Generated with CONAN\n")
            coordinates.to_csv(xyz, sep='\t', header=False, index=False, float_format='%.3f')
    
    def _initialize_functional_groups(self,parameters):

        # Depending on whether build_main is called from CONAN.py
        # or as standalone module, the structure library is somewhere else
        base_path = os.path.dirname(os.path.abspath(__file__))
        structure_library_path = os.path.join(base_path, 'structure_lib')
        if not os.path.exists(structure_library_path):
            structure_library_path = os.path.join(base_path, '..', 'structure_lib')
        structure_library_path = os.path.normpath(structure_library_path)

        self.group_list = []
        self.group_list.append(Functional_Group(parameters,structure_library_path))  

class Structure_1D(Structure):

    def stack(self,parameters,keywords):
        if not self._structure_df is None: # Check if any structure has been loaded
            self._stack_CNTs(parameters,keywords)

    # INTERFACE
    def add(self,parameters,keywrods):
        parameters['group_count'] = 1
        self._initialize_functional_groups(parameters)
        position = [self._structure_df.iloc[parameters['position'],1],
                    self._structure_df.iloc[parameters['position'],2],
                    self._structure_df.iloc[parameters['position'],3]]
        self.__add_group_on_position(position)

    # CONSTRUCTOR
    def __init__(self,parameters,keywords):
        self._build_CNT(parameters,keywords)
        self.bond_length=parameters['bond_length']

    # PRIVATE
    def __add_group_on_position(self,selected_position):
        added_group = self.group_list[0].remove_anchors()
        # give the group a random orientation first
        new_atom_coordinates = random_rotate_group_list(added_group.copy())
        
        # find the right orientation relative to local surface
        normal_vector = self.find_surface_normal_vector(selected_position)
        rotation_matrix = self.rotation_matrix_from_vectors(normal_vector)

        # finally rotate the group
        rotated_coordinates = []
        for atom in new_atom_coordinates:
            atom_coords = np.array(atom[1:4], dtype=float) # ensure that atom_coords has the right datatype
            rotated_coord = np.dot(rotation_matrix, atom_coords)
            rotated_coordinates.append([atom[0],rotated_coord[0],rotated_coord[1],rotated_coord[2],'functional'])

        # shift the coordinates to the selected position
        for atom in rotated_coordinates:
            atom[1] += selected_position[0]
            atom[2] += selected_position[1]
            atom[3] += selected_position[2]

        new_atoms_df = pd.DataFrame(rotated_coordinates, columns=['Species','x','y','z','group'])  # Update columns as needed
        self._structure_df = pd.concat([self._structure_df,new_atoms_df])

    def find_surface_normal_vector(self,position):

        surface_atoms = []
        # find adjacent atoms
        for i, atom in self._structure_df.iterrows():
            delta_x = atom['x']-position[0]
            delta_y = atom['y']-position[1]
            distance = math.sqrt((delta_x)**2+(delta_y)**2+(atom['z']-position[2])**2)
            if distance <= self.bond_length*1.2:
                # we append the mirrored atom and not the atom itself, since
                # atoms that are mirrored due to periodic boundary conditions would
                # make the averages later useless if we take the positions directly
                if distance >= 0.05: # We do not want to add the selected position itself
                    surface_atoms.append([atom['x'],atom['y'],atom['z']]) 

        # compute average position
        surface_atoms = np.array(surface_atoms)
        average_position = np.average(surface_atoms, axis=0)

        # this only works on curved surface (selected position and surface atoms are NOT in one plane)
        # on flat surfaces we have to use a different algorithm

        # compute normal vector
        position=np.array(position)
        normal_vector = average_position-position
        normal_magnitude = np.linalg.norm(normal_vector)
        normal_vector /= normal_magnitude

        return normal_vector
    
    def rotation_matrix_from_vectors(self,vec2):
        """ Find the rotation matrix that aligns vec1 to vec2 """
        vec1 = np.array([0,0,1])
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    def _build_CNT(self,parameters,keywords):

        if 'armchair' in keywords:
            tube_kind = 1
        elif 'zigzag' in keywords:
            tube_kind = 2
        else:
            ddict.printLog("No valid tube kind found in arguments, use 'zigzag' or 'armchair'")
            return None        

        tube_size=parameters['tube_size']
        tube_length=parameters['tube_length']

        # Load the provided distances and bond lengths.
        distance = float(parameters['bond_length'])
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
        self.radius = radius
        self._structure_df = pd.DataFrame(positions_tube)
        self._structure_df.insert(0, "Species", "C")
        self._structure_df.insert(4, "group", "Structure")
        self._structure_df.insert(5, "Molecule", int(1))
        self._structure_df.columns.values[1] = "x"
        self._structure_df.columns.values[2] = "y"
        self._structure_df.columns.values[3] = "z"
        self._structure_df.insert(6,"Label","X")
        counter=1
        for i,atom in self._structure_df.iterrows():
            self._structure_df.at[i,'Label'] = f"C{counter}"
            counter=counter+1
        return self._structure_df

    def _stack_CNTs(self,parameters,keywords):

        distance_tubes = parameters['tube_distance']
        radius_distance = self.radius + distance_tubes/2

        positions_tube = self._structure_df

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

        # Check all atoms in the positions_tube dataframe and shift all atoms that are outside the unit cell to the inside of the unit cell.
        for i in range(0, len(unit_cell)):
            if unit_cell.iloc[i,1] > unit_cell_x:
                unit_cell.iloc[i,1] = unit_cell.iloc[i,1] - unit_cell_x
            if unit_cell.iloc[i,1] < 0:
                unit_cell.iloc[i,1] = unit_cell.iloc[i,1] + unit_cell_x
            if unit_cell.iloc[i,2] > unit_cell_y:
                unit_cell.iloc[i,2] = unit_cell.iloc[i,2] - unit_cell_y
            if unit_cell.iloc[i,2] < 0:
                unit_cell.iloc[i,2] = unit_cell.iloc[i,2] + unit_cell_y

        # Now multiply the unit cell in x and y direction to fill the whole simulation box.
        multiplicity_x = parameters['multiplicity'][0]
        multiplicity_y = parameters['multiplicity'][1]

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
        positions_tube = super_cell.copy()
        positions_tube = pd.DataFrame(positions_tube)
        
        # now within the dataframe, the molcules need to be sorted by the following criteria: Species number -> Molecule number -> Label.
        # The according column names are 'Species', 'Molecule' and 'Label'. The first two are floats, the last one is a string.
        # In case of the label the sorting should be done like C1, C2, C3, ... C10, C11, ... C100, C101, ... C1000, C1001, ...
        # Extract the numerical part from the 'Label' column and convert it to integer
        positions_tube['Label_num'] = positions_tube['Label'].str.extract('(\d+)').astype(int)

        # Sort the dataframe by 'Element', 'Molecule', and 'Label_num'
        positions_tube = positions_tube.sort_values(by=['Species', 'Molecule', 'Label_num'])

        # Drop the 'Label_num' column as it's no longer needed
        positions_tube = positions_tube.drop(columns=['Label_num'])

        # Finally compute the PBC size of the simulation box. It is given by the multiplicity in x and y direction times the unit cell size.
        pbc_size_x = multiplicity_x * unit_cell_x
        pbc_size_y = multiplicity_y * unit_cell_y
        
        self._structure_df = positions_tube

        return self._structure_df
    
class Structure_2D(Structure):

    def functionalize_sheet(self,parameters):
        self._initialize_functional_groups(parameters)
        self.__add_groups_to_sheet()

    def available_positions(self):
        available_positions = []
        for i, position in self._structure_df.iterrows():
            available_positions.append([position[1],position[2],position[3]])
        return available_positions
    
    def add(self,parameters,keywords):
        parameters['group_count'] = 1
        self._initialize_functional_groups(parameters)
        position = [self._structure_df.iloc[parameters['position'],1],
                    self._structure_df.iloc[parameters['position'],2]]
        self.__add_group_on_position(position)
        
    
    # CONSTRUCTOR
    def __init__(self,bond_distance,sheet_size):
        self.bond_distance = bond_distance
        self.sheet_size = sheet_size
        self._create_sheet()

    # PRIVATE
    def _create_sheet(self):        
        self._define_unit_cell() 
        self._build_sheet()

    def _define_unit_cell(self):
        # Calculate the distance between rows of atoms in x-y-directions
        C_C_y_distance = self.bond_distance * math.cos(30 * math.pi / 180)
        C_C_x_distance = self.bond_distance * math.sin(30 * math.pi / 180)
        self._unit_cell_vectors =  [2*self.bond_distance + 2*C_C_x_distance,   # length of unit vector in x-direction
                                    2*C_C_y_distance                           # length of unit vector in y-direction
                                    ]
        # Create a list for the atomic positions inside the unit cell.
        self._positions_unitcell = [(0,0,0),
                            (C_C_x_distance,C_C_y_distance,0),
                            (C_C_x_distance + self.bond_distance,C_C_y_distance,0),
                            (2 * C_C_x_distance + self.bond_distance,0,0)]
        
        self._number_of_unit_cells = [
            math.floor(self.sheet_size[0]/self._unit_cell_vectors[0]),
            math.floor(self.sheet_size[1]/self._unit_cell_vectors[1])
        ]

        self.sheet_size = [
            self._unit_cell_vectors[0] * self._number_of_unit_cells[0],
            self._unit_cell_vectors[1] * self._number_of_unit_cells[1]
        ]

    def __add_groups_to_sheet(self):
        added_groups = []
        for group in self.group_list:
            added_groups.append(group.remove_anchors())
        position_list = self.available_positions()
        new_atoms = []
        random.seed(a=None, version=2)
        for i in range(len(self.group_list)):
            group = self.group_list[i]
            number_of_added_groups = 0
            for j in range(group.group_count):
                if position_list:
                    new_atoms += self.__add_group_on_random_position(added_groups[i],group.exclusion_radius,position_list)
                else:
                    print("Sheet size is not large enough!")
                    print(f"Generated sheet is missing {group.group_count-number_of_added_groups} groups")
                    break
                number_of_added_groups += 1
        # convert the list to a dataframe and add a column with the atom group (structural/functional)
        new_atoms_df = pd.DataFrame(new_atoms)
        new_atoms_df["group"] = pd.Series(["functional" for x in range(len(new_atoms_df.index))])
        new_atoms_df.columns = ['Species','x','y','z','group']
        #finally add atoms to sheet
        self._structure_df = pd.concat([self._structure_df,new_atoms_df])

    def __add_group_on_random_position(self,added_group,exclusion_radius,position_list):
        # select a position to add the group on
        selected_position = position_list[random.randint(0,len(position_list)-1)]
        # randomly rotate the group
        new_atom_coordinates = random_rotate_group_list(added_group.copy())
        # shift the coordinates to the selected position
        for atom in new_atom_coordinates:
            atom[1] += selected_position[0]
            atom[2] += selected_position[1]
        #Remove positions blocked by the new group
        position_list.remove(selected_position)
        self.__remove_adjacent_positions(position_list,selected_position,exclusion_radius)
        return new_atom_coordinates
    
    def __add_group_on_position(self,selected_position):
        added_group = self.group_list[0].remove_anchors()
        # randomly rotate the group
        new_atom_coordinates = random_rotate_group_list(added_group.copy())
        # shift the coordinates to the selected position
        for atom in new_atom_coordinates:
            atom[1] += selected_position[0]
            atom[2] += selected_position[1]
        new_atoms_df = pd.DataFrame(new_atom_coordinates, columns=['Species','x','y','z'])
        new_atoms_df["group"] = pd.Series(["functional" for x in range(len(new_atoms_df.index))])
        self._structure_df = pd.concat([self._structure_df,new_atoms_df])

    def __remove_adjacent_positions(self, position_list, selected_position, cutoff_distance):
        adjacent_positions = []
        for position in position_list:
            if positions_are_adjacent(position, selected_position, cutoff_distance, self.sheet_size):
                adjacent_positions.append(position)
        for adjacent_position in adjacent_positions:
            position_list.remove(adjacent_position)    

class Pore(Structure):

    # INTERFACE
    def add(self,parameters,keywrods):
        parameters['group_count'] = 1
        self._initialize_functional_groups(parameters)
        # get the coordinates of the selected position
        selected_position = [self._structure_df.iloc[parameters['position'],1],
                    self._structure_df.iloc[parameters['position'],2],
                    self._structure_df.iloc[parameters['position'],3]]
        self.__add_group_on_position(selected_position)

    # CONSTRUCTOR
    def __init__(self,parameters,keywords):
        self._build_pore(parameters,keywords)

    # PRIVATE
    def _build_pore(self,parameters,keywords):
        self.bond_length = parameters['bond_length']
        self.sheet_size = parameters['sheet_size']
        if 'closed' in keywords:
            pore_kind = 2
        else:
            pore_kind = 1
        # Generate substructures
        wall = Graphene(parameters['bond_length'],parameters['sheet_size'])
        cnt = Structure_1D(parameters,keywords)
        # If the user wants a closed pore, we copy the wall now without the hole
        if pore_kind == 2:
            wall2 = copy.deepcopy(wall)
        self._structure_df = wall._structure_df
        # make a hole in the wall
        pore_position = wall.make_pores(cnt.radius+1.0)
        # shift cnt position to hole
        cnt._structure_df['x'] += pore_position[1]
        cnt._structure_df['y'] += pore_position[2]
        self.pore_center = [pore_position[1],pore_position[2]]
        self.cnt_radius=[cnt.radius]
        # If the user wants an open pore, we copy it now with the hole
        if pore_kind == 1:
            wall2 = copy.deepcopy(wall)
        # 'clip off' te ends of the cnt for a smoother transition
        cnt._structure_df = cnt._structure_df[cnt._structure_df['z'] > 0.2]
        max_z = cnt._structure_df['z'].max()
        cnt._structure_df = cnt._structure_df[cnt._structure_df['z'] < (max_z-0.2)]
        # move the second wall
        wall2._structure_df['z'] += max_z
        # combine the pore and the wall
        self._structure_df = pd.concat([cnt._structure_df, wall._structure_df, wall2._structure_df])
        self._structure_df.reset_index(inplace=True, drop=True)
        # Insert correct values for 'Label' and 'Molecule' columns.
        counter=1
        for i,atom in self._structure_df.iterrows():
            self._structure_df.at[i,'Label'] = f"C{counter}"
            counter=counter+1
        self._structure_df.loc[:,'Molecule'] = int(1)
        # lastly we need to correct the sheet_size to reflect the actual size of the sheet
        # 
        max_x = self._structure_df['x'].max()   # determine the maximum x-value
        delta_x = abs(max_x - self.sheet_size[0]) # minimum image distance in x-direction
        # the sheet size needs to be scaled down so that the minimum image distance
        # is equal to the bond length
        self.sheet_size[0] -= (delta_x-self.bond_length)

        # we do the same in y direction. The only difference is that 
        # the distance should not be equal to one bond length
        max_y = self._structure_df['y'].max()
        delta_y = abs(max_y - self.sheet_size[1])
        self.sheet_size[1] -= (delta_y-self.bond_length*math.cos(30*math.pi/180))
        
    def __add_group_on_position(self,selected_position):
        added_group = self.group_list[0].remove_anchors()
        # give the group a random orientation first
        new_atom_coordinates = random_rotate_group_list(added_group.copy())
        
        # find out if the position is inside the pore or on a wall 
        distance_to_center=math.sqrt((selected_position[0]-self.pore_center[0])**2
                                    +(selected_position[1]-self.pore_center[1])**2)
        
        if(distance_to_center < self.cnt_radius[0]+0.4):
            self.add_group_in_pore(new_atom_coordinates,selected_position)
        else:
            self.add_group_on_wall(new_atom_coordinates,selected_position)

    def add_group_on_wall(self,new_atom_coordinates,selected_position):

        # find out which wall the selected position belongs to
        structure_center_z = self._structure_df['z'].max()/2.0

        # if the position is on the wall with z~0.0, we have to invert the group
        # otherwise we do not change anything
        if selected_position[2] < structure_center_z:
            for atom in new_atom_coordinates:
                atom[1] *= -1.0
                atom[2] *= -1.0
                atom[3] *= -1.0

        # move the group to the position
        for atom in new_atom_coordinates:
            atom[1] += selected_position[0]
            atom[2] += selected_position[1]
            atom[3] += selected_position[2]
        new_atoms_df = pd.DataFrame(new_atom_coordinates, columns=['Species','x','y','z'])
        new_atoms_df["group"] = pd.Series(["functional" for x in range(len(new_atoms_df.index))])
        self._structure_df = pd.concat([self._structure_df,new_atoms_df])

    def add_group_in_pore(self,new_atom_coordinates,selected_position):
        # find the right orientation relative to local surface
            normal_vector = self.find_surface_normal_vector(selected_position)
            rotation_matrix = self.rotation_matrix_from_vectors(normal_vector)

            # finally rotate the group
            rotated_coordinates = []
            for atom in new_atom_coordinates:
                atom_coords = np.array(atom[1:4], dtype=float) # ensure that atom_coords has the right datatype
                rotated_coord = np.dot(rotation_matrix, atom_coords)
                rotated_coordinates.append([atom[0],rotated_coord[0],rotated_coord[1],rotated_coord[2],'functional'])

            # shift the coordinates to the selected position
            for atom in rotated_coordinates:
                atom[1] += selected_position[0]
                atom[2] += selected_position[1]
                atom[3] += selected_position[2]

            new_atoms_df = pd.DataFrame(rotated_coordinates, columns=['Species','x','y','z','group'])  # Update columns as needed
            #new_atoms_df["group"] = "functional"
            self._structure_df = pd.concat([self._structure_df,new_atoms_df])

    def find_surface_normal_vector(self,position):

        """surface_atoms = []
        # find adjacent atoms
        for i, atom in self._structure_df.iterrows():
            delta_x = atom['x']-position[0]
            delta_x -= self.sheet_size[0] * round(delta_x / self.sheet_size[0])
            delta_y = atom['y']-position[1]
            delta_y -= self.sheet_size[1] * round(delta_y / self.sheet_size[1]) ## minimum immage distance in x-y-direction
            distance = math.sqrt((delta_x)**2+(delta_y)**2+(atom['z']-position[2])**2)
            if distance <= self.bond_length*1.2:
                # we append the mirrored atom and not the atom itself, since
                # atoms that are mirrored due to periodic boundary conditions would
                # make the averages later useless if we take the positions directly
                if distance >= 0.05: # We do not want to add the selected position itself
                    surface_atoms.append([position[0]+delta_x,position[1]+delta_y,atom['z']])       

        # compute average position
        surface_atoms = np.array(surface_atoms)
        average_position = np.average(surface_atoms, axis=0)

        # this only works on curved surface (selected position and surface atoms are NOT in one plane)
        # on flat surfaces we have to use a different algorithm

        # check if the local surface is curved
        if (np.linalg.norm(average_position-np.array(position)) < 0.01):
            print("WARNING WARNING WARNING")
            print(np.linalg.norm(average_position-np.array(position)))
            return(np.array([1,0,0]))

        # compute normal vector
        position=np.array(position)
        normal_vector = average_position-position"""
        normal_vector = np.array([self.pore_center[0]-position[0],self.pore_center[1]-position[1],0.0])
        normal_magnitude = np.linalg.norm(normal_vector)
        normal_vector /= normal_magnitude

        return normal_vector
    
    def rotation_matrix_from_vectors(self,vec2):
        vec1 = np.array([0,0,1])
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        
class Graphene(Structure_2D):

    #INTERFACE
    def  make_pores(self, pore_size):
        hole_position = self._make_circular_pore(pore_size)
        return hole_position

    # PRIVATE
    def _make_circular_pore(self,pore_site):

        # select atom closest to center as position
        atoms_df = self._structure_df.copy()
        selected_position = center_position(self.sheet_size, atoms_df)
        # find which atoms should be removed
        atoms_to_remove = []
        for i, atom in self._structure_df.iterrows():
            atom_position=[atom[1],atom[2]]
            if minimum_image_distance(atom_position, [selected_position[1],selected_position[2]], self.sheet_size) <= pore_site:
                atoms_to_remove.append(i)
        # remove atoms
        self._structure_df.drop(atoms_to_remove, inplace=True)

        return selected_position


    def _build_sheet(self):
        Z = [0.0]*4 # All Z-coordinates are 0
        coords = []
        #BUILD SHEET FROM MULTIPLE UNIT CELLS
        for i in range(self._number_of_unit_cells[0]):
            X = [self._positions_unitcell[0][0] + i * self._unit_cell_vectors[0],
                self._positions_unitcell[1][0] + i * self._unit_cell_vectors[0],
                self._positions_unitcell[2][0] + i * self._unit_cell_vectors[0],
                self._positions_unitcell[3][0] + i * self._unit_cell_vectors[0],]
            for j in range(self._number_of_unit_cells[1]):
                Y = [self._positions_unitcell[0][1] + j * self._unit_cell_vectors[1],
                    self._positions_unitcell[1][1] + j * self._unit_cell_vectors[1],
                    self._positions_unitcell[2][1] + j * self._unit_cell_vectors[1],
                    self._positions_unitcell[3][1] + j * self._unit_cell_vectors[1],]
                coords.append(["C",X[0],Y[0],Z[0],"Structure"])
                coords.append(["C",X[1],Y[1],Z[1],"Structure"])
                coords.append(["C",X[2],Y[2],Z[2],"Structure"])
                coords.append(["C",X[3],Y[3],Z[3],"Structure"])
        self._structure_df = pd.DataFrame(coords) 
        # Give appropriate column names
        self._structure_df.columns = ['Species','x','y','z','group']

class Boronnitride(Structure_2D):

    # INTERFACE
    def make_pores(self, pore_size):
        self.__make_triangular_pore(pore_size)

    # PRIVATE
    def __make_triangular_pore(self, pore_size):
        # pick starting position
        atoms_df = self._structure_df.copy()
        # ensure that the hole is not placed at the  border
        """dummy_df = atoms_df[atoms_df[1] < self.sheet_size[0] - 1.5*hole_size]
        dummy_df = dummy_df[dummy_df[2] < self.sheet_size[1] - 1.5*hole_size]
        dummy_df = dummy_df[dummy_df[1] > 1.5*hole_size]
        dummy_df = dummy_df[dummy_df[2] > 1.5*hole_size]
        dummy_df = dummy_df[dummy_df[0] == 'N'] # triangular holes start at N
        selected_position = dummy_df.iloc[random.randint(0,len(dummy_df[0])-1)]"""
        dummy_df = atoms_df[atoms_df['Species'] == 'N']
        selected_position = center_position(self.sheet_size, dummy_df)
        # find nearest atom in x-direction to get orientation of the triangle
        dummy_df = atoms_df[atoms_df['Species'] == 'B']
        dummy_df = dummy_df[dummy_df['y'] > (selected_position[2]-0.1)]
        dummy_df = dummy_df[dummy_df['y'] < (selected_position[2]+0.1)]
        nearest_atom_df = dummy_df
        nearest_atom_df['x'] = nearest_atom_df['x'].apply(lambda x: abs(x - selected_position[1]))
        nearest_atom = atoms_df.iloc[nearest_atom_df['x'].idxmin()]
        # now define triangle
        orientation_vector = [
            nearest_atom[1]-selected_position[1],
            nearest_atom[2]-selected_position[2]
        ]
        magnitude = math.sqrt((orientation_vector[0])**2 + (orientation_vector[1])**2)/pore_size
        orientation_vector[0] /= magnitude
        orientation_vector[1] /= magnitude
        orientation_vector[0] *= pore_size
        orientation_vector[1] *= pore_size
        triangle_position = [selected_position[1],selected_position[2]]
        tip1 = [selected_position[1]+orientation_vector[0],selected_position[2]+orientation_vector[1]]

        tip2, tip3 = find_triangle_tips(triangle_position,tip1)

        # cut the triangle out of the sheet
        atoms_to_remove = []
        for i, atom in self._structure_df.iterrows():
            point = (atom[1], atom[2])  # Assuming columns 1 & 2 are x and y coordinates
            if is_point_inside_triangle(tip1, tip2, tip3, point):
                atoms_to_remove.append(i)

        # Remove atoms that are inside the triangle
        self._structure_df.drop(atoms_to_remove, inplace=True)
    
    def _build_sheet(self):
        Z = [0.0]*4 # All Z-coordinates are 0
        coords = []
        #BUILD SHEET FROM MULTIPLE UNIT CELLS
        for i in range(self._number_of_unit_cells[0]):
            X = [self._positions_unitcell[0][0] + i * self._unit_cell_vectors[0],
                self._positions_unitcell[1][0] + i * self._unit_cell_vectors[0],
                self._positions_unitcell[2][0] + i * self._unit_cell_vectors[0],
                self._positions_unitcell[3][0] + i * self._unit_cell_vectors[0],]
            for j in range(self._number_of_unit_cells[1]):
                Y = [self._positions_unitcell[0][1] + j * self._unit_cell_vectors[1],
                    self._positions_unitcell[1][1] + j * self._unit_cell_vectors[1],
                    self._positions_unitcell[2][1] + j * self._unit_cell_vectors[1],
                    self._positions_unitcell[3][1] + j * self._unit_cell_vectors[1],]
                coords.append(["B",X[0],Y[0],Z[0],"Structure"])
                coords.append(["N",X[1],Y[1],Z[1],"Structure"])
                coords.append(["B",X[2],Y[2],Z[2],"Structure"])
                coords.append(["N",X[3],Y[3],Z[3],"Structure"])
        self._structure_df = pd.DataFrame(coords)
        # Give appropriate column names
        self._structure_df.columns = ['Species','x','y','z','group'] 

def center_position(sheet_size, atoms_df):
    # This function returns the coordinates of the atom that
    # is closest to the sheet center
    center_point = [
        sheet_size[0]/2,
        sheet_size[1]/2
    ]
    distance_to_center_point = []
    for i,atom in atoms_df.iterrows():
        distance_to_center_point.append(minimum_image_distance(center_point,[atom[1],atom[2]],sheet_size))
    center_position_index=distance_to_center_point.index(min(distance_to_center_point))
    center_position = atoms_df.iloc[int(center_position_index)]
    return center_position

def rotate_vector(vec, angle):
        rad = np.radians(angle)
        rotation_matrix = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
        return np.dot(rotation_matrix, vec)

def find_triangle_tips(center, tip1):
    vec1 = np.array(tip1) - np.array(center)
    vec2 = rotate_vector(vec1, 120)
    vec3 = rotate_vector(vec1, 240)
    tip2 = center + vec2
    tip3 = center + vec3
    return tip2, tip3

def minimum_image_distance(position1, position2, system_size):
    delta = np.zeros(2)
    for i in range(2):
        delta[i] = position1[i] - position2[i]
        delta[i] -= system_size[i] * round(delta[i] / system_size[i])
    return np.sqrt(np.sum(delta**2))

def positions_are_adjacent(position1, position2, cutoff_distance, system_size):
    return minimum_image_distance(position1, position2, system_size) < cutoff_distance

def random_rotation_matrix_2d():
    theta = np.random.uniform(0, 2*np.pi)
    cos, sin = np.cos(theta), np.sin(theta)
    return np.array([[cos, -sin], [sin, cos]])

def random_rotate_group_list(group_list):
    rotation_matrix = random_rotation_matrix_2d()
    rotated_group_list = [
        [atom[0]] + (rotation_matrix.dot(atom[1:3])).tolist() + [atom[3]]
        for atom in group_list
    ]
    return rotated_group_list

def is_point_inside_triangle(tip1, tip2, tip3, point):
    def area(a, b, c):
        return 0.5 * abs((a[0] - c[0]) * (b[1] - a[1]) - (a[0] - b[0]) * (c[1] - a[1]))

    A = area(tip1, tip2, tip3)
    A1 = area(point, tip2, tip3)
    A2 = area(tip1, point, tip3)
    A3 = area(tip1, tip2, point)

    # Barycentric coordinates
    l1 = A1 / A
    l2 = A2 / A
    l3 = A3 / A

    # Check if point is inside the triangle
    return 0 <= l1 <= 1 and 0 <= l2 <= 1 and 0 <= l3 <= 1 and abs(l1 + l2 + l3 - 1) < 1e-5
