import defdict as ddict
import Structures
import build_main as main
import pandas as pd
import time
import vmd_interface as vmd

class Interpreter:

    # INTERFACE
    def execute_command(self,parsed_command):
        #CONAN commands
        if parsed_command['COMMAND'] == 'build':
            self.__build(parsed_command['PARAMETERS'],parsed_command['KEYWORDS'])
        elif parsed_command['COMMAND'] == 'functionalize':
            self.__functionalize(parsed_command['PARAMETERS'])
        elif parsed_command['COMMAND'] == 'defects':
            self.__defects(parsed_command['PARAMETERS'])
        elif parsed_command['COMMAND'] == 'stack':
            self.current_structure.stack(parsed_command['PARAMETERS'],parsed_command['KEYWORDS'])
            self.current_structure.print_xyz_file(".tmp")
            if self.vmd_is_running:    
                vmd.update_structure()
        elif parsed_command['COMMAND'] == 'remove':
            self._remove_atom(parsed_command['PARAMETERS'],parsed_command['KEYWORDS'])
        elif parsed_command['COMMAND'] == 'add':
            self.current_structure.add(parsed_command['PARAMETERS'],parsed_command['KEYWORDS'])
            self.current_structure.print_xyz_file(".tmp")
            if self.vmd_is_running:
                vmd.update_structure()
        elif parsed_command['COMMAND'] == 'load':
            self._load_structure(parsed_command['KEYWORDS'])

        #VMD interface
        if parsed_command['COMMAND'] == 'vmd':
            if 'show_index' in parsed_command['KEYWORDS']:
                vmd.send_command("show_index")
            else:
                self.vmd_process = vmd.start_vmd()
                self.vmd_is_running = True

    # cleanup
    def exit(self):
        # print final structure
        if not self.current_structure is None:
            self.current_structure.print_xyz_file("structure") 
        # exit vmd
        if self.vmd_is_running:
            vmd.send_command("exit")
        # if vmd does not want to exit, exit again, but harder
            if self.vmd_process.poll() is None:
                self.vmd_process.terminate()

    # CONSTRUCTOR
    def __init__(self):
        self.current_structure = None
        self.vmd_is_running = False

    # PRIVATE
    def _remove_atom(self,parameters,keywords):
        if 'atom' in keywords:
            self.current_structure.remove_atom_by_index(parameters['index'])
            self.current_structure.print_xyz_file(".tmp")
            if self.vmd_is_running:
                vmd.update_structure()
    def _add_group(self,parameters,keywords):
        self.current_structure.add(parameters,keywords)
        self.current_structure.print_xyz_file(".tmp")
        if self.vmd_is_running:
                vmd.update_structure()
    def _load_structure(self, keywords):

        if len(keywords) > 1:
            ddict.printLog(f"'load' command accepts one argument. found: {len(keywords)}")
            return
        path_to_structure = keywords[0]
        try:
            with open(path_to_structure, 'r') as file:
                lines = file.readlines()
            
            
            atom_data = lines[2:] # Skip the first two lines (atom count and comment)
            atoms = []
            for line in atom_data:
                parts = line.split()
                if len(parts) >= 4:
                    atom_type = parts[0]
                    x, y, z = map(float, parts[1:4])
                    atoms.append([atom_type, x, y, z])
            
            self.df = pd.DataFrame(atoms, columns=['Atom', 'X', 'Y', 'Z'])
            
        except Exception as e:
            ddict.printLog(f"Failed to load structure from {path_to_structure}: {e}")
            return None


    def __defects(self,parameters):
        if self.current_structure == None:
            ddict.printLog("\033[31m cannot create pores, missing structure")
            return
        if not "pore_size" in parameters:
            ddict.printLog("'pore_size' parameter is missing")
            return
        self.current_structure.make_pores(parameters['pore_size'])
        self.current_structure.print_xyz_file(".tmp")
        #ddict.printLog("Structure changed")
        # load updated structure into vmd
        if self.vmd_is_running:
            vmd.update_structure() 

    def __functionalize(self,parameters):
        if self.current_structure == None:
            ddict.printLog("\033[31m cannot functionalize, missing structure \033[37m")
            return
        self.current_structure.functionalize_sheet(parameters)
        # print to temporary .xyz file
        self.current_structure.print_xyz_file(".tmp")
        #ddict.printLog("Structure functionalization finished")
        # load updated structure into vmd
        if self.vmd_is_running:
            vmd.update_structure() 

    def __build(self,parameters,keywords):

        default_bond_length = 1.42
        default_sheet_size = [20.0, 20.0]

        # check if all necessary parameters have been assigned
        if not "type" in parameters:
            ddict.printLog("\033[31m 'type' parameter is missing \033[37m")
            return None
        
        # set defaults
        if not "bond_length" in parameters:
            parameters['bond_length'] = default_bond_length
            #ddict.printLog(f"bond_length parameter set to default. ({default_bond_length})")
        if not "sheet_size" in parameters:
            parameters['sheet_size'] = default_sheet_size
            #ddict.printLog(f"sheet_size parameter set to default. ({default_sheet_size[0]}x{default_sheet_size[1]})")

        if parameters['type'] == "graphene":
            self.current_structure = Structures.Graphene(parameters['bond_length'],parameters['sheet_size'])
        elif parameters['type'] == "boronnitride":
            self.current_structure = Structures.Boronnitride(parameters['bond_length'],parameters['sheet_size'])
        elif parameters['type'] == "cnt":
            self.current_structure = Structures.Structure_1D(parameters,keywords)
        elif parameters['type'] == "pore":
            self.current_structure = Structures.Pore(parameters,keywords)
        else:
            raise InvalidCommand(f"unknown structure type '{parameters['type']}'.")
        
        self.number_of_structural_atoms = len(self.current_structure._structure_df['group'])

        # print to temporary .xyz file
        self.current_structure.print_xyz_file(".tmp")
        #ddict.printLog("Structure build finished")

        # load updated structure into vmd
        if self.vmd_is_running:
            vmd.update_structure()

class InvalidCommand(Exception):
    pass