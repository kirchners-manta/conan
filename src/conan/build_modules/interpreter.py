# import build_main as main
import os

import pandas as pd

import conan.build_modules.vmd_interface.vmd_interface as vmd
import conan.defdict as ddict
from conan.build_modules.structures import Boronnitride, Graphene, Pore, Structure1d


class Interpreter:

    # INTERFACE
    def execute_command(self, parsed_command):
        # CONAN commands
        if parsed_command["COMMAND"] == "build":
            self.__build(parsed_command["PARAMETERS"], parsed_command["KEYWORDS"])
        elif parsed_command["COMMAND"] == "functionalize":
            self.__functionalize(parsed_command["PARAMETERS"])
        elif parsed_command["COMMAND"] == "defects":
            self.__defects(parsed_command["PARAMETERS"], parsed_command["KEYWORDS"])
        elif parsed_command["COMMAND"] == "stack":
            self.current_structure.stack(parsed_command["PARAMETERS"], parsed_command["KEYWORDS"])
            self.current_structure.print_xyz_file(".tmp")
            if self.vmd_is_running:
                vmd.update_structure()

        elif parsed_command["COMMAND"] == "remove":
            self._remove_atom(parsed_command["PARAMETERS"], parsed_command["KEYWORDS"])
        elif parsed_command["COMMAND"] == "add":
            self.current_structure.add(parsed_command["PARAMETERS"])
            self.current_structure.print_xyz_file(".tmp")
            if self.vmd_is_running:
                vmd.update_structure()
        elif parsed_command["COMMAND"] == "load":
            self._load_structure(parsed_command["KEYWORDS"])
        elif parsed_command["COMMAND"] == "save":
            self._save_structure(parsed_command["KEYWORDS"])

        # VMD interface
        if parsed_command["COMMAND"] == "vmd":
            if "show_index" in parsed_command["KEYWORDS"]:
                vmd.send_command("show_index")
            elif "update" in parsed_command["KEYWORDS"]:
                vmd.update_structure()
            else:
                self.vmd_process = vmd.start_vmd()
                if self.vmd_process:
                    self.vmd_is_running = True

    # cleanup
    def exit(self):
        # print final structure
        if self.current_structure is not None:
            self.current_structure.print_xyz_file(self.current_structure.type)
            # exit vmd
        if self.vmd_is_running:
            vmd.send_command("exit")
            # if vmd does not want to exit, exit again, but harder
            if self.vmd_process.poll() is None:
                self.vmd_process.terminate()
        # remove temporary files
        if os.path.exists(".command_file"):
            os.remove(".command_file")
        if os.path.exists("structures/.tmp.xyz"):
            os.remove("structures/.tmp.xyz")
        if os.path.exists(".state.vmd"):
            os.remove(".state.vmd")

    # CONSTRUCTOR
    def __init__(self):
        self.current_structure = None
        self.vmd_is_running = False

    # PRIVATE
    def _save_structure(self, keywords):
        # this function just assumes that all keywords are given filenames
        if self.current_structure is not None:
            for file_name in keywords:
                self.current_structure.print_xyz_file(file_name)

    def _remove_atom(self, parameters, keywords):
        if "atom" in keywords:
            self.current_structure.remove_atom_by_index(parameters["index"])
            self.current_structure.print_xyz_file(".tmp")
            if self.vmd_is_running:
                vmd.update_structure()

    def _add_group(self, parameters, keywords):
        self.current_structure.add(parameters, keywords)
        self.current_structure.print_xyz_file(".tmp")
        if self.vmd_is_running:
            vmd.update_structure()

    def _load_structure(self, keywords):

        if len(keywords) > 1:
            ddict.printLog(f"'load' command accepts one argument. found: {len(keywords)}")
            return
        path_to_structure = keywords[0]
        try:
            with open(path_to_structure, "r") as file:
                lines = file.readlines()

            atom_data = lines[2:]  # Skip the first two lines (atom count and comment)
            atoms = []
            for line in atom_data:
                parts = line.split()
                if len(parts) >= 4:
                    atom_type = parts[0]
                    x, y, z = map(float, parts[1:4])
                    atoms.append([atom_type, x, y, z])

            self.df = pd.DataFrame(atoms, columns=["Atom", "X", "Y", "Z"])

        except Exception as e:
            ddict.printLog(f"Failed to load structure from {path_to_structure}: {e}")
            return None

    def __defects(self, parameters, keywords):
        if self.current_structure is None:
            ddict.printLog("\033[31m cannot create pores, missing structure")
            return
        if "pore_size" not in parameters:
            ddict.printLog("'pore_size' parameter is missing")
            return
        self.current_structure.make_pores(parameters, keywords)
        self.current_structure.print_xyz_file(".tmp")
        # ddict.printLog("Structure changed")
        # load updated structure into vmd
        if self.vmd_is_running:
            vmd.update_structure()

    def __functionalize(self, parameters):
        if self.current_structure is None:
            ddict.printLog("\033[31m cannot functionalize, missing structure \033[37m")
            return
        self.current_structure.functionalize_sheet(parameters)
        # print to temporary .xyz file
        self.current_structure.print_xyz_file(".tmp")
        # ddict.printLog("Structure functionalization finished")
        # load updated structure into vmd
        if self.vmd_is_running:
            vmd.update_structure()

    def __build(self, parameters, keywords):

        default_bond_length = 1.42
        default_sheet_size = [20.0, 20.0]

        # check if all necessary parameters have been assigned
        if "type" not in parameters:
            ddict.printLog("\033[31m 'type' parameter is missing \033[37m")
            return None

        # set defaults
        if "bond_length" not in parameters:
            parameters["bond_length"] = default_bond_length
            # ddict.printLog(f"bond_length parameter set to default. ({default_bond_length})")
        if "sheet_size" not in parameters:
            parameters["sheet_size"] = default_sheet_size
            # ddict.printLog(f"sheet_size parameter set to default. ({default_sheet_size[0]}x{default_sheet_size[1]})")

        if parameters["type"] == "graphene":
            self.current_structure = Graphene(parameters["bond_length"], parameters["sheet_size"])
        elif parameters["type"] == "boronnitride":
            self.current_structure = Boronnitride(parameters["bond_length"], parameters["sheet_size"])
        elif parameters["type"] == "cnt":
            self.current_structure = Structure1d(parameters, keywords)
        elif parameters["type"] == "pore":
            self.current_structure = Pore(parameters, keywords)
        else:
            raise InvalidCommand(f"unknown structure type '{parameters['type']}'.")

        if not self.current_structure._structure_df.empty:
            # print to temporary .xyz file
            self.current_structure.print_xyz_file(".tmp")
            # ddict.printLog("Structure build finished")

        # load updated structure into vmd
        if self.vmd_is_running:
            vmd.update_structure()


class InvalidCommand(Exception):
    pass
