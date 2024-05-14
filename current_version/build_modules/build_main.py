from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory
import sys
from input_handler import Parser
from interpreter import Interpreter


def main(args):
    history = InMemoryHistory()  # command history from prompt toolkit
    parser = Parser()
    interpreter = Interpreter()

    if args['input']:
        read_commands_from_input(args['input'], parser, interpreter)
    while True:
        try:
            # get command
            command = prompt('CONAN-build: ', history=history)
            if not command:  # if the command is empty we skip it
                continue
            if command == 'exit':
                print("exiting...")
                break
            elif command == 'undo':
                undo(parser, interpreter)
            else:
                # parse input
                parsed_command = parser.parse(command)
                # print(parsed_command)
                # execute command
                interpreter.execute_command(parsed_command)
        except KeyboardInterrupt:
            interpreter.exit()  # perform cleanup
            print("exiting...")
            break  # exit when the user hits 'Ctrl C'
        except Exception as e:  # If we encounter ANY error the cleanup needs to be performed
            #interpreter.exit() # perform cleanup
            print(f"ERROR: {e}")
            continue


def read_commands_from_input(input_file, parser, interpreter):
    with open(input_file, 'r') as file:
        for line in file:
            try:
                # get command
                command = line.strip()
                if not command:  # if the command is empty we skip it
                    continue
                if command == 'exit':
                    print("exiting...")
                    interpreter.exit()
                    sys.exit()
                else:
                    # parse input
                    parsed_command = parser.parse(command)
                    #print(parsed_command)
                    # execute command
                    if parser.parsing_successful:
                        interpreter.execute_command(parsed_command)
            except EOFError:
                break  # exit when input file ends
            except Exception as e:  # If we encounter ANY error the cleanup needs to be performed
                #interpreter.exit() # perform cleanup
                #    print(f"ERROR: {e}")
                continue


def undo(parser, interpreter):
    # delete the last command
    with open(".command_file", "r+") as f:
        current_position = previous_position = f.tell()
        while f.readline():
            previous_position = current_position
            current_position = f.tell()
        f.truncate(previous_position)

    # now execute all commands that are left
    parser.undo = True
    read_commands_from_input(".command_file", parser, interpreter)
    parser.undo = False
