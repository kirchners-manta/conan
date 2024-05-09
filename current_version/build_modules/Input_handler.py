from ply import lex
import defdict as ddict
import sys

class InvalidCommand(Exception):
    pass

class Parser:

    # INTERFACE
    def parse(self,command):
        parsed_command = {}
        token_list = []
        self.parsing_successful = False

        # split command into readable tokens
        self.lexer.tokenize_input(command)

        for token in self.lexer.lexer:
            token_list.append(token)

        # check if the first token is a command
        if not token_list[0].type == 'COMMAND':
            raise InvalidCommand("\033[31minvalid command\033[37m")
        
        parsed_command['COMMAND'] = token_list[0].value
        token_list.remove(token_list[0])
        
        # Now we evaluate the rest of the command
        # Take care of expressions involving an operator first
        expression_list = self.__check_for_invalid_expressions(token_list)
        # find the assignment expressions and combine them to parameters
        if expression_list is not None:
            parameters = {"PARAMETERS":self.__perform_assignment_operations(expression_list)}
            parsed_command.update(parameters)
        # all remaining tokens should be single keywords now, so we can just add the token values
        # as keyword_list to our parsed command
        keyword_list = []
        for token in token_list:
            keyword_list.append(token.value)
        keywords = {"KEYWORDS":keyword_list}
        parsed_command.update(keywords)
        self.parsing_successful = True

        # save the command in file (this is needed for the undo command to work)
        if not self.undo:
            self._save_command(command,parsed_command)
        return parsed_command


    # PRIVATE
    def _save_command(self,raw_command,parsed_command):
        # we append the command to a file if it is not the vmd command
        if not parsed_command['COMMAND'] == 'vmd':
            with open(".command_file", 'a') as file:
                file.write(f"\n{raw_command}")

    def __perform_assignment_operations(self,expression_list):
        parameters={}
        for expression in expression_list:
            if expression['operator'].value == '=':
                parameters[expression['operand1'].value] = expression['operand2'].value
        return parameters

    def __check_for_invalid_expressions(self,token_list):
        # get all operators
        operator_list = []
        for i,token in enumerate(token_list):
            if token.type == 'OPERATOR':
                operator_list.append([i,token])

        # if there are no operators we can return
        if not operator_list:
            return None
        
        # check if all operators have two operands
        operand_list = []
        for operator in operator_list:
            operand_list.append(operator[0]-1)
            operand_list.append(operator[0]+1)
        if (min(operand_list) < 0  # first element is an operator
            or max(operand_list) > len(token_list)-1 or  # last element is an operator
            len(operand_list) > len(set(operand_list))): # two operators share an operand
            return None
        else:
            # If all expressions are valid, we make a list with all operator/operand tokens
            expression_list = []
            for operator in operator_list:
                expression_list.append({ "operand1" : token_list[operator[0]-1],
                                         "operator" : token_list[operator[0]],
                                         "operand2" : token_list[operator[0]+1]})
            # we now need to remove all assigned tokens from the token list to process
            # them further
            
            for expression in expression_list:
                token_list.remove(expression['operand1'])
                token_list.remove(expression['operator'])
                token_list.remove(expression['operand2'])
            return expression_list

    def __init__(self):
        self.lexer = Lexer()
        self.undo = False
        with open('.command_file', 'w') as file:
            pass # generates the command file (needed for the undo command)

class Lexer:
    # INTERFACE
    def tokenize_input(self,command):
        self.lexer.input(command)

    def get_next_token(self):
        return self.lexer.token()

    # CONSTRUCTOR
    def __init__(self):
        self.__initialize_lexer()

    # PRIVATE
    def __initialize_lexer(self):
        tokens = [
            'COMMAND',
            'KEYWORD',
            'VALUE',
            'OPERATOR',
        ] # all token types have to be listed here

        t_ignore = ' \t'  # Ignore spaces and tabs between tokens

        # next we define all token types listed above
        def t_COMMAND(t):
            r'(vmd|build|load|functionalize|defects|stack|add|remove|exit|undo)'
            return t
        def t_KEYWORD(t):
            r'\"[^\"]*\"|[a-zA-Z_]+' # all letters as well as everything within " " is a keyword
            if t.value.startswith('"') and t.value.endswith('"'):
                # Remove quotes
                t.value = t.value[1:-1]
            return t
        # \d+ = digit | \. = dot | \s+ = space
        def t_VALUE(t):
            r'(\d+\.\d+|\d+)(\s+\d+\.\d+|\s+\d+)*' # first bracket: float or integer
                                                   # second bracket: list of flaots or integers
            # split the token into multiple characters
            values = t.value.split()
            # since all tokens are read in as strings we need to convert them accordingly
            t.value = [float(value) if '.' in value else int(value) for value in values]
            # if multiple values are read in we save them as list, if not we save them as single value
            if len(t.value) == 1:
                t.value = t.value[0]
            return t
        def t_OPERATOR(t):
            r'(=|at|to)'
            return t
        # This special token gets triggered whenever a non defined character is encountered
        def t_error(t):
            print(f"Illegal character '{t.value[0]}' at line {t.lexer.lineno}")
            t.lexer.skip(1)
        # Build the lexer
        self.lexer = lex.lex()