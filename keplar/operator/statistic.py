from bingo.symbolic_regression.agraph.operator_definitions import INTEGER, CONSTANT
from bingo.symbolic_regression.agraph.string_parsing import eq_string_to_infix_tokens, operators, operator_map, \
    functions, var_or_const_pattern, int_pattern, infix_to_postfix
from keplar.operator.operator import Operator


class Statistic(Operator):
    def __init__(self):
        super().__init__()

    def do(self, population):
        raise NotImplementedError


class BingoStatistic(Statistic):
    def __init__(self, str_equ):
        super().__init__()
        self.str_equ = str_equ

    def pos_do(self):
        str1=self.str_equ
        list_equ = eq_string_to_infix_tokens(str1)
        print(list_equ)
        postfix_tokens = infix_to_postfix(list_equ)
        print(postfix_tokens)
        stack = []
        constants = []
        command_to_i = {}
        i = 0
        command_array = []
        n_constants = 0
        for token in postfix_tokens:
            if token in operators:
                operands = stack.pop(), stack.pop()
                command = [operator_map[token], operands[1], operands[0]]
            elif token in functions:
                operand = stack.pop()
                print(operand)
                command = [operator_map[token], operand, operand]
            else:
                var_or_const = var_or_const_pattern.fullmatch(token)
                integer = int_pattern.fullmatch(token)
                if var_or_const:
                    groups = var_or_const.groups()
                    print(groups)
                    command = [operator_map[groups[0]], int(groups[1]),
                               int(groups[1])]
                elif integer:
                    operand = int(token)
                    command = [INTEGER, operand, operand]
                else:
                    try:
                        command = [CONSTANT, n_constants, n_constants]

                        constant = float(token)
                        constants.append(constant)
                        n_constants += 1
                    except ValueError as err:
                        raise RuntimeError(f"Unknown token {token}") from err
            if tuple(command) in command_to_i:
                stack.append(command_to_i[tuple(command)])
            else:
                command_to_i[tuple(command)] = i
                command_array.append(command)
                stack.append(i)
                i += 1
        print(command_array)
        print(constants)
        reversed_dict = dict((value, key) for key, value in command_to_i.items())
        print(reversed_dict)
