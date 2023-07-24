from sympy import sympify, symbols

from bingo.symbolic_regression.agraph.operator_definitions import INTEGER, CONSTANT
from bingo.symbolic_regression.agraph.string_parsing import eq_string_to_infix_tokens, operators, operator_map, \
    functions, var_or_const_pattern, int_pattern, infix_to_postfix
from keplar.operator.operator import Operator
from keplar.translator.translator import is_float


class Statistic(Operator):
    def __init__(self):
        super().__init__()

    def do(self, population):
        raise NotImplementedError


class TaylorStatistic(Statistic):
    def __init__(self, str_equ):
        super().__init__()
        self.final_statis = None
        self.str_equ = str_equ

    def pos_do(self):
        final_statis = {}
        str1 = self.str_equ
        list_equ = eq_string_to_infix_tokens(str1)
        print(list_equ)
        for token in range(len(list_equ)):
            if is_float(list_equ[token]) and list_equ[token - 1] != "^" and token != len(list_equ) - 1:
                print(float(list_equ[token]))
                if list_equ[token + 3] != "*" and list_equ[token + 3] != "^":
                    print("**" + list_equ[token])
                    print(list_equ[token + 2])
                    str_temp = list_equ[token + 2]
                    this_num = float(list_equ[token])
                    if token != 0:
                        if list_equ[token - 1] == "-":
                            this_num = this_num * (-1)
                    if str_temp not in final_statis:
                        final_statis.update({str_temp: this_num})
                    else:
                        now_num = final_statis[str_temp]
                        now_num += this_num
                        final_statis.update({str_temp: now_num})
                else:
                    this_num = float(list_equ[token])
                    if token != 0:
                        if list_equ[token - 1] == "-":
                            this_num = this_num * (-1)
                    i = token + 2
                    while list_equ[i] != "+" and list_equ[i] != "-":
                        i += 1
                    now_list = list_equ[token + 2:i]
                    now_str = ''.join(now_list)
                    if now_str not in final_statis:
                        final_statis.update({now_str: this_num})
                    else:
                        now_num = final_statis[now_str]
                        now_num += this_num
                        final_statis.update({now_str: now_num})

        print(final_statis)
        self.final_statis = final_statis


class BingoStatistic(Statistic):
    def __init__(self, str_equ):
        super().__init__()
        self.final_statis = None
        self.str_equ = str_equ

    def pos_do(self):
        str1 = self.str_equ
        list_equ = eq_string_to_infix_tokens(str1)
        print(list_equ)
        postfix_tokens = infix_to_postfix(list_equ)
        print(postfix_tokens)
        stack = []
        constants = []
        command_to_i = {}
        func_with_x = []
        i = 0
        command_array = []
        final_statis = {}
        n_constants = 0
        for token in postfix_tokens:
            if token in operators:
                operands = stack.pop(), stack.pop()
                command = [operator_map[token], operands[1], operands[0]]
                if token == "*":
                    if command_array[command[1]][0] == 1 and command_array[command[2]][0] == 0 and \
                            command_array[command[2]][1] == command_array[command[2]][2]:
                        str_temp = "X_" + str(command_array[command[2]][1])
                        if str_temp not in final_statis:
                            final_statis.update({str_temp: constants[command_array[command[1]][1]]})
                        else:
                            now_num = final_statis[str_temp]
                            now_num += constants[command_array[command[1]][1]]
                            final_statis.update({str_temp: now_num})
                    elif command_array[command[2]][0] == 1 and command_array[command[1]][0] == 0 and \
                            command_array[command[1]][1] == command_array[command[1]][2]:
                        str_temp = "X_" + str(command_array[command[1]][1])
                        if str_temp not in final_statis:
                            final_statis.update({str_temp: constants[command_array[command[2]][1]]})
                        else:
                            now_num = final_statis[str_temp]
                            now_num += constants[command_array[command[1]][1]]
                            final_statis.update({str_temp: now_num})
                elif token == "^":
                    print(command)

            elif token in functions:
                operand = stack.pop()
                # print(token)
                # print(command_array[operand][1])
                if command_array[operand][0] == 0 and command_array[operand][1] == command_array[operand][2]:
                    str_temp = "X_" + str(command_array[operand][1])
                    func_with_x.append([token, str_temp])
                command = [operator_map[token], operand, operand]
            else:
                var_or_const = var_or_const_pattern.fullmatch(token)
                integer = int_pattern.fullmatch(token)
                if var_or_const:
                    groups = var_or_const.groups()
                    # print(groups)
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

        for i in command_array:
            if i[0] == 0 and i[1] == i[2]:
                str_temp = "X_" + str(i[1])
                if str_temp not in final_statis:
                    final_statis.update({str_temp: 1})
                else:
                    now_num = final_statis[str_temp]
                    now_num += 1
                    final_statis.update({str_temp: now_num})

        reversed_dict = dict((value, key) for key, value in command_to_i.items())

        for i in func_with_x:
            str_final = i[0] + '(' + i[1] + ')'
            if i[1] in final_statis:
                now_num = final_statis[i[1]]
                now_num -= 1
                if now_num == 0:
                    del final_statis[i[1]]
                else:
                    final_statis.update({i[1]: now_num})
            if str_final not in final_statis:
                final_statis.update({str_final: 1})
            else:
                now_num = final_statis[str_final]
                now_num += 1
                final_statis.update({str_final: now_num})
        self.final_statis = final_statis
        print(final_statis)
