from sympy import sympify, symbols

from bingo.symbolic_regression.agraph.operator_definitions import INTEGER, CONSTANT
from bingo.symbolic_regression.agraph.string_parsing import eq_string_to_infix_tokens, operators, operator_map, \
    functions, var_or_const_pattern, int_pattern, infix_to_postfix
from gplearn._program import _Program
from gplearn.functions import _Function
from keplar.operator.operator import Operator
from keplar.translator.translator import is_float, lable_list_to_gp_list


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
        if is_float(list_equ[0]) and len(list_equ)>1:
            if list_equ[1] == '+' or list_equ[1] == '-':
                bug_num = list_equ[0]
                list_equ = list_equ[1:]
                list_equ.append('+')
                list_equ.append(bug_num)
                if list_equ[0] == '+':
                    list_equ = list_equ[1:]
                elif list_equ[0] == '-':
                    new_num = float(list_equ[1]) * (-1)
                    list_equ[1] = str(new_num)
                    list_equ = list_equ[1:]
        for token in range(len(list_equ)):
            if is_float(list_equ[token]) and list_equ[token - 1] != "^" and token != len(list_equ) - 1:
                print(float(list_equ[token]))
                if (token + 3 == len(list_equ) or list_equ[token + 3] != "*") and (
                        token + 3 == len(list_equ) or list_equ[token + 3] != "^"):
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


class OperonStatistic(Statistic):
    def __init__(self):
        super().__init__()

    def pos_do(self):
        pass


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
                    print(command_to_i)
                    if command_array[command[1]][0] == 1:
                        print("1")
                        print(command)
                        print(command[2])
                        print(stack)
                        #     and command_array[command[2]][0] == 0 and \
                        #     command_array[command[2]][1] == command_array[command[2]][2]:
                        # str_temp = "X_" + str(command_array[command[2]][1])
                        # if str_temp not in final_statis:
                        #     final_statis.update({str_temp: constants[command_array[command[1]][1]]})
                        # else:
                        #     now_num = final_statis[str_temp]
                        #     now_num += constants[command_array[command[1]][1]]
                        #     final_statis.update({str_temp: now_num})
                    elif command_array[command[2]][0] == 1:
                        print("2")
                        print(command)
                        print(command[1])
                        #     and command_array[command[1]][0] == 0 and \
                        #     command_array[command[1]][1] == command_array[command[1]][2]:
                        # str_temp = "X_" + str(command_array[command[1]][1])
                        # if str_temp not in final_statis:
                        #     final_statis.update({str_temp: constants[command_array[command[2]][1]]})
                        # else:
                        #     now_num = final_statis[str_temp]
                        #     now_num += constants[command_array[command[1]][1]]
                        #     final_statis.update({str_temp: now_num})
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


class GpStatistic(Statistic):
    def __init__(self, gp_program):
        super().__init__()
        self.gp_program = gp_program
        if not isinstance(self.gp_program, _Program):
            raise ValueError("输入类型必须为gplearn的program类型")

    def pos_do(self):
        data = self.gp_program.export_graphviz()
        print(data)
        print(type(data))
        list_str = data.split('\n')
        print(list_str)
        list_str = list_str[2:-1]
        print(list_str)
        lable_list = []
        arrow_list = []
        for tk in list_str:
            if '[' in tk:
                lable_list.append(tk)
            elif "->" in tk:
                arrow_list.append(tk)
            else:
                ValueError(f"出错了token{tk}")
        print(lable_list)
        print(arrow_list)
        lable_num_list = []
        lable_name_list = []
        for i in lable_list:
            temp_list = i.split(" ")
            print(temp_list)
            num_str = temp_list[0]
            lable_num_list.append(int(num_str))
            print(lable_num_list)
            name_temp = temp_list[1]
            name_temp_str = ""
            read_flag = False
            for j in name_temp:
                if not read_flag and j == '"':
                    read_flag = True
                elif read_flag and j != '"':
                    name_temp_str += j
                elif read_flag and j == '"':
                    break
                else:
                    continue
            lable_name_list.append(name_temp_str)
        print(lable_name_list)
        node_dict = {}
        for i in range(len(lable_name_list)):
            node_dict.update({lable_num_list[i]: lable_name_list[i]})
        print(node_dict)
        new_arrow_list = []
        left_arrow_list = []
        right_arrow_list = []
        for i in arrow_list:
            str_temp = i.split(' ')
            new_arrow_list.append(str_temp[0] + str_temp[1] + str_temp[2])
            left_arrow_list.append(str_temp[0])
            right_arrow_list.append(str_temp[2])
        print(new_arrow_list)
        print(left_arrow_list)
        print(right_arrow_list)
        new_tree = []
        while True:
            if lable_name_list[0] != 'sub' and lable_name_list[0] != 'add':
                break
            elif lable_name_list[0] == 'add':
                left_add_index = []
                for i in range(len(left_arrow_list)):
                    if left_arrow_list[i] == '0':
                        left_add_index.append(i)
                right_add_list = [right_arrow_list[left_add_index[0]], right_arrow_list[left_add_index[1]]]
                mid_point = int(right_add_list[0]) - int(right_add_list[1])
                if mid_point < 0:
                    mid_point *= (-1)
                new_tree.append(lable_name_list[1:mid_point + 1])
                new_tree.append(lable_name_list[mid_point + 1:])
                break
            elif lable_name_list[0] == 'sub':
                left_add_index = []
                for i in range(len(left_arrow_list)):
                    if left_arrow_list[i] == '0':
                        left_add_index.append(i)
                right_add_list = [right_arrow_list[left_add_index[0]], right_arrow_list[left_add_index[1]]]
                mid_point = int(right_add_list[0]) - int(right_add_list[1])
                if mid_point < 0:
                    mid_point *= (-1)
                new_tree.append(lable_name_list[1:mid_point + 1])
                temp_right = lable_name_list[mid_point + 1:]
                temp_right = ['neg'] + temp_right
                new_tree.append(temp_right)
                break

        print(new_tree)
        temp_ = []
        for i in new_tree:
            temp_.append(lable_list_to_gp_list(i))
        new_gpprog_list = []
        for i in temp_:
            gp_prog = _Program(function_set=["add", "sub", "mul", "div", "sqrt", "neg"],
                               arities={"add": 2, "sub": 2, "mul": 2, "div": 2, "sqrt": 1, "neg": 1},
                               init_depth=[3, 3], init_method="half and half", n_features=4, const_range=[0, 1],
                               metric="rmse",
                               p_point_replace=0.4, parsimony_coefficient=0.01, random_state=1, program=i)
            new_gpprog_list.append(gp_prog)
        return new_gpprog_list

    #     terminals = []
    #     if fade_nodes is None:
    #         fade_nodes = []
    #     output = 'digraph program {\nnode [style=filled]\n'
    #     for i, node in enumerate(self.gp_program._program):
    #         fill = '#cecece'
    #         if isinstance(node, _Function):
    #             if i not in fade_nodes:
    #                 fill = '#136ed4'
    #             terminals.append([node.arity, i])
    #             output += ('%d [label="%s", fillcolor="%s"] ;\n'
    #                        % (i, node.name, fill))
    #         else:
    #             if i not in fade_nodes:
    #                 fill = '#60a6f6'
    #             if isinstance(node, int):
    #                 if self.feature_names is None:
    #                     feature_name = 'X%s' % node
    #                 else:
    #                     feature_name = self.feature_names[node]
    #                 output += ('%d [label="%s", fillcolor="%s"] ;\n'
    #                            % (i, feature_name, fill))
    #             else:
    #                 output += ('%d [label="%.3f", fillcolor="%s"] ;\n'
    #                            % (i, node, fill))
    #             if i == 0:
    #                 # A degenerative program of only one node
    #                 return output + '}'
    #             terminals[-1][0] -= 1
    #             terminals[-1].append(i)
    #             while terminals[-1][0] == 0:
    #                 output += '%d -> %d ;\n' % (terminals[-1][1],
    #                                             terminals[-1][-1])
    #                 terminals[-1].pop()
    #                 if len(terminals[-1]) == 2:
    #                     parent = terminals[-1][-1]
    #                     terminals.pop()
    #                     if not terminals:
    #                         return output + '}'
    #                     terminals[-1].append(parent)
    #                     terminals[-1][0] -= 1
    #
    #     # We should never get here
    #     return None
