import re

import numpy as np
import pyoperon as Operon

from TaylorGP.src.taylorGP._program import _Program
from bingo.symbolic_regression import AGraph
from bingo.symbolic_regression.agraph.string_parsing import infix_to_postfix, postfix_to_command_array_and_constants, \
    eq_string_to_infix_tokens
from gplearn.functions import _Function, sqrt1, add2, mul2, div2, sub2, neg1, power2, exp1, sin1, cos1, log1
from keplar.population.function import arity_map, operator_map3, _function_map, map_F1, Operator_map_S, \
    operator_map_dsr, operator_map_dsr2, Operator_map_S1
from keplar.population.individual import Individual

from TaylorGP.src.taylorGP.functions import _function_map


# Operon从X_1开始
# 该函数是将种群中的一个个体转化成TaylorGP中的program格式
def trans_taylor_program(ind):
    length = len(ind.func)
    program = []
    for i in range(length):

        # 如果遍历列表中数据类型为浮点数，那么该数为常量
        if isinstance(ind.func[i], float):
            program.append(ind.func[i])
            # print(ind.func[i])

        # 如果遍历列表中数据类型为字符串类型，那么该数据为function类型
        elif isinstance(ind.func[i], str):
            if _function_map.get(ind.func[i]) is not None:
                # print(_function_map[ind.func[i]].name)
                program.append(_function_map[ind.func[i]])
            else:
                program.append(ind.func[i])

        # 如果遍历列表中数据类型为整数类型，那么该数据为变量（x1、x2、x3....）
        elif isinstance(ind.func[i], int):
            # print(ind.func[i])
            program.append(ind.func[i])

    return program


# 该函数是将TaylorGP中的program格式转化成框架中我们种群中的一个个体
def taylor_trans_population(program, population, idx):
    eq = []

    for i, node in enumerate(program):
        if isinstance(node, _Function):
            eq.append(node.name)
        else:
            eq.append(node)
    ind = Individual(eq)

    population.target_pop_list[idx] = ind

    return population


def taylor_trans_ind(program):
    eq = []

    for i, node in enumerate(program):
        if isinstance(node, _Function):
            eq.append(node.name)
        else:
            eq.append(node)
    ind = Individual(eq)

    return ind
def format_taylor(mylist,const_array):
    stack = []
    stack_i = 0
    # print(self.func)
    for i in mylist:
        tk = int(i)
        if tk < 3000:
            str_op = map_F1[tk]
            arity = arity_map[tk]
            if arity == 1:
                if str_op == "square":
                    op1 = stack[stack_i - 1]
                    equ_str = "((" + op1 + ")" + "^2" + ")"
                    stack[stack_i - 1] = equ_str
                else:
                    op1 = stack[stack_i - 1]
                    equ_str = "(" + str_op + "(" + op1 + ")" + ")"
                    stack[stack_i - 1] = equ_str
            if arity == 2:
                op1 = stack[stack_i - 1]
                op2 = stack[stack_i - 2]
                stack_i = stack_i - 1
                equ_str = "(" + op1 + " " + str_op + " " + op2 + ")"
                stack[stack_i - 1] = equ_str
        elif tk >= 5000:
            const = const_array[tk - 5000]
            stack.append(str(const))
            stack_i = stack_i + 1
        elif 3000 <= tk < 5000:
            x_num = tk - 3000
            x_str = "x" + str(x_num)
            stack.append(x_str)
            stack_i = stack_i + 1
        else:
            raise ValueError(f"编码无意义{tk}")
    return stack[0]


# from keplar.population.function import operator_map, arity_map, operator_map3, _function_map


def get_priority(op):
    if op == '+' or op == '-':
        return 0
    else:
        return 1


def is_float(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


def prefix_to_postfix(expression):
    stack = []
    operators = {'add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs',
                 'neg', 'inv', 'max', 'min', 'sin', 'cos', 'tan',
                 'sig', 'aq', 'pow', 'exp', 'square'}  # 可用的运算符

    for token in reversed(expression):
        if token in operators:  # 操作符
            # 弹出栈顶运算符，直到遇到更低优先级的运算符或左括号
            while stack and stack[-1] in operators and operators[token] <= operators[stack[-1]]:
                temp = stack.pop()
            stack.append(token)
        elif token == ')':  # 右括号
            stack.append(token)
        elif token == '(':  # 左括号
            # 弹出栈顶运算符，直到遇到右括号
            while stack and stack[-1] != ')':
                yield stack.pop()
            stack.pop()  # 弹出右括号
        else:  # 操作数
            yield token

    # 弹出栈中剩余的运算符
    while stack:
        yield stack.pop()


def bingo_infixstr_to_func(equ):
    equ = re.sub(r"\)\(", r')*(', equ)
    list_infix = []
    op_al = ""
    x_al = ""
    num_al = ""
    op_start = False
    x_start = False
    num_start = False
    i = 0
    # print(equ)
    equ = re.sub(r"(\d{1})e", r"\1", equ)
    equ = re.sub(r"\(-", r'(0-', equ)
    equ = re.sub(r"-(\d{1})", r"0 - \1", equ)
    # print(equ)
    while i < len(equ):
        if op_start:
            if equ[i].isalnum():
                op_al = op_al + str(equ[i])
                i = i + 1
            else:
                list_infix.append(op_al)
                op_al = ""
                op_start = False
        elif x_start:
            if equ[i] == "_":
                x_al = x_al + "_"
                i = i + 1
            elif equ[i].isdigit():
                x_al = x_al + str(equ[i])
                i = i + 1
            else:
                list_infix.append(x_al)
                x_al = ""
                x_start = False
        elif num_start:
            if equ[i].isdigit() or equ[i] == ".":
                num_al = num_al + str(equ[i])
                i = i + 1
            else:
                list_infix.append(num_al)
                num_al = ""
                num_start = False
        else:
            if equ[i].isalpha() and equ[i] != "X":
                op_start = True
                op_al = str(equ[i])
                i = i + 1
            elif equ[i] == " ":
                i = i + 1
            elif equ[i] == "X":
                x_start = True
                x_al = "X"
                i = i + 1
            elif equ[i].isdigit():
                num_start = True
                num_al = str(equ[i])
                i = i + 1
            else:
                list_infix.append(str(equ[i]))
                i = i + 1
    if num_al != "":
        list_infix.append(num_al)
    if x_al != "":
        list_infix.append(x_al)
    if op_al != "":
        list_infix.append(op_al)
    # print(list_infix)
    post_equ = infix_to_postfix(list_infix)
    # print(post_equ)
    stack = []
    const_array = []
    for node in post_equ:
        if str(node) == "nan":
            node = "0"
        if str(node) == "inf":
            node = "100000000000000000000000"

        if node in ["+", "-", "*", "/", "^"] or node.isalpha():
            # print(node)
            arity = arity_map[operator_map3[node]]
            if arity == 1:
                operand1 = stack.pop()
                sub_stack = []
                sub_stack.append(operator_map3[node])
                sub_stack.append(operand1)
                stack.append(sub_stack)
            elif arity == 2:
                operand2 = stack.pop()
                operand1 = stack.pop()
                sub_stack = []
                sub_stack.append(operator_map3[node])
                sub_stack.append(operand1)
                sub_stack.append(operand2)
                stack.append(sub_stack)
            else:
                raise ValueError("Arity>=3")
        elif is_float(node):
            const_code = len(const_array)
            const_array.append(float(node))
            stack.append(const_code + 5000)
        else:
            # print(node)
            var_code = int(node[2:]) + 3000
            stack.append(var_code)
    new_func_list = []
    stack1 = [stack]
    while stack1:
        item = stack1.pop()
        if isinstance(item, list):
            stack1.extend(reversed(item))
        else:
            new_func_list.append(item)
    # print(new_func_list)
    return new_func_list, const_array


def bingo_to_taylor_str(equ):
    equ = re.sub(r'X_(\d{3})', r'x\1', equ)
    equ = re.sub(r'X_(\d{2})', r'x\1', equ)
    equ = re.sub(r'X_(\d{1})', r'x\1', equ)



def to_gp(ind):
    func = ind.func
    list_program = []
    for i in func:
        int_i = int(i)
        if int_i < 3000:
            str_op = map_F1[int_i]
            _func = _function_map[str_op]
            list_program.append(_func)
        elif 3000 <= int_i < 5000:
            float_con = float('%.3f' % ind.const_array[int_i - 3000])
            list_program.append(float_con)
        elif int_i > 5000:
            x = int(int_i - 5000)
            list_program.append(x)

        else:
            raise ValueError("留空")
    return list_program


def to_taylor(ind, function_set,
              arities,
              init_depth,
              init_method,
              n_features,
              const_range,
              metric,
              p_point_replace,
              parsimony_coefficient,
              random_state):
    func = ind.func
    list_program = []
    for i in func:
        int_i = int(i)
        if int_i < 3000:
            str_op = map_F1[int_i]
            _func = _function_map[str_op]
            list_program.append(_func)
        elif 3000 <= int_i < 5000:
            float_con = float('%.3f' % ind.const_array[int_i - 3000])
            list_program.append(float_con)
        elif int_i > 5000:
            x = int(int_i - 5000)
            list_program.append(x)

        else:
            raise ValueError("留空")
    taylor_program = _Program(function_set,
                              arities,
                              init_depth,
                              init_method,
                              n_features,
                              const_range,
                              metric,
                              p_point_replace,
                              parsimony_coefficient,
                              random_state, program=list_program)
    return taylor_program


def infix_to_prefix(infix, len1, s2, top2):
    s1 = ""
    top1 = -1
    i = len1 - 1
    while i >= 0:
        if '0' <= infix[i] <= '9':
            s2[++top2] = infix[i]
            i = i - 1

        elif infix[i] == ')':
            s1[++top1] = ')'
            i = i - 1

        elif (infix[i] == '+' or
              infix[i] == '-' or
              infix[i] == '*' or
              infix[i] == '/'):

            if top1 == -1 or s1[top1] == ')' or get_priority(infix[i]) >= get_priority(s1[top1]):
                s1[++top1] = infix[i]
                i = i - 1
            else:
                s2[++top2] = s1[top1]
                top1 = top1 - 1

        if infix[i] == '(':
            while s1[top1] != ')':
                s2[++top2] = s1[top1]
                top1 = top1 - 1
            top1 = top1 - 1
            i = i - 1

    while top1 != -1:
        s2[++top2] = s1[top1]


class Stacks(object):  # 用列表实现栈
    def __init__(self):  # 实例化栈
        self.list = []

    def isEmpty(self):  # 判断栈空
        return self.list == []

    def push(self, item):  # 入栈
        self.list.append(item)

    def pop(self):  # 出栈
        return self.list.pop()

    def top(self):  # 返回顶部元素
        return self.list[len(self.list) - 1]

    def size(self):  # 返回栈大小
        return len(self.list)


def pre_to_mid(x):
    s = Stacks()
    list = x.split()  # 空格分割待字符串
    for par in list:
        if par in "+-*/":  # 遇到运算符则入栈
            s.push(par)
        else:  # 为数字时分两种情况：
            if s.top() in '+-*/':  # 栈顶为运算符
                s.push(par)  # 数字入栈
            else:  # 当前栈顶为数字
                while (not s.isEmpty()) and (not s.top() in '+-*/'):  # 若栈不空，且当前栈顶为数字，则循环计算
                    shu = s.pop()  # 运算符前的数字出栈
                    fu = s.pop()  # 运算符出栈
                    par = '(' + shu + fu + par + ')'  # 计算
                s.push(str(par))  # 算式入栈
    return s.pop()  # 返回最终算式


def trans_gp(gp_program):
    node = gp_program.program[0]
    if isinstance(node, float):
        ind = Individual(func=["5000"], const_array=[node])
        return ind
    if isinstance(node, int):
        int_x = 3000 + node
        str_x = int(int_x)
        ind = Individual(func=[str_x], const_array=[])
        return ind
    func = []
    const_array = []
    const_index = 0
    for i, node in enumerate(gp_program.program):
        if isinstance(node, _Function):
            op_name = node.name
            op_code = Operator_map_S1[op_name]
            func.append(op_code)
        else:
            if isinstance(node, int):
                x_code = 3000 + node
                func.append(x_code)
            elif isinstance(node, float):
                const_array.append(node)
                const_code = 5000 + const_index
                func.append(const_code)
                const_index = const_index + 1
            else:
                raise ValueError(f"未识别，字符{node}")
    ind = Individual(func=func, const_array=const_array)
    return ind


class DSRToKeplar():
    def __init__(self, T):
        self.T = T
        self.length_T = len(self.T)

        # self.operator_map_dsr = operator_map_dsr
        # self.poplation = poplation

    def do(self, poplation=None):
        f = [[] for i in range(self.length_T)]
        for i in range(self.length_T):
            for j in range(len(self.T[i])):
                f[i].append(int(operator_map_dsr[str(self.T[i][j])]))
            poplation.append(f[i])
        poplation.set_pop_size(self.length_T)
        return poplation

    def to_keplar(self, poplation=None):
        self.length_T = poplation.get_pop_size()
        T_new = [[] for i in range(self.length_T)]
        for i in range(self.length_T):
            for j in range(len(poplation.pop_list[i])):
                T_new[i].append(Operator_map_S[int(poplation.pop_list[i][j])])

        return T_new


class KeplarToDSR():

    def __init__(self):
        # self.poplation = poplation
        self.length_T = None
        # self.T = T
        # self.operator_map = operator_map

    def do(self, poplation=None):
        self.length_T = poplation.get_pop_size()
        T_new = [[] for i in range(self.length_T)]
        for i in range(self.length_T):
            for j in range(len(poplation.pop_list[i])):
                T_new[i].append(operator_map_dsr2[int(poplation.pop_list[i][j])])

        return T_new


def gp_to_bingo1(gp_equ):
    strx_ = re.sub(r'X(\d{3})', r'X_\1', gp_equ)
    strx_ = re.sub(r'X(\d{2})', r'X_\1', gp_equ)
    strx_ = re.sub(r'X(\d{1})', r'X_\1', gp_equ)
    return strx_


def bingo_to_gp(bg_equ):
    strx_ = re.sub(r'X_(\d{3})', r'X\1', bg_equ)
    strx_ = re.sub(r'X_(\d{2})', r'X\1', strx_)
    strx_ = re.sub(r'X_(\d{1})', r'X\1', strx_)
    return strx_


def gp_to_bingo(gp_equ):
    strx_ = re.sub(r'X(\d{1})', r'X_\1', gp_equ)
    strx_ = re.sub(r'-(\d{1}).(\d{3})', r'sub(0.000,\1.\2)', strx_)
    strx_ = re.sub(r'add', r'+', strx_)
    strx_ = re.sub(r'sub', '-', strx_)
    strx_ = re.sub(r'mul', r'*', strx_)
    strx_ = re.sub(r'div', r'/', strx_)
    strx_ = re.sub(r',', ' ', strx_)
    strx_ = re.sub(r'\(', r' ', strx_)
    strx_ = re.sub(r'\)', r' ', strx_)
    strx_ = re.sub(r'  ', ' ', strx_)
    strx_ = re.sub(r'   ', ' ', strx_)
    strx_ = pre_to_mid(str(strx_))
    # strx_ = infix_to_postfix(strx_)
    strx_ = "".join(strx_)
    strx_ = re.sub(r'-', ' - ', strx_)
    strx_ = re.sub(r'\+', ' + ', strx_)
    strx_ = re.sub(r'\*', ' * ', strx_)
    strx_ = re.sub(r'/', ' / ', strx_)
    strx_ = re.sub(r'X', r' X', strx_)
    strx_ = re.sub(r'0.000', '0.000 ', strx_)
    strx_ = re.sub(r'X_(\d{1})(\d{1})', r'X_\1 \2', strx_)
    strx_ = re.sub(r'(\d{1})0.000', r'\1 0.000', strx_)
    strx_ = re.sub(r'.(\d{3})(\d{1})', r'.\1 \2', strx_)
    return str(strx_)


class trans_Dsr():
    def __init__(self):
        pass

    def pop2Dsr(self, poplation, programs):
        pass

    def Dsr2pop(self, poplation, programs):
        pass


def postfix_to_infix(expression):
    stack = []

    for token in expression:
        if token.isalnum():  # 操作数，直接入栈
            stack.append(token)
        else:  # 运算符，弹出两个操作数并生成中缀表达式
            operand2 = stack.pop()
            operand1 = stack.pop()
            infix = "(" + operand1 + token + operand2 + ")"
            stack.append(infix)

    return stack.pop()  # 返回最终中缀表达式


# def trans_op(tree):
#     equ = []
#     for j in tree.Nodes:
#         if str(j.Name) == "variable":
#             equ.append("X_1")
#         elif str(j.Name) == "constant":
#             equ.append("1")
#         elif str(j.Name) == "square":
#             equ.append("sqrt")
#         elif str(j.Name) == "pow":
#             equ.append("^")
#         else:
#             equ.append(str(j.Name))
#     a, b = postfix_to_command_array_and_constants(equ)
#     ind = AGraph()
#     ind.command_array = a
#     ind._update()
#     return str(ind)

def trans_op0(equ):
    equ1 = re.sub(r'X(\d{3})', r'X_\1', equ)
    equ1 = re.sub(r'X(\d{2})', r'X_\1', equ1)
    equ1 = re.sub(r'X(\d{1})', r'X_\1', equ1)
    return equ1


def trans_op1(equ):
    equ1 = re.sub(r'X(\d{3})', r'X_\1', equ)
    equ1 = re.sub(r'X(\d{2})', r'X_\1', equ1)
    equ1 = re.sub(r'X(\d{1})', r'X_\1', equ1)
    pattern = r'(X_\d+)'
    output_string = re.sub(pattern, lambda m: m.group(1)[:-1] + str(int(m.group(1)[-1]) - 1), equ1)
    return output_string


def trans_op2(equ):
    equ1 = re.sub(r'x(\d{3})', r'X_\1', equ)
    equ1 = re.sub(r'x(\d{2})', r'X_\1', equ1)
    equ1 = re.sub(r'x(\d{1})', r'X_\1', equ1)
    pattern = r'(x_\d+)'
    output_string = re.sub(pattern, lambda m: m.group(1)[:-1] + str(int(m.group(1)[-1]) - 1), equ1)
    return output_string


def op_postfix_to_prefix(node_list):
    # print("--------------------------")
    # for node in node_list:
    #     print(node.Name)
    stack = []
    for node in node_list:
        if not node.IsLeaf:
            if node.Arity == 1:
                operand1 = stack.pop()
                sub_stack = []
                sub_stack.append(node)
                sub_stack.append(operand1)
                stack.append(sub_stack)
            elif node.Arity == 2:

                operand2 = stack.pop()
                operand1 = stack.pop()
                sub_stack = []
                sub_stack.append(node)
                sub_stack.append(operand1)
                sub_stack.append(operand2)
                stack.append(sub_stack)


            else:
                raise ValueError("Arity>=3")
        else:
            stack.append(node)

    new_node_list = []
    stack1 = [stack]
    while stack1:
        item = stack1.pop()
        if isinstance(item, list):
            stack1.extend(reversed(item))
        else:
            new_node_list.append(item)
    return new_node_list


def bgpostfix_to_gpprefix(post_equ_list):
    stack = []
    new_list = [sqrt1 if x == 'sqrt' else x for x in post_equ_list]
    new_list = [add2 if x == '+' else x for x in new_list]
    new_list = [sub2 if x == '-' else x for x in new_list]
    new_list = [mul2 if x == '*' else x for x in new_list]
    new_list = [div2 if x == '/' else x for x in new_list]
    new_list = [power2 if x == '^' else x for x in new_list]
    new_list = [exp1 if x == 'exp' else x for x in new_list]
    new_list = [sin1 if x == 'sin' else x for x in new_list]
    new_list = [cos1 if x == 'cos' else x for x in new_list]
    new_list = [log1 if x == 'log' else x for x in new_list]
    for token in new_list:
        if isinstance(token, _Function):
            if token.arity == 1:
                operand1 = stack.pop()
                sub_stack = []
                sub_stack.append(token)
                sub_stack.append(operand1)
                stack.append(sub_stack)
            elif token.arity == 2:
                operand2 = stack.pop()
                operand1 = stack.pop()
                sub_stack = []
                sub_stack.append(token)
                sub_stack.append(operand1)
                sub_stack.append(operand2)
                stack.append(sub_stack)
            else:
                raise ValueError("Arity>=3")
        else:
            if token[0] == 'x':
                token = token[1:]
                token = int(token)
            else:
                token = float(token)
            stack.append(token)
    new_node_list = []
    stack1 = [stack]
    while stack1:
        item = stack1.pop()
        if isinstance(item, list):
            stack1.extend(reversed(item))
        else:
            new_node_list.append(item)
    return new_node_list


def lable_list_to_gp_list(lable_list):
    for i in range(len(lable_list)):
        if lable_list[i][0] == 'X':
            lable_list[i] = int(lable_list[i][1:])
        elif is_float(lable_list[i]):
            lable_list[i] = float(lable_list[i])
        elif lable_list[i] == 'add':
            lable_list[i] = add2
        elif lable_list[i] == 'sub':
            lable_list[i] = sub2
        elif lable_list[i] == 'mul':
            lable_list[i] = mul2
        elif lable_list[i] == 'div':
            lable_list[i] = div2
        elif lable_list[i] == 'sqrt':
            lable_list[i] = sqrt1
        elif lable_list[i] == 'neg':
            lable_list[i] = neg1

    return lable_list


def trans_op(op_tree, variable_list):
    var_dict = {}
    for var in variable_list:
        var_dict[int(var.Hash)] = str(var.Name)
    func = []
    const_array = []
    c_num = 0
    const_code = 5000
    variable_code = 3000
    node_list = op_tree.Nodes
    node_list = op_postfix_to_prefix(node_list)
    for node in node_list:
        if node.IsLeaf:
            if node.IsConstant:
                token = str(const_code)
                const_array.append(node.Value)
                c_num = c_num + 1
                const_code = const_code + 1
                func.append(token)
            else:
                variable_code = variable_code + 1
                var_name = var_dict[node.HashValue]
                var_name = var_name[1:]
                token = 2999 + int(var_name)
                token = str(token)
                func.append(token)
        else:
            if node.Type == Operon.NodeType.Add:
                token = "1001"
                func.append(token)
            elif node.Type == Operon.NodeType.Sub:
                token = "1002"
                func.append(token)
            elif node.Type == Operon.NodeType.Mul:
                token = "1003"
                func.append(token)
            elif node.Type == Operon.NodeType.Aq:
                token = "1016"
                func.append(token)
            elif node.Type == Operon.NodeType.Pow:
                token = "1017"
                func.append(token)
            elif node.Type == Operon.NodeType.Div:
                token = "1004"
                func.append(token)
            elif node.Type == Operon.NodeType.Sqrt:
                token = "1005"
                func.append(token)
            elif node.Type == Operon.NodeType.Log:
                token = "1006"
                func.append(token)
            elif node.Type == Operon.NodeType.Abs:
                token = "1007"
                func.append(token)
            elif node.Type == Operon.NodeType.Fmax:
                token = "1010"
                func.append(token)
            elif node.Type == Operon.NodeType.Fmin:
                token = "1011"
                func.append(token)
            elif node.Type == Operon.NodeType.Sin:
                token = "1012"
                func.append(token)
            elif node.Type == Operon.NodeType.Cos:
                token = "1013"
                func.append(token)
            elif node.Type == Operon.NodeType.Tan:
                token = "1014"
                func.append(token)
            elif node.Type == Operon.NodeType.Exp:
                token = "1018"
                func.append(token)
            elif node.Type == Operon.NodeType.Square:
                token = "1019"
                func.append(token)

            else:
                raise ValueError(f"{node.Name}转换operon节点时未识别")
    return func, const_array


def to_bingo(ind):
    str_equ = ind.format()
    bingo_graph = AGraph(equation=str_equ)
    bingo_graph._update()
    return bingo_graph


def to_dsr(length, *T, **map):
    f = []
    for i in range(length):
        for j in range(len(T[i])):
            f[i].append(map[T[i][j]])


def to_op(ind, np_x, np_y):
    # print("----------------")
    str_equ = ind.format()
    # print(str_equ)
    # print(ind.format())
    str_equ = re.sub(r'x(\d{3})', r'X_\1', str_equ)
    str_equ = re.sub(r'x(\d{2})', r'X_\1', str_equ)
    str_equ = re.sub(r'x(\d{1})', r'X_\1', str_equ)
    # print(ind.func)
    str_equ = re.sub(r' pow ', r" ^ ", str_equ)
    str_equ = re.sub(r' power ', r" ^ ", str_equ)
    # print(str_equ)
    list_equ = eq_string_to_infix_tokens(str_equ)
    # print(list_equ)
    ds = Operon.Dataset(np.hstack([np_x, np_y]))
    # func = ind.func
    # list_prefix = []
    # for i in func:
    #     int_i = int(i)
    #     if int_i < 3000:
    #         str_op = map_F1[int_i]
    #         list_prefix.append(str_op)
    #     elif 3000 <= int_i < 5000:
    #         str_x = "X_" + str(int_i - 2999)
    #         list_prefix.append(str_x)
    #     elif int_i >= 5000:
    #         str_con = str(ind.const_array[int_i - 5000])
    #         list_prefix.append(str_con)
    #
    #     else:
    #         raise ValueError("留空")
    # print(list_equ)
    list_postfix = infix_to_postfix(list_equ)
    # print(list_postfix)
    node_list = []
    var_hash_dict = {}
    variables = ds.Variables
    for var in variables:
        print(var.Index)
        print(var.Name)
        print(var.Hash)
        hash_ = var.Hash
        var_hash_dict[var.Index] = var.Hash
        # print(len(list_postfix))
    for token in list_postfix:
        # print("*")
        token = re.sub(r'x(\d{3})', r'X_\1', token)
        token = re.sub(r'x(\d{2})', r'X_\1', token)
        token = re.sub(r'x(\d{1})', r'X_\1', token)
        # print(token)
        if is_float(token):
            node = Operon.Node.Constant(float(token))
            node_list.append(node)
        elif (token[0] == 'x' and token[1] == '_') or (token[0] == 'X' and token[1] == '_'):
            var_num_str = token[2:]
            var_num = int(var_num_str)
            node = Operon.Node.Variable(1)
            node.HashValue = var_hash_dict[var_num]
            node_list.append(node)
        elif token == '+':
            node = Operon.Node.Add()
            node_list.append(node)
        elif token == '-':
            node = Operon.Node.Sub()
            node_list.append(node)
        elif token == '*':
            node = Operon.Node.Mul()
            node_list.append(node)
        elif token == '/':
            node = Operon.Node.Div()
            node_list.append(node)
        elif token == '^':
            node = Operon.Node.Pow()
            node_list.append(node)
        elif token == 'exp':
            node = Operon.Node.Exp()
            node_list.append(node)
        elif token == 'log':
            node = Operon.Node.Log()
            node_list.append(node)
        elif token == 'sin':
            node = Operon.Node.Sin()
            node_list.append(node)
        elif token == 'cos':
            node = Operon.Node.Cos()
            node_list.append(node)
        elif token == 'tan':
            node = Operon.Node.Tan()
            node_list.append(node)
        elif token == 'tanh':
            node = Operon.Node.Tanh()
            node_list.append(node)
        elif token == 'sqrt':
            node = Operon.Node.Sqrt()
            node_list.append(node)
        elif token == 'cbrt':
            node = Operon.Node.Cbrt()
            node_list.append(node)
        elif token == 'dyn':
            node = Operon.Node.Dyn()
            node_list.append(node)
        elif token == 'power':
            node = Operon.Node.Pow()
            node_list.append(node)
        elif token == 'pow':
            node = Operon.Node.Pow()
            node_list.append(node)
        else:
            raise ValueError(f"通用个体转换为Operon个体时未识别,未识别字符为{token}")
    # print(len(node_list))
    op_tree = Operon.Tree(node_list)
    # print(op_tree.Length)
    op_tree.UpdateNodes()
    # print(Operon.InfixFormatter.Format(op_tree, ds, 5))
    return op_tree


def equ_to_op(equ, ds):
    list_infix = []
    op_al = ""
    x_al = ""
    num_al = ""
    op_start = False
    x_start = False
    num_start = False
    i = 0
    while i < len(equ):
        if op_start:
            if equ[i].isalnum():
                op_al = op_al + str(equ[i])
                i = i + 1
            else:
                list_infix.append(op_al)
                op_al = ""
                op_start = False
        elif x_start:
            if equ[i] == "_":
                x_al = x_al + "_"
                i = i + 1
            elif equ[i].isdigit():
                x_al = x_al + str(equ[i])
                i = i + 1
            else:
                list_infix.append(x_al)
                x_al = ""
                x_start = False
        elif num_start:
            if equ[i].isdigit() or equ[i] == ".":
                num_al = num_al + str(equ[i])
                i = i + 1
            else:
                list_infix.append(num_al)
                num_al = ""
                num_start = False
        else:
            if equ[i].isalpha() and equ[i] != "X":
                op_start = True
                op_al = str(equ[i])
                i = i + 1
            elif equ[i] == " ":
                continue
            elif equ[i] == "X":
                x_start = True
                x_al = "X"
                i = i + 1
            elif equ[i].isdigit():
                num_start = True
                num_al = str(equ[i])
                i = i + 1
            else:
                list_infix.append(str(equ[i]))
                i = i + 1
    if num_al != "":
        list_infix.append(num_al)
    if x_al != "":
        list_infix.append(x_al)
    if op_al != "":
        list_infix.append(op_al)
    post_equ = infix_to_postfix(list_infix)
    # print(post_equ)
    node_list = []
    var_hash = []
    variables = ds.Variables
    for var in variables:
        hash_ = var.Hash
        var_hash.append(hash_)
    for token in post_equ:
        if is_float(token):
            node = Operon.Node.Constant(float(token))
            node_list.append(node)
        elif token[0] == 'x' and token[1] == '_':
            var_num_str = token[2:]
            var_num = int(var_num_str)
            node = Operon.Node.Variable(1)
            node.HashValue = var_hash[var_num]
            node_list.append(node)
        elif token == '+':
            node = Operon.Node.Add()
            node_list.append(node)
        elif token == '-':
            node = Operon.Node.Sub()
            node_list.append(node)
        elif token == '*':
            node = Operon.Node.Mul()
            node_list.append(node)
        elif token == '/':
            node = Operon.Node.Div()
            node_list.append(node)
        elif token == '^':
            node = Operon.Node.Pow()
            node_list.append(node)
        elif token == 'exp':
            node = Operon.Node.Exp()
            node_list.append(node)
        elif token == 'log':
            node = Operon.Node.Log()
            node_list.append(node)
        elif token == 'sin':
            node = Operon.Node.Sin()
            node_list.append(node)
        elif token == 'cos':
            node = Operon.Node.Cos()
            node_list.append(node)
        elif token == 'tan':
            node = Operon.Node.Tan()
            node_list.append(node)
        elif token == 'tanh':
            node = Operon.Node.Tanh()
            node_list.append(node)
        elif token == 'sqrt':
            node = Operon.Node.Sqrt()
            node_list.append(node)
        elif token == 'cbrt':
            node = Operon.Node.Cbrt()
            node_list.append(node)
        elif token == 'dyn':
            node = Operon.Node.Dyn()
            node_list.append(node)
        else:
            raise ValueError(f"通用个体转换为Operon个体时未识别,未识别字符为{token}")
    op_tree = Operon.Tree(node_list)
    op_tree.UpdateNodes()
    # print(Operon.InfixFormatter.Format(op_tree, ds, 5))
    return op_tree
