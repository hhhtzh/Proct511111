import re

import numpy as np
# import pyoperon as Operon
from bingo.symbolic_regression import AGraph
from bingo.symbolic_regression.agraph.string_parsing import infix_to_postfix, postfix_to_command_array_and_constants
# from keplar.population.population import Population
from keplar.population.function import operator_map,operator_map_dsr,operator_map_dsr2

# class Operon(object):
#     def __init__(self):
#         pass
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


def trans_gp(gp_equ):
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


def to_gp(equ):
    strx = re.sub(r'.(\d{3}) (\d{1})', r'.\1\2', equ)
    strx = re.sub(r'(\d{1}) 0.000', r'\10.000', strx)
    strx = re.sub(r'X_(\d{1}) (\d{1})', r'\1\2', strx)
    strx = re.sub(r'0.000 ', '0.000', strx)
    strx = re.sub(r' X', r'X', strx)
    strx = re.sub(r' \+ ', r'+', strx)
    strx = re.sub(r' - ', r'-', strx)
    strx = re.sub(r' \* ', r'*', strx)
    strx = re.sub(r' / ', r'/', strx)


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


# def trans_op(equ):
#     equ1 = re.sub(r'X(\d{3})', r'X_\1', equ)
#     equ1 = re.sub(r'X(\d{2})', r'X_\1', equ1)
#     equ1 = re.sub(r'X(\d{1})', r'X_\1', equ1)
#     pattern = r'(X_\d+)'
#     output_string = re.sub(pattern, lambda m: m.group(1)[:-1] + str(int(m.group(1)[-1]) - 1), equ1)
#     return output_string

def op_postfix_to_prefix(node_list):
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
                operand1 = stack.pop()
                operand2 = stack.pop()
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


# def trans_op(op_tree, variable_list):
#     var_dict = {}
#     for var in variable_list:
#         var_dict[int(var.Hash)] = str(var.Name)
#     func = []
#     const_array = []
#     c_num = 0
#     const_code = 2000
#     variable_code = 3000
#     node_list = op_tree.Nodes
#     node_list = op_postfix_to_prefix(node_list)
#     for node in node_list:
#         if node.IsLeaf:
#             if node.IsConstant:
#                 token = str(const_code)
#                 const_array.append(node.Value)
#                 c_num = c_num + 1
#                 const_code = const_code + 1
#                 func.append(token)
#             else:
#                 token = variable_code
#                 variable_code = variable_code + 1
#                 var_name = var_dict[node.HashValue]
#                 var_name = var_name[1:]
#                 token = token + int(var_name)
#                 token = str(token)
#                 func.append(token)
#         else:
#             if node.Type == Operon.NodeType.Add:
#                 token = "1001"
#                 func.append(token)
#             elif node.Type == Operon.NodeType.Sub:
#                 token = "1002"
#                 func.append(token)
#             elif node.Type == Operon.NodeType.Mul:
#                 token = "1003"
#                 func.append(token)
#             elif node.Type == Operon.NodeType.Aq:
#                 token = "1016"
#                 func.append(token)
#             elif node.Type == Operon.NodeType.Pow:
#                 token = "1017"
#                 func.append(token)
#             elif node.Type == Operon.NodeType.Div:
#                 token = "1004"
#                 func.append(token)
#             elif node.Type == Operon.NodeType.Sqrt:
#                 token = "1005"
#                 func.append(token)
#             elif node.Type == Operon.NodeType.Log:
#                 token = "1006"
#                 func.append(token)
#             elif node.Type == Operon.NodeType.Abs:
#                 token = "1007"
#                 func.append(token)
#             elif node.Type == Operon.NodeType.Fmax:
#                 token = "1010"
#                 func.append(token)
#             elif node.Type == Operon.NodeType.Fmin:
#                 token = "1011"
#                 func.append(token)
#             elif node.Type == Operon.NodeType.Sin:
#                 token = "1012"
#                 func.append(token)
#             elif node.Type == Operon.NodeType.Cos:
#                 token = "1013"
#                 func.append(token)
#             elif node.Type == Operon.NodeType.Tan:
#                 token = "1014"
#                 func.append(token)
#             elif node.Type == Operon.NodeType.Exp:
#                 token = "1018"
#                 func.append(token)
#             elif node.Type == Operon.NodeType.Square:
#                 token = "1019"
#                 func.append(token)

#             else:
#                 raise ValueError(f"{node.Name}转换operon节点时未识别")
#     return func


def to_bingo(ind):
    code = ind.func

class DSRToKeplar():
    def __init__(self,T):
        self.T = T
        self.length_T = len(self.T)

        # self.operator_map_dsr = operator_map_dsr
        # self.poplation = poplation

    def do(self,poplation=None):
        f = [[] for i in range(self.length_T)]
        for i in range(self.length_T):
            for j in range(len(self.T[i])):
                f[i].append(int(operator_map_dsr[str(self.T[i][j])]))
            poplation.append(f[i])
        poplation.set_pop_size(self.length_T)
        return poplation
    
    def to_keplar(self,poplation=None):
        self.length_T = poplation.get_pop_size()
        T_new = [[] for i in range(self.length_T)]
        for i in range(self.length_T):
            for j in range(len(poplation.pop_list[i])):
                T_new[i].append(operator_map[int(poplation.pop_list[i][j])])
                # T_new[i].append(str(operator_map[int(poplation.pop_list[i][j])]))

        return T_new

class KeplarToDSR():

    def __init__(self):
        # self.poplation = poplation
        self.length_T = None
        # self.T = T
        # self.operator_map = operator_map

    def do(self,poplation=None):
        self.length_T = poplation.get_pop_size()
        T_new = [[] for i in range(self.length_T)]
        for i in range(self.length_T):
            for j in range(len(poplation.pop_list[i])):

                T_new[i].append(operator_map_dsr2[int(poplation.pop_list[i][j])])
                # T_new[i].append(str(operator_map[int(poplation.pop_list[i][j])]))
            # T_new[i] = T_new[i].strip('"')

        return T_new



           

# def to_dsr(length_T,*T,**operator_map1):
#         f = [[] for i in range(length_T)]
#         for i in range(length_T):
#             for j in range(len(T[i])):
#                 f[i].append(int(operator_map1[str(T[i][j])]))
#             print(f[i])


# def to_op(ind, np_x, np_y):
#     ds = Operon.Dataset(np.hstack([np_x, np_y]))
#     equ = ind.equation
#     list_infix = []
#     op_al = ""
#     x_al = ""
#     num_al = ""
#     op_start = False
#     x_start = False
#     num_start = False
#     i = 0
#     while i < len(equ):
#         if op_start:
#             if equ[i].isalnum():
#                 op_al = op_al + str(equ[i])
#                 i = i + 1
#             else:
#                 list_infix.append(op_al)
#                 op_al = ""
#                 op_start = False
#         elif x_start:
#             if equ[i] == "_":
#                 x_al = x_al + "_"
#                 i = i + 1
#             elif equ[i].isdigit():
#                 x_al = x_al + str(equ[i])
#                 i = i + 1
#             else:
#                 list_infix.append(x_al)
#                 x_al = ""
#                 x_start = False
#         elif num_start:
#             if equ[i].isdigit() or equ[i] == ".":
#                 num_al = num_al + str(equ[i])
#                 i = i + 1
#             else:
#                 list_infix.append(num_al)
#                 num_al = ""
#                 num_start = False
#         else:
#             if equ[i].isalpha() and equ[i] != "X":
#                 op_start = True
#                 op_al = str(equ[i])
#                 i = i + 1
#             elif equ[i] == " ":
#                 continue
#             elif equ[i] == "X":
#                 x_start = True
#                 x_al = "X"
#                 i = i + 1
#             elif equ[i].isdigit():
#                 num_start = True
#                 num_al = str(equ[i])
#                 i = i + 1
#             else:
#                 list_infix.append(str(equ[i]))
#                 i = i + 1
#     if num_al != "":
#         list_infix.append(num_al)
#     if x_al != "":
#         list_infix.append(x_al)
#     if op_al != "":
#         list_infix.append(op_al)
#     post_equ = infix_to_postfix(list_infix)
#     print(post_equ)
#     node_list = []
#     var_hash = []
#     variables = ds.Variables
#     for var in variables:
#         hash = var.Hash
#         var_hash.append(hash)
#     for token in post_equ:
#         if is_float(token):
#             node = Operon.Node.Constant(float(token))
#             node_list.append(node)
#         elif token[0] == 'X' and token[1] == '_':
#             var_num_str = token[2:]
#             var_num = int(var_num_str)
#             node = Operon.Node.Variable(1)
#             node.HashValue = var_hash[var_num]
#             node_list.append(node)
#         elif token == '+':
#             node = Operon.Node.Add()
#             node_list.append(node)
#         elif token == '-':
#             node = Operon.Node.Sub()
#             node_list.append(node)
#         elif token == '*':
#             node = Operon.Node.Mul()
#             node_list.append(node)
#         elif token == '/':
#             node = Operon.Node.Div()
#             node_list.append(node)
#         elif token == '^':
#             node = Operon.Node.Pow()
#             node_list.append(node)
#         elif token == 'exp':
#             node = Operon.Node.Exp()
#             node_list.append(node)
#         elif token == 'log':
#             node = Operon.Node.Log()
#             node_list.append(node)
#         elif token == 'sin':
#             node = Operon.Node.Sin()
#             node_list.append(node)
#         elif token == 'cos':
#             node = Operon.Node.Cos()
#             node_list.append(node)
#         elif token == 'tan':
#             node = Operon.Node.Tan()
#             node_list.append(node)
#         elif token == 'tanh':
#             node = Operon.Node.Tanh()
#             node_list.append(node)
#         elif token == 'sqrt':
#             node = Operon.Node.Sqrt()
#             node_list.append(node)
#         elif token == 'cbrt':
#             node = Operon.Node.Cbrt()
#             node_list.append(node)
#         elif token == 'dyn':
#             node = Operon.Node.Dyn()
#             node_list.append(node)
#         else:
#             raise ValueError(f"通用个体转换为Operon个体时未识别,未识别字符为{token}")
#     op_tree = Operon.Tree(node_list)
#     op_tree.UpdateNodes()
#     print(Operon.InfixFormatter.Format(op_tree, ds, 5))
#     return op_tree
