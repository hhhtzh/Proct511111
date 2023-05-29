import re

import numpy as np
import pyoperon as Operon
from bingo.symbolic_regression import AGraph
from bingo.symbolic_regression.agraph.string_parsing import infix_to_postfix, postfix_to_command_array_and_constants


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


def trans_op(equ):
    equ1 = re.sub(r'X(\d{3})', r'X_\1', equ)
    equ1 = re.sub(r'X(\d{2})', r'X_\1', equ1)
    equ1 = re.sub(r'X(\d{1})', r'X_\1', equ1)
    pattern = r'(X_\d+)'
    output_string = re.sub(pattern, lambda m: m.group(1)[:-1] + str(int(m.group(1)[-1]) - 1), equ1)
    return output_string


def to_op(ind,np_x,np_y):
    ds = Operon.Dataset(np.hstack([np_x, np_y]))
    equ = ind.equation
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
    print(post_equ)
    node_list=[]
    var_hash=[]
    variables=ds.Variables
    for var in variables:
        hash=var.Hash
        var_hash.append(hash)
    for token in post_equ:
        if is_float(token):
            node=Operon.Node.Constant(float(token))
            node_list.append(node)
        elif token[0]=='X' and token[1]== '_':
            var_num_str=token[2:]
            var_num=int(var_num_str)
            node=Operon.Node.Variable(1)
            node.HashValue=var_hash[var_num]
            node_list.append(node)
        elif token=='+':
            node=Operon.Node.Add()
            node_list.append(node)
        elif token=='-':
            node=Operon.Node.Sub()
            node_list.append(node)
        elif token=='*':
            node=Operon.Node.Mul()
            node_list.append(node)
        elif token=='/':
            node = Operon.Node.Div()
            node_list.append(node)
        elif token=='^':
            node = Operon.Node.Pow()
            node_list.append(node)
        elif token=='exp':
            node = Operon.Node.Exp()
            node_list.append(node)
        elif token=='log':
            node = Operon.Node.Log()
            node_list.append(node)
        elif token=='sin':
            node = Operon.Node.Sin()
            node_list.append(node)
        elif token=='cos':
            node = Operon.Node.Cos()
            node_list.append(node)
        elif token=='tan':
            node = Operon.Node.Tan()
            node_list.append(node)
        elif token=='tanh':
            node = Operon.Node.Tanh()
            node_list.append(node)
        elif token=='sqrt':
            node = Operon.Node.Sqrt()
            node_list.append(node)
        elif token=='cbrt':
            node = Operon.Node.Cbrt()
            node_list.append(node)
        elif token=='dyn':
            node = Operon.Node.Dyn()
            node_list.append(node)
        else:
            raise ValueError(f"通用个体转换为Operon个体时未识别,未识别字符为{token}")
    op_tree=Operon.Tree(node_list)
    op_tree.UpdateNodes()
    print(Operon.InfixFormatter.Format(op_tree, ds, 5))
    return op_tree
