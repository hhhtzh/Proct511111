import re

from bingo.symbolic_regression import AGraph
from bingo.symbolic_regression.agraph.string_parsing import infix_to_postfix, postfix_to_command_array_and_constants


def get_priority(op):
    if op == '+' or op == '-':
        return 0
    else:
        return 1


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


def to_op(ind):
    equ = ind.equation
    list_infix = []
    op_al = ""
    x_al = ""
    op_start = False
    x_start = False
    for token in equ:
        if op_start:
            if token.isalnum():
                op_al = op_al + str(token)
            else:
                list_infix.append(op_al)
                op_al = ""
                op_start = False
                if token != " ":
                    list_infix.append(token)
        elif x_start:
            if token == "_":
                x_al = x_al + "_"
            elif token.isdigit():
                x_al = x_al + str(token)
            else:
                list_infix.append(x_al)
                x_al = ""
                x_start = False
                if token != " ":
                    list_infix.append(token)
        else:
            if token.isalnum() and token != "X":
                op_start = True
                op_al = str(token)
            elif token == " ":
                pass
            elif token == "X":
                x_start = True
                x_al = "X"
            else:
                list_infix.append(str(token))
    print(list_infix)
    post_equ = infix_to_postfix(list_infix)
    print(post_equ)
