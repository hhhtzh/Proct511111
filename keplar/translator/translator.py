import re

from bingo.symbolic_regression.agraph.string_parsing import infix_to_postfix


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



def pop2Dsr(self,poplation,programs):
        pass

def Dsr2pop(self,poplation,programs):
        pass


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
