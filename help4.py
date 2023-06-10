# def is_operator(char):
#     operators = ['+', '-', '*', '/']
#     return char in operators
#
#
# def postfix_to_prefix(expression):
#     stack = []
#     for char in expression:
#         if is_operator(char):
#             operand2 = stack.pop()
#             operand1 = stack.pop()
#             stack.append(char + operand1 + operand2)
#         else:
#             stack.append(char)
#     return stack
#
#
# # 测试
# postfix_expression = ['2', '3', '4', '*', '+', '5', '-']
# prefix_expression = postfix_to_prefix(postfix_expression)
# print("前缀表达式：", prefix_expression)
# dict={"11":2}
# print(dict["11"])
# str_equ="(X_1)((X_1 - ((X_1)/(X_1)))((X_1)(X_1)) - (X_0 - (X_0)))"
from keplar.translator.translator import prefix_to_postfix


def prefix_to_postfix(expression):
    stack = []
    operators = {'add': 1, 'sub': 2, 'mul': 3, 'div': 4, 'sqrt': 5, 'log': 6, 'abs': 7,
                 'neg': 8, 'inv': 9, 'max': 10, 'min': 11, 'sin': 12, 'cos': 13, 'tan': 14,
                 'sig': 15, 'aq': 16, 'pow': 17, 'exp': 18, 'square': 19,}  # 可用的运算符

    for token in reversed(expression):
        if token in operators:  # 操作符
            # 弹出栈顶运算符，直到遇到更低优先级的运算符或左括号
            while stack and stack[-1] in operators and operators[token] <= operators[stack[-1]]:
                yield stack.pop()
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


prefix_expression = ['add', 'mul', '5', '2', '4']
postfix_expression = list(prefix_to_postfix(prefix_expression))
print("前缀表达式:", prefix_expression)
print("后缀表达式:", postfix_expression)
