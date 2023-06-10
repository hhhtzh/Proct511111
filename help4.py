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
str_equ="(X_1)((X_1 - ((X_1)/(X_1)))((X_1)(X_1)) - (X_0 - (X_0)))"

