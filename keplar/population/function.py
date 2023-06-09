import numpy as np
from gplearn.functions import _protected_division, _protected_sqrt, _Function, _protected_log, _protected_inverse, \
    _sigmoid

from dsr.dso.functions import unprotected_ops

operator_map = {1001: 'add', 1002: 'sub',1003: 'mul', 1004: 'div', 1005:'sqrt', 1006: 'log'
    , 1007: 'abs', 1008: 'neg', 1009: 'inv', 1010: 'max', 1011: 'min', 1012: 'sin',
                1013: 'cos', 1014: 'tan',1015:'sig',1016:'aq',1017:'pow',1018:'exp',1019:'square',
                3001:'X_1', 3002:'X_2', 3003:'X_3'
                }

operator_map_dsr = { 'add' : 1001 ,'sub':1002 ,'mul':1003, 'div':1004 ,'sqrt':1005, 'log':1006,'abs':1007,
                 'neg':1008, 'inv':1009, 'max':1010,'min':1011 , 'sin':1012, 'cos':1013, 'tan':1014,
                 'sig':1015, 'aq':1016, 'pow':1017, 'exp':1018, 'square':1019 ,
                 'x1':3001, 'x2':3002, 'x3':3003
                 }

operator_map_dsr2 = {1001: 'add', 1002: 'sub',1003: 'mul', 1004: 'div', 1005:'sqrt', 1006: 'log'
    , 1007: 'abs', 1008: 'neg', 1009: 'inv', 1010: 'max', 1011: 'min', 1012: 'sin',
                1013: 'cos', 1014: 'tan',1015:'sig',1016:'aq',1017:'pow',1018:'exp',1019:'square',
                3001:x1, 3002:0x10001, 3003:0x10002
                }

#从2000开始为常量，从3000开始为变量
x1= 
add2 = _Function(function=np.add, name='add', arity=2)
sub2 = _Function(function=np.subtract, name='sub', arity=2)
mul2 = _Function(function=np.multiply, name='mul', arity=2)
div2 = _Function(function=_protected_division, name='div', arity=2)
sqrt1 = _Function(function=_protected_sqrt, name='sqrt', arity=1)
log1 = _Function(function=_protected_log, name='log', arity=1)
neg1 = _Function(function=np.negative, name='neg', arity=1)
inv1 = _Function(function=_protected_inverse, name='inv', arity=1)
abs1 = _Function(function=np.abs, name='abs', arity=1)
max2 = _Function(function=np.maximum, name='max', arity=2)
min2 = _Function(function=np.minimum, name='min', arity=2)
sin1 = _Function(function=np.sin, name='sin', arity=1)
cos1 = _Function(function=np.cos, name='cos', arity=1)
tan1 = _Function(function=np.tan, name='tan', arity=1)
sig1 = _Function(function=_sigmoid, name='sig', arity=1)

_function_map = {'add': add2,
                 'sub': sub2,
                 'mul': mul2,
                 'div': div2,
                 'sqrt': sqrt1,
                 'log': log1,
                 'abs': abs1,
                 'neg': neg1,
                 'inv': inv1,
                 'max': max2,
                 'min': min2,
                 'sin': sin1,
                 'cos': cos1,
                 'tan': tan1}

# unprotected_ops = [
#     # Binary operators
#     Token(np.add, "add", arity=2, complexity=1),
#     Token(np.subtract, "sub", arity=2, complexity=1),
#     Token(np.multiply, "mul", arity=2, complexity=1),
#     Token(np.divide, "div", arity=2, complexity=2),

#     # Built-in unary operators
#     Token(np.sin, "sin", arity=1, complexity=3),
#     Token(np.cos, "cos", arity=1, complexity=3),
#     Token(np.tan, "tan", arity=1, complexity=4),
#     Token(np.exp, "exp", arity=1, complexity=4),
#     Token(np.log, "log", arity=1, complexity=4),
#     Token(np.sqrt, "sqrt", arity=1, complexity=4),
#     Token(np.square, "n2", arity=1, complexity=2),
#     Token(np.negative, "neg", arity=1, complexity=1),
#     Token(np.abs, "abs", arity=1, complexity=2),
#     Token(np.maximum, "max", arity=1, complexity=4),
#     Token(np.minimum, "min", arity=1, complexity=4),
#     Token(np.tanh, "tanh", arity=1, complexity=4),
#     Token(np.reciprocal, "inv", arity=1, complexity=2),

#     # Custom unary operators
#     Token(logabs, "logabs", arity=1, complexity=4),
#     Token(expneg, "expneg", arity=1, complexity=4),
#     Token(n3, "n3", arity=1, complexity=3),
#     Token(n4, "n4", arity=1, complexity=3),
#     Token(sigmoid, "sigmoid", arity=1, complexity=4),
#     Token(harmonic, "harmonic", arity=1, complexity=4)
# ]

function_map_dsr = {
    op.name : op for op in unprotected_ops
    }
# map_dsr = {op.name : op for op in unprotected_ops}
for i in operator_map_dsr2:
    # print(operator_map_dsr2[i])
    if function_map_dsr.get(operator_map_dsr2[i]) != None:
        operator_map_dsr2[i] = function_map_dsr[operator_map_dsr2[i]]

# for i in range(len(operator_map_dsr2)):
#     if operator_map_dsr2[i] in function_map_dsr.keys():
        # operator_map_dsr2[i] = function_map_dsr[operator_map_dsr2[i]]
    # print(operator_map_dsr[i])

# for i in range(len(operator_map_dsr)):

    # print(operator_map_dsr[i])







