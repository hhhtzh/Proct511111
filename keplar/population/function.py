import numpy as np
from gplearn.functions import _protected_division, _protected_sqrt, _Function, _protected_log, _protected_inverse, \
    _sigmoid

# 存放框架的字典编码方式，这一套编码为数字翻译成先序遍历结果1000（函数）、3000（变量）、5000（常量）
# 1000(函数)
map_F1 = {1001: 'add', 1002: 'sub', 1003: 'mul', 1004: 'div', 1005: 'sqrt', 1006: 'log'
    , 1007: 'abs', 1008: 'neg', 1009: 'inv', 1010: 'max', 1011: 'min', 1012: 'sin',
          1013: 'cos', 1014: 'tan', 1015: 'sig', 1016: 'aq', 1017: 'pow', 1018: 'exp', 1019: 'square',
          1020: '^'}

map_F1_format = {'+': 1001, '-': 1002, '*': 1003, '/': 1004, 'sqrt': 1005, 'log': 1006, 'abs': 1007,
                 'neg': 1008, 'inv': 1009, 'max': 1010, 'min': 1011, 'sin': 1012, 'cos': 1013, 'tan': 1014,
                 'sig': 1015, 'aq': 1016, 'pow': 1017, 'exp': 1018, 'square': 1019, '^': 1020
                 }
# 3000（变量）
map_F2 = {3000: 'x0', 3001: 'x1', 3002: 'x2', 3003: 'x3'}
# 5000（常量）
map_F3 = {}

# Operator_map_Format ={map_F1_format,map_F2,map_F3}

# 存放框架的字典编码方式，这一套编码为遍历结果翻译成数字 1000（函数）、3000（变量）、5000（常量）
# 1000(函数)
map_S1 = {'add': 1001, 'sub': 1002, 'mul': 1003, 'div': 1004, 'sqrt': 1005, 'log': 1006
    , 'abs': 1007, 'neg': 1008, 'inv': 1009, 'max': 1010, 'min': 1011, 1012: 'sin',
          1013: 'cos', 1014: 'tan', 1015: 'sig', 1016: 'aq', 1017: 'pow', 1018: 'exp', 1019: 'square',
          1020: '^'}

# 3000（变量）
map_S2_taylor = {'1': 3001, '2': 3002, '3': 3003, '4': 3004, '5': 3005}
map_S2_dsr = {'x1': 3001, 'x2': 3002, 'x3': 3003, 'x4': 3004, 'x5': 3005}
# 5000（常量）
map_S3 = {}

# Operator_map_S_dsr ={map_S1 , map_S2_dsr , map_S3}
# Operator_map_S_taylor ={map_S1 , map_S2_taylor , map_S3}


# ------------------------------------------------------------------------------------#

Operator_map_S = {'add': 1001, 'sub': 1002, 'mul': 1003, 'div': 1004, 'sqrt': 1005, 'log': 1006, 'abs': 1007,
                  'neg': 1008, 'inv': 1009, 'max': 1010, 'min': 1011, 'sin': 1012, 'cos': 1013, 'tan': 1014,
                  'sig': 1015, 'aq': 1016, 'pow': 1017, 'exp': 1018, 'square': 1019,
                  'x1': 5001, 'x2': 5002, 'x3': 5003
                  }

arity_map = {1001: 2, 1002: 2, 1003: 2, 1004: 2, 1005: 1, 1006: 1
    , 1007: 1, 1008: 1, 1009: 1, 1010: 2, 1011: 2, 1012: 1,
             1013: 1, 1014: 1, 1015: 1, 1016: 1, 1017: 1, 1018: 1, 1019: 1, 1020: 2
             }
operator_map2 = {1001: '+', 1002: '-', 1003: '*', 1004: '/', 1005: 'sqrt', 1006: 'log'
    , 1007: 'abs', 1008: 'neg', 1009: 'inv', 1010: 'max', 1011: 'min', 1012: 'sin',
                 1013: 'cos', 1014: 'tan', 1015: 'sig', 1016: 'aq', 1017: 'pow', 1018: 'exp', 1019: 'square',
                 5001: 'x1', 5002: 'x2', 5003: 'x3'
                 }

operator_map_dsr = {'add': 1001, 'sub': 1002, 'mul': 1003, 'div': 1004, 'sqrt': 1005, 'log': 1006, 'abs': 1007,
                    'neg': 1008, 'inv': 1009, 'max': 1010, 'min': 1011, 'sin': 1012, 'cos': 1013, 'tan': 1014,
                    'sig': 1015, 'aq': 1016, 'pow': 1017, 'exp': 1018, 'square': 1019,
                    'x1': 5001, 'x2': 5002, 'x3': 5003
                    }

operator_map_dsr2 = {1001: 'add', 1002: 'sub', 1003: 'mul', 1004: 'div', 1005: 'sqrt', 1006: 'log'
    , 1007: 'abs', 1008: 'neg', 1009: 'inv', 1010: 'max', 1011: 'min', 1012: 'sin',
                     1013: 'cos', 1014: 'tan', 1015: 'sig', 1016: 'aq', 1017: 'pow', 1018: 'exp', 1019: 'square',
                     5001: "x1", 5002: 'x2', 5003: 'x3'
                     }

operator_map3 = {'+': 1001, '-': 1002, '*': 1003, '/': 1004, 'sqrt': 1005, 'log': 1006, 'abs': 1007,
                 'neg': 1008, 'inv': 1009, 'max': 1010, 'min': 1011, 'sin': 1012, 'cos': 1013, 'tan': 1014,
                 'sig': 1015, 'aq': 1016, 'pow': 1017, 'exp': 1018, 'square': 1019, '^': 1020
                 }
# 从3000开始为常量，从5000开始为变量

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
                 '+': add2,
                 'sub': sub2,
                 '-': sub2,
                 'mul': mul2,
                 '*': mul2,
                 'div': div2,
                 '/': div2,
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
