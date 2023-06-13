import numpy as np
from gplearn.functions import _protected_division, _protected_sqrt, _Function, _protected_log, _protected_inverse, \
    _sigmoid

operator_map = {1001: 'add', 1002: 'sub',1003: 'mul', 1004: 'div', 1005:'sqrt', 1006: 'log'
    , 1007: 'abs', 1008: 'neg', 1009: 'inv', 1010: 'max', 1011: 'min', 1012: 'sin',
                1013: 'cos', 1014: 'tan',1015:'sig',1016:'aq',1017:'pow',1018:'exp',1019:'square',
                3001:'x1', 3002:'x2', 3003:'x3'
                }

operator_map1 = { 'add' : 1001 ,'sub':1002 ,'mul':1003, 'div':1004 ,'sqrt':1005, 'log':1006,'abs':1007,
                 'neg':1008, 'inv':1009, 'max':1010,'min':1011 , 'sin':1012, 'cos':1013, 'tan':1014,
                 'sig':1015, 'aq':1016, 'pow':1017, 'exp':1018, 'square':1019 ,
                 'x1':3001, 'x2':3002, 'x3':3003
                 }
<<<<<<< HEAD
#从2000开始为常量，从5000开始为变量
=======
#从5000开始为常量，从3000开始为变量
>>>>>>> 0f3da5f (commit)

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








