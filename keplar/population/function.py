import numpy as np
from gplearn.functions import _protected_division, _protected_sqrt

operator_map = {"00001": "+", "00010": "-", "00011": "*", "00100": "/", "00101": "^", "00110": "max"
    , "00111": "min", "01000": "sin", "01001": "cos", "01010": "sinh", "01011": "cosh", "01100": "exp",
                "01101": "neg", "01110": "abs", "01111": "sqrt"}

function_map = {"00001": np.add, "00010": np.subtract, "00011": np.multiply, "00100": _protected_division,
                "00101": _protected_sqrt, "00110": np.maximum
    , "00111": np.minimum, "01000": np.sin, "01001": np.cos, "01010": np.sinh, "01011": np.cosh
    , "01100": np.exp, "01101": np.negative, "01110": np.abs, "01111": _protected_sqrt}

arity_map = {"00001": 2, "00010": 2, "00011": 2, "00100": 2,
             "00101": 1, "00110": 2
    , "00111": 2, "01000": 1, "01001": 1, "01010": 1, "01011": 1
    , "01100": 1, "01101": 1, "01110": 1, "01111": 1}
