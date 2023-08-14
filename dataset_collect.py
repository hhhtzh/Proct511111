from keplar.data.data import Data

data = Data("pmlb", "1027_ESL", ["x1", "x2", "x3", 'y'])
data.read_file()
operators = ["+", "-", "*", "/", "sin", "exp", "cos", 'sqrt', 'log', 'sin', 'pow', 'exp', '^']
