from keplar.Algorithm.Alg import KeplarMBingo, KeplarMOperon
from keplar.data.data import Data
from keplar.population.population import Population

data = Data("pmlb", "1027_ESL", ["x1", "x2", "x3", "x4", 'y'])
data.read_file()
operators = ["+", "-", "*", "/", "sin", "exp", "sqrt", "^"]
population = Population(128)
kmb = KeplarMOperon(1000, None, None, None, 0.1, population, data, operators)
kmb.run()