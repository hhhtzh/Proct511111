from keplar.Algorithm.Alg import BingoAlg
from keplar.data.data import Data

data = Data("txt", "datasets/2.txt", ["x0", "x1", "x2", "x3", "x4", "y"])
data.read_file()
data.set_xy("y")
bingo = BingoAlg(1000, data, operators=["+", "-", "*", "/", "sin", "exp"])
bingo.run()
print(bingo.island.get3top())
