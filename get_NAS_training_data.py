from keplar.data.data import Data
from keplar.operator.check_pop import CheckPopulation
from keplar.operator.creator import GpCreator

data = Data("pmlb", "1027_ESL", ["x1", "x2", "x3", "x4", 'y'])
data.read_file()
x = data.get_np_x()
y = data.get_np_y()
creator = GpCreator(128, x, y, "Bingo", n_jobs=20)
population = creator.do()
for _ in range(1000):
    ck = CheckPopulation(data)
    ck.do(population)
