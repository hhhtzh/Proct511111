from keplar.Algorithm.TaylorGP_Algorithm import MTaylorGPAlg
from keplar.data.data import Data
from keplar.population.population import Population

data = Data("txt", "datasets/1.txt", ["x", "y"])
data.read_file()
data.set_xy("y")
pop = Population(100)
mt = MTaylorGPAlg(1000, data, population=pop,SR_method="Operon")
mt.run()
