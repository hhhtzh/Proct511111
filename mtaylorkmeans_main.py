from keplar.Algorithm.TaylorGP_Algorithm import MTaylorGPAlg, MTaylorKMeansAlg
from keplar.data.data import Data
from keplar.population.population import Population

data = Data("txt", "datasets/2.txt", ["x0", "x1", "x2", "x3", "x4", "y"])
data.read_file()
data.set_xy("y")
pop = Population(100)
mt = MTaylorKMeansAlg(1000, data, population=pop, SR_method="Operon")
mt.run()