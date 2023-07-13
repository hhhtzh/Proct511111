from keplar.population.population import Population

data = Data("pmlb", "1027_ESL", ["x1", "x2", "x3", 'y'])
data.read_file()
# data.set_xy("y")
pop = Population(100)
fit_list = []
time_list = []
mt = MTaylorKMeansAlg(1000, data, population=pop)
for i in range(10):
    mt.run()
    fit_list.append(mt.best_fit)
    time_list.append(mt.elapse_time)