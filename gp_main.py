from gplearn.genetic import SymbolicRegressor

sr = SymbolicRegressor(population_size=128, generations=1000,
                       function_set=["add", "sub", "mul", "div", "sin", "exp", "cos", 'sqrt', 'log', 'sin', 'pow', 'exp', '^'])
