class Individual:
    def __init__(self, func):
        self.equation = None
        # prefix list
        self.func = func
        self.evaluated = False
        self.fitness = None

    def format(self):
        for i in self.func:
            tk = int(i)
            if i<2000:

            elif i>2000 and i<3000:
            elif i>3000:
            else:
                raise ValueError(f"编码无意义{i}")

    def reset_equation(self, new_equation):
        self.equation = new_equation

    def get_fitness(self):
        return self.fitness

    def set_fitness(self, fitness):
        self.fitness = fitness
