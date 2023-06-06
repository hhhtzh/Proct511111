class Individual:
    def __init__(self, func):
        self.equation = None
        #prefix list
        self.func = func
        self.evaluated = False
        self.fitness = None

    def reset_equation(self, new_equation):
        self.equation = new_equation

    def get_fitness(self):
        return self.fitness

    def set_fitness(self, fitness):
        self.fitness = fitness
