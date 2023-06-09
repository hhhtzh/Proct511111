from keplar.population.function import operator_map


class Individual:
    def __init__(self, func):
        self.equation = None
        # prefix list
        self.func = func
        self.evaluated = False
        self.fitness = None
        self.const_array=[]

    def format(self):
        str_equ=[]
        for i in self.func:
            tk = int(i)
            if i<2000:
                str_op=operator_map[str(i)]
                str_equ.append()
            elif 2000 <= i < 3000:
                const=self.const_array[i]
                str_equ.append(str(const))
            elif i>3000:
                x_num=i-3000
                x_str="X_"+str(x_num)
                str_equ.append(x_str)
            else:
                raise ValueError(f"编码无意义{i}")

    def reset_equation(self, new_equation):
        self.equation = new_equation

    def get_fitness(self):
        return self.fitness

    def set_fitness(self, fitness):
        self.fitness = fitness
