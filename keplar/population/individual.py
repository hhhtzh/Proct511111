from keplar.population.function import  arity_map, operator_map2


class Individual:
    def __init__(self, func, const_array=None):
        if const_array is None:
            const_array = []
        self.equation = None
        # prefix list
        self.func = func
        self.evaluated = False
        self.fitness = None
        if const_array is None:
            self.const_array = []
        else:
            self.const_array = const_array

    def format(self):
        stack = []
        stack_i = 0
        for i in reversed(self.func):
            tk = int(i)
            if tk < 2000:
                str_op = operator_map2[tk]
                arity = arity_map[tk]
                if arity == 1:
                    if str_op == "square":
                        op1 = stack[stack_i - 1]
                        equ_str = "((" + op1 + ")" + "^2" + ")"
                        stack[stack_i - 1] = equ_str
                    else:
                        op1 = stack[stack_i - 1]
                        equ_str = "(" + str_op + "(" + op1 + ")" + ")"
                        stack[stack_i - 1] = equ_str
                if arity == 2:
                    op1 = stack[stack_i - 1]
                    op2 = stack[stack_i - 2]
                    stack_i = stack_i - 1
                    equ_str = "(" + op1 + " " + str_op + " " + op2 + ")"
                    stack[stack_i - 1] = equ_str
            elif 2000 <= tk < 3000:
                const = self.const_array[tk - 2000]
                stack.append(str(const))
                stack_i = stack_i + 1
            elif tk > 5000:
                x_num = tk - 5000
                x_str = "X_" + str(x_num)
                stack.append(x_str)
                stack_i = stack_i + 1
            else:
                raise ValueError(f"编码无意义{tk}")
        return stack[0]

    def reset_equation(self, new_equation):
        self.equation = new_equation

    def get_fitness(self):
        return self.fitness

    def set_fitness(self, fitness):
        self.fitness = fitness
