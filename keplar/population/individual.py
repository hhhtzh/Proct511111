from keplar.population.function import arity_map, operator_map2


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

        # genome每个个体的信息记录，它包括parent的元数据以及为产生当前程序而执行的遗传操作
        # example：
        # gnome = {'method': 'Hoist Mutation',
        #        'parent_idx': parent_index,
        #         parent_nodes': removed}
        self.genome = None

    def format(self):
        stack = []
        stack_i = 0
        # print(self.func)
        for i in reversed(self.func):
            tk = int(i)
            # print(tk)
            if tk < 3000:
                # print(stack_i)
                str_op = operator_map2[tk]
                arity = arity_map[tk]
                if arity == 1:
                    if str_op == "square":
                        op1 = stack.pop()
                        equ_str = "((" + op1 + ")" + "^2" + ")"
                        stack.append(equ_str)
                    else:
                        op1 = stack.pop()
                        equ_str = "(" + str_op + "(" + op1 + ")" + ")"
                        stack.append(equ_str)
                if arity == 2:
                    op1 = stack.pop()
                    op2 = stack.pop()
                    equ_str = "(" + op1 + " " + str_op + " " + op2 + ")"
                    stack.append(equ_str)
            elif 3000 <= tk < 5000:
                # print(stack_i)
                x_num = tk - 3000
                x_str = "X_" + str(x_num)
                stack.append(x_str)
            elif tk >= 5000:
                # print(stack_i)
                const = self.const_array[tk - 5000]
                stack.append(str(const))
            else:
                raise ValueError(f"编码无意义{tk}")
            # print(stack)
        return stack[0]

    def reset_equation(self, new_equation):
        self.equation = new_equation

    def get_fitness(self):
        return self.fitness

    def set_fitness(self, fitness):
        self.fitness = fitness
