from abc import abstractmethod


class Alg:
    def __init__(self, max_generation, up_op_list, down_op_list, eva_op_list, error_tolerance, population):
        self.max_generation = max_generation
        self.up_op_list = up_op_list
        self.down_op_list = down_op_list
        self.eva_op_list = eva_op_list
        self.error_tolerance = error_tolerance
        self.population = population
        self.age = 0

    @abstractmethod
    def run(self):
        raise NotImplementedError
    
class uDSR_Alg(Alg):
    def __init__(self, max_generation, up_op_list, down_op_list, eva_op_list, error_tolerance, population):
        super().__init__(max_generation, up_op_list, down_op_list, eva_op_list, error_tolerance, population)
        


class BingoAlg(Alg):

    def __init__(self, max_generation, up_op_list, down_op_list, eva_op_list, error_tolerance, population):
        super().__init__(max_generation, up_op_list, down_op_list,
                         eva_op_list, error_tolerance, population)

    def get_best_fitness(self):
        return self.get_best().get_fitness()

    def get_best(self):
        best_fitness = self.population.pop_list[0].get_fitness()
        for indv in self.population.pop_list:
            if indv.get_fitness() > best_fitness:
                best_fitness = indv.get_fitness()
        return indv

    def get_best_individual(self):

        return self.get_best().equation

    def run(self):
        generation_pop_size = self.population.get_pop_size()
        self.eva_op_list.do(self.population)
        now_error = self.get_best_fitness()
        while self.age < self.max_generation and now_error >= self.error_tolerance or str(now_error) == "nan":
            self.population = self.down_op_list.do(self.population)
            while generation_pop_size > self.population.get_pop_size():
                self.up_op_list.do(self.population)
            self.eva_op_list.do(self.population)
            now_error = self.get_best_fitness()
            best_ind = str(self.get_best_individual())
            self.age += 1
            print("第" + f"{self.age}代种群，" +
                  f"最佳个体适应度为{now_error}" + f"最佳个体为{best_ind}")
        best_ind = str(self.get_best_individual())
        print("迭代结束，共迭代" + f"{self.age}代" +
              f"最佳个体适应度为{now_error}" + f"最佳个体为{best_ind}")
