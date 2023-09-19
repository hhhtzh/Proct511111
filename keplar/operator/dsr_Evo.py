from keplar.operator.operator import Operator
from keplar.population.population import Population
from keplar.operator.composite_operator import CompositeOp

from dsr.dso.gp.gp_controller import GPController
from dsr.dso.memory import Batch, make_queue


class uDsrEvoCompositOp(CompositeOp):
    def __init__(self, op_list, iter_num):
        super().__init__()
        self.op_list = op_list
        self.iter_num = iter_num

    def do(self, population):
        num = 0
        while num < self.iter_num:
            for op in self.op_list:
                op.do(population)
            num += 1


class uDsrEvo():
    def __init__(self):
        pass

    def do(self, population=None):
        # 整个演化的过程，可以分解出来几部分，包括corssover，mutation，selection，evaluation
        population = Population(pop_size=30)
        # Run GP seeded with the current batch, returning elite samples
        if self.gp_controller is not None:
            deap_programs, deap_actions, deap_obs, deap_priors = self.gp_controller(actions)
            self.nevals += self.gp_controller.nevals

        # 初始化种群
