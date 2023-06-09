import pandas as pd

# from keplar.operator.creator import Creator
# from keplar.operator.crossover import BingoCrossover
# from keplar.operator.generator import Generator
from keplar.operator.operator import Operator


class CompositeOp(Operator):
    def __init__(self, op_list):
        super().__init__()
        self.op_list = op_list

    # def __init__(self, config_path):
    #     self.config = pd.read_json(config_path)
    #     operators = self.config["creator"]["operators"]
    #     self.creator = Creator(pop_size=self.config["creator"]["pop_size"], operators=operators)
    #     self.population=None
    #
    # def do(self, x,y=None,dx_dt=None):
    #     if self.config["creator"]["type"] == "bingo":
    #         self.creator.bingo_do(x=x, stack_size=self.config["creator"]["bingo"]["stack_size"])
    #         self.population=self.creator.population
    #     else:
    #         raise ValueError("creator类型未识别")
    #     self.generator = Generator(self.population,self.config["generation"]["generation_pop_size"],x,y,dx_dt)
    #     self.generator.do(self.config["generator"]["max_generation"],self.config["generator"]["error_tolerance"]

    def do(self, population):
        for op in self.op_list:
            op.do(population)


class CompositeOpReturn(CompositeOp):
    def __init__(self, op_list):
        super().__init__(op_list)

    def do(self, population):
        for op in self.op_list:
            re = op.do(population)
        return re
    
class uDsr_CompositeOp(CompositeOp):
    def __init__(self, op_list):
        super().__init__(op_list)
        # self.op_list =op_list

    def do(self, population=None):
        # return super().do(population)
        for op in self.op_list:
            op.exec()

# class UDsrEvoCompositOp(CompositeOp):
#     def __init__(self,op_list, num_gen):
#         self.num_gen
#     def do(self, population):
#         num = 0
#         while num < self.num_gen:
#             for op in self.op_list:
#                 op.do(population)
#             num + = 1


# bingocrossover = BinggoCrossover()
# # BinggoCrossoverdsr_regression = DSRRegression()
# bingomutation = BinggoMutation()
#
# binggocomp = UDsrEvoCompositOp([bingocrossover,bingomutation])
#
# dsr = CompositeOp([dsr,rl])
#
# udsr = CompositeOp([binggocomp,dsr])
#
# u_dsr = UDSRAlgorithm()
# u_dsr.run()
