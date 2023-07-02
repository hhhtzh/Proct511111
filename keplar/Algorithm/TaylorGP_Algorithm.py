import random
import sys
import time

import numpy as np

from keplar.Algorithm.Alg import Alg


# from keplar.operator.creator import OperonCreator
# from 

class TayloGPAlg(Alg):
    # def __init__(self, max_generation, up_op_list, down_op_list, eva_op_list, error_tolerance, population):
    #     super().__init__(max_generation, up_op_list, down_op_list, eva_op_list, error_tolerance, population)
    def __init__(self, max_generation, up_op_list, down_op_list, eval_op_list, error_tolerance, population):
        # super().__init__()
        super().__init__(max_generation, up_op_list, down_op_list, eval_op_list, error_tolerance, population)

    def run(self):
        # return super().run()
        # return super().run()
        print("done!")
