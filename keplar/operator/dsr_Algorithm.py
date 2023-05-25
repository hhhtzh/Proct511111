# import dsr.dso as dso
import pandas as pd
import numpy as np  

from keplar.operator.operator import Operator
from keplar.operator.composite_operator import CompositeOp
from keplar.operator.dsr_prework import pre_env
from keplar.operator.dsr_Deeplearning import uDsrDeeplearning
from keplar.operator.dsr_Evo import uDsrEvo
from keplar.operator.dsr_RL import uDsrRL
from keplar.operator.dsr_Evo import uDsrEvoCompositOp
from dsr.dso.core import DeepSymbolicOptimizer
from keplar.operator.creator import Creator,uDSRCreator
from copy import deepcopy
from dsr.dso.program import Program
from dsr.dso.train import Trainer
from dsr.dso.program import Program, from_tokens

class uDsrAlgorithm:
    def __init__(self,csv_filename,config_filename):
        # 读入数据
        self.csv_filename = csv_filename
        self.config_filename = config_filename

    def run(self):

        # 对数据进行预处理
        prepare_env = pre_env(self.csv_filename,self.config_filename)
        config = prepare_env.do()

        # 进行深度学习,并对model进行setup
        dsr_model = uDsrDeeplearning(deepcopy(config))
        dsr_model.do()

        # self.mytrain=model.trainer

        # population = uDSRCreator(30)
        # population.do()
        programs=Program
        
        # print("done!\n")

        # 进行演化算法
        dsr_Evo = uDsrEvo()

        # 进行强化学习
        dsr_RL =uDsrRL()

        # 进行组合
        dsr_2 =CompositeOp([dsr_Evo,dsr_RL]) 
        # 这里的dsr_Evo和dsr_RL都是CompositeOp类型的，可以演化多次再做一次强化学习（当前是一次演化一次强化学习）

        # 进行组合
        uDsr = uDsrEvoCompositOp([dsr_2],iter_num=20)
        uDsr.do()
        





        