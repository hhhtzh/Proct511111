# import dsr.dso as dso
import pandas as pd
import numpy as np
from keplar.operator.dsr_Deeplearning import uDsrDeeplearning,uDSR_Deeplearn
from copy import deepcopy
from dsr.dso.task import set_task
from dsr.dso.program import Program, from_tokens
from keplar.operator.creator import uDSR_Creator
from keplar.Algorithm.Alg import Alg
from keplar.operator.composite_operator import uDsr_CompositeOp
from keplar.population.function import operator_map,operator_map_dsr
from keplar.translator.translator import KeplarToDSR,DSRToKeplar
from keplar.population.population import Population
from keplar.operator.dsr_model import uDSR_Model,uDSR_Sample
class uDsrAlgorithm(Alg):
    def __init__(self, csv_filename, config_filename):
        # 读入数据
        self.csv_filename = csv_filename
        self.config_filename = config_filename

    def run(self):

        #前期的准备工作
        udsr_deeplearn=uDSR_Deeplearn(self.csv_filename,self.config_filename)
        udsr_deeplearn.exec()

        udsr_model=uDSR_Model(udsr_deeplearn.sess, udsr_deeplearn.policy, udsr_deeplearn.policy_optimizer, udsr_deeplearn.gp_controller,
                            udsr_deeplearn.logger,udsr_deeplearn.pool, **udsr_deeplearn.d)
        #创建一个新的model（RNN）
        self.model = udsr_model.pre_do()


        iter_num = 0
        done = False
        while not done:
            # 采样
            udsr_sample = uDSR_Sample(self.model)
            self.T, self.programs,self.actions, self.obs, self.priors= udsr_sample.do()
            #定义一个新的population
            population = Population(len(self.T))
            
            #创建population
            udsr_creator = uDSR_Creator(self.T)
            #对uDSR类型的编码进行转化population的编码
            udsr_trans = DSRToKeplar(self.T)

            # 在这一步可以将其他算法的poppulation传入，进行组合
            trans_udsr = KeplarToDSR()

            udsr_list = [udsr_creator ,udsr_trans ,trans_udsr]
            udsr_comop = uDsr_CompositeOp(udsr_list)
            T_new = udsr_comop.do(population)

            #将新的population的编码转化为uDSR类型的编码
            for i in range(len(self.programs)):
                self.programs[i].traversal = Program.library.tokenize(T_new[i])


            #一次迭代
            udsr_model.dsr_train.one_iter(programs = self.programs,actions=self.actions,obs=self.obs,priors=self.priors)
            iter_num += 1
            if iter_num > 20:
                done = True


