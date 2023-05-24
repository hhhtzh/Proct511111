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

        # 进行RNN深度学习,并对model进行setup
        dsr_model = uDsrDeeplearning(deepcopy(config))
        dsr_model.do()
        # train = dsr_model.do()
        # model  =  DeepSymbolicOptimizer(deepcopy(config))

        # model.setup()

        

        # self.mytrain=model.trainer


        # positional_entropy = None
        # top_samples_per_batch = list()
        # if self.debug >= 1:
        #     print("\nDEBUG: Policy parameter means:")
        #     self.print_var_means()

        # ewma = None if self.b_jumpstart else 0.0 # EWMA portion of baseline

        # start_time = time.time()
        # if self.verbose:
        #     print("-- RUNNING ITERATIONS START -------------")


        # Number of extra samples generated during attempt to get
        # batch_size new samples
        # n_extra = 0
        # # Record previous cache before new samples are added by from_tokens
        # s_history = list(Program.cache.keys())

        # actions, obs, priors = self.mytrain.policy.sample(self.mytrain.batch_size)
        # programs = [from_tokens(a) for a in actions]  

        # if self.mytrain.gp_controller is not None:
        #     deap_programs, deap_actions, deap_obs, deap_priors = self.mytrain.gp_controller(actions)
        #     self.mytrain.nevals += self.mytrain.gp_controller.nevals


        # print(self.mytrain.baseline)
        # print(self.mytrain.policy)
        # print(self.mytrain.batch_size)
        # # print(self.mytrain.)

        # actions, obs, priors = self.mytrain.policy.sample(self.mytrain.batch_size)
        # programs = [from_tokens(a) for a in actions]  

        # model.train()
        
        # trainer1 =model.trainer
        # warm_start = None
        # self.batch_size=100

        # warm_start = warm_start if warm_start is not None else self.batch_size
        # actions, obs, priors = model.policy.sample(warm_start)
        # programs = [from_tokens(a) for a in actions]
        # r = np.array([p.r for p in programs])
        # print(r.)
        # print(r[0].print_stats())
        # print(r.all)
        # self.actions, self.obs, self.priors =model.policy.sample(model.policy.batch_size)

        # programs = [from_tokens(a) for a in self.actions] 

        # # self.action.shape[0]=30
        # # self.action.shape[1]=30
        # for p in programs:
        #     print(p.str)


        population = uDSRCreator(30)
        population.do()
        # print("done!\n")

        # 进行演化算法
        dsr_Evo = uDsrEvo()

        # 进行强化学习
        dsr_RL =uDsrRL()
        # dsr_RL = uDsrRL(sees=trainer1.sess,policy= trainer1.policy,policy_optimizer= trainer1.policy_optimizer,gp_controller=trainer1.gp_controller,
        #                 logger=trainer1.logger,pool=trainer1.pool)
       

        # 进行组合
        dsr_2 =CompositeOp([dsr_Evo,dsr_RL]) 
        # 这里的dsr_Evo和dsr_RL都是CompositeOp类型的，可以演化多次再做一次强化学习（当前是一次演化一次强化学习）

        # 进行组合
        uDsr = uDsrEvoCompositOp([dsr_2],iter_num=20)
        uDsr.do()
        





        