# import dsr.dso as dso
import pandas as pd
import numpy as np

from keplar.operator.operator import Operator
from keplar.operator.composite_operator import CompositeOp
from keplar.operator.dsr_prework import pre_env
from keplar.operator.dsr_Deeplearning import uDsrDeeplearning,uDsr_deeplearn
# from keplar.operator.dsr_Evo import uDsrEvo
# from keplar.operator.dsr_RL import uDsrRL
from keplar.operator.dsr_Evo import uDsrEvoCompositOp
# from dsr.dso.core import DeepSymbolicOptimizer
# from keplar.operator.creator import Creator,uDSRCreator
from copy import deepcopy
from dsr.dso.task import set_task
# from dsr.dso.program import Program
# from dsr.dso.train import Trainer
from dsr.dso.program import Program, from_tokens

# from keplar.translator.translator import pop2Dsr, Dsr2pop, DSRTransPOP, POPTransPOP
from dsr.dso.train import Trainer

from keplar.operator.dsr_train import dsr_Train

from keplar.operator.creator import DsrCreator

from keplar.Algorithm.Alg import Alg

# from keplar.Algorithm.dsr_Alg import uDsr_Alg

from keplar.operator.composite_operator import uDsr_CompositeOp

from keplar.operator.dsr_loop import uDsr_loop


class uDsrAlgorithm(Alg):
    def __init__(self, csv_filename, config_filename):
        # 读入数据
        self.csv_filename = csv_filename
        self.config_filename = config_filename

    def run(self):

        udsr_1=uDsr_deeplearn(self.csv_filename,self.config_filename)
        
        # population = None
        udsr_1.exec()
        # udsr_2=uDSr_loop()


        # uDsr_CompositeOp(udsr_1,udsr_2)

        # 对数据进行预处理
        # prepare_env = pre_env(self.csv_filename, self.config_filename)
        # config = prepare_env.do()

        # # 进行深度学习环境配置,并对model进行setup
        # dsr_model = uDsrDeeplearning(deepcopy(config))
        # # programs=dsr_model.do()
        # dsr_model.do()

        # # set_task(dsr_model.config_task)

        # # dsr_model.trainer.run_one_step()

        # # while not dsr_model.trainer.done:
        # # dsr_model.trainer.run_step()
        # # print(programs.task.task_type+"kkk\n")

        # dsr_train = dsr_Train(dsr_model.sess, dsr_model.policy, dsr_model.policy_optimizer, dsr_model.gp_controller, dsr_model.logger,
        #                       dsr_model.pool, **dsr_model.config_training, **dsr_model.config_task)

        # #  dsr_train = dsr_Train(sess=dsr_model.sess,policy=dsr_model.policy,policy_optimizer=dsr_model.policy_optimizer
        # #                 ,gp_controller=dsr_model.gp_controller,logger=dsr_model.logger,
        # #                 pool=dsr_model.pool,**dsr_model.config_training, **dsr_model.config_task)

        # # while not dsr_train.done:
        # #     dsr_train.run_one_step()

        # # iter_num = 0
        # while not dsr_train.done:
        #     # 每次循环采样一次，采样大小为batch_size（默认为1000）
        #     programs, r, l, actions, obs, priors = dsr_train.dsr_sample()
        #     # programs,actions,obs,priors =dsr_train.dsr_sample()

        #     # 用DsrCreator将programs转化为population
        #     dsr_creator = DsrCreator()
        #     # print(programs[0].str)
        #     # programs[0].__repr__()

        #     population = dsr_creator.do(programs=programs)
        #     # print(expr[0])

        #     # 转化的population内部的equation为uDSR类型，需要转化为统一类型
        #     popChange = POPTransPOP()
        #     population = popChange.do(population)

        #     # 将population转化为programs，并将格式转回uDSR类型
        #     proG = DSRTransPOP(population)
        #     programs2 = proG.do(programs)

        #     # 用programs2进行训练
        #     programs2 = programs
        #     programs, actions, obs, priors = dsr_train.loop_one_step(
        #         programs2, actions, obs, priors)

            # programs,actions,obs,priors=dsr_train.loop_one_step(programs,actions,obs,priors)
            # dsr_train.done = True

            # dsr_train.loop_one_step(programs,actions,obs,priors)

            # iter_num += 1

        # iter_num = 0
        # while iter_num < 20:
        #     programs,r,l,expr =dsr_train.dsr_sample()
        #     population = Dsr2pop(programs)
        #     iter_num += 1

        # dsr_train.T_step()

        # dsr_train.T_step()
        # actions, obs, priors = dsr_train.policy.sample(dsr_train.batch_size)
        # programs = [from_tokens(a) for a in actions]

        # if dsr_model.sess is None:
        #     dsr_model.do()

        # dsr_train.get_program()
        # dsr_train.programs[0].__repr__()

        # print(Program.task.task_type)
        # dsr_train.re_P()
        # print(Program.task.task_type)

        # dsr_train.run_step()

        # while not dsr_train.done:
        #     result = dsr_train.one_step()

        # model  =  DeepSymbolicOptimizer(deepcopy(config))

        # programs = model.setup()

        # programs
        # uDsr_step =
        # model.programs = [Program.random() for _ in range(100)]

        # warm_start = 50
        # actions, obs, priors =model.policy.sample(warm_start)
        # programs = [from_tokens(a) for a in actions]
        # r = np.array([p.r for p in programs])
        # l = np.array([len(p.traversal) for p in programs])

        # print[l[0]]

        # self.traversal = [Program.library[t] for t in tokens]

        # config['task']['data']['train']['x'] = model.data['train']['x']

        # self.mytrain=model.trainer

        # population = uDSRCreator(30)
        # population.do()

        # print("done!\n")

        # # 进行演化算法
        # dsr_Evo = uDsrEvo()

        # # 进行强化学习
        # dsr_RL =uDsrRL()

        # # 进行组合
        # dsr_2 =CompositeOp([dsr_Evo,dsr_RL])
        # # 这里的dsr_Evo和dsr_RL都是CompositeOp类型的，可以演化多次再做一次强化学习（当前是一次演化一次强化学习）

        # # 进行组合
        # uDsr = uDsrEvoCompositOp([dsr_2],iter_num=20)
        # uDsr.do()
