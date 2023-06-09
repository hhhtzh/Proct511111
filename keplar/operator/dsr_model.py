
import numpy as np
import os
from dsr.dso.train import Trainer
import time
import tensorflow as tf
# from dso.program import Program, from_tokens
from dsr.dso.program import Program, from_tokens

from dsr.dso.utils import empirical_entropy, get_duration, weighted_quantile, pad_action_obs_priors
from dsr.dso.memory import Batch, make_queue
from dsr.dso.variance import quantile_variance
from dsr.dso.task import set_task,make_task
from itertools import compress
from textwrap import indent

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Set TensorFlow seed
tf.compat.v1.set_random_seed(0)

from keplar.operator.operator import Operator
from keplar.operator.dsr_train import dsr_Train

class uDSR_Model(Operator):

    def __init__(self,sess, policy, policy_optimizer, gp_controller, logger,
                        pool, **config_training):
        super().__init__()

        self.sess =sess
        self.policy = policy
        self.policy_optimizer =policy_optimizer
        self.gp_controller= gp_controller
        self.logger =logger
        self.pool =pool 
        self.config_training =config_training
        # self.config_task =config_task
        self.dsr_train =None

    def pre_do(self):
        self.dsr_train = dsr_Train(self.sess, self.policy, self.policy_optimizer, self.gp_controller, self.logger,
                        self.pool, **self.config_training)
        
        return self.dsr_train
    
    def do(self, population=None):

        self.dsr_train.loop_one_step()
    
    def exec(self, population=None):
        return super().exec(population)

    
class uDSR_Sample(Operator):
    def __init__(self,model):
        super().__init__()
        self.model =model
        self.T =None

    def do(self):
        programs, r, l, actions, obs, priors = self.model.dsr_sample()
        self.T = np.array([p.traversal for p in programs],dtype=object)
        
        return self.T,programs, actions, obs, priors
        
