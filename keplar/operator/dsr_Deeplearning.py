import dsr.dso.core as core
import tensorflow as tf
from dsr.dso.core import Program
from dsr.dso.core import DeepSymbolicOptimizer

from dsr.dso.config import load_config 
from dsr.dso.train import Trainer
from dsr.dso.checkpoint import Checkpoint
# from dsr.dso.train_stats import StatsLogger
from dsr.dso.prior import make_prior
# from dsr.dso.program import Program
# from dsr.dso.config import load_config
from dsr.dso.tf_state_manager import make_state_manager

from dsr.dso.policy.policy import make_policy
from dsr.dso.policy_optimizer import make_policy_optimizer
# from dsr.dso.core import 
from collections import defaultdict



class uDsrDeeplearning:
    def __init__(self,config=None):
        # self.config = self.set_config(DeepSymbolicOptimizer,config)
        self.sess  = None
        self.prior= None
        self.state_manager = None
        self.policy = None
        self.policy_optimizer = None
        self.trainer = None
        self.set_config(config)
        
    def set_config(self, config):
        config = load_config(config)

        self.config = defaultdict(dict, config)
        self.config_task = self.config["task"]
        self.config_prior = self.config["prior"]
        self.config_logger = self.config["logging"]
        self.config_training = self.config["training"]
        self.config_state_manager = self.config["state_manager"]
        self.config_policy = self.config["policy"]
        self.config_policy_optimizer = self.config["policy_optimizer"]
        self.config_gp_meld = self.config["gp_meld"]
        self.config_experiment = self.config["experiment"]
        self.config_checkpoint = self.config["checkpoint"]

    def do(self):

        Program.clear_cache()
        tf.compat.v1.reset_default_graph()

        self.pool = DeepSymbolicOptimizer().make_pool_and_set_task()
        # Generate objects needed for training and set seeds
        # DeepSymbolicOptimizer.pool = DeepSymbolicOptimizer.make_pool_and_set_task()
        DeepSymbolicOptimizer().set_seeds() # Must be called _after_ resetting graph and _after_ setting task

        # Limit TF to single thread to prevent "resource not available" errors in parallelized runs
        session_config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                                        inter_op_parallelism_threads=1)
        
        self.sess = tf.compat.v1.Session(config=session_config)

        # Setup logdirs and output files
        self.output_file = DeepSymbolicOptimizer().make_output_file()
        # DeepSymbolicOptimizer.save_config(DeepSymbolicOptimizer)

        # Prepare training parameters
        self.prior = make_prior
        self.state_manager = make_state_manager
        self.policy = make_policy
        self.policy_optimizer = make_policy_optimizer
        # self.trainer = 
        # self.checkpoint = Checkpoint()

        print("model already setup!\n")

        # return self.trainer
        

        
        


    






        