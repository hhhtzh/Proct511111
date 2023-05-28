# import dsr.dso.core as core
import tensorflow as tf
from dsr.dso.core import Program
# from dsr.dso.core import DeepSymbolicOptimizer
import os
from datetime import datetime

from dsr.dso.config import load_config 
from multiprocessing import Pool, cpu_count
import random
import zlib
from time import time

import numpy as np
from dsr.dso.train import Trainer
# from dsr.dso.checkpoint import Checkpoint
# from dsr.dso.train_stats import StatsLogger
from dsr.dso.prior import make_prior
# from dsr.dso.program import Program
# from dsr.dso.config import load_config
from dsr.dso.tf_state_manager import make_state_manager
from dsr.dso.task import set_task
from dsr.dso.train_stats import StatsLogger


from dsr.dso.policy.policy import make_policy
from dsr.dso.policy_optimizer import make_policy_optimizer
# from dsr.dso.core import 
from collections import defaultdict
from dsr.dso.gp.gp_controller import GPController

from keplar.operator.dsr_train import dsr_Train



class uDsrDeeplearning():
    def __init__(self,config=None):
        # self.config = self.set_config(DeepSymbolicOptimizer,config)
        # self.sess  = None
        # self.prior= None
        # self.state_manager = None
        # self.policy = None
        # self.policy_optimizer = None
        # self.trainer = None
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

        self.pool = self.make_pool_and_set_task()
        # Generate objects needed for training and set seeds
        # DeepSymbolicOptimizer.pool = DeepSymbolicOptimizer.make_pool_and_set_task()
        self.set_seeds() # Must be called _after_ resetting graph and _after_ setting task

        # Limit TF to single thread to prevent "resource not available" errors in parallelized runs
        session_config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                                        inter_op_parallelism_threads=1)
        
        self.sess = tf.compat.v1.Session(config=session_config)

        # Setup logdirs and output files
        self.output_file = self.make_output_file()
        # DeepSymbolicOptimizer.save_config(DeepSymbolicOptimizer)

        # Prepare training parameters
        # self.prior = self.make_prior()
        self.prior=make_prior(Program.library, self.config_prior)
        self.state_manager = make_state_manager(self.config_state_manager)
        self.policy = make_policy(self.sess,self.prior,self.state_manager,**self.config_policy)
        self.policy_optimizer = make_policy_optimizer(self.sess, self.policy,**self.config_policy_optimizer)
        # self.gp_controller = self.make_gp_controller()
        if self.config_gp_meld.pop("run_gp_meld", False):
            self.gp_controller = GPController(self.prior,
                                         self.config_prior,
                                         **self.config_gp_meld)
        else:
            self.gp_controller = None
        # self.logger = self.make_logger()
        self.logger=StatsLogger(self.sess,self.output_file,**self.config_logger)

        # self.trainer = dsr_Train(self.sess, self.policy, self.policy_optimizer, self.gp_controller, self.logger,self.pool, **self.config_training)
        # self.trainer = 

        # self.trainer = 
        # self.checkpoint = Checkpoint()

        print("model already setup!\n")
        # print(Program.task.task_type+' 1hhh\n')

        # return Program
        
        # print(Program.task.task_type+' hhh\n')

        # return self.trainer


    # def train_one_step(self, override=None):
    #     """
    #     Train one iteration.
    #     """

    #     # Setup the model
    #     # if self.sess is None:
    #     #     self.setup()

    #     # Run one step
    #     # assert not self.trainer.done, "Training has already completed!"
    #     self.trainer.run_one_step(override)
        
    #     # # Maybe save next checkpoint
    #     # self.checkpoint.update()

    #     # # If complete, return summary
    #     # if self.trainer.done:
    #     #     return self.finish()
        
    # def make_logger(self):
    #     logger = StatsLogger(self.sess,
    #                          self.output_file,
    #                          **self.config_logger)
    #     return logger
        # return self.trainer
    def set_seeds(self):
        """
        Set the tensorflow, numpy, and random module seeds based on the seed
        specified in config. If there is no seed or it is None, a time-based
        seed is used instead and is written to config.
        """

        seed = self.config_experiment.get("seed")

        # Default uses current time in milliseconds, modulo 1e9
        if seed is None:
            seed = round(time() * 1000) % int(1e9)
            self.config_experiment["seed"] = seed

        # Shift the seed based on task name
        # This ensures a specified seed doesn't have similarities across different task names
        task_name = Program.task.name
        shifted_seed = seed + zlib.adler32(task_name.encode("utf-8"))

        # Set the seeds using the shifted seed
        tf.compat.v1.set_random_seed(shifted_seed)
        np.random.seed(shifted_seed)
        random.seed(shifted_seed)

    # def make_prior(self):
    #     prior = make_prior(Program.library, self.config_prior)
    #     return prior

    # def make_state_manager(self):
    #     state_manager = make_state_manager(self.config_state_manager)
    #     return state_manager
    
    # def make_policy_optimizer(self):
    #     policy_optimizer = make_policy_optimizer(self.sess,
    #                                              self.policy,
    #                                              **self.config_policy_optimizer)
    #     return policy_optimizer

    # def make_policy(self):
    #     policy = make_policy(self.sess,
    #                          self.prior,
    #                          self.state_manager,
    #                          **self.config_policy)
    #     return policy

    # def make_gp_controller(self):
    #     if self.config_gp_meld.pop("run_gp_meld", False):
    #         gp_controller = GPController(self.prior,
    #                                      self.config_prior,
    #                                      **self.config_gp_meld)
    #     else:
    #         gp_controller = None
    #     return gp_controller
    
    def make_pool_and_set_task(self):
        # Create the pool and set the Task for each worker

        # Set complexity and const optimizer here so pool can access them
        # Set the complexity function
        complexity = self.config_training["complexity"]
        Program.set_complexity(complexity)

        # Set the constant optimizer
        const_optimizer = self.config_training["const_optimizer"]
        const_params = self.config_training["const_params"]
        const_params = const_params if const_params is not None else {}
        Program.set_const_optimizer(const_optimizer, **const_params)

        pool = None
        n_cores_batch = self.config_training.get("n_cores_batch")
        if n_cores_batch is not None:
            if n_cores_batch == -1:
                n_cores_batch = cpu_count()
            if n_cores_batch > 1:
                pool = Pool(n_cores_batch,
                            initializer=set_task,
                            initargs=(self.config_task,))

        # Set the Task for the parent process
        set_task(self.config_task)

        return pool
    

    def make_output_file(self):
        """Generates an output filename"""

        # If logdir is not provided (e.g. for pytest), results are not saved
        if self.config_experiment.get("logdir") is None:
            self.save_path = None
            print("WARNING: logdir not provided. Results will not be saved to file.")
            return None

        # When using run.py, timestamp is already generated
        timestamp = self.config_experiment.get("timestamp")
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
            self.config_experiment["timestamp"] = timestamp

        # Generate save path
        task_name = Program.task.name
        if self.config_experiment["exp_name"] is None:
            save_path = os.path.join(
                self.config_experiment["logdir"],
                '_'.join([task_name, timestamp]))
        else:
            save_path = os.path.join(
                self.config_experiment["logdir"],
                self.config_experiment["exp_name"])

        self.config_experiment["task_name"] = task_name
        self.config_experiment["save_path"] = save_path
        os.makedirs(save_path, exist_ok=True)

        seed = self.config_experiment["seed"]
        output_file = os.path.join(save_path,
                                   "dso_{}_{}.csv".format(task_name, seed))

        self.save_path = save_path

        return output_file


        
        


    






        