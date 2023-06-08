import pandas as pd
# import tensorflow as tf
from copy import deepcopy
from datetime import datetime
from dsr.dso.config import load_config 
# import click
import multiprocessing


from dsr.dso.run import print_summary
from keplar.operator.operator import Operator
import sys


class pre_env(Operator):
    def __init__(self,csv_filename, config_filename):
        super().__init__()
        self.type =None
        self.config =None
        self.data = csv_filename
        self.config = config_filename

    def init_config(self, config):
        self.config = config


    def do(self):
        messages = []
        runs =1
        n_cores_task = 1
        seed = None
        exp_name = None

        

        # Load the experiment config
        config = load_config(self.config)

        # Overwrite named benchmark (for tasks that support them)
        task_type = config["task"]["task_type"]
        if task_type == "regression":
            config["task"]["dataset"] = self.data


        # Save starting seed and run command
        config["experiment"]["starting_seed"] = config["experiment"]["seed"]
        # config["experiment"]["cmd"] = " ".join(sys.argv)
          # Set timestamp once to be used by all workers
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        config["experiment"]["timestamp"] = timestamp



        print_summary(config, runs, messages)

        # Generate configs (with incremented seeds) for each run
        configs = [deepcopy(config) for _ in range(runs)]
        for i, config in enumerate(configs):
            config["experiment"]["seed"] += i


        print("pre_env done!")




        