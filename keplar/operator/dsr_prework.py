import pandas as pd
# import tensorflow as tf
from copy import deepcopy
from datetime import datetime
from dsr.dso.config import load_config 
# import click
import multiprocessing


from dsr.dso.run import print_summary
import sys


class pre_env:
    def __init__(self, csv_filename, config_filename):
        super().__init__()
        self.data = csv_filename
        self.config = config_filename

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

        # Update save dir if provided
        if exp_name is not None:
            config["experiment"]["exp_name"] = exp_name

        # Overwrite config seed, if specified
        if seed is not None:
            if config["experiment"]["seed"] is not None:
                messages.append(
                    "INFO: Replacing config seed {} with command-line seed {}.".format(
                        config["experiment"]["seed"], seed))
            config["experiment"]["seed"] = seed

        # Save starting seed and run command
        config["experiment"]["starting_seed"] = config["experiment"]["seed"]
        config["experiment"]["cmd"] = " ".join(sys.argv)
          # Set timestamp once to be used by all workers
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        config["experiment"]["timestamp"] = timestamp

        # Fix incompatible configurations
        if n_cores_task == -1:
            n_cores_task = multiprocessing.cpu_count()
        if n_cores_task > runs:
            messages.append(
                    "INFO: Setting 'n_cores_task' to {} because there are only {} runs.".format(
                        runs, runs))
            n_cores_task = runs
        if config["training"]["verbose"] and n_cores_task > 1:
            messages.append(
                    "INFO: Setting 'verbose' to False for parallelized run.")
            config["training"]["verbose"] = False
        if config["training"]["n_cores_batch"] != 1 and n_cores_task > 1:
            messages.append(
                    "INFO: Setting 'n_cores_batch' to 1 to avoid nested child processes.")
            config["training"]["n_cores_batch"] = 1
        if config["gp_meld"]["run_gp_meld"] and n_cores_task > 1 and runs > 1:
            messages.append(
                    "INFO: Setting 'parallel_eval' to 'False' as we are already parallelizing.")
            config["gp_meld"]["parallel_eval"] = False



        print_summary(config, runs, messages)

        # Generate configs (with incremented seeds) for each run
        configs = [deepcopy(config) for _ in range(runs)]
        for i, config in enumerate(configs):
            config["experiment"]["seed"] += i


        print("pre_env done!")





        