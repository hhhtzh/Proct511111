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
from dsr.dso.task import set_task, make_task
from itertools import compress
from textwrap import indent

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Set TensorFlow seed
tf.compat.v1.set_random_seed(0)


class dsr_Train(Trainer):
    def __init__(self, sess, policy, policy_optimizer, gp_controller, logger,
                 pool, n_samples=200000, batch_size=1000, alpha=0.5,
                 epsilon=0.05, verbose=True, baseline="R_e",
                 b_jumpstart=False, early_stopping=True, debug=0,
                 use_memory=False, memory_capacity=1e3, warm_start=None, memory_threshold=None,
                 complexity="token", const_optimizer="scipy", const_params=None, n_cores_batch=1, **config_task):
        Trainer.__init__(self, sess, policy, policy_optimizer, gp_controller, logger,
                         pool, n_samples=200000, batch_size=1000, alpha=0.5,
                         epsilon=0.05, verbose=True, baseline="R_e",
                         b_jumpstart=False, early_stopping=True, debug=0,
                         use_memory=False, memory_capacity=1e3, warm_start=None, memory_threshold=None,
                         complexity="token", const_optimizer="scipy", const_params=None, n_cores_batch=1)
        # # self.programs = self.get_program()
        # Trainer.batch_size = batch_size
        # Trainer.policy = policy

        # self.sess = sess

        # self.config_task1 = config_task
        # # Programs=Program

        # # Initialize compute draw
        # self.sess.run(tf.compat.v1.global_variables_initializer())

        # self.policy = policy
        # self.policy_optimizer = policy_optimizer
        # self.gp_controller = gp_controller
        # self.logger = logger
        # self.pool = pool
        # self.n_samples = n_samples
        # self.batch_size = batch_size
        # self.alpha = alpha
        # self.epsilon = epsilon
        # self.verbose = verbose
        # self.baseline = baseline
        # self.b_jumpstart = b_jumpstart
        # self.early_stopping = early_stopping
        # self.debug = debug
        # self.use_memory = use_memory
        # self.memory_threshold = memory_threshold

        # # set_task(self.config_task1)
        protected = config_task["protected"] if "protected" in config_task else False

        Program.set_execute(protected)
        task = make_task(**config_task)
        Program.set_task(task)

        # # print(Program.task.task_type+"xxx\n")

        # # print("??????????????????????!!!!!")

        # if self.debug:
        #     tvars = tf.trainable_variables()
        #     def print_var_means():
        #         tvars_vals = self.sess.run(tvars)
        #         for var, val in zip(tvars, tvars_vals):
        #             print(var.name, "mean:", val.mean(), "var:", val.var())
        #     self.print_var_means = print_var_means

        # # Create the priority_queue if needed
        # if hasattr(self.policy_optimizer, 'pqt_k'):
        #     from dsr.dso.policy_optimizer.pqt_policy_optimizer import PQTPolicyOptimizer
        #     assert type(self.policy_optimizer) == PQTPolicyOptimizer
        #     # Create the priority queue
        #     k = self.policy_optimizer.pqt_k
        #     if k is not None and k > 0:
        #         self.priority_queue = make_queue(priority=True, capacity=k)
        # else:
        #     self.priority_queue = None

        # # Create the memory queue
        # if self.use_memory:
        #     # print("Using memory queue.1111111111111\n")
        #     assert self.epsilon is not None and self.epsilon < 1.0, \
        #         "Memory queue is only used with risk-seeking."
        #     self.memory_queue = make_queue(policy=self.policy, priority=False,
        #                                    capacity=int(memory_capacity))

        #     # Warm start the queue
        #     # TBD: Parallelize. Abstract sampling a Batch
        #     warm_start = warm_start if warm_start is not None else self.batch_size
        #     actions, obs, priors = policy.sample(warm_start)
        #     programs = [from_tokens(a) for a in actions]
        #     r = np.array([p.r for p in programs])
        #     l = np.array([len(p.traversal) for p in programs])
        #     on_policy = np.array([p.originally_on_policy for p in programs])
        #     sampled_batch = Batch(actions=actions, obs=obs, priors=priors,
        #                           lengths=l, rewards=r, on_policy=on_policy)
        #     self.memory_queue.push_batch(sampled_batch, programs)
        # else:
        #     self.memory_queue = None

        # self.nevals = 0 # Total number of sampled expressions (from RL or GP)
        # self.iteration = 0 # Iteration counter
        # self.r_best = -np.inf
        # self.p_r_best = None
        # self.done = False

    def one_iter(self, programs=None, actions=None, obs=None, priors=None):
        # print("node 0.1")
        # def run_one_step(self, override=None):
        """
        Executes one step of main training loop. If override is given,
        train on that batch. Otherwise, sample the batch to train on.

        Parameters
        ----------
        override : tuple or None
            Tuple of (actions, obs, priors, programs) to train on offline
            samples instead of sampled
        """

        override = None

        positional_entropy = None
        top_samples_per_batch = list()
        if self.debug >= 1:
            print("\nDEBUG: Policy parameter means:")
            self.print_var_means()
        ewma = None if self.b_jumpstart else 0.0  # EWMA portion of baseline
        start_time = time.time()
        if self.verbose:
            print("-- RUNNING ITERATIONS START -------------")
        # Number of extra samples generated during attempt to get
        # batch_size new samples
        n_extra = 0
        # Record previous cache before new samples are added by from_tokens
        s_history = list(Program.cache.keys())

        # Construct the actions, obs, priors, and programs
        # Shape of actions: (batch_size, max_length)
        # Shape of obs: (batch_size, obs_dim, max_length)
        # Shape of priors: (batch_size, max_length, n_choices)
        if override is None:
            # length = len(T)
            pass

            # pass
            # Sample batch of Programs from the Controller
            # actions, obs, priors = self.policy.sample(self.batch_size)
            # print("node 0.1")
            # programs = [from_tokens(a) for a in actions] 
            # print("node 0")

        else:
            # Train on the given batch of Programs
            actions, obs, priors, programs = override
            for p in programs:
                Program.cache[p.str] = p
        # Extra samples, previously already contained in cache,
        # that were geneated during the attempt to get
        # batch_size new samples for expensive reward evaluation
        if self.policy.valid_extended_batch:
            # print("node 1.1")
            self.policy.valid_extended_batch = False
            n_extra = self.policy.extended_batch[0]
            if n_extra > 0:
                # print("node 1")
                extra_programs = [from_tokens(a) for a in
                                  self.policy.extended_batch[1]]
                # Concatenation is fine because rnn_policy.sample_novel()
                # already made sure that offline batch and extended batch
                # are padded to the same trajectory length
                actions = np.concatenate([actions, self.policy.extended_batch[1]])
                obs = np.concatenate([obs, self.policy.extended_batch[2]])
                priors = np.concatenate([priors, self.policy.extended_batch[3]])
                programs = programs + extra_programs

        self.nevals += self.batch_size + n_extra

        # Run GP seeded with the current batch, returning elite samples
        if self.gp_controller is not None:
            # print("node 1.5")
            deap_programs, deap_actions, deap_obs, deap_priors = self.gp_controller(actions)
            self.nevals += self.gp_controller.nevals

            # Pad AOP if different sized
            if actions.shape[1] < deap_actions.shape[1]:
                # If RL shape is smaller than GP then pad
                pad_length = deap_actions.shape[1] - actions.shape[1]
                actions, obs, priors = pad_action_obs_priors(actions, obs, priors, pad_length)
            elif actions.shape[1] > deap_actions.shape[1]:
                # If GP shape is smaller than RL then pad
                pad_length = actions.shape[1] - deap_actions.shape[1]
                deap_actions, deap_obs, deap_priors = pad_action_obs_priors(deap_actions, deap_obs, deap_priors,
                                                                            pad_length)

            # print("node 2")
            # Combine RNN and deap programs, actions, obs, and priors
            programs = programs + deap_programs
            actions = np.append(actions, deap_actions, axis=0)
            obs = np.append(obs, deap_obs, axis=0)
            priors = np.append(priors, deap_priors, axis=0)

        # Compute rewards in parallel
        if self.pool is not None:
            # print("node 3.1")
            # Filter programs that need reward computing
            programs_to_optimize = list(set([p for p in programs if "r" not in p.__dict__]))
            pool_p_dict = {p.str: p for p in self.pool.map(work, programs_to_optimize)}
            programs = [pool_p_dict[p.str] if "r" not in p.__dict__ else p for p in programs]
            # print("node 3")

            # Make sure to update cache with new programs
            Program.cache.update(pool_p_dict)

        # Compute rewards (or retrieve cached rewards)
        r = np.array([p.r for p in programs])

        # Back up programs to save them properly later
        controller_programs = programs.copy() if self.logger.save_token_count else None

        # Need for Vanilla Policy Gradient (epsilon = null)
        l = np.array([len(p.traversal) for p in programs])
        s = [p.str for p in programs]  # Str representations of Programs
        on_policy = np.array([p.originally_on_policy for p in programs])
        invalid = np.array([p.invalid for p in programs], dtype=bool)

        if self.logger.save_positional_entropy:
            # print("node 4")
            positional_entropy = np.apply_along_axis(empirical_entropy, 0, actions)

        if self.logger.save_top_samples_per_batch > 0:
            # print("node 4.1")
            # sort in descending order: larger rewards -> better solutions
            sorted_idx = np.argsort(r)[::-1]
            top_perc = int(len(programs) * float(self.logger.save_top_samples_per_batch))
            for idx in sorted_idx[:top_perc]:
                top_samples_per_batch.append([self.iteration, r[idx], repr(programs[idx])])

        # Store in variables the values for the whole batch (those variables will be modified below)
        r_full = r
        l_full = l
        s_full = s
        actions_full = actions
        invalid_full = invalid
        r_max = np.max(r)

        """
        Apply risk-seeking policy gradient: compute the empirical quantile of
        rewards and filter out programs with lesser reward.
        """
        if self.epsilon is not None and self.epsilon < 1.0:
            # print("node 4.2")
            # Compute reward quantile estimate
            if self.use_memory:  # Memory-augmented quantile
                # print("node 4.3")
                # Get subset of Programs not in buffer
                unique_programs = [p for p in programs \
                                   if p.str not in self.memory_queue.unique_items]
                N = len(unique_programs)

                # Get rewards
                memory_r = self.memory_queue.get_rewards()
                # print("node 5")

                sample_r = [p.r for p in unique_programs]
                combined_r = np.concatenate([memory_r, sample_r])

                # Compute quantile weights
                memory_w = self.memory_queue.compute_probs()
                if N == 0:
                    print("WARNING: Found no unique samples in batch!")
                    combined_w = memory_w / memory_w.sum()  # Renormalize
                else:
                    sample_w = np.repeat((1 - memory_w.sum()) / N, N)
                    combined_w = np.concatenate([memory_w, sample_w])

                # Quantile variance/bias estimates
                if self.memory_threshold is not None:
                    print("Memory weight:", memory_w.sum())
                    if memory_w.sum() > self.memory_threshold:
                        quantile_variance(self.memory_queue, self.policy, self.batch_size, self.epsilon, self.iteration)

                # Compute the weighted quantile
                quantile = weighted_quantile(values=combined_r, weights=combined_w, q=1 - self.epsilon)

            else:  # Empirical quantile
                quantile = np.quantile(r, 1 - self.epsilon, interpolation="higher")

            # Filter quantities whose reward >= quantile
            keep = r >= quantile
            l = l[keep]
            s = list(compress(s, keep))
            invalid = invalid[keep]
            r = r[keep]
            programs = list(compress(programs, keep))
            actions = actions[keep, :]
            obs = obs[keep, :, :]
            priors = priors[keep, :, :]
            on_policy = on_policy[keep]

        # Clip bounds of rewards to prevent NaNs in gradient descent
        r = np.clip(r, -1e6, 1e6)

        # Compute baseline
        # NOTE: pg_loss = tf.reduce_mean((self.r - self.baseline) * neglogp, name="pg_loss")
        if self.baseline == "ewma_R":
            ewma = np.mean(r) if ewma is None else self.alpha * np.mean(r) + (1 - self.alpha) * ewma
            b = ewma
        elif self.baseline == "R_e":  # Default
            ewma = -1
            b = quantile
        elif self.baseline == "ewma_R_e":
            ewma = np.min(r) if ewma is None else self.alpha * quantile + (1 - self.alpha) * ewma
            b = ewma
        elif self.baseline == "combined":
            ewma = np.mean(r) - quantile if ewma is None else self.alpha * (np.mean(r) - quantile) + (
                        1 - self.alpha) * ewma
            b = quantile + ewma

        # Compute sequence lengths
        lengths = np.array([min(len(p.traversal), self.policy.max_length)
                            for p in programs], dtype=np.int32)

        # Create the Batch
        sampled_batch = Batch(actions=actions, obs=obs, priors=priors,
                              lengths=lengths, rewards=r, on_policy=on_policy)

        # Update and sample from the priority queue
        if self.priority_queue is not None:
            # print("node 6")
            self.priority_queue.push_best(sampled_batch, programs)
            pqt_batch = self.priority_queue.sample_batch(self.policy_optimizer.pqt_batch_size)
            # Train the policy
            summaries = self.policy_optimizer.train_step(b, sampled_batch, pqt_batch)
        else:
            # print("node 7")
            pqt_batch = None
            # Train the policy
            summaries = self.policy_optimizer.train_step(b, sampled_batch)

        # Walltime calculation for the iteration
        iteration_walltime = time.time() - start_time

        # Update the memory queue
        if self.memory_queue is not None:
            self.memory_queue.push_batch(sampled_batch, programs)

        # Update new best expression
        if r_max > self.r_best:
            self.r_best = r_max
            self.p_r_best = programs[np.argmax(r)]

            # self.p_r_best.

            # Print new best expression
            if self.verbose or self.debug:
                print("[{}] Training iteration {}, current best R: {:.4f}".format(get_duration(start_time),
                                                                                  self.iteration + 1, self.r_best))
                print("\n\t** New best")
                self.p_r_best.print_stats()

        # Collect sub-batch statistics and write output
        # self.logger.save_stats(r_full, l_full, actions_full, s_full,
        #                        invalid_full, r, l, actions, s, s_history,
        #                        invalid, self.r_best, r_max, ewma, summaries,
        #                        self.iteration, b, iteration_walltime,
        #                        self.nevals, controller_programs,
        #                        positional_entropy, top_samples_per_batch)

        # Stop if early stopping criteria is met
        if self.early_stopping and self.p_r_best.evaluate.get("success"):
            print("[{}] Early stopping criteria met; breaking early.".format(get_duration(start_time)))
            self.done = True

        if self.verbose and (self.iteration + 1) % 10 == 0:
            print("[{}] Training iteration {}, current best R: {:.4f}".format(get_duration(start_time),
                                                                              self.iteration + 1, self.r_best))

        if self.debug >= 2:
            print("\nParameter means after iteration {}:".format(self.iteration + 1))
            self.print_var_means()

        if self.nevals >= self.n_samples:
            self.done = True

        # if self.iteration==self.n_samples/self.batch_size-1:
        print("[{}] Training iteration {}, current best R: {:.4f}".format(get_duration(start_time), self.iteration + 1,
                                                                          self.r_best))
        print("\n\t** New best")
        self.p_r_best.print_stats()

        # Increment the iteration counter
        self.iteration += 1

        return programs, actions, obs, priors

        # print("??????????????????????")

        # self.programs = self.get_program()

    def run_step(self, override=None):

        # if override is None:
        #     # Sample batch of Programs from the Controller
        #     print("xxxxxxxxxxxxx\n")
        #     # Program.task.task_type='regression'
        #     # print(self.programs.task.task_type+"\n")
        #     print(Program.task.task_type+"xxx\n")

        #     actions, obs, priors = self.policy.sample(self.batch_size)
        #     programs = [from_tokens(a) for a in actions]     
        self.two_step()

    def dsr_sample(self, override=None):
        if override is None:
            # Sample batch of Programs from the Controller
            actions, obs, priors = self.policy.sample(self.batch_size)
            programs = [from_tokens(a) for a in actions]
            r = np.array([p.r for p in programs])
            l = np.array([len(p.traversal) for p in programs])
            # expr = np.array([repr(p.sympy_expr) for p in programs])
            # expr ={}

            # for i in range(len(r)):
            #     # print(programs[i].pretty())
            #     print("{}\n".format(indent(programs[i].pretty(), '\t  ')))

            # print(str(expr[i]))/

        return programs, r, l, actions, obs, priors

    # def loop_one_step(self, override=None):
    #     pass

    def T_step(self, override=None):

        if override is None:
            # Sample batch of Programs from the Controller
            print("xxxxxxxxxxxxx\n")
            # Program.task.task_type='regression'
            # print(self.programs.task.task_type+"\n")
            print(Program.task.task_type + "xxx\n")

            actions, obs, priors = self.policy.sample(self.batch_size)
            programs = [from_tokens(a) for a in actions]

            r = np.array([p.r for p in programs])
            l = np.array([len(p.traversal) for p in programs])
            R = np.array([p.sympy_expr for p in programs])

            # print(Program.library)
            # print(programs.library)
            # print(l)
            T = np.array([p.traversal for p in programs])
            print(T)
            # print(T[0])
            # print(l[0])
            # print(R[0])

            # print(len(T))
            # print(len(l))
            # print(len(R))

            # print(p.traversal)

            # r.__repr__()
            # print(r)
            # print(R)

            # programs.__str__()
        # self.two_step()

    def loop_one_step(self, programs=None, actions=None, obs=None, priors=None):
        # print("node 0.1")
        # def run_one_step(self, override=None):
        """
        Executes one step of main training loop. If override is given,
        train on that batch. Otherwise, sample the batch to train on.

        Parameters
        ----------
        override : tuple or None
            Tuple of (actions, obs, priors, programs) to train on offline
            samples instead of sampled
        """

        override = None

        positional_entropy = None
        top_samples_per_batch = list()
        if self.debug >= 1:
            print("\nDEBUG: Policy parameter means:")
            self.print_var_means()
        ewma = None if self.b_jumpstart else 0.0  # EWMA portion of baseline
        start_time = time.time()
        if self.verbose:
            print("-- RUNNING ITERATIONS START -------------")
        # Number of extra samples generated during attempt to get
        # batch_size new samples
        n_extra = 0
        # Record previous cache before new samples are added by from_tokens
        s_history = list(Program.cache.keys())

        # Construct the actions, obs, priors, and programs
        # Shape of actions: (batch_size, max_length)
        # Shape of obs: (batch_size, obs_dim, max_length)
        # Shape of priors: (batch_size, max_length, n_choices)
        if override is None:
            pass
            # Sample batch of Programs from the Controller
            # actions, obs, priors = self.policy.sample(self.batch_size)
            # print("node 0.1")
            # programs = [from_tokens(a) for a in actions]    
            # print("node 0")

        else:
            # Train on the given batch of Programs
            actions, obs, priors, programs = override
            for p in programs:
                Program.cache[p.str] = p
        # Extra samples, previously already contained in cache,
        # that were geneated during the attempt to get
        # batch_size new samples for expensive reward evaluation
        if self.policy.valid_extended_batch:
            # print("node 1.1")
            self.policy.valid_extended_batch = False
            n_extra = self.policy.extended_batch[0]
            if n_extra > 0:
                # print("node 1")
                extra_programs = [from_tokens(a) for a in
                                  self.policy.extended_batch[1]]
                # Concatenation is fine because rnn_policy.sample_novel()
                # already made sure that offline batch and extended batch
                # are padded to the same trajectory length
                actions = np.concatenate([actions, self.policy.extended_batch[1]])
                obs = np.concatenate([obs, self.policy.extended_batch[2]])
                priors = np.concatenate([priors, self.policy.extended_batch[3]])
                programs = programs + extra_programs

        self.nevals += self.batch_size + n_extra

        # Run GP seeded with the current batch, returning elite samples
        if self.gp_controller is not None:
            # print("node 1.5")
            deap_programs, deap_actions, deap_obs, deap_priors = self.gp_controller(actions)
            self.nevals += self.gp_controller.nevals

            # Pad AOP if different sized
            if actions.shape[1] < deap_actions.shape[1]:
                # If RL shape is smaller than GP then pad
                pad_length = deap_actions.shape[1] - actions.shape[1]
                actions, obs, priors = pad_action_obs_priors(actions, obs, priors, pad_length)
            elif actions.shape[1] > deap_actions.shape[1]:
                # If GP shape is smaller than RL then pad
                pad_length = actions.shape[1] - deap_actions.shape[1]
                deap_actions, deap_obs, deap_priors = pad_action_obs_priors(deap_actions, deap_obs, deap_priors,
                                                                            pad_length)

            # print("node 2")
            # Combine RNN and deap programs, actions, obs, and priors
            programs = programs + deap_programs
            actions = np.append(actions, deap_actions, axis=0)
            obs = np.append(obs, deap_obs, axis=0)
            priors = np.append(priors, deap_priors, axis=0)

        # Compute rewards in parallel
        if self.pool is not None:
            # print("node 3.1")
            # Filter programs that need reward computing
            programs_to_optimize = list(set([p for p in programs if "r" not in p.__dict__]))
            pool_p_dict = {p.str: p for p in self.pool.map(work, programs_to_optimize)}
            programs = [pool_p_dict[p.str] if "r" not in p.__dict__ else p for p in programs]
            # print("node 3")

            # Make sure to update cache with new programs
            Program.cache.update(pool_p_dict)

        # Compute rewards (or retrieve cached rewards)
        r = np.array([p.r for p in programs])

        # Back up programs to save them properly later
        controller_programs = programs.copy() if self.logger.save_token_count else None

        # Need for Vanilla Policy Gradient (epsilon = null)
        l = np.array([len(p.traversal) for p in programs])
        s = [p.str for p in programs]  # Str representations of Programs
        on_policy = np.array([p.originally_on_policy for p in programs])
        invalid = np.array([p.invalid for p in programs], dtype=bool)

        if self.logger.save_positional_entropy:
            # print("node 4")
            positional_entropy = np.apply_along_axis(empirical_entropy, 0, actions)

        if self.logger.save_top_samples_per_batch > 0:
            # print("node 4.1")
            # sort in descending order: larger rewards -> better solutions
            sorted_idx = np.argsort(r)[::-1]
            top_perc = int(len(programs) * float(self.logger.save_top_samples_per_batch))
            for idx in sorted_idx[:top_perc]:
                top_samples_per_batch.append([self.iteration, r[idx], repr(programs[idx])])

        # Store in variables the values for the whole batch (those variables will be modified below)
        r_full = r
        l_full = l
        s_full = s
        actions_full = actions
        invalid_full = invalid
        r_max = np.max(r)

        """
        Apply risk-seeking policy gradient: compute the empirical quantile of
        rewards and filter out programs with lesser reward.
        """
        if self.epsilon is not None and self.epsilon < 1.0:
            # print("node 4.2")
            # Compute reward quantile estimate
            if self.use_memory:  # Memory-augmented quantile
                # print("node 4.3")
                # Get subset of Programs not in buffer
                unique_programs = [p for p in programs \
                                   if p.str not in self.memory_queue.unique_items]
                N = len(unique_programs)

                # Get rewards
                memory_r = self.memory_queue.get_rewards()
                # print("node 5")

                sample_r = [p.r for p in unique_programs]
                combined_r = np.concatenate([memory_r, sample_r])

                # Compute quantile weights
                memory_w = self.memory_queue.compute_probs()
                if N == 0:
                    print("WARNING: Found no unique samples in batch!")
                    combined_w = memory_w / memory_w.sum()  # Renormalize
                else:
                    sample_w = np.repeat((1 - memory_w.sum()) / N, N)
                    combined_w = np.concatenate([memory_w, sample_w])

                # Quantile variance/bias estimates
                if self.memory_threshold is not None:
                    print("Memory weight:", memory_w.sum())
                    if memory_w.sum() > self.memory_threshold:
                        quantile_variance(self.memory_queue, self.policy, self.batch_size, self.epsilon, self.iteration)

                # Compute the weighted quantile
                quantile = weighted_quantile(values=combined_r, weights=combined_w, q=1 - self.epsilon)

            else:  # Empirical quantile
                quantile = np.quantile(r, 1 - self.epsilon, interpolation="higher")

            # Filter quantities whose reward >= quantile
            keep = r >= quantile
            l = l[keep]
            s = list(compress(s, keep))
            invalid = invalid[keep]
            r = r[keep]
            programs = list(compress(programs, keep))
            actions = actions[keep, :]
            obs = obs[keep, :, :]
            priors = priors[keep, :, :]
            on_policy = on_policy[keep]

        # Clip bounds of rewards to prevent NaNs in gradient descent
        r = np.clip(r, -1e6, 1e6)

        # Compute baseline
        # NOTE: pg_loss = tf.reduce_mean((self.r - self.baseline) * neglogp, name="pg_loss")
        if self.baseline == "ewma_R":
            ewma = np.mean(r) if ewma is None else self.alpha * np.mean(r) + (1 - self.alpha) * ewma
            b = ewma
        elif self.baseline == "R_e":  # Default
            ewma = -1
            b = quantile
        elif self.baseline == "ewma_R_e":
            ewma = np.min(r) if ewma is None else self.alpha * quantile + (1 - self.alpha) * ewma
            b = ewma
        elif self.baseline == "combined":
            ewma = np.mean(r) - quantile if ewma is None else self.alpha * (np.mean(r) - quantile) + (
                        1 - self.alpha) * ewma
            b = quantile + ewma

        # Compute sequence lengths
        lengths = np.array([min(len(p.traversal), self.policy.max_length)
                            for p in programs], dtype=np.int32)

        # Create the Batch
        sampled_batch = Batch(actions=actions, obs=obs, priors=priors,
                              lengths=lengths, rewards=r, on_policy=on_policy)

        # Update and sample from the priority queue
        if self.priority_queue is not None:
            # print("node 6")
            self.priority_queue.push_best(sampled_batch, programs)
            pqt_batch = self.priority_queue.sample_batch(self.policy_optimizer.pqt_batch_size)
            # Train the policy
            summaries = self.policy_optimizer.train_step(b, sampled_batch, pqt_batch)
        else:
            # print("node 7")
            pqt_batch = None
            # Train the policy
            summaries = self.policy_optimizer.train_step(b, sampled_batch)

        # Walltime calculation for the iteration
        iteration_walltime = time.time() - start_time

        # Update the memory queue
        if self.memory_queue is not None:
            self.memory_queue.push_batch(sampled_batch, programs)

        # Update new best expression
        if r_max > self.r_best:
            self.r_best = r_max
            self.p_r_best = programs[np.argmax(r)]

            # self.p_r_best.

            # Print new best expression
            if self.verbose or self.debug:
                print("[{}] Training iteration {}, current best R: {:.4f}".format(get_duration(start_time),
                                                                                  self.iteration + 1, self.r_best))
                print("\n\t** New best")
                self.p_r_best.print_stats()

        # Collect sub-batch statistics and write output
        # self.logger.save_stats(r_full, l_full, actions_full, s_full,
        #                        invalid_full, r, l, actions, s, s_history,
        #                        invalid, self.r_best, r_max, ewma, summaries,
        #                        self.iteration, b, iteration_walltime,
        #                        self.nevals, controller_programs,
        #                        positional_entropy, top_samples_per_batch)

        # Stop if early stopping criteria is met
        if self.early_stopping and self.p_r_best.evaluate.get("success"):
            print("[{}] Early stopping criteria met; breaking early.".format(get_duration(start_time)))
            self.done = True

        if self.verbose and (self.iteration + 1) % 10 == 0:
            print("[{}] Training iteration {}, current best R: {:.4f}".format(get_duration(start_time),
                                                                              self.iteration + 1, self.r_best))

        if self.debug >= 2:
            print("\nParameter means after iteration {}:".format(self.iteration + 1))
            self.print_var_means()

        if self.nevals >= self.n_samples:
            self.done = True

        if self.iteration == self.n_samples / self.batch_size - 1:
            print("[{}] Training iteration {}, current best R: {:.4f}".format(get_duration(start_time),
                                                                              self.iteration + 1, self.r_best))
            print("\n\t** New best")
            self.p_r_best.print_stats()

        # Increment the iteration counter
        self.iteration += 1

        return programs, actions, obs, priors
