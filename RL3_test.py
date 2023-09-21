import random

import numpy as np
import pandas as pd
import tensorflow as tf

from keplar.Algorithm.Alg import KeplarBingoAlg, GpBingo2Alg, KeplarOperonAlg
from keplar.data.data import Data
from keplar.operator.check_pop import CheckPopulation
from keplar.operator.composite_operator import CompositeOp, CompositeOpReturn
from keplar.operator.creator import OperonCreator, BingoCreator
from keplar.operator.crossover import BingoCrossover, OperonCrossover
from keplar.operator.evaluator import OperonEvaluator, BingoEvaluator, GpEvaluator, OperonSingleEvaluator
from keplar.operator.mutation import BingoMutation, OperonMutation
from keplar.operator.reinserter import KeplarReinserter
from keplar.operator.selector import BingoSelector
import pyoperon as Operon

data = Data("pmlb", "1027_ESL", ["x0", "x1", "x2", "x3", 'y'])
data.read_file()
x = data.get_np_x()
y = data.get_np_y()


def calculate_reward(list, action, new_list):
    # 计算适应度变化
    fitness_change = new_list[0] - list[0]
    mean_fitness_change = new_list[2] - list[2]
    reward = 0
    # 根据适应度变化分配奖励
    if fitness_change < 0:
        reward += 10.0  # 适应度下降，奖励为正数
    elif fitness_change < 0:
        reward -= 20.0  # 适应度上升，奖励为负数
    else:
        reward += 0.0  # 适应度没有变化，奖励为零

    if mean_fitness_change < 0:
        reward += 1
    elif mean_fitness_change > 0:
        reward -= 2
    else:
        reward += 0

    return reward


def calculate_returns(rewards, gamma=0.99):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


# 定义策略网络

import tensorflow as tf


class PolicyNetwork(tf.keras.Model):
    def __init__(self, num_actions):
        super(PolicyNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.dense3 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(num_actions, activation='softmax')

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        x = self.dense3(x)
        # print(x)
        return self.output_layer(x)


# 策略梯度训练函数
def train_policy_network(policy_network, states, actions, rewards, optimizer):
    with tf.GradientTape() as tape:
        action_probabilities = policy_network(states)
        # print(action_probabilities)
        actions_one_hot = tf.one_hot(actions, depth=3)
        selected_action_probabilities = tf.reduce_sum(action_probabilities * actions_one_hot, axis=1)
        loss = -tf.reduce_sum(tf.math.log(selected_action_probabilities) * rewards)

    gradients = tape.gradient(loss, policy_network.trainable_variables)

    # 添加梯度裁剪，防止梯度爆炸
    gradients, _ = tf.clip_by_global_norm(gradients, 5)

    optimizer.apply_gradients(zip(gradients, policy_network.trainable_variables))


# 设置环境和参数
num_actions = 5
num_episodes = 1000
learning_rate = 0.01

policy_network = PolicyNetwork(num_actions)
optimizer = tf.keras.optimizers.Adam(learning_rate)

operators = ["+", "-", "*", "/", "sin", "exp", "cos", 'sqrt', 'log', 'sin', 'pow', 'exp', '^']
bg_creator = BingoCreator(128, operators, x, 10, "Bingo")
bg_evaluator = BingoEvaluator(x, "exp", "lm", "self", y)
bg_crossover = BingoCrossover("Bingo")
bg_mutation = BingoMutation(x, operators, "Bingo")
bg_selector = BingoSelector(0.5, "tournament", "Bingo")
op_crossover = OperonCrossover(x, y, "Operon")
select = BingoSelector(0.2, "tournament", "Operon")
op_mutation = OperonMutation(0.6, 0.7, 0.8, 0.8, x, y, 10, 50, "balanced", "Operon")
data = pd.read_csv("NAStraining_data/recursion_training2.csv")
op_creator = OperonCreator("balanced", x, y, 128, "Operon")
op_evaluator = OperonEvaluator("RMSE", x, y, 0.7, True, "Operon")
evaluator = OperonEvaluator("RMSE", x, y, 0.7, True, "self")
gp_evaluator = GpEvaluator(x, y, "self", metric="rmse")
kb_gen_up_oplist = CompositeOp([bg_crossover, bg_mutation])
kb_gen_down_oplist = CompositeOpReturn([bg_selector])
kb_gen_eva_oplist = CompositeOp([bg_evaluator])
gen_up_oplist = CompositeOp([bg_crossover, bg_mutation])
gen_down_oplist = CompositeOpReturn([bg_selector])
gen_eva_oplist = CompositeOp([gp_evaluator])
op_up_list = [op_mutation, op_crossover]
eval_op_list = [evaluator]
population = op_creator.do()
evaluator.do(population)
ck = CheckPopulation(data)
# 训练强化学习代理
for episode in range(num_episodes):
    # 在每个episode开始时，初始化环境
    list1 = ck.do(population)
    # print(list1)
    state = np.array(list1)  # 状态
    # print(state[0])
    episode_states, episode_actions, episode_rewards = [], [], []
    generation_num = 0
    while True:
        # 选择动作
        # print(state)
        # action_probabilities = policy_network(np.array([state], dtype=np.float32))
        # action = np.random.choice(num_actions, p=action_probabilities.numpy()[0])
        action_probabilities = policy_network(np.array([state], dtype=np.float32))
        # print(action_probabilities)
        action_probabilities = np.where(np.isnan(action_probabilities), 1e-6, action_probabilities)
        action_probabilities /= np.sum(action_probabilities)
        print(action_probabilities)
        action = np.random.choice(num_actions, p=action_probabilities[0])
        pool_population = population
        new_ind = None
        # print(action)
        # 执行动作并观察奖励和新状态
        if action == 0:
            op_crossover.do(pool_population)
            new_ind = op_crossover.new_ind
            old_inds = op_crossover.old_inds
            eval = OperonSingleEvaluator("RMSE", x, y, 0.9, True, new_ind)
            new_fit = eval.do()
            eval = OperonSingleEvaluator("RMSE", x, y, 0.9, True, old_inds[0])
            old_fit1 = eval.do()
            eval = OperonSingleEvaluator("RMSE", x, y, 0.9, True, old_inds[1])
            old_fit2 = eval.do()
            if old_fit1 > new_fit and old_fit2 > new_fit:
                reward = 1
            else:
                reward = -1
        elif action == 1:
            op_mutation.do(pool_population)
            old_tree_list = op_mutation.old_tree_list
            new_tree_list = op_mutation.new_tree_list
            old_ind_list = []
            new_ind_list = []
            for i in old_tree_list:
                ind = Operon.Individual()
                ind.Genotype = i
                old_ind_list.append(ind)
            for i in new_tree_list:
                ind = Operon.Individual()
                ind.Genotype = i
                new_ind_list.append(ind)
            y = y.reshape([-1, 1])
            ds = Operon.Dataset(np.hstack([x, y]))
            target = ds.Variables[-1]
            inputs = Operon.VariableCollection(v for v in ds.Variables if v.Name != target.Name)
            rng = Operon.RomuTrio(random.randint(1, 1000000))
            training_range = Operon.Range(0, int(ds.Rows))
            test_range = Operon.Range(int(ds.Rows), ds.Rows)
            problem = Operon.Problem(ds, inputs, target.Name, training_range, test_range)
            interpreter = Operon.Interpreter()
            evaluator = Operon.Evaluator(problem, interpreter, "RMSE", True)
            old_fit_list = []
            new_fit_list = []
            for i in old_ind_list:
                ea = evaluator(rng, i)
                old_fit_list.append(ea[0])
            for i in new_ind_list:
                ea = evaluator(rng, i)
                new_fit_list.append(ea[0])
            old_fit_list = np.array(old_fit_list)
            new_fit_list = np.array(new_fit_list)
            mean_old = np.mean(old_fit_list)
            mean_new = np.mean(new_fit_list)
            reward = -(mean_new - mean_old)


        elif action == 2:
            bg_crossover.do(pool_population)
            reward = 1
        elif action == 3:
            bg_mutation.do(pool_population)
            reward = 1
        elif action == 4:
            rein = KeplarReinserter(pool_population, "self")
            rein.do(population)
            reward = 0
        else:
            raise ValueError("其他方法暂未实现")
        # print(population.pop_type)
        # print(len(population.pop_list))
        # print(population.pop_type)
        # print(len(population.pop_list))
        # for i in population.pop_list:
        #     print(i.format())
        # print(population.pop_type)
        evaluator.do(population)
        # print(population.pop_type)
        # print(population.pop_list[0].fitness)
        list1 = ck.do(population)
        # print(list1)
        new_state = np.array(list1)  # 假设新状态是一个一维向量，根据您的实际情况调整
        # reward = calculate_reward(state, action, new_state)  # 根据游戏的奖励函数计算奖励
        # print(reward)

        episode_states.append(state)
        episode_actions.append(action)
        episode_rewards.append(reward)

        state = new_state

        # 检查是否结束
        generation_num += 1
        # print(generation_num)
        if generation_num > 1000:
            break

        # 计算回报并更新策略网络
        print(episode_rewards)
        episode_returns = calculate_returns(episode_rewards)

        train_policy_network(policy_network, np.vstack(episode_states), np.array(episode_actions), episode_returns,
                             optimizer)

# 最后，您可以使用训练好的策略网络来玩游戏。
