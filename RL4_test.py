import random

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense
from keras.optimizers import Adam

from keplar.Algorithm.Alg import KeplarBingoAlg, KeplarOperonAlg, GpBingo2Alg
from keplar.data.data import Data
from keplar.operator.check_pop import CheckPopulation
from keplar.operator.composite_operator import CompositeOp, CompositeOpReturn
from keplar.operator.creator import BingoCreator, OperonCreator
from keplar.operator.crossover import BingoCrossover, OperonCrossover
from keplar.operator.evaluator import BingoEvaluator, OperonEvaluator, GpEvaluator
from keplar.operator.mutation import BingoMutation, OperonMutation
from keplar.operator.selector import BingoSelector

# 环境
state_dim = 6
action_dim = 3
action_high = 2
action_low = 0


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


# Actor网络
def build_actor():
    input_state = tf.keras.layers.Input(shape=(state_dim,))
    x = Dense(256, activation='relu')(input_state)
    x = Dense(128, activation='relu')(x)
    output_action = Dense(action_dim, activation='tanh', kernel_initializer='random_uniform')(x)
    scaled_action = tf.keras.layers.Lambda(lambda x: x * action_high)(output_action)
    model = tf.keras.models.Model(inputs=input_state, outputs=scaled_action)
    return model

# Critic网络
def build_critic():
    input_state = tf.keras.layers.Input(shape=(state_dim,))
    input_action = tf.keras.layers.Input(shape=(action_dim,))
    x = tf.keras.layers.Concatenate()([input_state, input_action])
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    output_value = Dense(1)(x)
    model = tf.keras.models.Model(inputs=[input_state, input_action], outputs=output_value)
    return model



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


# 建立Actor和Critic网络
actor_network = build_actor()
critic_network = build_critic()

# 建立目标Actor和目标Critic网络（用于延迟更新）
target_actor_network = build_actor()
target_critic_network = build_critic()

# 定义优化器
actor_optimizer = Adam(learning_rate=0.001)
critic_optimizer = Adam(learning_rate=0.002)

# 定义回报折扣因子
gamma = 0.99

# 定义噪声参数
exploration_noise = 0.1
noise_decay = 0.999

# 定义经验回放缓冲区
buffer_size = 100000
buffer = []

# 定义训练参数
batch_size = 64
actor_update_interval = 1
critic_update_interval = 1

# 训练DDPG代理
for episode in range(1000):
    population = op_creator.do()
    evaluator.do(population)
    ck = CheckPopulation(data)
    list1 = ck.do(population)
    print(list1)
    state = np.array(list1)  # 状态
    total_reward = 0

    while True:
        done = False
        # 在环境中采取动作
        action = actor_network.predict(np.array([state]))[0]
        action += exploration_noise * np.random.randn(action_dim)
        action = np.clip(action, action_low, action_high)
        print(action)
        action_probabilities = action / np.sum(action)
        # 采样一个动作
        action = np.random.choice(len(action_probabilities), p=action_probabilities)

        print("合为1的概率分布:", action_probabilities)
        print("采样的动作:", action)
        # 执行动作并观察下一个状态和奖励
        if action == 0:
            print("bg")
            bgsr = KeplarBingoAlg(1, kb_gen_up_oplist, kb_gen_down_oplist, kb_gen_eva_oplist, 0.001, population)
            bgsr.one_gen_run()

        elif action == 1:
            print("gpbg2")
            gpbg2 = GpBingo2Alg(1, gen_up_oplist, gen_down_oplist, gen_eva_oplist, 0.001, population)
            gpbg2.one_gen_run()

        elif action == 2:
            print("ko")
            opbg = KeplarOperonAlg(1, op_up_list, None, eval_op_list, -10, population, select, x, y, 128)
            opbg.one_gen_run()

        else:
            raise ValueError("其他方法暂未实现")

        evaluator.do(population)
        list1 = ck.do(population)
        new_state = np.array(list1)
        print(list1)
        reward = calculate_reward(state, action, new_state)
        next_state = new_state

        # 存储经验
        buffer.append((state, action, reward, next_state, done))

        state = next_state
        total_reward += reward

        # 如果缓冲区足够大，开始训练网络
        if len(buffer) >= batch_size:
            experiences = random.sample(buffer, batch_size)
            states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = map(np.array, zip(*experiences))

            # 训练Critic网络
            target_actions = target_actor_network.predict(next_states_batch)
            target_values = target_critic_network.predict([next_states_batch, target_actions])
            target_rewards = rewards_batch + gamma * target_values.squeeze() * (1 - dones_batch)
            critic_loss = critic_network.train_on_batch([states_batch, actions_batch], target_rewards)

            # 训练Actor网络
            actions_for_grad = actor_network.predict(states_batch)
            critic_grads = critic_network.gradient([states_batch, actions_for_grad])
            actor_optimizer.apply_gradients(zip(critic_grads, actor_network.trainable_variables))

            # 更新目标网络
            if episode % actor_update_interval == 0:
                target_actor_network.set_weights(actor_network.get_weights())
            if episode % critic_update_interval == 0:
                target_critic_network.set_weights(critic_network.get_weights())

        # 更新噪声参数
        exploration_noise *= noise_decay
        exploration_noise = max(exploration_noise, 0.01)

        if done:
            break

    print(f"Episode: {episode + 1}, Total Reward: {total_reward:.2f}")
