import random

import numpy as np
import pandas as pd
import tensorflow as tf
import keras.layers as layers
from collections import deque

from keplar.data.data import Data
from keplar.operator.check_pop import CheckPopulation
from keplar.operator.composite_operator import CompositeOp, CompositeOpReturn
from keplar.operator.creator import BingoCreator, OperonCreator
from keplar.operator.crossover import BingoCrossover, OperonCrossover
from keplar.operator.evaluator import BingoEvaluator, OperonEvaluator, GpEvaluator
from keplar.operator.mutation import BingoMutation, OperonMutation
from keplar.operator.selector import BingoSelector

data = Data("pmlb", "1027_ESL", ["x0", "x1", "x2", "x3", 'y'])
data.read_file()
x = data.get_np_x()
y = data.get_np_y()

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


# 定义经验回放缓冲区
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for experience in batch:
            state, action, reward, next_state, done = experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)


# 定义行动者网络（Actor）
class Actor(tf.keras.Model):
    def __init__(self, action_dim, max_action):
        super(Actor, self).__init__()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(32, activation='relu')
        self.output_layer = layers.Dense(action_dim, activation='softmax')

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        action = self.output_layer(x) * max_action
        return action


# 定义评论者网络（Critic）
class Critic(tf.keras.Model):
    def __init__(self):
        super(Critic, self).__init__()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(32, activation='relu')
        self.output_layer = layers.Dense(1)

    def call(self, state, action):
        x = tf.concat([state, action], axis=-1)
        x = self.dense1(x)
        x = self.dense2(x)
        q_value = self.output_layer(x)
        return q_value


# 定义DDPG代理
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(action_dim, max_action)
        self.target_actor = Actor(action_dim, max_action)
        self.critic = Critic()
        self.target_critic = Critic()
        self.max_action = max_action

        self.actor_optimizer = tf.optimizers.Adam(learning_rate=0.001)
        self.critic_optimizer = tf.optimizers.Adam(learning_rate=0.002)
        self.buffer = ReplayBuffer(buffer_size=10000)

    def train(self, batch_size=64):
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)

        with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
            target_actions = self.target_actor(next_states)
            q_values_next = self.target_critic(next_states, target_actions)
            q_targets = rewards + 0.99 * q_values_next * (1.0 - dones)

            q_values = self.critic(states, actions)
            critic_loss = tf.reduce_mean(tf.square(q_values - q_targets))

            actor_loss = -tf.reduce_mean(self.critic(states, self.actor(states)))

        actor_gradients = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        critic_gradients = critic_tape.gradient(critic_loss, self.critic.trainable_variables)

        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

        self.update_target_networks()

    def update_target_networks(self):
        actor_weights = self.actor.get_weights()
        target_actor_weights = self.target_actor.get_weights()
        for i in range(len(actor_weights)):
            target_actor_weights[i] = 0.995 * target_actor_weights[i] + 0.005 * actor_weights[i]
        self.target_actor.set_weights(target_actor_weights)

        critic_weights = self.critic.get_weights()
        target_critic_weights = self.target_critic.get_weights()
        for i in range(len(critic_weights)):
            target_critic_weights[i] = 0.995 * target_critic_weights[i] + 0.005 * critic_weights[i]
        self.target_critic.set_weights(target_critic_weights)


# 创建环境
state_dim = 6
action_dim = 1
max_action = 100.0
agent = DDPGAgent(state_dim, action_dim, max_action)

# 训练DDPG代理
for episode in range(1000):
    list1 = ck.do(population)
    state = np.array(list1)  # 状态
    state = np.expand_dims(state, axis=0)
    total_reward = 0
    for t in range(100):
        action = agent.actor(state)
        action = np.clip(action, 0, max_action)
        print(action)
        sequence = []
        for i in range(int(action[0][0])):
            sequence.append()
        next_state = state + action
        reward = -np.square(next_state)  # 这里定义一个简单的奖励函数
        agent.buffer.add(state, action, reward, next_state, False)
        if len(agent.buffer.buffer) > 64:
            agent.train()
        state = next_state
        total_reward += reward
    print("Episode: {}, Total Reward: {}".format(episode, total_reward))
