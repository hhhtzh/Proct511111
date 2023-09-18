import numpy as np
import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from transformers import AutoTokenizer, AutoModel

from keplar.Algorithm.Alg import KeplarBingoAlg, GpBingo2Alg, KeplarOperonAlg
from keplar.data.data import Data
from keplar.operator.check_pop import CheckPopulation
from keplar.operator.composite_operator import CompositeOp, CompositeOpReturn
from keplar.operator.creator import OperonCreator, BingoCreator
from keplar.operator.crossover import BingoCrossover, OperonCrossover
from keplar.operator.evaluator import OperonEvaluator, BingoEvaluator, GpEvaluator
from keplar.operator.mutation import BingoMutation, OperonMutation
from keplar.operator.selector import BingoSelector

# data = Data("pmlb", "1027_ESL", ["x0", "x1", "x2", "x3", 'y'])
data = Data("txt", "datasets/vla/two/1.txt", ["x0", "x1", "y"])
data.read_file()
data.set_xy("y")
x = data.get_np_x()
y = data.get_np_y()

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


# 将数学表达式转换为句向量
# def expression_to_sentence_vector(expression):
#     # 分词
#     tokens = tokenizer.encode(expression, add_special_tokens=True)
#
#     # 将分词转换为PyTorch张量
#     input_ids = torch.tensor(tokens).unsqueeze(0)
#
#     # 获取句向量
#     with torch.no_grad():
#         outputs = model(input_ids)
#         sentence_vector = outputs.last_hidden_state.mean(dim=1).squeeze()
#
#     return sentence_vector


def expression_to_sentence_vector(expression, max_seq_length=512):
    # 检查表达式长度是否超过最大序列长度
    if len(expression) > max_seq_length:
        # 如果超过最大长度，则截断表达式
        expression = expression[:max_seq_length]

    # 分词
    tokens = tokenizer.encode(expression, add_special_tokens=True)

    # 将分词转换为PyTorch张量
    input_ids = torch.tensor(tokens).unsqueeze(0)

    # 获取句向量
    with torch.no_grad():
        outputs = model(input_ids)
        sentence_vector = outputs.last_hidden_state.mean(dim=1).squeeze()

    return sentence_vector


# 定义演员（Actor）和评论家（Critic）神经网络
class Actor(tf.keras.Model):
    def __init__(self, num_actions):
        super(Actor, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
        self.output_layer = tf.keras.layers.Dense(num_actions, activation='softmax')

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        return self.output_layer(x)


# def calculate_reward(state, action, new_state):
#     # 奖励函数
#     if new_state[0] < state[0]:  # 如果适应度下降
#         return 1.0
#     elif new_state[0] > state[0]:  # 如果适应度上升
#         return -2.0
#     else:
#         return 0.0  # 适应度上升返回零奖励


def calculate_reward(list, new_list):
    # 计算适应度变化
    fitness_change = new_list[0] - list[0]
    mean_fitness_change = new_list[2] - list[2]
    reward = 0
    # 根据适应度变化分配奖励
    if fitness_change < 0:
        reward += 1.0  # 适应度下降，奖励为正数
    elif fitness_change < 0:
        reward -= 2.0  # 适应度上升，奖励为负数
    else:
        reward += 0.0  # 适应度没有变化，奖励为零

    if mean_fitness_change < 0:
        reward += 1
    elif mean_fitness_change > 0:
        reward -= 2
    else:
        reward += 0

    return reward


class Critic(tf.keras.Model):
    def __init__(self):
        super(Critic, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(32, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1, activation=None)

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        return self.output_layer(x)


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


# 定义Actor-Critic代理
class ActorCriticAgent:
    def __init__(self, num_actions):
        self.actor = Actor(num_actions)
        self.critic = Critic()
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def select_action(self, state):
        action_probs = self.actor(state)
        # 处理 NaN 值：将 NaN 替换为均等的概率分布
        # action_probs = tf.where(tf.math.is_nan(action_probs), tf.ones_like(action_probs) / num_actions, action_probs)
        print(action_probs)
        action = np.random.choice(num_actions, p=action_probs.numpy()[0])
        return action

    def train(self, states, actions, rewards, next_states, dones, gamma=0.99):
        with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
            # 计算演员策略
            action_probs = self.actor(states)
            chosen_action_probs = tf.reduce_sum(action_probs * tf.one_hot(actions, num_actions), axis=1)

            # 计算评论家的值估计
            values = self.critic(states)
            next_values = self.critic(next_states)

            # 计算TD误差和策略梯度
            td_error = rewards + gamma * next_values * (1 - dones) - values
            actor_loss = -tf.math.log(chosen_action_probs) * td_error
            critic_loss = tf.square(td_error)

        actor_gradients = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        critic_gradients = critic_tape.gradient(critic_loss, self.critic.trainable_variables)

        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))


# 初始化Agent
num_actions = 3  # 假设有3种动作选项
agent = ActorCriticAgent(num_actions)

# 定义训练参数
num_episodes = 1000
max_steps_per_episode = 100

for episode in range(num_episodes):
    list1 = ck.do(population)
    print(list1)
    expressions = []
    for i in population.pop_list:
        str_equ = i.format()
        expressions.append(str_equ)
    vector = []
    for expression in expressions:
        vector = expression_to_sentence_vector(expression)
    episode_reward = 0
    done = False

    for step in range(max_steps_per_episode):
        # 在调用 select_action 之前，将状态的形状调整为 (1, 768)
        state = np.array(vector, dtype=np.float32)  # 假设 list1 是形状为 (768,) 的状态
        print(np.shape(state))
        state = np.expand_dims(state, axis=0)  # 添加一个维度，将形状调整为 (1, 768)
        action = agent.select_action(state)
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
        # next_state, reward, done, _ = env.step(action)

        evaluator.do(population)
        expressions = []
        for i in population.pop_list:
            str_equ = i.format()
            expressions.append(str_equ)
        vector = []
        for expression in expressions:
            vector = expression_to_sentence_vector(expression)
            # print(f"Expression: {expression}")
            # print(f"Sentence Vector: {vector}")
            # print()
        new_list1 = ck.do(population)
        print(list1)
        reward = calculate_reward(list1, new_list1)  # 根据游戏的奖励函数计算奖励
        new_state = np.array(vector)
        new_list1 = list1
        print(list1)

        agent.train(
            np.array([state], dtype=np.float32),
            action,
            reward,
            np.array([new_state], dtype=np.float32),
            done
        )

        episode_reward += reward
        state = new_state

        if done:
            break

    print(f"Episode {episode + 1}: Total Reward: {episode_reward}")