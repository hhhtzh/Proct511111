import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from keras.layers import Input, Dense, LSTM
from keras.models import Model
from keras.optimizers import Adam
import tensorflow_probability as tfp
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from keplar.data.data import Data
from keplar.operator.check_pop import CheckPopulation
from keplar.operator.composite_operator import CompositeOpReturn, CompositeOp
from keplar.operator.creator import BingoCreator, OperonCreator
from keplar.operator.crossover import BingoCrossover, OperonCrossover
from keplar.operator.evaluator import OperonEvaluator, GpEvaluator, BingoEvaluator
from keplar.operator.linear_regression import SklearnTwoIndividualLinearRegression, \
    SklearnOneIndividualLinearRegression, SklearnLinearRegression
from keplar.operator.mutation import OperonMutation, BingoMutation
from keplar.operator.reinserter import KeplarReinserter
from keplar.operator.selector import BingoSelector

data = Data("pmlb", "1027_ESL", ["x0", "x1", "x2", "x3", 'y'])
# data = Data("txt_pmlb", "datasets/pmlb/val/503_wind.txt", ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13","y"])

data.read_file()
# data.set_xy("y")
x = data.get_np_x()
y = data.get_np_y()

from transformers import BertModel, BertTokenizer

# 指定本地模型路径
model_path = "model/bert-base-uncased"

# 加载本地模型
model = BertModel.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)


def calculate_reward(list, new_list):
    # 计算适应度变化
    fitness_change = new_list[0] - list[0]
    mean_fitness_change = new_list[2] - list[2]
    reward = 0
    # 根据适应度变化分配奖励
    if fitness_change < 0:
        reward += 5.0  # 适应度下降，奖励为正数
    elif fitness_change < 0:
        reward -= 10.0  # 适应度上升，奖励为负数
    else:
        reward += 0.0  # 适应度没有变化，奖励为零

    if mean_fitness_change < 0:
        reward += 1
    elif mean_fitness_change > 0:
        reward -= 2
    else:
        reward += 0

    return reward


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


# 定义Actor网络
class ActorNetwork(tf.keras.Model):
    def __init__(self, num_actions):
        super(ActorNetwork, self).__init__()
        self.num_actions = num_actions
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(64, activation='relu')
        self.dense3 = Dense(64, activation='relu')
        self.output_layer = Dense(num_actions, activation='softmax')

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        x = self.dense3(x)
        action_probabilities = self.output_layer(x)
        return action_probabilities


# 定义Critic网络
class CriticNetwork(tf.keras.Model):
    def __init__(self):
        super(CriticNetwork, self).__init__()
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(64, activation='relu')
        self.dense3 = Dense(64, activation='relu')
        self.output_layer = Dense(1, activation=None)

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        x = self.dense3(x)
        value = self.output_layer(x)
        return value


# 定义PPOAgent
class PPOAgent:
    def __init__(self, num_actions, clip_epsilon=0.2, gamma=0.99):
        self.num_actions = num_actions
        self.actor_network = ActorNetwork(num_actions)
        self.critic_network = CriticNetwork()
        self.actor_optimizer = Adam(learning_rate=0.001)
        self.critic_optimizer = Adam(learning_rate=0.001)
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma

    def get_action(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        # print("state:", state)
        action_probabilities = self.actor_network(state).numpy()[0]
        # print(action_probabilities)
        for i in action_probabilities:
            if str(i)=="nan":
                action = 4
                return action
        action = np.random.choice(self.num_actions, p=action_probabilities)
        return action

    def train(self, states, actions, rewards, dones, old_probs, epochs=5):
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)

        for _ in range(epochs):
            with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
                action_probabilities = self.actor_network(states)
                action_masks = tf.one_hot(actions, self.num_actions)
                # print(action_probabilities)
                print("action_mask:",action_masks)
                selected_action_probabilities = tf.reduce_sum(action_probabilities * action_masks, axis=1)
                old_action_masks = tf.one_hot(actions, self.num_actions)
                # print("old_action_masks:",old_action_masks)
                old_action_probabilities = old_action_masks * action_probabilities
                old_action_probabilities = tf.reduce_sum(old_action_probabilities, axis=1)
                print("selected_action_probabilities",selected_action_probabilities)
                print("old_action_probabilities",old_action_probabilities)
                ratio = selected_action_probabilities / (old_action_probabilities + 1e-5)
                # 将inf替换为999
                ratio = tf.where(tf.math.is_inf(ratio), 999.0, ratio)

                # 将nan替换为0
                ratio = tf.where(tf.math.is_nan(ratio), 0.0, ratio)
                print("reward:", rewards)
                advantages = calculate_advantages(rewards, dones, self.critic_network(states))
                advantages = tf.where(tf.math.is_inf(advantages), 999.0, advantages)

                # 将nan替换为0
                advantages = tf.where(tf.math.is_nan(advantages), 0.0, advantages)
                print("ratio:", ratio)
                print("advantages", advantages)
                surrogate_obj = tf.minimum(ratio * advantages, tf.clip_by_value(ratio, 1 - self.clip_epsilon,
                                                                                1 + self.clip_epsilon) * advantages)
                actor_loss = -tf.reduce_mean(surrogate_obj)
                print("actor_loss",actor_loss)

                values = self.critic_network(states)
                critic_loss = tf.reduce_mean(tf.square(rewards - values))

            actor_gradients = tape1.gradient(actor_loss, self.actor_network.trainable_variables)
            critic_gradients = tape2.gradient(critic_loss, self.critic_network.trainable_variables)

            self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor_network.trainable_variables))
            self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic_network.trainable_variables))


# 计算优势函数
def calculate_advantages(rewards, dones, values):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = 0
    last_value = 0
    print("奖励序列:",rewards)
    print("价值序列:",values)
    for t in reversed(range(len(rewards))):
        mask = 1 - dones[t]
        delta = rewards[t] + 0.99 * last_value * mask - values[t]
        advantages[t] = delta + 0.99 * 0.95 * last_advantage * mask
        last_advantage = advantages[t]
        last_value = values[t]
    return advantages


operators = ["+", "-", "*", "/", "sin", "exp", "cos", 'sqrt', 'log', 'sin', 'pow', 'exp', '^']
bg_creator = BingoCreator(128, operators, x, 10, "Bingo")
bg_evaluator = BingoEvaluator(x, "exp", "lm", "self", y)
bg_crossover = BingoCrossover("Bingo")
bg_mutation = BingoMutation(x, operators, "self")
bg_selector = BingoSelector(0.5, "tournament", "self")
op_crossover = OperonCrossover(x, y, "self")
select = BingoSelector(0.2, "tournament", "Operon")
op_mutation = OperonMutation(0.6, 0.7, 0.8, 0.8, x, y, 10, 50, "balanced", "self")
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
lr= SklearnOneIndividualLinearRegression(data)
lr1=SklearnLinearRegression(data)

# 创建PPOAgent
num_actions = 7  # 五个离散
agent = PPOAgent(num_actions)

# 定义训练参数
num_episodes = 1000
generation_num = 1000

# 在每个episode中生成动作序列并训练Agent
for episode in range(num_episodes):
    episode_states, episode_actions, episode_rewards, episode_length, episode_probs = [], [], [], [], []
    for _ in range(generation_num):
        done = False
        # state = np.random.rand(6)  # 初始状态，这里使用随机生成的示例状态
        # list1 = ck.do(population)
        # print(list1)
        # state = np.array(list1)  # 状态
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
        evaluator.do(population)
        list1 = ck.do(population)
        print("初始最好适应度:" + str(list1[0]) + ",平均适应度:" + str(list1[2]))
        state = np.array(vector)
        # print(state)
        # min_val = np.min(state)
        # max_val = np.max(state)
        # state = (state - min_val) / (state - min_val)
        # print(state)
        done = False
        episode_actions = []
        while not done:
            # print("不变的state:" + str(state))
            action = agent.get_action(state)
            # print(action)
            # next_state = np.random.rand(6)  # 模拟环境返回下一个状态，这里使用随机生成的示例状态
            if action == 6:
                done = True

            # done = np.random.choice([True, False])  # 模拟环境返回done信号，这里使用随机生成的示例done信号
            episode_actions.append(action)

        episode_states = []
        print(episode_actions)
        # reward = np.random.rand()  # 模拟环境返回奖励，这里使用随机生成的示例奖励
        # evaluator.do(population)
        # pool_pop = bg_selector.do(population)
        # evaluator.do(pool_pop)
        # list2=ck.do(pool_pop)
        # print("筛选后最好适应度:" + str(list2[0]) + ",平均适应度:" + str(list2[2]))
        pool_pop = population.copy()
        pool_pop.pop_type = "self"

        # print(population.pop_type)
        for i in episode_actions:
            # for ind in pool_pop.pop_list:
            #     print(ind.func)
            if i == 0:
                print("0")
                # print("state_shape:", np.shape(episode_states))
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
                state = np.array(vector)
                episode_states.append(state)
                op_crossover.do(pool_pop)

            elif i == 1:
                print("1")
                # print("state_shape:", np.shape(episode_states))

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
                state = np.array(vector)
                episode_states.append(state)
                op_mutation.do(pool_pop)
            elif i == 2:
                print("2")
                # print("state_shape:", np.shape(episode_states))
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
                state = np.array(vector)
                episode_states.append(state)
                bg_crossover.do(pool_pop)
            elif i == 3:
                print("3")
                # print("state_shape:", np.shape(episode_states))
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
                state = np.array(vector)
                episode_states.append(state)
                bg_mutation.do(pool_pop)

            elif i == 4:
                print("4")
                # print("state_shape:", np.shape(episode_states))

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
                state = np.array(vector)
                episode_states.append(state)
                lr.do(pool_pop)
            elif i ==5:
                print("5")
                # print("state_shape:", np.shape(episode_states))
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
                state = np.array(vector)
                episode_states.append(state)
                lr1.do(pool_pop)



            elif i == 6:
                print("6")
                # print("state_shape:", np.shape(episode_states))
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
                state = np.array(vector)
                episode_states.append(state)
                rein = KeplarReinserter(pool_pop, "self")
                rein.do(population)
            else:
                raise ValueError("出错了")

        print(population.pop_type)
        print(population.pop_list)
        evaluator.do(population)
        new_list1 = ck.do(population)
        ck.write_rl_json(population, episode_actions, "RL6_test17")
        print("最好适应度:" + str(new_list1[0]) + ",平均适应度:" + str(new_list1[2]))
        reward = calculate_reward(list1, new_list1)
        list1 = new_list1
        episode_rewards = []
        for _ in range(len(episode_actions)):
            episode_rewards.append(reward)
        for _ in range(len(episode_actions)):
            episode_length.append(0)
        episode_probs.append(agent.actor_network(np.array([state], dtype=np.float32))[0, action])

        # state = next_state

        # 计算优势函数并训练Agent
        print("reward_shape:", np.shape(episode_rewards))
        print("length_shape:", np.shape(episode_length))
        # advantages = calculate_advantages(episode_rewards, episode_length,
        #                                   agent.critic_network(np.array(episode_states, dtype=np.float32)))
        print("epsode_actions:", episode_actions)
        print("state_shape:", np.shape(episode_states))
        agent.train(episode_states, episode_actions, episode_rewards, episode_length, episode_probs)

# 最后，您可以使用训练好的Agent来生成动作序列。
