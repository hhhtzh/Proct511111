import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

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


# 创建Actor网络，用于生成动作
class ActorNetwork(tf.keras.Model):
    def __init__(self, action_dim):
        super(ActorNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(action_dim, activation='tanh')

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        actions = self.output_layer(x)
        return actions


# 创建Critic网络，用于评估动作价值
class CriticNetwork(tf.keras.Model):
    def __init__(self):
        super(CriticNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1, activation=None)

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        value = self.output_layer(x)
        return value


# PPO算法
class PPOAgent:
    def __init__(self, state_dim, action_dim, clip_epsilon=0.2, gamma=0.99):
        self.actor_network = ActorNetwork(action_dim)
        self.critic_network = CriticNetwork()
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma

    def get_action(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        actions = self.actor_network(state)
        return actions.numpy()

    def train(self, states, actions, rewards, next_states, dones, old_probs, epochs=5):
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)

        for _ in range(epochs):
            with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
                # 计算动作的概率分布
                probs = self.actor_network(states)
                dist = tfp.distributions.Normal(probs, 1)
                new_probs = dist.prob(actions)
                old_probs = tf.convert_to_tensor(old_probs, dtype=tf.float32)

                # 计算策略比率
                ratio = new_probs / (old_probs + 1e-5)
                clipped_ratio = tf.clip_by_value(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)

                # 计算PPO的损失函数
                surrogate_obj = tf.minimum(ratio * rewards, clipped_ratio * rewards)
                actor_loss = -tf.reduce_mean(surrogate_obj)

                # 计算Critic的损失函数
                values = self.critic_network(states)
                td_errors = rewards + self.gamma * self.critic_network(next_states) * (1 - dones) - values
                critic_loss = tf.reduce_mean(tf.square(td_errors))

            actor_gradients = tape1.gradient(actor_loss, self.actor_network.trainable_variables)
            critic_gradients = tape2.gradient(critic_loss, self.critic_network.trainable_variables)

            self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor_network.trainable_variables))
            self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic_network.trainable_variables))


if __name__ == "__main__":
    # 创建PPO代理
    state_dim = 6
    action_dim = 4  # 这里假设动作维度为1，你可以根据需要修改
    agent = PPOAgent(state_dim, action_dim)  # 生成动作序列的示例
    list1 = ck.do(population)
    state = np.array(list1)  # 初始状态
    max_steps = 100  # 控制序列的最大长度
    actions = []

    for _ in range(max_steps):
        action = agent.get_action(state)
        actions.append(action[0])
        next_state = np.random.randn(state_dim)  # 假设下一个状态是随机生成的
        reward = np.random.randn()  # 假设奖励也是随机的
        done = False  # 假设未结束
        state = next_state
    print(len(actions))
    print("Generated Action Sequence:", actions)
    for i in actions:
        print(i)
