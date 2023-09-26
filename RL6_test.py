import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, LSTM  # 使用LSTM作为RNN结构
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras import backend as K
from tensorflow_probability import distributions as tfd
import tensorflow_probability as tfp


# Actor网络，输出序列的概率分布
def build_actor_network(seq_length, num_actions):
    state_input = Input(shape=(6,))
    x = Dense(64, activation='relu')(state_input)

    # 使用LSTM作为RNN结构，输出一个序列
    rnn_layer = LSTM(64, return_sequences=True)(x)

    # 输出序列的概率分布
    logits = Dense(num_actions, activation='linear')(rnn_layer)
    probabilities = tf.keras.layers.Softmax()(logits)

    model = Model(inputs=state_input, outputs=probabilities)
    return model


# ...

# 在每个时间步选择动作
def choose_action(actor, state):
    action_probs = actor.predict(state)
    action = np.random.choice(len(action_probs[0]), p=action_probs[0])
    return action


# ...

# 在每个时间步训练Agent
def train(agent, states, actions, rewards, dones):
    # 训练逻辑
    pass


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


# ...
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


# 训练PPO Agent
seq_length = 10  # 序列的最大长度
num_actions = 4  # 假设有4个离散的动作
agent = PPOAgent(seq_length, num_actions)

# 定义环境和训练参数
num_episodes = 1000
state = np.random.rand(6)  # 初始状态

for episode in range(num_episodes):
    states, actions, rewards, dones = [], [], [], []
    done = False
    while not done:
        action = choose_action(agent.actor, state)
        next_state = np.random.rand(6)  # 模拟环境返回下一个状态
        reward = np.random.rand()  # 模拟环境返回奖励
        done = np.random.choice([True, False])  # 模拟环境返回done信号
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        state = next_state

        if len(states) >= seq_length or done:
            # 当序列长度达到最大值或者环境返回done信号时，进行一次训练
            train(agent, np.array(states), np.array(actions), np.array(rewards), np.array(dones))
            states, actions, rewards, dones = [], [], [], []
