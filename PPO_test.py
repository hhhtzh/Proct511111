import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


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
    action_dim = 1  # 这里假设动作维度为1，你可以根据需要修改
    agent = PPOAgent(state_dim, action_dim)

    # 生成动作序列的示例
    state = np.random.randn(state_dim)  # 初始状态
    max_steps = 100  # 控制序列的最大长度
    actions = []

    for _ in range(max_steps):
        action = agent.get_action(state)
        actions.append(action[0])
        next_state = np.random.randn(state_dim)  # 假设下一个状态是随机生成的
        reward = np.random.randn()  # 假设奖励也是随机的
        done = False  # 假设未结束
        state = next_state

    print("Generated Action Sequence:", actions)
