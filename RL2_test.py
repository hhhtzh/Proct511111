import numpy as np
import tensorflow as tf




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


# 定义Actor-Critic代理
class ActorCriticAgent:
    def __init__(self, num_actions):
        self.actor = Actor(num_actions)
        self.critic = Critic()
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def select_action(self, state):
        action_probs = self.actor(state)
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
    state = env.reset()
    episode_reward = 0

    for step in range(max_steps_per_episode):
        action = agent.select_action(np.array([state], dtype=np.float32))
        next_state, reward, done, _ = env.step(action)

        agent.train(
            np.array([state], dtype=np.float32),
            action,
            reward,
            np.array([next_state], dtype=np.float32),
            done
        )

        episode_reward += reward
        state = next_state

        if done:
            break

    print(f"Episode {episode + 1}: Total Reward: {episode_reward}")


