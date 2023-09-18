import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
import gym

# 环境

state_dim = 6
action_dim = 3
action_high = 2
action_low = 0

# DDPG参数
gamma = 0.99
tau = 0.001
exploration_noise = 0.1
buffer_size = 10000
batch_size = 64

# 经验回放缓冲区
replay_buffer = []

# Actor网络
def build_actor_network():
    input_state = Input(shape=(state_dim,))
    x = Dense(256, activation='relu')(input_state)
    x = Dense(128, activation='relu')(x)
    output_action = Dense(action_dim, activation='tanh')(x)
    output_action = tf.multiply(output_action, action_high)
    model = Model(inputs=input_state, outputs=output_action)
    return model

# Critic网络
def build_critic_network():
    input_state = Input(shape=(state_dim,))
    input_action = Input(shape=(action_dim,))
    x_state = Dense(256, activation='relu')(input_state)
    x_action = Dense(256, activation='relu')(input_action)
    x = tf.concat([x_state, x_action], axis=-1)
    x = Dense(128, activation='relu')(x)
    output_value = Dense(1)(x)
    model = Model(inputs=[input_state, input_action], outputs=output_value)
    return model

actor_network = build_actor_network()
critic_network = build_critic_network()

# 目标Actor和Critic网络（用于更新）
target_actor_network = build_actor_network()
target_critic_network = build_critic_network()
target_actor_network.set_weights(actor_network.get_weights())
target_critic_network.set_weights(critic_network.get_weights())

# Actor和Critic优化器
actor_optimizer = Adam(learning_rate=0.001)
critic_optimizer = Adam(learning_rate=0.002)

# 定义动作选择函数（添加探索噪声）
def select_action(state):
    action = actor_network.predict(state)
    noise = exploration_noise * np.random.randn(action_dim)
    action = np.clip(action + noise, action_low, action_high)
    return action

# 训练DDPG
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0

    while True:
        action = select_action(np.array([state]))
        next_state, reward, done, _ = env.step(action)
        replay_buffer.append((state, action, reward, next_state, done))

        if len(replay_buffer) > buffer_size:
            replay_buffer.pop(0)

        if len(replay_buffer) > batch_size:
            minibatch = random.sample(replay_buffer, batch_size)
            states = np.array([s[0] for s in minibatch])
            actions = np.array([s[1] for s in minibatch])
            rewards = np.array([s[2] for s in minibatch])
            next_states = np.array([s[3] for s in minibatch])
            dones = np.array([s[4] for s in minibatch])

            target_actions = target_actor_network.predict(next_states)
            target_values = target_critic_network.predict([next_states, target_actions])
            target_rewards = rewards + gamma * target_values * (1 - dones)

            critic_loss = critic_network.train_on_batch([states, actions], target_rewards)
            actor_loss = -critic_network.train_on_batch([states, actor_network.predict(states)], np.ones((batch_size, 1)))

            # 更新目标网络
            actor_weights = actor_network.get_weights()
            target_actor_weights = target_actor_network.get_weights()
            for i in range(len(actor_weights)):
                target_actor_weights[i] = tau * actor_weights[i] + (1 - tau) * target_actor_weights[i]
            target_actor_network.set_weights(target_actor_weights)

            critic_weights = critic_network.get_weights()
            target_critic_weights = target_critic_network.get_weights()
            for i in range(len(critic_weights)):
                target_critic_weights[i] = tau * critic_weights[i] + (1 - tau) * target_critic_weights[i]
            target_critic_network.set_weights(target_critic_weights)

            state = next_state
            episode_reward += reward

        if done:
            break

    print(f"Episode: {episode + 1}, Reward: {episode_reward}")

# 训练结束后，可以使用actor_network来选择动作
