import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras.optimizers import Adam
import gym

# 环境
env = gym.make('Pendulum-v0')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_high = env.action_space.high
action_low = env.action_space.low

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
    state = env.reset()
    total_reward = 0

    while True:
        # 在环境中采取动作
        action = actor_network.predict(np.array([state]))[0]
        action += exploration_noise * np.random.randn(action_dim)
        action = np.clip(action, action_low, action_high)

        # 执行动作并观察下一个状态和奖励
        next_state, reward, done, _ = env.step(action)

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
