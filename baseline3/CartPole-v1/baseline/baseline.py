import gymnasium as gym
import chex
import jax
from sbx import DQN , PPO

env = gym.make("CartPole-v1", render_mode="none")
model = PPO("MlpPolicy", env, tensorboard_log="./logs",verbose=1)
model.learn(total_timesteps=100_000, progress_bar=True)
model.save("ppo_cartpole")  

model = DQN("MlpPolicy", env, tensorboard_log="./logs",verbose=1)
model.learn(total_timesteps=100_000, progress_bar=True)
model.save("dqn_cartpole")  
