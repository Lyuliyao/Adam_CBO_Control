import gymnasium as gym
import chex
import jax
from sbx import DDPG , PPO, SAC, TD3, TQC, CrossQ
import time 


start_time = time.time()
env = gym.make("Pendulum-v1", render_mode="none")
model = DDPG("MlpPolicy", env, tensorboard_log="./logs/DDPG",verbose=1)
model.learn(total_timesteps=100_000, progress_bar=False)
model.save("DDPG_Pendulum")
end_time = time.time()
DDPG_time = end_time - start_time

start_time = time.time()
model = PPO("MlpPolicy", env, tensorboard_log="./logs/PPO",verbose=1)
model.learn(total_timesteps=100_000, progress_bar=False)
model.save("PPO_Pendulum")
end_time = time.time()
PPO_time = end_time - start_time

start_time = time.time()
model = SAC("MlpPolicy", env, tensorboard_log="./logs/SAC",verbose=1)
model.learn(total_timesteps=100_000, progress_bar=False)
model.save("SAC_Pendulum")
end_time = time.time()
SAC_time = end_time - start_time

start_time = time.time()
model = TD3("MlpPolicy", env, tensorboard_log="./logs/TD3",verbose=1)
model.learn(total_timesteps=100_000, progress_bar=False)
model.save("TD3_Pendulum")
end_time = time.time()
TD3_time = end_time - start_time


start_time = time.time()
model = TQC("MlpPolicy", env, tensorboard_log="./logs/TQC",verbose=1)
model.learn(total_timesteps=100_000, progress_bar=False)
model.save("TQC_Pendulum")
end_time = time.time()
TQC_time = end_time - start_time

start_time = time.time()
model = CrossQ("MlpPolicy", env, tensorboard_log="./logs/CrossQ",verbose=1)
model.learn(total_timesteps=100_000, progress_bar=False)
model.save("CrossQ_Pendulum")
end_time = time.time()
CrossQ_time = end_time - start_time


print(f"DDPG_time: {DDPG_time}")
print(f"PPO_time: {PPO_time}")
print(f"SAC_time: {SAC_time}")
print(f"TD3_time: {TD3_time}")
print(f"TQC_time: {TQC_time}")
print(f"CrossQ_time: {CrossQ_time}")
