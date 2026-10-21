import gymnasium as gym
from stable_baselines3 import PPO  
env=gym.make("LunarLander-v3")
lunar_model=PPO("MlpPolicy",env,verbose=1)
lunar_model.learn(total_timesteps=100_000)#100000 and 100_000 is identical python allows to use _ in nummeric strings to improve readability for large numbers

