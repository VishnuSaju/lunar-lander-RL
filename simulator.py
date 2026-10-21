#imports
import gymnasium as gym
from stable_baselines3 import PPO  
from pathlib import Path

env=gym.make("LunarLander-v3",render_mode="human")
model_path=Path("models/lunar_ppo_v2.zip")
if not model_path.exists():
    env.close()
    raise SystemExit("No trained model exists. Train the model first!!")

lunar_model=PPO.load(model_path,env=env)
observation, info = env.reset()
terminated = False
truncated = False
total_reward=0
while not(terminated or truncated):
     action,_=lunar_model.predict(observation,deterministic=True)
     observation, reward, terminated, truncated, info= env.step(action)
     total_reward+=reward

print("episode reward :",total_reward)
print("Terminated:", terminated)
print("Timed out:", truncated)
env.close()