#imports
import gymnasium as gym
from stable_baselines3 import PPO  
from pathlib import Path
import numpy as np

env=gym.make("LunarLander-v3")
model_path=Path("models/lunar_ppo_v2.zip")
if model_path.exists():
    lunar_model=PPO.load(model_path,env=env)
    
else:
    env.close()
    raise SystemExit("No trained model exists. Train the model first!!")

episode_rewards = []
for episode in range(100):
    observation, info = env.reset()
    total_reward = 0
    terminated = False
    truncated = False
    while not(terminated or truncated):
        action,_=lunar_model.predict(observation,deterministic=True)
        observation, reward, terminated, truncated, info= env.step(action)
        total_reward+=reward
    # Lunar Lander observation:
    # index 0 = horizontal position relative to landing pad
    # index 2 = horizontal velocity
    # index 4 = angle
    # index 6 = left leg contact
    # index 7 = right leg contact
    x_position = observation[0]
    x_velocity = observation[2]
    angle = observation[4]
    left_leg = bool(observation[6])
    right_leg = bool(observation[7])

    episode_rewards.append(total_reward)
    print(f"\nEpisode {episode + 1}")
    print(f"Reward: {total_reward:.2f}")
    print(f"Horizontal position: {x_position:.3f}")
    print(f"Horizontal velocity: {x_velocity:.3f}")
    print(f"Angle: {angle:.3f}")
    print(f"Left leg contact: {left_leg}")
    print(f"Right leg contact: {right_leg}")
    print(f"Terminated: {terminated}")
    print(f"Timed out: {truncated}")

input("Press Enter to continue to the next episode...")

print("\n--- Overall results ---")
print(f"Mean reward: {np.mean(episode_rewards):.2f}")
print(f"Highest reward: {np.max(episode_rewards):.2f}")
print(f"Lowest reward: {np.min(episode_rewards):.2f}")


print("episode reward :",total_reward)
print("Terminated:", terminated)
print("Timed out:", truncated)
env.close()