import gymnasium as gym
import time
env=gym.make("LunarLander-v3",render_mode="human")
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
observation, info = env.reset()
print("Game started!")
print("Initial observation:", observation)
for _ in range(300):
    observation, reward, terminated, truncated, info = env.step(0)
    print("Reward:", reward)
    print("terminated:",terminated)
    print("truncated:",truncated)
    env.render()
    time.sleep(1 / 60)
    if terminated or truncated:
        break

env.close()
print("Game closed.")