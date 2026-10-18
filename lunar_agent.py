import gymnasium as gym
import time
env=gym.make("LunarLander-v3",render_mode="human")
observation, info = env.reset()
print("Game started!")
print("Initial observation:", observation)

#input("Press Enter in this terminal to close the game...")
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