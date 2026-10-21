
import gymnasium as gym
from reward_system import CustomReward

# Create the original environment
env = gym.make("LunarLander-v3")

# Apply our custom reward system
env = CustomReward(env)

observation, info = env.reset()

print("Initial observation:", observation)

# Take 5 actions to test the wrapper
for step in range(5):
    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)

    print(f"\nStep: {step + 1}")
    print(f"Action: {action}")
    print(f"Modified reward: {reward:.3f}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")

    if terminated or truncated:
        break

env.close()
