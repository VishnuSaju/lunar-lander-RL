
import gymnasium as gym


class CustomReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.previous_distance = None

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)

        # Remember the initial horizontal distance
        self.previous_distance = abs(observation[0])

        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        x_position = observation[0]
        x_velocity = observation[2]
        y_velocity = observation[3]
        angle = observation[4]

        # Reward progress toward the center
        current_distance = abs(x_position)
        progress_reward = (self.previous_distance - current_distance) * 5.0
        distance_penalty=2.5*current_distance

        self.previous_distance = current_distance

        # Small penalties for unstable movement
        velocity_penalty = 0.1 * (abs(x_velocity) + abs(y_velocity))
        angle_penalty = 0.1 * abs(angle)

        custom_reward = (reward+ progress_reward- velocity_penalty- angle_penalty-distance_penalty)

        return observation, custom_reward, terminated, truncated, info
