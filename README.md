Lunar lander reinforcement learning

A reinforcement learning project that trains a PPO agent to control a lunar lander using Gymnasium and Stable-Baselines3.The goal is to teach the agent to land safely by interacting with a simulated environment and learning from rewards and penalties.

Features

PPO training: Train an agent using Proximal Policy Optimization.
Custom reward shaping: Experiment with additional rewards and penalties to influence landing behavior.
Model evaluation: Evaluate the trained agent over 100 episodes and calculate mean, highest, and lowest rewards.
Visual simulation: Watch the trained agent control the lander in the Lunar Lander environment.
Model saving: Save trained models so they can be loaded later without training from scratch.

Technologies

1:Python
2:gymnasium
3:stable-baselines3
4:PPO(proximal policy optimization)
5:NumPy
6:Box2D

how to use:

1:Train the agent(if not trained)
    By running lunar_agent.py python script
2:Evaluate the trained agent
    by running evaluate_agent.py python script
3:To watch the simulation 
    run simulator.py python script

note:The training script must be run first if the required saved model does not exist

Results:
The V2 model achieved the following results during evaluation on the original LunarLander-v3 environment:

Metric                    Result

Evaluation episodes         100

Mean reward                237.67

Highest reward             295.33

Lowest reward               3.31

Learning Goals

Understand reinforcement learning through a practical environment
Explore PPO-based policy training
Experiment with reward shaping
Compare agent performance across evaluation episodes

Future Improvements

Measure landing success rate using explicit success criteria.
Compare the default reward system against custom reward shaping.
Experiment with different training configurations.
Improve evaluation and visualization tools.