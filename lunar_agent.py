#imports
import gymnasium as gym
from stable_baselines3 import PPO  
from pathlib import Path
from reward_system import CustomReward

base_env=gym.make("LunarLander-v3") #creates environment and store in base_env

# Apply our custom reward system
env = CustomReward(base_env)

Path("models").mkdir(exist_ok=True) #mkdir create a models directory and exist_ok = True prevents error if dir already exists
model_path=Path("models/lunar_ppo_v2.zip") 
if model_path.exists():
    print("loading existing model......")
    lunar_model=PPO.load(model_path,env=env)
   
else:
    print("Creating new model.....")
    lunar_model=PPO("MlpPolicy",env,verbose=1)
    


lunar_model.learn(total_timesteps=250_000)
'''100000 and 100_000 is identical python allows to use _ in nummeric strings 
to improve readability for large numbers'''

lunar_model.save(model_path)
print("model saved successfully!!!!")
env.close()