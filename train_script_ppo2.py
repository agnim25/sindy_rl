import gym
import numpy as np
import os
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.sac.policies import MlpPolicy as MlpPolicySAC
from stable_baselines.ddpg.policies import MlpPolicy as MlpPolicyDDPG
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from stable_baselines import PPO1
from stable_baselines import SAC
from stable_baselines import DDPG
from stable_baselines import TRPO
from stable_baselines import A2C
from stable_baselines import ACKTR
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
import gym_custom
from gym_custom import CosineSineObservation
from stable_baselines.common.env_checker import check_env
from stable_baselines import results_plotter
from stable_baselines.common.evaluation import evaluate_policy



from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.common.noise import AdaptiveParamNoiseSpec, NormalActionNoise
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.callbacks import StopTrainingOnRewardThreshold
from stable_baselines.common.callbacks import EvalCallback

import dill as pickle

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# methods = ['trpo', 'acktr', 'sac', 'a2c', 'ppo', 'ddpg']

method = 'ddpg'
env_name = 'swimmer'
use_sindy = True

settings_cartpole = [('rk', 'swingupsafe', 'dm')]
setting_render_cartpole = ('rk', 'swingupsafe', 'dm')
ver = 'v0'
def get_cartpole_name(_ode_sol, _task, _rew):
    return '{}_cartpole_{}_{}'.format(_ode_sol, _task, _rew)

if env_name == 'cartpole':
    env_name = get_cartpole_name(*settings_cartpole)
    env = gym.make('ct_{}-{}'.format(env_name, ver))
else:
    env = gym.make(env_name)

log_dir = "./{}_{}".format(env_name, ver)
if use_sindy:
    env.use_sindy_model()
    # log_dir += '_sindy'

best_dir = os.path.join(log_dir, 'best_model_{}'.format(method))
train_dir = os.path.join(log_dir, 'log_{}'.format(method))
board_dir = os.path.join(log_dir, 'tb_{}'.format(method))
last_dir = os.path.join(log_dir, 'last_model_{}'.format(method))
model_filename = os.path.join(last_dir, 'last_model')
os.makedirs(log_dir, exist_ok=True)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(best_dir, exist_ok=True)
os.makedirs(board_dir, exist_ok=True)
os.makedirs(last_dir, exist_ok=True)

tot_steps = 100000
eval_freq = 1000
target_episodic_reward = 2000
seed_1 = 28549323
seed_1 = 5888
seed_1 = 25934

seed = seed_1
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# tensorboard --logdir ./a2c_cartpole_tensorboard/; ./ppo2_cartpole_tensorboard/
# tensorboard --logdir ./acktr_tensorboard
best_dir = None

"""
SAC
"""
# model = SAC(MlpPolicySAC, env_gym, verbose=1)
# model.learn(total_timesteps=50000)
# model.save("sac_cartpole")


"""
DDPG 
"""
# the noise objects for DDPG
n_actions = env.action_space.shape[-1]
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

model = DDPG(MlpPolicyDDPG, env, verbose=1, param_noise=param_noise, action_noise=action_noise, tensorboard_log=board_dir)
model.learn(total_timesteps=400000)
model.save("ddpg_mountaincar_sindy")



"""
TRPO 
"""
# model = TRPO(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=100000)
# model.save("trpo_cartpole")


"""
A2C
"""
# model = A2C(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=100000)
# model.save("a2c_cartpole")


"""
ACKTR
"""
# model = ACKTR(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=200000)
# model.save("acktr_cartpole")


# del model # remove to demonstrate saving and loading

# model = PPO2.load("ppo2_CartPolev1")
# model = PPO2.load("ppo2_cartpole_swingup")
# model = SAC.load("sac_cartpole")
# model = DDPG.load("ddpg_cartpole")
# model = TRPO.load("trpo_cartpole")
# model = A2C.load("a2c_cartpole")
# model = ACKTR.load("acktr_cartpole")

# Enjoy trained agent
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render('human')
