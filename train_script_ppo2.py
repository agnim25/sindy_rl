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
import gym_ct_cartpole
from gym_ct_cartpole import CosineSineObservation
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
env = 'mountaincar'
settings = [(env, 'rk', 'swingupsafe', 'dm')]
setting_render = (env, 'rk', 'swingupsafe', 'dm')
def get_name(_env, _ode_sol, _task, _rew):
    return '{}_{}_{}_{}'.format(_ode_sol, _env, _task, _rew)
ver = 'v0'
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


for setting in settings:
    class SaveOnBestTrainingRewardCallback(BaseCallback):
        """
        Callback for saving a model (the check is done every ``check_freq`` steps)
        based on the training reward (in practice, we recommend using ``EvalCallback``).

        :param check_freq: (int)
        :param log_dir: (str) Path to the folder where the model will be saved.
          It must contains the file created by the ``Monitor`` wrapper.
        :param verbose: (int)
        """
        def __init__(self, check_freq: int, log_dir: str, verbose=1):
            super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
            self.check_freq = check_freq
            self.log_dir = log_dir
            self.save_path = os.path.join(log_dir, 'best_model_{}'.format(method))
            self.best_mean_reward = -np.inf

        def _init_callback(self) -> None:
            # Create folder if needed
            if self.save_path is not None:
                os.makedirs(self.save_path, exist_ok=True)

        def _on_step(self) -> bool:
            if self.n_calls % self.check_freq == 0:

              # Retrieve training reward
              x, y = ts2xy(load_results(self.log_dir), 'timesteps')
              if len(x) > 0:
                  # Mean training reward over the last 100 episodes
                  mean_reward = np.mean(y[-100:])
                  if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

                  # New best model, you could save the agent here
                  if mean_reward > self.best_mean_reward:
                      self.best_mean_reward = mean_reward
                      # Example for saving best model
                      if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                      self.model.save(self.save_path)

            return True

    env_name = get_name(*setting)
    # Create log dicd
    # r
    log_dir = "./{}_{}".format(env_name, ver)
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
    # env = make_vec_env('ct_cartpole-v0', n_envs=1)

    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)


    env = gym.make('ct_{}-{}'.format(env_name, ver))
    env.use_sindy = True
    env.model = model

    # env = CosineSineObservation(env)
    eval_env = gym.make('ct_{}-{}'.format(env_name, ver))
    eval_env.use_sindy = True
    eval_env.model = model
    # eval_env = CosineSineObservation(eval_env)
    # check_env(env)
    env = Monitor(env, train_dir)
    # multiprocess environment

    # callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    stoptraining_callback = StopTrainingOnRewardThreshold(reward_threshold=target_episodic_reward)
    #assert method in ('ppo', 'sac', 'ddpg', 'trpo', 'a2c', 'acktr'), "Method not recognized"
    callback = EvalCallback(eval_env, stoptraining_callback, n_eval_episodes=1, eval_freq=eval_freq, log_path=train_dir,
                            best_model_save_path=best_dir)



    # model = PPO1(MlpPolicy, env, gamma=0.99, timesteps_per_actorbatch=1024, clip_param=0.2, entcoeff=0.001,
    #              optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
    #              lam=0.95, verbose=1,
    #                   tensorboard_log=board_dir)

    # model = PPO2(MlpPolicy, env, gamma=0.99, n_steps=1024, cliprange=0.1, ent_coef=1.183272853635661e-07,
    #              noptepochs=20, learning_rate=1.82682356104873e-05, nminibatches=32,
    #              lam=0.98, verbose=1,
    #              tensorboard_log=board_dir)

    # model = PPO2(MlpPolicy, env, gamma=0.999, n_steps=128, cliprange=0.2, ent_coef=6.86971575620569e-05,
    #              noptepochs=20, learning_rate=0.0008481901362984788, nminibatches=1,
    #              lam=0.99, verbose=1,
    #              tensorboard_log=board_dir)

    # model = PPO2(MlpPolicy, env, gamma=0.999, n_steps=4096, cliprange=0.2, ent_coef=0.001,
    #              noptepochs=10, learning_rate=2.0e-04, nminibatches=8,
    #              lam=0.99, verbose=1,
    #              tensorboard_log=board_dir)

    # model = PPO2(MlpPolicy, env, gamma=0.999, n_steps=8192, cliprange=0.2, ent_coef=0.001,
    #              noptepochs=20, learning_rate=3.0e-04, nminibatches=8,
    #              lam=0.995, verbose=1,
    #              tensorboard_log=board_dir, seed=seed)  # ppo2_11

    # model = PPO2(MlpPolicy, env, gamma=0.999, n_steps=8192, cliprange=0.2, ent_coef=0.001,
    #              noptepochs=30, learning_rate=3.0e-04, nminibatches=8,
    #              lam=0.995, verbose=1,
    #              tensorboard_log=board_dir, seed=seed)  # ppo2_13 model_success_4M

    # model = PPO2(MlpPolicy, env, gamma=0.999, n_steps=8192, cliprange=0.2, ent_coef=0.00001,
    #              noptepochs=30, learning_rate=3.0e-04, nminibatches=8,
    #              lam=0.995, verbose=1, cliprange_vf=-1,
    #              tensorboard_log=board_dir, seed=seed) # ppo2_19   .._model_success_10M_1

    # model = PPO2(MlpPolicy, env, gamma=0.999, n_steps=8192, cliprange=0.2, ent_coef=0.000001,
    #              noptepochs=30, learning_rate=3.0e-04, nminibatches=8,
    #              lam=0.995, verbose=1, cliprange_vf=-1,
    #              tensorboard_log=board_dir, seed=seed) # ppo2_20   .._model_success_10M_2

    # lr_0 = 3.0e-04
    # lr_f = 3.0e-05
    # k = (lr_0-lr_f)/lr_f
    # def lr_schedule(percent_left):
    #     return lr_0*percent_left+(1-percent_left)*lr_f
    #
    # model = PPO2(MlpPolicy, env, gamma=0.999, n_steps=8192, cliprange=0.2, ent_coef=0.00001,
    #              noptepochs=30, learning_rate=lr_schedule, nminibatches=8,
    #              lam=0.995, verbose=1, cliprange_vf=-1,
    #              tensorboard_log=board_dir, seed=seed) # ppo2_23

    # lr_0 = 4.0e-04
    # lr_f = 1.0e-04
    # k = (lr_0-lr_f)/lr_f
    # def lr_schedule(percent_left):
    #     return lr_0*percent_left+(1-percent_left)*lr_f
    #
    # model = PPO2(MlpPolicy, env, gamma=0.999, n_steps=8192, cliprange=0.2, ent_coef=0.00001,
    #              noptepochs=30, learning_rate=lr_schedule, nminibatches=8,
    #              lam=0.995, verbose=1, cliprange_vf=-1,
    #              tensorboard_log=board_dir, seed=seed) # ppo2_24 success_10M_3

    # lr_0 = 5.0e-04
    # lr_f = 5.0e-05
    # k = (lr_0-lr_f)/lr_f
    # def lr_schedule(percent_left):
    #     return lr_0*percent_left+(1-percent_left)*lr_f
    #
    # model = PPO2(MlpPolicy, env, gamma=0.999, n_steps=8192, cliprange=0.2, ent_coef=0.00001,
    #              noptepochs=30, learning_rate=lr_schedule, nminibatches=8,
    #              lam=0.995, verbose=1, cliprange_vf=-1,
    #              tensorboard_log=board_dir, seed=seed) # ppo2_29

    # lr_0 = 4.0e-04
    # lr_01 = 1.0e-04
    # lr_f = 1.0e-04
    # lr_f1 = 5.0e-05
    # k = (lr_0-lr_f)/lr_f
    # def pwl_lr_schedule(percent_left):
    #     percent_done = 1 - percent_left
    #     if percent_done < 0.33:
    #         return lr_0*percent_left+(1-percent_left)*lr_f
    #     else:
    #         tot = 1 - 0.33
    #         lamb = (percent_done-0.33)/tot
    #
    #         return lr_01*(1-lamb) + lr_f1*lamb


    # lr_0 = 4.0e-04
    # lr_01 = 1.0e-04
    # lr_f = 1.0e-04
    # lr_f1 = 5.0e-05
    # k = (lr_0 - lr_f) / lr_f
    #
    #
    # def pwl_lr_schedule(percent_left):
    #     percent_done = 1 - percent_left
    #     if percent_done < 0.33:
    #         return 2.0e-04
    #     else:
    #         tot = 1 - 0.33
    #         lamb = (percent_done - 0.33) / tot
    #
    #         return lr_01 * (1 - lamb) + lr_f1 * lamb


    # model = PPO2(MlpPolicy, env, gamma=0.999, n_steps=8192, cliprange=0.2, ent_coef=0.001,
    #                           noptepochs=30, learning_rate=1.5e-4, nminibatches=8,
    #                           lam=0.995, verbose=1,
    #                           tensorboard_log=board_dir, seed=seed)  # ppo2_32 quick and smooth learning

    # lr_0 = 2.0e-04
    # lr_f = 2.0e-05
    # k = (lr_0-lr_f)/lr_f
    # def lr_schedule(percent_left):
    #     return lr_0*percent_left+(1-percent_left)*lr_f
    #
    # model = PPO2(MlpPolicy, env, gamma=0.999, n_steps=8192, cliprange=0.2, ent_coef=0.002,
    #                           noptepochs=30, learning_rate=lr_schedule, nminibatches=8,
    #                           lam=0.995, verbose=1,
    #                           tensorboard_log=board_dir, seed=seed)  # ppo2_36 quicker and smoother learning, but jiggly in high reward

    # lr_01 = 1.5e-04
    # lr_f1 = 1e-05
    # #k = (lr_0 - lr_f) / lr_f
    # def pwl_lr_schedule(percent_left):
    #     percent_done = 1 - percent_left
    #     if percent_done < 0.2:
    #         return 1.5e-04
    #     else:
    #         tot = 1 - 0.2
    #         lamb = (percent_done - 0.2) / tot
    #
    #         return lr_01 * (1 - lamb) + lr_f1 * lamb
    #
    # # def lr_schedule(percent_left):
    # #     return lr_0 * percent_left + (1 - percent_left) * lr_f
    #
    #
    # model = PPO2(MlpPolicy, env, gamma=0.999, n_steps=8192, cliprange=0.2, ent_coef=0.002,
    #              noptepochs=30, learning_rate=pwl_lr_schedule, nminibatches=8,
    #              lam=0.995, verbose=1,
    #              tensorboard_log=board_dir,
    #              seed=seed)  # ppo2_38 quicker learning, but jiggly in high reward


    # model = PPO2(MlpPolicy, env, gamma=0.999, n_steps=8192, cliprange=0.2, ent_coef=0.01,
    #              noptepochs=30, learning_rate=3e-5, nminibatches=8,
    #              lam=0.995, verbose=1,
    #              tensorboard_log=board_dir,
    #              seed=seed)
    #
    # model.learn(total_timesteps=tot_steps, callback=callback)
    # model.save(model_filename)  # ppo2_39
    #

    # lr_01 = 1.e-04
    # lr_f1 = 0.5e-05
    # #k = (lr_0 - lr_f) / lr_f
    # def pwl_lr_schedule(percent_left):
    #     percent_done = 1 - percent_left
    #     if percent_done < 0.2:
    #         return 1.5e-04
    #     else:
    #         tot = 1 - 0.2
    #         lamb = (percent_done - 0.2) / tot
    #
    #         return lr_01 * (1 - lamb) + lr_f1 * lamb
    #
    # model = PPO2(MlpPolicy, env, gamma=0.999, n_steps=8192, cliprange=0.2, ent_coef=0.01,
    #              noptepochs=40, learning_rate=pwl_lr_schedule, nminibatches=8,
    #              lam=0.995, verbose=1,
    #              tensorboard_log=board_dir,
    #              seed=seed) ## ppo2_41... jumpiness dcreases once leraningrate drops to 4e-5
    #                                       but doesn't go away entirely.. could be n_epochs fault
    # large entropy bonus isn't that bad



    # model = PPO2(MlpPolicy, env, gamma=0.999, n_steps=8192, cliprange=0.2, ent_coef=0.01,
    #              noptepochs=20, learning_rate=1.5e-4, nminibatches=8,
    #              lam=0.995, verbose=1,
    #              tensorboard_log=board_dir,
    #              seed=seed) #ppo2_43

    # model = PPO2(MlpPolicy, env, gamma=0.999, n_steps=8192, cliprange=0.2, ent_coef=0.001,
    #              noptepochs=20, learning_rate=1.5e-4, nminibatches=8,
    #              lam=0.995, verbose=1,
    #              tensorboard_log=board_dir,
    #              seed=seed) #ppo2_44... like ppo2_32 but with epoch = 20

    # model = PPO2(MlpPolicy, env, gamma=0.999, n_steps=8192, cliprange=0.2, ent_coef=0.001,
    #              noptepochs=24, learning_rate=1.5e-4, nminibatches=8,
    #              lam=0.995, verbose=1, cliprange_vf=-1,
    #              tensorboard_log=board_dir,
    #              seed=seed) #ppo2_45 ..  slightly bigger epochs to be faster and cliprange_vf to prevent noisiness?

    # model = PPO2(MlpPolicy, env, gamma=0.999, n_steps=8192, cliprange=0.2, ent_coef=0.001,
    #              noptepochs=24, learning_rate=1.5e-4, nminibatches=8,
    #              lam=0.995, verbose=1, cliprange_vf=-1,
    #              tensorboard_log=board_dir,
    #              seed=seed) # ppo2_46 = trying ppo2_32 again with cliprange_vf= -1

## CONCLUSION: cliprange_vf= -1 is most likely trash: ppo2_32 didn't work at all

    # model = PPO2(MlpPolicy, env, gamma=0.999, n_steps=8192, cliprange=0.2, ent_coef=0.001,
    #              noptepochs=25, learning_rate=1.5e-4, nminibatches=8,
    #              lam=0.995, verbose=1,
    #              tensorboard_log=board_dir,
    #              seed=seed) ## ppo2_47 ... like ppo2_32 but with epoch = 25

    # lr_01 = 1.e-04
    # lr_f1 = 0.5e-05
    # #k = (lr_0 - lr_f) / lr_f
    # def pwl_lr_schedule(percent_left):
    #     percent_done = 1 - percent_left
    #     if percent_done < 0.2:
    #         return 1.5e-04
    #     else:
    #         tot = 1 - 0.2
    #         lamb = (percent_done - 0.2) / tot
    #
    #         return lr_01 * (1 - lamb) + lr_f1 * lamb

    # model = PPO2(MlpPolicy, env, gamma=0.999, n_steps=8192, cliprange=0.2, ent_coef=0.001,
    #              noptepochs=30, learning_rate=1.5e-4, nminibatches=8,
    #              lam=0.995, verbose=1,
    #              tensorboard_log=board_dir,
    #              seed=seed) ## ppo2_48 is just like ppo2_32

    # model = PPO2(MlpPolicy, env, gamma=0.999, n_steps=8192, cliprange=0.2, ent_coef=0.001,
    #                           noptepochs=30, learning_rate=1.5e-4, nminibatches=8,
    #                           lam=0.995, verbose=1,
    #                           tensorboard_log=board_dir, seed=seed)  #ppo2_49 copy cmd of ppo2_32 quick and smooth learning

    # lr_01 = 2.0e-04
    # lr_f1 = 1e-06
    # #k = (lr_0 - lr_f) / lr_f
    # def pwl_lr_schedule(percent_left):
    #     percent_done = 1 - percent_left
    #     if percent_done < 0.5:
    #         return 2e-04
    #     else:
    #         tot = 1 - 0.5
    #         lamb = (percent_done - 0.5) / tot
    #
    #         return lr_01 * (1 - lamb) + lr_f1 * lamb
    #
    # model = PPO2(MlpPolicy, env, gamma=0.999, n_steps=8192, cliprange=0.2, ent_coef=0.0005,
    #              noptepochs=20, learning_rate=pwl_lr_schedule, nminibatches=8,
    #              lam=0.995, verbose=1,
    #              tensorboard_log=board_dir,
    #              seed=seed) #ppo2_51... like ppo2_44 but with less entropy and more lr

# even super small learning (down to 1e-5) rate didn't affect the jumpiness of the curve in high reward
#     lr_01 = 4.0e-04
#     lr_f1 = 1.0e-04
#     #k = (lr_0 - lr_f) / lr_f
#     def pwl_lr_schedule(percent_left):
#         percent_done = 1 - percent_left
#         if percent_done < 0.5:
#             return 4e-04
#         else:
#             tot = 1 - 0.5
#             lamb = (percent_done - 0.5) / tot
#
#             return lr_01 * (1 - lamb) + lr_f1 * lamb
#
#     model = PPO2(MlpPolicy, env, gamma=0.999, n_steps=8192, cliprange=0.2, ent_coef=0.001,
#                  noptepochs=10, learning_rate=pwl_lr_schedule, nminibatches=8,
#                  lam=0.995, verbose=1,
#                  tensorboard_log=board_dir,
#                  seed=seed) #ppo2_52... trying out smaller epoch and higher learning rate in comparison to ppo2_51
#     lr_01 = 8.0e-04
#     lr_f1 = 1.0e-04
#     #k = (lr_0 - lr_f) / lr_f
#     def pwl_lr_schedule(percent_left):
#         percent_done = 1 - percent_left
#         if percent_done < 0.5:
#             return 8e-04
#         else:
#             tot = 1 - 0.5
#             lamb = (percent_done - 0.5) / tot
#
#             return lr_01 * (1 - lamb) + lr_f1 * lamb
#
#     model = PPO2(MlpPolicy, env, gamma=0.999, n_steps=8192, cliprange=0.2, ent_coef=0.01,
#                  noptepochs=10, learning_rate=pwl_lr_schedule, nminibatches=8,
#                  lam=0.995, verbose=1,
#                  tensorboard_log=board_dir,
#                  seed=seed) #ppo2_54... trying out smaller epoch and higher learning rate in comparison to ppo2_51
# ppo2_52 learned

# best_10Msuccess hyperparameters are:
#     lr_01 = 8.0e-04
#     lr_f1 = 1.0e-04
#     #k = (lr_0 - lr_f) / lr_f
#     def pwl_lr_schedule(percent_left):
#         percent_done = 1 - percent_left
#         if percent_done < 0.5:
#             return 8e-04
#         else:
#             tot = 1 - 0.5
#             lamb = (percent_done - 0.5) / tot
#
#             return lr_01 * (1 - lamb) + lr_f1 * lamb
#
#     model = PPO2(MlpPolicy, env, gamma=0.999, n_steps=8192, cliprange=0.2, ent_coef=0.001,
#                  noptepochs=10, learning_rate=pwl_lr_schedule, nminibatches=8,
#                  lam=0.995, verbose=1,
#                  tensorboard_log=board_dir,
#                  seed=seed) #ppo2_55... like ppo2_54 but less entropy

    # best learning behavior so far ... learns within 2M but still jiggly

    # ppo2_52 learned
    # lr_01 = 1.5e-03
    # lr_f1 = 1.0e-04
    #
    #
    # # k = (lr_0 - lr_f) / lr_f
    # def pwl_lr_schedule(percent_left):
    #     percent_done = 1 - percent_left
    #     if percent_done < 0.5:
    #         return 1.5e-03
    #     else:
    #         tot = 1 - 0.5
    #         lamb = (percent_done - 0.5) / tot
    #
    #         return lr_01 * (1 - lamb) + lr_f1 * lamb
    #
    #
    # model = PPO2(MlpPolicy, env, gamma=0.999, n_steps=8192, cliprange=0.2, ent_coef=0.0001,
    #              noptepochs=6, learning_rate=pwl_lr_schedule, nminibatches=8,
    #              lam=0.995, verbose=1,
    #              tensorboard_log=board_dir,
    #              seed=seed)  # ppo2_63... like ppo2_55 but less epochs more learning_rate

    # lr_01 = 8.0e-04
    # lr_f1 = 1.0e-04
    # #k = (lr_0 - lr_f) / lr_f
    # def pwl_lr_schedule(percent_left):
    #     percent_done = 1 - percent_left
    #     if percent_done < 0.5:
    #         return lr_01
    #     else:
    #         tot = 1 - 0.5
    #         lamb = (percent_done - 0.5) / tot
    #
    #         return lr_01 * (1 - lamb) + lr_f1 * lamb
    #
    # model = PPO2(MlpPolicy, env, gamma=0.999, n_steps=8192, cliprange=0.2, ent_coef=0.0005,
    #              noptepochs=10, learning_rate=pwl_lr_schedule, nminibatches=8,
    #              lam=0.995, verbose=1,
    #              tensorboard_log=board_dir,
    #              seed=seed) #ppo2_64... like ppo2_55 but less entropy

    # lr_01 = 8e-04
    # lr_f1 = 0.5e-04
    # #k = (lr_0 - lr_f) / lr_f
    # def pwl_lr_schedule(percent_left, rew):
    #     percent_done = 1 - percent_left
    #     if percent_done < 0.5:
    #         return lr_01
    #     else:
    #         tot = 1 - 0.5
    #         lamb = (percent_done - 0.5) / tot
    #
    #         #return lr_01 * (1 - lamb) + lr_f1 * lamb
    #         return lr_f1

# #   PPO_32 on swingupsafe also named 6Msuccess_custom_ADAM
# #   fast learning, but still jumpy.. learning rate needs to be below 1.5e-3 for stuff to work
#
#     rew_v = [700, 1300, 1400, 1500, 1600, 1800, 2001]
#     lr_v = [1.5e-3, 1.5e-3, 1e-4, 1e-4, 5e-5, 5e-5, 5e-5]
#
#     #k = (lr_0 - lr_f) / lr_f
#     def lr_rew_schedule(percent_left, rew):
#         for (r, lr) in zip(rew_v, lr_v):
#             if rew < r:
#                 return lr
#         return 0.0 # This should never happen anyway...
#
#
#     model = PPO2(MlpPolicy, env, gamma=0.999, n_steps=8192, cliprange=0.2, ent_coef=0.001,
#                  noptepochs=10, learning_rate=lr_rew_schedule, nminibatches=8,
#                  lam=0.995, verbose=1,
#                  tensorboard_log=board_dir,
#                  seed=seed) # like bset10Msuccess but with learning_rate - reward coupling
# # You need smaller learning rate for learning... 1.5e-3 tops

    # rew_v = [700, 1300, 1400, 1500, 1600, 1800, 2001]
    # lr_v = [1e-3, 1e-3, 1e-4, 1e-4, 5e-5, 5e-5, 5e-5]
    #
    #
    # # k = (lr_0 - lr_f) / lr_f
    # def lr_rew_schedule(percent_left, rew):
    #     for (r, lr) in zip(rew_v, lr_v):
    #         if rew < r:
    #             return lr
    #     return 0.0  # This should never happen anyway...
    #
    #
    # model = PPO2(MlpPolicy, env, gamma=0.999, n_steps=8192, cliprange=0.2, ent_coef=0.001,
    #              noptepochs=10, learning_rate=lr_rew_schedule, nminibatches=8,
    #              lam=0.995, verbose=1,
    #              tensorboard_log=board_dir,
    #              seed=seed)  # #ppo_34 worked nicely with lr = 1e-3

    # rew_v = [700, 1300, 1400, 1500, 1600, 1800, 2001]
    # lr_v = [1e-3, 1e-3, 1e-4, 1e-4, 5e-5, 5e-5, 5e-5]
    #
    #
    # # k = (lr_0 - lr_f) / lr_f
    # def lr_rew_schedule(percent_left, rew):
    #     for (r, lr) in zip(rew_v, lr_v):
    #         if rew < r:
    #             return lr
    #     return 0.0  # This should never happen anyway...
    #
    #
    # model = PPO2(MlpPolicy, env, gamma=0.999, n_steps=8192, cliprange=0.2, ent_coef=0.0001,
    #              noptepochs=10, learning_rate=lr_rew_schedule, nminibatches=8,
    #              lam=0.995, verbose=1,
    #              tensorboard_log=board_dir,
    #              seed=seed)  # #ppo_35 worked nicely and shows with lr = 1e-3 entropy can be reduced

    # rew_v = [700, 1300, 1400, 1500, 1600, 1800, 2001]
    # lr_v = [1e-3, 1e-3, 1e-4, 1e-4, 1e-4, 1e-4, 5e-5]
    #
    #
    # # k = (lr_0 - lr_f) / lr_f
    # def lr_rew_schedule(percent_left, rew):
    #     for (r, lr) in zip(rew_v, lr_v):
    #         if rew < r:
    #             return lr
    #     return 0.0  # This should never happen anyway...
    #
    #
    # model = PPO2(MlpPolicy, env, gamma=0.999, n_steps=8192, cliprange=0.2, ent_coef=0.0001,
    #              noptepochs=6, learning_rate=lr_rew_schedule, nminibatches=8,
    #              lam=0.995, verbose=1,
    #              tensorboard_log=board_dir,
    #              seed=seed) # it did learn but weirdly jumpy and slow ppo2_37

    # rew_v = [700, 1300, 1400, 1500, 1600, 1800, 2001]
    # lr_v = [1.5e-3, 1.5e-3, 1e-4, 1e-4, 1e-4, 1e-4, 5e-5]
    #
    # # k = (lr_0 - lr_f) / lr_f
    # def lr_rew_schedule(percent_left, rew):
    #     for (r, lr) in zip(rew_v, lr_v):
    #         if rew < r:
    #             return lr
    #     return 0.0  # This should never happen anyway...
    #
    #
    # model = PPO2(MlpPolicy, env, gamma=0.999, n_steps=8192, cliprange=0.2, ent_coef=0.0001,
    #              noptepochs=6, learning_rate=lr_rew_schedule, nminibatches=8,
    #              lam=0.995, verbose=1,
    #              tensorboard_log=board_dir,
    #              seed=seed) #epochs down doesn't change shit... it's still jumpy ppo2_37

    # rew_v = [700, 1300, 1400, 1500, 1600, 1800, 2001]
    # lr_v = [1e-3, 1e-3, 1e-4, 1e-4, 1e-4, 1e-4, 5e-5]
    #
    # # k = (lr_0 - lr_f) / lr_f
    # def lr_rew_schedule(percent_left, rew):
    #     for (r, lr) in zip(rew_v, lr_v):
    #         if rew < r:
    #             return lr
    #     return 0.0  # This should never happen anyway...
    #
    #
    # model = PPO2(MlpPolicy, env, gamma=0.999, n_steps=8192, cliprange=0.2, ent_coef=0.000001,
    #              noptepochs=10, learning_rate=lr_rew_schedule, nminibatches=8,
    #              lam=0.995, verbose=1,
    #              tensorboard_log=board_dir,
    #              seed=seed)
    #
    # model.learn(total_timesteps=tot_steps, callback=callback)
    # model.save(model_filename)

    # rew_v = [700, 1300, 1400, 1500, 1600, 1800, 2001]
    # lr_v = [1e-3, 1e-3, 1e-4, 1e-4, 1e-4, 1e-4, 5e-5]

    # # k = (lr_0 - lr_f) / lr_f
    # def lr_rew_schedule(percent_left, rew=0):
    #     for (r, lr) in zip(rew_v, lr_v):
    #         if rew < r:
    #             return lr
    #     return 0.0  # This should never happen anyway...


    # model = PPO2(MlpPolicy, env, gamma=0.999, n_steps=8192, cliprange=0.2, ent_coef=0.000001,
    #              noptepochs=10, learning_rate=lr_rew_schedule, nminibatches=8,
    #              lam=0.995, verbose=1,
    #              tensorboard_log=board_dir,
    #              seed=seed) #highest consistent reward ppo2_52 on swingup

    # model.learn(total_timesteps=tot_steps, callback=callback)
    # model.save(model_filename)

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
model.save("ddpg_cartpole")


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
