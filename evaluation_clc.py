import numpy as np
#from clc.clc import clc

from stable_baselines.common.vec_env import VecEnv


def evaluate_clc(clc_policy, env, n_eval_episodes=10, deterministic=True,
                    render=False, callback=None, reward_threshold=None,
                    return_episode_rewards=False):
    """
    Runs candidate for `n_eval_episodes` episodes and returns average reward.
    This is made to work only with one env.

    :param model: (BaseRLModel) The RL agent you want to evaluate.
    :param env: (gym.Env or VecEnv) The gym environment. In the case of a `VecEnv`
        this must contain only one environment.
    :param n_eval_episodes: (int) Number of episode to evaluate the agent
    :param deterministic: (bool) Whether to use deterministic or stochastic actions
    :param render: (bool) Whether to render the environment or not
    :param callback: (callable) callback function to do additional checks,
        called after each step.
    :param reward_threshold: (float) Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: (bool) If True, a list of reward per episode
        will be returned instead of the mean.
    :return: (float, float) Mean reward per episode, std of reward per episode
        returns ([float], [int]) when `return_episode_rewards` is True
    """
    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"

    obs_log = None
    a_log = None
    episode_rewards, episode_lengths = [], []
    for _ in range(n_eval_episodes):
        obs = env.reset()
        action = clc_policy.get_action(obs, reset=True, verbose=True)
        done, state = False, None
        episode_reward = 0.0
        episode_length = 0.0
        while not done:

            obs, reward, done, _info = env.step(action)
            if obs_log is None:
                obs_log = obs
            else:
                obs_log = np.vstack((obs_log, obs))

            if a_log is None:
                a_log = action
            else:
                a_log = np.vstack((a_log, action))
            episode_reward += reward
            if callback is not None:
                callback(locals(), globals())
            episode_length += 1
            if render:
                env.render()
            action = clc_policy.get_action(obs, reset=False, verbose=True)
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    if reward_threshold is not None:
        assert mean_reward > reward_threshold, 'Mean reward below threshold: '\
                                         '{:.2f} < {:.2f}'.format(mean_reward, reward_threshold)
    if return_episode_rewards:
        return episode_rewards, episode_lengths, obs_log, a_log
    return mean_reward, std_reward, obs_log, a_log
