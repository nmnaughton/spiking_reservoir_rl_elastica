from tqdm import tqdm

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys
from reservoir_rl import ReservoirNetworkSimulator
sys.path.append(os.path.abspath(os.path.join('..', 'elastica')))
from set_environment import Environment

import matplotlib.pyplot as plt
import gym
from gym import spaces
import pandas as pd
from spinup import vpg_tf1 as vpg
from spinup import sac_tf1 as sac
from spinup.utils.run_utils import ExperimentGrid
from spinup.utils.test_policy import load_policy_and_env, run_policy

def get_elastica_env(collect_data_for_postprocessing=False):
    return Environment(
        final_time=elastica_sim_time,
        num_steps_per_update=50,
        number_of_control_points=num_control_points,
        alpha=75,
        beta=75,
        COLLECT_DATA_FOR_POSTPROCESSING=collect_data_for_postprocessing,
        mode=4,
        target_position=[-0.4, 0.6, 0.0],
        target_v=0.5,
        boundary=[-0.6, 0.6, 0.3, 0.9, -0.6, 0.6],
        E=1e7,
        sim_dt=2.0e-4,
        n_elem=20,
        NU=30,
        dim=dim, # 3.0,
        max_rate_of_change_of_activation=np.infty)


class CustomEnv(gym.Env):

    def __init__(self, dim, num_control_points, n_reservoir_neurons, collect_data_for_postprocessing=False):
        super(CustomEnv).__init__()

        self.dim = dim
        self.num_control_points = num_control_points
        self.n_reservoir_neurons = n_reservoir_neurons
        self.elastica_env = get_elastica_env(collect_data_for_postprocessing)
        self.collect_data_for_postprocessing = collect_data_for_postprocessing
        if self.collect_data_for_postprocessing:
            self.elastica_env.render()

        if self.dim == 2.0:
            # normal direction activation (2D)
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.num_control_points,),
                dtype=np.float64)
            self.action = np.zeros(self.num_control_points)
        if self.dim == 3.0:
            # normal and/or binormal direction activation (3D)
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(2 * self.num_control_points,),
                dtype=np.float64)
            self.action = np.zeros(2 * self.num_control_points)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_reservoir_neurons,),
            dtype=np.float64)

        self.num_steps = 0
        self.total_reward = 0.0

    def reset(self):
        reservoir_network_simulator.initialize_reservoir()
        rod_state = self.elastica_env.reset()
        reservoir_state = reservoir_network_simulator.simulate_network_vanilla(rod_state)
        self.num_steps = 0
        self.total_reward = 0.0
        return reservoir_state

    def step(self, action):
        rod_state, reward, done, info = self.elastica_env.step(action)
        reservoir_state = reservoir_network_simulator.simulate_network_vanilla(rod_state)
        self.num_steps += 1
        self.total_reward += reward

        if self.num_steps == 999:
            print(self.total_reward / self.num_steps)

            if self.collect_data_for_postprocessing:
                self.elastica_env.close()
                self.elastica_env.post_processing("video.mp4")

        return reservoir_state, reward, done, info

    def render(self, mode='human'):
        return

def get_custom_env(collect_data_for_postprocessing=False):
    return CustomEnv(dim, num_control_points, n_reservoir_neurons, collect_data_for_postprocessing)


def plot(title='Spiking Reservoir + Vanilla Policy Gradient'):
    # Plot Rewards vs. Episodes
    rewards = []

    with open('progress.txt', 'r') as fp:
        lines = fp.readlines()
        for i, line in enumerate(lines):
            if i == 0:
                continue
            words = line.split()
            reward = float(words[1]) / 1000.
            rewards.append(reward)

    window_size = 10
    rewards_series = pd.Series(rewards)
    windows = rewards_series.rolling(window_size)
    moving_avg_rewards_10 = windows.mean().tolist()[window_size - 1:]

    print(max(rewards))

    plt.plot(rewards)
    plt.plot(moving_avg_rewards_10)
    plt.yticks(np.arange(-1.0, 2.1, 0.1))
    plt.legend(['Episode Score', '10 Episode Moving Average'])
    plt.grid(True)
    plt.xlabel('Episode')
    plt.ylabel('Episode Score')
    plt.title(title)
    # plt.savefig(title + '.png')
    plt.show()

if __name__ == "__main__":
    # Reservoir parameters
    dim = 2.0
    num_control_points = 3
    input_size = 11 + num_control_points * int(dim - 1)
    output_size = num_control_points * int(dim - 1)
    n_reservoir_neurons = 512
    elastica_sim_time = 10
    nengo_sim_time = 0.01
    bounds = [-1, 1]
    num_elastica_timesteps = int(elastica_sim_time/nengo_sim_time)
    weights_size = n_reservoir_neurons * output_size

    reservoir_network_simulator = ReservoirNetworkSimulator(
        input_size = input_size,
        n_neurons = n_reservoir_neurons,
        output_size = output_size,
        sim_time = nengo_sim_time,
        bounds = bounds,
        action_calculation_method = "full",
        num_coeff_per_action = 5,
        n_reservoir_output_neurons = n_reservoir_neurons)

    # SpinningUp parameters
    # env_fn = lambda : get_custom_env() #get_elastica_env()
    # logger_kwargs = dict(output_dir='./', exp_name='spinning_up_sac')
    # ac_kwargs = dict(hidden_sizes=[])

    # SAC parameters
    # lr = 0.01
    # batch_size = num_elastica_timesteps
    # ac_kwargs = dict(hidden_sizes=[3])

    # Run SpinningUp reinforcement learning algorithm
    # with tf.Graph().as_default():
        # vpg(env_fn=env_fn, ac_kwargs=ac_kwargs, logger_kwargs=logger_kwargs, steps_per_epoch=num_elastica_timesteps, epochs=500, pi_lr=0.01, vf_lr=0.01, seed=0)
        # vpg(env_fn=env_fn, logger_kwargs=logger_kwargs, steps_per_epoch=num_elastica_timesteps, epochs=500, seed=0)
        # sac(env_fn=env_fn, ac_kwargs=ac_kwargs, logger_kwargs=logger_kwargs, steps_per_epoch=num_elastica_timesteps, epochs=500, lr=lr, num_test_episodes=1)


    # Generate video off of saved policy
    # _, get_action =  load_policy_and_env(os.path.abspath('./'))
    # env =  get_custom_env(collect_data_for_postprocessing=True) # get_elastica_env(collect_data_for_postprocessing=True)
    # run_policy(env, get_action)

    # Plot the rewards and moving average rewards vs. episodes
    # plot()

    # Hyperparameter search
    env_fn = lambda : get_elastica_env()
    ac_kwargs = dict()
    hidden_sizes = [[512], [256, 256], [128, 128, 128, 128]]
    activations = [tf.tanh, tf.nn.relu]
    activations_str = ["tanh", "relu"]
    learning_rates = [0.01, 0.001]
    seeds = [0]

    for seed in seeds:
        for hidden_size in hidden_sizes:
            for learning_rate in learning_rates:
                for i, activation in enumerate(activations):
                    ac_kwargs['hidden_sizes'] = hidden_size
                    ac_kwargs['activation'] = activation
                    activation_str = activations_str[i]
                    logger_kwargs = dict(output_dir='./', exp_name='vpg_hyperparam_test')

                    with tf.Graph().as_default():
                        vpg(env_fn=env_fn, ac_kwargs=ac_kwargs, logger_kwargs=logger_kwargs, steps_per_epoch=num_elastica_timesteps, epochs=1000, seed=seed, pi_lr=learning_rate, vf_lr=learning_rate)

                    exp_save_file = f"hidden_size_{hidden_size}_activation_{activation_str}_learning_rate_{learning_rate}_seed{seed}.txt"
                    with open("progress.txt") as f1:
                        with open(exp_save_file, "w") as f2:
                            for line in f1:
                                f2.write(line)