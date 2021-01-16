from tqdm import tqdm

import logging
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
import nengo_loihi
from nxsdk.logutils.nxlogging import set_verbosity, LoggingLevel
set_verbosity(LoggingLevel.WARNING)
import pandas as pd
import pybulletgym
from spinup import vpg_tf1 as vpg
from spinup import trpo_tf1 as trpo
from spinup import ppo_tf1 as ppo
from spinup import ddpg_tf1 as ddpg
from spinup import sac_tf1 as sac
from spinup.utils.run_utils import ExperimentGrid
from spinup.utils.test_policy import load_policy_and_env, run_policy
from spinup.utils.mpi_tools import mpi_fork

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

def get_pendulum_env():
    return gym.make('InvertedPendulumPyBulletEnv-v0')

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
        # print(reward)
        if done:
            print("num_steps: ", self.num_steps)
            print("total_reward: ", self.total_reward)
            avg_reward = self.total_reward / self.num_steps
            print("avg_reward: ", avg_reward)
            print('\n')

            with open("logging.txt", "a") as f:
                f.write(str(avg_reward) + '\n')

            if self.collect_data_for_postprocessing:
                self.elastica_env.close()
                self.elastica_env.post_processing("video.mp4")

            reward = self.total_reward / self.num_steps

        return reservoir_state, reward, done, info

    def render(self, mode='human'):
        return

class CustomPendulumEnv(gym.Env):

    def __init__(self, collect_data_for_postprocessing=False):
        super(CustomEnv).__init__()

        self.pendulum_env = gym.make('InvertedPendulumPyBulletEnv-v0')

        self.collect_data_for_postprocessing = collect_data_for_postprocessing
        if self.collect_data_for_postprocessing:
            self.pendulum_env.render()

        self.pendulum_env.reset()

        input_size = 5 # pendulum_env observation_space
        output_size = 1 # pendulum_env action_space

        self.n_reservoir_neurons = 512
        self.num_steps = 0
        self.total_reward = 0.0

        self.reservoir_network_simulator = ReservoirNetworkSimulator(
            input_size = input_size,
            n_neurons = n_reservoir_neurons,
            output_size = output_size,
            sim_time = 0.01,
            action_calculation_method = "full")

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float64)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(n_reservoir_neurons,), dtype=np.float64)

    def reset(self):
        self.reservoir_network_simulator.initialize_reservoir()
        pendulum_state = self.pendulum_env.reset()
        reservoir_state = self.reservoir_network_simulator.simulate_network_vanilla(pendulum_state)

        self.num_steps = 0
        self.total_reward = 0.0

        return reservoir_state

    def step(self, action):
        pendulum_state, reward, done, info = self.pendulum_env.step(action)
        reservoir_state = self.reservoir_network_simulator.simulate_network_vanilla(pendulum_state)

        self.num_steps += 1
        self.total_reward += reward

        if done and self.collect_data_for_postprocessing:
            self.pendulum_env.close()

        return reservoir_state, reward, done, info

    def render(self, mode='human'):
        return

def get_custom_elastica_env(collect_data_for_postprocessing=False):
    return CustomEnv(dim, num_control_points, n_reservoir_neurons, collect_data_for_postprocessing)

def get_custom_pendulum_env(collect_data_for_postprocessing=False):
    return CustomPendulumEnv(collect_data_for_postprocessing)

def plot(title='Spiking Reservoir + Vanilla Policy Gradient', plot_min_max=False, batch_size=4000):
    # Plot Rewards vs. Episodes
    rewards = []

    with open('progress.txt', 'r') as fp:
        lines = fp.readlines()
        for i, line in enumerate(lines):
            if i == 0:
                continue
            words = line.split()

            if plot_min_max:
                rewards.append(float(words[3])/ 1000.)
                rewards.append(float(words[4])/ 1000.)
            else:
                rewards.append(float(words[1])/ 1000.)

    window_size = 10
    rewards_series = pd.Series(rewards)
    windows = rewards_series.rolling(window_size)
    moving_avg_rewards_10 = windows.mean().tolist()[window_size - 1:]

    print('max_reward:', max(rewards))

    x_vals = []
    for i in range(0, len(rewards)):
        x_vals.append(i / 1000 * (batch_size / 1000))

    plt.plot(x_vals, rewards)
    plt.plot(x_vals[9:], moving_avg_rewards_10)
    plt.yticks(np.arange(-1.0, 2.1, 0.1))
    plt.legend(['Episode Score', '10 Episode Moving Average'])
    plt.grid(True)
    plt.xlabel('Elastica Timestep (in millions)')
    plt.ylabel('Episode Score')
    # plt.title(title)
    plt.show()

def plot_pendulum():
    # Plot Rewards vs. Episodes
    rewards = []

    with open('progress.txt', 'r') as fp:
        lines = fp.readlines()
        for i, line in enumerate(lines):
            if i == 0:
                continue
            words = line.split()
            rewards.append(float(words[1]))

    window_size = 10
    rewards_series = pd.Series(rewards)
    windows = rewards_series.rolling(window_size)
    moving_avg_rewards_10 = windows.mean().tolist()[window_size - 1:]

    print(max(rewards))

    plt.plot(rewards)
    plt.plot(moving_avg_rewards_10)
    plt.yticks(np.arange(-100.0, 1100., 100))
    plt.legend(['Episode Score', '10 Episode Moving Average'])
    plt.grid(True)
    # plt.xlabel('Episode')
    plt.xlabel('Epoch')
    plt.ylabel('Episode Score')
    # plt.title(title)
    # plt.title('spiking_reservoir_512_learning_rate_0.01_seed_0_batch_size_2000_gradient_global_norm_clipping_0.00001_on_all_tensors')
    # plt.savefig(title + '.png')
    plt.show()

def plot_from_logging_file():
    legend = []

    with open('logging.txt', 'r') as f:
    # with open('spiking_reservoir_512_learning_rate_0.01_seed_0_batch_size_baseline_test.txt', 'r') as f:
        lines = f.readlines()
        rewards = []

        for i, line in enumerate(lines):
            if 'exp_name' in line:
                exp_name = line[10:]
                legend.append(exp_name)

                if len(rewards) > 0:
                    window_size = 10
                    rewards_series = pd.Series(rewards)
                    windows = rewards_series.rolling(window_size)
                    moving_avg_rewards_10 = windows.mean().tolist()[window_size - 1:]
                    plt.plot(moving_avg_rewards_10)
                    rewards = []

            else:
                rewards.append(float(line))

        window_size = 10
        rewards_series = pd.Series(rewards)
        windows = rewards_series.rolling(window_size)
        moving_avg_rewards_10 = windows.mean().tolist()[window_size - 1:]
        plt.plot(moving_avg_rewards_10)
        rewards = []

    plt.yticks(np.arange(-1.0, 2.1, 0.1))
    plt.grid(True)
    plt.legend(legend, prop={"size":5})
    plt.xlabel('Episode')
    plt.ylabel('Elastica Timestep')
    plt.show()

def plot_from_rewards_dir(pendulum_rewards=False):
    reward_dir = './rewards/'
    rewards = []
    moving_avg_rewards = []
    legend = []

    x_axis = []
    for i in range(10, 1001):
        x_axis.append(i)

    NUM_COLORS = 8 # 20

    cm = plt.get_cmap('gist_rainbow')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.set_prop_cycle(color = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])

    for file_name in os.listdir(reward_dir):
        legend.append(file_name.replace('.txt', '') + '_avg')
        # legend.append(file_name.replace('.txt', ''))
        path = os.path.join(reward_dir, file_name)

        current_rewards = []
        with open(path, 'r') as fp:
            lines = fp.readlines()
            for i, line in enumerate(lines):
                if i == 0:
                    continue
                words = line.split()
                reward = float(words[1]) # / 1000.
                if not pendulum_rewards:
                    reward /= 1000.
                current_rewards.append(reward)

        window_size = 10
        rewards_series = pd.Series(current_rewards)
        windows = rewards_series.rolling(window_size)
        current_moving_avg_rewards_10 = windows.mean().tolist()[window_size - 1:]

        rewards.append(current_rewards)
        moving_avg_rewards.append(current_moving_avg_rewards_10)

        plt.plot(x_axis[0:len(current_moving_avg_rewards_10)], current_moving_avg_rewards_10, alpha=0.7)
        # plt.plot(list(range(len(current_rewards))), current_rewards, alpha=0.7)
        # plt.plot(current_rewards, alpha=0.7)

    if pendulum_rewards:
        plt.yticks(np.arange(-100.0, 1100., 100))
        plt.xlabel('Episode')
    else:
        plt.yticks(np.arange(-1.0, 2.1, 0.1))
        plt.xlabel('Elastica timestep (in thousands)')

    plt.grid(True)
    plt.legend(legend, prop={"size":8})
    plt.ylabel('Episode Score')
    # plt.ylabel('Episode Score (avg. over 1000 Elastica timesteps per episode)')
    # plt.title('Spiking Reservoir + VPG on inverted pendulum varing beta1')
    # plt.title('Spiking Reservoir + VPG vs Solo VPG Hyperparameter Tuned')
    plt.show()

def seed_test():
    env_fn = lambda : get_custom_env()
    ac_kwargs = dict(hidden_sizes=[])

    seeds = [0, 1, 2, 3, 4]
    for seed in seeds:
        logger_kwargs = dict(output_dir='./', exp_name='spiking_reservoir_512_learning_rate_0.01_seed_{seed}_baselie')

        with tf.Graph().as_default():
            vpg(env_fn=env_fn, ac_kwargs=ac_kwargs, logger_kwargs=logger_kwargs, steps_per_epoch=num_elastica_timesteps, epochs=1000, pi_lr=0.01, vf_lr=0.01, seed=seed)

            exp_save_file = "spiking_reservoir_512_learning_rate_0.01_seed_"+ str(seed) +"_baseline.txt"
            with open("progress.txt") as f1:
                with open(exp_save_file, "w") as f2:
                    for line in f1:
                        f2.write(line)

def batch_size_test():
    # Run batch size experiment
    env_fn = lambda : get_custom_env()
    ac_kwargs = dict(hidden_sizes=[])

    # batch_size_multipliers = [1, 2, 3, 4, 8, 16, 32, 64, 128]
    batch_size_multipliers = [1]

    for multiplier in batch_size_multipliers:

        batch_size = int(multiplier * num_elastica_timesteps)
        episodes = int(1000 / multiplier) + 1
        exp_name = 'spiking_reservoir_512_learning_rate_0.01_seed_0_batch_size_' + str(batch_size) + '_gradient_global_norm_clipping_0.00001_on_all_tensors'

        with open("logging.txt", "a") as f:
            f.write("exp_name: " + exp_name + "\n")

        logger_kwargs = dict(output_dir='./', exp_name=exp_name)

        with tf.Graph().as_default():
            vpg(env_fn=env_fn, ac_kwargs=ac_kwargs, logger_kwargs=logger_kwargs, steps_per_epoch=batch_size, epochs=episodes, pi_lr=0.01, vf_lr=0.01, seed=0)

    plot_from_logging_file()

def train_on_pendulum():
    # SpinningUp parameters
    env_fn = lambda : get_custom_pendulum_env()
    logger_kwargs = dict(output_dir='./', exp_name='inverted_pendulum_spiking_reservoir_512_learning_rate_0.01_exponential_decay_seed_0')
    ac_kwargs = dict(hidden_sizes=[])

    # Run SpinningUp reinforcement learning algorithm
    with tf.Graph().as_default():
        vpg(env_fn=env_fn, ac_kwargs=ac_kwargs, logger_kwargs=logger_kwargs, steps_per_epoch=4000, epochs=500, pi_lr=0.01, vf_lr=0.01, seed=0)

def train_on_pendulum_baseline_rl_algos():
    env_fn = lambda : get_pendulum_env()
    steps_per_epoch = 4000
    epochs = 500

    # VPG
    with tf.Graph().as_default():
        logger_kwargs = dict(output_dir='./vpg', exp_name='inverted_pendulum_vpg')
        vpg(env_fn=env_fn, logger_kwargs=logger_kwargs, steps_per_epoch=steps_per_epoch, epochs=epochs)

    # TRPO
    with tf.Graph().as_default():
        logger_kwargs = dict(output_dir='./trpo', exp_name='inverted_pendulum_trpo')
        trpo(env_fn=env_fn, logger_kwargs=logger_kwargs, steps_per_epoch=steps_per_epoch, epochs=epochs)

    # PPO
    with tf.Graph().as_default():
        logger_kwargs = dict(output_dir='./ppo', exp_name='inverted_pendulum_ppo')
        trpo(env_fn=env_fn, logger_kwargs=logger_kwargs, steps_per_epoch=steps_per_epoch, epochs=epochs)

    # DDPG
    with tf.Graph().as_default():
        logger_kwargs = dict(output_dir='./ddpg', exp_name='inverted_pendulum_ddpg')
        ddpg(env_fn=env_fn, logger_kwargs=logger_kwargs, steps_per_epoch=steps_per_epoch, epochs=epochs)

    # SAC
    with tf.Graph().as_default():
        logger_kwargs = dict(output_dir='./sac', exp_name='inverted_pendulum_sac')
        sac(env_fn=env_fn, logger_kwargs=logger_kwargs, steps_per_epoch=steps_per_epoch, epochs=epochs)

    # Spiking Reservoir 512 + VPG
    env_fn = lambda : get_custom_pendulum_env()

    with tf.Graph().as_default():
        logger_kwargs = dict(output_dir='./spiking_reservoir_vpg', exp_name='inverted_pendulum_spiking_reservoir_512_learning_rate_0.01')
        ac_kwargs = dict(hidden_sizes=[])
        vpg(env_fn=env_fn, ac_kwargs=ac_kwargs, logger_kwargs=logger_kwargs, steps_per_epoch=steps_per_epoch, epochs=epochs, pi_lr=0.01, vf_lr=0.01, seed=0)

if __name__ == "__main__":
    nengo_loihi.set_defaults()

    # Reservoir parameters
    dim = 2.0
    num_control_points = 3
    input_size = 11 + num_control_points * int(dim - 1)
    output_size = num_control_points * int(dim - 1)
    n_reservoir_neurons = 512
    elastica_sim_time = 10 # 5
    nengo_sim_time = 0.01
    num_elastica_timesteps = int(elastica_sim_time/nengo_sim_time)
    weights_size = n_reservoir_neurons * output_size

    reservoir_network_simulator = ReservoirNetworkSimulator(
        input_size = input_size,
        n_neurons = n_reservoir_neurons,
        output_size = output_size,
        sim_time = nengo_sim_time,
        action_calculation_method = "full")

    # train_on_pendulum()
    # train_on_pendulum_baseline_rl_algos()

    # seed_test()
    # plot_from_logging_file()
    # batch_size_test()
    # plot_from_logging_file()

    # SpinningUp parameters
    env_fn = lambda : get_custom_elastica_env()
    logger_kwargs = dict(output_dir='./', exp_name='spiking_reservoir_512_learning_rate_0.01_seed_0_batch_size_1000_gradient_global_norm_clipping_0.00001_on_all_tensors_epochs_500')
    ac_kwargs = dict(hidden_sizes=[])

    # Run SpinningUp reinforcement learning algorithm
    with tf.Graph().as_default():
        vpg(env_fn=env_fn, ac_kwargs=ac_kwargs, logger_kwargs=logger_kwargs, steps_per_epoch=num_elastica_timesteps, epochs=1, pi_lr=0.01, vf_lr=0.01, seed=0)

        # Run w/ and w/o grad clipping.
        # vpg(env_fn=env_fn, ac_kwargs=ac_kwargs, logger_kwargs=logger_kwargs, steps_per_epoch=3 * num_elastica_timesteps, epochs=500, pi_lr=0.01, vf_lr=0.01, seed=0)

        # Need to divide all logged average rewards by 500! AND CHANGE num_elastica_timesteps to = 5
        # vpg(env_fn=env_fn, ac_kwargs=ac_kwargs, logger_kwargs=logger_kwargs, steps_per_epoch=num_elastica_timesteps, epochs=2000, pi_lr=0.01, vf_lr=0.01, seed=0)

        # vpg(env_fn=env_fn, logger_kwargs=logger_kwargs, steps_per_epoch=num_elastica_timesteps, epochs=500, seed=0)

    # Generate video off of saved policy
    # _, get_action =  load_policy_and_env(os.path.abspath('./spiking_reservoir_vpg'), itr=1263)
    # env = get_custom_elastica_env(collect_data_for_postprocessing=True) # get_custom_pendulum_env(collect_data_for_postprocessing=True) #
    # run_policy(env, get_action)

    # Plot the rewards and moving average rewards vs. episodes
    # plot()
    # plot_pendulum()
    # plot_from_rewards_dir(pendulum_rewards=True)

    # # Hyperparameter search
    # env_fn = lambda : get_elastica_env()
    # ac_kwargs = dict()
    # hidden_sizes = [[128, 128, 128, 128]]
    # activations = [tf.tanh, tf.nn.relu]
    # activations_str = ["tanh", "relu"]
    # learning_rates = [0.01]
    # seeds = [0]
    #
    # for seed in seeds:
    #     for hidden_size in hidden_sizes:
    #         for learning_rate in learning_rates:
    #             for i, activation in enumerate(activations):
    #                 ac_kwargs['hidden_sizes'] = hidden_size
    #                 ac_kwargs['activation'] = activation
    #                 activation_str = activations_str[i]
    #                 logger_kwargs = dict(output_dir='./', exp_name='vpg_hyperparam_test')
    #
    #                 with tf.Graph().as_default():
    #                     vpg(env_fn=env_fn, ac_kwargs=ac_kwargs, logger_kwargs=logger_kwargs, steps_per_epoch=num_elastica_timesteps, epochs=1000, seed=seed, pi_lr=learning_rate, vf_lr=learning_rate)
    #
    #                 exp_save_file = f"hidden_size_{hidden_size}_activation_{activation_str}_learning_rate_{learning_rate}_seed{seed}.txt"
    #                 with open("progress.txt") as f1:
    #                     with open(exp_save_file, "w") as f2:
    #                         for line in f1:
    #                             f2.write(line)
