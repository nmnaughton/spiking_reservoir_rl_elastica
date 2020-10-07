import numpy as np
import nengo
import random
import copy

import gym
from gym import wrappers
from time import time
from datetime import datetime

from tqdm import tqdm

import logging
import shutil
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import cma
import sys
import pickle
from time import sleep
from matplotlib import cm
import matplotlib.pyplot as plt
import scipy
from sklearn.linear_model import LinearRegression
# sys.path.append("../")
sys.path.append(os.path.abspath(os.path.join('..', 'elastica')))
from set_environment import Environment

from tensorforce import Agent
from tensorforce.execution import Runner
from tensorforce import Environment as TF_ENV
# import nengo_dl

class ReservoirNetworkSimulator:
    def __init__(self, input_size, n_neurons, output_size, sim_time,
                 bounds, action_calculation_method,
                 n_reservoir_output_neurons=None,
                 num_coeff_per_action=None):

        self.sim_time = sim_time
        self.input_size = input_size
        self.n_neurons = n_neurons
        self.n_reservoir_output_neurons = n_reservoir_output_neurons
        self.output_size = output_size

        # full = CMA-ES produces full W_out matrix, reduced = CMA-ES produces coefficients for linear recombination
        self.action_calculation_methods = ["full", "reduced"]
        self.action_calculation_method = action_calculation_method
        assert(self.action_calculation_method in self.action_calculation_methods )
        self.num_coeff_per_action = num_coeff_per_action

        # Sets up reservoir spiking neurons network with Nengo
        self.seed = 101
        self.W_in = np.load('W_in.npy')
        self.W_reservoir = np.load('W_reservoir.npy')

        self.states = []
        self.actions = []
        self.rewards = []
        self.reservoir_outputs = []
        self.reservoir_voltages = []
        self.reservoir_spikes = []

        network = nengo.Network(seed = self.seed)
        with network:
            def func(time):
                return self.state * 2000
            input_layer = nengo.Node(output=func, size_in = 0, size_out=self.input_size, label="Input Points")
            reservoir = nengo.Ensemble(n_neurons=self.n_neurons, dimensions=self.input_size, neuron_type=nengo.LIF())
            conn_in = nengo.Connection(input_layer, reservoir.neurons, synapse=None, transform=self.W_in)

            w_res_sparse = scipy.sparse.find(self.W_reservoir)
            w_res_sparse_nengo = nengo.Sparse(
                self.W_reservoir.shape,
                np.array([w_res_sparse[0],w_res_sparse[1]]).T,
                w_res_sparse[2]
                )
            self.con_res = nengo.Connection(reservoir.neurons, reservoir.neurons, transform = w_res_sparse_nengo) 
            self.output_probe = nengo.Probe(reservoir.neurons, 'output', synapse=0.01, sample_every=self.sim_time)
        self.sim = nengo.Simulator(network, progress_bar=False)

    def _get_action_matrix_multiplication(self, reservoir_output, W_out, collect_metadata):
        W_out = W_out.reshape((self.output_size, self.n_neurons))
        action = W_out @ reservoir_output
        return action

    def _get_action_linear_combination(self, reservoir_output, W_out, collect_metadata):
        coefficients = np.array(W_out)
        coefficients = coefficients.reshape((self.output_size, self.num_coeff_per_action))
        action = np.zeros(self.output_size)

        for i in range (0, self.output_size):
            action[i] = np.dot(coefficients[i,:], reservoir_output[i * self.num_coeff_per_action:(i + 1) * self.num_coeff_per_action])

        return action

    def _get_action(self, reservoir_output, W_out):
        if (self.action_calculation_method == "full"):
            return self._get_action_matrix_multiplication(reservoir_output, W_out, collect_metadata)
        else:
            return self._get_action_linear_combination(reservoir_output[-self.n_reservoir_output_neurons:], W_out, collect_metadata)

    def simulate_network(self, W_out, env, num_elastica_timesteps, collect_metadata=False, curr_dir='./', filter_output=True, make_vid=False):
        # Disable nengo cache warnings
        nengo.rc.set("decoder_cache", "enabled", "False")

        self.state = env.get_state()
        tot_reward = 0.0
        self.sim.reset()

        for i in range(num_elastica_timesteps):
            self.state, reward, done, _  = self.do_combined_step(W_out)
            tot_reward += reward
            if done:
                break

        avg_tot_reward = tot_reward / (i + 1)
        return avg_tot_reward

    def do_combined_step(self, W_out):
        self.sim.run(self.sim_time)
        # Take the output at the last Nengo simulation timestep
        reservoir_output = self.sim.data[self.output_probe][-1] * 1e-4
        action = self._get_action(reservoir_output, W_out)
        self.state, reward, done, _ = env.step(action)
        return state, reward, done, _ 

    def get_reservoir_state(self, action):
        self.state, reward, done, _ = env.step(action)
        self.sim.run(self.sim_time)
        reservoir_state = self.sim.data[self.output_probe][-1] * 1e-4
        # print(reservoir_state.min()*1000, reservoir_state.max()*1000, action)
        # print(reservoir_state.mean())
        return reservoir_state, done, reward
        # return self.state, done, reward

    def get_seed(self):
        return self.seed

    def get_state(self):
        return self.state

    def get_states(self):
        return self.states

    def get_rewards(self):
        return self.rewards

    def get_actions(self):
        return self.actions
 
def get_env(collect_data_for_postprocessing=False):
    elastica_dt = 2.0e-4
    env = Environment(
        final_time=episode_sim_time,
        num_steps_per_update=int(np.rint(sim_time/elastica_dt)),
        number_of_control_points=num_control_points,
        alpha=75,
        beta=75,
        COLLECT_DATA_FOR_POSTPROCESSING=collect_data_for_postprocessing,
        mode=4,
        target_position=[-0.4, 0.6, 0.0],
        target_v=0.5,
        boundary=[-0.6, 0.6, 0.3, 0.9, -0.6, 0.6],
        E=1e7,
        sim_dt=elastica_dt,
        n_elem=20,
        NU=30,
        dim=dim,
        max_rate_of_change_of_activation=np.infty,
    )
    return env

def generate_Ws(seed=101, density=0.20, spectral_radius=0.90):
    np.random.seed(seed)
    # W_in = np.ones((n_reservoir_neurons, input_size))*0.1
    W_in = np.random.uniform(bounds[0], bounds[1], (n_reservoir_neurons, input_size))
    print("Shape of W_in:", W_in.shape)
    # Draw W_reservoir from a random unifrom distribution
    W_reservoir = np.random.uniform(bounds[0], bounds[1], (n_reservoir_neurons, n_reservoir_neurons))
    print("Shape of W_reservoir:", W_reservoir.shape)
    # Create a mask to make W_rservoir a sparse matrix with a density of 20%
    mask = scipy.sparse.rand(n_reservoir_neurons, n_reservoir_neurons, density=density)
    mask = np.array(mask.todense())
    mask[np.where(mask > 0)] = 1
    W_reservoir = W_reservoir * mask

    # Set the spectral radius of W_reservoir is 0.90
    E, _ = np.linalg.eig(W_reservoir)
    e_max = np.max(np.abs(E))
    print(e_max)
    W_reservoir /= np.abs(e_max)/spectral_radius

    # Save W_in and W_reservoir to file
    np.save('W_in.npy', W_in)
    np.save('W_reservoir.npy', W_reservoir)

class CustomEnvironment(TF_ENV):

    def __init__(self):
        super().__init__()
        self.score_list = []

    def states(self):
        return dict(type='float', shape=(n_reservoir_neurons,))
        # return dict(type='float', shape=(input_size,))

    def actions(self):
        return dict(type='float', shape=(output_size,), min_value=-2, max_value=2)

    # Optional additional steps to close environment
    def close(self):
        super().close()

    def reset(self):
        reservoir_network_simulator.sim.reset()
        env.reset()
        reservoir_network_simulator.state = env.get_state()
        reservoir_network_simulator.sim.run(reservoir_network_simulator.sim_time)
        reservoir_state = reservoir_network_simulator.sim.data[reservoir_network_simulator.output_probe][-1] * 1e-4
        self.score = 0
        self.step_num = 0
        return reservoir_state
        # return env.get_state()

    def execute(self, actions):
        next_state, done, reward = reservoir_network_simulator.get_reservoir_state(actions)
        self.score += reward
        self.step_num += 1
        if done == True:
            self.score_list.append(self.score/self.step_num)
            print("    episode %d final score: %5.4f, reward on final timestep: %5.4f" % (i, self.score/self.step_num, reward))
        return next_state, done, reward

if __name__ == "__main__":

    nengo_package = nengo
    restart_from_previous = False
    overwrite_logging_file = True
    save_path = 'cma_2d_data_compact'
    logging_file_name = 'logging_2d_compact.txt'
    save_evaluate_video = False

    episode_sim_time = 10

    # Reservoir parameters
    dim = 2.0
    num_control_points = 3
    input_size = 11 + num_control_points * int(dim - 1)
    output_size = num_control_points * int(dim - 1)
    n_reservoir_neurons = 512 # 128
    sim_time = 0.01
    bounds = [-1, 1]
    num_elastica_timesteps = int(episode_sim_time/sim_time) #TODO(kshivvy): Refactor in terms of final episode time (in sec.) and num_steps_per_update
    weights_size = n_reservoir_neurons * output_size
    #########################

    generate_Ws()
    env = get_env()
    env.reset()

    reservoir_network_simulator = ReservoirNetworkSimulator(
                input_size = input_size,
                n_neurons = n_reservoir_neurons,
                output_size = output_size,
                sim_time = sim_time,
                bounds = bounds,
                action_calculation_method = "full",
                num_coeff_per_action = 5,
                n_reservoir_output_neurons = n_reservoir_neurons)

    tf_env = TF_ENV.create(environment=CustomEnvironment, max_episode_timesteps=num_elastica_timesteps)
    

    # # Using tensorforce reinforce algo. It does not seem to work. 
    # agent = Agent.create(
    #     agent='reinforce',
    #     environment=tf_env, 
    #     batch_size = 2,
    #     network=[ ],
    #     learning_rate=0.01,
    #     config=dict(seed=10), 
    #     )

    # The original agent that I used. This seems like it should be similar to reinforce but I 
    # have had a hard time understanding the documentation and have not dug into the code. 
    # this set up seems to work well. 
    learning_rate = 0.01
    decay_rate = 0.00
    num_steps = 2
    batch_size = 2
    agent = Agent.create(
        agent='tensorforce', 
        environment=tf_env, 
        update=dict(unit="episodes",batch_size=2),
        # optimizer=dict(optimizer='adam', learning_rate=learning_rate, amsgrad=True),
        optimizer=dict(optimizer='adam', learning_rate=dict(
                type='decaying', decay='inverse_time', unit='episodes',
                num_steps=num_steps, initial_value=learning_rate, decay_rate=decay_rate
            ),amsgrad=True),
        objective='policy_gradient', 
        # reward_estimation=dict(horizon=dict(
        #     type='linear', unit='episodes', num_steps=100,
        #     initial_value=10, final_value=10
        # )),
        reward_estimation=dict(horizon=10),
        policy=dict(network=[ ]),
        # policy=dict(network="auto"),
        config=dict(seed=10), 
        # I have found that seed=10 works as a good seed. Other seeds do not work as well 
        # which we need to explore carefully. Seed=2 gave consistent increase but not to as
        # good a score.
    )

    # Train for n episodes
    episodes = 100
    start_time = time()
    for i in range(episodes):
        # print("current learning rate:", learning_rate / (1 + decay_rate * i / num_steps))
        states = tf_env.reset()
        terminal = False

        while not terminal:
            actions = agent.act(states=states)
            states, terminal, reward = tf_env.execute(actions=actions)
            agent.observe(terminal=terminal, reward=reward)

    print("Training took", time() - start_time)
    print(tf_env.score_list)
    plt.plot(tf_env.score_list)
    plt.title('2d -- info_about_params')

    env = get_env(collect_data_for_postprocessing=True)
    COLLECT_DATA=True
    env.reset()
    tf_env.reset()
    total_reward = 0
    terminal = False
    itern = 0
    while not terminal:
        actions = agent.act(states=states, independent=True)
        states, terminal, reward = tf_env.execute(actions=actions)
        total_reward += reward
        itern += 1
    print('Evaluation Score:', total_reward/itern)
    # env.post_processing("RC_video.mp4", plot_3d_video= False )

    plt.show()
    print('done')


    agent.close()
    tf_env.close()
