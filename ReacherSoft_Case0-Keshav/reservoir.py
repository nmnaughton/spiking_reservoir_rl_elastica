import logging
import pickle
import random
import shutil
import sys
from time import time
from tqdm import tqdm
import os
import nengo
import nengo_loihi
nengo_loihi.set_defaults()
import numpy as np
import scipy

class ReservoirSimulator:
    def __init__(self, input_size, n_neurons, output_size, sim_time,
                 action_calculation_method, rod_state_q, reservoir_state_q,
                 n_reservoir_output_neurons=None,
                 num_coeff_per_action=None):
        self.sim_time = sim_time
        self.input_size = input_size
        self.rod_state = np.zeros(input_size)
        self.n_neurons = n_neurons
        self.n_reservoir_output_neurons = n_reservoir_output_neurons
        self.output_size = output_size
        self.n_reservoir_steps = 0
        self.sim = None

        # full = full W_out matrix, reduced = coefficients for linear recombination
        action_calculation_methods = ['full', 'reduced']
        assert(action_calculation_method in action_calculation_methods )
        self.action_calculation_method = action_calculation_method
        self.num_coeff_per_action = num_coeff_per_action

        # Sets up reservoir spiking neurons network with Nengo
        self.seed = 101
        self.W_in = np.load('W_in.npy')
        self.W_reservoir = np.load('W_reservoir.npy')
        self.alpha = 0.8 # Leakage rate
        self.reservoir_output = np.zeros(512)

        # Queue will act as shared memory between processes
        self.rod_state_q = rod_state_q
        self.reservoir_state_q = reservoir_state_q

        self.initialize_reservoir()

    def initialize_reservoir(self):
        # Disable nengo cache warnings
        nengo.rc.set("decoder_cache", "enabled", "False")

        if self.sim:
          self.sim.close()

        self.network = nengo.Network(seed = self.seed)
        self.n_reservoir_steps = 0

        with self.network:
            def get_scaled_rod_state(time):
                if (self.n_reservoir_steps % 10) == 0:
                    self.n_reservoir_steps = 0
                    self.rod_state = self.rod_state_q.get()
                self.n_reservoir_steps += 1
                return self.rod_state * 2000

            def set_scaled_reservoir_output(time, reservoir_output):
                reservoir_output = reservoir_output * 1e-4
                if (self.n_reservoir_steps % 10) == 0:
                    self.reservoir_state_q.put(reservoir_output)

            W_reservoir_sparse = scipy.sparse.find(self.W_reservoir)
            indicies = np.array([W_reservoir_sparse[0], W_reservoir_sparse[1]]).T
            W_reservoir_sparse_nengo = nengo.Sparse(self.W_reservoir.shape, indicies, W_reservoir_sparse[2])

            input_layer = nengo.Node(output=get_scaled_rod_state, size_in=0, size_out=self.input_size)
            reservoir = nengo.Ensemble(n_neurons=self.n_neurons, dimensions=self.input_size, neuron_type=nengo.LIF())
            conn_in = nengo.Connection(input_layer, reservoir.neurons, synapse=None, transform=self.W_in)
            conn_res = nengo.Connection(reservoir.neurons, reservoir.neurons, transform=W_reservoir_sparse_nengo)

            process_node = nengo.Node(output=set_scaled_reservoir_output, size_in=self.n_neurons)
            conn_proc = nengo.Connection(reservoir.neurons, process_node, synapse=0.01)

        self.sim = nengo_loihi.Simulator(self.network, progress_bar=False)

    def _get_action_matrix_multiplication(self, reservoir_output, W_out):
        W_out = W_out.reshape((self.output_size, self.n_neurons))
        action = W_out @ reservoir_output
        return action

    def _get_action_linear_combination(self, reservoir_output, W_out):
        coefficients = np.array(W_out)
        coefficients = coefficients.reshape((self.output_size, self.num_coeff_per_action))
        action = np.zeros(self.output_size)

        for i in range (0, self.output_size):
            action[i] = np.dot(coefficients[i,:], reservoir_output[i * self.num_coeff_per_action:(i + 1) * self.num_coeff_per_action])

        return action

    def _get_action(self, reservoir_output, W_out):
        if (self.action_calculation_method == 'full'):
            return self._get_action_matrix_multiplication(reservoir_output, W_out)
        else:
            return self._get_action_linear_combination(reservoir_output[-self.n_reservoir_output_neurons:], W_out)

    def simulate_network_continuous(self, num_elastica_timesteps):
        total_sim_time = int(self.sim_time * num_elastica_timesteps)
        self.sim.run(total_sim_time)
