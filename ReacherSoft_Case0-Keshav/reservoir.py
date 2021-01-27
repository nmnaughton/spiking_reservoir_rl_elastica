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
                 rod_state_q, reservoir_state_q,
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

        self.num_coeff_per_action = num_coeff_per_action

        # Sets up reservoir spiking neurons network with Nengo
        self.seed = 101
        self.alpha = 0.8 # Leakage rate
        self.reservoir_output = np.zeros(512)

        # Queue will act as shared memory between processes
        self.rod_state_q = rod_state_q
        self.reservoir_state_q = reservoir_state_q

        self.W_in = np.load('W_in.npy')
        self.W_reservoir = np.load('W_reservoir.npy')

        W_reservoir_sparse = scipy.sparse.find(self.W_reservoir)
        indicies = np.array([W_reservoir_sparse[0], W_reservoir_sparse[1]]).T
        self.W_reservoir_sparse_nengo = nengo.Sparse(self.W_reservoir.shape, indicies, W_reservoir_sparse[2])

        self.initialize_reservoir()

    def reset_reservoir(self):
        self.sim.close()
        self.initialize_reservoir()

    def initialize_reservoir(self):
        # Disable nengo cache warnings
        # nengo.rc.set("decoder_cache", "enabled", "False")

        if self.sim:
          self.sim.close()

        network = nengo.Network(seed = self.seed)
        self.n_reservoir_steps = 0
        self.n_reservoir_steps_output = 0

        with network:
            def get_scaled_rod_state(time):
                if (self.n_reservoir_steps % 10) == 0:
                    # print('trying to get rod state')
                    self.done, self.rod_state = self.rod_state_q.get()
                    # print('success getting rod state', self.done)
                self.n_reservoir_steps += 1
                return self.rod_state

            def set_scaled_reservoir_output(time, reservoir_output):
                if (self.n_reservoir_steps_output % 10) == 0:
                    # print('going to put reservoir state')
                    self.reservoir_state_q.put(reservoir_output)
                    # print('success putting reservoir state')
                self.n_reservoir_steps_output += 1

            
            input_layer  = nengo.Node(output=get_scaled_rod_state, size_in=0, size_out=self.input_size)
            reservoir    = nengo.Ensemble(n_neurons=self.n_neurons, dimensions=1, neuron_type=nengo_loihi.neurons.LIF(amplitude=0.001))
            process_node = nengo.Node(output=set_scaled_reservoir_output, size_in=self.n_neurons)

            conn_in = nengo.Connection(input_layer, reservoir.neurons, synapse=None, transform=self.W_in)
            self.W_out_eye = np.eye(self.n_neurons)
            conn_proc = nengo.Connection(reservoir.neurons, process_node, synapse=0.01)
            conn_res = nengo.Connection(reservoir.neurons, reservoir.neurons, transform=self.W_reservoir_sparse_nengo)

        self.sim = nengo_loihi.Simulator(network)

    def simulate_network_continuous(self, num_elastica_timesteps):
        total_sim_time = int(self.sim_time * num_elastica_timesteps)
        print('going to run for:', total_sim_time)
        # self.sim.run(total_sim_time)
        while True:
            self.sim.run_steps(10)
            # self.sim.run_steps(int(np.round(self.sim_time/0.001)))
            if self.done:
                print('resetting reservoir')
                self.reset_reservoir()

