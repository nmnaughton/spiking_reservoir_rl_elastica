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

n_reservoir_neurons = 512
bounds = [-1, 1]
input_size = 14 + 3 
output_size = 3 

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
        self.nengo_dt = 0.001

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
                    # print('getting rod state from queue')
                    self.done, self.rod_state = self.rod_state_q.get()
                self.n_reservoir_steps += 1
                return self.rod_state

            def set_scaled_reservoir_output(time, reservoir_output):
                if (self.n_reservoir_steps_output % 10) == 0:
                    # print('putting reservoir state in queue')
                    self.reservoir_state_q.put(reservoir_output)
                self.n_reservoir_steps_output += 1

            
            input_layer  = nengo.Node(output=get_scaled_rod_state, size_in=0, size_out=self.input_size)
            reservoir    = nengo.Ensemble(n_neurons=self.n_neurons, dimensions=1, neuron_type=nengo.LIF(amplitude=self.nengo_dt * 0.1))
            # reservoir    = nengo.Ensemble(n_neurons=self.n_neurons, dimensions=1, neuron_type=nengo_loihi.neurons.LIF(amplitude=self.nengo_dt))
            process_node = nengo.Node(output=set_scaled_reservoir_output, size_in=self.n_neurons)

            conn_in   = nengo.Connection(input_layer, reservoir.neurons, synapse=None, transform=self.W_in)
            conn_proc = nengo.Connection(reservoir.neurons, process_node, synapse=0.01)
            conn_res  = nengo.Connection(reservoir.neurons, reservoir.neurons, transform=self.W_reservoir_sparse_nengo)

        # self.sim = nengo_loihi.Simulator(network, dt=self.nengo_dt)
        self.sim = nengo.Simulator(network, dt=self.nengo_dt, progress_bar=False)

    def simulate_network_continuous(self, num_elastica_timesteps):
        while True:
            self.sim.run_steps(int(np.round(self.sim_time/self.nengo_dt)))
            if self.done:
                # print('resetting reservoir')
                self.reset_reservoir()

def generate_Ws(seed=101, density=0.20, spectral_radius=0.9):
    np.random.seed(seed)
    W_in = np.random.uniform(bounds[0], bounds[1], (n_reservoir_neurons, input_size))
    # W_in /= input_size

    # Sample W_reservoir from a random unifrom distribution
    W_reservoir = np.random.uniform(bounds[0], bounds[1], (n_reservoir_neurons, n_reservoir_neurons))

    # Create a mask to make W_reservoir a sparse matrix with a density of 20%
    mask = scipy.sparse.rand(n_reservoir_neurons, n_reservoir_neurons, density=density)
    mask = np.array(mask.todense())
    mask[np.where(mask > 0)] = 1
    W_reservoir = W_reservoir * mask

    # Set the spectral radius of W_reservoir to 0.90
    E, _ = np.linalg.eig(W_reservoir)
    e_max = np.max(np.abs(E))
    print('e_max:', e_max)
    W_reservoir /= np.abs(e_max)/spectral_radius
    # W_reservoir /= n_reservoir_neurons

    # Save W_in and W_reservoir to file
    np.save('W_in.npy', W_in)
    np.save('W_reservoir.npy', W_reservoir)

# Command to Run: cd Documents\NCSA\git_version\spiking_reservoir_rl_elastica-master && venv\Scripts\activate && cd ReacherSoft_Case0-Keshav && python reservoir_rl.py
def main():
    # Generate W_in and W_reservoir
    generate_Ws()

if __name__ == "__main__":
    main()
