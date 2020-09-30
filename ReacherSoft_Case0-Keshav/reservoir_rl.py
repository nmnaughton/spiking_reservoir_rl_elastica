import logging
import pickle
import random
import shutil
import sys
from time import time
from tqdm import tqdm
import os

import cma
import nengo
import numpy as np
import matplotlib.pyplot as plt
import scipy

sys.path.append(os.path.abspath(os.path.join('..', 'elastica')))
from set_environment import Environment

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

        # Metadata collection on reservoir simulation
        self.collect_metadata = False

        # Sets up reservoir spiking neurons network with Nengo
        self.seed = 101
        self.W_in = np.load('W_in.npy')
        self.W_reservoir = np.load('W_reservoir.npy')
        self.alpha = 0.8 # Leakage rate

    def _initialize_reservoir(self):
        # Disable nengo cache warnings
        nengo.rc.set("decoder_cache", "enabled", "False")

        self.state = np.zeros(self.input_size)
        self.network = nengo.Network(seed = self.seed)

        with self.network:
            def func(time):
                return self.state * 2000

            W_reservoir_sparse = scipy.sparse.find(self.W_reservoir)
            indicies = np.array([W_reservoir_sparse[0], W_reservoir_sparse[1]]).T
            W_reservoir_sparse_nengo = nengo.Sparse(self.W_reservoir.shape, indicies, W_reservoir_sparse[2])

            input_layer = nengo.Node(output=func, size_in=0, size_out=self.input_size)
            reservoir = nengo.Ensemble(n_neurons=self.n_neurons, dimensions=self.input_size, neuron_type=nengo.LIF())
            conn_in = nengo.Connection(input_layer, reservoir.neurons, synapse=None, transform=self.W_in)
            conn_res = nengo.Connection(reservoir.neurons, reservoir.neurons, transform=W_reservoir_sparse_nengo)
            self.output_probe = nengo.Probe(reservoir.neurons, 'output', synapse=0.01, sample_every=self.sim_time)

            if self.collect_metadata:
                self.spikes_probe = nengo.Probe(reservoir.neurons, sample_every=self.sim_time)
                self.voltage_probe = nengo.Probe(reservoir.neurons, 'voltage', synapse=0.01, sample_every=self.sim_time)

        self.sim = nengo.Simulator(self.network, progress_bar=False)

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
        if (self.action_calculation_method == "full"):
            return self._get_action_matrix_multiplication(reservoir_output, W_out)
        else:
            return self._get_action_linear_combination(reservoir_output[-self.n_reservoir_output_neurons:], W_out)

    def simulate_network(self, W_out, env, num_elastica_timesteps):
        self.state = env.get_state()
        self._initialize_reservoir()
        tot_reward = 0.0

        for i in range(num_elastica_timesteps):
            self.sim.run(self.sim_time)

            # Take the output at the last Nengo simulation timestep
            reservoir_output = self.sim.data[self.output_probe][-1] * 1e-4

            # Logic for using alpha (leakage rate).
            # if len(reservoir_state) == 0:
            #     reservoir_state = reservoir_output
            # else:
            #     reservoir_state = (1 - self.alpha)*reservoir_state + self.alpha * reservoir_output
            # action = self._get_action(reservoir_state, W_out)

            action = self._get_action(reservoir_output, W_out)
            self.state, reward, done, _ = env.step(action)
            tot_reward += reward

            if done:
                break

        avg_tot_reward = tot_reward / (i + 1)
        return avg_tot_reward

    def simulate_network_verbose(self, W_out, env, num_elastica_timesteps, save_dir='./', make_vid=False):
        # Video plotting will not work on Blue Waters
        if make_vid:
            env.render()

        if self.collect_metadata:
            states = []
            actions = []
            rewards = []

        self.state = env.get_state()
        self._initialize_reservoir()
        tot_reward = 0.0

        for i in tqdm(range(num_elastica_timesteps)):
            self.sim.run(self.sim_time)

            # Take the output at the last Nengo simulation timestep
            reservoir_output = self.sim.data[self.output_probe][-1] * 1e-4
            action = self._get_action(reservoir_output, W_out)
            self.state, reward, done, _ = env.step(action)
            tot_reward += reward

            if self.collect_metadata:
                states.append(self.state)
                actions.append(action)
                rewards.append(reward)

            if done:
                break

        if self.collect_metadata:
            np.save(os.path.join(save_dir, "states.npy"), np.array(states))
            np.save(os.path.join(save_dir, "actions.npy"), np.array(actions))
            np.save(os.path.join(save_dir, "rewards.npy"), np.array(rewards))

            reservoir_outputs = self.sim.data[self.output_probe]
            reservoir_voltages = self.sim.data[self.voltage_probe]
            reservoir_spikes = self.sim.data[self.spikes_probe]

            np.save(os.path.join(save_dir, "reservoir_outputs.npy"), np.array(reservoir_outputs))
            np.save(os.path.join(save_dir, "reservoir_voltages.npy"), np.array(reservoir_voltages))
            np.save(os.path.join(save_dir, "reservoir_spikes.npy"), np.array(reservoir_spikes))

        if make_vid:
            env.close()
            env.post_processing("video.mp4")
            if save_dir != './':
                shutil.move("2D_2d_video.mp4", curr_dir)
                # shutil.move("2D_3d_video.mp4", curr_dir)

        avg_tot_reward = tot_reward / (i + 1)
        print(f"avg_tot_reward: {avg_tot_reward}")
        return avg_tot_reward

    def set_collect_metadata(self, collect_metadata):
        self.collect_metadata = collect_metadata
        self._initialize_reservoir()

class CMAEngine:
    def __init__(self, weights_size, initial_step_size, population_size):
        self.set_cma_es_solver(weights_size, initial_step_size, population_size)
        self.num_cma_iterations = 0

    def run_cma_es(self, num_generations=1):
        global fitness_fn

        self.cma_es_solver.optimize(
            objective_fct=fitness_fn,
            iterations=num_generations,
            min_iterations=num_generations,
            n_jobs=-1)

        cma_es_result = self.cma_es_solver.result
        best_W_out = cma_es_result[0]

        self.num_cma_iterations += 1
        return best_W_out

    def save(self, curr_dir):
        with open(os.path.join(curr_dir, "saved_cma_es.txt"), "wb") as f:
            f.write(self.cma_es_solver.pickle_dumps())

    def load(self, file_path):
        with open(file_path, "rb") as f:
            cma_es_str = f.read()
            self.cma_es_solver = pickle.loads(cma_es_str)

    def plot(self, curr_dir):
        self.cma_es_solver.logger.plot()
        cma.plot()
        plt.savefig(os.path.join(curr_dir, "cmea_es_run.png"))

    def set_cma_es_solver(self, weights_size, initial_step_size, population_size, opts = {}):
        random_weights = np.zeros(weights_size)

        opts['popsize'] = population_size

        self.cma_es_solver = cma.CMAEvolutionStrategy(
            random_weights,
            initial_step_size,
            opts)

    def set_fitness_fn(self, fitness_fn):
        self.fitness_fn = fitness_fn

def get_env(collect_data_for_postprocessing=False):
    return Environment(
        final_time=5,
        num_steps_per_update=50,
        number_of_control_points=3,
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
        dim=3.0, # 2.0,
        max_rate_of_change_of_activation=np.infty)

# Reservoir parameters
input_size = 17 # 14
output_size = 6 # 3
n_reservoir_neurons = 512
sim_time = 0.01
bounds = [-1, 1]
action_calculation_method = "full" # "reduced"
num_coeff_per_action = 5

# Changes number of parameters CMA-ES optimizes for and number of reservoir nerons.
# full = CMA-ES produces full W_out matrix, reduced = CMA-ES produces coefficients for linear recombination
weights_size = n_reservoir_neurons * output_size
n_reservoir_output_neurons = n_reservoir_neurons
if (action_calculation_method == "reduced"):
    weights_size = output_size * num_coeff_per_action
    n_reservoir_output_neurons = weights_size

reservoir_network_simulator = ReservoirNetworkSimulator(
        input_size = input_size,
        n_neurons = n_reservoir_neurons,
        output_size = output_size,
        sim_time = sim_time,
        bounds = bounds,
        action_calculation_method = action_calculation_method,
        num_coeff_per_action = num_coeff_per_action,
        n_reservoir_output_neurons = n_reservoir_output_neurons)

num_elastica_timesteps = int(100 * 5) #TODO(kshivvy): Refactor in terms of final episode time (in sec.) and num_steps_per_update

def fitness_fn(W_out):
    global reservoir_network_simulator
    global num_elastica_timesteps

    num_trials = 4
    mean_accumulated_reward = 0.0

    for i in range(num_trials):
        env = get_env()
        env.reset()
        mean_accumulated_reward += reservoir_network_simulator.simulate_network(W_out, env, num_elastica_timesteps)

    mean_accumulated_reward /= num_trials
    # print(f"mean_accumulated_reward: {mean_accumulated_reward}")
    with open("logging.txt", "a") as myfile:
        myfile.write(f"mean_accumulated_reward: {mean_accumulated_reward}\n")
    fitness = -1.0 * mean_accumulated_reward
    return fitness

def train(cma_save_file=''):
    # CMA-ES parameters
    global weights_size
    initial_step_size = 1.0
    population_size = 128
    num_cma_generations = 20

    cma_engine = CMAEngine(
            weights_size = weights_size,
            initial_step_size = initial_step_size,
            population_size = population_size)

    if cma_save_file:
        cma_engine.load(cma_save_file)

    # Run CMA-ES
    best_W_out = cma_engine.run_cma_es(num_cma_generations)

    # Postprocessing
    dir = 'cma_es_data'
    os.makedirs(dir, exist_ok=True)
    np.save(os.path.join(dir, "best_W_out.npy"), best_W_out)
    cma_engine.save(dir)
    # cma_engine.plot(dir)
    avg_tot_reward = evaluate(best_W_out, dir)

    return best_W_out

def evaluate(W_out, save_dir='./', set_collect_metadata=True, make_vid=True):
    global reservoir_network_simulator
    global num_elastica_timesteps

    if set_collect_metadata:
        reservoir_network_simulator.set_collect_metadata(set_collect_metadata)

    env = get_env(collect_data_for_postprocessing=True)
    env.reset()
    avg_tot_reward = reservoir_network_simulator.simulate_network_verbose(W_out, env, num_elastica_timesteps, save_dir=save_dir, make_vid=make_vid)
    return avg_tot_reward

def generate_cma_plots(load_dir='./', save_dir='./'):
    file_path = os.path.join(load_dir, 'saved_cma_es.txt')

    if save_dir != './':
        shutil.copyfile(file_path, os.path.join(save_dir, 'saved_cma_es.txt'))

    cma_engine = CMAEngine(weights_size = 10,
                           initial_step_size = 10,
                           population_size = 10)

    cma_engine.load(file_path)
    cma_engine.plot(file_path)

def generate_Ws(seed=101, density=0.20, spectral_radius=0.9):
    np.random.seed(seed)
    W_in = np.random.uniform(bounds[0], bounds[1], (n_reservoir_neurons, input_size))

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
    W_reservoir /= np.abs(e_max)/spectral_radius

    # Save W_in and W_reservoir to file
    np.save('W_in.npy', W_in)
    np.save('W_reservoir.npy', W_reservoir)

# Command to Run: cd Documents\NCSA\git_version\spiking_reservoir_rl_elastica-master && venv\Scripts\activate && cd ReacherSoft_Case0-Keshav && python reservoir_rl.py
def main():
    # Generate W_in and W_reservoir
    # generate_Ws()

    # Train: Run CMA-ES
    cma_save_file = ''
    best_W_out = train(cma_save_file)

    # Evaluate: Generate video/metadata for a W_out
    # W_out = np.load('best_W_out.npy')
    # evaluate(W_out)

    # Plot: Plot CMA-ES simulation data. Currently does not work with saved_cma_es.txt files from Blue Waters.
    # generate_cma_plots()

if __name__ == "__main__":
    main()
