import numpy as np
import nengo
import random
import copy

import gym
from gym import wrappers
from time import time

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
        self.alpha = 0.8 # Leakage rate

        self.states = []
        self.actions = []
        self.rewards = []
        self.reservoir_outputs = []
        self.reservoir_voltages = []
        self.reservoir_spikes = []

    # NOTE: Normalizing the action makes it very difficult for CMA-ES to converge
    def _normalize_action_clip(self, action):
        for i in range(len(action)):
            if action[i] < -1.0:
                action[i] = -1.0
            elif action[i] > 1.0:
                action[i] = 1.0

        return action

    def _normalize_action(self, action):
        for i in range(len(action)):
            action[i] /= 10
        return action

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

    def _get_action(self, reservoir_output, W_out, collect_metadata):
        if (self.action_calculation_method == "full"):
            return self._get_action_matrix_multiplication(reservoir_output, W_out, collect_metadata)
        else:
            return self._get_action_linear_combination(reservoir_output[-self.n_reservoir_output_neurons:], W_out, collect_metadata)

    def simulate_network(self, W_out, env, num_elastica_timesteps, collect_metadata=False, curr_dir='./', filter_output=True):
        # Disable nengo cache warnings
        nengo.rc.set("decoder_cache", "enabled", "False")

        if collect_metadata:
            env.render()

        self.state = env.get_state()
        tot_reward = 0.0
        for i in range(num_elastica_timesteps):
            with nengo.Network(seed = self.seed) as network:
                input_layer = nengo.Node(output=self.state, size_out=self.input_size)
                reservoir = nengo.Ensemble(n_neurons=self.n_neurons, dimensions=self.input_size, neuron_type=nengo.LIF())

                nengo.Connection(input_layer, reservoir.neurons, synapse=None, transform=self.W_in)
                nengo.Connection(reservoir.neurons, reservoir.neurons, transform=self.W_reservoir)

                spikes_probe = nengo.Probe(reservoir.neurons)

                if collect_metadata:
                    output_probe = nengo.Probe(reservoir.neurons, 'output', synapse=0.005)
                    voltage_probe = nengo.Probe(reservoir.neurons, 'voltage', synapse=0.005)

            with nengo.Simulator(network, progress_bar=False) as sim:
                sim.run(self.sim_time)

            # Take the output at the last Nengo simulation timestep
            # reservoir_output = sim.data[output_probe][-1]

            # Sum the output over all Nengo simulation timesteps
            reservoir_output = np.sum(sim.data[spikes_probe], axis=0)

            # Logic for using alpha (leakage rate)
            # if len(reservoir_state) == 0:
            #     reservoir_state = reservoir_output
            # else:
            #     reservoir_state = (1 - self.alpha)*reservoir_state + self.alpha * reservoir_output
            # action = self._get_action(reservoir_state, W_out, collect_metadata)

            action = self._get_action(reservoir_output, W_out, collect_metadata)
            # action = W_out @ reservoir_output
            self.state, reward, done, _ = env.step(action)

            if collect_metadata:
                self.states.append(self.state)
                self.actions.append(action)
                self.rewards.append(reward)

                self.reservoir_outputs.append(sim.data[output_probe])
                self.reservoir_voltages.append(sim.data[voltage_probe])
                self.reservoir_spikes.append(sim.data[spikes_probe])

            tot_reward += reward

            if done:
                break

        if collect_metadata:
            np.save(os.path.join(curr_dir, "states.npy"), np.array(self.states))
            np.save(os.path.join(curr_dir, "actions.npy"), np.array(self.actions))
            np.save(os.path.join(curr_dir, "rewards.npy"), np.array(self.rewards))

            np.save(os.path.join(curr_dir, "reservoir_outputs.npy"), np.array(self.reservoir_outputs))
            np.save(os.path.join(curr_dir, "reservoir_voltages.npy"), np.array(self.reservoir_voltages))
            np.save(os.path.join(curr_dir, "reservoir_spikes.npy"), np.array(self.reservoir_spikes))

            env.close()
            env.post_processing("video.mp4")
            if curr_dir != './':
                shutil.move("2D_2d_video.mp4", curr_dir)
                # shutil.move("2D_3d_video.mp4", curr_dir)

        avg_tot_reward = tot_reward / (i + 1)
        # print(f"avg_tot_reward: {avg_tot_reward}")
        return avg_tot_reward

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
            n_jobs=8) #-1

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

class SupervisedLearningEngine():

    def __init__(self, reservoir_output_file_path, reload_reservoir_data=False):
        self.seed = 101
        self.input_size = 14
        self.output_size = 3
        self.n_reservoir_neurons = 128
        self.W_in = np.load('W_in.npy')
        self.W_reservoir = np.load('W_reservoir.npy')

        self._load_state_action_pairs()

        if reload_reservoir_data:
            self._generate_reservoir_outputs()

        self.reservoir_outputs = np.load(reservoir_output_file_path)
        self.state = self.states[0]

        print(self.states.shape)
        print(self.actions.shape)
        print(self.reservoir_outputs.shape)

    def _load_state_action_pairs(self):
        file_path = 'state_action_pairs.npz'
        npz_file = np.load('state_action_pairs.npz')

        self.states  = npz_file['state']
        actions = npz_file['action']

        self.num_data_pts = actions.shape[0]

        self.actions = np.zeros((self.num_data_pts, self.output_size))
        self.actions[1:] = actions[0:(self.num_data_pts - 1)]

    def _generate_reservoir_outputs(self):
        self.spikes = []
        self.decoded_outputs = []
        self.voltages = []

        self.summed_spikes = []
        self.summed_decoded_outputs = []
        self.summed_voltages = []

        for i in tqdm(range(self.num_data_pts)):
            self.state = self.states[i]

            with nengo.Network(seed = self.seed) as network:
                input_layer = nengo.Node(output=self.state, size_out=self.input_size)
                reservoir = nengo.Ensemble(n_neurons=self.n_reservoir_neurons, dimensions=self.input_size, neuron_type=nengo.LIF())

                nengo.Connection(input_layer, reservoir.neurons, synapse=None, transform=self.W_in)
                nengo.Connection(reservoir.neurons, reservoir.neurons, transform=self.W_reservoir)

                spikes_probe = nengo.Probe(reservoir.neurons)
                decoded_output_probe = nengo.Probe(reservoir.neurons, 'output', synapse=0.005)
                voltage_probe = nengo.Probe(reservoir.neurons, 'voltage', synapse=0.005)

            with nengo.Simulator(network, progress_bar=False) as sim:
                sim.run(0.01)

            # Take the output at the last Nengo simulation timestep
            spikes = sim.data[spikes_probe][-1]
            decoded_output = sim.data[decoded_output_probe][-1]
            voltages = sim.data[voltage_probe][-1]

            # Sum the output over all Nengo simulation timesteps
            summed_spikes = np.sum(sim.data[spikes_probe], axis=0)
            summed_decoded_output = np.sum(sim.data[decoded_output_probe], axis=0)
            summed_voltages = np.sum(sim.data[voltage_probe], axis=0)

            self.spikes.append(spikes)
            self.decoded_outputs.append(decoded_output)
            self.voltages.append(voltages)

            self.summed_spikes.append(summed_spikes)
            self.summed_decoded_outputs.append(summed_decoded_output)
            self.summed_voltages.append(summed_voltages)

        self.spikes = np.array(self.spikes)
        self.decoded_outputs = np.array(self.decoded_outputs)
        self.voltages = np.array(self.voltages)

        self.summed_spikes = np.array(self.summed_spikes)
        self.summed_decoded_outputs = np.array(self.summed_decoded_outputs)
        self.summed_voltages = np.array(self.summed_voltages)

        np.save('spikes.npy', self.spikes)
        np.save('decoded_outputs.npy', self.decoded_outputs)
        np.save('voltages.npy', self.voltages)

        np.save('summed_spikes.npy', self.summed_spikes)
        np.save('summed_decoded_outputs.npy', self.summed_decoded_outputs)
        np.save('summed_voltages.npy', self.summed_voltages)

    def run_gradient_descent_tf(self):
        # Source: https://donaldpinckney.com/books/tensorflow/book/ch2-linreg/2018-03-21-multi-variable.html

        # Define data placeholders
        reservoir_output = tf.placeholder(tf.float32, shape=(self.n_reservoir_neurons, None))
        action = tf.placeholder(tf.float32, shape=(self.output_size, None))

        # Define trainable variables
        W_out = tf.get_variable("W_out", shape=(self.output_size, self.n_reservoir_neurons))

        # Define model output
        predicted_action = tf.matmul(W_out, reservoir_output)

        # Define loss function
        L = tf.reduce_sum((predicted_action - action)**2)

        # Define optimizer
        # optimizer = tf.train.AdamOptimizer().minimize(L)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(L)

        # Create a session and initialize variables
        session = tf.Session()
        session.run(tf.global_variables_initializer())

        # Main optimization loop
        for i in range(100):
            _, current_loss, current_W_out = session.run([optimizer, L, W_out], feed_dict={
                reservoir_output: self.reservoir_outputs.transpose(),
                action: self.actions.transpose()
            })
            # print("t = %g, loss = %g, W_out = %s, b = %s" % (t, current_loss, str(current_W_out), str(current_b)))
            print("t = %g, loss = %g" % (i, current_loss))

        np.save('best_W_out.npy', current_W_out)

    def run_linear_regression(self):
        window = 10

        def moving_average(a, n=3) :
            ret = np.cumsum(a, dtype=float)
            ret[n:] = ret[n:] - ret[:-n]
            return ret[n - 1:] / n

        smooth_actions = np.zeros((moving_average(self.actions[:,0],window).shape[0],self.actions.shape[1]))
        for i in range(self.actions.shape[1]):
            smooth_actions[:,i] = moving_average(self.actions[:,i],window)

        smooth_reservoir_outputs = np.zeros((moving_average(self.reservoir_outputs[:,0],window).shape[0],self.reservoir_outputs.shape[1]))
        for i in range(self.reservoir_outputs.shape[1]):
            smooth_reservoir_outputs[:,i] = moving_average(self.reservoir_outputs[:,i],window)

        reg_smooth = LinearRegression().fit(smooth_reservoir_outputs, smooth_actions)
        r_squared = reg_smooth.score(smooth_reservoir_outputs, smooth_actions)
        print(f"r_squared: {r_squared}")
        np.save('best_W_out.npy', reg_smooth.coef_)
        print(reg_smooth.coef_)

def get_env(collect_data_for_postprocessing=False):
    env = Environment(
        final_time=7,
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
        dim=2.0,
        max_rate_of_change_of_activation=np.infty,
    )
    return env

# Reservoir parameters
input_size = 14
output_size = 3
n_reservoir_neurons = 128
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

num_elastica_timesteps = 100 * 7 #TODO(kshivvy): Refactor in terms of final episode time (in sec.) and num_steps_per_update

def fitness_fn(W_out):
    global reservoir_network_simulator
    global num_elastica_timesteps

    num_trials = 3
    mean_accumulated_reward = 0.0

    for i in range(num_trials):
        env = get_env()
        env.reset()
        mean_accumulated_reward += reservoir_network_simulator.simulate_network(W_out, env, num_elastica_timesteps)

    mean_accumulated_reward /= num_trials
    # print(f"mean_accumulated_reward: {mean_accumulated_reward}")
    fitness = -1.0 * mean_accumulated_reward
    return fitness

def train(cma_save_file=''):
    # CMA-ES parameters
    global weights_size
    initial_step_size = 0.0001
    population_size = 4 #max(32, int(weights_size * (weights_size ** 0.5)))
    num_cma_generations = 2 #100

    cma_engine = CMAEngine(
            weights_size = weights_size,
            initial_step_size = initial_step_size,
            population_size = population_size)

    if cma_save_file:
        cma_engine.load(cma_save_file)

    for i in range(num_cma_generations):
        # Create base directory for current CMA-ES generation
        curr_dir = "cma_es_data/generation_" + str(i+1) + "/"
        os.makedirs(curr_dir, exist_ok=True)

        # Run CMA-ES for one generation
        best_W_out = cma_engine.run_cma_es()
        # print(f"best_W_out: {best_W_out}")
        np.save(os.path.join(curr_dir, "best_W_out.npy"), best_W_out)
        cma_engine.save(curr_dir)
        # cma_engine.plot(curr_dir)

        # Simulate the W_out generated by CMA-ES for the current generation
        avg_tot_reward = evaluate(best_W_out, curr_dir)

        # Stop simulation if our solution is good enough
        if avg_tot_reward >= 1.5:
            break

    return best_W_out

def evaluate(W_out, curr_dir='./', filter_output=True):
    global reservoir_network_simulator
    global num_elastica_timesteps

    env = get_env(collect_data_for_postprocessing=True)
    env.reset()
    avg_tot_reward = reservoir_network_simulator.simulate_network(W_out, env, num_elastica_timesteps, collect_metadata=True, curr_dir=curr_dir, filter_output=filter_output)
    return avg_tot_reward

def generate_cma_plots(load_dir, save_dir='./'):
    file_path = os.path.join(load_dir, 'saved_cma_es.txt')
    shutil.copyfile(file_path, os.path.join(save_dir, 'saved_cma_es.txt'))

    cma_engine = CMAEngine(weights_size = 10,
                           initial_step_size = 10,
                           population_size = 10)
    cma_engine.plot(save_dir)

def generate_Ws(seed=101, density=0.20, spectral_radius=0.9):
    np.random.seed(seed)
    W_in = np.random.uniform(bounds[0], bounds[1], (n_reservoir_neurons, input_size))

    # Draw W_reservoir from a random unifrom distribution
    W_reservoir = np.random.uniform(bounds[0], bounds[1], (n_reservoir_neurons, n_reservoir_neurons))

    # Create a mask to make W_rservoir a sparse matrix with a density of 20%
    mask = scipy.sparse.rand(n_reservoir_neurons, n_reservoir_neurons, density=density)
    mask = np.array(mask.todense())
    mask[np.where(mask > 0)] = 1
    W_reservoir = W_reservoir * mask

    # Set the spectral radius of W_reservoir is 0.90
    E, _ = np.linalg.eig(W_reservoir)
    e_max = np.max(np.abs(E))
    W_reservoir /= np.abs(e_max)/spectral_radius

    # Save W_in and W_reservoir to file
    np.save('W_in.npy', W_in)
    np.save('W_reservoir.npy', W_reservoir)

# Command to Run: cd Documents\NCSA\elastica-python-CoRL_cases && venv\Scripts\activate && cd ReacherSoft_Case0-Keshav && python reservoir_rl.py
def main():
    # generate_Ws()

    # Directory cleanup. NOTE: IT WILL DELETE YOUR PREVIOUS RESULTS! BACK THEM UP
    if os.path.isdir('./cma_es_data'):
        shutil.rmtree('./cma_es_data')

    if os.path.isdir('./BlueWatersResults'):
        shutil.rmtree('./BlueWatersResults')
        os.mkdir('./BlueWatersResults')

    # Run CMA-ES
    cma_save_file = ''
    best_W_out = train(cma_save_file)

    # Postprocessing
    last_gen_dir =  os.path.join('./cma_es_data', os.listdir('./cma_es_data')[-1])
    W_out = np.load(os.path.join(last_gen_dir, 'best_W_out.npy'))
    evaluate(W_out, curr_dir='./BlueWatersResults')

    # supervised_learning_engine = SupervisedLearningEngine(reservoir_output_file_path='summed_spikes.npy', reload_reservoir_data=True)
    # supervised_learning_engine.run_linear_regression()
    # curr_dir = 'C:/Users/kesha/Documents/NCSA/elastica-python-CoRL_cases/ReacherSoft_Case0-Keshav/lin_reg_weights'
    # W_out = np.load('best_W_out.npy')
    # evaluate(W_out, curr_dir=curr_dir)

if __name__ == "__main__":
    main()
