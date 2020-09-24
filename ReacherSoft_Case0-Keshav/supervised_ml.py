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

sys.path.append("../")
from set_environment import Environment
from reservoir_rl import generate_Ws, ReservoirNetworkSimulator

class SupervisedLearningEngine():

    def __init__():
        initialize_reservoir_network_simulator()

    def initialize_reservoir_network_simulator():
        input_size = 13
        output_size = 2
        n_reservoir_neurons = 512
        sim_time = 0.01
        bounds = [-1, 1]
        action_calculation_method = 'full'
        num_coeff_per_action = None

        # Changes number of parameters CMA-ES optimizes for and number of reservoir nerons.
        # 'full' = CMA-ES produces full W_out matrix,
        # 'reduced' = CMA-ES produces coefficients for linear recombination
        weights_size = n_reservoir_neurons * output_size
        n_reservoir_output_neurons = n_reservoir_neurons

        if (action_calculation_method == 'reduced'):
            weights_size = output_size * num_coeff_per_action
            n_reservoir_output_neurons = weights_size

        generate_Ws(bounds=bounds, n_reservoir_neurons=n_reservoir_neurons, input_size=input_size)

        self.reservoir_network_simulator = ReservoirNetworkSimulator(
                input_size = input_size,
                n_neurons = n_reservoir_neurons,
                output_size = output_size,
                sim_time = sim_time,
                bounds = bounds,
                action_calculation_method = action_calculation_method,
                num_coeff_per_action = num_coeff_per_action,
                n_reservoir_output_neurons = n_reservoir_output_neurons)


# Command to Run: cd Documents\NCSA\elastica-python-CoRL_cases && venv\Scripts\activate && cd ReacherSoft_Case0-Keshav && python supervised_ml.py
def main():
    pass

if __name__ == "__main__":
    main()
