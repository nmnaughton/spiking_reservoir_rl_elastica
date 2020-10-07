# -*- coding: utf-8 -*-
"""
A Continuous Control Implementation for TensorFlow 2.0
Author: W.J.A. van Heeswijk
Date: 11-8-2020
This code is supplemental to the following note:
'Implementing Gaussian Actor Networks for  Continuous Control in TensorFlow 2.0'
https://www.researchgate.net/publication/343714359_Implementing_Gaussian_Actor_Networks_for_Continuous_Control_in_TensorFlow_20
Corresponding blog post:
https://towardsdatascience.com/a-minimal-working-example-for-continuous-policy-gradients-in-tensorflow-2-0-d3413ec38c6b
Python 3.8 and TensorFlow 2.3 were used to write the algorithm
This code has been published under the GNU GPLv3 license
"""
# Utilities
from tqdm import tqdm

# Needed for training the network
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow.keras.initializers as initializers

# Needed for animation
import matplotlib.pyplot as plt

# Disable GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Import spiking reservoir and Elastica
import sys
from reservoir_rl import ReservoirNetworkSimulator
sys.path.append(os.path.abspath('elastica'))
from set_environment import Environment
import scipy

"""Construct the actor network with mu and sigma as output"""
def ConstructActorNetwork(bias_mu, bias_sigma):
    inputs = layers.Input(shape=(512,)) #input dimension
    # hidden1 = layers.Dense(256, activation="relu",kernel_initializer=initializers.he_normal())(inputs)
    # hidden2 = layers.Dense(256, activation="relu",kernel_initializer=initializers.he_normal())(hidden1)
    mu = layers.Dense(3, activation="linear",kernel_initializer=initializers.Zeros(),
                         bias_initializer=initializers.Constant(bias_mu))(inputs)
    sigma = layers.Dense(3, activation="softplus",kernel_initializer=initializers.Zeros(),
                         bias_initializer=initializers.Constant(bias_sigma))(inputs)

    actor_network = keras.Model(inputs=inputs, outputs=[mu,sigma])
    return actor_network

"""Weighted Gaussian log likelihood loss function"""
def CustomLossGaussian(state, action, reward):
        # Obtain mu and sigma from actor network
        nn_mu, nn_sigma = actor_network(state)

        # Obtain pdf of Gaussian distribution
        pdf_value = tf.exp(-0.5 *((action - nn_mu) / (nn_sigma))**2) * 1/(nn_sigma*tf.sqrt(2 *np.pi))

        # Compute log probability
        log_probability = tf.math.log(pdf_value + 1e-5)

        # Compute weighted loss
        loss_actor = - reward * log_probability
        return loss_actor

def get_env(collect_data_for_postprocessing=False):
    return Environment(
        final_time=10,
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
        dim=2.0, # 3.0,
        max_rate_of_change_of_activation=np.infty)


mu_target = 4.0
target_range = 0.25
max_reward = 1.0

# Create actor network
bias_mu = np.zeros(3)  #bias 0.0 yields mu=0.0 with linear activation function
bias_sigma = np.ones(3) * 0.55 #bias 0.55 yields sigma=1.0 with softplus activation function
actor_network = ConstructActorNetwork(bias_mu , bias_sigma)
opt = keras.optimizers.Adam(learning_rate=0.01) # 0.01

# Initialize arrays for plot
epoch_ar = []
mu_ar = []
sigma_ar=[]
reward_ar=[]
target_ar=[]

# Reservoir parameters
input_size = 14 # 17
output_size = 3 # 6
n_reservoir_neurons = 512
sim_time = 0.01
bounds = [-1, 1]
action_calculation_method = "full" # NOT USED FOR REINFORCE
num_coeff_per_action = 20 # NOT USED FOR REINFORCE

reservoir_network_simulator = ReservoirNetworkSimulator(
        input_size = input_size,
        n_neurons = n_reservoir_neurons,
        output_size = output_size,
        sim_time = sim_time,
        bounds = bounds,
        action_calculation_method = action_calculation_method,
        num_coeff_per_action = num_coeff_per_action,
        n_reservoir_output_neurons = n_reservoir_neurons)

num_episodes = 500
num_elastica_timesteps = int(100 * 10) #TODO(kshivvy): Refactor in terms of final episode time (in sec.) and num_steps_per_update

for episode_num in tqdm(range(num_episodes)):

    # generate_Ws()

    env = get_env()
    rod_state = env.reset()
    reservoir_network_simulator.initialize_reservoir()
    avg_reward = 0.0
    gradients = []

    for elastica_timestep in range(num_elastica_timesteps):
        reservoir_output = reservoir_network_simulator.simulate_network_vanilla(rod_state)

        # Obtain mu and sigma from network
        reservoir_output_tf = tf.convert_to_tensor([reservoir_output], np.float32)
        mu, sigma = actor_network(reservoir_output_tf)

        # Draw action from normal distribution
        action = tf.random.normal([3,], mean=mu, stddev=sigma)
        # print(action)

        # Compute reward
        rod_state, reward, done, _ = env.step(action.numpy()[0])
        avg_reward += reward

        # Update network weights
        with tf.GradientTape() as tape:
            # Compute Gaussian loss
            loss_value = CustomLossGaussian(reservoir_output_tf, action, reward) * 100

            # Compute gradients
            gradient = tape.gradient(loss_value, actor_network.trainable_variables)
            opt.apply_gradients(zip(gradient, actor_network.trainable_variables))

            gradients.append(gradient)

        if done:
            break

    avg_reward /= elastica_timestep

    # Batch update gradients every two episodes
    # if np.mod(episode_num, 2) == 0:
    #     for gradient in gradients:
    #         opt.apply_gradients(zip(gradient, actor_network.trainable_variables))

    # Update console output and plot
    if np.mod(episode_num, 1) == 0:
        with open("logging.txt", "a") as myfile:
            myfile.write(f"======episode {episode_num}======\n")
            myfile.write(f"mu: {mu.numpy()}\n")
            myfile.write(f"sigma: {sigma.numpy()}\n")
            myfile.write(f"loss: {loss_value.numpy()}\n")
            myfile.write(f"reward: {avg_reward}\n")
             # actor_network.save("trained_model")

#plt.show()
