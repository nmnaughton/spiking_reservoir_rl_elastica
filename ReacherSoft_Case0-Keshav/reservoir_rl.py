# Force CPU, disable GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import gym
import logging
import multiprocessing as mp
import numpy as np
import scipy
import sys
# import tensorflow as tf

# Import Elastica
from reservoir import ReservoirSimulator
sys.path.append(os.path.abspath(os.path.join('..', 'elastica')))
from set_environment import Environment

# Import Nengo and Intel nxsdk packages
import nengo_loihi
nengo_loihi.set_defaults()
try:
    from nxsdk.logutils.nxlogging import set_verbosity, LoggingLevel
    set_verbosity(LoggingLevel.WARNING)
except:
    print('nxsdk is not installed')

# Import SpinningUp RL packages
# from spinup import vpg_tf1 as vpg
from spinup import ppo_pytorch as ppo

def get_elastica_env(collect_data_for_postprocessing=False):
    sim_dt=2.0e-4
    RL_update_interval = 0.01
    return Environment(
        final_time=elastica_sim_time,
        num_steps_per_update=int(np.round( RL_update_interval/sim_dt)),
        number_of_control_points=num_control_points,
        alpha=75,
        beta=75,
        COLLECT_DATA_FOR_POSTPROCESSING=collect_data_for_postprocessing,
        mode=4,
        target_position=[-0.4, 0.6, 0.0],
        target_v=0.5,
        boundary=[-0.6, 0.6, 0.3, 0.9, -0.6, 0.6],
        E=1e7,
        sim_dt=sim_dt,
        n_elem=20,
        NU=30,
        dim=dim,
        max_rate_of_change_of_activation=np.infty)

class ElasticaEnvWrapper(gym.Env):

    def __init__(self, dim, num_control_points, n_reservoir_neurons, collect_data_for_postprocessing=False):
        super(ElasticaEnvWrapper).__init__()
        self.dim = dim
        self.num_control_points = num_control_points
        self.n_reservoir_neurons = n_reservoir_neurons

        self.elastica_env = get_elastica_env(collect_data_for_postprocessing)
        self.collect_data_for_postprocessing = collect_data_for_postprocessing
        if self.collect_data_for_postprocessing:
            self.elastica_env.render()

        self.p = None
        self.rod_state_q = None
        self.reservoir_state_q = None

        if self.dim == 2.0:
            # normal direction activation (2D)
            self.action_space = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.num_control_points,),
                dtype=np.float64)
            self.action = np.zeros(self.num_control_points)
        if self.dim == 3.0:
            # normal and/or binormal direction activation (3D)
            self.action_space = gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(2 * self.num_control_points,),
                dtype=np.float64)
            self.action = np.zeros(2 * self.num_control_points)

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_reservoir_neurons,),
            dtype=np.float64)

        self.num_steps = 0
        self.total_reward = 0.0

        self.manager = mp.Manager()
        self.rod_state_q = self.manager.Queue()
        self.reservoir_state_q = self.manager.Queue()

        self.p = mp.Process(target=reservoir_worker_func, args=(input_size, n_reservoir_neurons, output_size, nengo_sim_time, self.rod_state_q, self.reservoir_state_q, num_RL_timesteps,))
        self.p.start()

    def reset(self):
        rod_state = self.elastica_env.reset()

        # the first entry is a reset flag
        self.rod_state_q.put([True, np.concatenate((rod_state,self.action))])
        reservoir_state = self.reservoir_state_q.get()

        self.num_steps = 0
        self.total_reward = 0.0

        return reservoir_state

    def step(self, action):
        rod_state, reward, done, info = self.elastica_env.step(action)
        self.rod_state_q.put([False,np.concatenate((rod_state,action))])
        # print(np.concatenate((rod_state,action)))
        reservoir_state = self.reservoir_state_q.get()
        # print(reservoir_state)
        self.num_steps += 1
        # print('step #', self.num_steps)
        self.total_reward += reward

        if done:
            avg_reward = self.total_reward / self.num_steps

            print("num_steps: ", self.num_steps)
            print("total_reward: ", self.total_reward)
            print("avg_reward: ", avg_reward)
            print(' ')

            with open("logging.txt", "a") as f:
                f.write(str(reward) + '\n')

            if self.collect_data_for_postprocessing:
                self.elastica_env.close()
                self.elastica_env.post_processing("video.mp4")

        return reservoir_state, reward, done, info

    def render(self, mode='human'):
        return

def reservoir_worker_func(input_size, n_reservoir_neurons, output_size, nengo_sim_time, rod_state_q, reservoir_state_q, num_RL_timesteps):

    reservoir_simulator = ReservoirSimulator(
        input_size=input_size,
        n_neurons=n_reservoir_neurons,
        output_size=output_size,
        sim_time=nengo_sim_time,
        rod_state_q=rod_state_q,
        reservoir_state_q=reservoir_state_q)

    reservoir_simulator.simulate_network_continuous(num_RL_timesteps)
    print("FINISHED WORKER, returning")
    return

# python -W ignore reservoir_rl.py
if __name__ == "__main__":
    # Reservoir parameters
    dim = 2.0
    num_control_points = 3
    output_size = num_control_points * int(dim - 1)
    input_size = 11 + num_control_points * int(dim - 1) + output_size
    n_reservoir_neurons = 512
    elastica_sim_time = 5
    num_episodes_per_epoch = 4
    nengo_sim_time = 0.01
    num_RL_timesteps = int(np.round(elastica_sim_time/nengo_sim_time * num_episodes_per_epoch))
    weights_size = n_reservoir_neurons * output_size

    # SpinningUp parameters
    env_fn = lambda : ElasticaEnvWrapper(dim, num_control_points, n_reservoir_neurons)
    logger_kwargs = dict(output_dir='./', exp_name='ppo_spiking_reservoir_512_learning_rate_0.01_seed_0_batch_size_1000_epochs_500')
    ac_kwargs = dict(hidden_sizes=[])

    # Run SpinningUp reinforcement learning algorithm
    # with tf.Graph().as_default():
        # vpg(env_fn=env_fn, ac_kwargs=ac_kwargs, logger_kwargs=logger_kwargs, steps_per_epoch=num_elastica_timesteps, epochs=1, pi_lr=0.01, vf_lr=0.01, seed=0)

    ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, logger_kwargs=logger_kwargs, steps_per_epoch=num_RL_timesteps, epochs=50, seed=0)

    print('shutting down with errors. These could be fixed by editing spinning up code. ')

        # if ElasticaEnvWrapper.p:
    #     print('terminating previous process')
    #     ElasticaEnvWrapper.p.terminate()